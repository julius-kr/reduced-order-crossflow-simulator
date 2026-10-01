"""Legacy CFD: the vendored 3D-hydrodynamic-flow-sculpting code, its safe library loader, the translation of
projects to library entries, and the isolated worker that runs the upstream operators.

The real map libraries are not part of the repository; these tests build synthetic libraries in the upstream
format (a pickled dictionary of index maps plus a parameter tree) with known maps.
"""
import gzip
import os
import pickle
import shutil

import numpy as np
import pytest

from crossflow_simulator import legacy_cfd
from crossflow_simulator.backends.legacy_cfd_worker import GRID, load_library_file, reflected
from crossflow_simulator.comparison import (
    BackendRun, ComparisonService, accuracy_against, build_comparison, realistic_seconds,
)
from crossflow_simulator.config import ChannelGeometry, OperatingConditions, PillarSpec, ProjectConfig
from crossflow_simulator.simulator import simulate

from conftest import ROOT, UM, make_config

# A library entry of the published pillar library (first key in its sorted order): base A-B at y 0-0.375,
# tip C 0.5625 upstream at y 0.375, height 0.375, in library units (channel 1.5 x 1.5).
ENTRY = (1.5, 1.5, 0.375, 0.0, 0.375, 3.4375, 0.375)
SIDE = 300 * UM  # square channel: 200 um per library unit
UNIT = SIDE / 1.5


def library_pillar(pillar_id="tri", x=600 * UM, mount="bottom", mirror=False):
    """The ENTRY pillar in project coordinates (base across the channel at x, tip upstream)."""
    ya, yb, xc, yc = (v * UNIT for v in (ENTRY[3], ENTRY[4], ENTRY[5] - 4.0, ENTRY[6]))
    vertices = [(0.0, ya), (0.0, yb), (xc, yc)]
    if mirror:
        vertices = [(vx, SIDE - vy) for vx, vy in vertices]
    return PillarSpec(pillar_id, vertices=vertices, height=ENTRY[2] * UNIT, x=x, mount=mount)


def square_case(pillars=(), layout="layers_y", rates=(1.0, 2.0)):
    cfg = make_config(layout=layout, rates=rates, channel=ChannelGeometry(width=SIDE, height=SIDE, length=3000 * UM),
                      pillars=list(pillars))
    cfg.grid.output_ny = 64
    return cfg


def shift_map(offset=20):
    """Backward map of a library (top-mounted) pillar: every node above mid-height takes its value from
    ``offset`` nodes towards y = 0."""
    cells = np.arange(GRID * GRID)
    z, y = np.divmod(cells, GRID)
    return np.where(z >= GRID // 2, z * GRID + np.clip(y - offset, 0, GRID - 1), cells).astype(np.int32)


def write_library(folder, entries: dict, compress=False):
    """A pillar library in the upstream format (``library_generation_pillar.build_and_save_library``)."""
    target = folder / "generated_library_pillar"
    target.mkdir(parents=True, exist_ok=True)
    library = {legacy_cfd.library_key(values): index_map for values, index_map in entries.items()}
    tree: dict = {}
    for values in entries:
        node = tree
        for value in values:
            node = node.setdefault(np.float64(value), {})
    index = {key: sorted({np.float64(values[i]) for values in entries}) for i, key in enumerate(legacy_cfd.PILLAR_KEYS)}
    index["tree"] = tree
    np.save(target / "library_pillar.npy", library, allow_pickle=True)
    np.save(target / "library_pillar_dict.npy", index, allow_pickle=True)
    if compress:
        with open(target / "library_pillar.npy", "rb") as src, gzip.open(target / "library_pillar.npy.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        (target / "library_pillar.npy").unlink()
    return target


@pytest.fixture
def library(tmp_path, monkeypatch):
    folder = tmp_path / "libraries"
    write_library(folder, {ENTRY: shift_map()})
    monkeypatch.setenv(legacy_cfd.LIBRARY_ENV, str(folder))
    return folder


@pytest.fixture
def no_library(tmp_path, monkeypatch):
    monkeypatch.setenv(legacy_cfd.LIBRARY_ENV, str(tmp_path / "empty"))


# ----------------------------------------------------------------------------- vendored code and loader


def test_vendored_upstream_code_matches_the_pinned_revision():
    report = legacy_cfd.verify_upstream()
    assert report["ok"], report
    assert (ROOT / legacy_cfd.UPSTREAM_FOLDER / "LICENSE").read_text(encoding="utf-8").lstrip().startswith(
        "GNU GENERAL PUBLIC LICENSE")


class _RunsCode:
    def __reduce__(self):
        return (os.getcwd, ())


def test_library_loader_reads_numpy_dictionaries_and_refuses_anything_else(tmp_path):
    data = {"a": np.arange(5, dtype=np.int32), "tree": {np.float64(1.5): {np.float64(0.25): {}}},
            "values": [np.float64(0.375), 2], "mask": np.ones(3, dtype=bool)}
    np.save(tmp_path / "lib.npy", data, allow_pickle=True)
    with open(tmp_path / "lib.npy", "rb") as src, gzip.open(tmp_path / "lib.npy.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    for name in ("lib.npy", "lib.npy.gz"):
        loaded = load_library_file(tmp_path / name)
        assert np.array_equal(loaded["a"], data["a"]) and loaded["tree"] == data["tree"]
        assert loaded["values"] == data["values"] and loaded["mask"].dtype == bool
    np.save(tmp_path / "evil.npy", {"x": _RunsCode()}, allow_pickle=True)
    with pytest.raises(pickle.UnpicklingError, match="only NumPy arrays"):
        load_library_file(tmp_path / "evil.npy")
    np.save(tmp_path / "plain.npy", np.zeros(4))
    with pytest.raises(ValueError, match="expected a pickled dictionary"):
        load_library_file(tmp_path / "plain.npy")


# ----------------------------------------------------------------------------- translation


def test_library_pillars_are_matched_directly_and_through_exact_symmetries(library):
    plan = legacy_cfd.build_plan(square_case([library_pillar()]))
    assert plan["unsupported"] == []
    (op,) = plan["operations"]
    assert op["key"] == legacy_cfd.library_key(ENTRY) and op["flip_z"] and not op["mirror_y"]
    assert op["deviation"] < 1e-9
    assert [s["operations"] for s in plan["stations"]] == [0, 1]
    top = legacy_cfd.build_plan(square_case([library_pillar(mount="top")]))["operations"][0]
    assert top["key"] == op["key"] and not top["flip_z"] and not top["mirror_y"]
    mirrored = legacy_cfd.build_plan(square_case([library_pillar(mirror=True)]))["operations"][0]
    assert mirrored["key"] == op["key"] and mirrored["mirror_y"] and mirrored["map_key"] != op["map_key"]


def test_cases_outside_the_libraries_are_explained(library):
    moved = library_pillar()
    moved.y = 20 * UM
    reasons = legacy_cfd.build_plan(square_case([moved]))["unsupported"]
    assert len(reasons) == 1 and "no library entry" in reasons[0] and "nearest library pillar" in reasons[0]
    assert "differs by up to 20 um" in reasons[0]
    rectangle = square_case([library_pillar()])
    rectangle.channel.height = 150 * UM
    assert "square channel" in legacy_cfd.build_plan(rectangle)["unsupported"][0]
    square = PillarSpec("sq", vertices=[(0, 0), (60 * UM, 0), (60 * UM, 60 * UM), (0, 60 * UM)], height=75 * UM,
                        x=600 * UM, y=40 * UM)
    assert "4-vertex footprint" in legacy_cfd.build_plan(square_case([square]))["unsupported"][0]
    ports = ProjectConfig.load(ROOT / "examples" / "serial_three_inlets.json")
    assert "inlet ports" in legacy_cfd.build_plan(ports)["unsupported"][0]
    contrast = square_case()
    contrast.materials[1].viscosity *= 4
    assert "viscosity contrast" in legacy_cfd.build_plan(contrast)["unsupported"][0]
    pressure = square_case()
    pressure.operating = OperatingConditions(drive="pressure")
    assert any("pressure-driven" in r for r in legacy_cfd.build_plan(pressure)["unsupported"])


def test_square_library_example_uses_a_library_entry(tmp_path, monkeypatch):
    entry = (1.5, 1.5, 1.125, 0.0, 0.75, 4.75, 0.0)  # the example's pillar (three quarters of the height)
    monkeypatch.setenv(legacy_cfd.LIBRARY_ENV, str(tmp_path / "libraries"))
    write_library(tmp_path / "libraries", {entry: shift_map()})
    plan = legacy_cfd.build_plan(ProjectConfig.load(ROOT / "examples" / "square_library_pillars.json"))
    assert plan["unsupported"] == []
    key = legacy_cfd.library_key(entry)
    # both pillars are bottom-mounted (z-reflected library maps); the second one is the mirror image
    assert [(op["pillar"], op["key"], op["flip_z"], op["mirror_y"]) for op in plan["operations"]] == [
        ("wall", key, True, False), ("mirrored", key, True, True)]


PUBLISHED = ROOT / legacy_cfd.UPSTREAM_FOLDER  # published libraries and streamlines, when installed


# The library entries (ph, ya, yb, xc, yc; channel 1.5 x 1.5) of the examples the legacy CFD runs, per pillar in
# streamwise order. All are bottom-mounted (z-reflected maps) and entries of the published pillar library.
LIBRARY_EXAMPLES = {
    "square_library_pillars": [(1.125, 0.0, 0.75, 4.75, 0.0), (1.125, 0.75, 1.5, 4.75, 1.5)],
    "square_centred_pillar": [(0.75, 0.375, 1.125, 4.75, 0.75)],
    "square_full_width_wedge": [(0.5, 0.0, 1.5, 4.75, 0.75)],
    "square_pillar_sequence": [(0.75, 0.0, 0.75, 4.75, 0.0), (0.75, 0.75, 1.5, 4.75, 1.5)] * 2,
    "square_upstream_pillar_array": [(1.0, 0.0, 1.125, 3.4375, 0.5625)],
}


def example_keys(name):
    return [legacy_cfd.library_key((1.5, 1.5, *entry)) for entry in LIBRARY_EXAMPLES[name]]


@pytest.mark.parametrize("name", sorted(LIBRARY_EXAMPLES))
def test_legacy_cfd_examples_translate_to_their_library_entries(name, tmp_path, monkeypatch):
    from crossflow_simulator.backends import get_backend

    write_library(tmp_path / "libraries", {(1.5, 1.5, *e): shift_map() for e in LIBRARY_EXAMPLES[name]})
    monkeypatch.setenv(legacy_cfd.LIBRARY_ENV, str(tmp_path / "libraries"))
    cfg = ProjectConfig.load(ROOT / "examples" / f"{name}.json")
    plan = legacy_cfd.build_plan(cfg)
    assert plan["unsupported"] == []
    assert [op["key"] for op in plan["operations"]] == example_keys(name)
    assert all(op["deviation"] == 0.0 and op["flip_z"] and not op["mirror_y"] for op in plan["operations"])
    assert get_backend("legacy").check(cfg) == []  # the legacy solver runs them too: all four solvers compare


@pytest.mark.skipif(not (PUBLISHED / "generated_library_pillar" / "library_pillar_dict.npy").exists(),
                    reason="published pillar library not installed")
@pytest.mark.parametrize("name", sorted(LIBRARY_EXAMPLES))
def test_published_library_has_every_example_pillar(name):
    plan = legacy_cfd.build_plan(ProjectConfig.load(ROOT / "examples" / f"{name}.json"))
    assert plan["unsupported"] == [] and [op["key"] for op in plan["operations"]] == example_keys(name)
    assert all(op["deviation"] == 0.0 for op in plan["operations"])


STREAMLINES = PUBLISHED / "streamline_files_pillar" / "cw1.5000_ch1.5000_ph0.3750_ya0.0000_yb0.3750_xc3.4375_yc0.0000.txt"


@pytest.mark.skipif(not STREAMLINES.exists(), reason="published streamline data not installed")
def test_published_streamlines_put_the_library_pillars_against_z_equals_ch():
    xs, ys, zs, _, ye, ze = np.loadtxt(STREAMLINES, comments="%")[:, :6].T
    moved = np.hypot(ye - ys, ze - zs)
    behind = (xs < 0.05) & (ye < 0.25)  # streamlines ending behind the base edge (y 0 to 0.375)
    near_ceiling = moved[behind & (ze > 1.25)].mean()
    near_floor = moved[behind & (ze < 0.25)].mean()
    assert near_ceiling > 3 * near_floor  # a pillar of a quarter of the height, standing against z = ch
    assert legacy_cfd.LIBRARY_MOUNT == "top"


@pytest.mark.skipif(not STREAMLINES.exists(), reason="published streamline data not installed")
def test_comsol_time_per_map_comes_from_the_published_export_stamps():
    stamps = legacy_cfd.comsol_export_intervals(STREAMLINES.parent)
    assert stamps["files"] == legacy_cfd.COMSOL_PILLAR["maps"] == 900
    assert stamps["mean_seconds"] == pytest.approx(legacy_cfd.COMSOL_PILLAR["seconds_per_map"], abs=1.0)
    assert stamps["max_gap_seconds"] < 1800  # one uninterrupted batch
    assert stamps["first"].strftime("%Y-%m-%d %H:%M") == legacy_cfd.COMSOL_PILLAR["first_export"]
    assert stamps["last"].strftime("%Y-%m-%d %H:%M") == legacy_cfd.COMSOL_PILLAR["last_export"]


def test_missing_library_only_matters_when_a_pillar_needs_it(no_library):
    assert legacy_cfd.build_plan(square_case())["unsupported"] == []
    reasons = legacy_cfd.build_plan(square_case([library_pillar()]))["unsupported"]
    assert len(reasons) == 1 and "library not installed" in reasons[0]


def test_reflected_maps_are_conjugated_by_the_reflection():
    index_map = shift_map()
    flipped = reflected(index_map, flip_z=True, mirror_y=False)
    image = np.arange(GRID * GRID).reshape(GRID, GRID)
    direct = image.reshape(-1)[index_map].reshape(GRID, GRID)
    via_reflection = np.flipud(np.flipud(image).reshape(-1)[index_map].reshape(GRID, GRID))
    assert np.array_equal(image.reshape(-1)[flipped].reshape(GRID, GRID), via_reflection)
    assert not np.array_equal(direct, via_reflection)
    assert np.array_equal(reflected(flipped, flip_z=True, mirror_y=False), index_map)


# ----------------------------------------------------------------------------- worker


def test_worker_applies_the_library_maps_through_the_upstream_operators(library, tmp_path):
    cases = {"bottom": square_case([library_pillar()]), "top": square_case([library_pillar(mount="top")])}
    service = ComparisonService(tmp_path / "work", worker_env={legacy_cfd.LIBRARY_ENV: str(library)})
    try:
        runs = {name: service.run_backend("legacy_cfd", cfg, name, timeout=300) for name, cfg in cases.items()}
        hello = service.clients["legacy_cfd"].hello
    finally:
        service.close()
    assert hello["isolation_problems"] == [] and hello["identity"]["source_revision"] == legacy_cfd.UPSTREAM["revision"]
    for name, run in runs.items():
        assert run.status == "success", run.message
        result = run.result
        inlet = legacy_cfd.initial_profile(cases[name])["codes"]
        assert np.array_equal(result.sections[0].codes, inlet)
        index_map = reflected(shift_map(), flip_z=True, mirror_y=False) if name == "bottom" else shift_map()
        expected = inlet.reshape(-1)[index_map].reshape(GRID, GRID)
        assert np.array_equal(result.final.codes, expected)
        assert [s.name for s in result.sections] == ["inlet plane", "after group 1: tri"]
        assert result.final.grid_kind == "legacy_nodes" and result.backend == "legacy_cfd"
        log = result.diagnostics["operation_log"]
        assert log[0]["flip_z"] == (name == "bottom") and log[0]["changed_fraction"] > 0.01
        # its realistic time includes the COMSOL simulation behind the one map applied
        assert result.diagnostics["comsol_seconds"] == legacy_cfd.COMSOL_PILLAR["seconds_per_map"]
        assert realistic_seconds(run) == pytest.approx(run.solve_seconds + 135.0)
    assert runs["bottom"].cache_state == "library loaded" and runs["top"].cache_state == "library in memory"
    # bottom mount moves the lower half, top mount the upper half
    moved = [np.argwhere(r.result.final.codes != r.result.sections[0].codes)[:, 0] for r in runs.values()]
    assert moved[0].max() < GRID // 2 <= moved[1].min()


def test_compressed_libraries_are_read_too(tmp_path, monkeypatch):
    folder = tmp_path / "gz"
    write_library(folder, {ENTRY: shift_map()}, compress=True)
    monkeypatch.setenv(legacy_cfd.LIBRARY_ENV, str(folder))
    assert legacy_cfd.library_files("pillar")["data"].name == "library_pillar.npy.gz"
    assert legacy_cfd.build_plan(square_case([library_pillar()]))["unsupported"] == []


def test_straight_channel_equals_the_new_solver_and_enters_the_accuracy_table(no_library, tmp_path):
    cfg = ProjectConfig.load(ROOT / "examples" / "parallel_unequal.json")
    cfg.grid.output_ny = 64
    service = ComparisonService(tmp_path / "work")
    try:
        run = service.run_backend("legacy_cfd", cfg, "straight", timeout=300)
    finally:
        service.close()
    assert run.status == "success", run.message
    assert run.cache_state == "library not needed"
    assert run.result.diagnostics["comsol_seconds"] == 0 and realistic_seconds(run) == run.solve_seconds
    new = simulate(cfg)
    agreement = accuracy_against(new, run.result)
    assert agreement["min_iou"] > 0.99 and agreement["max_flux_misassigned"] < 0.005
    # the CFD reference is the new result here: only the bookkeeping is tested
    runs = {"new": BackendRun("new", "success", result=new), "legacy_cfd": run,
            "cfd": BackendRun("cfd", "success", result=new)}
    comparison = build_comparison(cfg, runs)
    assert set(comparison.accuracy) == {"new", "legacy_cfd"}
    assert comparison.accuracy_vs_legacy_cfd["new"]["within_targets"]  # on the 250-node legacy CFD grid
    assert "accuracy_vs_legacy_cfd" in comparison.summary()


def test_unsupported_run_reports_the_reason_without_a_worker(tmp_path):
    cfg = ProjectConfig.load(ROOT / "examples" / "right_triangle_bottom.json")
    service = ComparisonService(tmp_path / "work")
    try:
        run = service.run_backend("legacy_cfd", cfg, "x")
    finally:
        service.close()
    assert run.status == "unsupported" and run.message.startswith("Not supported by Legacy CFD: pillar maps")
    assert "legacy_cfd" not in service.clients


def test_sections_are_reported_at_the_new_solver_stations(library):
    cfg = square_case([library_pillar()])
    new = simulate(cfg)
    stations = legacy_cfd.build_plan(cfg)["stations"]
    assert [s["x"] for s in stations] == pytest.approx([s.x for s in new.sections])
    assert [s["name"] for s in stations[1:]] == [s.name for s in new.sections[1:]]
