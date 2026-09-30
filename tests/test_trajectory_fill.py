"""Output cells without a valid trajectory take the stream label of their valid neighbours."""
import json

import numpy as np
import pytest

from crossflow_simulator.cache import memory_cache
from crossflow_simulator.config import PillarSpec, ProjectConfig
from crossflow_simulator.maps import fill_trajectory_gaps
from crossflow_simulator.models.domain import section_solid
from crossflow_simulator.results import SimulationResult
from crossflow_simulator.simulator import _fill_warnings, simulate

from conftest import ROOT, UM

Y, Z = np.arange(12.0), np.arange(10.0)  # unit cell centres, (nz, ny) = (10, 12)


def section(left: int = 1, right: int = 2, split: int = 6) -> np.ndarray:
    codes = np.full((len(Z), len(Y)), right, dtype=np.int32)
    codes[:, :split] = left
    return codes


def fill(codes, solid=None, max_fill_cells=100, max_failed_fraction=0.5):
    # the synthetic sections are small, so the 1% guard is relaxed except where it is tested
    return fill_trajectory_gaps(codes, solid, Y, Z, max_fill_cells, max_failed_fraction)


def test_isolated_failed_cells_take_the_surrounding_label():
    codes = section()
    codes[2, 2] = codes[7, 9] = -1
    out, filled, report = fill(codes)
    assert out[2, 2] == 1 and out[7, 9] == 2
    assert filled.sum() == 2 and filled[2, 2] and filled[7, 9]
    assert report["failed"] == report["filled"] == 2 and report["unfilled"] == 0
    assert np.array_equal(out[~filled], codes[~filled])  # valid cells are never touched


def test_a_cell_on_an_interface_takes_the_majority_label():
    codes = section(split=5)
    codes[4, 5] = -1  # neighbours: 3 of label 1 (column 4), 5 of label 2
    assert fill(codes)[0][4, 5] == 2
    codes = section(split=6)
    codes[4, 5] = -1  # neighbours: 5 of label 1, 3 of label 2
    assert fill(codes)[0][4, 5] == 1


def test_ties_go_to_the_nearest_valid_centre_then_the_lower_code():
    unit = np.arange(3.0)
    codes = np.array([[-1, -1, 3],
                      [2, -1, -1],
                      [2, -1, 3]], dtype=np.int32)
    out, _, _ = fill_trajectory_gaps(codes, None, unit, unit, 100, 1.0)
    assert out[1, 1] == 2  # two of each; label 2 has the orthogonal (nearer) neighbour
    codes = np.array([[4, -1, 5]] * 3, dtype=np.int32)
    out, _, _ = fill_trajectory_gaps(codes, None, unit, unit, 100, 1.0)
    assert (out[:, 1] == 4).all()  # equal counts, equal distances: the lower code


def test_patches_fill_from_the_outside_in():
    codes = section()
    codes[3:6, 1:4] = -1  # 3 x 3 patch inside label 1: the centre has no valid neighbour at first
    out, filled, report = fill(codes)
    assert (out[3:6, 1:4] == 1).all() and filled.sum() == 9 and report["rounds"] == 2


def test_a_large_patch_stays_unfilled_and_warns():
    codes = section()
    codes[1:5, 1:4] = -1  # 12 cells
    codes[8, 10] = -1
    out, filled, report = fill(codes, max_fill_cells=10)
    assert (out[1:5, 1:4] == -1).all() and out[8, 10] == 2 and filled.sum() == 1
    assert report["unfilled_patches"] == [12] and report["unfilled"] == 12
    assert _fill_warnings("after group 1: p1", report) == [
        "after group 1: p1: 12 cells in 1 patch of 12 cells without valid trajectories were not filled (patches "
        "over model.max_fill_cells = 10 cells stay unfilled)."]


def test_more_than_one_percent_failed_fills_nothing_and_warns():
    codes = section()
    codes[0, :2] = -1  # 2 of 120 cells: 1.7%
    out, filled, report = fill(codes, max_failed_fraction=0.01)
    assert not filled.any() and (out == codes).all() and report["skipped"].startswith("1.67% of the flow cells")
    assert "no cells were filled" in _fill_warnings("s", report)[0]
    codes[0, 1] = 1  # 1 of 120: 0.8%
    assert fill(codes, max_failed_fraction=0.01)[1].sum() == 1


def test_solid_cells_are_never_filled_nor_used_across():
    codes = section()
    solid = np.zeros(codes.shape, dtype=bool)
    solid[4:7, 4:7] = True
    codes[solid] = -1  # a pillar cross-section without trajectories
    codes[2, 2] = -1
    out, filled, report = fill(codes, solid)
    assert (out[solid] == -1).all() and not filled[solid].any()
    assert out[2, 2] == 1 and report["failed"] == 1 and report["flow_cells"] == codes.size - 9
    # a failed cell walled in by solid except for one diagonal neighbour between two solid cells stays unfilled
    codes = section()
    solid = np.zeros(codes.shape, dtype=bool)
    solid[3:6, 2:5] = True
    solid[4, 3] = False
    solid[3, 4] = False  # the open diagonal neighbour; (3, 3) and (4, 4) are solid
    codes[4, 3] = -1
    out, filled, report = fill(codes, solid)
    assert out[4, 3] == -1 and not filled.any() and report["unfilled"] == 1
    solid[4, 4] = False  # an open orthogonal neighbour
    out, filled, _ = fill(codes, solid)
    assert out[4, 3] == 1 and filled[4, 3]


def test_the_fill_is_deterministic_and_mirror_equivariant():
    y, z = np.arange(60.0), np.arange(40.0)
    codes = np.ones((40, 60), dtype=np.int32)
    codes[:, 25:] = 2
    codes[15:, 45:] = 3
    codes[np.random.default_rng(3).random(codes.shape) < 0.008] = -1
    first = fill_trajectory_gaps(codes, None, y, z, 100)
    second = fill_trajectory_gaps(codes.copy(), None, y, z, 100)
    assert first[2]["filled"] > 5 and first[2]["unfilled"] == 0
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    mirrored = fill_trajectory_gaps(codes[:, ::-1].copy(), None, y, z, 100)
    assert np.array_equal(mirrored[0][:, ::-1], first[0])


def test_section_solid_marks_pillar_cells_at_the_station():
    cfg = ProjectConfig.load(ROOT / "examples" / "right_triangle_bottom.json")
    pillar = cfg.pillars[0]
    y = np.linspace(2.5, 297.5, 60) * UM
    z = np.linspace(2.5, 147.5, 30) * UM
    inside = section_solid(cfg, pillar.x + 10 * UM, y, z)
    assert inside.any() and not inside[z > pillar.height].any()
    assert not section_solid(cfg, pillar.x - 10 * UM, y, z).any()
    cfg.pillars[0] = PillarSpec(pillar.id, pillar.vertices, height=pillar.height, x=pillar.x, y=pillar.y, mount="top")
    top = section_solid(cfg, pillar.x + 10 * UM, y, z)
    assert top.any() and not top[z < cfg.channel.height - pillar.height].any()


# --------------------------------------------------------------------------- the solver on the examples

EXAMPLES = [e["file"][:-5] for e in json.loads((ROOT / "examples" / "index.json").read_text())["examples"]]


@pytest.fixture(scope="module")
def results():
    memory_cache().clear()
    return {name: simulate(ProjectConfig.load(ROOT / "examples" / f"{name}.json")) for name in EXAMPLES}


FAILED = ("wall_contact", "entered_solid", "stagnation", "max_length")


@pytest.mark.parametrize("name", EXAMPLES)
def test_examples_have_no_grey_cells_and_count_every_filled_one(results, name):
    result = results[name]
    assert result.ok, result.message
    assert not (result.final.codes < 0).any()
    for s in result.sections[1:]:
        status = s.diagnostics.get("trajectory_status")
        if status is None:  # a straight channel: no traced section
            continue
        failed = sum(status[k] for k in FAILED)
        assert status["filled_from_neighbours"] == failed == int(s.filled.sum())
        assert s.filled.mean() < 0.005
        assert np.all(s.status[s.filled] != 0)  # the raw status of a filled cell is kept
    assert not any("were not filled" in w for w in result.warnings)


def test_filled_cells_are_saved_and_the_raw_share_still_judges_the_numerics(results, tmp_path):
    result = results["two_nearby_pillars"]
    final = result.final
    assert final.filled.sum() > 0
    loaded = SimulationResult.load(result.save(tmp_path / "r.npz"))
    assert np.array_equal(loaded.final.filled, final.filled) and loaded.final.filled.dtype == bool
    check = next(c for c in result.validity["checks"] if c["check"] == "trajectories")
    assert check["detail"].startswith(f"{100 * final.filled.mean():.3f}% of cells without a valid trajectory")


def test_cached_and_fresh_maps_give_the_same_filled_section():
    cfg = ProjectConfig.load(ROOT / "examples" / "right_triangle_bottom.json")
    memory_cache().clear()
    fresh = simulate(cfg)
    reused = simulate(cfg)
    assert not fresh.diagnostics["map_reused"] and reused.diagnostics["map_reused"]
    assert fresh.final.filled.sum() > 0
    for a, b in zip(fresh.sections, reused.sections):
        assert np.array_equal(a.codes, b.codes)
        assert (a.filled is None and b.filled is None) or np.array_equal(a.filled, b.filled)
        assert a.diagnostics.get("trajectory_status") == b.diagnostics.get("trajectory_status")


def test_conservative_transport_fills_the_same_gaps():
    cfg = ProjectConfig.load(ROOT / "examples" / "right_triangle_bottom.json")
    cfg.model.transport = "conservative"
    result = simulate(cfg)
    final = result.final
    groups = result.diagnostics["conservative_transport"]["groups"]
    assert groups[-1]["empty_cells"] == groups[-1]["filled_from_neighbours"] == int(final.filled.sum()) > 0
    totals = final.fractions.sum(axis=0)
    assert np.allclose(totals, 1.0, atol=1e-5) and not (final.codes < 0).any()
    kk, jj = np.nonzero(final.filled)
    assert np.all(final.fractions[final.codes[kk, jj] - 1, kk, jj] == 1.0)
    assert final.diagnostics["unresolved_flux_fraction"] < 1e-9
