"""CFD reference: convective operator, full-channel solves, pressure drive, caching, and accuracy metrics."""
import json

import numpy as np
import pytest

from crossflow_simulator import cache as cache_mod
from crossflow_simulator import cfd, metrics, physics
from crossflow_simulator.comparison import BackendRun, ComparisonService, accuracy_against, build_comparison, \
    export_comparison
from crossflow_simulator.config import InletSpec, OperatingConditions, PillarSpec, ProjectConfig
from crossflow_simulator.models.domain import port_opening
from crossflow_simulator.simulator import simulate
from crossflow_simulator.solvers.convection import convective_acceleration
from crossflow_simulator.solvers.stokes import RectilinearGrid, StokesProblem, StokesSolution
from crossflow_simulator.transport import REACHED_PLANE

from conftest import ROOT, UM, make_config

TRIANGLE = [(0.0, 0.0), (150 * UM, 0.0), (0.0, 120 * UM)]


def small(cfg, cells=8, check=True, inertia=True):
    cfg.cfd.cells = cells
    cfg.cfd.convergence_check = check
    cfg.cfd.inertia = inertia
    return cfg


@pytest.fixture
def disk_cache(tmp_path, monkeypatch):
    """A real, enabled disk cache in a temporary directory (the suite disables it by default)."""
    store = cache_mod.DiskCache(tmp_path / "cache")
    store.enabled = True
    monkeypatch.setattr(cache_mod, "_disk", store)
    return store


# ----------------------------------------------------------------------------- operators


def _analytic(x, y, z):
    u = np.sin(np.pi * y) * np.sin(np.pi * z) * (1 + 0.5 * np.cos(2 * np.pi * x))
    v = 0.3 * np.sin(2 * np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    w = -0.2 * np.cos(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    return u, v, w


def test_convective_operator_converges_at_second_order():
    errors = []
    for n in (16, 32):
        grid = RectilinearGrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
        fields = [_analytic(*np.meshgrid(*grid.face_coordinates(c), indexing="ij"))[c] for c in range(3)]
        acceleration = convective_acceleration(StokesSolution(grid, np.zeros(grid.shape, bool), *fields,
                                                              p=np.zeros(grid.shape)))
        error = 0.0
        for c in range(3):
            X, Y, Z = np.meshgrid(*grid.face_coordinates(c), indexing="ij")
            h = 1e-6
            vel = _analytic(X, Y, Z)
            exact = 0.0
            for d, (dx, dy, dz) in enumerate(((h, 0, 0), (0, h, 0), (0, 0, h))):
                exact = exact + vel[d] * (_analytic(X + dx, Y + dy, Z + dz)[c] - _analytic(X - dx, Y - dy, Z - dz)[c]) / (2 * h)
            inner = (slice(1, -1),) * 3
            error = max(error, float(np.max(np.abs(acceleration[c][inner] - exact[inner]))))
        errors.append(error)
    assert errors[0] / errors[1] > 3.5


def test_two_openings_on_one_wall_both_inject():
    grid = RectilinearGrid(np.linspace(0, 1e-3, 41), np.linspace(0, 3e-4, 13), np.linspace(0, 1.5e-4, 7))
    ports = [InletSpec(f"p{k}", "A", "port", flow_rate=1e-11, station=s, wall="top", position=1.5e-4, diameter=1e-4)
             for k, s in enumerate((3e-4, 7e-4))]
    openings = [port_opening(grid, port)[0] for port in ports]
    problem = StokesProblem(grid, np.zeros(grid.shape, bool), 1.0, openings)
    values = problem.boundary_values(opening_rates=[2.0, 3.0])
    top = values[problem.face_index[2][:, :, -1].ravel()].reshape(grid.shape[0], grid.shape[1])
    area = grid.face_area(2)[:, :, 0]
    assert -np.sum(top * area) == pytest.approx(5.0)


# ----------------------------------------------------------------------------- full-channel solves


def test_straight_channel_is_exact_including_inertia():
    cfg = small(make_config(rates=(1.0, 2.0, 3.0), layout="layers_y"), cells=12)
    cfg.channel.length = 800 * UM
    result = cfd.run_cfd(cfg, use_cache=False)
    assert result.ok, result.message
    first, final = result.sections[0], result.sections[-1]
    assert np.array_equal(first.codes, final.codes)
    for level in result.diagnostics["cfd"]["levels"]:
        assert level["picard"]["inertia_effect"] < 1e-8  # developed flow has no convective acceleration
        assert level["mass_balance"]["relative_error"] < 1e-6
    fractions = final.diagnostics["flux_fraction"]
    assert fractions[1] == pytest.approx(1 / 6, abs=0.006) and fractions[3] == pytest.approx(0.5, abs=0.006)
    assert result.diagnostics["cfd"]["mesh_convergence"]["uncertainty_m"] == 0.0
    assert result.validity["numerics_ok"]
    lines = result.pathlines
    assert lines is not None and np.all(lines.status == REACHED_PLANE)
    assert np.nanmax(lines.points[:, :, 0]) == pytest.approx(800 * UM)


def test_pressure_drive_recovers_the_duct_resistance():
    cfg = make_config(rates=(1.0,), layout="single")
    cfg.channel.length = 1500 * UM
    cfg.operating = OperatingConditions(drive="pressure", outlet_pressure=50.0)
    inlet = cfg.inlets[0]
    inlet.flow_rate, inlet.pressure, inlet.tubing_resistance = None, 2000.0, 5e12
    small(cfg, cells=12, check=False, inertia=False)
    result = cfd.run_cfd(cfg, use_cache=False)
    assert result.ok, result.message
    mu = cfg.materials[0].viscosity
    expected = (2000.0 - 50.0) / (5e12 + physics.duct_resistance(1500 * UM, cfg.channel.width, cfg.channel.height, mu))
    rate = result.diagnostics["network"]["inlet_rates_m3_s"]["in1"]
    assert rate == pytest.approx(expected, rel=0.02)


def test_unsupported_cases_are_explained():
    cfg = make_config()
    cfg.operating = OperatingConditions(drive="pressure")
    cfg.materials[1].viscosity *= 4
    assert any("viscosity contrast" in r for r in cfd.unsupported(cfg))


def pillar_case(**kwargs):
    cfg = make_config(pillars=[PillarSpec("tri", TRIANGLE, height=90 * UM, x=250 * UM, y=40 * UM)], **kwargs)
    cfg.channel.length = 1000 * UM
    return cfg


def test_inertia_correction_scales_with_density():
    effects = []
    for density in (1000.0, 2000.0):
        cfg = small(pillar_case(), cells=8, check=False)
        for material in cfg.materials:
            material.density = density
        flow = cfd.solve_level(cfg, cfd.build_domain(cfg, 8), use_cache=False)
        assert flow.info["picard"]["converged"]
        effects.append(flow.info["picard"]["inertia_effect"])
    assert effects[1] / effects[0] == pytest.approx(2.0, rel=0.1)  # first order in Re at creeping flow


def test_viscosity_contrast_couples_the_material_field():
    cfg = small(pillar_case(), cells=12, check=False)  # 8 cells cannot resolve a 4:1 viscosity interface
    cfg.materials[0].viscosity = 4 * cfg.materials[1].viscosity  # lower layer 4x more viscous, equal flow rates
    result = cfd.run_cfd(cfg, use_cache=False)
    assert result.ok, result.message
    level = result.diagnostics["cfd"]["levels"][0]
    assert level["coupling"]["converged"] and level["picard"]["converged"]
    area = result.final.cell_area()
    viscous = float(area[result.final.codes == 1].sum() / area.sum())
    assert viscous > 0.55  # the slower, more viscous layer needs more of the section for the same flow
    assert result.final.diagnostics["flux_fraction"][1] == pytest.approx(0.5, abs=0.01)


def test_new_solver_agrees_with_the_cfd_reference_on_a_partial_pillar():
    cfg = small(pillar_case(), cells=12)
    reference = cfd.run_cfd(cfg, use_cache=False)
    assert reference.ok, reference.message
    mesh = reference.diagnostics["cfd"]["mesh_convergence"]
    assert mesh["cells"] == [8, 12] and mesh["uncertainty_m"] > 0
    # paths trapped at a pillar's stagnation line are restarted and reported, not lost
    assert "restarted_at_stagnation" in reference.final.diagnostics["trajectory_status"]
    assert np.mean(reference.final.codes < 0) < 0.001
    assert reference.final.diagnostics["unresolved_flux_fraction"] < 0.001
    assert [s.x for s in reference.sections] == pytest.approx([s.x for s in simulate(cfg).sections])
    new = simulate(cfg)
    accuracy = accuracy_against(reference, new)
    assert accuracy["min_iou"] > 0.95
    assert accuracy["max_interface_shift_m"] < 0.02 * cfg.channel.height + mesh["uncertainty_m"]
    assert accuracy["within_targets"]


# ----------------------------------------------------------------------------- keys, caching, metrics


def test_cache_keys_follow_what_the_flow_depends_on():
    cfg = pillar_case()
    base_field, base_result = cfd.field_key(cfg, 8), cfd.result_key(cfg)
    other = cfg.copy()
    other.model.fidelity = "reference"
    other.display.length_unit = "mm"
    assert cfd.result_key(other) == base_result  # the new solver's settings do not matter
    other.materials[0].color = "#123456"
    assert cfd.field_key(other, 8) == base_field and cfd.result_key(other) != base_result
    flow = cfg.copy()
    flow.inlets[0].flow_rate *= 2
    assert cfd.field_key(flow, 8) != base_field
    moved = cfg.copy()
    moved.pillars[0].x += 10 * UM
    assert cfd.field_key(moved, 8) != base_field
    assert cfg.input_hash() == small(cfg.copy(), cells=16).input_hash()  # CFD settings stay out of the new hash


def test_results_are_cached_and_restamped(disk_cache):
    cfg = small(pillar_case(), cells=8, check=False)
    first = cfd.run_cfd(cfg)
    assert first.ok and first.diagnostics["cache"] == "computed"
    assert cfd.cached_result(cfg) is not None
    edited = cfg.copy()
    edited.model.fidelity = "reference"
    again = cfd.run_cfd(edited)
    assert again.diagnostics["cache"].startswith("disk cache") and again.input_hash == edited.input_hash()
    assert cfd.is_current(again, edited)
    assert np.array_equal(again.final.codes, first.final.codes)
    changed = cfg.copy()
    changed.inlets[0].flow_rate *= 1.5
    assert not cfd.is_current(again, changed) and cfd.cached_result(changed) is None
    # Recompute: the stored result and flow fields are deleted and everything is computed and stored again
    fresh = cfd.run_cfd(cfg, recompute=True)
    assert fresh.ok and fresh.diagnostics["cache"] == "recomputed (2 stored entries cleared)"
    assert fresh.diagnostics["computed_seconds"] == fresh.timings["total"]
    assert np.array_equal(fresh.final.codes, first.final.codes)
    stored = cfd.cached_result(cfg)
    assert stored is not None and stored.diagnostics["computed_seconds"] == fresh.diagnostics["computed_seconds"]
    assert cfd.forget(cfg) == 2 and cfd.cached_result(cfg) is None and cfd.forget(cfg) == 0


def test_interface_shift_is_the_mean_normal_displacement():
    z = np.arange(40)[:, None] * np.ones((1, 60))
    a = np.where(z < 20, 1, 2)
    b = np.where(z < 23, 1, 2)
    assert metrics.interface_shift(a, b, 1, (0.5, 1.0)) == pytest.approx(1.5)  # 3 rows of 0.5
    assert metrics.interface_shift(a, a, 1, (0.5, 1.0)) == 0.0


def test_accuracy_of_a_result_against_itself_is_perfect():
    result = simulate(make_config(rates=(1.0, 2.0)))
    accuracy = accuracy_against(result, result)
    assert accuracy["min_iou"] == 1.0 and accuracy["max_interface_shift_m"] == 0.0
    assert accuracy["within_targets"]


# ----------------------------------------------------------------------------- worker and comparison


def test_cfd_backend_in_a_worker_with_comparison_and_export(tmp_path):
    cfg = small(ProjectConfig.load(ROOT / "examples" / "right_triangle_bottom.json"), cells=8)
    cfg.grid.output_ny = 64
    service = ComparisonService(tmp_path / "work")
    try:
        comparison = service.run_set(cfg, ("new", "cfd"))
    finally:
        service.close()
    runs = comparison.runs
    assert runs["new"].status == "success" and runs["cfd"].status == "success", runs["cfd"].message
    assert runs["cfd"].result.backend == "cfd" and runs["cfd"].memory.get("peak_bytes")
    accuracy = comparison.accuracy["new"]
    assert accuracy["min_iou"] > 0.9 and accuracy["speed"]["value"] > 1
    out = export_comparison(comparison, tmp_path / "export")
    for name in ("cfd_result.npz", "new_result.npz", "side_by_side.png", "comparison.json"):
        assert (out / name).exists(), name
    summary = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert summary["accuracy_vs_cfd"]["new"]["within_targets"] in (True, False)


def test_comparison_without_cfd_has_no_accuracy():
    result = simulate(make_config())
    comparison = build_comparison(make_config(), {"new": BackendRun("new", "success", result=result)})
    assert comparison.accuracy == {}
