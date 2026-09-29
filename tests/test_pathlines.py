"""Forward pathlines for the 3D view: display data, but they must still be physically consistent."""
import numpy as np
import pytest
from matplotlib.path import Path as PolygonPath

from crossflow_simulator import physics
from crossflow_simulator.cache import memory_cache
from crossflow_simulator.config import InletSpec, PillarSpec
from crossflow_simulator.results import SimulationResult
from crossflow_simulator.simulator import simulate
from crossflow_simulator.transport import REACHED_PLANE

from conftest import UL_MIN, UM, make_config

TRIANGLE = [(0.0, 0.0), (150 * UM, 0.0), (0.0, 120 * UM)]


def valid_samples(lines, i):
    row, time = lines.points[i].astype(float), lines.time[i].astype(float)
    keep = np.isfinite(time)
    return row[keep], time[keep]


def test_straight_channel_pathlines_are_straight_and_timed_by_the_developed_profile():
    cfg = make_config()
    cfg.channel.length = 1500 * UM
    result = simulate(cfg)
    lines = result.pathlines
    W, H = cfg.channel.width, cfg.channel.height
    total = sum(i.flow_rate for i in cfg.inlets)
    assert lines.outlet_x == pytest.approx(1500 * UM)
    assert np.all(lines.status == REACHED_PLANE)
    for i in range(0, len(lines.code), 17):
        points, time = valid_samples(lines, i)
        assert points[0, 0] == pytest.approx(0.0) and points[-1, 0] == pytest.approx(1500 * UM)
        assert np.ptp(points[:, 1]) == 0.0 and np.ptp(points[:, 2]) == 0.0
        y, z = points[0, 1], points[0, 2]
        u = physics.rectangular_duct_profile(np.array([y]), np.array([z]), W, H)[0, 0] * total / (W * H)
        assert time[-1] == pytest.approx(1500 * UM / u, rel=0.02)


def test_seeds_are_flux_weighted_and_coloured_by_the_inlet_pattern():
    result = simulate(make_config(rates=(1.0, 3.0)))
    lines = result.pathlines
    # Layers in z carrying 1:3 of the flow: the seed counts follow the flux, not the area.
    share = np.mean(lines.code == 2)
    assert share == pytest.approx(0.75, abs=0.06)
    lower = lines.points[lines.code == 1, 0, 2]
    upper = lines.points[lines.code == 2, 0, 2]
    assert lower.max() < upper.min()


def test_port_pathlines_start_in_the_opening_and_leave_through_the_outlet():
    cfg = make_config(rates=(2.0,), layout="single")
    cfg.inlets.append(InletSpec("p1", "B", "port", flow_rate=2 * UL_MIN, station=500 * UM, wall="top",
                                position=150 * UM, diameter=100 * UM))
    result = simulate(cfg)
    lines = result.pathlines
    port = lines.origin == 1
    assert port.any() and np.all(lines.code[port] == 2) and np.all(lines.code[~port] == 1)
    assert port.mean() == pytest.approx(0.5, abs=0.06)  # half of the flow enters through the port
    starts = np.array([valid_samples(lines, i)[0][0] for i in np.nonzero(port)[0]])
    assert np.all(np.hypot(starts[:, 0] - 500 * UM, starts[:, 1] - 150 * UM) <= 50 * UM + 1e-9)
    assert np.all(starts[:, 2] > 0.99 * cfg.channel.height)
    assert np.mean(lines.status == REACHED_PLANE) > 0.97


def test_pathlines_go_around_a_partial_pillar():
    pillar = PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM)
    cfg = make_config(pillars=[pillar])
    result = simulate(cfg)
    lines = result.pathlines
    points = lines.points.reshape(-1, 3).astype(float)
    points = points[np.isfinite(points[:, 0])]
    inside = PolygonPath(pillar.footprint()).contains_points(points[:, :2]) & (points[:, 2] < pillar.height)
    assert inside.mean() < 1e-3  # only staircase corners of the voxelized footprint
    assert np.mean(lines.status == REACHED_PLANE) > 0.97
    # Fluid passes over the partial-height pillar, above its top.
    over = PolygonPath(pillar.footprint()).contains_points(points[:, :2])
    assert over.sum() > 50
    assert np.mean(points[over, 2] > pillar.height) > 0.98


def test_pathlines_survive_save_and_load(tmp_path):
    result = simulate(make_config(pillars=[PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM)]))
    loaded = SimulationResult.load(result.save(tmp_path / "r.npz"))
    for name in ("points", "time"):
        assert np.array_equal(getattr(loaded.pathlines, name), getattr(result.pathlines, name), equal_nan=True)
    for name in ("code", "status", "origin"):
        assert np.array_equal(getattr(loaded.pathlines, name), getattr(result.pathlines, name))
    assert loaded.pathlines.outlet_x == result.pathlines.outlet_x


def test_reused_maps_recolour_pathlines_and_rescale_their_times():
    base = make_config(pillars=[PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM)])
    memory_cache().clear()
    first = simulate(base)
    faster = base.copy()
    for inlet in faster.inlets:
        inlet.flow_rate *= 2.0
    reused = simulate(faster)
    assert reused.diagnostics["map_reused"]
    assert np.allclose(reused.pathlines.time, first.pathlines.time / 2.0, rtol=1e-5, equal_nan=True)
    swapped = base.copy()
    swapped.inlets[0].pattern_row, swapped.inlets[1].pattern_row = 1, 0
    recoloured = simulate(swapped)
    assert recoloured.diagnostics["map_reused"]
    assert np.array_equal(recoloured.pathlines.code, 3 - first.pathlines.code)


def test_pathline_failure_never_fails_the_calculation(monkeypatch):
    import crossflow_simulator.simulator as simulator

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(simulator, "compute_pathlines", broken)
    memory_cache().clear()
    result = simulate(make_config())
    assert result.ok and result.pathlines is None
    assert any("3D pathlines unavailable" in w for w in result.warnings)
