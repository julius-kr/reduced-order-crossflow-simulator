"""Lubrication-pressure and hydraulic-network solvers against analytical solutions."""
import numpy as np
import pytest

from crossflow_simulator import physics
from crossflow_simulator.solvers.network import solve_network
from crossflow_simulator.solvers.pressure import solve_lubrication


# ----------------------------------------------------------------------------- lubrication pressure

def plates(nx=40, ny=20, length=2e-3, width=1e-3, gap=1e-4):
    x_edges, y_edges = np.linspace(0, length, nx + 1), np.linspace(0, width, ny + 1)
    return x_edges, y_edges, np.full((ny, nx), gap)


def test_constant_gap_gives_linear_pressure_and_exact_flux():
    x_edges, y_edges, gap = plates()
    mu, dp = 1e-3, 100.0
    sol = solve_lubrication(gap, x_edges, y_edges, mu, inlet_pressure=dp, outlet_pressure=0.0)
    xc = 0.5 * (x_edges[1:] + x_edges[:-1])
    assert np.allclose(sol.pressure, dp * (1 - xc / x_edges[-1])[None, :], rtol=0, atol=1e-10 * dp)
    expected = 1e-3 * 1e-4**3 / (12 * mu) * dp / 2e-3
    assert sol.flow_rate == pytest.approx(expected, rel=1e-12)
    assert sol.relative_residual <= 1e-8 and sol.flux_imbalance <= 1e-6 and sol.converged


def test_result_is_independent_of_grid_aspect_ratio():
    """Physical spacings: the answer must not depend on nx/ny (the legacy solver ignored spacing)."""
    rates = []
    for nx, ny in ((40, 20), (20, 60), (80, 10)):
        x_edges, y_edges, gap = plates(nx, ny)
        rates.append(solve_lubrication(gap, x_edges, y_edges, 1e-3, 100.0).flow_rate)
    assert np.ptp(rates) <= 1e-10 * rates[0]


def test_discontinuous_gap_uses_harmonic_transmissibility():
    """Two gap heights in series along x: exact series-resistance flow rate on a matching grid."""
    x_edges, y_edges, gap = plates(nx=40)
    gap[:, 20:] = 0.5e-4
    mu, dp = 1e-3, 100.0
    sol = solve_lubrication(gap, x_edges, y_edges, mu, dp)
    k1, k2 = 1e-4**3 / (12 * mu), 0.5e-4**3 / (12 * mu)
    resistance = (1e-3 / k1 + 1e-3 / k2) / 1e-3
    assert sol.flow_rate == pytest.approx(dp / resistance, rel=1e-12)


def test_zero_driving_pressure_gives_zero_flow():
    x_edges, y_edges, gap = plates()
    sol = solve_lubrication(gap, x_edges, y_edges, 1e-3, inlet_pressure=5.0, outlet_pressure=5.0)
    unit_flux = 1e-3 * 1e-4**3 / (12 * 1e-3) / 2e-3  # flow for 1 Pa across the plates
    assert abs(sol.flow_rate) < 1e-12 * unit_flux
    assert np.allclose(sol.pressure, 5.0)


def test_flow_rate_mode_and_factorization_reuse():
    x_edges, y_edges, gap = plates()
    gap[5:15, 10:20] = 0.3e-4
    first = solve_lubrication(gap, x_edges, y_edges, 1e-3, inlet_pressure=None, flow_rate=1e-12)
    second = solve_lubrication(gap, x_edges, y_edges, 1e-3, inlet_pressure=None, flow_rate=2e-12)
    assert first.flow_rate == pytest.approx(1e-12)
    assert second.pressure_drop == pytest.approx(2 * first.pressure_drop)
    assert second.timings["factorization_cached"] == 1.0
    assert first.flux_imbalance < 1e-10


def test_depth_integrated_flux_and_mean_velocity_are_distinct():
    x_edges, y_edges, gap = plates()
    sol = solve_lubrication(gap, x_edges, y_edges, 1e-3, 100.0)
    qx, _ = sol.depth_integrated_flux()
    ux, _ = sol.mean_velocity()
    assert np.allclose(ux, qx / 1e-4)
    assert np.allclose(qx.sum(axis=0) * (y_edges[1] - y_edges[0]), sol.flow_rate)


def test_isolated_fluid_regions_are_reported():
    x_edges, y_edges, gap = plates()
    gap[:, 18:22] = 0.0
    gap[8:12, 18:22] = 1e-4  # opening in the wall
    gap[0:3, 5:8] = 0.0  # small obstacle
    sol = solve_lubrication(gap, x_edges, y_edges, 1e-3, 100.0)
    assert sol.isolated_cells == 0
    enclosed = gap.copy()
    enclosed[2:8, 2:8] = 0.0
    enclosed[3:7, 3:7] = 1e-4
    assert solve_lubrication(enclosed, x_edges, y_edges, 1e-3, 100.0).isolated_cells == 16


# ----------------------------------------------------------------------------- hydraulic network

def test_network_matches_hand_calculation():
    W, H, L, mu = 300e-6, 150e-6, 2e-3, 0.02
    R_tube = 5e13
    inlets = [{"id": "a", "station": 0.0, "pressure": 2000.0, "resistance": R_tube, "viscosity": mu}]
    sol = solve_network(W, H, L, 0.0, inlets)
    R_channel = physics.duct_resistance(L, W, H, mu)
    assert sol.inlet_rates["a"] == pytest.approx(2000.0 / (R_tube + R_channel), rel=1e-10)


def test_network_rejects_backflow():
    W, H, L, mu = 300e-6, 150e-6, 2e-3, 0.02
    inlets = [{"id": "a", "station": 0.0, "pressure": 5000.0, "resistance": 1e12, "viscosity": mu},
              {"id": "b", "station": 1e-3, "pressure": 10.0, "resistance": 1e12, "viscosity": mu}]
    with pytest.raises(ValueError, match="backflow"):
        solve_network(W, H, L, 0.0, inlets)
