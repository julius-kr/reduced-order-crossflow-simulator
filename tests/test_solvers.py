"""Cross-section solver against analytical solutions."""
import numpy as np
import pytest

from crossflow_simulator import physics
from crossflow_simulator.solvers.section import flux_matched_velocity, solve_axial_velocity


# ----------------------------------------------------------------------------- cross-section (variable viscosity)

def test_section_solver_converges_at_second_order():
    W, H = 2.0, 1.0
    errors = []
    for n in (16, 32, 64):
        yf, zf = np.linspace(0, W, 2 * n + 1), np.linspace(0, H, n + 1)
        flow = solve_axial_velocity(yf, zf, 1.0)
        yc, zc = 0.5 * (yf[1:] + yf[:-1]), 0.5 * (zf[1:] + zf[:-1])
        exact = physics.rectangular_duct_profile(yc, zc, W, H, normalization="none")
        errors.append(np.max(np.abs(flow.velocity - exact)) / exact.max())
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > 1.8)


def test_two_layer_viscosity_matches_analytic_plane_flow():
    """Wide channel, two horizontal layers: compare the centreline to layered plane Poiseuille flow."""
    W, H, G = 40.0, 1.0, 1.0
    mu1, mu2, h1 = 4.0, 1.0, 0.4
    ny, nz = 20, 400
    yf, zf = np.linspace(0, W, ny + 1), np.linspace(0, H, nz + 1)
    zc = 0.5 * (zf[1:] + zf[:-1])
    mu = np.where(zc[:, None] < h1, mu1, mu2) * np.ones((1, ny))
    flow = solve_axial_velocity(yf, zf, mu, G)
    # Layer 1: u = -G z^2 / (2 mu1) + c1 z; layer 2: u = -G z^2 / (2 mu2) + c2 z + d2.
    # Conditions: continuity of u and of the shear stress at h1, no slip at z = H.
    M = np.array([[h1, -h1, -1.0], [mu1, -mu2, 0.0], [0.0, H, 1.0]])
    rhs = np.array([G * h1**2 / (2 * mu1) - G * h1**2 / (2 * mu2), 0.0, G * H**2 / (2 * mu2)])
    c1, c2, d2 = np.linalg.solve(M, rhs)
    exact = np.where(zc < h1, -G * zc**2 / (2 * mu1) + c1 * zc, -G * zc**2 / (2 * mu2) + c2 * zc + d2)
    centre = flow.velocity[:, ny // 2]
    assert np.max(np.abs(centre - exact)) < 5e-3 * exact.max()


def test_flux_matched_profile_carries_the_rate():
    yf, zf = np.linspace(0, 3e-4, 61), np.linspace(0, 1.5e-4, 31)
    flow = flux_matched_velocity(yf, zf, 0.02, 3e-11)
    assert flow.flow_rate == pytest.approx(3e-11)
    assert np.sum(flow.velocity * flow.cell_area) == pytest.approx(3e-11)

