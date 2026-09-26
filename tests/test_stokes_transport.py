"""3D Stokes solver and streamline integration."""
import numpy as np
import pytest

from crossflow_simulator import physics
from crossflow_simulator.solvers.section import flux_matched_velocity
from crossflow_simulator.solvers.stokes import RectilinearGrid, StokesProblem, StokesSolution, WallOpening
from crossflow_simulator.transport import REACHED_PLANE, StaggeredInterpolator, trace


def duct(n: int, length: float = 3.0, W: float = 2.0, H: float = 1.0, solid=None, openings=(), mu=1.0):
    grid = RectilinearGrid(np.linspace(0, length, int(length * n) + 1), np.linspace(0, W, int(W * n) + 1),
                           np.linspace(0, H, int(H * n) + 1))
    solid = np.zeros(grid.shape, bool) if solid is None else solid(grid)
    return grid, StokesProblem(grid, solid, mu, openings)


def developed(grid, rate=1.0, mu=1.0):
    return flux_matched_velocity(grid.yf, grid.zf, mu, rate).velocity.T


@pytest.fixture(scope="module")
def straight():
    grid, problem = duct(8)
    profile = developed(grid)
    solution = problem.solve(problem.boundary_values(inflow=profile, outflow=profile))
    return grid, profile, solution


def test_straight_duct_is_discretely_x_invariant(straight):
    grid, profile, solution = straight
    assert np.max(np.abs(solution.u - profile[None])) < 1e-6 * profile.max()
    assert np.max(np.abs(solution.v)) < 1e-6 * profile.max()
    assert np.max(np.abs(solution.w)) < 1e-6 * profile.max()
    assert solution.info["relative_residual"] < 1e-6
    assert np.max(np.abs(solution.divergence())) < 1e-8


def test_duct_flow_converges_to_the_analytical_profile():
    errors = []
    for n in (8, 16):
        grid, problem = duct(n, length=0.5)
        profile = developed(grid)
        solution = problem.solve(problem.boundary_values(inflow=profile, outflow=profile))
        exact = physics.rectangular_duct_profile(grid.centers[1], grid.centers[2], 2.0, 1.0).T / 2.0
        errors.append(np.max(np.abs(solution.u[len(grid.xf) // 2] - exact)) / exact.max())
    assert np.log2(errors[0] / errors[1]) > 1.8


def test_pressure_drop_matches_duct_resistance(straight):
    grid, _, solution = straight
    length = grid.centers[0][-1] - grid.centers[0][0]
    exact = physics.duct_pressure_gradient(1.0, 2.0, 1.0, 1.0) * length
    assert solution.pressure_drop() == pytest.approx(exact, rel=0.07)  # 8 cells across H; O(h^2)


def test_viscosity_scales_pressure_not_velocity():
    grid, p1 = duct(6, length=2.0, solid=lambda g: _block(g))
    _, p2 = duct(6, length=2.0, solid=lambda g: _block(g), mu=7.0)
    profile = developed(grid)
    s1 = p1.solve(p1.boundary_values(inflow=profile, outflow=profile))
    s2 = p2.solve(p2.boundary_values(inflow=profile, outflow=profile))
    assert np.max(np.abs(s1.u - s2.u)) < 1e-6 * np.max(np.abs(s1.u))
    assert s2.pressure_drop() == pytest.approx(7.0 * s1.pressure_drop(), rel=1e-6)


def _block(grid):
    xc, yc, zc = grid.centers
    return ((np.abs(xc - 1.0) < 0.3)[:, None, None] & (yc < 0.9)[None, :, None] & (zc < 0.6)[None, None, :])


def test_port_injection_balances_flux():
    grid = RectilinearGrid(np.linspace(0, 3, 37), np.linspace(0, 2, 25), np.linspace(0, 1, 13))
    xc, yc = grid.centers[0], grid.centers[1]
    r2 = ((xc[:, None] - 1.0) ** 2 + (yc[None, :] - 1.0) ** 2) / 0.4**2
    profile = np.where(r2 < 1, 1 - r2, 0.0)
    area = np.diff(grid.xf)[:, None] * np.diff(grid.yf)[None, :]
    profile = profile / (profile * area).sum()
    problem = StokesProblem(grid, np.zeros(grid.shape, bool), 1.0, [WallOpening("top", profile, "p")])
    inflow, outflow = developed(grid, 1.0), developed(grid, 1.5)
    solution = problem.solve(problem.boundary_values(inflow=inflow, outflow=outflow, opening_rates=[0.5]))
    assert solution.plane_flux(0) == pytest.approx(1.0)
    assert solution.plane_flux(-1) == pytest.approx(1.5)
    assert np.max(np.abs(solution.divergence())) < 1e-7


def test_lateral_mirror_symmetry_of_the_flow():
    def solid(g):
        xc, yc, zc = g.centers
        return (np.abs(xc - 1.5)[:, None, None] < 0.3) & (np.abs(yc - 1.0)[None, :, None] < 0.35) & (zc < 0.5)[None, None, :]

    grid, problem = duct(8, solid=solid)
    profile = developed(grid)
    s = problem.solve(problem.boundary_values(inflow=profile, outflow=profile))
    assert np.max(np.abs(s.u - s.u[:, ::-1, :])) < 1e-6 * s.u.max()
    assert np.max(np.abs(s.v + s.v[:, ::-1, :])) < 1e-6 * s.u.max()


# ----------------------------------------------------------------------------- transport

def uniform_solution(u0=1.0, v0=0.0, w0=0.0):
    grid = RectilinearGrid(np.linspace(0, 2, 21), np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    solid = np.zeros(grid.shape, bool)
    return StokesSolution(grid, solid, np.full(grid.face_shape(0), u0), np.full(grid.face_shape(1), v0),
                          np.full(grid.face_shape(2), w0), np.zeros(grid.shape))


def test_subpixel_displacement_accumulates_instead_of_freezing():
    """Legacy rounding erased sub-pixel motion; continuous coordinates must accumulate it exactly."""
    solution = uniform_solution(v0=0.001)
    interp = StaggeredInterpolator(solution)
    # Stay away from walls and the developed inflow/outflow planes, where the ghost conditions
    # (v = w = 0) do not match this synthetic uniform field.
    seeds = np.array([[1.8, 0.5, 0.5], [1.8, 0.45, 0.55]])
    result = trace(interp, seeds, stop_x=0.2, step=0.05)
    assert np.all(result.status == REACHED_PLANE)
    assert np.allclose(result.end[:, 1], seeds[:, 1] - 0.0016, atol=1e-9)
    assert np.allclose(result.end[:, 2], seeds[:, 2])


def test_zero_forcing_is_identity(straight):
    grid, _, solution = straight
    interp = StaggeredInterpolator(solution)
    rng = np.random.default_rng(3)
    yz = rng.uniform([0.05, 0.05], [1.95, 0.95], size=(300, 2))
    seeds = np.column_stack([np.full(300, grid.xf[-1]), yz])
    result = trace(interp, seeds, stop_x=grid.xf[0])
    assert np.all(result.status == REACHED_PLANE)
    assert np.max(np.abs(result.end[:, 1:] - yz)) < 1e-9


def test_integration_accuracy_converges(straight):
    """Tighter tolerance -> smaller error against an accurate reference in a curved field."""
    def solid(g):
        xc, yc, zc = g.centers
        return (np.abs(xc - 1.5)[:, None, None] < 0.25) & (yc < 1.2)[None, :, None] & (zc < 0.55)[None, None, :]

    grid, problem = duct(8, solid=solid)
    profile = developed(grid)
    interp = StaggeredInterpolator(problem.solve(problem.boundary_values(inflow=profile, outflow=profile)))
    rng = np.random.default_rng(5)
    seeds = np.column_stack([np.full(200, grid.xf[-1]), rng.uniform(0.2, 1.8, 200), rng.uniform(0.1, 0.9, 200)])
    reference = trace(interp, seeds, grid.xf[0], integrator="rk4", step=0.005)
    errors = []
    for tolerance in (1e-2, 1e-3, 1e-4):
        result = trace(interp, seeds, grid.xf[0], error_tolerance=tolerance)
        ok = (result.status == REACHED_PLANE) & (reference.status == REACHED_PLANE)
        errors.append(np.mean(np.linalg.norm(result.end[ok, 1:] - reference.end[ok, 1:], axis=1)))
    assert errors[0] > errors[1] > errors[2]


def test_trajectories_never_leave_the_channel(straight):
    grid, _, solution = straight
    interp = StaggeredInterpolator(solution)
    seeds = np.column_stack([np.full(4, grid.xf[-1]), [1e-4, 2.0 - 1e-4, 1.0, 1.0], [0.5, 0.5, 1e-4, 1.0 - 1e-4]])
    result = trace(interp, seeds, grid.xf[0])
    assert np.all(result.status == REACHED_PLANE)
    assert np.all((result.end[:, 1] > 0) & (result.end[:, 1] < 2.0) & (result.end[:, 2] > 0) & (result.end[:, 2] < 1.0))

