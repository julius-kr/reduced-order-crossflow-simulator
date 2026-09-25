"""Rectangular-duct profile: equivalence with the legacy modal sum, convergence, scaling."""
import time

import numpy as np
import pytest

from crossflow_simulator import physics
from crossflow_simulator.solvers.section import solve_axial_velocity


def legacy_double_loop(y, z, width, height, n_terms):
    """Independent re-statement of the legacy nested modal sum (normalized coordinates)."""
    Y, Z = np.meshgrid(y * width, z * height, indexing="xy")
    velocity = np.zeros_like(Y)
    for m in range(1, 2 * n_terms, 2):
        sin_y = np.sin(m * np.pi * Y / width)
        for n in range(1, 2 * n_terms, 2):
            coefficient = 16.0 / (np.pi**4 * m * n * ((m / width) ** 2 + (n / height) ** 2))
            velocity += coefficient * sin_y * np.sin(n * np.pi * Z / height)
    return velocity / np.trapezoid(np.trapezoid(velocity, y, axis=1), z)


@pytest.mark.parametrize("width,height", [(2.0, 1.0), (1.0, 1.0), (1.0, 3.0)])
def test_separable_series_equals_legacy_modal_sum(width, height):
    y, z = np.linspace(0, 1, 41), np.linspace(0, 1, 31)
    reference = legacy_double_loop(y, z, width, height, 30)
    fast = physics.rectangular_duct_profile(y * width, z * height, width, height, 30, "double_series", "trapezoid")
    assert np.max(np.abs(fast - reference)) <= 1e-10 * np.max(np.abs(reference))


def test_separable_series_is_much_faster_than_the_legacy_loop():
    y, z = np.linspace(0, 1, 120), np.linspace(0, 1, 80)
    started = time.perf_counter()
    legacy_double_loop(y, z, 2.0, 1.0, 40)
    legacy_time = time.perf_counter() - started
    physics._profile_cache.clear()
    started = time.perf_counter()
    physics.rectangular_duct_profile(y * 2.0, z, 2.0, 1.0, 40, "double_series", "trapezoid")
    fast_time = time.perf_counter() - started
    assert legacy_time / fast_time > 20


@pytest.mark.parametrize("width,height", [(3.0, 1.0), (1.0, 1.0), (0.5, 2.0)])
def test_single_series_agrees_with_double_series_and_vanishes_on_walls(width, height):
    y = np.linspace(0, width, 61)
    z = np.linspace(0, height, 41)
    single = physics.rectangular_duct_profile(y, z, width, height, 60, "single_series", "none")
    double = physics.rectangular_duct_profile(y, z, width, height, 400, "double_series", "none")
    assert np.max(np.abs(single - double)) < 2e-4 * single.max()
    # The single series cancels the parabola on the side walls with terms decaying like n^-3.
    for edge in (single[0], single[-1], single[:, 0], single[:, -1]):
        assert np.max(np.abs(edge)) < 2e-5 * single.max()


def test_analytic_normalization_gives_unit_mean_velocity():
    n = 400
    y = (np.arange(n) + 0.5) * (2.0 / n)
    z = (np.arange(n // 2) + 0.5) * (1.0 / (n // 2))
    u = physics.rectangular_duct_profile(y, z, 2.0, 1.0)
    assert u.mean() == pytest.approx(1.0, rel=1e-4)


def test_flux_integral_matches_parallel_plate_limit_and_symmetry():
    wide = physics.duct_flow_shape_integral(1000.0, 1.0) / 1000.0
    assert wide == pytest.approx(1.0 / 12.0, rel=1e-3)
    assert physics.duct_flow_shape_integral(2.0, 1.0) == pytest.approx(physics.duct_flow_shape_integral(1.0, 2.0))


def test_cached_profiles_are_read_only():
    y = np.linspace(0, 1, 11)
    u = physics.rectangular_duct_profile(y, y, 1.0, 1.0)
    with pytest.raises(ValueError):
        u[0, 0] = 1.0


def test_resistance_matches_finite_volume_solution():
    W, H, mu = 300e-6, 150e-6, 0.02
    edges_y, edges_z = np.linspace(0, W, 161), np.linspace(0, H, 81)
    flow = solve_axial_velocity(edges_y, edges_z, mu, pressure_gradient=1000.0)
    expected = 1000.0 / physics.duct_resistance(1.0, W, H, mu)
    assert flow.flow_rate == pytest.approx(expected, rel=2e-3)


def test_reynolds_number():
    re = physics.reynolds_number(1e-9, 1e-4, 1e-4, 1000.0, 1e-3)
    assert re == pytest.approx(1000.0 * 0.1 * 1e-4 / 1e-3)
