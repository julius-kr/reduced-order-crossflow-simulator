"""Rectangular-duct flow, flux integration, and nondimensional groups.

Two representations of fully developed laminar flow in a ``W x H`` duct are
provided:

``double_series``
    The truncated double sine series used by the legacy code, evaluated as
    ``U = S_z @ C.T @ S_y.T`` with each basis evaluated once. With
    ``normalization="trapezoid"`` on normalized coordinates it reproduces the
    legacy ``rectangular_channel_velocity_2d`` to floating-point tolerance.
``single_series``
    The classical single series (parabola plus hyperbolic correction), which
    converges much faster and is the default for the new solver.

Physical scaling: ``u = G / mu * phi(y, z)`` where ``G = -dp/dx`` and ``phi``
is the dimensionless (m^2) shape returned with ``normalization="none"``.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .cache import MemoryCache

_profile_cache = MemoryCache(max_bytes=256 * 2**20, name="duct-profiles")

Method = Literal["double_series", "single_series"]
Normalization = Literal["none", "trapezoid", "analytic"]


def _double_series_shape(y: np.ndarray, z: np.ndarray, width: float, height: float, n_terms: int) -> np.ndarray:
    modes = np.arange(1, 2 * n_terms, 2, dtype=float)
    s_y = np.sin(np.pi * np.outer(y, modes) / width)
    s_z = np.sin(np.pi * np.outer(z, modes) / height)
    m = modes[:, None]
    n = modes[None, :]
    # coefficient[m, n] matches the legacy double loop (m -> y, n -> z)
    coefficient = 16.0 / (np.pi**4 * m * n * ((m / width) ** 2 + (n / height) ** 2))
    return s_z @ coefficient.T @ s_y.T


def _single_series_shape(y: np.ndarray, z: np.ndarray, width: float, height: float, n_terms: int) -> np.ndarray:
    """phi(y, z) with mu * lap(phi) = -1 and phi = 0 on the walls, shape (len(z), len(y))."""
    if width < height:  # expand along the smaller dimension for fastest convergence
        return _single_series_shape(z, y, height, width, n_terms).T
    n = np.arange(1, 2 * n_terms, 2, dtype=float)
    parabola = 0.5 * z * (height - z)
    a = np.abs(y - 0.5 * width)[:, None] * (n[None, :] * np.pi / height)  # (ny, nt)
    b = (n * np.pi * width / (2.0 * height))[None, :]
    # cosh(a)/cosh(b) evaluated without overflow for a <= b
    ratio = np.exp(a - b) * (1.0 + np.exp(-2.0 * a)) / (1.0 + np.exp(-2.0 * b))
    s_z = np.sin(np.outer(z, n) * np.pi / height)  # (nz, nt)
    correction = (s_z / n**3) @ ratio.T  # (nz, ny)
    return parabola[:, None] - (4.0 * height**2 / np.pi**3) * correction


def duct_flow_shape_integral(width: float, height: float, n_terms: int = 200) -> float:
    """Integral of phi over the section, so that ``Q = G / mu * integral``."""
    if width < height:
        width, height = height, width
    n = np.arange(1, 2 * n_terms, 2, dtype=float)
    series = np.sum(np.tanh(n * np.pi * width / (2.0 * height)) / n**5)
    return width * height**3 / 12.0 * (1.0 - 192.0 * height / (np.pi**5 * width) * series)


def rectangular_duct_profile(
    y: np.ndarray,
    z: np.ndarray,
    width: float,
    height: float,
    n_terms: int = 100,
    method: Method = "single_series",
    normalization: Normalization = "analytic",
) -> np.ndarray:
    """Fully developed axial velocity on the (z, y) tensor grid, shape ``(len(z), len(y))``.

    Coordinates are in the same units as ``width``/``height``. Normalizations:
    ``none`` returns the shape ``phi`` (u = G/mu * phi); ``analytic`` scales to
    unit mean velocity using the exact flux; ``trapezoid`` scales to unit mean
    using trapezoidal quadrature over the supplied coordinates (legacy convention).
    The returned array is cached and read-only.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    if n_terms <= 0:
        raise ValueError("n_terms must be positive.")
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    key = ("duct", method, normalization, n_terms, float(width), float(height), y.tobytes(), z.tobytes())
    cached = _profile_cache.get(key)
    if cached is not None:
        return cached
    if method == "double_series":
        shape = _double_series_shape(y, z, width, height, n_terms)
    elif method == "single_series":
        shape = _single_series_shape(y, z, width, height, n_terms)
    else:
        raise ValueError(f"Unknown method {method!r}")
    if normalization == "none":
        profile = shape  # both series represent phi with lap(phi) = -1 in physical units
    elif normalization == "trapezoid":
        area = (y[-1] - y[0]) * (z[-1] - z[0])
        mean = np.trapezoid(np.trapezoid(shape, y, axis=1), z) / area
        if mean == 0:
            raise ValueError("average velocity cannot be zero.")
        profile = shape / mean
    elif normalization == "analytic":
        if method == "single_series":
            mean = duct_flow_shape_integral(width, height, max(n_terms, 200)) / (width * height)
        else:
            modes = np.arange(1, 2 * n_terms, 2, dtype=float)
            m, n = modes[:, None], modes[None, :]
            coefficient = 16.0 / (np.pi**4 * m * n * ((m / width) ** 2 + (n / height) ** 2))
            mean = float(np.sum(coefficient * 4.0 / (np.pi**2 * m * n)))
        profile = shape / mean
    else:
        raise ValueError(f"Unknown normalization {normalization!r}")
    profile = np.ascontiguousarray(profile)
    profile.setflags(write=False)
    _profile_cache.put(key, profile)
    return profile


def duct_pressure_gradient(flow_rate: float, width: float, height: float, viscosity: float) -> float:
    """G = -dp/dx that drives ``flow_rate`` through the duct (Pa/m)."""
    return flow_rate * viscosity / duct_flow_shape_integral(width, height)


def duct_resistance(length: float, width: float, height: float, viscosity: float) -> float:
    """Hydraulic resistance (Pa s / m^3) of a straight duct segment."""
    return viscosity * length / duct_flow_shape_integral(width, height)


def cell_centers(extent: float, n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) * (extent / n)


def integrate_flux(velocity: np.ndarray, cell_area: float | np.ndarray, mask: np.ndarray | None = None) -> float:
    """Volumetric flux of a cell-centred axial velocity field (optionally masked)."""
    weights = velocity * cell_area
    if mask is not None:
        weights = np.where(mask, weights, 0.0)
    return float(np.sum(weights))


def cumulative_flow_fraction(coordinate_edges: np.ndarray, flux_per_cell: np.ndarray) -> np.ndarray:
    """Cumulative flux fraction at cell edges, from 0 to 1."""
    cumulative = np.concatenate([[0.0], np.cumsum(flux_per_cell)])
    total = cumulative[-1]
    if total <= 0:
        raise ValueError("total flow must be positive.")
    return cumulative / total


def hydraulic_diameter(width: float, height: float) -> float:
    return 2.0 * width * height / (width + height)


def reynolds_number(flow_rate: float, width: float, height: float, density: float, viscosity: float) -> float:
    """Re = rho U D_h / mu with U the mean velocity and D_h the hydraulic diameter."""
    mean_velocity = flow_rate / (width * height)
    return density * mean_velocity * hydraulic_diameter(width, height) / viscosity


def peclet_number(flow_rate: float, width: float, height: float, diffusivity: float) -> float:
    if diffusivity <= 0:
        return float("inf")
    return flow_rate / (width * height) * hydraulic_diameter(width, height) / diffusivity
