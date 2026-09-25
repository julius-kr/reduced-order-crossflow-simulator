"""Sparse finite-volume lubrication (Hele-Shaw) pressure solver.

Replaces the legacy nested-loop SOR solver. Differences from legacy, all
intentional corrections:

* physical, possibly different, spacings ``dx`` and ``dy``;
* dimensional mobility ``k = h^3 / (12 mu)`` (depth-integrated flux per unit
  width ``q = -k grad p``, mean velocity ``q / h``);
* harmonic transmissibilities across discontinuous gap heights;
* zero-flux side walls applied directly, prescribed inlet/outlet pressures
  eliminated consistently (the matrix stays symmetric positive definite);
* convergence diagnostics computed from the same face fluxes used in assembly.

Fields are stored ``[y, x]`` on a cell-centred grid. Cells with ``gap <= 0``
are solid obstacles. The model assumes a slowly varying gap; abrupt
partial-height steps or square channels do not satisfy that automatically, so
this solver is an exploratory approximation, not the validated pillar model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.linalg import factorized

from ..cache import MemoryCache
from ..geometry import stable_hash

_factor_cache = MemoryCache(max_bytes=256 * 2**20, name="lubrication-factorizations")


@dataclass
class LubricationSolution:
    x_edges: np.ndarray
    y_edges: np.ndarray
    gap: np.ndarray  # (ny, nx) m
    viscosity: float
    pressure: np.ndarray  # (ny, nx) Pa, NaN in solid or isolated cells
    flux_x: np.ndarray  # (ny, nx + 1) m^3/s through x-faces
    flux_y: np.ndarray  # (ny + 1, nx) m^3/s through y-faces
    flow_rate: float
    inlet_pressure: float
    outlet_pressure: float
    relative_residual: float
    flux_imbalance: float
    max_cell_imbalance: float
    isolated_cells: int
    converged: bool
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def pressure_drop(self) -> float:
        return self.inlet_pressure - self.outlet_pressure

    def depth_integrated_flux(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centred q = (qx, qy) in m^2/s (display quantity, not the conservative flux)."""
        hy = np.diff(self.y_edges)[:, None]
        hx = np.diff(self.x_edges)[None, :]
        qx = 0.5 * (self.flux_x[:, :-1] + self.flux_x[:, 1:]) / hy
        qy = 0.5 * (self.flux_y[:-1, :] + self.flux_y[1:, :]) / hx
        return qx, qy

    def mean_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        qx, qy = self.depth_integrated_flux()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.gap > 0, qx / self.gap, 0.0), np.where(self.gap > 0, qy / self.gap, 0.0)


def mobility(gap: np.ndarray, viscosity: float) -> np.ndarray:
    return np.where(gap > 0, gap**3 / (12.0 * viscosity), 0.0)


def _connected_to_boundary(fluid: np.ndarray) -> np.ndarray:
    labels, _ = ndimage.label(fluid)
    boundary_labels = np.union1d(labels[:, 0], labels[:, -1])
    boundary_labels = boundary_labels[boundary_labels > 0]
    return np.isin(labels, boundary_labels)


def solve_lubrication(
    gap: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    viscosity: float,
    inlet_pressure: float | None = 1.0,
    outlet_pressure: float = 0.0,
    flow_rate: float | None = None,
    tolerance: float = 1e-8,
) -> LubricationSolution:
    """Solve ``div(k grad p) = 0`` with inlet/outlet pressures (or a prescribed flow rate)."""
    started = time.perf_counter()
    gap = np.asarray(gap, dtype=float)
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)
    ny, nx = gap.shape
    if len(x_edges) != nx + 1 or len(y_edges) != ny + 1:
        raise ValueError("edges must have one more entry than the gap array along each axis")
    if (inlet_pressure is None) == (flow_rate is None):
        raise ValueError("Specify exactly one of inlet_pressure or flow_rate.")
    if viscosity <= 0:
        raise ValueError("viscosity must be positive")
    hx = np.diff(x_edges)
    hy = np.diff(y_edges)
    k = mobility(gap, viscosity)
    fluid = gap > 0
    active = fluid & _connected_to_boundary(fluid)
    isolated = int(np.sum(fluid & ~active))

    # Face transmissibilities (harmonic across the face; zero next to solids).
    with np.errstate(divide="ignore"):
        resist_x = 0.5 * hx[None, :] / np.where(active, k, np.inf)
        resist_y = 0.5 * hy[:, None] / np.where(active, k, np.inf)
        t_x = hy[:, None] / (resist_x[:, :-1] + resist_x[:, 1:])
        t_y = hx[None, :] / (resist_y[:-1, :] + resist_y[1:, :])
        t_in = hy / resist_x[:, 0]
        t_out = hy / resist_x[:, -1]
    t_x = np.nan_to_num(t_x, nan=0.0, posinf=0.0)
    t_y = np.nan_to_num(t_y, nan=0.0, posinf=0.0)
    t_in = np.nan_to_num(t_in, nan=0.0, posinf=0.0)
    t_out = np.nan_to_num(t_out, nan=0.0, posinf=0.0)

    index = -np.ones((ny, nx), dtype=np.int64)
    index[active] = np.arange(int(active.sum()))
    n = int(active.sum())
    if n == 0:
        raise ValueError("No fluid cell connects the inlet to the outlet.")
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    for a, b, t in ((index[:, :-1], index[:, 1:], t_x), (index[:-1, :], index[1:, :], t_y)):
        mask = (a >= 0) & (b >= 0) & (t > 0)
        ia, ib, tv = a[mask], b[mask], t[mask]
        rows.extend([ia, ib])
        cols.extend([ib, ia])
        vals.extend([-tv, -tv])
        np.add.at(diag, ia, tv)
        np.add.at(diag, ib, tv)
    left, right = index[:, 0], index[:, -1]
    np.add.at(diag, left[left >= 0], t_in[left >= 0])
    np.add.at(diag, right[right >= 0], t_out[right >= 0])
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    matrix = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    assembled = time.perf_counter()

    key = stable_hash({"t_x": t_x, "t_y": t_y, "t_in": t_in, "t_out": t_out})
    solve = _factor_cache.get(key)
    cached = solve is not None
    if solve is None:
        solve = factorized(matrix)
        _factor_cache.put(key, solve, nbytes=matrix.nnz * 64)

    def boundary_rhs(p_in: float, p_out: float) -> np.ndarray:
        rhs = np.zeros(n)
        np.add.at(rhs, left[left >= 0], t_in[left >= 0] * p_in)
        np.add.at(rhs, right[right >= 0], t_out[right >= 0] * p_out)
        return rhs

    p_in = outlet_pressure + 1.0 if inlet_pressure is None else float(inlet_pressure)
    rhs = boundary_rhs(p_in, outlet_pressure)
    p_active = solve(rhs)
    pressure = np.full((ny, nx), np.nan)
    pressure[active] = p_active
    solved = time.perf_counter()

    def fluxes(p: np.ndarray, p_in: float, p_out: float) -> tuple[np.ndarray, np.ndarray]:
        pz = np.where(active, p, 0.0)
        fx = np.zeros((ny, nx + 1))
        fy = np.zeros((ny + 1, nx))
        fx[:, 1:-1] = t_x * (pz[:, :-1] - pz[:, 1:])
        fx[:, 0] = t_in * (p_in - pz[:, 0])
        fx[:, -1] = t_out * (pz[:, -1] - p_out)
        fy[1:-1, :] = t_y * (pz[:-1, :] - pz[1:, :])
        return fx, fy

    flux_x, flux_y = fluxes(pressure, p_in, outlet_pressure)
    q_in = float(flux_x[:, 0].sum())
    if flow_rate is not None:
        if q_in <= 0:
            raise ValueError("Geometry admits no through-flow.")
        scale = flow_rate / q_in
        pressure = outlet_pressure + scale * (pressure - outlet_pressure)
        p_in = outlet_pressure + scale * (p_in - outlet_pressure)
        flux_x, flux_y = flux_x * scale, flux_y * scale
        rhs = boundary_rhs(p_in, outlet_pressure)
        p_active = pressure[active]
        q_in = float(flux_x[:, 0].sum())
    q_out = float(flux_x[:, -1].sum())
    residual = float(np.linalg.norm(matrix @ p_active - rhs) / max(np.linalg.norm(rhs), 1e-300))
    divergence = (flux_x[:, 1:] - flux_x[:, :-1]) + (flux_y[1:, :] - flux_y[:-1, :])
    scale_q = max(abs(q_in), 1e-300)
    imbalance = abs(q_in - q_out) / scale_q
    max_cell = float(np.max(np.abs(divergence[active]))) / scale_q
    converged = residual <= tolerance and imbalance <= 1e-6
    return LubricationSolution(
        x_edges=x_edges,
        y_edges=y_edges,
        gap=gap,
        viscosity=viscosity,
        pressure=pressure,
        flux_x=flux_x,
        flux_y=flux_y,
        flow_rate=q_in,
        inlet_pressure=p_in,
        outlet_pressure=outlet_pressure,
        relative_residual=residual,
        flux_imbalance=imbalance,
        max_cell_imbalance=max_cell,
        isolated_cells=isolated,
        converged=converged,
        timings={
            "assembly": assembled - started,
            "solve": solved - assembled,
            "total": time.perf_counter() - started,
            "factorization_cached": float(cached),
        },
    )
