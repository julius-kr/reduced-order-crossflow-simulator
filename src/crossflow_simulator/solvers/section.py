"""Fully developed axial flow with a variable viscosity field.

Solves ``div_yz(mu(y, z) grad_yz(u)) = -G`` on a cell-centred rectilinear
cross-section with no-slip walls (and optional solid cells), using harmonic
face viscosity. This determines local mobility and flux partitioning for a
common pressure gradient; it is not a model of a 3D injection junction.

The stencil is identical to the one the 3D staggered Stokes solver uses for
x-invariant flow, so profiles computed here are discretely fully developed
inflow/outflow conditions for that solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


@dataclass
class SectionFlow:
    y_edges: np.ndarray
    z_edges: np.ndarray
    velocity: np.ndarray  # (nz, ny), m/s for the given pressure gradient
    flow_rate: float
    pressure_gradient: float
    relative_residual: float

    @property
    def cell_area(self) -> np.ndarray:
        return np.outer(np.diff(self.z_edges), np.diff(self.y_edges))


def solve_axial_velocity(
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    viscosity: float | np.ndarray,
    pressure_gradient: float = 1.0,
    solid: np.ndarray | None = None,
) -> SectionFlow:
    y_edges = np.asarray(y_edges, dtype=float)
    z_edges = np.asarray(z_edges, dtype=float)
    hy, hz = np.diff(y_edges), np.diff(z_edges)
    ny, nz = len(hy), len(hz)
    mu = np.broadcast_to(np.asarray(viscosity, dtype=float), (nz, ny))
    if np.any(~(mu > 0)):
        raise ValueError("viscosity must be positive everywhere")
    fluid = np.ones((nz, ny), dtype=bool) if solid is None else ~np.asarray(solid, dtype=bool)
    index = -np.ones((nz, ny), dtype=np.int64)
    index[fluid] = np.arange(int(fluid.sum()))
    n = int(fluid.sum())
    area = np.outer(hz, hy)

    rows, cols, vals = [], [], []
    diag = np.zeros(n)

    def couple(ia: np.ndarray, ib: np.ndarray, t: np.ndarray) -> None:
        rows.extend([ia, ib, ia, ib])
        cols.extend([ib, ia, ia, ib])
        vals.extend([-t, -t, t, t])

    # y-direction faces between (k, j-1) and (k, j)
    half_l = 0.5 * hy[:-1][None, :]
    half_r = 0.5 * hy[1:][None, :]
    t_y = hz[:, None] / (half_l / mu[:, :-1] + half_r / mu[:, 1:])
    both = fluid[:, :-1] & fluid[:, 1:]
    couple(index[:, :-1][both], index[:, 1:][both], t_y[both])
    # z-direction faces between (k-1, j) and (k, j)
    half_b = 0.5 * hz[:-1][:, None]
    half_t = 0.5 * hz[1:][:, None]
    t_z = hy[None, :] / (half_b / mu[:-1, :] + half_t / mu[1:, :])
    both = fluid[:-1, :] & fluid[1:, :]
    couple(index[:-1, :][both], index[1:, :][both], t_z[both])

    # Walls: domain boundary and solid neighbours (no-slip at the shared face).
    wall_t = np.zeros((nz, ny))
    pad = np.pad(fluid, 1, constant_values=False)
    wall_t += np.where(~pad[1:-1, :-2], hz[:, None] * mu / (0.5 * hy[None, :]), 0.0)
    wall_t += np.where(~pad[1:-1, 2:], hz[:, None] * mu / (0.5 * hy[None, :]), 0.0)
    wall_t += np.where(~pad[:-2, 1:-1], hy[None, :] * mu / (0.5 * hz[:, None]), 0.0)
    wall_t += np.where(~pad[2:, 1:-1], hy[None, :] * mu / (0.5 * hz[:, None]), 0.0)
    np.add.at(diag, index[fluid], wall_t[fluid])

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    matrix = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    rhs = pressure_gradient * area[fluid]
    solution = spsolve(matrix.tocsc(), rhs)
    residual = float(np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), 1e-300))
    velocity = np.zeros((nz, ny))
    velocity[fluid] = solution
    return SectionFlow(
        y_edges=y_edges,
        z_edges=z_edges,
        velocity=velocity,
        flow_rate=float(np.sum(velocity * area)),
        pressure_gradient=pressure_gradient,
        relative_residual=residual,
    )


def flux_matched_velocity(
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    viscosity: float | np.ndarray,
    flow_rate: float,
    solid: np.ndarray | None = None,
) -> SectionFlow:
    """Developed profile carrying ``flow_rate`` (linear scaling of a unit-gradient solve)."""
    unit = solve_axial_velocity(y_edges, z_edges, viscosity, 1.0, solid)
    scale = flow_rate / unit.flow_rate
    return SectionFlow(
        y_edges=unit.y_edges,
        z_edges=unit.z_edges,
        velocity=unit.velocity * scale,
        flow_rate=flow_rate,
        pressure_gradient=scale,
        relative_residual=unit.relative_residual,
    )
