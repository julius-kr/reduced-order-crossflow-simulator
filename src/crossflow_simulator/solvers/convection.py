"""Convective acceleration ``(u . grad) u`` of a staggered (MAC) velocity field.

Used by the CFD reference to include inertia through Picard iterations: the
convective term of the previous iterate is applied to the Stokes operator as a
body force. The term is evaluated in advective form with central differences
at every face. Tangential neighbours across walls and into solids use the same
no-slip reflection ghosts as the streamline interpolator, and the advecting
velocity at a face is the average of the four surrounding faces of the other
component.
"""
from __future__ import annotations

import numpy as np

from ..transport import StaggeredInterpolator
from .stokes import StokesSolution


def _along(axis: int, values: np.ndarray) -> list[int]:
    shape = [1, 1, 1]
    shape[axis] = len(values)
    return shape


def convective_acceleration(solution: StokesSolution,
                            interpolator: StaggeredInterpolator | None = None) -> list[np.ndarray]:
    """``(u . grad) u`` for each velocity component on its own faces (m/s^2)."""
    grid = solution.grid
    interp = interpolator or StaggeredInterpolator(solution)
    fields = (solution.u, solution.v, solution.w)
    out = []
    for c in range(3):
        f = fields[c]
        faces = grid.faces[c]
        centers = grid.centers[c]
        padded = interp.data[c]  # one reflection ghost layer on each side of the tangential axes
        total = np.zeros_like(f)
        # d f / d x_c from the neighbouring faces of the same component (interior faces only)
        derivative = np.zeros_like(f)
        inner = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower = [slice(None)] * 3
        inner[c], upper[c], lower[c] = slice(1, -1), slice(2, None), slice(None, -2)
        spacing = (faces[2:] - faces[:-2]).reshape(_along(c, faces[2:]))
        derivative[tuple(inner)] = (f[tuple(upper)] - f[tuple(lower)]) / spacing
        total += f * derivative
        # advecting components at the c-faces, and tangential derivatives
        for d in range(3):
            if d == c:
                continue
            other = fields[d]
            d_faces = [slice(None)] * 3
            d_faces_up = [slice(None)] * 3
            d_faces[d], d_faces_up[d] = slice(None, -1), slice(1, None)
            cell_mean = 0.5 * (other[tuple(d_faces)] + other[tuple(d_faces_up)])  # d at cell centres
            advecting = np.empty_like(f)
            first, last = [slice(None)] * 3, [slice(None)] * 3
            first[c], last[c] = slice(0, 1), slice(-1, None)
            advecting[tuple(first)] = np.take(cell_mean, [0], axis=c)
            advecting[tuple(last)] = np.take(cell_mean, [-1], axis=c)
            weight = ((faces[1:-1] - centers[:-1]) / (centers[1:] - centers[:-1])).reshape(
                _along(c, faces[1:-1]))
            low, high = [slice(None)] * 3, [slice(None)] * 3
            low[c], high[c] = slice(None, -1), slice(1, None)
            advecting[tuple(inner)] = (1.0 - weight) * cell_mean[tuple(low)] + weight * cell_mean[tuple(high)]
            coords = interp.center_axes[d].coords  # tangential centres including the ghost positions
            up, down = [slice(None)] * 3, [slice(None)] * 3
            for e in range(3):
                if e != c:
                    up[e] = down[e] = slice(1, -1)
            up[d], down[d] = slice(2, None), slice(None, -2)
            distance = (coords[2:] - coords[:-2]).reshape(_along(d, coords[2:]))
            total += advecting * (padded[tuple(up)] - padded[tuple(down)]) / distance
        out.append(total)
    return out


def momentum_volumes(solution: StokesSolution) -> list[np.ndarray]:
    """Volume of each face's momentum control volume (half cells at the domain boundary)."""
    grid = solution.grid
    volumes = []
    for c in range(3):
        widths = list(grid.widths)
        h = grid.widths[c]
        half = 0.5 * np.concatenate([[h[0]], h[:-1] + h[1:], [h[-1]]])
        widths[c] = half
        volumes.append(widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :])
    return volumes


def convective_force(solution: StokesSolution, density: float,
                     interpolator: StaggeredInterpolator | None = None) -> np.ndarray:
    """``-rho (u . grad) u`` integrated over the momentum control volumes, as a full face vector (N).

    The ordering matches :class:`~crossflow_simulator.solvers.stokes.StokesProblem`: all u faces,
    then v, then w, each in C order.
    """
    acceleration = convective_acceleration(solution, interpolator)
    volumes = momentum_volumes(solution)
    return np.concatenate([(-density * a * v).ravel() for a, v in zip(acceleration, volumes)])
