"""Continuous streamline integration through staggered 3D velocity fields.

Positions are floating point throughout; nothing is resampled onto a grid
during integration. Streamlines are integrated in arc length
(``dX/ds = +-U/|U|``) with a midpoint (RK2) or classical RK4 scheme, so the
step count depends on the path length rather than on the local speed, and
slow near-wall trajectories are not truncated. Residence time is accumulated
as ``dt = ds / |U|``.

Trajectories that cannot be completed are never clamped into valid results.
Each seed ends with an explicit status:

``REACHED_PLANE``  crossed the stop plane (the group's inflow plane)
``REACHED_OPENING`` reached a wall opening (port); ``origin`` names it
``WALL``            reached a no-slip wall outside every opening
``SOLID``           entered an obstacle cell
``STAGNATION``      speed vanished (stagnation point or recirculation)
``MAX_LENGTH``      path exceeded the allowed length (recirculation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import ndimage

from .solvers.stokes import DIRICHLET, GHOST, RectilinearGrid, StokesSolution, _face_status

REACHED_PLANE, REACHED_OPENING, WALL, SOLID, STAGNATION, MAX_LENGTH, ACTIVE = 0, 1, 2, 3, 4, 5, 9
STATUS_NAMES = {
    REACHED_PLANE: "reached_plane",
    REACHED_OPENING: "reached_opening",
    WALL: "wall_contact",
    SOLID: "entered_solid",
    STAGNATION: "stagnation",
    MAX_LENGTH: "max_length",
}


@dataclass(frozen=True)
class OpeningShape:
    """Geometric footprint of a wall opening used to classify trajectory ends."""

    wall: str  # "top" | "bottom" | "left" | "right"
    center: tuple[float, float]  # (x, y) on top/bottom walls, (x, z) on side walls
    radius: float

    def contains(self, x: np.ndarray, t: np.ndarray, tolerance: float) -> np.ndarray:
        return (x - self.center[0]) ** 2 + (t - self.center[1]) ** 2 <= (self.radius + tolerance) ** 2


class _Axis:
    """Fractional-index lookup along one lattice axis (arithmetic when uniform)."""

    def __init__(self, coords: np.ndarray):
        self.coords = np.asarray(coords, dtype=float)
        self.n = len(self.coords)
        spacing = np.diff(self.coords)
        self.uniform = bool(np.allclose(spacing, spacing[0], rtol=1e-9, atol=0.0))
        self.origin = float(self.coords[0])
        self.inverse = 1.0 / float(spacing[0])
        self.positions = np.arange(self.n, dtype=float)

    def index(self, x: np.ndarray) -> np.ndarray:
        if self.uniform:
            f = (x - self.origin) * self.inverse
        else:
            f = np.interp(x, self.coords, self.positions)
        return np.minimum(np.maximum(f, 0.0), self.n - 1.0)


class StaggeredInterpolator:
    """Trilinear interpolation of a MAC velocity field that honours no-slip walls.

    Each component is interpolated on its own staggered lattice. Tangential
    components are extended by reflection ghosts across domain walls and into
    solids next to fluid faces, so the interpolated velocity vanishes on the
    wall and solid surfaces instead of half a cell inside them. The wall-normal
    component additionally decays quadratically next to no-penetration faces
    (see :meth:`_normal_factor`).
    """

    def __init__(self, solution: StokesSolution):
        self.grid = solution.grid
        self.solid = solution.solid
        self.any_solid = bool(solution.solid.any())
        fluid = ~solution.solid
        self.data: list[np.ndarray] = []
        self.flags: list[np.ndarray] = []
        self.flag_strides: list[tuple[int, int, int]] = []
        for c, values in enumerate((solution.u, solution.v, solution.w)):
            status = _face_status(self.grid, fluid, c)
            # Faces where the normal velocity vanishes because of a wall or solid (not inflow/outflow/ports).
            flags = (status == GHOST) | ((status == DIRICHLET) & (np.asarray(values) == 0.0)
                                         & _wall_or_solid_face(self.grid, fluid, c))
            self.flags.append(np.ascontiguousarray(flags).ravel())
            self.flag_strides.append(tuple(int(v) for v in np.array(flags.strides) // flags.itemsize))
            data = _reflect_into_solids(np.array(values, dtype=float), status, c)
            for d in range(3):
                if d == c:
                    continue
                first = -np.take(data, [0], axis=d)
                last = -np.take(data, [-1], axis=d)
                data = np.concatenate([first, data, last], axis=d)
            self.data.append(np.ascontiguousarray(data))
        self.face_shapes = [self.grid.face_shape(c) for c in range(3)]
        self.face_axes = [_Axis(self.grid.faces[d]) for d in range(3)]
        centers = []
        for d in range(3):
            f, c = self.grid.faces[d], self.grid.centers[d]
            centers.append(_Axis(np.concatenate([[2 * f[0] - c[0]], c, [2 * f[-1] - c[-1]]])))
        self.center_axes = centers
        self.bounds = np.array([[g[0], g[-1]] for g in self.grid.faces])
        self.reference_speed = max(float(np.max(np.abs(solution.u))), 1e-300)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        fi = [self.face_axes[d].index(points[:, d]) for d in range(3)]
        ci = [self.center_axes[d].index(points[:, d]) for d in range(3)]
        out = np.empty_like(points)
        for c in range(3):
            index = np.vstack([fi[d] if d == c else ci[d] for d in range(3)])
            value = ndimage.map_coordinates(self.data[c], index, order=1, mode="nearest", prefilter=False)
            out[:, c] = value * self._normal_factor(c, fi[c], ci)
        return out

    def _normal_factor(self, c: int, position: np.ndarray, ci: list[np.ndarray]) -> np.ndarray:
        """Quadratic decay of the wall-normal component next to no-penetration faces.

        No-slip plus continuity make the normal velocity vanish like d^2 at a
        wall (d the wall distance), while linear interpolation gives d. Without
        this correction near-wall trajectories drift into walls and solids.
        """
        shape = self.face_shapes[c]
        strides = self.flag_strides[c]
        i0 = np.minimum(position.astype(np.int64), shape[c] - 2)
        t = position - i0
        flat = i0 * strides[c]
        for d in range(3):
            if d != c:
                # tangential lattices carry one ghost layer on each side
                nearest = np.minimum(np.maximum(np.rint(ci[d]).astype(np.int64) - 1, 0), shape[d] - 1)
                flat = flat + nearest * strides[d]
        flags = self.flags[c]
        lower = flags[flat]
        upper = flags[flat + strides[c]]
        return np.where(lower, np.where(upper, 4.0 * t * (1.0 - t), t), np.where(upper, 1.0 - t, 1.0))

    def cell_index(self, points: np.ndarray) -> list[np.ndarray]:
        shape = self.grid.shape
        return [np.minimum(self.face_axes[d].index(points[:, d]).astype(np.int64), shape[d] - 1) for d in range(3)]

    def in_solid(self, points: np.ndarray) -> np.ndarray:
        if not self.any_solid:
            return np.zeros(len(points), dtype=bool)
        i, j, k = self.cell_index(points)
        return self.solid[i, j, k]


def _wall_or_solid_face(grid: RectilinearGrid, fluid: np.ndarray, component: int) -> np.ndarray:
    """Faces of ``component`` lying on a side wall or on a solid surface (never inflow/outflow planes)."""
    n = grid.shape[component]
    pad_width = [(0, 0)] * 3
    pad_width[component] = (1, 1)
    padded = np.pad(fluid, pad_width, constant_values=component == 0)  # x-planes are open, walls are solid
    lower = np.take(padded, np.arange(0, n + 1), axis=component)
    upper = np.take(padded, np.arange(1, n + 2), axis=component)
    return lower != upper


def _reflect_into_solids(data: np.ndarray, status: np.ndarray, component: int) -> np.ndarray:
    """Give ghost faces the mirrored value of adjacent fluid faces (tangential directions only)."""
    ghost = status == GHOST
    if not ghost.any():
        return data
    total = np.zeros_like(data)
    count = np.zeros_like(data)
    valid = ~ghost
    for d in range(3):
        if d == component:
            continue
        for shift in (1, -1):
            neighbour = np.roll(data, shift, axis=d)
            neighbour_valid = np.roll(valid, shift, axis=d)
            edge = [slice(None)] * 3
            edge[d] = 0 if shift == 1 else -1
            neighbour_valid[tuple(edge)] = False
            use = ghost & neighbour_valid
            total[use] -= neighbour[use]
            count[use] += 1
    filled = data.copy()
    has = count > 0
    filled[has] = total[has] / count[has]
    return filled


@dataclass
class TraceResult:
    end: np.ndarray  # (N, 3)
    status: np.ndarray  # (N,) int8
    origin: np.ndarray  # (N,) int16: -1 invalid, 0 plane, k >= 1 opening index + 1
    residence_time: np.ndarray
    path_length: np.ndarray
    wall_limited: np.ndarray | None = None  # trajectories whose wall approach was step-limited
    steps: int = 0
    counts: dict[str, int] = field(default_factory=dict)


def trace(
    velocity: StaggeredInterpolator,
    seeds: np.ndarray,
    stop_x: float,
    direction: int = -1,
    openings: Sequence[OpeningShape] = (),
    step: float | None = None,
    integrator: str = "rk2",
    max_length: float | None = None,
    cancel: Callable[[], bool] | None = None,
    error_tolerance: float = 1e-3,
    use_kernel: bool = True,
) -> TraceResult:
    """Integrate streamlines from ``seeds`` until they reach ``x = stop_x`` or an opening.

    ``rk2`` is the midpoint rule with per-trajectory adaptive steps: the
    difference between the Euler and midpoint displacements estimates the local
    error, which is kept below ``error_tolerance`` times the smallest cell
    width. Steps never exceed one cell, so walls and solids cannot be skipped.
    ``rk4`` uses fixed steps of ``step``.
    """
    grid = velocity.grid
    seeds = np.asarray(seeds, dtype=float)
    n = len(seeds)
    hmin = min(float(w.min()) for w in grid.widths)
    ds = 0.5 * hmin if step is None else float(step)
    ds_max, ds_min = max(hmin, ds), 0.02 * hmin
    tolerance = error_tolerance * hmin
    adaptive = integrator != "rk4"
    extent = grid.faces[0][-1] - grid.faces[0][0]
    max_length = 6.0 * extent if max_length is None else max_length
    # Step budget: generous for trajectories of twice the domain length; trajectories trapped at
    # stagnation corners exhaust it and are reported as MAX_LENGTH instead of stalling the batch.
    max_steps = (4 * (int(np.ceil(2.0 * extent / ds)) + 1) if adaptive
                 else int(np.ceil(max_length / ds)) + 1)
    if adaptive and use_kernel and n:
        from . import transport_kernel

        inputs = transport_kernel.kernel_inputs(velocity)
        if inputs is not None:
            return _trace_compiled(velocity, inputs, seeds, stop_x, direction, openings, ds, ds_min, ds_max,
                                   tolerance, max_length, max_steps, cancel)
    step_size = np.full(n, ds)
    (x0, x1), (y0, y1), (z0, z1) = velocity.bounds

    pos = seeds.copy()
    wall_limited = np.zeros(n, dtype=bool)
    walls = (("left", 1, y0), ("right", 1, y1), ("bottom", 2, z0), ("top", 2, z1))
    opening_tolerance = 0.05 * float(grid.widths[0].min())
    status = np.full(n, ACTIVE, dtype=np.int8)
    origin = np.full(n, -1, dtype=np.int16)
    residence = np.zeros(n)
    length = np.zeros(n)
    stagnation = 1e-9 * velocity.reference_speed
    active = np.arange(n)
    step_count = 0

    inner_lo = np.array([-np.inf, y0 + 1e-3 * float(grid.widths[1].min()), z0 + 1e-3 * float(grid.widths[2].min())])
    inner_hi = np.array([np.inf, y1 - 1e-3 * float(grid.widths[1].min()), z1 - 1e-3 * float(grid.widths[2].min())])

    def direction_field(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Query inside the walls: intermediate RK stages may step past a wall, where the reflection
        # ghosts would reverse the tangential velocity.
        vel = velocity(np.clip(p, inner_lo, inner_hi))
        speed = np.linalg.norm(vel, axis=1)
        safe = np.where(speed > stagnation, speed, 1.0)
        return direction * vel / safe[:, None], speed

    for step_count in range(1, max_steps + 1):
        if not len(active):
            break
        if cancel is not None and step_count % 25 == 0 and cancel():
            from .solvers.stokes import SolveCancelled

            raise SolveCancelled()
        p = pos[active]
        h = step_size[active]
        d1, s1 = direction_field(p)
        accept = np.ones(len(active), dtype=bool)
        if integrator == "rk4":
            d2, s2 = direction_field(p + 0.5 * ds * d1)
            d3, s3 = direction_field(p + 0.5 * ds * d2)
            d4, s4 = direction_field(p + ds * d3)
            move = (d1 + 2 * d2 + 2 * d3 + d4) / 6.0
            speed = np.minimum.reduce([s1, s2, s3, s4])
            mean_speed = (s1 + 2 * s2 + 2 * s3 + s4) / 6.0
        else:
            d2, s2 = direction_field(p + 0.5 * h[:, None] * d1)
            move = d2
            speed = np.minimum(s1, s2)
            mean_speed = s2
            if adaptive:
                error = 0.5 * h * np.linalg.norm(d2 - d1, axis=1)
                accept = (error <= tolerance) | (h <= ds_min * (1 + 1e-9))
                grow = np.clip(0.9 * np.sqrt(tolerance / np.maximum(error, 1e-300)), 0.3, 2.0)
                step_size[active] = np.clip(h * grow, ds_min, ds_max)
        new = p + h[:, None] * move
        new[~accept] = p[~accept]
        # Wall-normal step limiter: near a no-slip wall the normal velocity vanishes like d^2 and the
        # tangential like d, so exact trajectories approach walls at most exponentially. A fixed step
        # would overshoot; allow at most half the remaining wall distance per step (except over ports).
        limited = np.zeros(len(active), dtype=bool)
        for wall, axis, plane in walls:
            lower = wall in ("left", "bottom")
            d_old = (p[:, axis] - plane) if lower else (plane - p[:, axis])
            d_new = (new[:, axis] - plane) if lower else (plane - new[:, axis])
            floor = 1e-3 * float(grid.widths[axis].min())
            target = np.maximum(0.5 * d_old, floor)
            approach = d_new < target
            if not approach.any():
                continue
            # Exempt only trajectories whose wall crossing point (the point classification will
            # use) lies over an opening; otherwise use the end point's projection onto the wall.
            delta = d_old - d_new
            theta = np.where(d_new < 0, np.clip(d_old / np.where(delta == 0, 1.0, delta), 0.0, 1.0), 1.0)
            probe = p + theta[:, None] * (new - p)
            in_plane = probe[:, 1] if wall in ("top", "bottom") else probe[:, 2]
            over_opening = np.zeros(len(active), dtype=bool)
            for opening in openings:
                if opening.wall == wall:
                    over_opening |= opening.contains(probe[:, 0], in_plane, opening_tolerance)
            limit = approach & ~over_opening
            if limit.any():
                new[limit, axis] = plane + target[limit] if lower else plane - target[limit]
                limited |= limit
        wall_limited[active] |= limited
        stalled = (speed <= stagnation) & accept
        taken = np.where(accept, h, 0.0)
        residence[active] += np.where(stalled, 0.0, taken / np.maximum(mean_speed, stagnation))
        length[active] += taken

        # Stop plane crossing (interpolate the exact crossing point).
        crossed = (new[:, 0] <= stop_x) if direction < 0 else (new[:, 0] >= stop_x)
        if crossed.any():
            denom = new[crossed, 0] - p[crossed, 0]
            theta = np.where(np.abs(denom) > 0, (stop_x - p[crossed, 0]) / np.where(denom == 0, 1.0, denom), 1.0)
            new[crossed] = p[crossed] + theta[:, None] * (new[crossed] - p[crossed])
        result = np.full(len(active), ACTIVE, dtype=np.int8)
        result[crossed] = REACHED_PLANE
        org = np.full(len(active), -1, dtype=np.int16)
        org[crossed] = 0

        # Wall crossing: locate the exact crossing point and classify it against the openings.
        for wall, axis, plane in walls:
            beyond = (new[:, axis] < plane) if wall in ("left", "bottom") else (new[:, axis] > plane)
            hit = beyond & (result == ACTIVE)
            if not hit.any():
                continue
            delta = new[hit, axis] - p[hit, axis]
            theta = np.clip((plane - p[hit, axis]) / np.where(delta == 0, 1.0, delta), 0.0, 1.0)
            new[hit] = p[hit] + theta[:, None] * (new[hit] - p[hit])
            new[hit, axis] = plane
            in_plane = new[:, 1] if wall in ("top", "bottom") else new[:, 2]
            matched = np.zeros(len(active), dtype=bool)
            for k, opening in enumerate(openings):
                if opening.wall != wall:
                    continue
                inside = hit & ~matched & opening.contains(new[:, 0], in_plane, opening_tolerance)
                org[inside] = k + 1
                result[inside] = REACHED_OPENING
                matched |= inside
            result[hit & ~matched] = WALL
        still = result == ACTIVE
        if still.any():
            solid = np.zeros(len(active), dtype=bool)
            solid[still] = velocity.in_solid(new[still])
            if solid.any():
                # Same limiter for solid surfaces: the crossed voxel face is known exactly.
                idx = np.nonzero(solid)[0]
                p_cell = velocity.cell_index(p[idx])
                n_cell = velocity.cell_index(new[idx])
                for axis in range(3):
                    moved = n_cell[axis] != p_cell[axis]
                    if not moved.any():
                        continue
                    up = n_cell[axis] > p_cell[axis]
                    faces = grid.faces[axis]
                    face = np.where(up, faces[np.minimum(p_cell[axis] + 1, len(faces) - 1)], faces[p_cell[axis]])
                    d_old = np.abs(p[idx, axis] - face)
                    target = np.maximum(0.5 * d_old, 1e-3 * float(grid.widths[axis].min()))
                    adjusted = np.where(up, face - target, face + target)
                    new[idx[moved], axis] = adjusted[moved]
                limited[idx] = True
                wall_limited[active[idx]] = True
                bad = velocity.in_solid(new[idx])
                result[idx[bad]] = SOLID
        result[(result == ACTIVE) & stalled] = STAGNATION
        result[(result == ACTIVE) & (length[active] > max_length)] = MAX_LENGTH
        pos[active] = new
        status[active] = result
        origin[active] = org
        active = active[result == ACTIVE]
    status[active] = MAX_LENGTH
    counts = {name: int(np.sum(status == code)) for code, name in STATUS_NAMES.items()}
    counts["wall_limited"] = int(wall_limited.sum())
    return TraceResult(pos, status, origin, residence, length, wall_limited, step_count, counts)


def _trace_compiled(velocity, inputs, seeds, stop_x, direction, openings, ds, ds_min, ds_max, tolerance,
                    max_length, max_steps, cancel, chunk: int = 4096) -> TraceResult:
    """Same algorithm as the NumPy loop above, in the optional compiled kernel (chunked for cancellation)."""
    from . import transport_kernel

    grid = velocity.grid
    (x0, x1), (y0, y1), (z0, z1) = velocity.bounds
    wall_ids = {"left": 0, "right": 1, "bottom": 2, "top": 3}
    table = np.array([[wall_ids[o.wall], o.center[0], o.center[1], o.radius] for o in openings], dtype=float)
    table = table.reshape(-1, 4)
    widths_min = np.array([float(w.min()) for w in grid.widths])
    lo = np.array([-np.inf, y0 + 1e-3 * widths_min[1], z0 + 1e-3 * widths_min[2]])
    hi = np.array([np.inf, y1 - 1e-3 * widths_min[1], z1 - 1e-3 * widths_min[2]])
    params = np.array([stop_x, float(direction), ds, ds_min, ds_max, tolerance, max_length,
                       1e-9 * velocity.reference_speed, 0.05 * widths_min[0], float(max_steps)])
    parts = []
    for start in range(0, len(seeds), chunk):
        if cancel is not None and cancel():
            from .solvers.stokes import SolveCancelled

            raise SolveCancelled()
        parts.append(transport_kernel.trace_kernel(
            np.ascontiguousarray(seeds[start:start + chunk]), *inputs["data"], *inputs["flags"], inputs["faces"],
            inputs["centers"], inputs["uniform"], inputs["origin"], inputs["inverse"], inputs["solid"],
            inputs["any_solid"], table, velocity.bounds, lo, hi, widths_min, params))
    end, status, origin, residence, length, limited = (np.concatenate(p) for p in zip(*parts))
    counts = {name: int(np.sum(status == code)) for code, name in STATUS_NAMES.items()}
    counts["wall_limited"] = int(limited.sum())
    return TraceResult(end, status, origin, residence, length, limited, 0, counts)


def is_valid(status: np.ndarray) -> np.ndarray:
    return (status == REACHED_PLANE) | (status == REACHED_OPENING)
