"""Forward pathlines from the inlets to the outlet, for the 3D view.

Display only: sections, flux fractions, and diagnostics never use them.

Seeds are flux weighted. Inlet-pattern seeds are drawn on the inlet plane with
density proportional to the developed axial velocity, and every port receives
seeds in proportion to its flow rate, spread over its opening with the
Poiseuille inflow weight. Through the developed segments between groups a
pathline keeps its ``(y, z)`` and moves with the developed velocity of the
segment's flow rate. Inside a group it is integrated forward through the
group's Stokes field with the same tracer as the sections, stopping at a
sequence of ``x`` planes (the group's graded grid faces, so samples are dense
near features) to record positions and residence times.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import physics
from .results import Pathlines
from .transport import REACHED_PLANE, trace

MAX_PLANES_PER_GROUP = 120


class _DevelopedSpeed:
    """Developed axial velocity per unit mean velocity at arbitrary (y, z)."""

    def __init__(self, width: float, height: float, ny: int = 121, nz: int = 61):
        self.y = np.linspace(0.0, width, ny)
        self.z = np.linspace(0.0, height, nz)
        profile = physics.rectangular_duct_profile(self.y, self.z, width, height, normalization="analytic")
        self.shape = np.clip(np.asarray(profile), 0.0, None)  # series truncation leaves -0 at the walls
        self._interp = RegularGridInterpolator((self.z, self.y), self.shape, bounds_error=False, fill_value=0.0)

    def __call__(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        # Floor near the walls, where the developed speed vanishes and times would diverge.
        return np.maximum(self._interp(np.column_stack([z, y])), 0.02)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """``n`` points with density proportional to the axial velocity (flux weighted)."""
        cells = (0.5 * (self.shape[1:, 1:] + self.shape[:-1, :-1])).ravel()
        pick = rng.choice(cells.size, size=n, p=cells / cells.sum())
        iz, iy = np.unravel_index(pick, (len(self.z) - 1, len(self.y) - 1))
        y = self.y[iy] + rng.random(n) * np.diff(self.y)[iy]
        z = self.z[iz] + rng.random(n) * np.diff(self.z)[iz]
        return y, z


def _port_seeds(shape, n: int, width: float, height: float, inset: float,
                rng: np.random.Generator) -> np.ndarray:
    """Seeds inside a circular wall opening, weighted by the Poiseuille inflow ``1 - r^2/R^2``."""
    points = []
    while sum(len(p) for p in points) < n:
        r = shape.radius * np.sqrt(rng.random(4 * n))
        keep = rng.random(4 * n) < 1.0 - (r / shape.radius) ** 2
        angle = 2 * np.pi * rng.random(4 * n)
        points.append(np.column_stack([r * np.cos(angle), r * np.sin(angle)])[keep])
    a, b = np.concatenate(points)[:n].T
    x = shape.center[0] + a
    if shape.wall in ("top", "bottom"):
        y = np.clip(shape.center[1] + b, inset, width - inset)
        z = np.full(n, height - inset if shape.wall == "top" else inset)
    else:
        z = np.clip(shape.center[1] + b, inset, height - inset)
        y = np.full(n, inset if shape.wall == "left" else width - inset)
    return np.column_stack([x, y, z])


def compute_pathlines(
    links: Sequence,
    domains: Sequence,
    pattern: Callable[[np.ndarray, np.ndarray], np.ndarray],
    width: float,
    height: float,
    pattern_rate: float,
    group_rates: Sequence[tuple[float, Sequence[float]]],
    port_codes: Sequence[Sequence[int]],
    port_origins: Sequence[Sequence[int]],
    outlet_x: float,
    n_lines: int = 360,
    step_fraction: float = 0.5,
    integrator: str = "rk2",
    tolerance: float = 1e-3,
    cancel: Callable[[], bool] | None = None,
    seed: int = 0,
) -> Pathlines:
    """Pathlines through the whole channel.

    ``group_rates[g] = (upstream rate, port rates)`` and ``port_codes[g]`` / ``port_origins[g]``
    give each group's ports (stream codes and 1-based port numbers) in the order of the domain's
    openings.
    """
    rng = np.random.default_rng(seed)
    speed = _DevelopedSpeed(width, height)
    area = width * height
    total = pattern_rate + sum(sum(ports) for _, ports in group_rates)
    n_pattern = max(int(round(n_lines * pattern_rate / total)), 8) if pattern_rate > 0 else 0
    port_counts = [[max(int(round(n_lines * q / total)), 4) if q > 0 else 0 for q in ports]
                   for _, ports in group_rates]
    n = n_pattern + sum(sum(c) for c in port_counts)
    x_in = min([0.0] + [link.x_start for link in links])

    planes = []
    for link in links:
        faces = np.asarray(link.flow.domain.grid.faces[0], dtype=float)[1:]
        if len(faces) > MAX_PLANES_PER_GROUP:
            faces = faces[np.linspace(0, len(faces) - 1, MAX_PLANES_PER_GROUP).round().astype(int)]
        planes.append(faces)
    # Columns: inlet, then per group (entry plane, port seeds, planes), then the outlet.
    m = 2 + sum(2 + len(p) for p in planes)
    points = np.full((n, m, 3), np.nan)
    time = np.full((n, m), np.nan)
    code = np.full(n, -1, dtype=np.int32)
    origin = np.zeros(n, dtype=np.int16)
    status = np.full(n, REACHED_PLANE, dtype=np.int8)

    pos = np.full((n, 3), np.nan)
    clock = np.zeros(n)
    alive = np.zeros(n, dtype=bool)
    if n_pattern:
        y0, z0 = speed.sample(n_pattern, rng)
        pos[:n_pattern] = np.column_stack([np.full(n_pattern, x_in), y0, z0])
        code[:n_pattern] = pattern(y0, z0)
        alive[:n_pattern] = True
    column = 0
    points[alive, column], time[alive, column] = pos[alive], 0.0

    def straight(to_x: float, rate: float) -> None:
        nonlocal column
        column += 1
        idx = np.nonzero(alive & (pos[:, 0] < to_x))[0]
        if len(idx):
            u = speed(pos[idx, 1], pos[idx, 2]) * rate / area
            clock[idx] += (to_x - pos[idx, 0]) / u
            pos[idx, 0] = to_x
        points[alive, column], time[alive, column] = pos[alive], clock[alive]

    start = n_pattern
    for g, link in enumerate(links):
        upstream, ports = group_rates[g]
        straight(link.x_start, upstream)
        shapes = link.flow.domain.opening_shapes
        grid = link.flow.domain.grid
        hmin = min(float(w.min()) for w in grid.widths)
        column += 1
        for k, count in enumerate(port_counts[g]):
            if not count:
                continue
            seeds = _port_seeds(shapes[k], count, width, height, 0.05 * hmin, rng)
            idx = np.arange(start, start + count)
            start += count
            pos[idx], clock[idx], alive[idx] = seeds, 0.0, True
            code[idx], origin[idx] = port_codes[g][k], port_origins[g][k]
            points[idx, column], time[idx, column] = seeds, 0.0
        for x_plane in planes[g]:
            if cancel is not None and cancel():
                from .solvers.stokes import SolveCancelled

                raise SolveCancelled()
            column += 1
            idx = np.nonzero(alive & (pos[:, 0] < x_plane))[0]
            if len(idx):
                result = trace(link.flow.interpolator, pos[idx], stop_x=float(x_plane), direction=1,
                               openings=shapes, step=step_fraction * hmin, integrator=integrator,
                               cancel=cancel, error_tolerance=tolerance)
                pos[idx] = result.end
                clock[idx] += result.residence_time
                ended = result.status != REACHED_PLANE
                status[idx[ended]] = result.status[ended]
                points[idx[ended], column], time[idx[ended], column] = pos[idx[ended]], clock[idx[ended]]
                alive[idx[ended]] = False
            points[alive, column], time[alive, column] = pos[alive], clock[alive]
    straight(outlet_x, total)
    return Pathlines(points=points.astype(np.float32), time=time.astype(np.float32), code=code,
                     status=status, origin=origin, outlet_x=float(outlet_x))


def recolor(lines: Pathlines, pattern: Callable[[np.ndarray, np.ndarray], np.ndarray],
            time_scale: float = 1.0) -> Pathlines:
    """Reused pathlines with the inlet pattern re-evaluated at their seeds and times rescaled."""
    code = lines.code.copy()
    from_pattern = lines.origin == 0
    if from_pattern.any():
        seeds = lines.points[from_pattern, 0]
        code[from_pattern] = pattern(seeds[:, 1].astype(float), seeds[:, 2].astype(float))
    return Pathlines(points=lines.points, time=(lines.time * time_scale).astype(np.float32), code=code,
                     status=lines.status, origin=lines.origin, outlet_x=lines.outlet_x)
