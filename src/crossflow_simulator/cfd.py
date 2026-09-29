"""CFD reference: the whole channel in one 3D Navier-Stokes solve, without reduced-order assumptions.

The new solver is a reduced-order model. It splits the channel into feature groups, assumes fully
developed flow at each group's margins, neglects inertia, and composes section maps traced on
a coarse lattice. The CFD reference removes those assumptions:

* **One domain.** A single staggered finite-volume grid runs from an inflow buffer upstream of
  x = 0 to an outflow buffer downstream of the outlet, refined at every pillar and port. The
  inlet pattern enters developed at the far end of the inflow buffer; ports inject Poiseuille
  inflow through the wall.
* **Steady incompressible Navier-Stokes.** Inertia enters through Picard iterations that apply
  the lagged convective term as a body force (under-relaxed if needed). With a viscosity
  contrast, the viscosity field follows the materials: every fluid cell is traced to its
  source and the flow is solved again until the material field stops changing.
* **Direct tracing.** Every output cell of every section is traced back individually to its
  source through the full field. There is no map interpolation and no group composition, and
  each section's flux uses the computed axial velocity.
* **Mesh-convergence estimate.** A second grid with 2/3 of the cells is solved as well. The
  interface shift between the two grids gives a grid-convergence-index estimate of the
  reference's own numerical uncertainty (first-order convergence and safety factor 1.25, because
  staircase pillars converge at first order).

It shares the finite-volume building blocks and the streamline tracer with the new solver, so it
is a higher-fidelity numerical reference, not an independent code. CFD from another code, imported
as ``crossflow-reference/1`` data, is the independent check (see ``calibration.py``).

Flow fields and results are cached on disk. The field key contains only what the flow depends on,
so changing colours, the output grid, or the new solver's settings reuses the solved field and
only repeats the tracing.
"""
from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import geometry, metrics, physics, solver_identity
from .cache import Stopwatch, disk_cache
from .config import InletSpec, ProjectConfig, validate_config
from .inlets import partition_pattern, pattern_grid
from .materials import stream_table
from .models.domain import channel_extent, plan_groups, port_opening
from .models.validity import MODEL_CARDS
from .pathlines import _DevelopedSpeed, _port_seeds
from .results import Pathlines, Section, SimulationResult, failed_result
from .simulator import output_grid
from .solvers.convection import convective_force
from .solvers.section import flux_matched_velocity
from .solvers.stokes import (
    RectilinearGrid, SolveCancelled, StokesProblem, StokesSolution, WallOpening, graded_faces, snap_faces,
)
from .transport import (
    MAX_LENGTH, REACHED_OPENING, REACHED_PLANE, STAGNATION, STATUS_NAMES, OpeningShape, StaggeredInterpolator, trace,
)

CFD_VERSION = "cfd-reference/3"
PRESETS = {"quick": 16, "standard": 24, "fine": 32}
X_SPACING = 4.0  # streamwise spacing far from features, in fine cells
SOLVER_TOLERANCE = 1e-12
DIVERGENCE_TARGET = 1e-5  # cell divergence / cell flux; a solve above it is refined with a 100x tighter tolerance
CORRECTION_TOLERANCE = 1e-6  # preconditioned; a Picard step needs accuracy relative to its own size only
CORRECTION_RESIDUAL = 0.05
PICARD_TOLERANCE = 1e-4  # largest velocity change of a Picard step, relative to the largest velocity
MAX_PICARD = 30
COUPLING_TOLERANCE = 1e-3  # fraction of fluid cells whose material changed
COUPLING_FLIP_TOLERANCE = 1e-2  # ... or at most this fraction (single interface cells flipping) while
COUPLING_VELOCITY_TOLERANCE = 2e-3  # the velocity changes by less than this (relative RMS)
MAX_COUPLING = 12
TRACE_TOLERANCE = 2.5e-4  # local streamline error per step / smallest cell (the new solver uses 1e-3)
STEP_BUDGET = 8.0  # slow paths along walls in wakes need many small steps over the whole channel
SUBCELL = 3  # interface cells of a section are sampled SUBCELL x SUBCELL times for flux fractions
N_PATHLINES = 360
MAX_PATHLINE_PLANES = 240
CONVERGENCE_RATIO = 2.0 / 3.0
GCI_SAFETY = 1.25
GCI_ORDER = 1.0
TARGET_FRACTION_OF_H = 0.02  # interface acceptance target of the validation workflow (calibration.TARGETS)

ProgressFn = Callable[[str, float, str], None]


# --------------------------------------------------------------------------- capability and keys


def _viscosities(cfg: ProjectConfig) -> dict[int, float]:
    codes = cfg.stream_codes()
    return {codes[i.id]: cfg.material(i.material).viscosity for i in cfg.inlets}


def _contrast(cfg: ProjectConfig) -> bool:
    return len({cfg.material(i.material).viscosity for i in cfg.active_inlets()}) > 1


def unsupported(cfg: ProjectConfig) -> list[str]:
    """Cases the CFD reference cannot compute faithfully (empty when supported)."""
    reasons = []
    if cfg.operating.drive == "pressure":
        if _contrast(cfg):
            reasons.append("pressure-driven flow with a viscosity contrast (the inlet rates would depend on "
                           "the material field)")
        if cfg.channel.length is None:
            reasons.append("pressure-driven flow needs the channel length (outlet position)")
    return reasons


def convergence_cells(cfg: ProjectConfig) -> int | None:
    if not cfg.cfd.convergence_check:
        return None
    coarse = int(round(CONVERGENCE_RATIO * cfg.cfd.cells))
    return coarse if 8 <= coarse < cfg.cfd.cells else None


def outlet_position(cfg: ProjectConfig) -> float:
    """Streamwise position of the outlet plane (as in the new solver's pathlines)."""
    return max([channel_extent(cfg)] + [g.x_end for g in plan_groups(cfg)])


def stations(cfg: ProjectConfig) -> list[tuple[str, float]]:
    """Section stations: the new solver's (inlet and after each feature group), for aligned comparisons."""
    groups = plan_groups(cfg)
    if not groups:
        return [("inlet plane", 0.0), ("outlet (straight channel)", float(cfg.channel.length or channel_extent(cfg)))]
    return [("inlet plane", 0.0)] + [(f"after {g.label}", float(g.x_end)) for g in groups]


def _field_inputs(cfg: ProjectConfig) -> dict:
    """Everything the flow field depends on (and nothing else)."""
    data = cfg.to_dict()
    active_inlets = [d for d, inlet in zip(data["inlets"], cfg.inlets) if inlet.enabled]
    active_pillars = [d for d, pillar in zip(data["pillars"], cfg.pillars) if pillar.enabled]
    inputs = {"channel": data["channel"], "pillars": active_pillars, "operating": data["operating"]}
    if _contrast(cfg) or cfg.operating.drive == "pressure":
        inputs["inlets"] = active_inlets
        inputs["pattern"] = data["pattern"]
        inputs["materials"] = [{"id": m.id, "viscosity": m.viscosity, "density": m.density} for m in cfg.materials]
    else:
        rates = {i.id: float(i.flow_rate) for i in cfg.active_inlets()}
        total = sum(rates.values())
        inputs["pattern_rate"] = sum(rates[i.id] for i in cfg.pattern_inlets())
        inputs["ports"] = [[p.station, p.wall, p.position, p.diameter, rates[p.id]] for p in cfg.port_inlets()]
        inputs["viscosity"] = cfg.material(cfg.active_inlets()[0].material).viscosity
        inputs["density"] = sum(rates[i.id] * cfg.material(i.material).density for i in cfg.active_inlets()) / total
    return inputs


def field_key(cfg: ProjectConfig, cells: int) -> str:
    return geometry.stable_hash({
        "version": CFD_VERSION, "inputs": _field_inputs(cfg), "cells": cells, "buffer": cfg.cfd.buffer,
        "inertia": cfg.cfd.inertia, "outlet_x": outlet_position(cfg),
    })


def result_key(cfg: ProjectConfig) -> str:
    """Key of a complete CFD result: flow fields plus everything the sections and pathlines depend on."""
    data = cfg.to_dict()
    levels = [cfg.cfd.cells] + [c for c in [convergence_cells(cfg)] if c]
    y_edges, z_edges = output_grid(cfg)
    return geometry.stable_hash({
        "version": CFD_VERSION, "fields": [field_key(cfg, c) for c in levels], "stations": stations(cfg),
        "output": [len(y_edges), len(z_edges)], "inlets": data["inlets"], "pattern": data["pattern"],
        "materials": data["materials"], "pathlines": N_PATHLINES,
    })


def estimate_seconds(cfg: ProjectConfig) -> float:
    """Rough run time on the development machine (measured: about 20 s per 145k-cell Stokes solve)."""
    total = 0.0
    for cells in [cfg.cfd.cells] + [c for c in [convergence_cells(cfg)] if c]:
        W, H = cfg.channel.width, cfg.channel.height
        ny = cells if W <= H else cells * W / H
        nz = cells if H <= W else cells * H / W
        fine = min(W / ny, H / nz)
        nx = (outlet_position(cfg) + 2 * cfg.cfd.buffer * max(W, H)) / (X_SPACING * fine) + 40 * (
            len(cfg.active_pillars()) + len(cfg.port_inlets()))
        n = nx * ny * nz
        # measured: viscosity coupling costs about 13 plain solves (re-assembly, tracing every cell, 4-6 rounds)
        solves = (13.0 if _contrast(cfg) else 1.0) + (0.6 if cfg.cfd.inertia else 0.0)
        if cfg.operating.drive == "pressure":
            solves += len(cfg.port_inlets()) + 1
        total += 1.4e-4 * n * solves + 2.0
    return total


# --------------------------------------------------------------------------- domain


@dataclass
class Domain:
    cells: int
    grid: RectilinearGrid
    solid: np.ndarray
    openings: list[WallOpening]
    shapes: list[OpeningShape]
    ports: list[InletSpec]
    x_in: float
    x_out: float
    fine: float
    warnings: list[str] = field(default_factory=list)

    @property
    def hmin(self) -> float:
        return min(float(w.min()) for w in self.grid.widths)

    def describe(self) -> dict:
        nx, ny, nz = self.grid.shape
        return {"cells_across": self.cells, "grid": [nx, ny, nz], "cells": int(nx * ny * nz),
                "fluid_cells": int((~self.solid).sum()), "x_range_m": [self.x_in, self.x_out],
                "fine_spacing_m": self.fine, "smallest_cell_m": self.hmin}


def _aligned_edges(pillars) -> tuple[list[float], list[float]]:
    """x and y coordinates of footprint extents and of footprint edges parallel to an axis."""
    xs, ys = [], []
    for pillar in pillars:
        fp = pillar.footprint()
        xs += [float(fp[:, 0].min()), float(fp[:, 0].max())]
        ys += [float(fp[:, 1].min()), float(fp[:, 1].max())]
        span = max(np.ptp(fp[:, 0]), np.ptp(fp[:, 1]))
        for a, b in zip(fp, np.roll(fp, -1, axis=0)):
            if abs(a[0] - b[0]) <= 1e-9 * span:
                xs.append(float(a[0]))
            if abs(a[1] - b[1]) <= 1e-9 * span:
                ys.append(float(a[1]))
    return sorted(set(xs)), sorted(set(ys))


def _snap_graded(faces: np.ndarray, points: list[float], min_fraction: float = 0.4) -> np.ndarray:
    """Move the nearest face onto each point of a graded axis; drop faces left too close to a snapped one."""
    faces = faces.copy()
    snapped = np.zeros(len(faces), dtype=bool)
    snapped[[0, -1]] = True
    for point in points:
        if not faces[0] < point < faces[-1]:
            continue
        i = int(np.argmin(np.abs(faces - point)))
        if snapped[i]:
            continue
        faces[i] = point
        snapped[i] = True
    keep = np.ones(len(faces), dtype=bool)
    for i in np.nonzero(snapped)[0]:
        for j in (i - 1, i + 1):
            if 0 < j < len(faces) - 1 and not snapped[j]:
                local = faces[j + 1] - faces[j - 1]  # the two cells around face j
                if abs(faces[j] - faces[i]) < min_fraction * 0.5 * local:
                    keep[j] = False
    return faces[keep]


def _partition_edges(cfg: ProjectConfig) -> tuple[list[float], list[float]]:
    """Inlet layer boundaries (y, z). With a viscosity contrast the inflow viscosity jumps there, so
    faces on them make the imposed inflow split the flux exactly as the pattern prescribes."""
    if not _contrast(cfg) or cfg.operating.drive == "pressure":
        return [], []
    partition = _partition(cfg, {i.id: float(i.flow_rate) for i in cfg.active_inlets()}, True)
    ys = [float(v) for edges in partition.col_edges for v in edges[1:-1]]
    zs = [float(v) for edges in partition.row_edges for v in edges[1:-1]]
    return ys, zs


def build_domain(cfg: ProjectConfig, cells: int, outlet_x: float | None = None) -> Domain:
    W, H = cfg.channel.width, cfg.channel.height
    ny = cells if W <= H else int(round(cells * W / H))
    nz = cells if H <= W else int(round(cells * H / W))
    fine = min(W / ny, H / nz)
    size = max(W, H)
    outlet_x = outlet_position(cfg) if outlet_x is None else outlet_x
    x_in, x_out = -cfg.cfd.buffer * size, outlet_x + cfg.cfd.buffer * size
    pillars = cfg.active_pillars()
    heights = [p.height if p.mount == "bottom" else H - p.height for p in pillars]
    layer_y, layer_z = _partition_edges(cfg)
    zf = snap_faces(np.linspace(0.0, H, nz + 1), heights + layer_z)
    # Faces snapped onto the footprints' extents and axis-aligned edges, so these edges are exact on every
    # grid level (otherwise the staircase aliases differently per level and the convergence check is noise).
    edge_x, edge_y = _aligned_edges(pillars)
    yf = snap_faces(np.linspace(0.0, W, ny + 1), edge_y + layer_y)
    intervals = []
    for pillar in pillars:
        fp = pillar.footprint()
        intervals.append((float(fp[:, 0].min()), float(fp[:, 0].max())))
    for port in cfg.port_inlets():
        radius = 0.5 * float(port.diameter)
        intervals.append((port.station - radius, port.station + radius))
    xf = _snap_graded(graded_faces(x_in, x_out, target=X_SPACING * fine, refine=intervals, fine=fine), edge_x)
    grid = RectilinearGrid(xf, yf, zf)
    zc = grid.centers[2]
    solid = np.zeros(grid.shape, dtype=bool)
    warnings = []
    for pillar in pillars:
        fp = pillar.footprint()
        column = geometry.raster_fraction(fp, xf, yf, subsamples=8) >= 0.5
        layers = zc < pillar.height if pillar.mount == "bottom" else zc > H - pillar.height
        solid |= column[:, :, None] & layers[None, None, :]
        across = min(np.ptp(fp[:, 0]), np.ptp(fp[:, 1])) / fine
        if across < 3:
            warnings.append(f"Pillar {pillar.id!r} spans only {across:.1f} CFD cells; use a finer CFD grid.")
    openings, shapes = [], []
    for port in cfg.port_inlets():
        opening, shape, faces = port_opening(grid, port)
        if faces < 12:
            warnings.append(f"Port {port.id!r} is covered by only {faces} CFD wall faces; use a finer CFD grid.")
        axis = 2 if port.wall in ("top", "bottom") else 1
        side = 0 if port.wall in ("bottom", "left") else -1
        if np.any(np.take(solid, [side], axis=axis).squeeze(axis) & (opening.profile != 0)):
            raise ValueError(f"Port {port.id!r} opens into a solid obstacle.")
        openings.append(opening)
        shapes.append(shape)
    return Domain(cells, grid, solid, openings, shapes, cfg.port_inlets(), x_in, x_out, fine, warnings)


# --------------------------------------------------------------------------- flow


@dataclass
class LevelFlow:
    domain: Domain
    solution: StokesSolution
    interpolator: StaggeredInterpolator
    rates: dict[str, float]
    partition: object  # PatternPartition
    info: dict


def _plane_pressure(solution: StokesSolution, x: float) -> float:
    """Mean pressure over the fluid of the cross-section at ``x`` (linear between cell centres)."""
    fluid = ~solution.solid
    means = np.array([np.mean(layer[mask]) if mask.any() else np.nan for layer, mask in zip(solution.p, fluid)])
    xc = solution.grid.centers[0]
    valid = np.isfinite(means)
    return float(np.interp(x, xc[valid], means[valid]))


def _port_pressure(solution: StokesSolution, opening: WallOpening) -> float:
    """Inflow-weighted mean pressure in the cells behind a port opening."""
    axis = 2 if opening.wall in ("top", "bottom") else 1
    side = 0 if opening.wall in ("bottom", "left") else -1
    cells = np.take(solution.p, [side], axis=axis).squeeze(axis)
    weights = np.abs(opening.profile)
    return float(np.nansum(weights * cells) / weights.sum())


def _speed(solution: StokesSolution) -> float:
    return max(float(np.max(np.abs(a))) for a in (solution.u, solution.v, solution.w)) or 1.0


def navier_stokes(problem: StokesProblem, values: np.ndarray, density: float, inertia: bool,
                  cancel: Callable[[], bool] | None = None, initial: StokesSolution | None = None,
                  base: StokesSolution | None = None) -> tuple[StokesSolution, dict]:
    """Steady Navier-Stokes by Picard iteration on top of the Stokes operator.

    Every iterate is ``u_stokes + S0(G)``, with ``S0`` the Stokes solution operator for zero boundary
    values and ``G`` an accumulated body force. A step solves ``S0(F(u) - G)`` (``F`` the convective
    force of the current iterate), adds it with relaxation ``omega`` and updates ``G``; it has
    converged when the unrelaxed step changes the velocity by less than ``PICARD_TOLERANCE``.
    """
    if base is None:
        base = problem.solve(values, tolerance=SOLVER_TOLERANCE, cancel=cancel, initial=initial)
        if base.info.get("max_cell_divergence", 0.0) > DIVERGENCE_TARGET:
            # Cell-by-cell viscosity jumps (traced material fields) need a tighter solve for continuity.
            first = base.info
            base = problem.solve(values, tolerance=1e-2 * SOLVER_TOLERANCE, cancel=cancel, initial=base)
            base.info["refined_after"] = {k: first.get(k) for k in ("iterations", "max_cell_divergence")}
    stokes = {k: base.info.get(k) for k in ("method", "iterations", "relative_residual", "max_cell_divergence",
                                            "unknowns", "solve_seconds")}
    info: dict = {"stokes": stokes, "picard": None}
    solution = base
    if inertia and density > 0:
        speed = _speed(base)
        accumulated = np.zeros(problem.n_faces)
        zero = np.zeros(problem.n_faces)
        omega, previous, converged, history = 1.0, math.inf, False, []
        for iteration in range(1, MAX_PICARD + 1):
            interpolator = StaggeredInterpolator(solution)
            residual = convective_force(solution, density, interpolator) - accumulated
            step = problem.solve(zero, tolerance=CORRECTION_TOLERANCE, cancel=cancel, force=residual,
                                 velocity_scale=speed, max_residual=CORRECTION_RESIDUAL)
            change = _speed(step) / speed
            solution = solution.combine(step, 1.0, omega)
            accumulated = accumulated + omega * residual
            history.append({"iteration": iteration, "relative_change": change, "relaxation": omega,
                            "linear_iterations": step.info.get("iterations"),
                            "relative_residual": step.info.get("relative_residual")})
            if change < PICARD_TOLERANCE:
                converged = True
                break
            if change > previous:
                omega *= 0.5
                if omega < 1.0 / 16:
                    break
            previous = change
        effect = max(float(np.max(np.abs(a - b))) for a, b in
                     ((solution.u, base.u), (solution.v, base.v), (solution.w, base.w))) / speed
        info["picard"] = {"iterations": len(history), "converged": converged, "history": history,
                          "inertia_effect": effect, "density_kg_m3": density}
    solution.info = dict(base.info, navier_stokes=info["picard"])
    return solution, info


def _pressure_driven(cfg: ProjectConfig, domain: Domain, problem: StokesProblem, unit: np.ndarray,
                     cancel) -> tuple[dict[str, float], StokesSolution, dict]:
    """Inlet flow rates from the channel's exact creeping-flow resistance matrix.

    One unit solve per source (the inlet pattern and each port) gives the pressure at every source
    node per unit flow; together with each inlet's supply pressure and tubing resistance this is
    a linear system for the rates, and the field is the superposition of the unit solutions.
    """
    W, H = cfg.channel.width, cfg.channel.height
    q_ref = 1e-3 * W * H  # 1 mm/s mean velocity; the problem is linear, so the scale is arbitrary
    ports = domain.ports
    n_sources = 1 + len(ports)
    solutions = []
    for source in range(n_sources):
        e = np.zeros(n_sources)
        e[source] = q_ref
        values = problem.boundary_values(inflow=e[0] * unit, outflow=q_ref * unit, opening_rates=list(e[1:]))
        solutions.append(problem.solve(values, tolerance=SOLVER_TOLERANCE, cancel=cancel))
    L = float(cfg.channel.length)
    response = np.zeros((n_sources, n_sources))  # node pressure above the outlet per unit source flow
    for source, sol in enumerate(solutions):
        outlet = _plane_pressure(sol, L)
        response[0, source] = (_plane_pressure(sol, 0.0) - outlet) / q_ref
        for k, opening in enumerate(domain.openings):
            response[1 + k, source] = (_port_pressure(sol, opening) - outlet) / q_ref
    inlets = cfg.active_inlets()
    node = [0 if i.placement == "pattern" else 1 + ports.index(i) for i in inlets]
    n = len(inlets)
    matrix = np.zeros((n, n))
    rhs = np.zeros(n)
    for a, inlet in enumerate(inlets):
        matrix[a, a] += float(inlet.tubing_resistance)
        for b in range(n):
            matrix[a, b] += response[node[a], node[b]]
        rhs[a] = float(inlet.pressure) - cfg.operating.outlet_pressure
    q = np.linalg.solve(matrix, rhs)
    if np.any(q <= 0):
        bad = [i.id for i, v in zip(inlets, q) if v <= 0]
        raise ValueError(f"Backflow into inlet(s) {bad}: the supply pressures cannot drive every inlet.")
    rates = {inlet.id: float(v) for inlet, v in zip(inlets, q)}
    source_rates = np.zeros(n_sources)
    for a, v in zip(node, q):
        source_rates[a] += v
    field = None
    for sol, rate in zip(solutions, source_rates):
        field = sol.scaled(rate / q_ref) if field is None else field.combine(sol, 1.0, rate / q_ref)
    field.info = dict(solutions[0].info)
    info = {"inlet_rates_m3_s": rates, "source_response_Pa_s_m3": response.tolist(), "unit_solves": n_sources,
            "assumption": "exact creeping-flow channel response; inertia added afterwards at these rates"}
    return rates, field, info


def _trace_sources(interpolator: StaggeredInterpolator, domain: Domain, points: np.ndarray, pattern,
                   port_codes: list[int], cancel) -> tuple[np.ndarray, object]:
    """Stream code of each point from tracing it back to the inflow plane or a port.

    A backward path that passes a stagnation line on a pillar face can be trapped there: arc-length
    steps overshoot the zero-velocity point back and forth. Such fluid lies on the dividing surface
    between two routes around the pillar, so its origin is ambiguous at the grid resolution anyway.
    It is restarted one fine cell upstream of where it stalled (or diagonally, if that point is solid),
    and the restarts are counted in ``result.counts["stagnation_restarts"]``.
    """
    result = trace(interpolator, points, stop_x=domain.x_in, direction=-1, openings=domain.shapes,
                   step=0.5 * domain.hmin, cancel=cancel, error_tolerance=TRACE_TOLERANCE, step_budget=STEP_BUDGET)
    stuck = np.nonzero(np.isin(result.status, (MAX_LENGTH, STAGNATION)))[0]
    restarts = 0
    h = domain.fine
    (x0, x1), (y0, y1), (z0, z1) = (tuple(f[[0, -1]]) for f in domain.grid.faces)
    for shift in ((-h, 0, 0), (-h, h, 0), (-h, -h, 0), (-h, 0, h), (-h, 0, -h), (-2 * h, 0, 0)):
        if not len(stuck):
            break
        start = result.end[stuck] + np.array(shift)
        start[:, 1] = np.clip(start[:, 1], y0 + 1e-3 * h, y1 - 1e-3 * h)
        start[:, 2] = np.clip(start[:, 2], z0 + 1e-3 * h, z1 - 1e-3 * h)
        free = ~interpolator.in_solid(start) & (start[:, 0] > x0)
        if not free.any():
            continue
        retry = trace(interpolator, start[free], stop_x=domain.x_in, direction=-1, openings=domain.shapes,
                      step=0.5 * domain.hmin, cancel=cancel, error_tolerance=TRACE_TOLERANCE, step_budget=STEP_BUDGET)
        ok = np.isin(retry.status, (REACHED_PLANE, REACHED_OPENING))
        index = stuck[free][ok]
        result.status[index] = retry.status[ok]
        result.end[index] = retry.end[ok]
        result.origin[index] = retry.origin[ok]
        result.residence_time[index] += retry.residence_time[ok]
        restarts += int(ok.sum())
        stuck = np.setdiff1d(stuck, index)
    result.counts = dict(result.counts or {}, stagnation_restarts=restarts)
    code = np.full(len(points), -1, dtype=np.int32)
    plane = result.status == REACHED_PLANE
    if plane.any():
        code[plane] = pattern(result.end[plane, 1], result.end[plane, 2])
    opening = result.status == REACHED_OPENING
    if opening.any():
        code[opening] = np.asarray(port_codes, dtype=np.int32)[result.origin[opening] - 1]
    return code, result


def solve_level(cfg: ProjectConfig, domain: Domain, cancel=None, report: Callable[[str], None] | None = None,
                use_cache: bool = True) -> LevelFlow:
    """Flow on one grid level (cached on disk by the field key)."""
    say = report or (lambda message: None)
    started = time.perf_counter()
    grid = domain.grid
    codes = cfg.stream_codes()
    mu_of = _viscosities(cfg)
    contrast = _contrast(cfg)
    yf, zf = grid.faces[1], grid.faces[2]
    yc, zc = grid.centers[1], grid.centers[2]
    unit = flux_matched_velocity(yf, zf, 1.0, 1.0).velocity.T  # (ny, nz), unit flow, uniform viscosity
    key = field_key(cfg, domain.cells)
    stored = disk_cache().load("cfd-field", key) if use_cache else None
    if stored is not None and stored["u"].shape == grid.face_shape(0):
        info = json.loads(str(stored["info"]))
        rates = json.loads(str(stored["rates"]))
        solution = StokesSolution(grid, domain.solid, stored["u"], stored["v"], stored["w"], stored["p"], {})
        info["field_cache"] = "disk"
        partition = _partition(cfg, rates, contrast)
        return LevelFlow(domain, solution, StaggeredInterpolator(solution), rates, partition, info)

    single_mu = cfg.material(cfg.active_inlets()[0].material).viscosity
    network = None
    base = None
    if cfg.operating.drive == "pressure":
        say("resistance matrix (one creeping-flow solve per source)")
        problem = StokesProblem(grid, domain.solid, single_mu, domain.openings)
        rates, base, network = _pressure_driven(cfg, domain, problem, unit, cancel)
    else:
        rates = {i.id: float(i.flow_rate) for i in cfg.active_inlets()}
        problem = None
    partition = _partition(cfg, rates, contrast)
    pattern_rate = sum(rates[i.id] for i in cfg.pattern_inlets())
    port_rates = [rates[p.id] for p in domain.ports]
    total = pattern_rate + sum(port_rates)
    active = cfg.active_inlets()
    density = sum(rates[i.id] * cfg.material(i.material).density for i in active) / total
    mean_mu = sum(rates[i.id] * cfg.material(i.material).viscosity for i in active) / total
    port_codes = [codes[p.id] for p in domain.ports]

    fluid = ~domain.solid
    history = []
    converged = True
    solution = None
    ns_info: dict = {}
    if contrast:
        Yc, Zc = np.meshgrid(yc, zc, indexing="ij")
        inflow_mu = np.vectorize(mu_of.get)(partition.evaluate(Yc, Zc)).astype(float)  # (ny, nz)
        mu_field = np.full(grid.shape, mean_mu)
        outflow_mu = np.full((len(yc), len(zc)), mean_mu)
        X, Y, Z = np.meshgrid(*grid.centers, indexing="ij")
        cell_points = np.column_stack([X[fluid], Y[fluid], Z[fluid]])
        converged = False
        def solve(inertia: bool):
            problem = StokesProblem(grid, domain.solid, mu_field, domain.openings)
            inflow = flux_matched_velocity(yf, zf, inflow_mu.T, pattern_rate).velocity.T
            outflow = flux_matched_velocity(yf, zf, outflow_mu.T, total).velocity.T
            values = problem.boundary_values(inflow=inflow, outflow=outflow, opening_rates=port_rates)
            return (problem, *navier_stokes(problem, values, density, inertia, cancel, initial=solution))

        # The material field is iterated in creeping flow; inertia (a small correction in this regime) is
        # added once on the converged viscosity field, instead of in every coupling iteration.
        for iteration in range(1, MAX_COUPLING + 1):
            say(f"viscosity coupling iteration {iteration}")
            previous = solution
            problem, solution, ns_info = solve(inertia=False)
            # RMS change: single interface cells flipping material give large but local maximum changes
            velocity_change = (float("inf") if previous is None else float(np.sqrt(
                sum(np.sum((a - b) ** 2) for a, b in ((solution.u, previous.u), (solution.v, previous.v),
                                                      (solution.w, previous.w)))
                / sum(np.sum(a ** 2) for a in (solution.u, solution.v, solution.w)))))
            material, _ = _trace_sources(StaggeredInterpolator(solution), domain, cell_points, partition.evaluate,
                                         port_codes, cancel)
            valid = material >= 0
            new_mu = mu_field.copy()
            flat = new_mu[fluid]
            flat[valid] = np.vectorize(mu_of.get)(material[valid])
            new_mu[fluid] = flat
            changed = float(np.mean(~np.isclose(new_mu[fluid], mu_field[fluid], rtol=1e-12)))
            mu_field = new_mu
            outflow_mu = mu_field[-1]
            history.append({"iteration": iteration, "changed_cell_fraction": changed,
                            "velocity_change": velocity_change, "untraced_cells": int((~valid).sum())})
            if iteration > 1 and (changed < COUPLING_TOLERANCE or (
                    changed < COUPLING_FLIP_TOLERANCE and velocity_change < COUPLING_VELOCITY_TOLERANCE)):
                converged = True
                break
        if cfg.cfd.inertia:
            say("Navier-Stokes solve on the converged viscosity field")
            problem, solution, ns_info = solve(inertia=True)
    else:
        if problem is None:
            say("Navier-Stokes solve")
            problem = StokesProblem(grid, domain.solid, single_mu, domain.openings)
        values = problem.boundary_values(inflow=pattern_rate * unit, outflow=total * unit, opening_rates=port_rates)
        solution, ns_info = navier_stokes(problem, values, density, cfg.cfd.inertia, cancel, base=base)
    outlet = float(cfg.channel.length or outlet_position(cfg))
    inflow_flux, outflow_flux = solution.plane_flux(0), solution.plane_flux(-1)
    # Continuity check on every cross-section plane outside the port openings.
    xf = grid.faces[0]
    plane_flux = np.sum(solution.u * grid.face_area(0), axis=(1, 2))
    expected = np.full(len(xf), inflow_flux)
    inside_port = np.zeros(len(xf), dtype=bool)
    for port, rate in zip(domain.ports, port_rates):
        radius = 0.5 * float(port.diameter)
        expected[xf >= port.station + radius] += rate
        inside_port |= np.abs(xf - port.station) < radius
    plane_error = float(np.max(np.abs(plane_flux - expected)[~inside_port]) / total)
    info = {
        **domain.describe(),
        "unknowns": int(problem.K.shape[0]),
        **ns_info,
        "coupling": {"iterations": history, "converged": converged} if contrast else None,
        "mass_balance": {"inflow_m3_s": inflow_flux, "ports_m3_s": float(sum(port_rates)),
                         "outflow_m3_s": outflow_flux, "planes_checked": int((~inside_port).sum()),
                         "relative_error": plane_error},
        "pressure_drop_Pa": _plane_pressure(solution, 0.0) - _plane_pressure(solution, outlet),
        "network": network,
        "reynolds_number": physics.reynolds_number(total, cfg.channel.width, cfg.channel.height, density, mean_mu),
        "seconds": time.perf_counter() - started,
        "field_cache": "computed",
        "warnings": domain.warnings,
    }
    if use_cache:
        disk_cache().save("cfd-field", key, {"u": solution.u, "v": solution.v, "w": solution.w, "p": solution.p,
                                              "info": np.array(json.dumps(_clean(info))),
                                              "rates": np.array(json.dumps(rates))})
    return LevelFlow(domain, solution, StaggeredInterpolator(solution), rates, partition, info)


def _partition(cfg: ProjectConfig, rates: dict[str, float], contrast: bool):
    grid_codes, _, _ = pattern_grid(cfg)
    code_to_id = {v: k for k, v in cfg.stream_codes().items()}
    rate_grid = np.vectorize(lambda c: rates[code_to_id[int(c)]])(grid_codes).astype(float)
    return partition_pattern(cfg, rates=rate_grid, viscosity_coupling=contrast)


# --------------------------------------------------------------------------- sections and pathlines


def _expected(cfg: ProjectConfig, rates: dict[str, float], x: float) -> dict[int, float]:
    codes = cfg.stream_codes()
    out = {codes[i.id]: rates[i.id] for i in cfg.pattern_inlets()}
    for port in cfg.port_inlets():
        if port.station < x:
            out[codes[port.id]] = rates[port.id]
    return out


def sample_section(cfg: ProjectConfig, flow: LevelFlow, name: str, x: float, y_edges: np.ndarray,
                   z_edges: np.ndarray, cancel=None) -> Section:
    """Trace every output cell centre at ``x`` back to its source; flux from the computed axial velocity."""
    yc, zc = 0.5 * (y_edges[1:] + y_edges[:-1]), 0.5 * (z_edges[1:] + z_edges[:-1])
    Y, Z = np.meshgrid(yc, zc)
    seeds = np.column_stack([np.full(Y.size, x), Y.ravel(), Z.ravel()])
    codes = cfg.stream_codes()
    code, result = _trace_sources(flow.interpolator, flow.domain, seeds, flow.partition.evaluate,
                                  [codes[p.id] for p in flow.domain.ports], cancel)
    axial = flow.interpolator(seeds)[:, 0].reshape(Y.shape)
    pattern_hit = result.status == REACHED_PLANE
    donor_y = np.where(pattern_hit, result.end[:, 1], np.nan).reshape(Y.shape)
    donor_z = np.where(pattern_hit, result.end[:, 2], np.nan).reshape(Y.shape)
    counts = {STATUS_NAMES.get(int(s), str(int(s))): int((result.status == s).sum()) for s in np.unique(result.status)}
    counts["restarted_at_stagnation"] = int(result.counts.get("stagnation_restarts", 0))
    section = Section(name=name, x=float(x), y_edges=y_edges, z_edges=z_edges, codes=code.reshape(Y.shape),
                      status=result.status.reshape(Y.shape).astype(np.int8), axial_velocity=axial,
                      donor_y=donor_y, donor_z=donor_z, residence_time=result.residence_time.reshape(Y.shape),
                      diagnostics={"expected_flux_m3_s": _expected(cfg, flow.rates, x), "trajectory_status": counts})
    section.fractions = _subcell_fractions(cfg, flow, section, x, cancel)
    weights = axial * section.cell_area()
    total = float(weights.sum())
    expected = section.diagnostics["expected_flux_m3_s"]
    expected_total = sum(expected.values())
    fractions = {int(c): float(np.sum(section.fractions[c - 1] * weights)) / total for c in expected}
    section.diagnostics["flux_fraction"] = fractions
    section.diagnostics["flux_fraction_error"] = {c: fractions[c] - q / expected_total for c, q in expected.items()}
    section.diagnostics["unresolved_flux_fraction"] = float(
        np.sum((1.0 - section.fractions.sum(axis=0)) * weights)) / total
    section.diagnostics["flux_basis"] = (f"computed axial velocity (CFD field); cells on an interface or without a "
                                         f"valid trajectory sampled {SUBCELL}x{SUBCELL} times")
    return section


def _subcell_fractions(cfg: ProjectConfig, flow: LevelFlow, section: Section, x: float, cancel) -> np.ndarray:
    """Per-stream fraction of each output cell: 1 inside a stream, sub-sampled where codes change or fail."""
    codes = section.codes
    n_streams = len(cfg.inlets)
    fractions = np.zeros((n_streams,) + codes.shape, dtype=np.float32)
    for code in np.unique(codes[codes > 0]):
        fractions[code - 1][codes == code] = 1.0
    padded = np.pad(codes, 1, mode="edge")
    mixed = codes < 0
    for dz, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        mixed |= padded[1 + dz:padded.shape[0] - 1 + dz, 1 + dy:padded.shape[1] - 1 + dy] != codes
    k, j = np.nonzero(mixed)
    if not len(k):
        return fractions
    offsets = (np.arange(SUBCELL) + 0.5) / SUBCELL
    y0, dy = section.y_edges[j], np.diff(section.y_edges)[j]
    z0, dz = section.z_edges[k], np.diff(section.z_edges)[k]
    ys = (y0[:, None, None] + dy[:, None, None] * offsets[None, None, :]).repeat(SUBCELL, axis=1)
    zs = (z0[:, None, None] + dz[:, None, None] * offsets[None, :, None]).repeat(SUBCELL, axis=2)
    points = np.column_stack([np.full(ys.size, x), ys.ravel(), zs.ravel()])
    stream_codes = cfg.stream_codes()
    sub, _ = _trace_sources(flow.interpolator, flow.domain, points, flow.partition.evaluate,
                            [stream_codes[p.id] for p in flow.domain.ports], cancel)
    sub = sub.reshape(len(k), SUBCELL * SUBCELL)
    for code in range(1, n_streams + 1):
        fractions[code - 1][k, j] = np.mean(sub == code, axis=1)
    return fractions


def mesh_convergence(fine: list[Section], coarse: list[Section], cells: tuple[int, int], height: float) -> dict:
    """Interface shift and overlap between the two grid levels, and the resulting uncertainty (GCI)."""
    ratio = cells[1] / cells[0]
    out = []
    for f, c in zip(fine, coarse):
        spacing = (float(np.diff(f.z_edges).mean()), float(np.diff(f.y_edges).mean()))
        present = sorted(int(v) for v in np.unique(np.concatenate([f.codes.ravel(), c.codes.ravel()])) if v >= 0)
        per_stream = {}
        for code in present:
            per_stream[code] = {
                "iou": metrics.iou(c.codes, f.codes, code),
                "interface_shift_m": metrics.interface_shift(c.codes, f.codes, code, spacing),
                "interface_distance_m": metrics.interface_distance(c.codes, f.codes, code, spacing),
                "flux_fraction_difference": (f.diagnostics.get("flux_fraction", {}).get(code, 0.0)
                                             - c.diagnostics.get("flux_fraction", {}).get(code, 0.0)),
            }
        shifts = [e["interface_shift_m"] for e in per_stream.values() if np.isfinite(e["interface_shift_m"])]
        shift = max(shifts) if shifts else 0.0
        ious = [e["iou"] for e in per_stream.values() if np.isfinite(e["iou"])]
        out.append({"station": f.name, "x_m": f.x, "per_stream": per_stream, "interface_shift_m": shift,
                    "min_iou": min(ious) if ious else float("nan"),
                    "uncertainty_m": GCI_SAFETY * shift / (ratio ** GCI_ORDER - 1.0)})
    final = out[-1]
    return {
        "cells": list(cells), "ratio": ratio, "assumed_order": GCI_ORDER, "safety_factor": GCI_SAFETY,
        "stations": out, "interface_shift_m": final["interface_shift_m"], "uncertainty_m": final["uncertainty_m"],
        "uncertainty_fraction_of_H": final["uncertainty_m"] / height, "min_iou": final["min_iou"],
        "max_uncertainty_m": max(s["uncertainty_m"] for s in out),
        "method": "grid convergence index: safety factor x interface shift between the grids / (ratio^order - 1); "
                  "interface shift = area where a stream differs / its interface length",
    }


def cfd_pathlines(cfg: ProjectConfig, flow: LevelFlow, outlet_x: float, n_lines: int = N_PATHLINES,
                  cancel=None, seed: int = 0) -> Pathlines:
    """Forward pathlines through the full field from x = 0 (and the ports) to the outlet."""
    rng = np.random.default_rng(seed)
    domain = flow.domain
    W, H = cfg.channel.width, cfg.channel.height
    codes = cfg.stream_codes()
    pattern_rate = sum(flow.rates[i.id] for i in cfg.pattern_inlets())
    port_rates = [flow.rates[p.id] for p in domain.ports]
    total = pattern_rate + sum(port_rates)
    n_pattern = max(int(round(n_lines * pattern_rate / total)), 8)
    port_counts = [max(int(round(n_lines * q / total)), 4) for q in port_rates]
    n = n_pattern + sum(port_counts)
    faces = domain.grid.faces[0]
    planes = faces[(faces > 0.0) & (faces < outlet_x)]
    if len(planes) > MAX_PATHLINE_PLANES - 1:
        planes = planes[np.linspace(0, len(planes) - 1, MAX_PATHLINE_PLANES - 1).round().astype(int)]
    planes = np.append(planes, outlet_x)
    events = [(float(x), 1, -1) for x in planes]
    events += [(float(p.station - 0.5 * p.diameter), 0, k) for k, p in enumerate(domain.ports)]
    events.sort()
    m = 1 + len(events)
    points = np.full((n, m, 3), np.nan)
    times = np.full((n, m), np.nan)
    code = np.full(n, -1, dtype=np.int32)
    origin = np.zeros(n, dtype=np.int16)
    status = np.full(n, REACHED_PLANE, dtype=np.int8)
    alive = np.zeros(n, dtype=bool)
    pos = np.full((n, 3), np.nan)
    clock = np.zeros(n)
    step = 0.5 * domain.hmin

    def advance(idx: np.ndarray, x_plane: float) -> None:
        result = trace(flow.interpolator, pos[idx], stop_x=x_plane, direction=1, openings=domain.shapes, step=step,
                       cancel=cancel, error_tolerance=TRACE_TOLERANCE, step_budget=STEP_BUDGET)
        pos[idx] = result.end
        clock[idx] += result.residence_time
        ended = result.status != REACHED_PLANE
        status[idx[ended]] = result.status[ended]
        alive[idx[ended]] = False
        return ended

    y0, z0 = _DevelopedSpeed(W, H).sample(n_pattern, rng)
    pos[:n_pattern] = np.column_stack([np.full(n_pattern, domain.x_in), y0, z0])
    code[:n_pattern] = flow.partition.evaluate(y0, z0)
    alive[:n_pattern] = True
    advance(np.arange(n_pattern), 0.0)  # through the inflow buffer; times start at x = 0
    clock[:] = 0.0
    points[alive, 0], times[alive, 0] = pos[alive], 0.0
    start = n_pattern
    for column, (x_event, kind, k) in enumerate(events, start=1):
        if cancel is not None and cancel():
            raise SolveCancelled()
        if kind == 0:
            count = port_counts[k]
            idx = np.arange(start, start + count)
            start += count
            pos[idx] = _port_seeds(domain.shapes[k], count, W, H, 0.05 * domain.hmin, rng)
            clock[idx], alive[idx] = 0.0, True
            code[idx], origin[idx] = codes[domain.ports[k].id], k + 1
            points[idx, column], times[idx, column] = pos[idx], 0.0
            continue
        idx = np.nonzero(alive & (pos[:, 0] < x_event))[0]
        if len(idx):
            ended = advance(idx, x_event)
            points[idx[ended], column], times[idx[ended], column] = pos[idx[ended]], clock[idx[ended]]
        points[alive, column], times[alive, column] = pos[alive], clock[alive]
    return Pathlines(points=points.astype(np.float32), time=times.astype(np.float32), code=code, status=status,
                     origin=origin, outlet_x=float(outlet_x))


# --------------------------------------------------------------------------- application service


def cached_result(cfg: ProjectConfig) -> SimulationResult | None:
    """The stored CFD result for this case, if one was computed before (no computation)."""
    try:
        key = result_key(cfg)
    except Exception:  # noqa: BLE001 - invalid geometry: nothing cached
        return None
    arrays = disk_cache().load("cfd-result", key)
    if arrays is None:
        return None
    try:
        result = SimulationResult.from_arrays(arrays)
    except (ValueError, KeyError):
        return None
    return _restamp(result, cfg)


def _restamp(result: SimulationResult, cfg: ProjectConfig) -> SimulationResult:
    """A cached result presented for ``cfg`` (same CFD key; the new solver's settings may differ)."""
    result.config = cfg.to_dict()
    result.input_hash = cfg.input_hash()
    computed = result.diagnostics.get("computed_seconds", result.timings.get("total"))
    result.diagnostics["cache"] = f"disk cache (computed in {computed:.1f} s)" if computed else "disk cache"
    return result


def forget(cfg: ProjectConfig) -> int:
    """Delete the stored CFD result and flow fields of this case (both grid levels); the number removed."""
    try:
        keys = [("cfd-result", result_key(cfg))] + [("cfd-field", field_key(cfg, cells)) for cells in
                                                     [cfg.cfd.cells, convergence_cells(cfg)] if cells]
    except Exception:  # noqa: BLE001 - invalid geometry: nothing stored
        return 0
    return sum(disk_cache().delete(namespace, key) for namespace, key in keys)


def is_current(result: SimulationResult | None, cfg: ProjectConfig) -> bool:
    if result is None:
        return False
    try:
        return result.provenance.get("cfd_key") == result_key(cfg)
    except Exception:  # noqa: BLE001
        return False


def run_cfd(cfg: ProjectConfig, progress: ProgressFn | None = None, cancel: Callable[[], bool] | None = None,
            use_cache: bool = True, recompute: bool = False) -> SimulationResult:
    """Compute (or load) the CFD reference for ``cfg``. Always returns a result with an explicit status.

    ``recompute`` first deletes the stored result and flow fields of this case, so everything is
    computed again (a measured compute time) and stored anew.
    """
    started = time.perf_counter()
    watch = Stopwatch()
    report = progress or (lambda stage, fraction, message: None)
    base = {"backend": "cfd", "solver": dict(solver_identity(), model="cfd_reference", cfd_version=CFD_VERSION),
            "config": cfg.to_dict(), "input_hash": cfg.input_hash()}
    try:
        report("validate", 0.0, "Validating configuration")
        issues = validate_config(cfg)
        errors = [i for i in issues if i.level == "error"]
        if errors:
            return failed_result("failed", "Invalid configuration: " + "; ".join(str(e) for e in errors), **base)
        reasons = unsupported(cfg)
        if reasons:
            return failed_result("unsupported", "Not supported by the CFD reference: " + "; ".join(reasons), **base)
        key = result_key(cfg)
        removed = forget(cfg) if recompute else 0
        if use_cache:
            cached = cached_result(cfg)
            if cached is not None:
                report("done", 1.0, "Loaded the stored CFD reference")
                cached.timings = dict(cached.timings, cache_lookup=time.perf_counter() - started)
                return cached
        outlet_x = outlet_position(cfg)
        levels = [cfg.cfd.cells] + [c for c in [convergence_cells(cfg)] if c]
        spans = [(0.02, 0.6), (0.72, 0.85)]
        y_edges, z_edges = output_grid(cfg)
        station_list = stations(cfg)
        flows: list[LevelFlow] = []
        level_sections: list[list[Section]] = []
        for level, cells in enumerate(levels):
            lo, hi = spans[level]
            label = "reference grid" if level == 0 else "convergence grid"
            report("mesh", lo, f"CFD {label}: {cells} cells across")
            with watch.time(f"mesh_{cells}"):
                domain = build_domain(cfg, cells, outlet_x)
            with watch.time(f"flow_{cells}"):
                flow = solve_level(cfg, domain, cancel, lambda msg, _lo=lo, _l=label: report(
                    "flow", _lo, f"CFD {_l} ({domain.describe()['cells']:,} cells): {msg}"), use_cache)
            flows.append(flow)
            report("sections", hi, f"CFD {label}: tracing {len(station_list)} sections cell by cell")
            with watch.time(f"sections_{cells}"):
                level_sections.append([sample_section(cfg, flow, name, x, y_edges, z_edges, cancel)
                                       for name, x in station_list])
        fine = flows[0]
        sections = level_sections[0]
        convergence = None
        if len(levels) > 1:
            convergence = mesh_convergence(sections, level_sections[1], (levels[1], levels[0]), cfg.channel.height)
        report("pathlines", 0.9, "Tracing CFD pathlines for the 3D view")
        warnings = [f"CFD grid: {w}" for w in fine.domain.warnings]
        with watch.time("pathlines"):
            try:
                pathlines = cfd_pathlines(cfg, fine, outlet_x, cancel=cancel)
            except SolveCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - display data only
                pathlines = None
                warnings.append(f"3D pathlines unavailable: {type(exc).__name__}: {exc}")
        report("checks", 0.97, "Checking the CFD reference")
        result = _assemble(cfg, flows, sections, convergence, pathlines, warnings, key, base)
        result.timings = dict(watch.timings, total=time.perf_counter() - started)
        result.diagnostics["computed_seconds"] = result.timings["total"]
        if recompute:
            result.diagnostics["cache"] = f"recomputed ({removed} stored entries cleared)"
        if use_cache:
            disk_cache().save("cfd-result", key, result.to_arrays())
        report("done", 1.0, "Done")
        return result
    except SolveCancelled:
        return failed_result("cancelled", "Cancelled by request.", timings=watch.timings, **base)
    except Exception as exc:  # noqa: BLE001 - reported as a failed result at the application boundary
        return failed_result("failed", f"{type(exc).__name__}: {exc}", timings=watch.timings,
                             diagnostics={"traceback": traceback.format_exc()}, **base)


def _assemble(cfg: ProjectConfig, flows: list[LevelFlow], sections: list[Section], convergence: dict | None,
              pathlines: Pathlines | None, warnings: list[str], key: str, base: dict) -> SimulationResult:
    fine = flows[0]
    H = cfg.channel.height
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "kind": "numerics"})

    residual = max(f.info["stokes"]["relative_residual"] for f in flows)
    check("linear_solver", residual <= 1e-5, f"largest relative residual {residual:.1e}")
    divergence = max(f.info["stokes"]["max_cell_divergence"] for f in flows)
    check("divergence", divergence <= 1e-4, f"largest cell divergence {divergence:.1e} (relative to the cell flux)")
    balance = max(f.info["mass_balance"]["relative_error"] for f in flows)
    check("mass_balance", balance <= 1e-4, f"flux through every cross-section plane matches the inflow upstream of "
                                           f"it to {balance:.1e} of the total")
    picard = fine.info.get("picard")
    if cfg.cfd.inertia:
        converged = all(f.info.get("picard") and f.info["picard"]["converged"] for f in flows)
        check("inertia_iteration", converged,
              f"{picard['iterations']} Picard steps; inertia changes the velocity by "
              f"{100 * picard['inertia_effect']:.3g}% of its maximum" if picard else "not run")
        if not converged:
            raise RuntimeError(f"The Navier-Stokes (Picard) iteration did not converge at Re = "
                               f"{fine.info['reynolds_number']:.3g}; the CFD reference supports moderate Re only.")
    coupling = fine.info.get("coupling")
    if coupling is not None:
        check("viscosity_coupling", coupling["converged"],
              f"{len(coupling['iterations'])} iterations; last change "
              f"{100 * coupling['iterations'][-1]['changed_cell_fraction']:.3f}% of the fluid cells")
        if not coupling["converged"]:
            warnings.append("CFD viscosity coupling did not converge; the material field still changes.")
    final = sections[-1]
    invalid = float(np.mean(final.codes < 0))
    check("trajectories", invalid <= 0.005, f"{100 * invalid:.3f}% of final-section cells without a valid trajectory")
    errors = [abs(v) for s in sections for v in s.diagnostics["flux_fraction_error"].values()]
    max_error = max(errors) if errors else 0.0
    check("flux_conservation", max_error <= 0.005, f"max per-stream flux-fraction error {100 * max_error:.3f} pp")
    if convergence is not None:
        uncertainty = convergence["uncertainty_m"]
        check("mesh_convergence", uncertainty <= TARGET_FRACTION_OF_H * H,
              f"interface uncertainty {uncertainty * 1e6:.3g} um ({100 * uncertainty / H:.2f}% of H; target "
              f"{100 * TARGET_FRACTION_OF_H:.0f}%) from {convergence['cells'][0]} vs {convergence['cells'][1]} cells")
    numerics_ok = all(c["ok"] for c in checks)
    if convergence is not None:
        status = (f"CFD reference; mesh uncertainty {convergence['uncertainty_m'] * 1e6:.2g} um "
                  f"({100 * convergence['uncertainty_fraction_of_H']:.2g}% of H)")
    else:
        status = "CFD reference; mesh uncertainty not estimated (convergence check off)"
    if not numerics_ok:
        status += "; numerical checks below target"
    streams = stream_table(cfg, fine.rates)
    return SimulationResult(
        status="success",
        message=f"{len(sections)} sections; {status}",
        streams=streams,
        sections=sections,
        diagnostics={
            "cfd": {"levels": [f.info for f in flows], "mesh_convergence": convergence,
                    "stations_m": [s.x for s in sections]},
            "pattern_partition": fine.partition.describe(),
            "reynolds_number": fine.info["reynolds_number"],
            "max_flux_fraction_error": max_error,
            "section_checks": [{"section": s.name,
                                "max_flux_fraction_error": max(abs(v) for v in s.diagnostics["flux_fraction_error"].values()),
                                "unresolved_flux_fraction": s.diagnostics["unresolved_flux_fraction"]}
                               for s in sections],
            "network": fine.info.get("network"),
            "cache": "computed",
        },
        validity={"model": "cfd_reference", "model_card": MODEL_CARDS["cfd_reference"], "status": status,
                  "inside_verified_envelope": numerics_ok, "numerics_ok": numerics_ok, "envelope_ok": True,
                  "externally_validated": False, "checks": checks},
        warnings=warnings,
        pathlines=pathlines,
        provenance={"models": {"flow": "cfd_reference"}, "cfd": {"cells": cfg.cfd.cells,
                    "convergence_cells": convergence_cells(cfg), "inertia": cfg.cfd.inertia, "buffer": cfg.cfd.buffer},
                    "cfd_key": key, "tracing_tolerance": TRACE_TOLERANCE,
                    "calibration": "none (reference solution)"},
        **base,
    )


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
