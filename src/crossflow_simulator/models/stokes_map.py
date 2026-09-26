"""Flow fields of feature groups from the 3D Stokes solver.

Equal-property path
    Stokes flow is linear in the source rates (upstream channel and ports), so
    solved fields are kept per geometry and any request in their span is a
    superposition; a new flow ratio costs at most one solve, a common viscosity
    only scales pressure, and scaling all rates scales the field. Solutions are
    cached in memory and on disk under a key made of the grid, solid mask, and
    opening profiles.

Viscosity-coupled path (experimental)
    With a viscosity contrast the flow depends on where each material is. The
    field is iterated: solve with the current viscosity field, trace every
    fluid cell back to its source to update the field, and repeat until the
    material assignment stops changing. Inflow/outflow profiles are the
    developed variable-viscosity profiles of the same discretization. This is
    not validated against reference data and is reported as experimental.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .. import cache as cache_mod
from ..geometry import stable_hash
from ..solvers.section import flux_matched_velocity
from ..solvers.stokes import StokesProblem, StokesSolution
from ..transport import REACHED_OPENING, REACHED_PLANE, StaggeredInterpolator, trace
from .domain import GroupDomain

SPAN_VERSION = "stokes-span/1"


@dataclass
class GroupFlow:
    domain: GroupDomain
    solution: StokesSolution
    interpolator: StaggeredInterpolator
    upstream_rate: float
    port_rates: list[float]
    viscosity: float | np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def total_rate(self) -> float:
        return self.upstream_rate + sum(self.port_rates)


def _developed(domain: GroupDomain, viscosity: float | np.ndarray, rate: float) -> np.ndarray:
    grid = domain.grid
    if rate == 0:
        return np.zeros((grid.shape[1], grid.shape[2]))
    mu = viscosity if np.ndim(viscosity) == 0 else np.asarray(viscosity).T  # (nz, ny) for the section solver
    return flux_matched_velocity(grid.yf, grid.zf, mu, rate).velocity.T  # (ny, nz)


def _store_nbytes(store: dict) -> int:
    total = sum(sum(a.nbytes for a in sol.arrays().values()) for sol in store["solutions"])
    problem = store.get("problem")
    if problem is not None:
        total += 4 * (problem.K.data.nbytes + problem.K.indices.nbytes)
    return total


def solve_for_rates(
    domain: GroupDomain,
    rates: Sequence[float],
    method: str = "auto",
    tolerance: float = 1e-12,
    cancel: Callable[[], bool] | None = None,
) -> tuple[StokesSolution, dict]:
    """Unit-viscosity Stokes field for source rates ``[upstream, port_1, ...]`` (m^3/s).

    Creeping flow is linear in the source rates. Solutions already computed for
    this geometry are kept (memory and disk) with their normalized rate vectors;
    a request inside their span is a superposition, anything else costs exactly
    one new solve (whose operator and preconditioner are reused).
    """
    rates = np.asarray(rates, dtype=float)
    total = float(rates.sum())
    if total <= 0:
        raise ValueError("The group must carry a positive flow rate.")
    target = rates / total
    key = stable_hash({"version": SPAN_VERSION, "domain": domain.key(), "tolerance": tolerance})
    memory = cache_mod.memory_cache()
    store = memory.get(("span", key))
    source = "memory"
    grid = domain.grid
    if store is None:
        store = {"rates": [], "solutions": [], "problem": None}
        stored = cache_mod.disk_cache().load("stokes-span", key)
        if stored is not None and stored["rates"].shape[1] == len(rates):
            for i, r in enumerate(stored["rates"]):
                store["rates"].append(r)
                store["solutions"].append(StokesSolution(grid, domain.solid, stored[f"u{i}"], stored[f"v{i}"],
                                                         stored[f"w{i}"], stored[f"p{i}"], {"cached": True}))
            source = "disk"
    if store["rates"]:
        basis = np.array(store["rates"]).T
        coefficients, *_ = np.linalg.lstsq(basis, target, rcond=None)
        if np.linalg.norm(basis @ coefficients - target) <= 1e-12 * np.linalg.norm(target):
            solution = _combine(store["solutions"], total * coefficients, grid, domain.solid)
            memory.put(("span", key), store, _store_nbytes(store))
            return solution, {"cache": f"{source} (superposition of {len(store['rates'])} solved rate vectors)",
                              "key": key}
    started = time.perf_counter()
    if store["problem"] is None:
        store["problem"] = StokesProblem(grid, domain.solid, 1.0, domain.openings)
    problem = store["problem"]
    unit_profile = _developed(domain, 1.0, 1.0)
    values = problem.boundary_values(inflow=target[0] * unit_profile, outflow=unit_profile,
                                     opening_rates=list(target[1:]))
    solved = problem.solve(values, method=method, tolerance=tolerance, cancel=cancel)
    store["rates"].append(target)
    store["solutions"].append(solved)
    memory.put(("span", key), store, _store_nbytes(store))
    arrays = {"rates": np.array(store["rates"])}
    for i, sol in enumerate(store["solutions"]):
        arrays.update({f"u{i}": sol.u, f"v{i}": sol.v, f"w{i}": sol.w, f"p{i}": sol.p})
    cache_mod.disk_cache().save("stokes-span", key, arrays)
    info = {
        "cache": "computed" if len(store["rates"]) == 1 else f"computed (rate vector {len(store['rates'])})",
        "key": key,
        "seconds": time.perf_counter() - started,
        "solves": [solved.info],
        "operator_timings": dict(problem.timings),
    }
    return solved.scaled(total), info


def _combine(solutions: list[StokesSolution], coefficients, grid, solid) -> StokesSolution:
    u = sum(c * s.u for c, s in zip(coefficients, solutions))
    v = sum(c * s.v for c, s in zip(coefficients, solutions))
    w = sum(c * s.w for c, s in zip(coefficients, solutions))
    p = sum(c * s.p for c, s in zip(coefficients, solutions))
    return StokesSolution(grid, solid, u, v, w, p, {"superposition": list(map(float, coefficients))})


def equal_property_flow(
    domain: GroupDomain,
    upstream_rate: float,
    port_rates: Sequence[float],
    viscosity: float,
    method: str = "auto",
    tolerance: float = 1e-12,
    cancel: Callable[[], bool] | None = None,
) -> GroupFlow:
    unit, info = solve_for_rates(domain, [upstream_rate, *port_rates], method, tolerance, cancel)
    solution = StokesSolution(domain.grid, domain.solid, unit.u, unit.v, unit.w, viscosity * unit.p, unit.info)
    info = dict(info)
    info["superposition"] = {"upstream_rate": upstream_rate, "port_rates": list(port_rates)}
    info["pressure_drop_Pa"] = solution.pressure_drop()
    info["flux_in_m3_s"] = solution.plane_flux(0)
    info["flux_out_m3_s"] = solution.plane_flux(-1)
    return GroupFlow(domain, solution, StaggeredInterpolator(solution), upstream_rate, list(port_rates), viscosity, info)


def coupled_viscosity_flow(
    domain: GroupDomain,
    upstream_rate: float,
    port_rates: Sequence[float],
    upstream_viscosity: Callable[[np.ndarray, np.ndarray], np.ndarray],
    port_viscosities: Sequence[float],
    reference_viscosity: float,
    method: str = "auto",
    tolerance: float = 1e-12,
    max_iterations: int = 8,
    change_tolerance: float = 2e-3,
    step_fraction: float = 0.5,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> GroupFlow:
    """Iterate the Stokes solve with the viscosity field implied by the traced material origins."""
    grid = domain.grid
    fluid = ~domain.solid
    xc, yc, zc = grid.centers
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    cell_points = np.column_stack([X[fluid], Y[fluid], Z[fluid]])
    start = equal_property_flow(domain, upstream_rate, port_rates, reference_viscosity, method, tolerance, cancel)
    flow = start
    mu = np.full(grid.shape, reference_viscosity)
    history = []
    converged = False
    inflow_mu = upstream_viscosity(np.repeat(yc[:, None], len(zc), 1).ravel(),
                                   np.repeat(zc[None, :], len(yc), 0).ravel()).reshape(len(yc), len(zc))
    for iteration in range(1, max_iterations + 1):
        if progress:
            progress(f"viscosity coupling iteration {iteration}")
        result = trace(flow.interpolator, cell_points, stop_x=grid.faces[0][0], direction=-1,
                       openings=domain.opening_shapes, step=step_fraction * min(float(w.min()) for w in grid.widths),
                       cancel=cancel)
        new_mu = mu.copy()
        values = np.full(len(cell_points), np.nan)
        plane = result.status == REACHED_PLANE
        values[plane] = upstream_viscosity(result.end[plane, 1], result.end[plane, 2])
        opening = result.status == REACHED_OPENING
        values[opening] = np.asarray(port_viscosities, dtype=float)[result.origin[opening] - 1]
        valid = np.isfinite(values)
        flat = new_mu[fluid]
        flat[valid] = values[valid]
        new_mu[fluid] = flat
        changed = float(np.mean(~np.isclose(new_mu[fluid], mu[fluid], rtol=1e-12)))
        mu = new_mu
        outflow_mu = mu[-1]
        inflow = _developed(domain, inflow_mu, upstream_rate)
        outflow = _developed(domain, outflow_mu, upstream_rate + float(np.sum(port_rates)))
        problem = StokesProblem(grid, domain.solid, mu, domain.openings)
        values_bc = problem.boundary_values(inflow=inflow, outflow=outflow, opening_rates=list(port_rates))
        solution = problem.solve(values_bc, method=method, tolerance=tolerance, cancel=cancel)
        previous = flow.solution
        flow = GroupFlow(domain, solution, StaggeredInterpolator(solution), upstream_rate, list(port_rates), mu)
        velocity_change = float(np.max(np.abs(solution.u - previous.u)) / max(np.max(np.abs(solution.u)), 1e-300))
        history.append({
            "iteration": iteration,
            "changed_cell_fraction": changed,
            "velocity_change": velocity_change,
            "invalid_trajectories": int(np.sum(~valid)),
            "residual": solution.info.get("relative_residual"),
        })
        if iteration > 1 and changed < change_tolerance and velocity_change < 1e-2:
            converged = True
            break
    flow.info = {
        "coupling": {"iterations": history, "converged": converged},
        "pressure_drop_Pa": flow.solution.pressure_drop(),
        "flux_in_m3_s": flow.solution.plane_flux(0),
        "flux_out_m3_s": flow.solution.plane_flux(-1),
        "cache": "not cached (viscosity-coupled)",
    }
    return flow
