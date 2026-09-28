"""Headless application service shared by the CLI, the GUI worker, and tests.

``simulate(config, progress, cancel)`` runs the stages

    validate -> plan groups -> (pressure network) -> inlet pattern -> group
    flows -> transport sections -> conservation/quality checks -> result

and always returns a :class:`SimulationResult`. Invalid input, numerical
failure, and cancellation produce explicit non-success results; a failed
solve is never presented as a prediction.
"""
from __future__ import annotations

import time
import traceback
from typing import Callable

import numpy as np

from . import physics, solver_identity
from .cache import Stopwatch, memory_cache
from .config import ProjectConfig, validate_config
from .inlets import partition_pattern, pattern_grid
from .geometry import stable_hash
from .maps import (
    KIND_PATTERN, ChainLink, ConservativeRemap, FlowChain, SectionMap, adaptive_section, group_map,
    pattern_fractions, subcell_fractions,
)
from .materials import stream_table
from .models.domain import build_domain, plan_groups
from .models.stokes_map import coupled_viscosity_flow, equal_property_flow
from .models.validity import assess
from .results import Section, SimulationResult, failed_result
from .solvers.network import solve_network
from .solvers.section import flux_matched_velocity
from .solvers.stokes import SolveCancelled

ProgressFn = Callable[[str, float, str], None]
MAP_CACHE_VERSION = "section-maps/1"


def output_grid(cfg: ProjectConfig) -> tuple[np.ndarray, np.ndarray]:
    W, H = cfg.channel.width, cfg.channel.height
    ny = cfg.grid.output_ny
    nz = cfg.grid.output_nz or max(8, int(round(ny * H / W)))
    return np.linspace(0.0, W, ny + 1), np.linspace(0.0, H, nz + 1)


def developed_velocity(y_edges: np.ndarray, z_edges: np.ndarray, width: float, height: float,
                       flow_rate: float, viscosity: float | np.ndarray) -> np.ndarray:
    """Developed axial velocity on the output grid carrying ``flow_rate`` (variable viscosity allowed)."""
    if np.ndim(viscosity) == 0:
        yc, zc = 0.5 * (y_edges[1:] + y_edges[:-1]), 0.5 * (z_edges[1:] + z_edges[:-1])
        profile = physics.rectangular_duct_profile(yc, zc, width, height, normalization="analytic")
        return np.asarray(profile) * flow_rate / (width * height)
    return flux_matched_velocity(y_edges, z_edges, viscosity, flow_rate).velocity


def simulate(cfg: ProjectConfig, progress: ProgressFn | None = None,
             cancel: Callable[[], bool] | None = None) -> SimulationResult:
    started = time.perf_counter()
    watch = Stopwatch()
    report = progress or (lambda stage, fraction, message: None)
    base = {
        "backend": "new",
        "solver": solver_identity(),
        "config": cfg.to_dict(),
        "input_hash": cfg.input_hash(),
    }

    def checkpoint() -> None:
        if cancel is not None and cancel():
            raise SolveCancelled()

    try:
        report("validate", 0.0, "Validating configuration")
        with watch.time("validate"):
            issues = validate_config(cfg)
        errors = [i for i in issues if i.level == "error"]
        if errors:
            return failed_result(
                "failed", "Invalid configuration: " + "; ".join(str(e) for e in errors),
                diagnostics={"issues": [vars(i) for i in issues]}, timings=watch.timings, **base,
            )
        warnings = [f"{i.path}: {i.message}" for i in issues if i.level == "warning"]
        notes = [f"{i.path}: {i.message}" for i in issues if i.level == "info"]
        W, H = cfg.channel.width, cfg.channel.height
        codes = cfg.stream_codes()
        active = cfg.active_inlets()

        report("plan", 0.05, "Planning inlet and pillar operations")
        with watch.time("plan"):
            groups = plan_groups(cfg)
            domains = [build_domain(cfg, g) for g in groups]
        for domain in domains:
            warnings.extend(f"{domain.group.label}: {w}" for w in domain.warnings)
        checkpoint()

        viscosities = {i.id: cfg.material(i.material).viscosity for i in active}
        mu_values = sorted(set(viscosities.values()))
        contrast = len(mu_values) > 1
        coupling = contrast and cfg.model.viscosity_coupling in ("auto", "iterate")
        code_to_mu = {codes[i.id]: cfg.material(i.material).viscosity for i in cfg.inlets}

        # ---- flow rates (prescribed, or from the pressure network)
        network_info = None
        if cfg.operating.drive == "pressure":
            report("network", 0.1, "Solving the hydraulic network")
            with watch.time("network"):
                rates, network_info = _pressure_driven_rates(cfg, domains, checkpoint)
        else:
            rates = {i.id: float(i.flow_rate) for i in active}
        streams = stream_table(cfg, rates)

        report("pattern", 0.15, "Partitioning the inlet pattern by flux")
        with watch.time("pattern"):
            grid_codes, _, pattern_inlets = pattern_grid(cfg)
            rate_grid = np.zeros(grid_codes.shape)
            code_to_id = {v: k for k, v in codes.items()}
            for index, c in np.ndenumerate(grid_codes):
                rate_grid[index] = rates[code_to_id[int(c)]]
            partition = partition_pattern(cfg, rates=rate_grid, viscosity_coupling=coupling)
        warnings.extend(partition.notes if any("infeasible" in n for n in partition.notes) else [])
        pattern_rate = float(rate_grid.sum())
        reference_mu = float(sum(rates[i] * viscosities[i] for i in rates) / sum(rates.values()))
        if not contrast:
            reference_mu = mu_values[0]
        elif not coupling:
            warnings.append("Viscosity contrast ignored (coupling off): equal-property map with flux-weighted "
                            f"viscosity {reference_mu:.4g} Pa s.")

        # ---- reuse of transport maps: an equal-property flow field (and therefore every donor map) is
        # unchanged when only the inlet pattern changes (materials, colours, pattern flow ratios at the same
        # total flow). Then the analytic pattern is re-evaluated at the stored donor coordinates.
        y_edges, z_edges = output_grid(cfg)
        group_rates, running = [], pattern_rate
        for domain in domains:
            port_rates = [rates[pid] for pid in domain.port_ids]
            group_rates.append([running, port_rates])
            running += sum(port_rates)
        reuse_key = None
        total_rate = sum(rates.values())
        if not coupling and cfg.model.transport == "categorical" and domains:
            # Creeping-flow maps depend on flow *fractions* only: common viscosity and overall flow scale
            # change pressures and speeds, not streamline geometry.
            fractions_key = [[up / total_rate, [q / total_rate for q in ports]] for up, ports in group_rates]
            reuse_key = ("section-maps", stable_hash({
                "version": MAP_CACHE_VERSION, "domains": [d.key() for d in domains], "rates": fractions_key,
                "grid": [len(y_edges), len(z_edges), W, H],
                "tracing": [cfg.model.step_fraction, cfg.model.integrator, cfg.model.tracing_tolerance],
                "stride": cfg.grid.map_stride, "ports": [[codes[p] for p in d.port_ids] for d in domains],
            }))
        cached_maps = memory_cache().get(reuse_key) if reuse_key else None

        # ---- group flows
        links: list[ChainLink] = []
        flow_info = [] if cached_maps is None else _rescaled_flow_info(cached_maps, reference_mu, total_rate)
        upstream_rate = pattern_rate
        coupling_converged = True
        for g, domain in enumerate(domains if cached_maps is None else []):
            checkpoint()
            fraction = 0.2 + 0.45 * g / max(len(domains), 1)
            report("flow", fraction, f"Solving Stokes flow for {domain.group.label}")
            port_rates = [rates[pid] for pid in domain.port_ids]
            with watch.time("flow"):
                if coupling:
                    prefix = FlowChain(links, partition.evaluate, cfg.model.step_fraction, cfg.model.integrator, cancel,
                              cfg.model.tracing_tolerance)

                    def upstream_mu(yq: np.ndarray, zq: np.ndarray, _prefix=prefix, _g=g) -> np.ndarray:
                        traced = _prefix.trace(yq, zq, _g - 1)
                        return np.array([code_to_mu.get(int(c), reference_mu) for c in traced.code])

                    port_mu = [viscosities[pid] for pid in domain.port_ids]
                    flow = coupled_viscosity_flow(
                        domain, upstream_rate, port_rates, upstream_mu, port_mu, reference_mu,
                        cfg.model.linear_solver, cfg.model.tolerance, cfg.model.max_coupling_iterations,
                        step_fraction=cfg.model.step_fraction, cancel=cancel,
                        progress=lambda msg: report("flow", fraction, msg),
                    )
                    coupling_converged &= bool(flow.info["coupling"]["converged"])
                else:
                    flow = equal_property_flow(domain, upstream_rate, port_rates, reference_mu,
                                               cfg.model.linear_solver, cfg.model.tolerance, cancel)
            links.append(ChainLink(flow, [codes[pid] for pid in domain.port_ids]))
            info = dict(flow.info)
            info.update({
                "group": domain.group.label,
                "x_range_m": [domain.group.x_start, domain.group.x_end],
                "ports": domain.port_ids,
                "pillars": [p.id for p in domain.group.pillars],
                "resolution": domain.resolution,
                "upstream_rate_m3_s": upstream_rate,
                "port_rates_m3_s": port_rates,
            })
            flow_info.append(info)
            upstream_rate += sum(port_rates)

        # ---- transport and sections
        chain = FlowChain(links, partition.evaluate, cfg.model.step_fraction, cfg.model.integrator, cancel,
                              cfg.model.tracing_tolerance)
        yc, zc = 0.5 * (y_edges[1:] + y_edges[:-1]), 0.5 * (z_edges[1:] + z_edges[:-1])
        Y, Z = np.meshgrid(yc, zc)
        sections: list[Section] = []
        expected = {codes[i.id]: rates[i.id] for i in pattern_inlets}

        def section_velocity(section_codes: np.ndarray, rate: float) -> np.ndarray:
            if not (contrast and coupling):
                return developed_velocity(y_edges, z_edges, W, H, rate, reference_mu)
            mu_field = np.vectorize(lambda c: code_to_mu.get(int(c), reference_mu))(section_codes).astype(float)
            return developed_velocity(y_edges, z_edges, W, H, rate, mu_field)

        report("transport", 0.7, "Evaluating the inlet section")
        with watch.time("transport"):
            n_streams = len(cfg.inlets)
            # Sub-cell supersampling for flux diagnostics; finer output grids need fewer sub-samples.
            samples = int(np.clip(round(640 / max(len(yc), len(zc))), 1, 4))
            inlet_codes = partition.evaluate(Y, Z).astype(np.int32)
            sections.append(Section(
                name="inlet pattern (x = 0)", x=0.0, y_edges=y_edges, z_edges=z_edges, codes=inlet_codes,
                axial_velocity=section_velocity(inlet_codes, pattern_rate),
                fractions=pattern_fractions(y_edges, z_edges, partition.evaluate, n_streams,
                                            samples).astype(np.float32),
                diagnostics={"expected_flux_m3_s": dict(expected)},
            ))
            flux_rate = pattern_rate
            stored_maps = []
            for g, domain in enumerate(domains):
                checkpoint()
                report("transport", 0.7 + 0.2 * g / max(len(domains), 1), f"Section after {domain.group.label}")
                for pid in domain.port_ids:
                    expected[codes[pid]] = rates[pid]
                flux_rate += sum(rates[pid] for pid in domain.port_ids)
                if cached_maps is None:
                    smap = adaptive_section(chain, g, yc, zc, stride=cfg.grid.map_stride)
                    chain.section_maps[g] = (smap, yc, zc)
                    stored_maps.append(smap)
                    x_end = links[g].x_end
                else:
                    smap = _reapply_map(cached_maps["maps"][g], partition.evaluate)
                    x_end = domain.group.x_end
                fractions = subcell_fractions(smap.code, smap.kind, smap.donor_y, smap.donor_z, y_edges, z_edges,
                                              partition.evaluate, n_streams, samples)
                sections.append(Section(
                    name=f"after {domain.group.label}", x=x_end, y_edges=y_edges, z_edges=z_edges,
                    codes=smap.code, status=smap.status,
                    axial_velocity=section_velocity(smap.code, flux_rate),
                    fractions=fractions.astype(np.float32),
                    donor_y=smap.donor_y, donor_z=smap.donor_z, residence_time=smap.residence,
                    diagnostics={**smap.diagnostics, "expected_flux_m3_s": dict(expected)},
                ))
            if not domains:
                sections.append(Section(
                    name="outlet (straight developed channel)", x=float(cfg.channel.length or 0.0),
                    y_edges=y_edges, z_edges=z_edges, codes=inlet_codes.copy(),
                    axial_velocity=sections[0].axial_velocity,
                    fractions=sections[0].fractions.copy(),
                    diagnostics={"expected_flux_m3_s": dict(expected), "note": "identity map: no features"},
                ))

        if cached_maps is None and reuse_key is not None:
            memory_cache().put(reuse_key, {"flow_info": flow_info, "maps": stored_maps,
                                           "viscosity": reference_mu, "total_rate": total_rate})

        conservative_info = None
        if cfg.model.transport == "conservative" and links:
            report("transport", 0.9, "Conservative flux remap")
            with watch.time("conservative_transport"):
                conservative_info = _conservative_transport(cfg, links, sections, y_edges, z_edges, rates, codes,
                                                            domains, partition.evaluate, cancel)

        # ---- checks
        report("checks", 0.95, "Checking flux and numerical quality")
        with watch.time("checks"):
            stream_codes = [s["code"] for s in streams if s["enabled"]]
            section_checks = []
            for section in sections:
                exp = section.diagnostics["expected_flux_m3_s"]
                total = sum(exp.values())
                achieved = _flux_fractions(section, stream_codes)
                errors_pp = {int(c): achieved.get(int(c), 0.0) - q / total for c, q in exp.items()}
                section.diagnostics["flux_fraction"] = {int(c): achieved.get(int(c), 0.0) for c in exp}
                section.diagnostics["flux_fraction_error"] = errors_pp
                section.diagnostics["unresolved_flux_fraction"] = achieved.get(-1, 0.0)
                section_checks.append({
                    "section": section.name,
                    "max_flux_fraction_error": max(abs(v) for v in errors_pp.values()),
                    "unresolved_flux_fraction": achieved.get(-1, 0.0),
                })
            final = sections[-1]
            max_error = max(c["max_flux_fraction_error"] for c in section_checks)
            invalid = float(np.mean(final.codes < 0))
            total_rate = sum(rates.values())
            densest = max(cfg.material(i.material).density for i in active)
            reynolds = physics.reynolds_number(total_rate, W, H, densest, mu_values[0])
            diffusivities = [cfg.material(i.material).diffusivity for i in active]
            peclet = (min(physics.peclet_number(total_rate, W, H, d) for d in diffusivities if d)
                      if any(diffusivities) else None)
            validity = assess(
                reynolds=reynolds,
                viscosity_ratio=mu_values[-1] / mu_values[0],
                coupling_used=coupling,
                coupling_converged=coupling_converged,
                aspect_ratio=H / W,
                resolution_warnings=[w for d in domains for w in d.warnings],
                max_flux_error=max_error,
                invalid_fraction=invalid,
            )
            if not validity["inside_verified_envelope"]:
                warnings.append(f"Validity: {validity['status']}")

        timings = dict(watch.timings)
        timings["total"] = time.perf_counter() - started
        report("done", 1.0, "Done")
        return SimulationResult(
            status="success",
            message=f"{len(sections)} sections; {validity['status']}",
            streams=streams,
            sections=sections,
            diagnostics={
                "issues": [vars(i) for i in issues],
                "notes": notes,
                "groups": flow_info,
                "pattern_partition": partition.describe(),
                "section_checks": section_checks,
                "max_flux_fraction_error": max_error,
                "reynolds_number": reynolds,
                "peclet_number": peclet,
                "viscosity_contrast": contrast,
                "viscosity_coupling": coupling,
                "reference_viscosity_Pa_s": reference_mu,
                "network": network_info,
                "conservative_transport": conservative_info,
                "traced_points": chain.traced_points,
                "map_lookups": chain.looked_up_points,
                "map_reused": cached_maps is not None,
                "cache": memory_cache().stats(),
            },
            timings=timings,
            validity=validity,
            warnings=warnings,
            provenance={
                "models": {"flow": validity["model"], "pattern": "pattern_partition"},
                "fidelity": cfg.model.fidelity,
                "integrator": cfg.model.integrator,
                "calibration": "none (no external reference data)",
            },
            **base,
        )
    except SolveCancelled:
        return failed_result("cancelled", "Cancelled by request.", timings=watch.timings, **base)
    except Exception as exc:  # noqa: BLE001 - reported as a failed result at the application boundary
        return failed_result(
            "failed", f"{type(exc).__name__}: {exc}", timings=watch.timings,
            diagnostics={"traceback": traceback.format_exc()}, **base,
        )


def _rescaled_flow_info(cached: dict, viscosity: float, total_rate: float) -> list[dict]:
    """Flow diagnostics of a reused map, rescaled to the current viscosity and total flow rate."""
    q = total_rate / cached["total_rate"]
    mu_q = q * viscosity / cached["viscosity"]
    out = []
    for info in cached["flow_info"]:
        info = dict(info)
        for key in ("pressure_drop_Pa",):
            if info.get(key) is not None:
                info[key] = info[key] * mu_q
        for key in ("flux_in_m3_s", "flux_out_m3_s", "upstream_rate_m3_s"):
            if info.get(key) is not None:
                info[key] = info[key] * q
        if info.get("port_rates_m3_s") is not None:
            info["port_rates_m3_s"] = [r * q for r in info["port_rates_m3_s"]]
        info["cache"] = "map reused (flow geometry unchanged)"
        out.append(info)
    return out


def _reapply_map(smap, pattern) -> "SectionMap":
    """Stored donor map with the (possibly changed) inlet pattern re-evaluated at its donor coordinates."""
    code = smap.code.copy()
    mask = smap.kind == KIND_PATTERN
    code[mask] = pattern(smap.donor_y[mask], smap.donor_z[mask])
    diagnostics = dict(smap.diagnostics, reused_map=True)
    return SectionMap(code, smap.kind, smap.status, smap.donor_y, smap.donor_z, smap.residence, diagnostics)


def _flux_fractions(section: Section, stream_codes: list[int]) -> dict[int, float]:
    weights = section.axial_velocity * section.cell_area()
    total = float(weights.sum())
    if section.fractions is not None:
        out = {int(c): float(np.sum(section.fractions[c - 1] * weights)) / total for c in stream_codes}
        out[-1] = float(np.sum((1.0 - section.fractions.sum(axis=0)) * weights)) / total
        return out
    out = {int(c): float(weights[section.codes == c].sum()) / total for c in stream_codes}
    out[-1] = float(weights[section.codes < 0].sum()) / total
    return out


def _pressure_driven_rates(cfg: ProjectConfig, domains, checkpoint) -> tuple[dict[str, float], dict]:
    from .models.stokes_map import solve_for_rates

    W, H = cfg.channel.width, cfg.channel.height
    if cfg.channel.length is None:
        raise ValueError("Pressure-driven operation needs the channel length (outlet position).")
    extra: dict[float, float] = {}
    for domain in domains:
        checkpoint()
        if not domain.group.pillars:
            continue
        unit, _ = solve_for_rates(domain, [1.0] + [0.0] * len(domain.openings), cfg.model.linear_solver,
                                  cfg.model.tolerance)
        xc = domain.grid.centers[0]
        straight = physics.duct_resistance(float(xc[-1] - xc[0]), W, H, 1.0)
        extra[float(domain.group.feature_start)] = max(unit.pressure_drop() - straight, 0.0)
    inlets = []
    for inlet in cfg.active_inlets():
        inlets.append({
            "id": inlet.id,
            "station": 0.0 if inlet.placement == "pattern" else inlet.station,
            "pressure": inlet.pressure,
            "resistance": inlet.tubing_resistance,
            "viscosity": cfg.material(inlet.material).viscosity,
        })
    solution = solve_network(W, H, cfg.channel.length, cfg.operating.outlet_pressure, inlets, extra)
    return solution.inlet_rates, {
        "inlet_rates_m3_s": solution.inlet_rates,
        "node_pressures_Pa": solution.node_pressures,
        "node_stations_m": solution.node_stations,
        "segment_rates_m3_s": solution.segment_rates,
        "pillar_extra_resistance_Pa_s_m3_per_Pa_s": extra,
        "iterations": solution.iterations,
        "converged": solution.converged,
        "assumption": "lumped duct resistances; junction losses neglected",
    }


def _conservative_transport(cfg, links, sections, y_edges, z_edges, rates, codes, domains, pattern_fn,
                            cancel) -> dict:
    """Sequential flux-conservative remap through each group; fills ``fractions`` on each section."""
    n_streams = len(cfg.inlets)
    yc, zc = 0.5 * (y_edges[1:] + y_edges[:-1]), 0.5 * (z_edges[1:] + z_edges[:-1])
    area = np.outer(np.diff(z_edges), np.diff(y_edges))
    # Initial fractions: supersample the analytic inlet pattern (area fractions within each cell).
    s = 4
    offsets = (np.arange(s) + 0.5) / s
    sy = (y_edges[:-1, None] + np.diff(y_edges)[:, None] * offsets).ravel()
    sz = (z_edges[:-1, None] + np.diff(z_edges)[:, None] * offsets).ravel()
    SY, SZ = np.meshgrid(sy, sz)
    pattern_codes = pattern_fn(SY, SZ)
    fractions = np.zeros((n_streams, len(zc), len(yc)))
    for code in np.unique(pattern_codes):
        fractions[code - 1] = (pattern_codes == code).reshape(len(zc), s, len(yc), s).mean(axis=(1, 3))
    sections[0].fractions = fractions.astype(np.float32)
    info = {"groups": []}
    for g, link in enumerate(links):
        gmap = group_map(link.flow, yc, zc, stride=cfg.grid.map_stride, step_fraction=cfg.model.step_fraction,
                         integrator=cfg.model.integrator, cancel=cancel, tolerance=cfg.model.tracing_tolerance)
        port_rates = [rates[pid] for pid in domains[g].port_ids]
        remap = ConservativeRemap(gmap, (y_edges, z_edges), sections[g].axial_velocity * area,
                                  sections[g + 1].axial_velocity * area, port_rates)
        port_streams = [codes[pid] - 1 for pid in domains[g].port_ids]
        fractions, remap_info = remap.apply(fractions, port_streams)
        sections[g + 1].fractions = fractions.astype(np.float32)
        info["groups"].append({**remap_info, "map": gmap.diagnostics})
    info["note"] = "Sequential conservative remap; section flux fractions use these fractions."
    return info
