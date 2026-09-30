"""Model cards and machine-readable validity envelopes.

A model is *numerically verified* when its discretization passes analytical,
convergence, conservation, and symmetry tests (see ``tests/``). It is
*validated* only when compared against independent CFD or experiment within
a stated envelope. No external reference data were available when this
version was built, so every model reports ``externally_validated: False`` and
predictions are labelled accordingly. Interpolation inside a validated
envelope would be permitted; everything else is explicitly experimental.
"""
from __future__ import annotations

from typing import Any

MODEL_CARDS: dict[str, dict[str, Any]] = {
    "stokes_map": {
        "title": "3D Stokes flow map per feature group",
        "description": (
            "Staggered finite-volume Stokes solution of each interacting group of ports and partial-height "
            "pillars, with fully developed inflow/outflow, followed by continuous backward streamline tracing "
            "to donor coordinates. Equal-property flows use cached unit basis solutions and superposition."
        ),
        "assumptions": [
            "steady, incompressible, Newtonian, isothermal flow of miscible liquids",
            "creeping flow (inertia neglected), rigid walls, no diffusion",
            "fully developed flow at group inflow/outflow planes (margins set by model settings)",
            "circular ports with Poiseuille inflow normal to the wall",
            "pillars voxelized on the flow grid; heights aligned with grid faces",
        ],
        "verified_by": [
            "discrete x-invariance of straight-duct flow (matches the cross-section solver to round-off)",
            "second-order convergence to the analytical rectangular-duct profile",
            "divergence-free fields and exact flux balance",
            "Stokes reversibility: a pillar followed by its mirror image composes to the identity",
            "lateral and top-bottom reflection equivariance",
            "invariance of the map to common viscosity and flow-rate scaling",
        ],
        "externally_validated": False,
        "envelope": {
            "reynolds_max": 1.0,
            "viscosity_ratio": [1.0, 1.0],
            "aspect_ratio_H_over_W": [0.1, 10.0],
            "min_gap_cells": 3.0,
            "min_port_faces": 12,
        },
    },
    "stokes_map_viscosity_coupled": {
        "title": "3D Stokes flow map with viscosity coupling (experimental)",
        "description": (
            "As stokes_map, but the viscosity field follows the traced material origins and the Stokes "
            "solution is iterated to self-consistency; developed variable-viscosity profiles at the planes."
        ),
        "assumptions": ["as stokes_map", "sharp interfaces; viscosity piecewise constant per material"],
        "verified_by": [
            "variable-viscosity cross-section solver against the analytical two-layer solution",
            "reduces to stokes_map for equal viscosities",
        ],
        "externally_validated": False,
        "envelope": {"reynolds_max": 1.0, "viscosity_ratio": [0.1, 10.0], "status": "experimental"},
    },
    "cfd_reference": {
        "title": "CFD reference (full-channel 3D Navier-Stokes)",
        "description": (
            "One staggered finite-volume solve of the steady incompressible Navier-Stokes equations over the "
            "whole channel, from an inflow buffer upstream of x = 0 to an outflow buffer beyond the outlet, "
            "refined at every feature; inertia by Picard iterations, viscosity following the materials. Every "
            "output cell is traced back individually, and a second, coarser grid estimates the mesh uncertainty."
        ),
        "assumptions": [
            "steady, incompressible, Newtonian, isothermal flow of miscible liquids",
            "sharp interfaces (no diffusion); density enters only the inertia term (flux-weighted mean)",
            "inlet pattern developed at the end of the inflow buffer; circular ports with Poiseuille inflow",
            "pillars voxelized on the CFD grid (staircase surfaces converge at first order)",
        ],
        "verified_by": [
            "the verification of the shared Stokes discretization and tracer (see stokes_map)",
            "second-order convective operator on analytic fields",
            "exact developed flow in a straight channel, including inertia",
            "grid-convergence estimate reported with every result",
        ],
        "externally_validated": False,
        "envelope": {"reynolds_max": 50.0, "status": "numerical reference; shares code with the new solver"},
    },
    "legacy_cfd": {
        "title": "Legacy CFD (3D-hydrodynamic-flow-sculpting map library)",
        "description": (
            "The vendored upstream tool: sections are predicted by chaining index maps that its authors built "
            "from COMSOL streamlines of individual geometries on a 250 x 250 section grid. The adapter selects "
            "the library entry for each pillar and runs the unchanged upstream operators in their own worker."
        ),
        "assumptions": [
            "only geometries in the library: triangular pillars in a square channel, library parameter values",
            "each pillar acts independently (maps chained in streamwise order; no interaction between features)",
            "maps quantized to the nearest of 250 x 250 section nodes; one fluid, creeping flow",
            "library pillars top-mounted (measured from the published streamlines); bottom mounts and mirror "
            "images by exact symmetry",
        ],
        "verified_by": [
            "vendored code checked against the pinned upstream revision (SHA-256 of every file)",
            "adapter tests with synthetic libraries (identity, shifted, and reflected maps)",
        ],
        "externally_validated": False,
        "envelope": {"status": "upstream library coverage only; COMSOL settings of the libraries are not documented"},
    },
    "pattern_partition": {
        "title": "Flux-constrained inlet pattern",
        "description": "Patch boundaries of the prescribed inlet pattern carry each inlet's flow rate exactly "
                       "under the developed (optionally variable-viscosity) profile.",
        "externally_validated": False,
        "envelope": {},
    },
}


def assess(
    *,
    reynolds: float,
    viscosity_ratio: float,
    coupling_used: bool,
    coupling_converged: bool,
    aspect_ratio: float,
    resolution_warnings: list[str],
    max_flux_error: float,
    invalid_fraction: float,
    flux_tolerance: float = 0.005,
    invalid_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Summarize whether a run lies inside the verified envelope."""
    model = "stokes_map_viscosity_coupled" if coupling_used else "stokes_map"
    envelope = MODEL_CARDS[model]["envelope"]
    checks = []

    def check(name: str, ok: bool, detail: str, kind: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "kind": kind})

    check("creeping_flow", reynolds <= envelope["reynolds_max"],
          f"Re = {reynolds:.3g} (limit {envelope['reynolds_max']})", "envelope")
    lo, hi = envelope["viscosity_ratio"]
    check("viscosity_ratio", lo <= viscosity_ratio <= hi, f"max/min viscosity = {viscosity_ratio:.3g}", "envelope")
    if not coupling_used and viscosity_ratio > 1.0 + 1e-12:
        check("viscosity_coupling", False, "viscosity contrast present but coupling disabled: equal-property map",
              "envelope")
    a_lo, a_hi = MODEL_CARDS["stokes_map"]["envelope"]["aspect_ratio_H_over_W"]
    check("aspect_ratio", a_lo <= aspect_ratio <= a_hi, f"H/W = {aspect_ratio:.3g}", "envelope")
    if coupling_used:
        check("coupling_converged", coupling_converged, "viscosity coupling iterations converged", "numerics")
    check("resolution", not resolution_warnings, "; ".join(resolution_warnings) or "resolved", "numerics")
    check("flux_conservation", max_flux_error <= flux_tolerance,
          f"max per-stream flux-fraction error {100 * max_flux_error:.3f} pp (target {100 * flux_tolerance:.2f} pp)",
          "numerics")
    check("trajectories", invalid_fraction <= invalid_tolerance,
          f"{100 * invalid_fraction:.3f}% of cells without a valid trajectory "
          f"(target {100 * invalid_tolerance:.2f}%)", "numerics")
    envelope_ok = all(c["ok"] for c in checks if c["kind"] == "envelope")
    numerics_ok = all(c["ok"] for c in checks if c["kind"] == "numerics")
    if coupling_used:
        label = "experimental (viscosity-coupled, not externally validated)"
    elif not envelope_ok:
        label = "outside verified envelope (extrapolation, experimental)"
    elif not numerics_ok:
        label = "numerical quality below target (check it against the CFD reference)"
    else:
        label = "inside numerically verified envelope; not externally validated"
    return {
        "model": model,
        "model_card": MODEL_CARDS[model],
        "status": label,
        "inside_verified_envelope": envelope_ok and numerics_ok and not coupling_used,
        "envelope_ok": envelope_ok,
        "numerics_ok": numerics_ok,
        "externally_validated": False,
        "checks": checks,
    }
