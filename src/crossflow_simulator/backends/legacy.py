"""Adapter that runs the frozen legacy solver in an isolated worker.

Responsibilities, all on the new-solver side of the process boundary:

* locate and verify the snapshot (``legacy/manifest.json`` hashes);
* decide whether a physical case can be expressed with the original
  operators, and say why not when it cannot (never silently dropping fluid
  properties or simplifying geometry);
* assemble the original operator sequence explicitly (the archived
  ``simulator.py`` is empty) with documented adapter conventions;
* normalize the worker output into a :class:`SimulationResult` on physical
  axes while keeping the raw legacy labels and reporting label defects.

Adapter conventions (they are reported in the effective settings):

* channel coordinates are normalized by ``W`` and ``H`` as the legacy code expects;
* serial ports on one wall are expressed as legacy ``Inlet`` objects varying in
  ``x`` only, listed bottom-to-top, so the legacy inference chooses vertical stacking;
* the legacy diameter curvature uses ``sum(D_i) / W`` over all inlets with a diameter;
* each pillar is placed in its own legacy lubrication window of length ``W``
  centred on the footprint; the legacy "slope" is the longest oblique edge;
* the secondary-flow step count is chosen from the legacy grid (unless set
  explicitly) so that the fastest label moves at least one grid cell per step.
  The operator's own default (120 steps over pseudo-time 0.12) moves labels by
  at most 0.001 per step, which rounds back to the same node on any grid below
  about 500 nodes, so the pillar would never change the section.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .. import physics
from ..config import ProjectConfig
from ..materials import stream_table
from ..results import Section, SimulationResult, failed_result
from ..simulator import output_grid
from .base import WORKER_FLAG, Backend, bundle_root, frozen, worker_executable
from .client import LaunchSpec


class LegacySnapshotError(RuntimeError):
    pass


def legacy_root() -> Path:
    """The ``legacy/`` directory holding ``manifest.json`` and ``python/``."""
    override = os.environ.get("CROSSFLOW_LEGACY_ROOT")
    candidates = [Path(override)] if override else []
    candidates.append(bundle_root() / "legacy")
    for candidate in candidates:
        if (candidate / "manifest.json").exists() and (candidate / "python" / "src").is_dir():
            return candidate
    raise LegacySnapshotError(
        "Legacy snapshot not found. Set CROSSFLOW_LEGACY_ROOT to the directory containing manifest.json."
    )


def load_manifest(root: Path | None = None) -> dict:
    root = root or legacy_root()
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def verify_snapshot(root: Path | None = None) -> dict[str, Any]:
    """Compare every archived file with the manifest hashes; list unexpected files."""
    root = root or legacy_root()
    manifest = load_manifest(root)
    archive = root / "python"
    mismatched, missing = [], []
    for entry in manifest["files"]:
        path = archive / entry["path"]
        if not path.exists():
            missing.append(entry["path"])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            mismatched.append(entry["path"])
    listed = {entry["path"] for entry in manifest["files"]}
    extra = sorted(
        p.relative_to(archive).as_posix() for p in archive.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.relative_to(archive).as_posix() not in listed
    )
    return {
        "ok": not (mismatched or missing or extra),
        "mismatched": mismatched,
        "missing": missing,
        "unexpected": extra,
        "files": len(manifest["files"]),
        "source_revision": manifest["snapshot"]["source_revision"],
    }


def legacy_python() -> str:
    """Interpreter for the legacy worker (overridable for a dedicated legacy environment)."""
    return os.environ.get("CROSSFLOW_LEGACY_PYTHON", sys.executable)


def worker_script() -> Path:
    return Path(__file__).resolve().with_name("legacy_worker.py")


def run_legacy_tests(extra_args: list[str] | None = None, timeout: float = 600.0) -> subprocess.CompletedProcess:
    """Run the six original tests against the frozen package in an isolated interpreter."""
    root = legacy_root()
    src, tests, config = root / "python" / "src", root / "python" / "tests", root / "python" / "pyproject.toml"
    boot = (
        "import sys, pathlib; sys.dont_write_bytecode = True; src = pathlib.Path(sys.argv[1]).resolve(); "
        "sys.path.insert(0, str(src)); import crossflow_simulator as m; "
        "assert pathlib.Path(m.__file__).resolve().is_relative_to(src), m.__file__; "
        "import pytest; raise SystemExit(pytest.main(sys.argv[2:]))"
    )
    args = [str(tests), "-q", "-p", "no:cacheprovider", "-c", str(config), "--rootdir", str(root / "python")]
    return subprocess.run([legacy_python(), "-I", "-c", boot, str(src), *args, *(extra_args or [])],
                          capture_output=True, text=True, timeout=timeout)


class LegacyBackend(Backend):
    name = "legacy"
    label = "Legacy solver"

    def launch_spec(self) -> LaunchSpec:
        root = legacy_root()
        args = ["--legacy-src", str(root / "python" / "src"), "--manifest", str(root / "manifest.json")]
        if frozen():
            return LaunchSpec([worker_executable(), WORKER_FLAG, "legacy", *args])
        return LaunchSpec([legacy_python(), "-I", str(worker_script()), *args])

    # ------------------------------------------------------------------ capability and plan
    def check(self, cfg: ProjectConfig) -> list[str]:
        return build_plan(cfg)["unsupported"]

    def effective_settings(self, cfg: ProjectConfig) -> dict[str, Any]:
        plan = build_plan(cfg)
        return {"conventions": plan["conventions"], "operations": [o["op"] for o in plan["operations"]],
                "grid": plan["grid"], "unsupported": plan["unsupported"], "ignored": plan["ignored"]}

    def make_job(self, cfg: ProjectConfig, output: Path) -> dict[str, Any]:
        plan = build_plan(cfg)
        if plan["unsupported"]:
            raise ValueError("Not supported by legacy: " + "; ".join(plan["unsupported"]))
        return {"plan": {"operations": plan["operations"]}, "output": str(output)}

    def load_result(self, message: dict, cfg: ProjectConfig) -> SimulationResult:
        base = {"backend": "legacy", "config": cfg.to_dict(), "input_hash": cfg.input_hash()}
        if message.get("status") != "success":
            return failed_result(message.get("status", "failed"), message.get("message", "Legacy run failed."),
                                 diagnostics={"traceback": message.get("traceback")}, **base)
        with np.load(message["output"], allow_pickle=False) as data:
            labels = data["labels"].astype(np.int32)
            y_n, z_n = data["y"], data["z"]
            meta = json.loads(str(data["meta"]))
        return normalize_output(cfg, labels, y_n, z_n, meta, message, base)


def normalize_output(cfg: ProjectConfig, labels, y_n, z_n, meta: dict, message: dict, base: dict) -> SimulationResult:
    W, H = cfg.channel.width, cfg.channel.height
    y, z = y_n * W, z_n * H
    rates = {i.id: float(i.flow_rate or 0.0) for i in cfg.active_inlets()}
    streams = stream_table(cfg, rates)
    requested = sorted(int(s["code"]) for s in streams if s["enabled"])
    produced = sorted(int(v) for v in np.unique(labels))
    unmatched = [v for v in produced if v not in requested]
    missing = [v for v in requested if v not in produced]
    total = sum(rates.values())
    # Comparison metric only: flux of the legacy labels under the analytic duct profile.
    velocity = np.asarray(physics.rectangular_duct_profile(y, z, W, H, normalization="analytic")) * total / (W * H)
    expected = {int(s["code"]): float(s["flow_rate_m3_s"] or 0.0) for s in streams if s["enabled"]}
    section = Section(
        name="legacy final section", x=0.0, y_edges=y, z_edges=z, codes=labels, axial_velocity=velocity,
        grid_kind="legacy_nodes",
        diagnostics={"expected_flux_m3_s": expected,
                     "note": "Nodal wall-inclusive legacy grid; flux fractions use the new analytic duct profile "
                             "as a common comparison metric."},
    )
    sor = [m for m in meta.get("messages", []) if "pressure solver" in m.lower()]
    not_converged = [m for m in sor if "did not converge" in m]
    warnings = []
    if unmatched or missing:
        warnings.append(f"Legacy labels differ from the requested stream codes: produced {produced}, "
                        f"requested {requested} (unmatched {unmatched}, missing {missing}).")
    if not_converged:
        warnings.append("Legacy SOR pressure solver reported non-convergence (print-only in legacy).")
    frozen_ops = [f"{i}: {e['op']}" for i, e in enumerate(meta.get("log", []))
                  if e["op"].startswith("pillar_") and e.get("changed_fraction") == 0.0]
    if frozen_ops:
        warnings.append(f"Legacy pillar operation changed no labels ({', '.join(frozen_ops)}). With per-step motion "
                        "below half a grid cell the legacy nearest-node resampling freezes; lower "
                        "legacy.secondary_steps or set it to 0 (automatic).")
    return SimulationResult(
        status="success",
        message="legacy heuristic surrogate result (behavioural baseline, not validated)",
        solver=meta.get("identity", {}),
        streams=streams,
        sections=[section],
        diagnostics={
            "requested_labels": requested,
            "produced_labels": produced,
            "unmatched_labels": unmatched,
            "missing_labels": missing,
            "legacy_messages": meta.get("messages", []),
            "pressure_solver_messages": sor,
            "operation_log": meta.get("log", []),
            "isolation_problems": meta.get("isolation_problems", []),
            "unavailable": ["convergence residuals", "flux conservation", "pressure drop", "validity envelope"],
        },
        timings={**meta.get("timings", {}), "total": float(message.get("solve_seconds", 0.0))},
        validity={"status": "legacy heuristic surrogate; not validated", "externally_validated": False},
        warnings=warnings,
        provenance={"models": {"flow": "legacy operator sequence"}, "adapter": "backends/legacy.py"},
        **base,
    )


def secondary_step_count(settings, ny_nodes: int, nz_nodes: int) -> int:
    """Step count for the legacy secondary-flow operator.

    The legacy operator normalizes the fastest speed to 1 and resamples labels at
    the nearest node after every step, so a step shorter than half a cell moves
    nothing. With ``secondary_steps = 0`` the adapter takes the largest count whose
    step still spans one cell of the coarser grid direction.
    """
    if settings.secondary_steps:
        return int(settings.secondary_steps)
    cells = max(min(ny_nodes, nz_nodes) - 1, 1)
    return max(1, int(np.floor(settings.secondary_pseudo_time * cells)))


def build_plan(cfg: ProjectConfig) -> dict[str, Any]:
    """Express a physical case with the original legacy operators, or explain why it cannot be."""
    L = cfg.legacy
    W, H = cfg.channel.width, cfg.channel.height
    unsupported: list[str] = []
    ignored: list[str] = []
    conventions: list[str] = [
        "coordinates normalized by W and H; legacy nodal grid includes the walls",
        "stream codes passed as legacy labels; raw output labels are kept",
    ]
    codes = cfg.stream_codes()
    active = cfg.active_inlets()
    if cfg.operating.drive != "flow_rate":
        unsupported.append("pressure-driven operation (legacy has no hydraulic network)")
    viscosities = {cfg.material(i.material).viscosity for i in active}
    if len(viscosities) > 1:
        unsupported.append("viscosity contrast (legacy has no fluid properties; they would be silently dropped)")
    densities = {cfg.material(i.material).density for i in active}
    if len(densities) > 1:
        ignored.append("density differences (no effect in either creeping-flow model)")
    ny_nodes = L.grid_ny or cfg.grid.output_ny
    _, z_edges = output_grid(cfg)
    nz_nodes = L.grid_nz or (len(z_edges) - 1)
    ops: list[dict[str, Any]] = [{"op": "cross_section", "ny": int(ny_nodes), "nz": int(nz_nodes)}]

    pattern = cfg.pattern_inlets()
    ports = cfg.port_inlets()
    layout = cfg.pattern.layout
    rate = lambda inlet: float(inlet.flow_rate or 0.0)  # noqa: E731
    if ports:
        walls = {p.wall for p in ports}
        if layout != "single":
            unsupported.append("ports downstream of a multi-inlet pattern (legacy stacks one layout only)")
        elif walls - {"top", "bottom"}:
            unsupported.append("side-wall ports (legacy has no side-wall port model)")
        elif len(walls) > 1:
            unsupported.append("ports on both top and bottom walls (legacy stacks one direction only)")
        else:
            wall = walls.pop()
            first = pattern[0]
            sequence = [first, *ports] if wall == "top" else [*reversed(ports), first]
            span = max(max(p.station for p in ports), 1e-30)
            ops.append({"op": "stack_auto", "inlets": [
                {"x": (0.0 if i.placement == "pattern" else i.station / span), "y": 0.5, "z": 1.0,
                 "flow_rate": rate(i), "label": codes[i.id]} for i in sequence]})
            conventions.append(f"serial ports on the {wall} wall listed bottom-to-top for legacy vertical stacking")
            if L.serial_curvature:
                diameters = [i.diameter for i in [first, *ports] if i.diameter]
                if first.diameter is None:
                    conventions.append("first (pattern) inlet has no diameter; excluded from sum(D_i)")
                ops.append({"op": "diameter_curvature", "diameter_ratio": float(sum(diameters) / W),
                            "strength": L.curvature_strength, "exponent": L.curvature_exponent,
                            "gamma": L.curvature_gamma})
    elif layout == "single":
        ops.append({"op": "fill", "label": codes[pattern[0].id]})
    elif layout == "layers_y":
        ordered = sorted(pattern, key=lambda i: i.pattern_col)
        ops.append({"op": "stack_auto", "inlets": [
            {"x": 0.0, "y": (i.pattern_col + 0.5) / len(ordered), "z": 0.5, "flow_rate": rate(i),
             "label": codes[i.id]} for i in ordered]})
    elif layout == "layers_z":
        ordered = sorted(pattern, key=lambda i: i.pattern_row)
        labels = [codes[i.id] for i in ordered]
        ops.append({"op": "stack_poiseuille", "orientation": "vertical", "flow_rates": [rate(i) for i in ordered],
                    "start_label": min(labels)})
        conventions.append("vertical layers use PoiseuilleStackingOperator (legacy inference only handles x-varying "
                           "inlets); legacy assigns consecutive labels from the smallest code")
    elif layout == "grid":
        rows, cols = cfg.pattern.rows, cfg.pattern.cols
        ops.append({"op": "stack_auto", "inlets": [
            {"x": 0.0, "y": (i.pattern_col + 0.5) / cols, "z": (i.pattern_row + 0.5) / rows, "flow_rate": rate(i),
             "label": codes[i.id]} for i in pattern]})
        conventions.append("grid pattern passed as an inlet array; legacy array stacking ignores per-inlet rates")

    secondary_steps = secondary_step_count(L, int(ny_nodes), int(nz_nodes))
    for pillar in sorted(cfg.active_pillars(), key=lambda p: float(p.footprint()[:, 0].mean())):
        fp = pillar.footprint()
        ratio = pillar.height / H
        if not 0.0 < ratio < 1.0:
            unsupported.append(f"pillar {pillar.id!r}: full-height pillars (legacy needs 0 < h/H < 1)")
            continue
        if pillar.group is not None and any(p.group == pillar.group and p.id != pillar.id for p in cfg.active_pillars()):
            unsupported.append(f"pillar {pillar.id!r}: composite pillar groups")
            continue
        if L.pillar_model == "lubrication_divergence":
            if pillar.mount != "bottom":
                unsupported.append(f"pillar {pillar.id!r}: top mount (legacy lubrication-divergence model is "
                                   "bottom-mounted only)")
                continue
            centre = 0.5 * (fp[:, 0].min() + fp[:, 0].max())
            local = np.column_stack([(fp[:, 0] - centre + 0.5 * W) / W, fp[:, 1] / W])
            if local[:, 0].min() < 0 or local[:, 0].max() > 1:
                unsupported.append(f"pillar {pillar.id!r}: longer than the legacy lubrication window (W)")
                continue
            edges = [(local[i], local[(i + 1) % len(local)]) for i in range(len(local))]
            oblique = [e for e in edges if abs(e[1][0] - e[0][0]) > 1e-9 and abs(e[1][1] - e[0][1]) > 1e-9]
            if not oblique:
                unsupported.append(f"pillar {pillar.id!r}: no oblique edge for the legacy slope-driven model")
                continue
            start, end = max(oblique, key=lambda e: float(np.hypot(*(e[1] - e[0]))))
            ops.append({
                "op": "pillar_lubrication_divergence", "vertices": local.tolist(), "height_ratio": ratio,
                "nx": L.lubrication_nx, "ny": L.lubrication_ny, "slope_start": start.tolist(),
                "slope_end": end.tolist(), "center_z_factor": L.center_z_factor, "radius_y": L.secondary_radius_y,
                "radius_z": L.secondary_radius_z, "strength": L.secondary_strength, "n_steps": secondary_steps,
                "pseudo_time": L.secondary_pseudo_time,
            })
            conventions.append(f"pillar {pillar.id!r}: legacy window of length W centred at x = {centre:.4g} m; "
                               "slope = longest oblique edge")
        else:
            ops.append({"op": "pillar_partial_height", "parameters": {
                "center_y": float(fp[:, 1].mean() / W), "height_ratio": ratio,
                "width_ratio": float(np.ptp(fp[:, 1]) / W), "shape": "rectangular", "mount": pillar.mount}})
            conventions.append(f"pillar {pillar.id!r}: polygon reduced to its lateral extent (legacy surrogate)")
    if any(o["op"] == "pillar_lubrication_divergence" for o in ops):
        per_step = L.secondary_pseudo_time / max(secondary_steps, 1)
        source = "explicit" if L.secondary_steps else "chosen from the legacy grid"
        conventions.append(f"legacy secondary flow: {secondary_steps} steps ({source}), at most {per_step:.3g} of the "
                           f"unit section per step on a {int(ny_nodes)} x {int(nz_nodes)} node grid")
    if any(o["op"] in ("stack_auto", "stack_poiseuille") for o in ops):
        # Only computed when a stacking operator consumes the profiles (no inflated legacy timings).
        if L.velocity_profile == "rectangular_2d":
            ops.insert(1, {"op": "velocity_profiles", "width": W / H, "height": 1.0, "n_terms": L.n_terms})
            conventions.append(f"stacking profiles from legacy rectangular_channel_velocity_2d, {L.n_terms} terms")
        else:
            conventions.append("stacking profiles: legacy default 1D parabola")
    return {
        "operations": ops,
        "unsupported": unsupported,
        "ignored": ignored,
        "conventions": conventions,
        "grid": {"ny_nodes": int(ny_nodes), "nz_nodes": int(nz_nodes), "kind": "legacy_nodes"},
    }
