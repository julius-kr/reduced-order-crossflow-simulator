"""Legacy CFD (the vendored 3D-hydrodynamic-flow-sculpting tool) as an isolated backend.

The translation lives in ``legacy_cfd.py``; the upstream code runs only in
``legacy_cfd_worker.py``, a script in its own interpreter that never imports this package.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .. import metrics, physics
from ..config import ProjectConfig
from ..legacy_cfd import GRID, UPSTREAM, UPSTREAM_FILES, build_plan, comsol_seconds, upstream_root
from ..materials import stream_table
from ..results import Section, SimulationResult, failed_result
from .base import WORKER_FLAG, Backend, frozen, worker_executable
from .client import LaunchSpec


def worker_script() -> Path:
    return Path(__file__).resolve().with_name("legacy_cfd_worker.py")


class LegacyCFDBackend(Backend):
    name = "legacy_cfd"
    label = "Legacy CFD"

    def launch_spec(self) -> LaunchSpec:
        args = ["--upstream", str(upstream_root()), "--expect-sha256", UPSTREAM_FILES["computation.py"],
                "--revision", UPSTREAM["revision"]]
        if frozen():
            return LaunchSpec([worker_executable(), WORKER_FLAG, "legacy_cfd", *args])
        return LaunchSpec([sys.executable, "-I", str(worker_script()), *args])

    def check(self, cfg: ProjectConfig) -> list[str]:
        return build_plan(cfg)["unsupported"]

    def effective_settings(self, cfg: ProjectConfig) -> dict[str, Any]:
        plan = build_plan(cfg)
        return {
            "model": f"COMSOL streamline-map library lookup ({UPSTREAM['name']} @ {UPSTREAM['revision'][:7]})",
            "grid": f"{GRID} x {GRID} nodes (upstream index maps, walls included)",
            "library_maps": [op["map_key"] for op in plan["operations"]] or "none needed (no pillars)",
            "library_file": plan["library"] or "not needed",
            "conventions": plan["conventions"],
            "ignored": plan["ignored"],
            "unsupported": plan["unsupported"],
        }

    def make_job(self, cfg: ProjectConfig, output: Path) -> dict[str, Any]:
        plan = build_plan(cfg)
        if plan["unsupported"]:
            raise ValueError("Not supported by legacy CFD: " + "; ".join(plan["unsupported"]))
        initial = dict(plan["initial"])
        path = output.with_name(output.stem + "_inlet.npy")  # large arrays never travel through the protocol
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, initial.pop("codes"), allow_pickle=False)
        initial["file"] = str(path)
        job_plan = {"initial": initial, "operations": plan["operations"], "stations": plan["stations"],
                    "library": plan["library"]}
        return {"plan": job_plan, "output": str(output)}

    def load_result(self, message: dict, cfg: ProjectConfig) -> SimulationResult:
        base = {"backend": "legacy_cfd", "config": cfg.to_dict(), "input_hash": cfg.input_hash()}
        if message.get("status") != "success":
            return failed_result(message.get("status", "failed"), message.get("message", "Legacy CFD run failed."),
                                 diagnostics={"traceback": message.get("traceback")}, **base)
        with np.load(message["output"], allow_pickle=False) as data:
            codes = data["codes"].astype(np.int32)
            meta = json.loads(str(data["meta"]))
        return normalize_output(cfg, codes, meta, message, base)


def normalize_output(cfg: ProjectConfig, codes: np.ndarray, meta: dict, message: dict, base: dict) -> SimulationResult:
    plan = build_plan(cfg)
    W, H = cfg.channel.width, cfg.channel.height
    y, z = np.linspace(0.0, W, GRID), np.linspace(0.0, H, GRID)
    rates = {i.id: float(i.flow_rate or 0.0) for i in cfg.active_inlets()}
    streams = stream_table(cfg, rates)
    total = sum(rates.values())
    # Comparison metric only: flux of the labels under the analytic duct profile (as for legacy).
    velocity = np.asarray(physics.rectangular_duct_profile(y, z, W, H, normalization="analytic")) * total / (W * H)
    expected = {int(s["code"]): float(s["flow_rate_m3_s"] or 0.0) for s in streams if s["enabled"]}
    requested = sorted(expected)
    sections = []
    for station, section_codes in zip(plan["stations"], codes):
        section = Section(name=station["name"], x=float(station["x"]), y_edges=y, z_edges=z, codes=section_codes,
                          axial_velocity=velocity, grid_kind="legacy_nodes",
                          diagnostics={"expected_flux_m3_s": expected, "library_operations": station["operations"]})
        section.diagnostics["flux_fraction"] = metrics.flux_fractions(section, requested)
        sections.append(section)
    final = sections[-1].codes
    produced = sorted(int(v) for v in np.unique(final) if v >= 0)
    unassigned = float(np.mean(final < 0))
    warnings = []
    if unassigned:
        warnings.append(f"{100 * unassigned:.2f}% of the final section has no stream (upstream map left it empty).")
    unchanged = [e["pillar"] for e in meta.get("log", []) if e.get("changed_fraction") == 0.0]
    if unchanged:
        warnings.append(f"Library map left the section unchanged for pillar(s) {', '.join(unchanged)}.")
    missing = [c for c in requested if c not in produced]
    if missing:
        warnings.append(f"Stream codes {missing} are absent from the final section.")
    return SimulationResult(
        status="success",
        message="legacy CFD: COMSOL streamline-map library (upstream tool, not validated here)",
        solver=meta.get("identity", {}),
        streams=streams,
        sections=sections,
        diagnostics={
            "requested_labels": requested,
            "produced_labels": produced,
            "unassigned_fraction": unassigned,
            "operation_log": meta.get("log", []),
            # the COMSOL simulations behind the maps applied (estimated; the solve time excludes them)
            "comsol_seconds": comsol_seconds(len(meta.get("log", []))),
            "library": meta.get("library"),
            "library_cache": meta.get("library_cache"),
            "conventions": plan["conventions"],
            "isolation_problems": meta.get("isolation_problems", []),
            "unavailable": ["convergence residuals", "flux conservation", "pressure drop", "mesh uncertainty"],
        },
        timings={**meta.get("timings", {}), "total": float(message.get("solve_seconds", 0.0))},
        validity={"status": "library lookup of precomputed COMSOL streamline maps; not validated here",
                  "model_card": "legacy_cfd", "externally_validated": False},
        warnings=warnings,
        provenance={"models": {"flow": "upstream index maps"}, "adapter": "legacy_cfd.py",
                    "upstream": dict(UPSTREAM)},
        **base,
    )
