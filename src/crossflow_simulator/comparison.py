"""Legacy/new/CFD execution, comparable timings, aligned metrics, accuracy against CFD, and export.

Rules implemented here (plan section 11, extended by the CFD reference):

* both backends receive the same physical case (same input hash) through
  explicit adapters; each shows its effective settings, and cases the legacy
  adapter cannot represent are reported as *not supported by legacy* with the
  reason instead of being simplified;
* measured runs execute sequentially in already-started workers (no resource
  contention); the benchmark alternates the order across trials and reports
  trial counts, medians, and spread for cold, warm, and cached scopes;
* a speedup ``legacy_time / new_time`` is only reported for two successful
  runs with the same timing scope; cached lookups are never compared against
  fresh legacy solves;
* legacy labels are regridded onto the new cell-centred grid by nearest
  neighbour *for display and metrics only*; both native results are kept;
* accuracy is measured only against the CFD reference (``cfd.py``), on its
  grid and with its axial velocity, and is reported together with the CFD's
  own mesh uncertainty. A difference smaller than that uncertainty cannot be
  resolved by the reference.

Agreement with legacy is a behavioural comparison, not evidence of accuracy.
"""
from __future__ import annotations

import json
import statistics
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import metrics
from .backends import get_backend
from .calibration import TARGETS
from .backends.client import WorkerClient
from .config import ProjectConfig
from .materials import colors_for
from .results import SimulationResult, failed_result

UpdateFn = Callable[[str, dict], None]


@dataclass
class BackendRun:
    backend: str
    status: str
    message: str = ""
    job_id: int | None = None
    solve_seconds: float | None = None  # numerical work inside the worker
    end_to_end_seconds: float | None = None  # submit -> result file loaded, parent side
    startup_seconds: float | None = None  # worker start, reported separately
    cache_state: str = "uncached"
    memory: dict = field(default_factory=dict)
    effective_settings: dict = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)
    result: SimulationResult | None = None

    def summary(self) -> dict[str, Any]:
        data = {k: v for k, v in asdict(self).items() if k != "result"}
        if self.result is not None:
            data["stage_timings"] = self.result.timings
            data["warnings"] = self.result.warnings
            data["validity"] = self.result.validity.get("status")
        return data


@dataclass
class ComparisonResult:
    comparison_id: str
    input_hash: str
    config: dict
    legacy_snapshot: dict
    runs: dict[str, BackendRun]
    metrics: dict = field(default_factory=dict)
    speedup: dict = field(default_factory=dict)
    alignment: dict = field(default_factory=dict)
    benchmark: dict | None = None
    accuracy: dict = field(default_factory=dict)  # per backend: agreement with the CFD reference
    created: float = field(default_factory=time.time)

    def is_current(self, cfg: ProjectConfig) -> bool:
        return self.input_hash == cfg.input_hash() and all(
            r.result is None or r.result.input_hash in ("", self.input_hash) for r in self.runs.values()
        )

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "crossflow-comparison/1",
            "comparison_id": self.comparison_id,
            "input_hash": self.input_hash,
            "created_unix": self.created,
            "legacy_snapshot": self.legacy_snapshot,
            "runs": {k: r.summary() for k, r in self.runs.items()},
            "metrics": self.metrics,
            "speedup": self.speedup,
            "alignment": self.alignment,
            "benchmark": self.benchmark,
            "accuracy_vs_cfd": self.accuracy,
            "interpretation": "Agreement with legacy is behavioural, not physical validation. Accuracy is measured "
                              "against the CFD reference within its reported mesh uncertainty; the CFD reference is "
                              "not externally validated.",
        }


def _snapshot_info() -> dict:
    try:
        from .backends.legacy import load_manifest, verify_snapshot

        manifest = load_manifest()
        check = verify_snapshot()
        return {"source_revision": manifest["snapshot"]["source_revision"],
                "version": manifest["solver_identity"]["version"], "verified": check["ok"]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


class ComparisonService:
    """Owns persistent isolated workers for every backend (each started on demand)."""

    def __init__(self, workdir: str | Path | None = None, on_update: UpdateFn | None = None,
                 worker_env: dict[str, str] | None = None):
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="crossflow-compare-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.on_update = on_update
        self.backends = {name: get_backend(name) for name in ("legacy", "new", "cfd")}
        self.clients: dict[str, WorkerClient] = {}
        self.worker_env = worker_env or {}
        self._startup: dict[str, float] = {}

    def _notify(self, kind: str, data: dict) -> None:
        if self.on_update is not None:
            self.on_update(kind, data)

    def client(self, name: str) -> WorkerClient:
        client = self.clients.get(name)
        if client is None or not client.alive:
            backend = self.backends[name]
            spec = backend.launch_spec()
            spec.env.update(self.worker_env)
            client = WorkerClient(name, spec, on_message=lambda m: self._notify("message", m))
            client.start()
            self.clients[name] = client
            self._startup[name] = client.startup_seconds or 0.0
        return client

    def close(self) -> None:
        for client in self.clients.values():
            client.terminate()
        self.clients.clear()

    def restart(self, name: str) -> None:
        client = self.clients.pop(name, None)
        if client is not None:
            client.kill()

    # ------------------------------------------------------------------ single backend run
    def run_backend(self, name: str, cfg: ProjectConfig, comparison_id: str, timeout: float | None = None,
                    clear_cache: bool = False, recompute: bool = False) -> BackendRun:
        """Run one backend in its worker. ``recompute`` (CFD only) discards the stored CFD result first."""
        backend = self.backends[name]
        try:
            unsupported = backend.check(cfg)
            effective = backend.effective_settings(cfg)
        except Exception as exc:  # noqa: BLE001
            return BackendRun(name, "failed", f"Adapter error: {exc}")
        if unsupported:
            label = "legacy" if name == "legacy" else backend.label
            run = BackendRun(name, "unsupported", f"Not supported by {label}: " + "; ".join(unsupported),
                             effective_settings=effective, unsupported=unsupported)
            self._notify("run", {"backend": name, "comparison_id": comparison_id, "status": run.status, "run": run})
            return run
        started_before = name in self.clients and self.clients[name].alive
        try:
            client = self.client(name)
        except Exception as exc:  # noqa: BLE001
            return BackendRun(name, "failed", f"Worker failed to start: {exc}", effective_settings=effective)
        output = self.workdir / f"{comparison_id}_{name}.npz"
        self._notify("pending", {"backend": name, "comparison_id": comparison_id})
        job = backend.make_job(cfg, output)
        job["comparison_id"] = comparison_id
        job["clear_cache"] = clear_cache
        job["recompute"] = recompute and name == "cfd"
        t0 = time.perf_counter()
        job_id = client.submit(job)
        self._notify("submitted", {"backend": name, "job_id": job_id, "comparison_id": comparison_id})
        try:
            message = client.wait(job_id, timeout)
        except TimeoutError:
            client.cancel(job_id)
            client.kill()
            return BackendRun(name, "cancelled", f"Timed out after {timeout} s; worker restarted", job_id=job_id,
                              effective_settings=effective)
        except Exception as exc:  # noqa: BLE001
            return BackendRun(name, "failed", str(exc), job_id=job_id, effective_settings=effective)
        result = backend.load_result(message, cfg)
        elapsed = time.perf_counter() - t0
        if name == "cfd":
            cache_state = str(result.diagnostics.get("cache", "computed"))
        else:
            cache_state = "cleared" if clear_cache else ("n/a" if name == "legacy" else "warm-worker")
        run = BackendRun(
            backend=name, status=result.status, message=result.message, job_id=job_id,
            solve_seconds=message.get("solve_seconds"), end_to_end_seconds=elapsed,
            startup_seconds=None if started_before else self._startup.get(name),
            cache_state=cache_state, memory=message.get("memory", {}), effective_settings=effective, result=result,
        )
        self._notify("run", {"backend": name, "comparison_id": comparison_id, "status": run.status,
                             "job_id": job_id, "run": run})
        return run

    # ------------------------------------------------------------------ paired run
    def run_set(self, cfg: ProjectConfig, names: tuple[str, ...] = ("new", "legacy"),
                timeouts: dict[str, float | None] | None = None, clear_cache: bool = False,
                recompute_cfd: bool = False) -> ComparisonResult:
        """Run the named backends sequentially (in the given order) on the same case."""
        timeouts = {"new": 600.0, "legacy": 1800.0, "cfd": 7200.0, **(timeouts or {})}
        comparison_id = uuid.uuid4().hex[:12]
        runs: dict[str, BackendRun] = {}
        for name in names:
            runs[name] = self.run_backend(name, cfg, comparison_id, timeouts.get(name),
                                          clear_cache=clear_cache and name == "new",
                                          recompute=recompute_cfd and name == "cfd")
        comparison = build_comparison(cfg, runs, comparison_id)
        self._notify("comparison", {"comparison_id": comparison_id})
        return comparison

    def run_pair(self, cfg: ProjectConfig, timeout_new: float | None = 600.0,
                 timeout_legacy: float | None = 1800.0, clear_cache: bool = False,
                 order: tuple[str, ...] = ("new", "legacy")) -> ComparisonResult:
        return self.run_set(cfg, order, {"new": timeout_new, "legacy": timeout_legacy}, clear_cache)

    # ------------------------------------------------------------------ benchmark
    def benchmark(self, cfg: ProjectConfig, trials: int = 3, include_cold: bool = True,
                  timeout_legacy: float | None = 1800.0) -> ComparisonResult:
        """Repeatable protocol: sequential runs, alternating order, cold/warm/cached scopes."""
        comparison_id = uuid.uuid4().hex[:12]
        protocol = {
            "trials": trials,
            "execution": "sequential in isolated workers; order alternates between trials",
            "scopes": {
                "cold": "fresh worker: startup + first solve (reported separately)",
                "warm_uncached": "already-started worker, new-solver caches cleared before each run",
                "cached": "already-started worker, new-solver flow cache kept (not an algorithmic comparison)",
            },
            "timing": "solve_seconds measured inside the worker around the complete numerical pipeline",
        }
        samples: dict[str, dict[str, list[float]]] = {n: {"warm_uncached": [], "cached": []} for n in self.backends}
        cold: dict[str, dict] = {}
        unsupported = self.backends["legacy"].check(cfg)
        names = ["new"] if unsupported else ["new", "legacy"]
        runs: dict[str, BackendRun] = {}
        if include_cold:
            for name in names:
                self.restart(name)
                t0 = time.perf_counter()
                run = self.run_backend(name, cfg, comparison_id, timeout_legacy, clear_cache=True)
                cold[name] = {"startup_seconds": self._startup.get(name), "first_solve_seconds": run.solve_seconds,
                              "total_seconds": time.perf_counter() - t0, "status": run.status}
                runs[name] = run
        for trial in range(trials):
            order = names if trial % 2 == 0 else list(reversed(names))
            for name in order:
                run = self.run_backend(name, cfg, comparison_id, timeout_legacy, clear_cache=True)
                runs[name] = run
                if run.status == "success" and run.solve_seconds is not None:
                    samples[name]["warm_uncached"].append(run.solve_seconds)
        for trial in range(trials):
            run = self.run_backend("new", cfg, comparison_id, None, clear_cache=False)
            if run.status == "success" and run.solve_seconds is not None:
                samples["new"]["cached"].append(run.solve_seconds)
        if unsupported:
            runs["legacy"] = BackendRun("legacy", "unsupported", "Not supported by legacy: " + "; ".join(unsupported),
                                        unsupported=unsupported)
        comparison = ComparisonResult(comparison_id, cfg.input_hash(), cfg.to_dict(), _snapshot_info(), runs)
        comparison.metrics, comparison.alignment = compare_results(runs.get("legacy"), runs.get("new"))
        stats = {name: {scope: describe(values) for scope, values in scopes.items() if values}
                 for name, scopes in samples.items()}
        comparison.speedup = speedup(runs.get("legacy"), runs.get("new"), stats)
        comparison.benchmark = {"protocol": protocol, "cold": cold, "statistics": stats, "samples": samples}
        return comparison


def build_comparison(cfg: ProjectConfig, runs: dict[str, BackendRun], comparison_id: str | None = None,
                     ) -> ComparisonResult:
    """A comparison record with every metric that the available runs allow."""
    comparison = ComparisonResult(comparison_id or uuid.uuid4().hex[:12], cfg.input_hash(), cfg.to_dict(),
                                  _snapshot_info() if "legacy" in runs else {}, dict(runs))
    comparison.metrics, comparison.alignment = compare_results(runs.get("legacy"), runs.get("new"))
    comparison.speedup = speedup(runs.get("legacy"), runs.get("new"))
    reference = runs.get("cfd")
    if reference is not None and reference.result is not None and reference.result.ok:
        for name in ("new", "legacy"):
            run = runs.get(name)
            if run is not None and run.result is not None and run.result.ok and run.result.final is not None:
                comparison.accuracy[name] = accuracy_against(reference.result, run.result)
                comparison.accuracy[name]["speed"] = speed_against_cfd(run, reference)
    return comparison


def cfd_compute_seconds(run: BackendRun) -> float | None:
    """How long the CFD reference took to compute (also when this run loaded it from the disk cache)."""
    if run.result is not None:
        value = run.result.diagnostics.get("computed_seconds")
        if value:
            return float(value)
    return run.solve_seconds


def speed_against_cfd(run: BackendRun | SimulationResult, reference: BackendRun) -> dict[str, Any]:
    seconds = run.solve_seconds if isinstance(run, BackendRun) else run.timings.get("total")
    cfd_seconds = cfd_compute_seconds(reference)
    if not seconds or not cfd_seconds:
        return {"value": None, "reason": "timings unavailable"}
    return {"value": cfd_seconds / seconds, "seconds": seconds, "cfd_seconds": cfd_seconds,
            "note": "CFD compute time (measured when it was computed) / this solve time; different grids and models"}


def accuracy_against(reference: SimulationResult, result: SimulationResult, station: int | None = None,
                     ) -> dict[str, Any]:
    """Agreement of ``result`` with the CFD reference at one station (default: the final sections).

    Codes are compared on the CFD section grid (a result on another grid is regridded by nearest
    neighbour). The flux a stream gains or loses is weighted with the CFD's own axial velocity, and the
    interface target of the validation workflow is widened by the CFD's mesh uncertainty.
    """
    same_stations = len(result.sections) == len(reference.sections)
    index = -1 if station is None or not same_stations else station
    target = reference.sections[index]
    source = result.sections[index] if same_stations else result.final
    aligned = (source.grid_kind == "cell_centred" and source.codes.shape == target.codes.shape
               and np.allclose(source.y_edges, target.y_edges) and np.allclose(source.z_edges, target.z_edges))
    predicted = source.codes if aligned else metrics.regrid_nearest(source, target.y, target.z)
    spacing = (float(np.diff(target.z_edges).mean()), float(np.diff(target.y_edges).mean()))
    height = float(target.z_edges[-1] - target.z_edges[0])
    weights = np.clip(target.axial_velocity, 0.0, None) * target.cell_area()
    total = float(weights.sum())
    convergence = (reference.diagnostics.get("cfd") or {}).get("mesh_convergence") or {}
    uncertainty = 0.0
    if convergence:
        stations_info = convergence.get("stations") or []
        entry = stations_info[index] if stations_info and -len(stations_info) <= index < len(stations_info) else None
        uncertainty = float((entry or convergence).get("uncertainty_m", 0.0))
    codes = sorted(int(c) for c in np.unique(target.codes) if c >= 0)
    per_stream = {}
    for code in codes:
        a = metrics.centroid(predicted, code, target.y, target.z)
        b = metrics.centroid(target.codes, code, target.y, target.z)
        per_stream[code] = {
            "iou": metrics.iou(predicted, target.codes, code),
            "interface_shift_m": metrics.interface_shift(predicted, target.codes, code, spacing),
            "interface_distance_m": metrics.interface_distance(predicted, target.codes, code, spacing),
            "flux_misassigned": float((weights[predicted == code].sum() - weights[target.codes == code].sum()) / total),
            "centroid_displacement_m": float(np.hypot(a[0] - b[0], a[1] - b[1])),
        }
    shifts = [e["interface_shift_m"] for e in per_stream.values() if np.isfinite(e["interface_shift_m"])]
    ious = [e["iou"] for e in per_stream.values() if np.isfinite(e["iou"])]
    fluxes = [abs(e["flux_misassigned"]) for e in per_stream.values()]
    shift = max(shifts) if shifts else 0.0
    targets = {
        "interface_shift_m": TARGETS["interface_distance_fraction_of_height"] * height + uncertainty,
        "iou_min": TARGETS["iou_min"],
        "flux_misassigned_max": TARGETS["flux_fraction_error_max"],
    }
    passed = {
        "interface": shift <= targets["interface_shift_m"],
        "iou": (min(ious) if ious else 1.0) >= targets["iou_min"],
        "flux": (max(fluxes) if fluxes else 0.0) <= targets["flux_misassigned_max"],
    }
    return {
        "available": True,
        "station": target.name,
        "x_m": target.x,
        "per_stream": per_stream,
        "max_interface_shift_m": shift,
        "max_interface_shift_fraction_of_H": shift / height,
        "min_iou": min(ious) if ious else float("nan"),
        "max_flux_misassigned": max(fluxes) if fluxes else 0.0,
        "differing_cell_fraction": float(np.mean(predicted != target.codes)),
        "cfd_uncertainty_m": uncertainty,
        "within_cfd_uncertainty": bool(shift <= uncertainty) if uncertainty > 0 else shift == 0.0,
        "targets": targets,
        "passed": passed,
        "within_targets": all(passed.values()),
        "regridded": not aligned,
        "note": "Interface target 2% of H widened by the CFD mesh uncertainty (as for external references).",
    }


def describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    q = np.percentile(ordered, [25, 75]) if len(ordered) > 1 else [ordered[0], ordered[0]]
    return {"n": len(ordered), "median": statistics.median(ordered), "min": ordered[0], "max": ordered[-1],
            "iqr": float(q[1] - q[0])}


def speedup(legacy: BackendRun | None, new: BackendRun | None, stats: dict | None = None) -> dict[str, Any]:
    if legacy is None or new is None:
        return {"value": None, "reason": "both backends must run"}
    if legacy.status != "success" or new.status != "success":
        return {"value": None, "reason": f"no speedup for non-successful runs (legacy {legacy.status}, "
                                         f"new {new.status})"}
    if stats and stats.get("legacy", {}).get("warm_uncached") and stats.get("new", {}).get("warm_uncached"):
        l_med = stats["legacy"]["warm_uncached"]["median"]
        n_med = stats["new"]["warm_uncached"]["median"]
        return {"value": l_med / n_med, "scope": "median warm-worker uncached solve time",
                "legacy_seconds": l_med, "new_seconds": n_med,
                "note": "Different models and grids: this is an end-to-end runtime ratio for the same physical case, "
                        "not an equal-accuracy comparison."}
    if legacy.solve_seconds and new.solve_seconds:
        if new.cache_state not in ("cleared", "uncached"):
            return {"value": None, "reason": "new run may have used cached flow fields; not comparable to a fresh "
                                             "legacy solve (use the benchmark action)"}
        return {"value": legacy.solve_seconds / new.solve_seconds, "scope": "single warm-worker solve",
                "legacy_seconds": legacy.solve_seconds, "new_seconds": new.solve_seconds,
                "note": "Single trial; use the benchmark action for a distribution."}
    return {"value": None, "reason": "timings unavailable"}


def aligned_codes(legacy: SimulationResult, new: SimulationResult) -> tuple[np.ndarray, np.ndarray, dict]:
    """Legacy labels regridded (nearest) onto the new cell-centred grid; display/metrics only."""
    target = new.final
    source = legacy.final
    legacy_on_new = metrics.regrid_nearest(source, target.y, target.z)
    info = {
        "display_grid": f"new solver output grid {target.codes.shape[1]} x {target.codes.shape[0]} (cell-centred)",
        "operation": "nearest-neighbour regridding of legacy nodal labels (metrics and display only)",
        "legacy_native_grid": f"{source.codes.shape[1]} x {source.codes.shape[0]} nodes including walls",
        "planes": "final sections (developed flow after the last feature)",
    }
    return legacy_on_new, target.codes, info


def compare_results(legacy_run: BackendRun | None, new_run: BackendRun | None) -> tuple[dict, dict]:
    if not legacy_run or not new_run or not (legacy_run.result and legacy_run.result.ok and legacy_run.result.final
                                             and new_run.result and new_run.result.ok and new_run.result.final):
        return {"available": False, "reason": "both results must be successful"}, {}
    legacy, new = legacy_run.result, new_run.result
    a, b, alignment = aligned_codes(legacy, new)
    section = new.final
    spacing = (float(np.diff(section.z_edges).mean()), float(np.diff(section.y_edges).mean()))
    codes = [int(s["code"]) for s in new.streams if s["enabled"]]
    weights = section.axial_velocity * section.cell_area()
    total = float(weights.sum())
    per_stream = {}
    for code in codes:
        entry = {
            "iou": metrics.iou(a, b, code),
            "interface_distance_m": metrics.interface_distance(a, b, code, spacing),
            "flux_fraction_legacy": float(weights[a == code].sum()) / total,
            "flux_fraction_new": float(weights[b == code].sum()) / total,
        }
        cl = metrics.centroid(a, code, section.y, section.z)
        cn = metrics.centroid(b, code, section.y, section.z)
        entry["centroid_displacement_m"] = float(np.hypot(cl[0] - cn[0], cl[1] - cn[1]))
        per_stream[code] = entry
    return {
        "available": True,
        "per_stream": per_stream,
        "differing_cell_fraction": float(np.mean(a != b)),
        "legacy_unmatched_labels": legacy.diagnostics.get("unmatched_labels", []),
        "note": "Behavioural comparison only; neither result is a reference.",
    }, alignment


def export_comparison(comparison: ComparisonResult, directory: str | Path, length_unit: str = "um") -> Path:
    """Write every native result, aligned differences, figures, timings, protocol, configs, and versions."""
    from .plotter import sections_figure

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(comparison.config, indent=2) + "\n", encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps(_clean(comparison.summary()), indent=2) + "\n", encoding="utf-8")
    effective = {k: {"effective_settings": r.effective_settings, "unsupported": r.unsupported}
                 for k, r in comparison.runs.items()}
    (out / "effective_settings.json").write_text(json.dumps(effective, indent=2) + "\n", encoding="utf-8")
    for name, run in comparison.runs.items():
        if run.result is not None:
            run.result.save(out / f"{name}_result.npz")
    panels = []
    for name, title in (("legacy", "Legacy solver"), ("new", "New solver"), ("cfd", "CFD reference")):
        run = comparison.runs.get(name)
        if run is None:
            continue
        subtitle = ""
        if name == "cfd":
            seconds = cfd_compute_seconds(run)
            subtitle = f"computed in {seconds:.3g} s" if seconds else ""
        elif run.solve_seconds is not None:
            subtitle = f"solve {run.solve_seconds:.3g} s"
        accuracy = comparison.accuracy.get(name)
        if accuracy:
            subtitle += (f"; vs CFD: IoU {accuracy['min_iou']:.3f}, shift "
                         f"{accuracy['max_interface_shift_m'] * 1e6:.2g} um")
        result = run.result
        panels.append((title, result, colors_for(result.streams) if result is not None and result.streams else {},
                       subtitle))
    sections_figure(panels, length_unit).savefig(out / "side_by_side.png", dpi=150)
    legacy, new = comparison.runs.get("legacy"), comparison.runs.get("new")
    lr = legacy.result if legacy else None
    nr = new.result if new else None
    if lr is not None and nr is not None and lr.ok and nr.ok:
        a, b, _ = aligned_codes(lr, nr)
        np.savez_compressed(out / "differences.npz", legacy_on_new_grid=a, new=b, differs=(a != b),
                            y_m=nr.final.y, z_m=nr.final.z)
    names = ", ".join(f"{k}_result.npz" for k in comparison.runs)
    (out / "README.txt").write_text(
        "Solver comparison export\n"
        f"comparison_id: {comparison.comparison_id}\ninput_hash: {comparison.input_hash}\n\n"
        "config.json            shared physical input\n"
        "effective_settings.json each backend's effective settings and unsupported features\n"
        f"{names}  native results (crossflow-result/1)\n"
        "differences.npz        legacy labels regridded (nearest) onto the new grid, and the difference mask\n"
        "side_by_side.png       final sections on physical axes (legacy, new, CFD reference)\n"
        "comparison.json        statuses, timings, memory, metrics, speedup scope, accuracy against the CFD\n"
        "                       reference, benchmark protocol, versions\n\n"
        "Agreement between the solvers is a behavioural comparison, not physical validation. Accuracy is\n"
        "measured against the CFD reference within its reported mesh uncertainty.\n",
        encoding="utf-8",
    )
    return out


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def unsupported_result(cfg: ProjectConfig, reasons: list[str]) -> SimulationResult:
    return failed_result("unsupported", "Not supported by legacy: " + "; ".join(reasons), backend="legacy",
                         config=cfg.to_dict(), input_hash=cfg.input_hash())
