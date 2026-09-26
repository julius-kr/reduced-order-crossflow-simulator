"""Isolated worker that executes the frozen legacy solver.

This file is executed as a *script* in its own interpreter (``python -I``),
never imported into the new package's process. It imports only the standard
library, NumPy, and the archived ``crossflow_simulator`` package from the
legacy snapshot, and it refuses to run if anything else was loaded under that
name or if an archived source file no longer matches the manifest.

The legacy ``simulator.py`` is empty, so the adapter in ``backends/legacy.py``
sends an explicit operation plan; this worker only instantiates the original
operator classes and functions with the given parameters. No numerical
operation is replaced or corrected here.

Protocol: one JSON object per line on stdin/stdout (see ``backends/protocol``
in the new package; kept independent here on purpose).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

PROTOCOL = "crossflow-worker/1"


def _peak_memory() -> dict:
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return {"peak_bytes": None, "method": "unavailable (GetProcessMemoryInfo failed)"}
            return {"peak_bytes": int(counters.PeakWorkingSetSize), "current_bytes": int(counters.WorkingSetSize),
                    "method": "process-lifetime peak working set (GetProcessMemoryInfo)"}
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1 if sys.platform == "darwin" else 1024
        return {"peak_bytes": int(peak * scale), "method": "process-lifetime max RSS (getrusage)"}
    except Exception:  # noqa: BLE001
        return {"peak_bytes": None, "method": "unavailable"}


class Channel:
    def __init__(self) -> None:
        self._out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def send(self, message: dict) -> None:
        with self._lock:
            self._out.write(json.dumps(message) + "\n")
            self._out.flush()


class _LegacyFinder:
    """Resolve ``crossflow_simulator`` and its submodules only from the archived snapshot.

    Installed first on ``sys.meta_path`` so no other finder (an editable
    install hook, or a packaged application's frozen importer, which also
    contains a package of the same name) can serve new-package modules.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "crossflow_simulator" and not fullname.startswith("crossflow_simulator."):
            return None
        parts = fullname.split(".")
        base = self.root.joinpath(*parts)
        if (base / "__init__.py").exists():
            return importlib.util.spec_from_file_location(fullname, base / "__init__.py",
                                                          submodule_search_locations=[str(base)])
        if base.with_suffix(".py").exists():
            return importlib.util.spec_from_file_location(fullname, base.with_suffix(".py"))
        raise ImportError(f"{fullname} is not part of the legacy snapshot")


def _load_legacy(legacy_src: Path, manifest_path: Path) -> tuple[object, dict]:
    sys.dont_write_bytecode = True
    for name in list(sys.modules):
        if name == "crossflow_simulator" or name.startswith("crossflow_simulator."):
            raise SystemExit("crossflow_simulator was imported before the legacy path was selected")
    sys.meta_path.insert(0, _LegacyFinder(legacy_src))
    sys.path.insert(0, str(legacy_src))
    import crossflow_simulator as legacy  # noqa: PLC0415

    loaded = Path(legacy.__file__).resolve()
    if not loaded.is_relative_to(legacy_src.resolve()):
        raise SystemExit(f"Legacy worker loaded crossflow_simulator from {loaded}, not {legacy_src}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_root = legacy_src.parent  # legacy/python
    mismatches = []
    for entry in manifest["files"]:
        path = archive_root / entry["path"]
        if not entry["path"].startswith("src/"):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if digest != entry["sha256"]:
            mismatches.append(entry["path"])
    if mismatches:
        raise SystemExit(f"Archived legacy source differs from the manifest: {mismatches}")
    identity = {
        "name": manifest["solver_identity"]["name"],
        "version": manifest["solver_identity"]["version"],
        "source_revision": manifest["snapshot"]["source_revision"],
        "package_path": str(loaded.parent),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    try:
        import numpy

        identity["numpy"] = numpy.__version__
    except ImportError:  # pragma: no cover
        identity["numpy"] = None
    return legacy, identity


def _check_isolation(legacy_src: Path) -> list[str]:
    """Every loaded crossflow_simulator module must come from the archived snapshot."""
    problems = []
    for name, module in list(sys.modules.items()):
        if name == "crossflow_simulator" or name.startswith("crossflow_simulator."):
            file = getattr(module, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(legacy_src.resolve()):
                problems.append(f"{name} -> {file}")
    return problems


def _execute(plan: dict, cancel: threading.Event, send, job_id) -> dict:
    import numpy as np

    from crossflow_simulator.cross_section import CrossSection
    from crossflow_simulator.inlets import Inlet
    from crossflow_simulator.operators import (
        AutoInletStackingOperator, InletDiameterDeformationOperator, PoiseuilleStackingOperator,
    )

    timings: dict[str, float] = {}
    log: list[dict] = []
    captured = io.StringIO()
    cs = None
    profiles = {"y": None, "z": None}
    ops = plan["operations"]
    for index, op in enumerate(ops):
        if cancel.is_set():
            raise KeyboardInterrupt("cancelled")
        kind = op["op"]
        send({"type": "progress", "job_id": job_id, "stage": kind, "fraction": index / max(len(ops), 1),
              "message": f"legacy operation {index + 1}/{len(ops)}: {kind}"})
        before = None if cs is None else np.array(cs.labels, copy=True)
        started = time.perf_counter()
        with contextlib.redirect_stdout(captured):
            if kind == "cross_section":
                cs = CrossSection(ny=op["ny"], nz=op["nz"])
            elif kind == "fill":
                cs.fill(op["label"])
            elif kind == "velocity_profiles":
                from crossflow_simulator.physics import effective_velocity_profile, rectangular_channel_velocity_2d

                u2d = rectangular_channel_velocity_2d(cs.y, cs.z, op["width"], op["height"], op["n_terms"])
                profiles["z"] = effective_velocity_profile(u2d, cs.y, integrate_axis=1)
                profiles["y"] = effective_velocity_profile(u2d, cs.z, integrate_axis=0)
            elif kind == "stack_auto":
                inlets = [Inlet(i["x"], i["y"], i["z"], i["flow_rate"], i["label"]) for i in op["inlets"]]
                cs = AutoInletStackingOperator(inlets, profiles["y"], profiles["z"]).apply(cs)
            elif kind == "stack_poiseuille":
                profile = profiles["z"] if op["orientation"] == "vertical" else profiles["y"]
                cs = PoiseuilleStackingOperator(op["flow_rates"], op["orientation"], op["start_label"],
                                                profile).apply(cs)
            elif kind == "diameter_curvature":
                cs = InletDiameterDeformationOperator(op["diameter_ratio"], op["strength"], op["exponent"],
                                                      op["gamma"]).apply(cs)
            elif kind == "pillar_lubrication_divergence":
                cs = _lubrication_divergence_pillar(cs, op)
            elif kind == "pillar_partial_height":
                from crossflow_simulator.pillars import PartialHeightPillarOperator

                cs = PartialHeightPillarOperator(**op["parameters"]).apply(cs)
            else:
                raise ValueError(f"Unknown legacy operation {kind!r}")
        elapsed = time.perf_counter() - started
        timings[f"{index:02d}_{kind}"] = elapsed
        entry = {"op": kind, "seconds": elapsed}
        if before is not None and before.shape == np.shape(cs.labels):
            # Measured outside the timed region: how much this operation changed the section.
            entry["changed_fraction"] = float(np.mean(np.asarray(cs.labels) != before))
        log.append(entry)
    labels = np.asarray(cs.labels)
    return {
        "labels": labels,
        "y": np.asarray(cs.y),
        "z": np.asarray(cs.z),
        "timings": timings,
        "log": log,
        "messages": captured.getvalue().splitlines(),
    }


def _lubrication_divergence_pillar(cs, op: dict):
    """The original lubrication -> divergence forcing -> localized secondary-flow sequence."""
    import numpy as np
    from crossflow_simulator.lubrication import (
        PolygonPillarFootprint, conductance_from_gap, gap_height_from_pillar, polygon_pillar_height_map,
        solve_lubrication_pressure, velocity_from_pressure,
    )
    from crossflow_simulator.secondary_flow import (
        DivergenceDrivenLocalizedSecondaryFlowOperator, local_vertical_forcing_from_lubrication,
        slope_center_from_endpoints,
    )

    x = np.linspace(0.0, 1.0, op["nx"])
    y = np.linspace(0.0, 1.0, op["ny"])
    footprint = PolygonPillarFootprint([tuple(v) for v in op["vertices"]], op["height_ratio"])
    height = polygon_pillar_height_map(x, y, footprint)
    conductance = conductance_from_gap(gap_height_from_pillar(height))
    pressure = solve_lubrication_pressure(conductance)
    velocity_x, velocity_y = velocity_from_pressure(pressure, conductance, x, y)
    slope_start, slope_end = tuple(op["slope_start"]), tuple(op["slope_end"])
    forcing = local_vertical_forcing_from_lubrication(x, y, velocity_x, velocity_y, slope_start, slope_end,
                                                      pillar_height=height)
    forcing_y = np.interp(cs.y, y, forcing)
    center_y, center_z = slope_center_from_endpoints(slope_start, slope_end, op["center_z_factor"], op["height_ratio"])
    operator = DivergenceDrivenLocalizedSecondaryFlowOperator(
        forcing_y=forcing_y, center_y=center_y, center_z=center_z, radius_y=op["radius_y"],
        radius_z=op["radius_z"], strength=op["strength"], n_steps=op["n_steps"], pseudo_time=op["pseudo_time"],
    )
    return operator.apply(cs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-src", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    channel = Channel()
    sys.stdout = sys.stderr  # stray prints must never corrupt the protocol stream
    try:
        _, identity = _load_legacy(args.legacy_src, args.manifest)
        import numpy as np
    except SystemExit as exc:
        channel.send({"type": "fatal", "protocol": PROTOCOL, "message": str(exc)})
        return 2
    started = time.perf_counter()
    channel.send({"type": "hello", "protocol": PROTOCOL, "backend": "legacy", "identity": identity,
                  "pid": os.getpid(), "isolation_problems": _check_isolation(args.legacy_src)})
    jobs: list[dict] = []
    cancel_flags: dict = {}
    available = threading.Condition()
    stop = threading.Event()

    def reader() -> None:
        for line in sys.stdin:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            with available:
                if message.get("type") == "cancel":
                    cancel_flags.setdefault(message.get("job_id"), threading.Event()).set()
                elif message.get("type") == "shutdown":
                    stop.set()
                else:
                    jobs.append(message)
                available.notify_all()
        stop.set()
        with available:
            available.notify_all()

    threading.Thread(target=reader, daemon=True).start()
    while True:
        with available:
            while not jobs and not stop.is_set():
                available.wait()
            if not jobs and stop.is_set():
                break
            job = jobs.pop(0)
        job_id = job.get("job_id")
        cancel = cancel_flags.setdefault(job_id, threading.Event())
        if job.get("type") != "run":
            continue
        t0 = time.perf_counter()
        try:
            output = _execute(job["plan"], cancel, channel.send, job_id)
            meta = {
                "identity": identity,
                "timings": output["timings"],
                "log": output["log"],
                "messages": output["messages"],
                "isolation_problems": _check_isolation(args.legacy_src),
            }
            path = Path(job["output"])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, labels=output["labels"], y=output["y"], z=output["z"],
                                meta=np.array(json.dumps(meta)))
            channel.send({"type": "result", "job_id": job_id, "status": "success", "output": str(path),
                          "solve_seconds": time.perf_counter() - t0, "timings": output["timings"],
                          "memory": _peak_memory(), "worker_uptime": time.perf_counter() - started})
        except KeyboardInterrupt:
            channel.send({"type": "result", "job_id": job_id, "status": "cancelled", "message": "Cancelled",
                          "solve_seconds": time.perf_counter() - t0})
        except Exception as exc:  # noqa: BLE001 - reported to the parent as a failed run
            channel.send({"type": "result", "job_id": job_id, "status": "failed",
                          "message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                          "solve_seconds": time.perf_counter() - t0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
