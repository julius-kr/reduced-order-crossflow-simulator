"""Isolated worker process for the new solver.

Launched with ``python -I`` and an explicit source root, so it loads exactly
the new package that launched it (asserted in the hello message) and never the
legacy snapshot. Jobs arrive as JSON lines; results are written to NPZ files
and announced on the protocol stream. Cancellation is cooperative between and
inside numerical stages.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from .protocol import PROTOCOL, Channel, peak_memory


def main() -> int:
    channel = Channel()
    sys.stdout = sys.stderr
    import crossflow_simulator
    from crossflow_simulator import solver_identity
    from crossflow_simulator.config import ProjectConfig
    from crossflow_simulator.simulator import simulate

    identity = solver_identity()
    identity.update({"python": sys.version.split()[0], "executable": sys.executable})
    try:
        import numpy
        import scipy

        identity.update({"numpy": numpy.__version__, "scipy": scipy.__version__})
    except ImportError:  # pragma: no cover
        pass
    loaded = [name for name in sys.modules if name.startswith("crossflow_simulator")]
    foreign = [n for n in loaded if getattr(sys.modules[n], "__file__", None)
               and not Path(sys.modules[n].__file__).resolve().is_relative_to(
                   Path(crossflow_simulator.__file__).resolve().parent)]
    started = time.perf_counter()
    channel.send({"type": "hello", "protocol": PROTOCOL, "backend": "new", "identity": identity,
                  "pid": os.getpid(), "isolation_problems": foreign})

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
        if job.get("type") != "run":
            continue
        job_id = job.get("job_id")
        cancel = cancel_flags.setdefault(job_id, threading.Event())
        last = {"t": 0.0}

        def progress(stage: str, fraction: float, message: str, _job=job_id) -> None:
            now = time.perf_counter()
            if now - last["t"] >= 0.1 or fraction >= 1.0:  # rate-limited progress
                last["t"] = now
                channel.send({"type": "progress", "job_id": _job, "stage": stage, "fraction": fraction,
                              "message": message})

        t0 = time.perf_counter()
        try:
            if job.get("clear_cache"):
                from crossflow_simulator.cache import memory_cache

                memory_cache().clear()
            config = ProjectConfig.from_dict(job["config"])
            result = simulate(config, progress=progress, cancel=cancel.is_set)
            path = Path(job["output"])
            result.save(path)
            channel.send({"type": "result", "job_id": job_id, "status": result.status, "message": result.message,
                          "output": str(path), "solve_seconds": time.perf_counter() - t0,
                          "timings": result.timings, "memory": peak_memory(),
                          "worker_uptime": time.perf_counter() - started})
        except Exception as exc:  # noqa: BLE001 - reported as a failed run
            channel.send({"type": "result", "job_id": job_id, "status": "failed",
                          "message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                          "solve_seconds": time.perf_counter() - t0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
