"""Bridges between isolated solver workers and the Qt event loop.

Workers report on background reader threads; everything reaches widgets
through queued Qt signals, so solver processes never touch the GUI. Every
submission carries a monotonically increasing job ID; results from
superseded jobs are discarded, so rapid edits cannot display stale output.
"""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

from ..backends import get_backend
from ..backends.client import WorkerClient
from ..comparison import ComparisonService
from ..config import ProjectConfig
from .qt import QObject, Signal


class SolverRunner(QObject):
    """Persistent new-solver worker with latest-job-wins semantics."""

    progress = Signal(int, float, str)  # job_id, fraction, message
    finished = Signal(int, object)  # job_id, SimulationResult
    failed = Signal(int, str)
    started = Signal(int)

    def __init__(self, workdir: Path | None = None):
        super().__init__()
        self.backend = get_backend("new")
        self.workdir = workdir or Path(tempfile.mkdtemp(prefix="crossflow-gui-"))
        self.client: WorkerClient | None = None
        self.latest_job: int | None = None
        self.running: dict[int, ProjectConfig] = {}
        self.history: dict[str, list[float]] = {"preview": [], "medium": [], "reference": []}
        self._submitted: dict[int, float] = {}
        self._lock = threading.Lock()

    def _ensure(self) -> WorkerClient:
        if self.client is None or not self.client.alive:
            self.client = self.backend.client(on_message=self._on_message)
            self.client.start()
        return self.client

    def submit(self, cfg: ProjectConfig) -> int:
        client = self._ensure()
        for job_id in list(self.running):  # cancel superseded work
            client.cancel(job_id)
        job_id = WorkerClient.next_job_id()
        with self._lock:
            self.latest_job = job_id
            self.running[job_id] = cfg.copy()
            self._submitted[job_id] = time.perf_counter()
        client.submit(self.backend.make_job(cfg, self.workdir / f"job{job_id}.npz"), job_id=job_id)
        self.started.emit(job_id)
        return job_id

    def cancel(self) -> None:
        if self.client is not None:
            for job_id in list(self.running):
                self.client.cancel(job_id)

    def restart(self) -> None:
        if self.client is not None:
            self.client.kill()
        self.client = None
        self.running.clear()

    def estimate(self, fidelity: str) -> float | None:
        samples = self.history.get(fidelity) or []
        return sorted(samples)[len(samples) // 2] if samples else None

    def shutdown(self) -> None:
        if self.client is not None:
            self.client.terminate()

    def _on_message(self, message: dict) -> None:
        job_id = message.get("job_id")
        kind = message.get("type")
        if kind == "progress" and job_id == self.latest_job:
            self.progress.emit(job_id, float(message.get("fraction", 0.0)), str(message.get("message", "")))
        elif kind == "result":
            with self._lock:
                cfg = self.running.pop(job_id, None)
                started = self._submitted.pop(job_id, None)
            if job_id != self.latest_job or cfg is None:
                return  # superseded: never shown
            try:
                result = self.backend.load_result(message, cfg)
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(job_id, str(exc))
                return
            if result.ok and started is not None:
                self.history.setdefault(cfg.model.fidelity, []).append(time.perf_counter() - started)
            self.finished.emit(job_id, result)


class ComparisonRunner(QObject):
    """Runs paired comparisons and benchmarks on a background thread with live per-backend updates."""

    update = Signal(str, dict)  # kind, data
    done = Signal(object)  # ComparisonResult
    error = Signal(str)

    def __init__(self, workdir: Path | None = None):
        super().__init__()
        self.service = ComparisonService(workdir, on_update=lambda kind, data: self.update.emit(kind, data))
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_pair(self, cfg: ProjectConfig, timeout_legacy: float = 1800.0) -> None:
        self._start(lambda: self.service.run_pair(cfg.copy(), clear_cache=True, timeout_legacy=timeout_legacy))

    def run_set(self, cfg: ProjectConfig, names: tuple[str, ...], recompute_cfd: bool = False) -> None:
        """Run the named backends in order (new first, then legacy and/or the CFD reference)."""
        self._start(lambda: self.service.run_set(cfg.copy(), tuple(names), clear_cache=True,
                                                 recompute_cfd=recompute_cfd))

    def benchmark(self, cfg: ProjectConfig, trials: int) -> None:
        self._start(lambda: self.service.benchmark(cfg.copy(), trials=trials))

    def cancel(self) -> None:
        """Hard-stop the workers (legacy and CFD numerics run long); they restart on demand."""
        for name in list(self.service.clients):
            self.service.restart(name)

    def shutdown(self) -> None:
        self.service.close()

    def _start(self, job) -> None:
        if self.busy:
            return

        def work() -> None:
            try:
                self.done.emit(job())
            except Exception as exc:  # noqa: BLE001
                self.error.emit(f"{type(exc).__name__}: {exc}")

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
