"""Persistent worker processes with an asynchronous job interface.

A :class:`WorkerClient` owns one isolated worker process. Jobs get
monotonically increasing IDs; results and progress arrive on a reader thread
and are delivered through callbacks (the GUI forwards them to the Qt event
loop) or collected with :meth:`wait`. Cancellation is cooperative; a worker
blocked inside a non-interruptible numerical call can be terminated and is
restarted on the next submission.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class LaunchSpec:
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


class WorkerError(RuntimeError):
    pass


class WorkerClient:
    _ids = itertools.count(1)
    _ids_lock = threading.Lock()

    def __init__(self, name: str, spec: LaunchSpec, on_message: Callable[[dict], None] | None = None):
        self.name = name
        self.spec = spec
        self.on_message = on_message
        self.process: subprocess.Popen | None = None
        self.hello: dict | None = None
        self.startup_seconds: float | None = None
        self._hello_event = threading.Event()
        self._results: dict[int, dict] = {}
        self._events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._stderr: deque[str] = deque(maxlen=200)
        self._fatal: str | None = None

    # ------------------------------------------------------------------ lifecycle
    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, timeout: float = 120.0) -> dict:
        if self.alive and self.hello is not None:
            return self.hello
        self._hello_event.clear()
        self._fatal = None
        env = dict(os.environ)
        env.update(self.spec.env)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        started = time.perf_counter()
        self.process = subprocess.Popen(
            self.spec.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, env=env, cwd=self.spec.cwd, creationflags=flags,
        )
        threading.Thread(target=self._read_stdout, args=(self.process,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self.process,), daemon=True).start()
        if not self._hello_event.wait(timeout):
            self.terminate()
            raise WorkerError(f"{self.name} worker did not start within {timeout:.0f} s: {self.stderr_tail()}")
        if self._fatal:
            raise WorkerError(f"{self.name} worker refused to start: {self._fatal}")
        self.startup_seconds = time.perf_counter() - started
        return self.hello  # type: ignore[return-value]

    def terminate(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                    process.stdin.flush()
                    process.wait(timeout=2.0)
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    process.kill()
                    process.wait(timeout=5.0)
        finally:
            self.process = None
            self.hello = None
            with self._lock:
                for job_id, event in self._events.items():
                    if job_id not in self._results:
                        self._results[job_id] = {"type": "result", "job_id": job_id, "status": "cancelled",
                                                 "message": "Worker terminated"}
                        event.set()

    def kill(self) -> None:
        """Hard stop for a worker stuck in a non-interruptible computation."""
        process = self.process
        if process is not None and process.poll() is None:
            process.kill()
        self.terminate()

    # ------------------------------------------------------------------ jobs
    @classmethod
    def next_job_id(cls) -> int:
        with cls._ids_lock:
            return next(cls._ids)

    def submit(self, payload: dict[str, Any], job_id: int | None = None) -> int:
        if not self.alive:
            self.start()
        job_id = self.next_job_id() if job_id is None else job_id
        message = dict(payload, type="run", job_id=job_id)
        with self._lock:
            self._events[job_id] = threading.Event()
        try:
            self.process.stdin.write(json.dumps(message) + "\n")  # type: ignore[union-attr]
            self.process.stdin.flush()  # type: ignore[union-attr]
        except (OSError, ValueError) as exc:
            raise WorkerError(f"{self.name} worker is not accepting jobs: {exc}") from exc
        return job_id

    def cancel(self, job_id: int) -> None:
        if self.alive:
            try:
                self.process.stdin.write(json.dumps({"type": "cancel", "job_id": job_id}) + "\n")  # type: ignore
                self.process.stdin.flush()  # type: ignore[union-attr]
            except (OSError, ValueError):
                pass

    def wait(self, job_id: int, timeout: float | None = None) -> dict:
        event = self._events.get(job_id)
        if event is None:
            raise KeyError(job_id)
        deadline = None if timeout is None else time.perf_counter() + timeout
        while not event.wait(0.2):
            if not self.alive and job_id not in self._results:
                raise WorkerError(f"{self.name} worker exited unexpectedly: {self.stderr_tail()}")
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError(f"{self.name} job {job_id} exceeded {timeout} s")
        return self._results[job_id]

    def run(self, payload: dict[str, Any], timeout: float | None = None) -> dict:
        return self.wait(self.submit(payload), timeout)

    def stderr_tail(self, lines: int = 20) -> str:
        return "\n".join(list(self._stderr)[-lines:])

    # ------------------------------------------------------------------ readers
    def _read_stdout(self, process: subprocess.Popen) -> None:
        for line in process.stdout:  # type: ignore[union-attr]
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._stderr.append(f"[non-protocol output] {line.rstrip()}")
                continue
            kind = message.get("type")
            if kind == "hello":
                self.hello = message
                self._hello_event.set()
            elif kind == "fatal":
                self._fatal = message.get("message", "unknown error")
                self._hello_event.set()
            elif kind == "result":
                job_id = message.get("job_id")
                with self._lock:
                    self._results[job_id] = message
                    event = self._events.get(job_id)
                if event is not None:
                    event.set()
            if self.on_message is not None:
                try:
                    self.on_message(dict(message, worker=self.name))
                except Exception:  # noqa: BLE001 - never let a UI callback kill the reader
                    pass
        self._hello_event.set()

    def _read_stderr(self, process: subprocess.Popen) -> None:
        for line in process.stderr:  # type: ignore[union-attr]
            self._stderr.append(line.rstrip())
