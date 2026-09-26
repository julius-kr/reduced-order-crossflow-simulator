"""Worker protocol shared by the new-solver worker and the parent process.

Messages are JSON objects, one per line:

parent -> worker
    ``{"type": "run", "job_id": int, "comparison_id": str|None, "config"|"plan": ..., "output": path}``
    ``{"type": "cancel", "job_id": int}``, ``{"type": "shutdown"}``
worker -> parent
    ``{"type": "hello", "backend", "identity", "pid", "isolation_problems"}``
    ``{"type": "progress", "job_id", "stage", "fraction", "message"}`` (rate-limited)
    ``{"type": "result", "job_id", "status", "output", "solve_seconds", "timings", "memory"}``
    ``{"type": "fatal", "message"}``

Large arrays never travel through the protocol; results are NPZ files. The
legacy worker re-implements this small protocol so it has no dependency on
the new package.
"""
from __future__ import annotations

import json
import os
import sys
import threading

PROTOCOL = "crossflow-worker/1"


class Channel:
    """Line-buffered writer on a private duplicate of the original stdout."""

    def __init__(self) -> None:
        self._out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def send(self, message: dict) -> None:
        with self._lock:
            self._out.write(json.dumps(message) + "\n")
            self._out.flush()


def peak_memory() -> dict:
    """Process-lifetime peak memory of the calling process, with the measurement method."""
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
        return {"peak_bytes": int(peak * (1 if sys.platform == "darwin" else 1024)),
                "method": "process-lifetime max RSS (getrusage)"}
    except Exception:  # noqa: BLE001
        return {"peak_bytes": None, "method": "unavailable"}
