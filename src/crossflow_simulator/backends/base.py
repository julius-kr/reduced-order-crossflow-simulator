"""Solver backend interface shared by the new solver and the legacy adapter."""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from ..config import ProjectConfig
from ..results import SimulationResult
from .client import LaunchSpec, WorkerClient

WORKER_FLAG = "--crossflow-worker"


def frozen() -> bool:
    """True inside a packaged (PyInstaller) application."""
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """Directory holding bundled assets (packaged app) or the repository root (source checkout)."""
    if frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[3]


def worker_executable() -> str:
    """Executable for isolated workers.

    In the packaged application the windowed GUI executable has no usable
    stdin/stdout, so workers run through the console executable shipped next to it.
    """
    if frozen():
        candidate = Path(sys.executable).with_name("crossflow.exe" if sys.platform == "win32" else "crossflow")
        if candidate.exists():
            return str(candidate)
    return sys.executable


class Backend(ABC):
    name: str = ""
    label: str = ""

    @abstractmethod
    def launch_spec(self) -> LaunchSpec: ...

    @abstractmethod
    def make_job(self, cfg: ProjectConfig, output: Path) -> dict[str, Any]: ...

    @abstractmethod
    def load_result(self, message: dict, cfg: ProjectConfig) -> SimulationResult: ...

    def check(self, cfg: ProjectConfig) -> list[str]:
        """Reasons this backend cannot represent the case faithfully (empty when supported)."""
        return []

    def effective_settings(self, cfg: ProjectConfig) -> dict[str, Any]:
        return {}

    def client(self, on_message: Callable[[dict], None] | None = None) -> WorkerClient:
        return WorkerClient(self.name, self.launch_spec(), on_message)
