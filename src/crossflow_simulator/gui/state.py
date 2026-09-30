"""Project state: current configuration, validation, undo/redo, save/load, and autosave recovery."""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import ConfigError, ProjectConfig, validate_config
from .qt import QObject, QTimer, Signal


def autosave_path(path: Path | None) -> Path:
    from ..cache import default_cache_dir

    if path is not None:
        return path.with_name(path.stem + ".autosave.json")
    return default_cache_dir().parent / "untitled.autosave.json"


class ProjectState(QObject):
    """Single source of truth for the edited project; every edit goes through :meth:`apply`."""

    changed = Signal()  # configuration changed (any edit, undo, redo, load)
    saved = Signal(str)

    def __init__(self, config: ProjectConfig, path: Path | None = None, autosave_seconds: float = 20.0):
        super().__init__()
        self.config = config
        self.path = path
        self.dirty = False
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.autosave)
        self._timer.start(int(autosave_seconds * 1000))

    # ------------------------------------------------------------------ editing
    def snapshot(self) -> dict:
        return self.config.to_dict()

    def apply(self, mutate, description: str = "") -> list[str]:
        """Apply ``mutate(config_copy)``; keep it only if it parses. Returns validation messages."""
        before = self.snapshot()
        candidate = self.config.copy()
        mutate(candidate)
        try:
            candidate = ProjectConfig.from_dict(candidate.to_dict())
        except ConfigError as exc:
            return [str(i) for i in exc.issues]
        if candidate.to_dict() == before:
            return self.issues()
        self._undo.append(before)
        del self._undo[:-200]
        self._redo.clear()
        self.config = candidate
        self.dirty = True
        self.changed.emit()
        return self.issues()

    def issues(self) -> list[str]:
        return [str(i) for i in validate_config(self.config)]

    def errors(self) -> list:
        return [i for i in validate_config(self.config) if i.level == "error"]

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> None:
        if self._undo:
            self._redo.append(self.snapshot())
            self.config = ProjectConfig.from_dict(self._undo.pop())
            self.dirty = True
            self.changed.emit()

    def redo(self) -> None:
        if self._redo:
            self._undo.append(self.snapshot())
            self.config = ProjectConfig.from_dict(self._redo.pop())
            self.dirty = True
            self.changed.emit()

    # ------------------------------------------------------------------ files
    def load(self, path: str | Path) -> None:
        self.config = ProjectConfig.load(path)
        self.path = Path(path)
        self.dirty = False
        self._undo.clear()
        self._redo.clear()
        self.changed.emit()

    def replace(self, config: ProjectConfig, path: Path | None = None) -> None:
        self.config = config
        self.path = path
        self.dirty = path is None
        self._undo.clear()
        self._redo.clear()
        self.changed.emit()

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("No file name")
        self.config.save(target)
        self.path = target
        self.dirty = False
        try:
            autosave_path(target).unlink()
        except OSError:
            pass
        self.saved.emit(str(target))
        return target

    def autosave(self) -> None:
        if not self.dirty:
            return
        target = autosave_path(self.path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {"saved_unix": time.time(), "source": str(self.path) if self.path else None,
                       "config": self.config.to_dict()}
            target.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def recoverable(path: Path | None) -> dict | None:
        target = autosave_path(path)
        if not target.exists():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if path is not None and path.exists() and path.stat().st_mtime > payload.get("saved_unix", 0):
            return None
        return payload
