"""Solver backends. Each runs in its own isolated worker process.

``new``     the new implementation in this package (``src/``)
``legacy``  the frozen original solver in ``legacy/python`` (never imported here)
``cfd``     the full-channel CFD reference (``cfd.py``), in its own worker
"""
from __future__ import annotations

from .base import Backend


def get_backend(name: str) -> Backend:
    if name == "new":
        from .new import NewBackend

        return NewBackend()
    if name == "legacy":
        from .legacy import LegacyBackend

        return LegacyBackend()
    if name == "cfd":
        from .cfd import CFDBackend

        return CFDBackend()
    raise KeyError(f"Unknown backend {name!r}; expected 'new', 'legacy' or 'cfd'.")


__all__ = ["Backend", "get_backend"]
