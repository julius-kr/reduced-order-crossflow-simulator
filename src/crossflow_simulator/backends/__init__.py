"""Solver backends. Each runs in its own isolated worker process.

``new``     the new implementation in this package (``src/``)
``legacy``  the frozen original solver in ``legacy/python`` (never imported here)
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
    raise KeyError(f"Unknown backend {name!r}; expected 'new' or 'legacy'.")


__all__ = ["Backend", "get_backend"]
