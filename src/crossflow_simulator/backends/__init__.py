"""Solver backends. Each runs in its own isolated worker process.

``new``     the new implementation in this package (``src/``)
``legacy``  the frozen original solver in ``legacy/python`` (never imported here)
``cfd``     the full-channel CFD reference (``cfd.py``), in its own worker
``legacy_cfd`` the vendored 3D-hydrodynamic-flow-sculpting tool (``legacy_cfd.py``; upstream code runs
            only in its own worker script)
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
    if name == "legacy_cfd":
        from .legacy_cfd import LegacyCFDBackend

        return LegacyCFDBackend()
    raise KeyError(f"Unknown backend {name!r}; expected 'new', 'legacy', 'cfd' or 'legacy_cfd'.")


__all__ = ["Backend", "get_backend"]
