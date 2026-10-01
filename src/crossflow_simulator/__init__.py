"""Reduced-order crossflow simulator (new implementation).

The frozen original solver lives in ``legacy/python/`` and is only ever
executed by an isolated worker process (see ``crossflow_simulator.backends``).
"""
from __future__ import annotations

__version__ = "0.2.0"

SOLVER_NAME = "crossflow-new"


def solver_identity() -> dict[str, str]:
    """Identity record embedded in every result produced by this package."""
    from pathlib import Path

    return {
        "name": SOLVER_NAME,
        "version": __version__,
        "package_path": str(Path(__file__).resolve().parent),
    }
