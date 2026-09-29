"""The CFD reference as an isolated backend.

It runs in its own worker process (the new-solver worker with ``"solver": "cfd"`` jobs), so a
long CFD solve never delays the interactive new-solver previews and can be cancelled by
restarting its worker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from .new import NewBackend


class CFDBackend(NewBackend):
    name = "cfd"
    label = "CFD reference"

    def make_job(self, cfg: ProjectConfig, output: Path) -> dict[str, Any]:
        return {"config": cfg.to_dict(), "output": str(output), "solver": "cfd"}

    def check(self, cfg: ProjectConfig) -> list[str]:
        from ..cfd import unsupported

        return unsupported(cfg)

    def effective_settings(self, cfg: ProjectConfig) -> dict[str, Any]:
        from ..cfd import X_SPACING, convergence_cells, estimate_seconds, outlet_position

        size = max(cfg.channel.width, cfg.channel.height)
        return {
            "model": "full-channel steady 3D Navier-Stokes, one domain (CFD reference; see model card)",
            "cells_across_smaller_dimension": cfg.cfd.cells,
            "convergence_grid_cells": convergence_cells(cfg) or "off",
            "inertia": "Picard iterations" if cfg.cfd.inertia else "off (creeping flow)",
            "domain_m": [-cfg.cfd.buffer * size, outlet_position(cfg) + cfg.cfd.buffer * size],
            "streamwise_spacing": f"fine cell at features, growing to {X_SPACING:g} cells elsewhere",
            "sections": "every output cell traced back individually",
            "estimated_seconds": round(estimate_seconds(cfg)),
        }
