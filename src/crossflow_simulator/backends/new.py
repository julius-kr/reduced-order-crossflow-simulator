"""The new solver as an isolated backend."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import crossflow_simulator

from ..config import ProjectConfig
from ..models.domain import FIDELITY_CELLS, flow_cells
from ..results import SimulationResult, failed_result
from ..simulator import output_grid
from .base import WORKER_FLAG, Backend, frozen, worker_executable
from .client import LaunchSpec

_BOOT = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from crossflow_simulator.backends.new_worker import main; raise SystemExit(main())"
)


def source_root() -> Path:
    """Directory containing the loaded ``crossflow_simulator`` package (``src/`` in a checkout)."""
    return Path(crossflow_simulator.__file__).resolve().parents[1]


class NewBackend(Backend):
    name = "new"
    label = "New solver"

    def launch_spec(self) -> LaunchSpec:
        if frozen():
            return LaunchSpec([worker_executable(), WORKER_FLAG, "new"])
        return LaunchSpec([sys.executable, "-I", "-c", _BOOT, str(source_root())])

    def make_job(self, cfg: ProjectConfig, output: Path) -> dict[str, Any]:
        return {"config": cfg.to_dict(), "output": str(output)}

    def effective_settings(self, cfg: ProjectConfig) -> dict[str, Any]:
        y_edges, z_edges = output_grid(cfg)
        ny, nz = flow_cells(cfg)
        return {
            "model": "3D Stokes flow map per feature group (see model card)",
            "fidelity": cfg.model.fidelity,
            "flow_cells_across_smaller_dimension": FIDELITY_CELLS[cfg.model.fidelity],
            "flow_cells_y_z": [ny, nz],
            "output_grid": f"{len(y_edges) - 1} x {len(z_edges) - 1} cell-centred, physical axes",
            "linear_solver": cfg.model.linear_solver,
            "solver_tolerance": cfg.model.tolerance,
            "tracing": f"{cfg.model.integrator}, tolerance {cfg.model.tracing_tolerance}",
            "viscosity_coupling": cfg.model.viscosity_coupling,
            "transport": cfg.model.transport,
            "drive": cfg.operating.drive,
        }

    def load_result(self, message: dict, cfg: ProjectConfig) -> SimulationResult:
        output = message.get("output")
        if output and Path(output).exists():
            return SimulationResult.load(output)
        return failed_result(message.get("status", "failed") if message.get("status") != "success" else "failed",
                             message.get("message", "Worker returned no result file."),
                             backend="new", config=cfg.to_dict(), input_hash=cfg.input_hash(),
                             diagnostics={"traceback": message.get("traceback")})
