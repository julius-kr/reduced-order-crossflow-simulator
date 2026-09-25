"""Profile one 3D Stokes solve (development aid, not a benchmark).

Builds a 300 x 150 um channel with an off-centre partial-height block and
prints where the time goes (assembly, factorization or MINRES, preconditioner).

Usage (repository root)::

    python benchmarks/profile_stokes.py [--cells 12] [--method auto|direct|minres] [--top 25]
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from crossflow_simulator.solvers.section import flux_matched_velocity  # noqa: E402
from crossflow_simulator.solvers.stokes import RectilinearGrid, StokesProblem  # noqa: E402

W, H, L = 300e-6, 150e-6, 900e-6
MU = 0.02
RATE = 4e-9 / 60.0  # 4 uL/min


def build(cells: int) -> tuple[RectilinearGrid, StokesProblem]:
    grid = RectilinearGrid(np.linspace(0.0, L, 6 * cells + 1), np.linspace(0.0, W, 2 * cells + 1),
                           np.linspace(0.0, H, cells + 1))
    xc, yc, zc = grid.centers
    solid = ((np.abs(xc - 0.5 * L) < 0.1 * L)[:, None, None] & (yc < 0.45 * W)[None, :, None]
             & (zc < 0.6 * H)[None, None, :])
    return grid, StokesProblem(grid, solid, MU)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=int, default=12, help="cells across the channel height")
    parser.add_argument("--method", default="auto", choices=["auto", "direct", "minres"])
    parser.add_argument("--top", type=int, default=25, help="number of profile rows to print")
    args = parser.parse_args()

    started = time.perf_counter()
    grid, problem = build(args.cells)
    assembly = time.perf_counter() - started
    profile = flux_matched_velocity(grid.yf, grid.zf, MU, RATE).velocity.T
    values = problem.boundary_values(inflow=profile, outflow=profile)
    profiler = cProfile.Profile()
    profiler.enable()
    solution = problem.solve(values, method=args.method)
    profiler.disable()
    info = solution.info
    print(f"grid {grid.shape}, {info['unknowns']} unknowns, {info['method']}: assembly {assembly:.2f} s, "
          f"solve {info['solve_seconds']:.2f} s, {info['iterations']} iterations, "
          f"residual {info['relative_residual']:.1e}")
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
