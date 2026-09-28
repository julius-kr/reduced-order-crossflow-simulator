"""Time the legacy kernels inside an isolated interpreter that can only import the frozen snapshot.

Usage: ``python -I legacy_kernels.py <legacy_src> <output.npz>``. Prints a JSON timing record and saves
the kernel outputs so the parent process can check numerical equivalence of the new kernels.
"""
from __future__ import annotations

import contextlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

legacy_src = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
sys.dont_write_bytecode = True
sys.path.insert(0, str(legacy_src))

import numpy as np  # noqa: E402

import crossflow_simulator  # noqa: E402

assert Path(crossflow_simulator.__file__).resolve().is_relative_to(legacy_src)

from crossflow_simulator.lubrication import (  # noqa: E402
    PolygonPillarFootprint, polygon_pillar_height_map, solve_lubrication_pressure,
)
from crossflow_simulator.physics import rectangular_channel_velocity_2d  # noqa: E402


def timed(fn, repeats):
    times, result = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - started)
    return result, times


record = {"velocity": [], "pressure": []}
arrays = {}
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3
for ny, nz in ((160, 100), (300, 300)):
    y, z = np.linspace(0, 1, ny), np.linspace(0, 1, nz)
    u, times = timed(lambda: rectangular_channel_velocity_2d(y, z, 2.0, 1.0, 100), 1 if ny == 300 else repeats)
    arrays[f"velocity_{ny}x{nz}"] = u
    record["velocity"].append({"ny": ny, "nz": nz, "n_terms": 100, "seconds": times,
                               "median_seconds": statistics.median(times)})
for nx, ny in ((41, 31), (81, 61)):
    x, y = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    pillar = PolygonPillarFootprint([(0.3, 0.2), (0.65, 0.2), (0.65, 0.75)], 0.6)
    k = (1 - polygon_pillar_height_map(x, y, pillar)) ** 3
    with contextlib.redirect_stdout(io.StringIO()) as log:
        p, times = timed(lambda: solve_lubrication_pressure(k), 1 if nx == 81 else repeats)
    arrays[f"pressure_{nx}x{ny}"] = p
    arrays[f"conductance_{nx}x{ny}"] = k
    record["pressure"].append({"nx": nx, "ny": ny, "seconds": times, "median_seconds": statistics.median(times),
                               "messages": log.getvalue().splitlines()[:1]})
np.savez_compressed(output, **arrays)
print(json.dumps(record))
