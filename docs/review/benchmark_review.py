"""Benchmarks of the original solver, recorded before the rewrite; no solver code is changed.

Run from the repository root with an environment containing NumPy, SciPy,
and pytest: python -I docs/review/benchmark_review.py [--output PATH]
The sparse prototype deliberately reproduces the original discrete equations,
including their missing physical grid spacing; it is not the new solver's PDE.

Source selection: after the layout migration this harness measures the frozen
snapshot in legacy/python/src (originally the top-level python tree at d69ab5f).
benchmark_results.json is the historical record and is never overwritten;
reruns write to --output (default: benchmark_results_rerun.json).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import platform
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LEGACY_SRC = ROOT / "legacy" / "python" / "src"
sys.path.insert(0, str(LEGACY_SRC))

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from crossflow_simulator.cross_section import CrossSection
from crossflow_simulator.inlets import Inlet
from crossflow_simulator.lubrication import (
    PolygonPillarFootprint, polygon_pillar_height_map, solve_lubrication_pressure,
)
from crossflow_simulator.operators import AutoInletStackingOperator
from crossflow_simulator.physics import rectangular_channel_velocity_2d
from crossflow_simulator.secondary_flow import DivergenceDrivenLocalizedSecondaryFlowOperator
import crossflow_simulator

if not Path(crossflow_simulator.__file__).resolve().is_relative_to(LEGACY_SRC.resolve()):
    raise SystemExit(f"Refusing to benchmark: crossflow_simulator loaded from {crossflow_simulator.__file__}, not {LEGACY_SRC}")


def timed(fn, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, {"seconds": times, "median_seconds": statistics.median(times)}


def sparse_legacy_system(k):
    ny, nx = k.shape
    ids = np.arange(nx * ny).reshape(ny, nx)
    interior = ids[1:-1, 1:-1].ravel()
    center = k[1:-1, 1:-1]
    neighbors = (k[1:-1, 2:], k[1:-1, :-2], k[2:, 1:-1], k[:-2, 1:-1])
    coefficients = [(0.5 * (center + neighbor)).ravel() for neighbor in neighbors]
    rows = [interior]
    cols = [interior]
    values = [sum(coefficients)]
    for offset, coefficient in zip((1, -1, nx, -nx), coefficients):
        rows.append(interior)
        cols.append(interior + offset)
        values.append(-coefficient)
    for boundary, neighbor in ((ids[0, 1:-1], ids[1, 1:-1]), (ids[-1, 1:-1], ids[-2, 1:-1])):
        rows.extend((boundary, boundary))
        cols.extend((boundary, neighbor))
        values.extend((np.ones(boundary.size), -np.ones(boundary.size)))
    boundary = np.concatenate((ids[:, 0], ids[:, -1]))
    rows.append(boundary)
    cols.append(boundary)
    values.append(np.ones(boundary.size))
    matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(k.size, k.size)).tocsc()
    rhs = np.zeros(k.size)
    rhs[ids[:, 0]] = 1
    return matrix, rhs


def separable_velocity(y, z, width=2.0, height=1.0, n_terms=100):
    modes = np.arange(1, 2 * n_terms, 2, dtype=float)
    sy = np.sin(np.pi * y[:, None] * modes[None, :])
    sz = np.sin(np.pi * z[:, None] * modes[None, :])
    m, n = modes[:, None], modes[None, :]
    coeff = 16 / (np.pi**4 * m * n * ((m / width)**2 + (n / height)**2))
    velocity = sz @ coeff.T @ sy.T
    return velocity / np.trapezoid(np.trapezoid(velocity, y, axis=1), z)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "benchmark_results_rerun.json")
    output = parser.parse_args().output
    if output.resolve() == (Path(__file__).parent / "benchmark_results.json").resolve():
        raise SystemExit("benchmark_results.json is the historical record; choose another --output.")
    report = {"environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count()}, "pressure": [], "velocity": []}
    for nx, ny in ((41, 31), (81, 61)):
        x, y = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
        pillar = PolygonPillarFootprint([(0.3, 0.2), (0.65, 0.2), (0.65, 0.75)], 0.6)
        k = (1 - polygon_pillar_height_map(x, y, pillar))**3
        logs = io.StringIO()
        with contextlib.redirect_stdout(logs):
            old, old_time = timed(lambda: solve_lubrication_pressure(k))
        def sparse_run():
            a, b = sparse_legacy_system(k)
            return spsolve(a, b).reshape(k.shape)
        new, new_time = timed(sparse_run)
        a, b = sparse_legacy_system(k)
        entry = {"nx": nx, "ny": ny, "legacy": old_time, "sparse_including_assembly": new_time, "speedup": old_time["median_seconds"] / new_time["median_seconds"], "max_pressure_difference": float(np.max(np.abs(new-old))), "legacy_relative_residual": float(np.linalg.norm(a @ old.ravel()-b) / np.linalg.norm(b)), "sparse_relative_residual": float(np.linalg.norm(a @ new.ravel()-b) / np.linalg.norm(b)), "legacy_messages": logs.getvalue().splitlines()}
        report["pressure"].append(entry)
        print(json.dumps({"pressure": entry}), flush=True)
    for ny, nz, repeats in ((160, 100, 3), (300, 300, 1)):
        y, z = np.linspace(0, 1, ny), np.linspace(0, 1, nz)
        old, old_time = timed(lambda: rectangular_channel_velocity_2d(y, z, 2, 1, 100), repeats)
        new, new_time = timed(lambda: separable_velocity(y, z), repeats)
        entry = {"ny": ny, "nz": nz, "n_terms": 100, "legacy": old_time, "separable": new_time, "speedup": old_time["median_seconds"] / new_time["median_seconds"], "max_velocity_difference": float(np.max(np.abs(old-new)))}
        report["velocity"].append(entry)
        print(json.dumps({"velocity": entry}), flush=True)
    cs = CrossSection(ny=128, nz=128)
    cs.labels = 1 + (cs.y[None, :] > 0.5).astype(int) + 2 * (cs.z[:, None] > 0.5).astype(int)
    forcing = np.sin(2 * np.pi * cs.y)
    outputs = {}
    for steps in (20, 120, 240):
        result = DivergenceDrivenLocalizedSecondaryFlowOperator(forcing_y=forcing, n_steps=steps).apply(cs)
        outputs[str(steps)] = float(np.mean(result.labels != cs.labels))
    a = DivergenceDrivenLocalizedSecondaryFlowOperator(forcing_y=forcing, n_steps=20, strength=0.01).apply(cs)
    b = DivergenceDrivenLocalizedSecondaryFlowOperator(forcing_y=forcing, n_steps=20, strength=0.5).apply(cs)
    inlets = [Inlet(0, y, z, 1, label) for y, z, label in ((0.25, 0.25, 10), (0.75, 0.25, 20), (0.25, 0.75, 30), (0.75, 0.75, 40))]
    array_a = AutoInletStackingOperator(inlets).apply(cs)
    inlets[0].flow_rate = 9
    array_b = AutoInletStackingOperator(inlets).apply(cs)
    report["correctness_probes"] = {"changed_label_fraction_by_step_count": outputs, "strength_0_01_and_0_5_identical_with_normalization": bool(np.array_equal(a.labels, b.labels)), "array_flow_rate_change_1_to_9_identical": bool(np.array_equal(array_a.labels, array_b.labels)), "array_requested_labels": [10, 20, 30, 40], "array_actual_labels": array_a.unique_labels().tolist()}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"correctness_probes": report["correctness_probes"]}), flush=True)


if __name__ == "__main__":
    main()
