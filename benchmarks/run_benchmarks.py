"""Repeatable benchmarks of both backends (plan section 4 and work package B5).

Sections
    kernels     legacy kernels (isolated interpreter) versus new kernels on the *same* discrete
                problems (equivalence) and the corrected new formulations
    scenarios   new-solver timing scopes: fresh, cached map update, flow change, medium and reference fidelity
    end_to_end  paired legacy/new runs of the shared-capability examples in isolated workers
                (cold start, warm uncached, cached; sequential, alternating order)

Usage (repository root)::

    python benchmarks/run_benchmarks.py [--trials 3] [--quick] [--skip-end-to-end]

Writes ``benchmarks/results/benchmark_<timestamp>.json`` and prints a Markdown summary.
Kernel speedups do not multiply into an application speedup; see the end-to-end section.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CROSSFLOW_DISABLE_DISK_CACHE", "1")

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import spsolve  # noqa: E402

from crossflow_simulator import __version__, physics  # noqa: E402
from crossflow_simulator.backends.legacy import legacy_python, legacy_root  # noqa: E402
from crossflow_simulator.cache import memory_cache  # noqa: E402
from crossflow_simulator.comparison import ComparisonService  # noqa: E402
from crossflow_simulator.config import ProjectConfig  # noqa: E402
from crossflow_simulator.simulator import simulate  # noqa: E402
from crossflow_simulator.solvers.pressure import solve_lubrication  # noqa: E402


def environment() -> dict:
    info = {
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "crossflow_version": __version__,
    }
    try:
        import pyamg

        info["pyamg"] = pyamg.__version__
    except ImportError:
        info["pyamg"] = None
    return info


def timed(fn, repeats: int) -> tuple[object, dict]:
    times, result = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - started)
    return result, {"seconds": times, "median_seconds": statistics.median(times), "n": repeats}


def legacy_equivalent_system(k: np.ndarray):
    """The legacy SOR equations assembled sparsely (no physical spacing), for the equivalence check only."""
    ny, nx = k.shape
    ids = np.arange(nx * ny).reshape(ny, nx)
    interior = ids[1:-1, 1:-1].ravel()
    centre = k[1:-1, 1:-1]
    coefficients = [(0.5 * (centre + nb)).ravel() for nb in (k[1:-1, 2:], k[1:-1, :-2], k[2:, 1:-1], k[:-2, 1:-1])]
    rows, cols, vals = [interior], [interior], [sum(coefficients)]
    for offset, c in zip((1, -1, nx, -nx), coefficients):
        rows.append(interior)
        cols.append(interior + offset)
        vals.append(-c)
    for boundary, neighbour in ((ids[0, 1:-1], ids[1, 1:-1]), (ids[-1, 1:-1], ids[-2, 1:-1])):
        rows.extend((boundary, boundary))
        cols.extend((boundary, neighbour))
        vals.extend((np.ones(boundary.size), -np.ones(boundary.size)))
    edge = np.concatenate((ids[:, 0], ids[:, -1]))
    rows.append(edge)
    cols.append(edge)
    vals.append(np.ones(edge.size))
    matrix = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(k.size, k.size)).tocsc()
    rhs = np.zeros(k.size)
    rhs[ids[:, 0]] = 1.0
    return matrix, rhs


def kernels(repeats: int) -> dict:
    out_npz = Path(tempfile.mkdtemp()) / "legacy_kernels.npz"
    completed = subprocess.run(
        [legacy_python(), "-I", str(ROOT / "benchmarks" / "legacy_kernels.py"),
         str(legacy_root() / "python" / "src"), str(out_npz), str(repeats)],
        capture_output=True, text=True, timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    legacy = json.loads(completed.stdout.strip().splitlines()[-1])
    legacy_arrays = np.load(out_npz)
    report = {"velocity": [], "pressure": []}
    for entry in legacy["velocity"]:
        ny, nz = entry["ny"], entry["nz"]
        y, z = np.linspace(0, 1, ny), np.linspace(0, 1, nz)

        def same_sum():
            physics._profile_cache.clear()
            return physics.rectangular_duct_profile(y * 2.0, z, 2.0, 1.0, 100, "double_series", "trapezoid")

        u, t_same = timed(same_sum, repeats)
        _, t_single = timed(lambda: (physics._profile_cache.clear(),
                                     physics.rectangular_duct_profile(y * 2.0, z, 2.0, 1.0))[1], repeats)
        diff = float(np.max(np.abs(u - legacy_arrays[f"velocity_{ny}x{nz}"])))
        report["velocity"].append({
            "grid": f"{ny} x {nz}", "legacy": entry, "new_same_modal_sum": t_same, "new_single_series": t_single,
            "speedup_same_sum": entry["median_seconds"] / t_same["median_seconds"],
            "max_abs_difference_same_sum": diff,
        })
    for entry in legacy["pressure"]:
        nx, ny = entry["nx"], entry["ny"]
        k = legacy_arrays[f"conductance_{nx}x{ny}"]
        p_legacy = legacy_arrays[f"pressure_{nx}x{ny}"]

        def equivalent():
            a, b = legacy_equivalent_system(k)
            return spsolve(a, b).reshape(k.shape)

        p_new, t_equiv = timed(equivalent, repeats)
        # Corrected physics on the same geometry: a 1 x 1 mm region with a 100 um gap, cell-centred.
        gap = 1e-4 * np.cbrt(k[:-1, :-1])
        x_edges, y_edges = np.linspace(0, 1e-3, gap.shape[1] + 1), np.linspace(0, 1e-3, gap.shape[0] + 1)
        _, t_corrected = timed(lambda: solve_lubrication(gap, x_edges, y_edges, 1e-3, 1.0), repeats)
        report["pressure"].append({
            "grid": f"{nx} x {ny}", "legacy_sor": entry, "new_sparse_same_equations": t_equiv,
            "new_corrected_finite_volume": t_corrected,
            "speedup_same_equations": entry["median_seconds"] / t_equiv["median_seconds"],
            "max_abs_difference_same_equations": float(np.max(np.abs(p_new - p_legacy))),
            "note": "Legacy SOR stops at an update tolerance; the sparse solve is exact for the same equations.",
        })
    return report


def scenarios(repeats: int) -> dict:
    out = {}
    cfg = ProjectConfig.load(ROOT / "examples" / "right_triangle_bottom.json")

    def fresh():
        memory_cache().clear()
        return simulate(cfg)

    result, out["fresh_preview_single_pillar"] = timed(fresh, repeats)
    out["fresh_preview_single_pillar"]["stages"] = result.timings
    changed = cfg.copy()
    changed.inlets[0].flow_rate *= 0.5
    changed.inlets[1].flow_rate *= 1.5
    simulate(cfg)
    result, out["cached_map_update_160x80"] = timed(lambda: simulate(changed), repeats)
    assert result.diagnostics["map_reused"]
    big = cfg.copy()
    big.grid.output_ny, big.grid.output_nz = 300, 300
    simulate(big)
    big2 = big.copy()
    big2.inlets[0].flow_rate *= 0.5
    result, out["cached_map_update_300x300"] = timed(lambda: simulate(big2), repeats)
    serial = ProjectConfig.load(ROOT / "examples" / "serial_three_inlets.json")
    simulate(serial)
    factors = iter(1.1 + 0.2 * np.arange(repeats + 1))

    def port_change():
        # A new port flow ratio every repeat: flow fields come from the solved span (at most one
        # new solve per group), but the streamline maps must be traced again.
        changed = serial.copy()
        changed.inlets[1].flow_rate *= next(factors)
        return simulate(changed)

    result, out["port_flow_change_new_ratio_retrace"] = timed(port_change, repeats)
    assert not result.diagnostics["map_reused"]
    for fidelity in ("medium", "reference"):  # the app's Accuracy Medium and High
        finer = cfg.copy()
        finer.model.fidelity = fidelity

        def fresh_finer(finer=finer):
            memory_cache().clear()
            return simulate(finer)

        key = f"fresh_{fidelity}_single_pillar"
        result, out[key] = timed(fresh_finer, 1)
        out[key]["max_flux_fraction_error"] = result.diagnostics["max_flux_fraction_error"]
    out["targets"] = {"cached_300x300_update_s": 0.1, "interactive_with_map_s": 1.0, "fresh_reduced_model_s": 2.0}
    return out


def end_to_end(trials: int, names: list[str]) -> dict:
    service = ComparisonService(worker_env={"CROSSFLOW_DISABLE_DISK_CACHE": "1"})
    out = {}
    try:
        for name in names:
            cfg = ProjectConfig.load(ROOT / "examples" / f"{name}.json")
            comparison = service.benchmark(cfg, trials=trials)
            out[name] = {"benchmark": comparison.benchmark, "speedup": comparison.speedup,
                         "metrics": comparison.metrics}
            print(f"  {name}: {comparison.speedup}", file=sys.stderr)
    finally:
        service.close()
    return out


def markdown(report: dict) -> str:
    lines = ["# Benchmark summary", "", f"Environment: {json.dumps(report['environment'])}", ""]
    if "kernels" in report:
        lines += ["## Kernels (medians)", "", "| Kernel | Legacy | New (same problem) | Speedup | Max difference |",
                  "|---|---:|---:|---:|---:|"]
        for v in report["kernels"]["velocity"]:
            lines.append(f"| velocity {v['grid']} | {v['legacy']['median_seconds']:.4g} s | "
                         f"{v['new_same_modal_sum']['median_seconds'] * 1e3:.3g} ms | {v['speedup_same_sum']:.0f}x | "
                         f"{v['max_abs_difference_same_sum']:.2e} |")
        for p in report["kernels"]["pressure"]:
            lines.append(f"| pressure {p['grid']} | {p['legacy_sor']['median_seconds']:.4g} s | "
                         f"{p['new_sparse_same_equations']['median_seconds'] * 1e3:.3g} ms | "
                         f"{p['speedup_same_equations']:.0f}x | {p['max_abs_difference_same_equations']:.2e} |")
    if "scenarios" in report:
        lines += ["", "## New-solver scopes (medians)", ""]
        for key, value in report["scenarios"].items():
            if isinstance(value, dict) and "median_seconds" in value:
                lines.append(f"- {key}: {value['median_seconds'] * 1e3:.1f} ms (n={value['n']})")
    if "end_to_end" in report:
        lines += ["", "## End to end (isolated workers)", "", "| Case | Legacy warm | New warm uncached | New cached | "
                  "Speedup (warm) |", "|---|---:|---:|---:|---:|"]
        for name, entry in report["end_to_end"].items():
            stats = entry["benchmark"]["statistics"]
            legacy = stats.get("legacy", {}).get("warm_uncached", {}).get("median")
            new = stats.get("new", {}).get("warm_uncached", {}).get("median")
            cached = stats.get("new", {}).get("cached", {}).get("median")
            fmt = lambda v: "—" if v is None else f"{v:.3g} s"  # noqa: E731
            speed = entry["speedup"].get("value")
            lines.append(f"| {name} | {fmt(legacy)} | {fmt(new)} | {fmt(cached)} | "
                         f"{'—' if speed is None else f'{speed:.1f}x'} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="fewer repeats and cases")
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument("--skip-kernels", action="store_true")
    args = parser.parse_args()
    repeats = 1 if args.quick else 3
    report = {"schema": "crossflow-benchmark/1", "created_unix": time.time(), "environment": environment()}
    if not args.skip_kernels:
        print("kernels...", file=sys.stderr)
        report["kernels"] = kernels(repeats)
    print("scenarios...", file=sys.stderr)
    report["scenarios"] = scenarios(repeats)
    if not args.skip_end_to_end:
        print("end to end...", file=sys.stderr)
        names = ["parallel_unequal", "right_triangle_bottom"] if args.quick else [
            "parallel_unequal", "array_2x2_unequal", "serial_three_inlets", "right_triangle_bottom",
            "two_nearby_pillars"]
        report["end_to_end"] = end_to_end(args.trials, names)
    target = ROOT / "benchmarks" / "results" / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    summary = markdown(report)
    target.with_suffix(".md").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"\nWrote {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
