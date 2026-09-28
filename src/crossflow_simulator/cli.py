"""Command-line interface: headless runs, comparison, benchmarks, validation, and maintenance.

    crossflow run CONFIG [--backend new|legacy] [--inline] [--out result.npz] [--figure fig.png]
    crossflow compare CONFIG [--export DIR]
    crossflow bench CONFIG [--trials N]
    crossflow validate CONFIG
    crossflow sweep CONFIG --set PATH --values V1,V2,... [--csv out.csv]
    crossflow validate-references DIR [--out report.json]
    crossflow legacy-tests
    crossflow verify-snapshot
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


def _print(data) -> None:
    print(json.dumps(data, indent=2, default=str))


def cmd_run(args: argparse.Namespace) -> int:
    from .config import ProjectConfig

    cfg = ProjectConfig.load(args.config)
    if args.fidelity:
        cfg.model.fidelity = args.fidelity
    if args.inline and args.backend == "new":
        from .simulator import simulate

        def progress(stage, fraction, message):
            if not args.quiet:
                print(f"[{fraction:5.1%}] {stage}: {message}", file=sys.stderr)

        result = simulate(cfg, progress=progress)
    else:
        import tempfile

        from .backends import get_backend

        backend = get_backend(args.backend)
        reasons = backend.check(cfg)
        if reasons:
            print(f"Not supported by {args.backend}: " + "; ".join(reasons), file=sys.stderr)
            return 3
        client = backend.client(on_message=lambda m: (not args.quiet and m.get("type") == "progress"
                                                      and print(f"[{m['fraction']:5.1%}] {m['message']}",
                                                                file=sys.stderr)))
        client.start()
        output = Path(args.out) if args.out else Path(tempfile.mkdtemp()) / f"{args.backend}.npz"
        message = client.run(backend.make_job(cfg, output))
        client.terminate()
        result = backend.load_result(message, cfg)
    if args.out and result.sections:
        result.save(args.out)
    if args.figure and result.ok:
        from .materials import colors_for
        from .plotter import result_figure

        result_figure(result, colors_for(result.streams, cfg.display.color_by), cfg.display.length_unit).savefig(
            args.figure, dpi=150)
    summary = {
        "status": result.status,
        "message": result.message,
        "backend": result.backend,
        "timings_s": result.timings,
        "warnings": result.warnings,
        "validity": result.validity.get("status"),
    }
    if result.ok and result.final is not None:
        summary["final_flux_fractions"] = result.final.diagnostics.get("flux_fraction")
    _print(summary)
    return 0 if result.ok else 1


def cmd_compare(args: argparse.Namespace) -> int:
    from .comparison import ComparisonService, export_comparison
    from .config import ProjectConfig

    cfg = ProjectConfig.load(args.config)
    service = ComparisonService()
    try:
        comparison = service.run_pair(cfg, clear_cache=True)
    finally:
        service.close()
    summary = comparison.summary()
    _print({"comparison_id": summary["comparison_id"],
            "runs": {k: {x: v.get(x) for x in ("status", "message", "solve_seconds", "end_to_end_seconds",
                                               "startup_seconds", "cache_state", "unsupported")}
                     for k, v in summary["runs"].items()},
            "speedup": summary["speedup"], "metrics": summary["metrics"]})
    if args.export:
        export_comparison(comparison, args.export, cfg.display.length_unit)
        print(f"Exported to {args.export}", file=sys.stderr)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .comparison import ComparisonService
    from .config import ProjectConfig

    cfg = ProjectConfig.load(args.config)
    service = ComparisonService()
    try:
        comparison = service.benchmark(cfg, trials=args.trials)
    finally:
        service.close()
    summary = comparison.summary()
    _print({"comparison_id": summary["comparison_id"], "speedup": summary["speedup"],
            "statistics": summary["benchmark"]["statistics"], "cold": summary["benchmark"]["cold"]})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from .config import ProjectConfig, validate_config

    cfg = ProjectConfig.load(args.config)
    issues = validate_config(cfg)
    for issue in issues:
        print(issue)
    return 1 if any(i.level == "error" for i in issues) else 0


def _set_path(data: dict, path: str, value) -> None:
    keys = path.split(".")
    target = data
    for key in keys[:-1]:
        target = target[int(key)] if isinstance(target, list) else target[key]
    last = keys[-1]
    if isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value


def cmd_sweep(args: argparse.Namespace) -> int:
    from .config import ProjectConfig
    from .simulator import simulate

    base = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows = []
    for raw in args.values.split(","):
        data = json.loads(json.dumps(base))
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        _set_path(data, args.set, value)
        cfg = ProjectConfig.from_dict(data)
        started = time.perf_counter()
        result = simulate(cfg)
        row = {"value": raw, "status": result.status, "seconds": time.perf_counter() - started,
               "validity": result.validity.get("status")}
        if result.ok:
            for code, fraction in result.final.diagnostics.get("flux_fraction", {}).items():
                row[f"flux_fraction_{code}"] = fraction
            row["max_flux_error"] = result.diagnostics.get("max_flux_fraction_error")
        rows.append(row)
        print(json.dumps(row, default=str), file=sys.stderr)
    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    _print(rows)
    return 0


def cmd_references(args: argparse.Namespace) -> int:
    from .calibration import validation_report

    report = validation_report(args.directory, holdout_fraction=args.holdout)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _print({"summary": report["summary"], "envelope": report["envelope"],
            "cases": {k: v["passed"] for k, v in report["cases"].items()}})
    return 0 if all(v["passed"] for v in report["cases"].values()) else 1


def cmd_legacy_tests(args: argparse.Namespace) -> int:
    from .backends.legacy import run_legacy_tests

    completed = run_legacy_tests()
    print(completed.stdout)
    print(completed.stderr, file=sys.stderr)
    return completed.returncode


def cmd_verify_snapshot(args: argparse.Namespace) -> int:
    from .backends.legacy import legacy_root, verify_snapshot
    from .migration import migration_status

    report = {"snapshot": verify_snapshot(), "migration": migration_status(legacy_root().parent)}
    _print(report)
    ok = report["snapshot"]["ok"] and report["migration"]["ok"]
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crossflow", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one backend on a project file")
    run.add_argument("config")
    run.add_argument("--backend", choices=["new", "legacy"], default="new")
    run.add_argument("--inline", action="store_true", help="run the new solver in this process")
    run.add_argument("--fidelity", choices=["preview", "medium", "reference"])
    run.add_argument("--out")
    run.add_argument("--figure")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)
    compare = sub.add_parser("compare", help="run the new and the legacy solver on the same case")
    compare.add_argument("config")
    compare.add_argument("--export")
    compare.set_defaults(func=cmd_compare)
    bench = sub.add_parser("bench", help="repeatable legacy/new benchmark protocol")
    bench.add_argument("config")
    bench.add_argument("--trials", type=int, default=3)
    bench.set_defaults(func=cmd_bench)
    validate = sub.add_parser("validate", help="validate a project file")
    validate.add_argument("config")
    validate.set_defaults(func=cmd_validate)
    sweep = sub.add_parser("sweep", help="parameter sweep of the new solver")
    sweep.add_argument("config")
    sweep.add_argument("--set", required=True, help="dotted JSON path, e.g. inlets.1.flow_rate_m3_s")
    sweep.add_argument("--values", required=True, help="comma-separated JSON values")
    sweep.add_argument("--csv")
    sweep.set_defaults(func=cmd_sweep)
    refs = sub.add_parser("validate-references", help="evaluate the new solver against a reference directory")
    refs.add_argument("directory")
    refs.add_argument("--holdout", type=float, default=0.3)
    refs.add_argument("--out")
    refs.set_defaults(func=cmd_references)
    sub.add_parser("legacy-tests", help="run the original tests against the frozen snapshot").set_defaults(
        func=cmd_legacy_tests)
    sub.add_parser("verify-snapshot", help="check snapshot hashes and migration state").set_defaults(
        func=cmd_verify_snapshot)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
