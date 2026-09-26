"""Command-line interface: headless runs, validation, and parameter sweeps.

    crossflow run CONFIG [--out result.npz] [--figure fig.png]
    crossflow validate CONFIG
    crossflow sweep CONFIG --set PATH --values V1,V2,... [--csv out.csv]
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
    from .simulator import simulate

    cfg = ProjectConfig.load(args.config)
    if args.fidelity:
        cfg.model.fidelity = args.fidelity

    def progress(stage, fraction, message):
        if not args.quiet:
            print(f"[{fraction:5.1%}] {stage}: {message}", file=sys.stderr)

    result = simulate(cfg, progress=progress)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crossflow", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the new solver on a project file")
    run.add_argument("config")
    run.add_argument("--fidelity", choices=["preview", "medium", "reference"])
    run.add_argument("--out")
    run.add_argument("--figure")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)
    validate = sub.add_parser("validate", help="validate a project file")
    validate.add_argument("config")
    validate.set_defaults(func=cmd_validate)
    sweep = sub.add_parser("sweep", help="parameter sweep of the new solver")
    sweep.add_argument("config")
    sweep.add_argument("--set", required=True, help="dotted JSON path, e.g. inlets.1.flow_rate_m3_s")
    sweep.add_argument("--values", required=True, help="comma-separated JSON values")
    sweep.add_argument("--csv")
    sweep.set_defaults(func=cmd_sweep)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
