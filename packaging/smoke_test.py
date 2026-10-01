"""Clean-machine smoke test of the packaged application (no source checkout or Python needed).

    python packaging/smoke_test.py dist/crossflow

Checks: both executables exist; the bundled snapshot and the vendored legacy CFD code verify; the
legacy worker runs the original operators; a shipped shared-capability preset runs through all four
isolated backends (new, legacy, legacy CFD, and a small CFD reference); the report exports with the
accuracy against the CFD; when the pillar map library is bundled, the legacy CFD runs the square library
example with it. Run it on a machine without the development environment for the release gate.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(exe: Path, *args: str, timeout: float = 900) -> subprocess.CompletedProcess:
    return subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=timeout)


def main() -> int:
    dist = Path(sys.argv[1] if len(sys.argv) > 1 else "dist/crossflow").resolve()
    cli = dist / ("crossflow.exe" if sys.platform == "win32" else "crossflow")
    gui = dist / ("crossflow-gui.exe" if sys.platform == "win32" else "crossflow-gui")
    failures = []
    for exe in (cli, gui):
        if not exe.exists():
            failures.append(f"missing {exe.name}")
    if failures:
        print("\n".join(failures))
        return 1
    snapshot = run(cli, "verify-snapshot")
    report = json.loads(snapshot.stdout) if snapshot.stdout.strip().startswith("{") else {}
    if not report.get("snapshot", {}).get("ok"):
        failures.append(f"bundled snapshot does not verify: {snapshot.stdout[-400:]} {snapshot.stderr[-400:]}")
    if not report.get("legacy_cfd_upstream", {}).get("ok"):
        failures.append(f"bundled legacy CFD code does not verify: {report.get('legacy_cfd_upstream')}")
    preset = next((p for p in (dist / "_internal" / "examples", dist / "examples") if p.exists()), None)
    if preset is None:
        failures.append("examples not bundled")
    else:
        out = Path(tempfile.mkdtemp()) / "report"
        compare = run(cli, "compare", str(preset / "parallel_unequal.json"), "--cfd", "--cfd-cells", "12",
                      "--legacy-cfd", "--export", str(out))
        if compare.returncode != 0 or not (out / "side_by_side.png").exists():
            failures.append(f"comparison failed: {compare.stdout[-600:]} {compare.stderr[-600:]}")
        else:
            summary = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
            statuses = {k: v["status"] for k, v in summary["runs"].items()}
            if statuses != {"legacy": "success", "new": "success", "cfd": "success", "legacy_cfd": "success"}:
                failures.append(f"unexpected statuses {statuses}")
            elif not summary["accuracy_vs_cfd"].get("new", {}).get("within_targets"):
                failures.append(f"new solver outside the CFD targets: {summary['accuracy_vs_cfd'].get('new')}")
        if report.get("legacy_cfd_pillar_library", "not installed") != "not installed":
            out = Path(tempfile.mkdtemp()) / "library"
            mapped = run(cli, "compare", str(preset / "square_library_pillars.json"), "--no-legacy", "--legacy-cfd",
                         "--export", str(out))
            summary = json.loads((out / "comparison.json").read_text(encoding="utf-8")) if mapped.returncode == 0 else {}
            run_info = summary.get("runs", {}).get("legacy_cfd", {})
            if run_info.get("status") != "success":
                failures.append(f"legacy CFD with the bundled pillar library failed: {run_info or mapped.stdout[-600:]} "
                                f"{mapped.stderr[-600:]}")
    print("\n".join(failures) if failures else "Packaged application smoke test passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
