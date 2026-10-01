"""Entry point of the packaged application (both executables).

Dispatch happens *before* the new package is imported, because the legacy
worker must load the archived ``crossflow_simulator`` package, which has the
same name as the new one:

* ``--crossflow-worker legacy ...``  run the bundled legacy worker script
* ``--crossflow-worker legacy_cfd ...`` run the bundled legacy CFD worker script (upstream code)
* ``--crossflow-worker new``         run the new-solver worker
* ``<cli command> ...``              command-line interface (run, compare, ...)
* no arguments                       desktop application
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

CLI_COMMANDS = {"run", "compare", "validate", "sweep", "validate-references", "legacy-tests", "verify-snapshot", "gui"}


def _bundle() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def main() -> int:
    local = os.environ.get("LOCALAPPDATA") or str(Path.home())
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(local) / "crossflow_simulator" / "numba_cache"))
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "--crossflow-worker":
        if argv[1] in ("legacy", "legacy_cfd"):
            script = _bundle() / "workers" / f"{argv[1]}_worker.py"
            sys.argv = [str(script), *argv[2:]]
            runpy.run_path(str(script), run_name="__main__")
            return 0
        from crossflow_simulator.backends.new_worker import main as worker

        return worker()
    if argv and argv[0] in CLI_COMMANDS:
        from crossflow_simulator.cli import main as cli

        return cli(argv)
    from crossflow_simulator.gui.app import main as gui

    return gui(argv)


if __name__ == "__main__":
    raise SystemExit(main())
