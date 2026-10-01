"""Record the frozen legacy solver's outputs and timings for the shared-capability examples.

These are *legacy behaviour* fixtures, not truth: they let tests detect any
change in how the archived code is executed (adapter plan, worker, or
environment). Accuracy is judged against analytical and external references,
never against these files.

Run from the repository root: ``python benchmarks/capture_legacy_baseline.py``.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from crossflow_simulator.backends.legacy import LegacyBackend, build_plan  # noqa: E402
from crossflow_simulator.config import ProjectConfig  # noqa: E402
from crossflow_simulator.geometry import stable_hash  # noqa: E402

TARGET = ROOT / "reference_data" / "legacy_baseline"


def portable(identity: dict) -> dict:
    """The worker identity with local paths made relative to the repository (file name only outside it)."""
    out = dict(identity)
    for key in ("package_path", "executable"):
        if key in out:
            path = Path(out[key])
            try:
                out[key] = path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                out[key] = path.name
    return out


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    index = json.loads((ROOT / "examples" / "index.json").read_text(encoding="utf-8"))["examples"]
    backend = LegacyBackend()
    client = backend.client()
    hello = client.start()
    records = {}
    workdir = Path(tempfile.mkdtemp())
    for entry in index:
        if not entry["shared_capability"]:
            continue
        name = Path(entry["file"]).stem
        cfg = ProjectConfig.load(ROOT / "examples" / entry["file"])
        plan = build_plan(cfg)
        started = time.perf_counter()
        message = client.run(backend.make_job(cfg, workdir / f"{name}.npz"))
        elapsed = time.perf_counter() - started
        if message["status"] != "success":
            print(f"{name}: {message['status']} {message.get('message')}", file=sys.stderr)
            continue
        with np.load(message["output"], allow_pickle=False) as data:
            labels = data["labels"]
            meta = json.loads(str(data["meta"]))
        np.savez_compressed(TARGET / f"{name}.npz", labels=labels)
        records[name] = {
            "labels_sha256": hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest(),
            "labels_shape": list(labels.shape),
            "labels_dtype": str(labels.dtype),
            "unique_labels": np.unique(labels).tolist(),
            "plan_hash": stable_hash(plan["operations"]),
            "input_hash": cfg.input_hash(),
            "solve_seconds": message["solve_seconds"],
            "end_to_end_seconds": elapsed,
            "stage_seconds": message["timings"],
            "legacy_messages": meta["messages"],
        }
        print(f"{name}: {labels.shape} labels {np.unique(labels).tolist()} in {message['solve_seconds']:.2f} s")
    client.terminate()
    manifest = {
        "schema": "crossflow-legacy-baseline/1",
        "note": "Legacy behaviour, not truth. Timings are single runs on the recording machine.",
        "recorded_with": {
            "legacy_identity": portable(hello["identity"]),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "cases": records,
    }
    (TARGET / "baseline.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
