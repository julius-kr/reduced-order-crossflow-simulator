"""Isolated workers, backend identity, snapshot integrity, baseline reproduction, and paired comparison."""
import hashlib
import json
import shutil

import numpy as np
import pytest

from crossflow_simulator.backends import get_backend
from crossflow_simulator.backends.client import WorkerError
from crossflow_simulator.backends.legacy import build_plan, legacy_root, run_legacy_tests, verify_snapshot
from crossflow_simulator.comparison import ComparisonService, export_comparison
from crossflow_simulator.config import OperatingConditions, PillarSpec, ProjectConfig
from crossflow_simulator.migration import find_references

from conftest import ROOT, UM, make_config

pytestmark = pytest.mark.legacy


# ----------------------------------------------------------------------------- snapshot and migration

def test_snapshot_matches_manifest():
    report = verify_snapshot()
    assert report["ok"], report


def test_original_tests_pass_against_the_frozen_snapshot():
    completed = run_legacy_tests()
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "6 passed" in completed.stdout


def test_no_executable_reference_to_the_removed_python_tree():
    assert find_references(ROOT) == []


def test_new_package_never_loads_the_legacy_snapshot():
    import sys

    import crossflow_simulator

    assert "legacy" not in crossflow_simulator.__file__.replace("\\", "/").split("/src/")[0].split("/")[-1:]
    legacy_src = str(legacy_root() / "python" / "src")
    assert legacy_src not in sys.path


# ----------------------------------------------------------------------------- workers

@pytest.fixture(scope="module")
def service(tmp_path_factory):
    service = ComparisonService(tmp_path_factory.mktemp("compare"))
    yield service
    service.close()


def test_worker_identities_and_isolation(service):
    legacy = service.client("legacy").hello
    new = service.client("new").hello
    assert legacy["identity"]["name"] == "legacy" and not legacy["isolation_problems"]
    assert new["identity"]["name"] == "crossflow-new" and not new["isolation_problems"]
    assert "legacy" in legacy["identity"]["package_path"].replace("\\", "/")
    assert new["identity"]["package_path"].replace("\\", "/").endswith("src/crossflow_simulator")
    assert legacy["pid"] != new["pid"]


def test_tampered_snapshot_is_refused(tmp_path, monkeypatch):
    copy = tmp_path / "legacy"
    shutil.copytree(legacy_root(), copy, ignore=shutil.ignore_patterns("__pycache__"))
    target = copy / "python" / "src" / "crossflow_simulator" / "operators.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# modified\n", encoding="utf-8")
    monkeypatch.setenv("CROSSFLOW_LEGACY_ROOT", str(copy))
    client = get_backend("legacy").client()
    with pytest.raises(WorkerError, match="differs from the manifest"):
        client.start()
    client.terminate()


def test_legacy_reproduces_the_recorded_baseline(service):
    baseline = json.loads((ROOT / "reference_data" / "legacy_baseline" / "baseline.json").read_text(encoding="utf-8"))
    recorded = baseline["recorded_with"]["legacy_identity"]
    current = service.client("legacy").hello["identity"]
    if current.get("numpy") != recorded.get("numpy"):
        pytest.skip(f"environment differs from the recorded baseline (numpy {current.get('numpy')} vs "
                    f"{recorded.get('numpy')})")
    backend = get_backend("legacy")
    for name in ("parallel_unequal", "array_2x2_unequal", "serial_three_inlets", "right_triangle_bottom"):
        cfg = ProjectConfig.load(ROOT / "examples" / f"{name}.json")
        run = service.run_backend("legacy", cfg, "baseline", timeout=600)
        assert run.status == "success", run.message
        labels = np.load(service.workdir / "baseline_legacy.npz")["labels"]
        digest = hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()
        assert digest == baseline["cases"][name]["labels_sha256"], name
        assert backend.check(cfg) == []


def test_legacy_pillar_moves_labels_and_reports_frozen_settings(service):
    # The legacy operator's own default (120 steps) moves labels by less than half a cell per
    # step and freezes; the adapter's automatic step count must let the pillar act.
    cfg = ProjectConfig.load(ROOT / "examples" / "right_triangle_bottom.json")
    run = service.run_backend("legacy", cfg, "pillar-auto", timeout=600)
    assert run.status == "success", run.message
    pillar_ops = [e for e in run.result.diagnostics["operation_log"] if e["op"].startswith("pillar_")]
    assert pillar_ops and all(e["changed_fraction"] > 0.01 for e in pillar_ops)
    assert not any("changed no labels" in w for w in run.result.warnings)

    cfg.legacy.secondary_steps = 120
    run = service.run_backend("legacy", cfg, "pillar-frozen", timeout=600)
    assert run.status == "success", run.message
    assert any("changed no labels" in w for w in run.result.warnings)


def test_capability_checks_explain_unsupported_cases():
    tri = [(0.0, 0.0), (150 * UM, 0.0), (0.0, 120 * UM)]
    top = make_config(pillars=[PillarSpec("t", tri, height=90 * UM, x=300 * UM, y=40 * UM, mount="top")])
    assert any("top mount" in r for r in build_plan(top)["unsupported"])
    pressure = make_config()
    pressure.operating = OperatingConditions(drive="pressure")
    assert any("pressure" in r for r in build_plan(pressure)["unsupported"])


# ----------------------------------------------------------------------------- paired comparison

def test_paired_run_same_input_and_export(service, tmp_path):
    cfg = ProjectConfig.load(ROOT / "examples" / "parallel_unequal.json")
    comparison = service.run_pair(cfg, clear_cache=True)
    assert comparison.runs["legacy"].status == "success" and comparison.runs["new"].status == "success"
    assert comparison.runs["new"].result.input_hash == comparison.input_hash
    assert comparison.metrics["available"]
    assert comparison.speedup["value"] is not None
    assert comparison.is_current(cfg)
    edited = cfg.copy()
    edited.inlets[0].flow_rate *= 2
    assert not comparison.is_current(edited)
    out = export_comparison(comparison, tmp_path / "export")
    for name in ("config.json", "comparison.json", "effective_settings.json", "legacy_result.npz",
                 "new_result.npz", "side_by_side.png", "differences.npz", "README.txt"):
        assert (out / name).exists(), name
    summary = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert summary["legacy_snapshot"]["verified"] is True


def test_unsupported_legacy_case_keeps_new_result_and_gives_no_speedup(service):
    cfg = ProjectConfig.load(ROOT / "examples" / "right_triangle_top.json")
    cfg.grid.output_ny = 64
    comparison = service.run_pair(cfg)
    assert comparison.runs["legacy"].status == "unsupported"
    assert comparison.runs["legacy"].unsupported
    assert comparison.runs["new"].status == "success"
    assert comparison.speedup["value"] is None


def test_timeout_cancels_and_restarts_the_worker(service):
    cfg = ProjectConfig.load(ROOT / "examples" / "two_nearby_pillars.json")
    run = service.run_backend("legacy", cfg, "timeout", timeout=0.3)
    assert run.status == "cancelled"
    quick = ProjectConfig.load(ROOT / "examples" / "parallel_unequal.json")
    assert service.run_backend("legacy", quick, "after", timeout=120).status == "success"


def test_benchmark_protocol(service):
    cfg = ProjectConfig.load(ROOT / "examples" / "parallel_unequal.json")
    comparison = service.benchmark(cfg, trials=2, include_cold=True)
    stats = comparison.benchmark["statistics"]
    assert stats["legacy"]["warm_uncached"]["n"] == 2 and stats["new"]["warm_uncached"]["n"] == 2
    assert stats["new"]["cached"]["n"] == 2
    assert comparison.speedup["scope"] == "median warm-worker uncached solve time"
    assert set(comparison.benchmark["cold"]) == {"new", "legacy"}
