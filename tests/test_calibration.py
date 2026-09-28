"""Reference-data contract, family-wise holdout, evaluation, and envelopes."""
import json
import shutil

import numpy as np
import pytest

from crossflow_simulator.calibration import (
    ReferenceError, evaluate, load_directory, load_reference, split_by_family, validation_report,
)
from crossflow_simulator.simulator import simulate

from conftest import ROOT

ANALYTICAL = ROOT / "reference_data" / "analytical"


def test_analytical_references_load_and_pass():
    report = validation_report(ANALYTICAL)
    assert all(case["passed"] for case in report["cases"].values())
    assert report["envelope"]["externally_validated"] is False
    assert report["summary"]["holdout"]["cases"] >= 1


def test_viscosity_partition_matches_the_closed_form_interface():
    case = load_reference(ANALYTICAL / "layered_viscosity_wide_4p0.json")
    result = evaluate(case, simulate(case.config))
    assert result["max_interface_distance_fraction"] < 0.005


def test_references_without_reuse_permission_are_rejected(tmp_path):
    shutil.copy(ANALYTICAL / "lateral_layers_equal.json", tmp_path / "x.json")
    shutil.copy(ANALYTICAL / "lateral_layers_equal.npz", tmp_path / "x.npz")
    meta = json.loads((tmp_path / "x.json").read_text(encoding="utf-8"))
    meta["source"]["reuse_permitted"] = False
    (tmp_path / "x.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ReferenceError, match="reuse"):
        load_reference(tmp_path / "x.json")


def test_labels_must_use_stream_codes_of_the_config(tmp_path):
    shutil.copy(ANALYTICAL / "lateral_layers_equal.json", tmp_path / "x.json")
    data = dict(np.load(ANALYTICAL / "lateral_layers_equal.npz"))
    data["labels"] = data["labels"] * 10
    np.savez(tmp_path / "x.npz", **data)
    with pytest.raises(ReferenceError, match="stream codes"):
        load_reference(tmp_path / "x.json")


def test_holdout_split_keeps_families_together():
    cases = load_directory(ANALYTICAL)
    train, holdout = split_by_family(cases, holdout_fraction=0.5, seed=3)
    assert not ({c.family for c in train} & {c.family for c in holdout})
    assert len(train) + len(holdout) == len(cases)
