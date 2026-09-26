"""Application service: statuses, progress, cancellation, persistence, and the example projects."""
import json

import numpy as np
import pytest

from crossflow_simulator.config import PillarSpec, ProjectConfig
from crossflow_simulator.results import SimulationResult
from crossflow_simulator.simulator import simulate

from conftest import UM, make_config

TRIANGLE = [(0.0, 0.0), (150 * UM, 0.0), (0.0, 120 * UM)]


def test_invalid_configuration_is_a_failed_result_not_a_prediction():
    cfg = make_config()
    cfg.inlets[0].flow_rate = -1.0
    result = simulate(cfg)
    assert result.status == "failed"
    assert not result.sections
    assert result.diagnostics["issues"]


def test_numerical_failure_is_reported(monkeypatch):
    from crossflow_simulator.solvers import stokes

    def broken(self, *args, **kwargs):
        raise RuntimeError("Stokes solve failed: relative residual 1e-2")

    from crossflow_simulator.cache import memory_cache

    memory_cache().clear()  # force a fresh solve rather than a cached basis
    monkeypatch.setattr(stokes.StokesProblem, "solve", broken)
    cfg = make_config(pillars=[PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM)])
    result = simulate(cfg)
    assert result.status == "failed" and "residual" in result.message


def test_cancellation_returns_a_cancelled_result():
    cfg = make_config(pillars=[PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM)])
    result = simulate(cfg, cancel=lambda: True)
    assert result.status == "cancelled"


def test_progress_reports_stages_in_order():
    stages = []
    simulate(make_config(), progress=lambda stage, fraction, message: stages.append((stage, fraction)))
    assert stages[0][0] == "validate" and stages[-1] == ("done", 1.0)
    fractions = [f for _, f in stages]
    assert fractions == sorted(fractions)


def test_result_round_trip(tmp_path):
    cfg = make_config(pillars=[PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM)])
    result = simulate(cfg)
    path = result.save(tmp_path / "result.npz")
    loaded = SimulationResult.load(path)
    assert loaded.status == result.status and loaded.input_hash == cfg.input_hash()
    for a, b in zip(result.sections, loaded.sections):
        assert np.array_equal(a.codes, b.codes)
        assert a.name == b.name
    assert loaded.validity["model"] == result.validity["model"]
    with np.load(path, allow_pickle=False) as data:
        json.loads(str(data["meta"]))


def test_result_records_units_provenance_and_validity():
    result = simulate(make_config())
    assert result.units["length"] == "m"
    assert result.solver["name"] == "crossflow-new"
    assert result.validity["externally_validated"] is False
    assert "total" in result.timings


def test_empty_channel_is_identity():
    result = simulate(make_config(rates=(1.0, 2.0, 3.0), layout="layers_y"))
    assert np.array_equal(result.sections[0].codes, result.final.codes)


@pytest.mark.parametrize("name", [
    "empty_channel", "parallel_unequal", "array_2x2_unequal", "right_triangle_bottom", "right_triangle_reflected",
    "right_triangle_top", "serial_three_inlets", "two_nearby_pillars", "pressure_driven_serial",
])
def test_example_projects_run(examples_dir, name):
    cfg = ProjectConfig.load(examples_dir / f"{name}.json")
    cfg.grid.output_ny = 96
    result = simulate(cfg)
    assert result.ok, result.message
    assert result.diagnostics["max_flux_fraction_error"] < 0.01
    assert float(np.mean(result.final.codes < 0)) < 0.005


def test_viscosity_contrast_example_runs_coupled(examples_dir):
    cfg = ProjectConfig.load(examples_dir / "viscosity_contrast_layers.json")
    cfg.grid.output_ny = 96
    result = simulate(cfg)
    assert result.ok
    assert result.diagnostics["viscosity_coupling"] is True
    assert "experimental" in result.validity["status"]
    assert result.diagnostics["groups"][0]["coupling"]["converged"]


def test_pressure_driven_rates_come_from_the_network(examples_dir):
    cfg = ProjectConfig.load(examples_dir / "pressure_driven_serial.json")
    cfg.grid.output_ny = 64
    result = simulate(cfg)
    network = result.diagnostics["network"]
    assert network["converged"]
    rates = network["inlet_rates_m3_s"]
    # Same supply pressure; the downstream port sees a lower channel pressure and so delivers more.
    assert rates["in2"] > rates["in1"] > 0
    assert network["segment_rates_m3_s"][-1] == pytest.approx(rates["in1"] + rates["in2"])
    assert result.final.diagnostics["flux_fraction"][2] == pytest.approx(
        rates["in2"] / (rates["in1"] + rates["in2"]), abs=0.006)


def test_conservative_transport_conserves_flux_exactly():
    cfg = make_config(pillars=[PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM)])
    cfg.model.transport = "conservative"
    result = simulate(cfg)
    info = result.diagnostics["conservative_transport"]["groups"][0]
    fractions = result.final.diagnostics["flux_fraction_error"]
    assert max(abs(v) for v in fractions.values()) < 1e-3 + info["lost_flux_fraction"]
    assert result.final.fractions is not None
