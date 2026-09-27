"""Each known legacy defect: confirmed in the frozen code, and its replacement behaviour in the new solver.

The legacy side runs in an isolated interpreter that can only import the
archived snapshot; the new side runs in this process. Legacy results are
behaviour records, not references.
"""
import json
import subprocess
import sys

import numpy as np
import pytest

from crossflow_simulator.backends.legacy import legacy_python, legacy_root
from crossflow_simulator.config import FluidProperties, InitialPattern, PillarSpec
from crossflow_simulator.materials import PALETTE, palette_color
from crossflow_simulator.simulator import simulate
from crossflow_simulator.solvers.pressure import solve_lubrication

from conftest import ROOT, UM, make_config

pytestmark = pytest.mark.legacy


@pytest.fixture(scope="module")
def legacy():
    script = ROOT / "tests" / "legacy_probes" / "probe_defects.py"
    completed = subprocess.run([legacy_python(), "-I", str(script), str(legacy_root() / "python" / "src")],
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def grid_config(rates):
    cfg = make_config(rates=rates, layout="grid", pattern=InitialPattern(layout="grid", rows=2, cols=2))
    for i, inlet in enumerate(cfg.inlets):
        inlet.pattern_row, inlet.pattern_col = divmod(i, 2)
    return cfg


def test_probe_ran_against_the_snapshot(legacy):
    assert "legacy" in legacy["loaded_from"].replace("\\", "/")


def test_array_flow_rates(legacy):
    assert legacy["array_flow_rate_ignored"] is True
    a = simulate(grid_config((1.0, 2.0, 3.0, 4.0)))
    b = simulate(grid_config((9.0, 2.0, 3.0, 4.0)))
    assert not np.array_equal(a.final.codes, b.final.codes)
    fractions = b.final.diagnostics["flux_fraction"]
    assert fractions[1] == pytest.approx(9.0 / 18.0, abs=0.005)


def test_stable_labels(legacy):
    assert legacy["array_labels"] == [10, 11, 12, 13]  # requested 10, 20, 30, 40
    cfg = grid_config((1.0, 2.0, 3.0, 4.0))
    result = simulate(cfg)
    assert sorted(np.unique(result.final.codes).tolist()) == sorted(cfg.stream_codes().values())
    assert [s["inlet_id"] for s in result.streams] == [i.id for i in cfg.inlets]


def test_single_inlet(legacy):
    assert legacy["single_inlet_supported"] is False
    result = simulate(make_config(rates=(1.0,), layout="single"))
    assert result.ok and np.all(result.final.codes == 1)


def test_subpixel_motion(legacy):
    changed = legacy["changed_fraction_by_steps"]
    assert changed["20"] > 0 and changed["120"] == 0.0 and changed["240"] == 0.0
    # Replacement: continuous coordinates (see test_stokes_transport.test_subpixel_displacement_*);
    # here, a pillar deformation is independent of the output resolution used for sampling.
    tri = [(0.0, 0.0), (150 * UM, 0.0), (0.0, 120 * UM)]
    coarse = make_config(pillars=[PillarSpec("t", tri, height=90 * UM, x=300 * UM, y=40 * UM)])
    coarse.grid.output_ny = 64
    fine = make_config(pillars=[PillarSpec("t", tri, height=90 * UM, x=300 * UM, y=40 * UM)])
    fine.grid.output_ny = 192
    fc = simulate(coarse).final.diagnostics["flux_fraction"][1]
    ff = simulate(fine).final.diagnostics["flux_fraction"][1]
    assert fc == pytest.approx(ff, abs=0.006)


def test_arbitrary_strength_normalization(legacy):
    assert legacy["strength_ignored_with_normalization"] is True
    # Replacement: no strength parameter; deformation follows physical geometry (see test_models) and
    # is invariant to the overall flow-rate scale, as creeping flow requires.
    tri = [(0.0, 0.0), (150 * UM, 0.0), (0.0, 120 * UM)]
    slow = simulate(make_config(rates=(1.0, 1.0), pillars=[PillarSpec("t", tri, height=90 * UM, x=300 * UM, y=40 * UM)]))
    fast = simulate(make_config(rates=(3.0, 3.0), pillars=[PillarSpec("t", tri, height=90 * UM, x=300 * UM, y=40 * UM)]))
    assert np.mean(slow.final.codes != fast.final.codes) < 1e-3


def test_flux_versus_mean_velocity(legacy):
    assert legacy["velocity_scaling_half_gap"] == pytest.approx(0.125, rel=1e-6)  # h^3: a flux, named velocity
    edges_x, edges_y = np.linspace(0, 1e-3, 21), np.linspace(0, 5e-4, 11)
    full = solve_lubrication(np.full((10, 20), 1e-4), edges_x, edges_y, 1e-3, 1.0)
    half = solve_lubrication(np.full((10, 20), 5e-5), edges_x, edges_y, 1e-3, 1.0)
    assert half.flow_rate / full.flow_rate == pytest.approx(0.125)
    assert half.mean_velocity()[0][5, 10] / full.mean_velocity()[0][5, 10] == pytest.approx(0.25)


def test_convergence_status(legacy):
    assert legacy["non_convergence_is_print_only"] is True
    # Replacement: structured status; failures become failed results (test_simulator.test_numerical_failure_*).
    sol = solve_lubrication(np.full((10, 20), 1e-4), np.linspace(0, 1e-3, 21), np.linspace(0, 5e-4, 11), 1e-3, 1.0)
    assert sol.converged is True and sol.relative_residual < 1e-8


def test_material_colours(legacy):
    assert legacy["palette_size"] == 8
    colours = {palette_color(i) for i in range(24)}
    assert len(colours) == 24 and len(PALETTE) >= 20


def test_viscosity_is_not_silently_dropped():
    from crossflow_simulator.backends.legacy import build_plan

    cfg = make_config(materials=[FluidProperties("A", 0.04), FluidProperties("B", 0.01)])
    assert any("viscosity" in reason for reason in build_plan(cfg)["unsupported"])
