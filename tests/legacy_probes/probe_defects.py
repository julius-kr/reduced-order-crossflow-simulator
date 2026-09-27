"""Probe known defects of the frozen legacy solver. Run as a script with ``python -I``.

Usage: ``python -I probe_defects.py <legacy_src>``; prints one JSON object.
Only the archived package is importable here (asserted).
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

legacy_src = Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(legacy_src))

import numpy as np  # noqa: E402

import crossflow_simulator  # noqa: E402

assert Path(crossflow_simulator.__file__).resolve().is_relative_to(legacy_src), crossflow_simulator.__file__

from crossflow_simulator.cross_section import CrossSection  # noqa: E402
from crossflow_simulator.inlets import Inlet  # noqa: E402
from crossflow_simulator.lubrication import (  # noqa: E402
    conductance_from_gap, solve_lubrication_pressure, velocity_from_pressure,
)
from crossflow_simulator.operators import AutoInletStackingOperator  # noqa: E402
from crossflow_simulator.plotter import DEFAULT_COLORS  # noqa: E402
from crossflow_simulator.secondary_flow import DivergenceDrivenLocalizedSecondaryFlowOperator  # noqa: E402

out: dict = {"loaded_from": crossflow_simulator.__file__}

cs = CrossSection(ny=128, nz=128)
cs.labels = 1 + (cs.y[None, :] > 0.5).astype(int) + 2 * (cs.z[:, None] > 0.5).astype(int)
forcing = np.sin(2 * np.pi * cs.y)
out["changed_fraction_by_steps"] = {
    str(n): float(np.mean(DivergenceDrivenLocalizedSecondaryFlowOperator(forcing_y=forcing, n_steps=n).apply(cs).labels
                          != cs.labels)) for n in (20, 120, 240)}
a = DivergenceDrivenLocalizedSecondaryFlowOperator(forcing_y=forcing, n_steps=20, strength=0.01).apply(cs)
b = DivergenceDrivenLocalizedSecondaryFlowOperator(forcing_y=forcing, n_steps=20, strength=0.5).apply(cs)
out["strength_ignored_with_normalization"] = bool(np.array_equal(a.labels, b.labels))

inlets = [Inlet(0, y, z, 1, label) for y, z, label in ((0.25, 0.25, 10), (0.75, 0.25, 20), (0.25, 0.75, 30),
                                                        (0.75, 0.75, 40))]
first = AutoInletStackingOperator(inlets).apply(cs)
inlets[0].flow_rate = 9
second = AutoInletStackingOperator(inlets).apply(cs)
out["array_flow_rate_ignored"] = bool(np.array_equal(first.labels, second.labels))
out["array_labels"] = np.unique(first.labels).tolist()

try:
    AutoInletStackingOperator([Inlet(0, 0.5, 0.5, 1, 1)]).apply(cs)
    out["single_inlet_supported"] = True
except ValueError:
    out["single_inlet_supported"] = False

# "velocity" is k * grad p with k = h^3: it scales like the depth-integrated flux (h^3), not the mean velocity (h^2).
x = np.linspace(0, 1, 21)
y = np.linspace(0, 1, 11)
ratios = []
for gap in (1.0, 0.5):
    k = conductance_from_gap(np.full((11, 21), gap))
    with contextlib.redirect_stdout(io.StringIO()):
        p = solve_lubrication_pressure(k)
    ux, _ = velocity_from_pressure(p, k, x, y)
    ratios.append(float(ux[5, 10]))
out["velocity_scaling_half_gap"] = ratios[1] / ratios[0]

buffer = io.StringIO()
gap = np.full((11, 21), 1.0)
gap[3:8, 8:13] = 0.3  # a pillar: the linear initial guess is no longer the solution
with contextlib.redirect_stdout(buffer):
    solve_lubrication_pressure(conductance_from_gap(gap), max_iterations=3)
out["non_convergence_is_print_only"] = "did not converge" in buffer.getvalue()
out["palette_size"] = len(DEFAULT_COLORS)
print(json.dumps(out))
