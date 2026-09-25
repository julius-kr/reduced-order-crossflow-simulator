"""Lumped hydraulic network for pressure-driven operation.

Each inlet is a pressure source behind its tubing/port resistance, feeding a
node of the main channel: all pattern inlets feed the node at ``x = 0``; each
port feeds a node at its station. Channel segments between nodes are
straight-duct resistances evaluated with the flux-weighted viscosity of the
streams they carry, plus the extra resistance of pillar groups from the
Stokes solution. Junction losses are neglected (a stated model assumption).
Inlet backflow is reported as an error rather than silently accepted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import physics


@dataclass
class NetworkSolution:
    inlet_rates: dict[str, float]
    node_pressures: list[float]
    node_stations: list[float]
    segment_rates: list[float]
    iterations: int
    converged: bool


def solve_network(
    width: float,
    height: float,
    length: float,
    outlet_pressure: float,
    inlets: list[dict],
    extra_resistance: dict[float, float] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 20,
) -> NetworkSolution:
    """Solve for inlet flow rates.

    ``inlets``: dicts with ``id``, ``station`` (0 for pattern inlets),
    ``pressure``, ``resistance``, ``viscosity``. ``extra_resistance`` maps a
    station (upper end of the segment it lies in) to an additional
    resistance at unit viscosity.
    """
    stations = sorted({float(i["station"]) for i in inlets})
    if stations[0] != 0.0:
        stations = [0.0, *stations]
    if length <= stations[-1]:
        raise ValueError("The channel length must extend beyond the last inlet station in pressure-driven mode.")
    nodes = stations
    n = len(nodes)
    segment_lengths = np.diff([*nodes, length])
    rates = {i["id"]: 1e-12 for i in inlets}
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        # Flux-weighted viscosity of each segment from the current rate estimate.
        mu_seg = []
        for k in range(n):
            upstream = [i for i in inlets if i["station"] <= nodes[k] + 1e-15]
            q = np.array([max(rates[i["id"]], 0.0) for i in upstream])
            mu = np.array([i["viscosity"] for i in upstream])
            mu_seg.append(float(np.sum(q * mu) / np.sum(q)) if q.sum() > 0 else float(mu.mean()))
        resistance = []
        for k in range(n):
            r = physics.duct_resistance(segment_lengths[k], width, height, mu_seg[k])
            for station, extra in (extra_resistance or {}).items():
                if nodes[k] <= station < ([*nodes, length])[k + 1]:
                    r += extra * mu_seg[k]
            resistance.append(r)
        # Unknowns: node pressures P_0..P_{n-1}; outlet pressure fixed.
        A = np.zeros((n, n))
        b = np.zeros(n)
        for k in range(n):
            g = 1.0 / resistance[k]
            A[k, k] += g
            if k + 1 < n:
                A[k, k + 1] -= g
                A[k + 1, k + 1] += 0.0
            else:
                b[k] += g * outlet_pressure
            if k > 0:
                g_prev = 1.0 / resistance[k - 1]
                A[k, k] += g_prev
                A[k, k - 1] -= g_prev
            for inlet in inlets:
                if abs(inlet["station"] - nodes[k]) < 1e-15 or (k == 0 and inlet["station"] <= 0.0):
                    gi = 1.0 / inlet["resistance"]
                    A[k, k] += gi
                    b[k] += gi * inlet["pressure"]
        pressures = np.linalg.solve(A, b)
        new_rates = {}
        for inlet in inlets:
            k = 0 if inlet["station"] <= 0.0 else nodes.index(float(inlet["station"]))
            new_rates[inlet["id"]] = (inlet["pressure"] - pressures[k]) / inlet["resistance"]
        change = max(abs(new_rates[k] - rates[k]) for k in rates) / max(max(abs(v) for v in new_rates.values()), 1e-300)
        rates = new_rates
        if change < tolerance:
            converged = True
            break
    backflow = [k for k, v in rates.items() if v <= 0]
    if backflow:
        raise ValueError(f"Prescribed pressures drive backflow into inlet(s) {backflow}; not supported.")
    segment_rates = [
        float(sum(rates[i["id"]] for i in inlets if i["station"] <= nodes[k] + 1e-15)) for k in range(n)
    ]
    return NetworkSolution(rates, pressures.tolist(), nodes, segment_rates, iteration, converged)
