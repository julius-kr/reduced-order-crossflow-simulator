"""Inlet topology: prescribed cross-sectional patterns and physical port stations.

Two workflows are kept separate and nothing is inferred from coordinates:

* a *prescribed pattern* of material patches entering at ``x = 0`` (single
  inlet, lateral layers, vertical layers, or a rows x cols grid), each patch
  with its own flow rate and stable inlet ID;
* *physical ports* in channel walls at explicit streamwise stations, handled
  by junction flow maps downstream.

A pattern partition is flux-constrained: every patch carries exactly its
prescribed share of the flow under the developed velocity field (including a
viscosity contrast, when present). General grids of flow rates are not
separable, so an aligned rectangular partition is only used when it can match
every patch; otherwise a staggered partition is used and reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import least_squares

from . import physics
from .config import InletSpec, ProjectConfig
from .solvers.section import solve_axial_velocity

PartitionKind = Literal["single", "layers_y", "layers_z", "aligned", "staggered_rows", "staggered_columns"]


class AmbiguousLayoutError(ValueError):
    pass


@dataclass
class FluxTable:
    """Integral image of an axial velocity field for fast region-flux queries."""

    y_edges: np.ndarray
    z_edges: np.ndarray
    cumulative: np.ndarray  # (nz+1, ny+1)

    @classmethod
    def from_velocity(cls, y_edges: np.ndarray, z_edges: np.ndarray, velocity: np.ndarray) -> "FluxTable":
        flux = velocity * np.outer(np.diff(z_edges), np.diff(y_edges))
        cumulative = np.zeros((len(z_edges), len(y_edges)))
        cumulative[1:, 1:] = np.cumsum(np.cumsum(flux, axis=0), axis=1)
        return cls(y_edges, z_edges, cumulative)

    @property
    def total(self) -> float:
        return float(self.cumulative[-1, -1])

    def at(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the integral image at (y, z)."""
        y = np.clip(np.asarray(y, dtype=float), self.y_edges[0], self.y_edges[-1])
        z = np.clip(np.asarray(z, dtype=float), self.z_edges[0], self.z_edges[-1])
        j = np.clip(np.searchsorted(self.y_edges, y) - 1, 0, len(self.y_edges) - 2)
        k = np.clip(np.searchsorted(self.z_edges, z) - 1, 0, len(self.z_edges) - 2)
        ty = (y - self.y_edges[j]) / (self.y_edges[j + 1] - self.y_edges[j])
        tz = (z - self.z_edges[k]) / (self.z_edges[k + 1] - self.z_edges[k])
        c = self.cumulative
        return ((1 - tz) * ((1 - ty) * c[k, j] + ty * c[k, j + 1])
                + tz * ((1 - ty) * c[k + 1, j] + ty * c[k + 1, j + 1]))

    def region(self, y0, y1, z0, z1) -> np.ndarray:
        return self.at(y1, z1) - self.at(y0, z1) - self.at(y1, z0) + self.at(y0, z0)

    def invert_y(self, target: float, z0: float, z1: float) -> float:
        nodes = self.at(self.y_edges, np.full_like(self.y_edges, z1)) - self.at(self.y_edges, np.full_like(self.y_edges, z0))
        return float(np.interp(target, nodes, self.y_edges))

    def invert_z(self, target: float, y0: float, y1: float) -> float:
        nodes = self.at(np.full_like(self.z_edges, y1), self.z_edges) - self.at(np.full_like(self.z_edges, y0), self.z_edges)
        return float(np.interp(target, nodes, self.z_edges))


@dataclass
class PatternPartition:
    """Flux-constrained geometry of the prescribed inlet pattern at x = 0."""

    kind: PartitionKind
    width: float
    height: float
    codes: np.ndarray  # (rows, cols) stream codes, row 0 at the bottom, col 0 on the left
    row_edges: list[np.ndarray]  # staggered_columns: per column; otherwise one array
    col_edges: list[np.ndarray]  # staggered_rows: per row; otherwise one array
    target_fraction: np.ndarray  # (rows, cols)
    achieved_fraction: np.ndarray  # (rows, cols)
    iterations: int = 0
    converged: bool = True
    notes: list[str] = field(default_factory=list)

    def evaluate(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Stream code at arbitrary points of the x = 0 plane (vectorized)."""
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        out = np.zeros(np.broadcast(y, z).shape, dtype=np.int32)
        y, z = np.broadcast_arrays(y, z)
        rows, cols = self.codes.shape
        if self.kind == "staggered_columns":
            col = _locate(self.col_edges[0], y)
            for c in range(cols):
                sel = col == c
                row = _locate(self.row_edges[c], z[sel])
                out[sel] = self.codes[row, c]
            return out
        row = _locate(self.row_edges[0], z)
        if self.kind == "staggered_rows":
            for r in range(rows):
                sel = row == r
                out[sel] = self.codes[r, _locate(self.col_edges[r], y[sel])]
            return out
        return self.codes[row, _locate(self.col_edges[0], y)]

    def max_fraction_error(self) -> float:
        return float(np.max(np.abs(self.achieved_fraction - self.target_fraction)))

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "row_edges_m": [e.tolist() for e in self.row_edges],
            "col_edges_m": [e.tolist() for e in self.col_edges],
            "target_fraction": self.target_fraction.tolist(),
            "achieved_fraction": self.achieved_fraction.tolist(),
            "max_fraction_error": self.max_fraction_error(),
            "iterations": self.iterations,
            "converged": self.converged,
            "notes": list(self.notes),
        }


def _locate(edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def pattern_grid(cfg: ProjectConfig) -> tuple[np.ndarray, np.ndarray, list[InletSpec]]:
    """Stream codes and flow rates arranged as (rows, cols) from the pattern inlets."""
    inlets = cfg.pattern_inlets()
    codes_map = cfg.stream_codes()
    layout = cfg.pattern.layout
    if layout == "single":
        rows, cols = 1, 1
    elif layout == "layers_y":
        rows, cols = 1, len(inlets)
    elif layout == "layers_z":
        rows, cols = len(inlets), 1
    else:
        rows, cols = cfg.pattern.rows, cfg.pattern.cols
    codes = np.zeros((rows, cols), dtype=np.int32)
    rates = np.zeros((rows, cols))
    for inlet in inlets:
        r = 0 if layout in ("single", "layers_y") else inlet.pattern_row
        c = 0 if layout in ("single", "layers_z") else inlet.pattern_col
        codes[r, c] = codes_map[inlet.id]
        rates[r, c] = inlet.flow_rate if inlet.flow_rate is not None else np.nan
    return codes, rates, inlets


def developed_flux_table(width: float, height: float, viscosity: np.ndarray | float, n_y: int = 240) -> FluxTable:
    """Integral image of the developed profile on a fine midpoint grid."""
    n_z = max(24, int(round(n_y * height / width)))
    y_edges = np.linspace(0.0, width, n_y + 1)
    z_edges = np.linspace(0.0, height, n_z + 1)
    if np.ndim(viscosity) == 0:
        y_c, z_c = 0.5 * (y_edges[1:] + y_edges[:-1]), 0.5 * (z_edges[1:] + z_edges[:-1])
        velocity = physics.rectangular_duct_profile(y_c, z_c, width, height, normalization="none") / float(viscosity)
    else:
        velocity = solve_axial_velocity(y_edges, z_edges, viscosity).velocity
    return FluxTable.from_velocity(y_edges, z_edges, velocity)


def partition_pattern(
    cfg: ProjectConfig,
    rates: np.ndarray | None = None,
    viscosity_coupling: bool | None = None,
    tolerance: float = 1e-4,
    max_iterations: int = 40,
) -> PatternPartition:
    """Flux-constrained partition of the prescribed pattern.

    ``rates`` overrides the inlet flow rates (used in pressure-driven mode).
    With a viscosity contrast among pattern materials, the developed profile
    is recomputed from the current partition until the boundaries converge.
    """
    codes, config_rates, inlets = pattern_grid(cfg)
    rates = config_rates if rates is None else np.asarray(rates, dtype=float)
    if np.any(~(rates > 0)):
        raise ValueError("Every pattern inlet needs a positive flow rate.")
    W, H = cfg.channel.width, cfg.channel.height
    target = rates / rates.sum()
    code_to_mu = {cfg.stream_codes()[i.id]: cfg.material(i.material).viscosity for i in cfg.inlets}
    viscosities = {code_to_mu[c] for c in codes.ravel()}
    contrast = len(viscosities) > 1
    if viscosity_coupling is None:
        viscosity_coupling = cfg.model.viscosity_coupling != "off"
    couple = contrast and viscosity_coupling

    table = developed_flux_table(W, H, next(iter(viscosities)))
    partition = _partition_once(cfg, codes, target, table)
    notes: list[str] = []
    if contrast and not couple:
        notes.append("Viscosity contrast ignored in the pattern partition (coupling disabled).")
    iterations, converged = 0, True
    if couple:
        converged = False
        y_c = 0.5 * (table.y_edges[1:] + table.y_edges[:-1])
        z_c = 0.5 * (table.z_edges[1:] + table.z_edges[:-1])
        Y, Z = np.meshgrid(y_c, z_c)
        for iterations in range(1, max_iterations + 1):
            mu_field = np.vectorize(code_to_mu.get)(partition.evaluate(Y, Z)).astype(float)
            table = developed_flux_table(W, H, mu_field)
            updated = _partition_once(cfg, codes, target, table)
            change = max(
                max(float(np.max(np.abs(a - b))) for a, b in zip(updated.row_edges, partition.row_edges)) / H,
                max(float(np.max(np.abs(a - b))) for a, b in zip(updated.col_edges, partition.col_edges)) / W,
            )
            partition = updated
            if change < tolerance:
                converged = True
                break
        notes.append(f"Viscosity-coupled partition: {iterations} iterations, converged={converged}.")
    partition.iterations = iterations
    partition.converged = converged
    partition.notes = notes + partition.notes
    return partition


def _partition_once(cfg: ProjectConfig, codes: np.ndarray, target: np.ndarray, table: FluxTable) -> PatternPartition:
    W, H = cfg.channel.width, cfg.channel.height
    rows, cols = codes.shape
    layout = cfg.pattern.layout
    total = table.total
    notes: list[str] = []

    def cumulative_edges(fractions: np.ndarray, invert) -> np.ndarray:
        cum = np.cumsum(fractions)[:-1] * total
        return np.array([0.0, *[invert(c) for c in cum], 1.0])

    if layout == "single":
        kind: PartitionKind = "single"
        row_edges, col_edges = [np.array([0.0, H])], [np.array([0.0, W])]
    elif layout == "layers_y":
        kind = "layers_y"
        edges = cumulative_edges(target[0], lambda c: table.invert_y(c, 0.0, H))
        edges[-1] = W
        row_edges, col_edges = [np.array([0.0, H])], [edges]
    elif layout == "layers_z":
        kind = "layers_z"
        edges = cumulative_edges(target[:, 0], lambda c: table.invert_z(c, 0.0, W))
        edges[-1] = H
        row_edges, col_edges = [edges], [np.array([0.0, W])]
    else:
        choice = cfg.pattern.partition
        aligned = None
        if choice in ("auto", "aligned"):
            aligned = _aligned_partition(target, table, W, H)
            if aligned is None and choice == "aligned":
                raise ValueError(
                    "An aligned rectangular partition cannot carry the prescribed per-inlet flow rates; "
                    "choose a staggered partition or supply a junction map."
                )
            if aligned is None:
                notes.append("Aligned partition infeasible for these flow rates; using staggered rows.")
        if aligned is not None:
            kind = "aligned"
            row_edges, col_edges = [aligned[0]], [aligned[1]]
        elif choice == "staggered_columns":
            kind = "staggered_columns"
            col_e = cumulative_edges(target.sum(axis=0), lambda c: table.invert_y(c, 0.0, H))
            col_e[-1] = W
            row_edges = []
            for c in range(cols):
                base = float(table.region(col_e[c], col_e[c + 1], 0.0, 0.0))
                cum = np.cumsum(target[:, c])[:-1] * total + base
                inner = [table.invert_z(v, col_e[c], col_e[c + 1]) for v in cum]
                row_edges.append(np.array([0.0, *inner, H]))
            col_edges = [col_e]
        else:
            kind = "staggered_rows"
            row_e = cumulative_edges(target.sum(axis=1), lambda c: table.invert_z(c, 0.0, W))
            row_e[-1] = H
            col_edges = []
            for r in range(rows):
                cum = np.cumsum(target[r])[:-1] * total
                inner = [table.invert_y(v, row_e[r], row_e[r + 1]) for v in cum]
                col_edges.append(np.array([0.0, *inner, W]))
            row_edges = [row_e]
    partition = PatternPartition(kind, W, H, codes, row_edges, col_edges, target, np.zeros_like(target), notes=notes)
    partition.achieved_fraction = _achieved(partition, table)
    return partition


def _aligned_partition(target: np.ndarray, table: FluxTable, W: float, H: float, tol: float = 1e-3):
    rows, cols = target.shape
    total = table.total

    def edges_from(params: np.ndarray, n: int, span: float) -> np.ndarray:
        weights = np.exp(params)
        return np.concatenate([[0.0], np.cumsum(weights)[:-1] / weights.sum() * span, [span]])

    def residual(params: np.ndarray) -> np.ndarray:
        z_e = edges_from(params[:rows], rows, H)
        y_e = edges_from(params[rows:], cols, W)
        Z0, Y0 = np.meshgrid(z_e[:-1], y_e[:-1], indexing="ij")
        Z1, Y1 = np.meshgrid(z_e[1:], y_e[1:], indexing="ij")
        return (table.region(Y0, Y1, Z0, Z1) / total - target).ravel()

    row_guess = np.log(np.maximum(target.sum(axis=1), 1e-6))
    col_guess = np.log(np.maximum(target.sum(axis=0), 1e-6))
    fit = least_squares(residual, np.concatenate([row_guess, col_guess]), xtol=1e-12, ftol=1e-12)
    if np.max(np.abs(fit.fun)) > tol:
        return None
    return edges_from(fit.x[:rows], rows, H), edges_from(fit.x[rows:], cols, W)


def _achieved(partition: PatternPartition, table: FluxTable) -> np.ndarray:
    rows, cols = partition.codes.shape
    achieved = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            if partition.kind == "staggered_columns":
                z_e, y_e = partition.row_edges[c], partition.col_edges[0]
            elif partition.kind == "staggered_rows":
                z_e, y_e = partition.row_edges[0], partition.col_edges[r]
            else:
                z_e, y_e = partition.row_edges[0], partition.col_edges[0]
            achieved[r, c] = float(table.region(y_e[c], y_e[c + 1], z_e[r], z_e[r + 1]))
    return achieved / table.total


def infer_pattern_from_points(points: Sequence[tuple[float, float, float]], tolerance: float = 1e-9) -> str:
    """Classify legacy-style inlet coordinates, refusing ambiguous cases.

    Returns ``single``, ``serial_ports``, ``layers_y``, ``layers_z``, or ``grid``.
    A single inlet is supported. Mixed variation (for example both ``x`` and
    ``y``) is ambiguous: a physical array can mean parallel ports at successive
    stations or a prescribed cross-sectional pattern, so it must be specified.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError("Expected a non-empty list of (x, y, z) points.")
    if len(array) == 1:
        return "single"
    varies = np.ptp(array, axis=0) > tolerance
    if varies.tolist() == [True, False, False]:
        return "serial_ports"
    if varies.tolist() == [False, True, False]:
        return "layers_y"
    if varies.tolist() == [False, False, True]:
        return "layers_z"
    if varies.tolist() == [False, True, True]:
        ys = np.unique(np.round(array[:, 1] / tolerance))
        zs = np.unique(np.round(array[:, 2] / tolerance))
        if len(ys) * len(zs) == len(array):
            return "grid"
        raise AmbiguousLayoutError("Points in the y-z plane do not form a complete rows x cols grid.")
    raise AmbiguousLayoutError(
        "Inlet coordinates vary along several axes; specify pattern inlets and port stations explicitly."
    )
