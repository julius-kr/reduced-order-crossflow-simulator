"""Reference-data contract, evaluation against references, family-wise holdout, and envelopes.

A reference case is a pair ``<name>.json`` (metadata) + ``<name>.npz`` (arrays).
Required metadata (``crossflow-reference/1``):

``source``        kind (``cfd`` | ``experiment`` | ``analytical`` | ``numerical_self``), tool,
                  citation, license, ``reuse_permitted`` (must be true to be used)
``config``        the physical case as a ``crossflow-project/1`` project
``plane``         observation plane: ``x_m`` and a description (inlet, post-pillar, outlet ...)
``quantity``      ``labels`` (stream codes of the config on a cell-centred grid) or
                  ``interface`` (interface height ``z(y)`` between lower and upper codes)
``orientation``   coordinate convention (x streamwise, y from the left wall looking
                  downstream, z from the bottom wall); references in other conventions
                  must be transformed before import
``family``        geometry/operating-condition family: training and holdout sets are split
                  by complete families, never by images of one case
``numerics``      solver, tolerances, mesh-convergence evidence (CFD); for experiments the
                  registration and imaging uncertainty
``uncertainty``   optional ``interface_m`` (used to widen acceptance thresholds)

Only ``cfd`` and ``experiment`` sources count as external validation. Analytical and
numerical self-references verify numerics and are reported as such.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import metrics
from .config import ProjectConfig
from .results import Section, SimulationResult

SCHEMA = "crossflow-reference/1"
EXTERNAL_KINDS = {"cfd", "experiment"}
# Initial acceptance targets; to be refined with reference uncertainty.
TARGETS = {
    "interface_distance_fraction_of_height": 0.02,
    "iou_min": 0.90,
    "flux_fraction_error_max": 0.005,
    "rms_displacement_fraction": 0.05,
}


class ReferenceError(ValueError):
    pass


@dataclass
class ReferenceCase:
    name: str
    meta: dict
    arrays: dict[str, np.ndarray]
    config: ProjectConfig

    @property
    def family(self) -> str:
        return self.meta["family"]

    @property
    def external(self) -> bool:
        return self.meta["source"]["kind"] in EXTERNAL_KINDS


def load_reference(path: str | Path) -> ReferenceCase:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA:
        raise ReferenceError(f"{path.name}: schema must be {SCHEMA!r}")
    for key in ("source", "config", "plane", "quantity", "orientation", "family", "numerics"):
        if key not in meta:
            raise ReferenceError(f"{path.name}: missing metadata field {key!r}")
    source = meta["source"]
    for key in ("kind", "citation", "license", "reuse_permitted"):
        if key not in source:
            raise ReferenceError(f"{path.name}: source.{key} is required")
    if not source["reuse_permitted"]:
        raise ReferenceError(f"{path.name}: reuse is not permitted by the source licence")
    if source["kind"] not in EXTERNAL_KINDS | {"analytical", "numerical_self"}:
        raise ReferenceError(f"{path.name}: unknown source kind {source['kind']!r}")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    config = ProjectConfig.from_dict(meta["config"])
    if meta["quantity"] == "labels":
        for key in ("labels", "y_edges_m", "z_edges_m"):
            if key not in arrays:
                raise ReferenceError(f"{path.name}: labels references need array {key!r}")
        if arrays["labels"].shape != (len(arrays["z_edges_m"]) - 1, len(arrays["y_edges_m"]) - 1):
            raise ReferenceError(f"{path.name}: labels shape does not match the grid edges")
        unknown = set(np.unique(arrays["labels"]).tolist()) - set(config.stream_codes().values()) - {-1}
        if unknown:
            raise ReferenceError(f"{path.name}: labels {sorted(unknown)} are not stream codes of the config")
    elif meta["quantity"] == "interface":
        for key in ("y_m", "z_interface_m"):
            if key not in arrays:
                raise ReferenceError(f"{path.name}: interface references need array {key!r}")
        if "lower_codes" not in meta:
            raise ReferenceError(f"{path.name}: interface references need 'lower_codes'")
    else:
        raise ReferenceError(f"{path.name}: unsupported quantity {meta['quantity']!r}")
    return ReferenceCase(path.stem, meta, arrays, config)


def load_directory(directory: str | Path) -> list[ReferenceCase]:
    return [load_reference(p) for p in sorted(Path(directory).glob("*.json")) if _is_reference(p)]


def _is_reference(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("schema") == SCHEMA
    except (OSError, json.JSONDecodeError):
        return False


def section_at(result: SimulationResult, x: float) -> Section:
    """The computed section nearest the reference plane (sections exist after each feature group)."""
    return min(result.sections, key=lambda s: abs(s.x - x))


def evaluate(case: ReferenceCase, result: SimulationResult) -> dict[str, Any]:
    if not result.ok:
        return {"case": case.name, "status": result.status, "passed": False, "message": result.message}
    H = case.config.channel.height
    section = section_at(result, float(case.meta["plane"]["x_m"]))
    tolerance = float(case.meta.get("uncertainty", {}).get("interface_m", 0.0))
    out: dict[str, Any] = {"case": case.name, "family": case.family, "source": case.meta["source"]["kind"],
                           "section": section.name, "status": "success"}
    if case.meta["quantity"] == "labels":
        ref = Section("reference", section.x, case.arrays["y_edges_m"], case.arrays["z_edges_m"],
                      case.arrays["labels"].astype(np.int32))
        predicted = metrics.regrid_nearest(section, ref.y, ref.z)
        spacing = (float(np.diff(ref.z_edges).mean()), float(np.diff(ref.y_edges).mean()))
        per_stream = {}
        weights = None
        for code in sorted(c for c in np.unique(ref.codes) if c >= 0):
            iou = metrics.iou(predicted, ref.codes, code)
            distance = metrics.interface_distance(predicted, ref.codes, code, spacing)
            per_stream[int(code)] = {"iou": iou, "interface_distance_m": distance}
        area = ref.cell_area()
        velocity = section.axial_velocity
        if velocity is not None and velocity.shape == ref.codes.shape:
            weights = velocity * area
        else:
            weights = area
        for code, entry in per_stream.items():
            entry["flux_fraction_error"] = float((weights[predicted == code].sum() - weights[ref.codes == code].sum())
                                                 / weights.sum())
        distances = [e["interface_distance_m"] for e in per_stream.values() if np.isfinite(e["interface_distance_m"])]
        out["per_stream"] = per_stream
        out["mean_interface_distance_fraction"] = float(np.mean(distances) / H) if distances else 0.0
        out["min_iou"] = float(min(e["iou"] for e in per_stream.values()))
        out["max_flux_fraction_error"] = float(max(abs(e["flux_fraction_error"]) for e in per_stream.values()))
        passed = (out["mean_interface_distance_fraction"] <= TARGETS["interface_distance_fraction_of_height"]
                  + tolerance / H and out["min_iou"] >= TARGETS["iou_min"]
                  and out["max_flux_fraction_error"] <= TARGETS["flux_fraction_error_max"])
    else:
        lower = set(case.meta["lower_codes"])
        height = metrics.interface_height(section.codes, section.z, lower)
        predicted = np.interp(case.arrays["y_m"], section.y, height)
        reference = case.arrays["z_interface_m"]
        valid = np.isfinite(predicted) & np.isfinite(reference)
        error = np.abs(predicted[valid] - reference[valid])
        out["mean_interface_distance_fraction"] = float(error.mean() / H) if valid.any() else float("nan")
        out["max_interface_distance_fraction"] = float(error.max() / H) if valid.any() else float("nan")
        passed = bool(valid.any()) and out["mean_interface_distance_fraction"] <= (
            TARGETS["interface_distance_fraction_of_height"] + tolerance / H)
    out["passed"] = bool(passed)
    return out


def split_by_family(cases: list[ReferenceCase], holdout_fraction: float = 0.3,
                    seed: int = 0) -> tuple[list[ReferenceCase], list[ReferenceCase]]:
    """Train/holdout split by complete families (explicit ``split`` metadata wins)."""
    explicit = {c.name: c.meta.get("split") for c in cases}
    families = sorted({c.family for c in cases if not explicit[c.name]})
    rng = np.random.default_rng(seed)
    rng.shuffle(families)
    n_hold = int(round(holdout_fraction * len(families)))
    holdout_families = set(families[:n_hold])
    train, holdout = [], []
    for case in cases:
        split = explicit[case.name] or ("holdout" if case.family in holdout_families else "train")
        (holdout if split == "holdout" else train).append(case)
    shared = {c.family for c in train} & {c.family for c in holdout}
    if shared:
        raise ReferenceError(f"Families {sorted(shared)} appear in both training and holdout sets")
    return train, holdout


def envelope(cases: list[ReferenceCase], evaluations: dict[str, dict]) -> dict[str, Any]:
    """Parameter ranges covered by passing *external* holdout references (empty without them)."""
    passing = [c for c in cases if c.external and evaluations.get(c.name, {}).get("passed")]
    if not passing:
        return {"externally_validated": False, "ranges": {},
                "note": "No passing external (CFD/experimental) references: nothing is externally validated."}

    def values(fn: Callable[[ProjectConfig], float]) -> list[float]:
        return [float(fn(c.config)) for c in passing]

    ranges = {
        "aspect_ratio_H_over_W": values(lambda c: c.channel.height / c.channel.width),
        "pillar_height_ratio": [p.height / c.config.channel.height for c in passing for p in c.config.active_pillars()],
        "port_diameter_over_W": [i.diameter / c.config.channel.width for c in passing for i in c.config.port_inlets()],
        "viscosity_ratio": values(lambda c: max(m.viscosity for m in c.materials) / min(m.viscosity for m in c.materials)),
    }
    return {"externally_validated": True,
            "ranges": {k: [min(v), max(v)] for k, v in ranges.items() if v},
            "families": sorted({c.family for c in passing})}


def validation_report(directory: str | Path, simulate_fn: Callable[[ProjectConfig], SimulationResult] | None = None,
                      holdout_fraction: float = 0.3) -> dict[str, Any]:
    from .simulator import simulate

    simulate_fn = simulate_fn or simulate
    cases = load_directory(directory)
    train, holdout = split_by_family(cases, holdout_fraction)
    evaluations = {}
    for case in cases:
        evaluations[case.name] = evaluate(case, simulate_fn(case.config))
    summary = {}
    for label, subset in (("train", train), ("holdout", holdout)):
        rows = [evaluations[c.name] for c in subset]
        summary[label] = {"cases": len(rows), "passed": sum(r["passed"] for r in rows),
                          "external": sum(c.external for c in subset)}
    return {
        "schema": "crossflow-validation-report/1",
        "targets": TARGETS,
        "summary": summary,
        "cases": evaluations,
        "envelope": envelope(holdout, evaluations),
        "note": "Analytical and numerical self-references verify numerics; only CFD/experimental references validate.",
    }


@dataclass
class ReferenceWriter:
    """Helper for producing reference files in the contract format."""

    directory: Path
    written: list[str] = field(default_factory=list)

    def write(self, name: str, meta: dict, arrays: dict[str, np.ndarray]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        meta = {"schema": SCHEMA, **meta}
        (self.directory / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(self.directory / f"{name}.npz", **arrays)
        self.written.append(name)
        return self.directory / f"{name}.json"
