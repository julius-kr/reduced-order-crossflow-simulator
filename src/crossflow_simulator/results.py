"""Versioned result contracts and NPZ persistence (no pickled objects).

A result carries its own configuration snapshot, input hash, solver identity,
model provenance and validity, timings, diagnostics, and every computed
cross-section. Failed or unsupported runs are results too, with an explicit
``status``; they never masquerade as predictions.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

RESULT_SCHEMA = "crossflow-result/1"
Status = Literal["success", "failed", "cancelled", "unsupported"]

_SECTION_ARRAYS = ("y_edges", "z_edges", "codes", "status", "axial_velocity", "fractions", "donor_y", "donor_z",
                   "residence_time", "filled")


@dataclass
class Section:
    """One cross-section at a streamwise station. Arrays are stored ``[z, y]``."""

    name: str
    x: float
    y_edges: np.ndarray
    z_edges: np.ndarray
    codes: np.ndarray  # int32 stream codes, -1 where no valid trajectory exists
    status: np.ndarray | None = None
    axial_velocity: np.ndarray | None = None
    fractions: np.ndarray | None = None  # (n_streams, nz, ny) flux fractions, conservative transport
    donor_y: np.ndarray | None = None
    donor_z: np.ndarray | None = None
    residence_time: np.ndarray | None = None
    # True where the cell had no valid trajectory and took its neighbours' label (``status`` keeps the raw one)
    filled: np.ndarray | None = None
    grid_kind: Literal["cell_centred", "legacy_nodes"] = "cell_centred"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        if self.grid_kind == "legacy_nodes":
            return self.y_edges
        return 0.5 * (self.y_edges[1:] + self.y_edges[:-1])

    @property
    def z(self) -> np.ndarray:
        if self.grid_kind == "legacy_nodes":
            return self.z_edges
        return 0.5 * (self.z_edges[1:] + self.z_edges[:-1])

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (float(self.y_edges[0]), float(self.y_edges[-1]), float(self.z_edges[0]), float(self.z_edges[-1]))

    def cell_area(self) -> np.ndarray:
        if self.grid_kind == "legacy_nodes":
            # nodal grid: trapezoid weights (half weights on walls)
            wy = np.gradient(self.y_edges) if len(self.y_edges) > 1 else np.ones(1)
            wz = np.gradient(self.z_edges) if len(self.z_edges) > 1 else np.ones(1)
            wy = wy.copy(); wz = wz.copy()
            wy[[0, -1]] *= 0.5 if len(wy) > 2 else 1.0
            wz[[0, -1]] *= 0.5 if len(wz) > 2 else 1.0
            return np.outer(wz, wy)
        return np.outer(np.diff(self.z_edges), np.diff(self.y_edges))


@dataclass
class Pathlines:
    """Forward pathlines from the inlets to the outlet, used only for display (3D view animation).

    Line ``i`` is sampled at ``points[i, j]`` (``[x, y, z]``) with cumulative residence time
    ``time[i, j]``; samples a line does not have (before its port, after it ends) are NaN.
    ``status`` uses the transport codes; ``REACHED_PLANE`` (0) means the line left through the
    outlet. ``origin`` is 0 for the inlet pattern and ``k`` for the k-th port (1-based, project order
    of the port inlets).
    """

    points: np.ndarray  # (n, m, 3) float32
    time: np.ndarray  # (n, m) float32
    code: np.ndarray  # (n,) int32 stream code
    status: np.ndarray  # (n,) int8
    origin: np.ndarray  # (n,) int16
    outlet_x: float = 0.0

    ARRAYS = ("points", "time", "code", "status", "origin")

    def __post_init__(self) -> None:
        n = len(self.code)
        if self.points.shape[0] != n or self.points.ndim != 3 or self.points.shape[2] != 3:
            raise ValueError("pathline points must have shape (n, m, 3)")
        if self.time.shape != self.points.shape[:2]:
            raise ValueError("pathline times must have shape (n, m)")
        if self.status.shape != (n,) or self.origin.shape != (n,):
            raise ValueError("pathline status and origin must have shape (n,)")


@dataclass
class SimulationResult:
    status: Status
    message: str = ""
    backend: str = "new"
    solver: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    input_hash: str = ""
    streams: list[dict[str, Any]] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    validity: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    pathlines: Pathlines | None = None
    units: dict[str, str] = field(default_factory=lambda: {
        "length": "m", "flow_rate": "m3/s", "velocity": "m/s", "pressure": "Pa", "time": "s", "viscosity": "Pa s",
    })
    schema: str = RESULT_SCHEMA

    @property
    def ok(self) -> bool:
        return self.status == "success"

    @property
    def final(self) -> Section | None:
        return self.sections[-1] if self.sections else None

    def stream_by_code(self) -> dict[int, dict[str, Any]]:
        return {int(s["code"]): s for s in self.streams}

    # ---------------------------------------------------------------- persistence
    def meta(self) -> dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("sections", "pathlines")}
        data["pathlines"] = None if self.pathlines is None else {"outlet_x": self.pathlines.outlet_x}
        data["sections"] = [
            {
                "name": s.name,
                "x": s.x,
                "grid_kind": s.grid_kind,
                "diagnostics": s.diagnostics,
                "arrays": [a for a in _SECTION_ARRAYS if getattr(s, a) is not None],
            }
            for s in self.sections
        ]
        return _jsonable(data)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """The NPZ payload: JSON metadata plus named arrays."""
        arrays: dict[str, np.ndarray] = {"meta": np.array(json.dumps(self.meta()))}
        for index, section in enumerate(self.sections):
            for name in _SECTION_ARRAYS:
                value = getattr(section, name)
                if value is not None:
                    arrays[f"s{index}_{name}"] = np.asarray(value)
        if self.pathlines is not None:
            for name in Pathlines.ARRAYS:
                arrays[f"pathlines_{name}"] = np.asarray(getattr(self.pathlines, name))
        return arrays

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_arrays())
        return path

    @classmethod
    def from_arrays(cls, data) -> "SimulationResult":
        """Inverse of :meth:`to_arrays` (``data`` is a mapping such as an open NPZ file)."""
        meta = json.loads(str(data["meta"]))
        if meta.get("schema") != RESULT_SCHEMA:
            raise ValueError(f"Unsupported result schema {meta.get('schema')!r}")
        sections = []
        for index, info in enumerate(meta.pop("sections")):
            arrays = {name: data[f"s{index}_{name}"] for name in info["arrays"]}
            _check_section_arrays(arrays)
            sections.append(Section(name=info["name"], x=info["x"], grid_kind=info["grid_kind"],
                                    diagnostics=info["diagnostics"], **arrays))
        info = meta.pop("pathlines", None)
        pathlines = None
        if info is not None:
            pathlines = Pathlines(outlet_x=float(info["outlet_x"]),
                                  **{name: data[f"pathlines_{name}"] for name in Pathlines.ARRAYS})
        return cls(sections=sections, pathlines=pathlines, **meta)

    @classmethod
    def load(cls, path: str | Path) -> "SimulationResult":
        with np.load(path, allow_pickle=False) as data:
            return cls.from_arrays(data)


def _check_section_arrays(arrays: dict[str, np.ndarray]) -> None:
    codes = arrays["codes"]
    if codes.ndim != 2 or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("section codes must be a 2D integer array")
    for name in ("status", "axial_velocity", "donor_y", "donor_z", "residence_time", "filled"):
        if name in arrays and arrays[name].shape != codes.shape:
            raise ValueError(f"section array {name} has shape {arrays[name].shape}, expected {codes.shape}")
    if "fractions" in arrays and arrays["fractions"].shape[1:] != codes.shape:
        raise ValueError("section fractions must have shape (n_streams, nz, ny)")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def failed_result(status: Status, message: str, **kwargs: Any) -> SimulationResult:
    return SimulationResult(status=status, message=message, **kwargs)
