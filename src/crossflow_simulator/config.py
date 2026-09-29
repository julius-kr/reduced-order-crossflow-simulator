"""Typed, versioned project configuration (SI units internally).

Conventions used everywhere in the new implementation:

* ``x`` is streamwise, ``y`` lateral (channel width ``W``), ``z`` vertical
  (channel height ``H``); the channel occupies ``0 <= y <= W``, ``0 <= z <= H``.
* Cross-section fields are stored as ``[z, y]``; top-view fields as ``[y, x]``;
  3D staggered fields as ``[x, y, z]``.
* Project JSON stores SI numbers under unit-suffixed keys (``width_m``). String
  quantities such as ``"300 um"`` are accepted on input and converted to SI.

Inlets are either part of a prescribed cross-sectional *pattern* entering at
``x = 0`` or physical *ports* at explicit streamwise stations. Nothing is
inferred from coordinates.
"""
from __future__ import annotations

import copy
import json
import math
import types
import typing
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import geometry
from .units import KEY_SUFFIX, parse_quantity

SCHEMA = "crossflow-project/1"


def _q(kind: str, default: Any = MISSING) -> Any:
    """Dataclass field carrying a physical quantity of the given kind."""
    if default is MISSING:
        return field(metadata={"kind": kind})
    return field(default=default, metadata={"kind": kind})


# --------------------------------------------------------------------------- contracts


@dataclass
class ChannelGeometry:
    width: float = _q("length")
    height: float = _q("length")
    length: float | None = _q("length", None)


@dataclass
class FluidProperties:
    id: str
    viscosity: float = _q("viscosity", 1e-3)
    density: float = _q("density", 1000.0)
    diffusivity: float | None = _q("diffusivity", None)
    temperature: float = _q("temperature", 293.15)
    model: Literal["newtonian"] = "newtonian"
    name: str = ""
    color: str | None = None


@dataclass
class InletSpec:
    id: str
    material: str
    placement: Literal["pattern", "port"] = "pattern"
    flow_rate: float | None = _q("flow_rate", None)
    pressure: float | None = _q("pressure", None)
    tubing_resistance: float | None = _q("resistance", None)
    # Pattern placement (cross-sectional patch entering at x = 0).
    pattern_row: int = 0
    pattern_col: int = 0
    # Port placement (opening in a channel wall at a streamwise station).
    station: float = _q("length", 0.0)
    wall: Literal["top", "bottom", "left", "right"] = "top"
    position: float = _q("length", 0.0)  # y for top/bottom walls, z for side walls
    diameter: float | None = _q("length", None)
    shape: Literal["circle"] = "circle"
    orientation: Literal["normal"] = "normal"
    enabled: bool = True


@dataclass
class InitialPattern:
    layout: Literal["single", "layers_y", "layers_z", "grid"] = "single"
    rows: int = 1
    cols: int = 1
    partition: Literal["auto", "aligned", "staggered_rows", "staggered_columns"] = "auto"


@dataclass
class PillarSpec:
    id: str
    vertices: list[tuple[float, float]] = _q("length")  # local footprint
    height: float = _q("length")
    x: float = _q("length", 0.0)
    y: float = _q("length", 0.0)
    rotation: float = _q("angle", 0.0)
    mirror_y: bool = False
    mount: Literal["bottom", "top"] = "bottom"
    group: str | None = None
    enabled: bool = True

    def footprint(self) -> np.ndarray:
        """Absolute x-y footprint vertices, counter-clockwise."""
        local = geometry.as_vertices(self.vertices)
        return geometry.ensure_ccw(geometry.transform(local, self.x, self.y, self.rotation, self.mirror_y))


@dataclass
class OperatingConditions:
    drive: Literal["flow_rate", "pressure"] = "flow_rate"
    outlet_pressure: float = _q("pressure", 0.0)
    diffusion: bool = False


@dataclass
class GridSpec:
    output_ny: int = 160
    output_nz: int = 0  # 0: chosen from the physical aspect ratio
    flow_cells_y: int = 0  # 0: fidelity preset
    flow_cells_z: int = 0  # 0: from aspect ratio
    map_stride: int = 4
    lubrication_cells_x: int = 160
    lubrication_cells_y: int = 64
    dtype: Literal["float64"] = "float64"


@dataclass
class ModelSettings:
    fidelity: Literal["preview", "medium", "reference"] = "preview"
    junction_model: Literal["stokes_map", "ideal_layers"] = "stokes_map"
    pillar_model: Literal["stokes_map"] = "stokes_map"
    viscosity_coupling: Literal["auto", "off", "iterate"] = "auto"
    transport: Literal["categorical", "conservative"] = "categorical"
    integrator: Literal["rk2", "rk4"] = "rk2"  # rk2: adaptive midpoint; rk4: fixed step
    step_fraction: float = 0.5
    tracing_tolerance: float = 1e-3  # local position error per step / smallest flow cell (rk2)
    interaction_distance: float | None = _q("length", None)
    upstream_margin: float = 0.75
    downstream_margin: float = 1.0
    linear_solver: Literal["auto", "direct", "minres"] = "auto"
    tolerance: float = 1e-12  # preconditioned MINRES tolerance; the true residual is checked separately
    max_coupling_iterations: int = 8
    convergence_check: bool = False


@dataclass
class LegacySettings:
    """How the legacy adapter assembles the original operator sequence."""

    velocity_profile: Literal["rectangular_2d", "parabolic_1d"] = "rectangular_2d"
    n_terms: int = 100
    serial_curvature: bool = True
    curvature_strength: float = 0.18
    curvature_exponent: float = 1.0
    curvature_gamma: float = 1.0
    pillar_model: Literal["lubrication_divergence", "partial_height_surrogate"] = "lubrication_divergence"
    lubrication_nx: int = 41
    lubrication_ny: int = 31
    secondary_strength: float = 0.08
    secondary_steps: int = 0  # 0: chosen from the legacy grid so labels can move (see backends/legacy.py)
    secondary_pseudo_time: float = 0.12
    secondary_radius_y: float = 0.30
    secondary_radius_z: float = 0.30
    center_z_factor: float = 0.55
    grid_ny: int = 0  # 0: same as the new solver's output grid
    grid_nz: int = 0


@dataclass
class CFDSettings:
    """CFD reference settings (see ``cfd.py``). Not part of the new solver's input hash."""

    cells: int = 24  # grid cells across the smaller channel dimension on the reference grid
    convergence_check: bool = True  # also solve a coarser grid (2/3 of the cells) to estimate mesh uncertainty
    inertia: bool = True  # steady Navier-Stokes (Picard iterations); off: creeping flow
    buffer: float = 2.0  # inflow and outflow buffers, in channel sizes max(W, H)


@dataclass
class DisplaySettings:
    length_unit: str = "um"
    flow_unit: str = "uL/min"
    viscosity_unit: str = "mPa s"
    pressure_unit: str = "mbar"
    color_by: Literal["material", "inlet"] = "material"


@dataclass
class ProjectConfig:
    channel: ChannelGeometry
    materials: list[FluidProperties]
    inlets: list[InletSpec]
    pattern: InitialPattern = field(default_factory=InitialPattern)
    pillars: list[PillarSpec] = field(default_factory=list)
    operating: OperatingConditions = field(default_factory=OperatingConditions)
    grid: GridSpec = field(default_factory=GridSpec)
    model: ModelSettings = field(default_factory=ModelSettings)
    legacy: LegacySettings = field(default_factory=LegacySettings)
    cfd: CFDSettings = field(default_factory=CFDSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    name: str = "untitled"
    notes: str = ""
    schema: str = SCHEMA

    # ------------------------------------------------------------------ helpers
    def material(self, material_id: str) -> FluidProperties:
        for m in self.materials:
            if m.id == material_id:
                return m
        raise KeyError(material_id)

    def active_inlets(self) -> list[InletSpec]:
        return [i for i in self.inlets if i.enabled]

    def stream_codes(self) -> dict[str, int]:
        """Stable integer code per inlet (1-based position in the inlet list)."""
        return {inlet.id: index + 1 for index, inlet in enumerate(self.inlets)}

    def pattern_inlets(self) -> list[InletSpec]:
        return [i for i in self.active_inlets() if i.placement == "pattern"]

    def port_inlets(self) -> list[InletSpec]:
        ports = [i for i in self.active_inlets() if i.placement == "port"]
        return sorted(ports, key=lambda i: (i.station, self.inlets.index(i)))

    def active_pillars(self) -> list[PillarSpec]:
        return [p for p in self.pillars if p.enabled]

    def to_dict(self) -> dict[str, Any]:
        return _to_json(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    def input_hash(self) -> str:
        """Hash of every input that can affect a new-solver result (display and CFD settings excluded)."""
        data = self.to_dict()
        data.pop("display", None)
        data.pop("cfd", None)
        data.pop("name", None)
        data.pop("notes", None)
        return geometry.stable_hash(data)

    def copy(self) -> "ProjectConfig":
        return copy.deepcopy(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        schema = data.get("schema", SCHEMA)
        if schema != SCHEMA:
            raise ConfigError([Issue("error", "schema", f"Unsupported project schema {schema!r}; expected {SCHEMA!r}.")])
        return _from_json(cls, data, "")

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- serialization


def _json_key(f: Any) -> str:
    kind = f.metadata.get("kind")
    return f"{f.name}_{KEY_SUFFIX[kind]}" if kind else f.name


def _to_json(obj: Any) -> Any:
    if is_dataclass(obj):
        return {_json_key(f): _to_json(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _strip_optional(tp: Any) -> tuple[Any, bool]:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return (args[0] if len(args) == 1 else tp), True
    return tp, False


def _convert(tp: Any, value: Any, kind: str | None, path: str) -> Any:
    tp, optional = _strip_optional(tp)
    if value is None:
        if optional:
            return None
        raise ConfigError([Issue("error", path, "Value is required.")])
    origin = typing.get_origin(tp)
    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError([Issue("error", path, "Expected an object.")])
        return _from_json(tp, value, path)
    if origin is list:
        (item_type,) = typing.get_args(tp)
        if not isinstance(value, list):
            raise ConfigError([Issue("error", path, "Expected a list.")])
        return [_convert(item_type, v, kind, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is tuple:
        item_types = typing.get_args(tp)
        if not isinstance(value, (list, tuple)) or len(value) != len(item_types):
            raise ConfigError([Issue("error", path, f"Expected {len(item_types)} values.")])
        return tuple(_convert(t, v, kind, f"{path}[{i}]") for i, (t, v) in enumerate(zip(item_types, value)))
    if origin is Literal:
        allowed = typing.get_args(tp)
        if value not in allowed:
            raise ConfigError([Issue("error", path, f"{value!r} is not one of {list(allowed)}.")])
        return value
    if tp is float:
        try:
            return parse_quantity(value, kind) if kind else float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError([Issue("error", path, str(exc))]) from None
    if tp is int:
        if isinstance(value, bool) or not float(value).is_integer():
            raise ConfigError([Issue("error", path, f"Expected an integer, got {value!r}.")])
        return int(value)
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError([Issue("error", path, f"Expected true/false, got {value!r}.")])
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError([Issue("error", path, f"Expected a string, got {value!r}.")])
        return value
    return value


def _from_json(cls: type, data: dict[str, Any], path: str) -> Any:
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known: set[str] = set()
    for f in fields(cls):
        kind = f.metadata.get("kind")
        key = _json_key(f)
        known.update({key, f.name})
        if key in data:
            raw = data[key]
        elif f.name in data:
            raw = data[f.name]
        else:
            if f.default is MISSING and f.default_factory is MISSING:
                raise ConfigError([Issue("error", f"{path}.{key}".lstrip("."), "Missing required field.")])
            continue
        kwargs[f.name] = _convert(hints[f.name], raw, kind, f"{path}.{key}".lstrip("."))
    unknown = set(data) - known
    if unknown:
        raise ConfigError([Issue("error", path or "<root>", f"Unknown field(s): {sorted(unknown)}")])
    return cls(**kwargs)


# --------------------------------------------------------------------------- validation


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning" | "info"
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.path}: {self.message}"


class ConfigError(ValueError):
    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("; ".join(str(i) for i in issues))


def validate_config(cfg: ProjectConfig) -> list[Issue]:
    """Validate geometry, topology, and operating conditions. Errors block a solve."""
    issues: list[Issue] = []
    add = lambda level, path, message: issues.append(Issue(level, path, message))  # noqa: E731

    W, H = cfg.channel.width, cfg.channel.height
    for name, value in (("channel.width_m", W), ("channel.height_m", H)):
        if not (math.isfinite(value) and value > 0):
            add("error", name, "Must be a positive finite length.")
    if issues:
        return issues
    if cfg.channel.length is not None and cfg.channel.length <= 0:
        add("error", "channel.length_m", "Must be positive when given.")

    # Materials
    material_ids = [m.id for m in cfg.materials]
    if not material_ids:
        add("error", "materials", "At least one material is required.")
    for index, m in enumerate(cfg.materials):
        p = f"materials[{index}]"
        if material_ids.count(m.id) > 1:
            add("error", f"{p}.id", f"Duplicate material id {m.id!r}.")
        if not (m.viscosity > 0 and math.isfinite(m.viscosity)):
            add("error", f"{p}.viscosity_Pa_s", "Viscosity must be positive.")
        if not (m.density > 0 and math.isfinite(m.density)):
            add("error", f"{p}.density_kg_m3", "Density must be positive.")
        if m.diffusivity is not None and m.diffusivity < 0:
            add("error", f"{p}.diffusivity_m2_s", "Diffusivity must be non-negative.")

    # Inlets and driving conditions
    inlet_ids = [i.id for i in cfg.inlets]
    active = cfg.active_inlets()
    if not active:
        add("error", "inlets", "At least one enabled inlet is required.")
    for index, inlet in enumerate(cfg.inlets):
        p = f"inlets[{index}]"
        if inlet_ids.count(inlet.id) > 1:
            add("error", f"{p}.id", f"Duplicate inlet id {inlet.id!r}.")
        if inlet.material not in material_ids:
            add("error", f"{p}.material", f"Unknown material {inlet.material!r}.")
        if not inlet.enabled:
            continue
        if cfg.operating.drive == "flow_rate":
            if inlet.pressure is not None:
                add("error", f"{p}.pressure_Pa", "Over-specified: flow-rate drive does not accept an inlet pressure.")
            if inlet.flow_rate is None or not inlet.flow_rate > 0:
                add("error", f"{p}.flow_rate_m3_s", "Flow-rate drive requires a positive flow rate.")
        else:
            if inlet.flow_rate is not None:
                add("error", f"{p}.flow_rate_m3_s", "Over-specified: pressure drive computes flow rates; remove the flow rate.")
            if inlet.pressure is None:
                add("error", f"{p}.pressure_Pa", "Pressure drive requires an inlet pressure.")
            if inlet.tubing_resistance is None or not inlet.tubing_resistance > 0:
                add("error", f"{p}.tubing_resistance_Pa_s_m3", "Pressure drive requires a positive tubing/port resistance.")
        if inlet.placement == "port":
            _validate_port(cfg, inlet, p, add)

    _validate_pattern(cfg, add)
    _validate_ports_together(cfg, add)
    _validate_pillars(cfg, add)

    if cfg.operating.diffusion:
        add("error", "operating.diffusion", "Scalar diffusion is not implemented; results would be misleading.")
    if cfg.model.step_fraction <= 0 or cfg.model.step_fraction > 1:
        add("error", "model.step_fraction", "Must be in (0, 1].")
    if cfg.grid.output_ny < 8:
        add("error", "grid.output_ny", "Output grid needs at least 8 cells across the width.")
    if not 8 <= cfg.cfd.cells <= 64:
        add("error", "cfd.cells", "CFD reference grid needs 8 to 64 cells across the smaller channel dimension.")
    if not cfg.cfd.buffer >= 0.5:
        add("error", "cfd.buffer", "CFD inflow/outflow buffers must be at least half a channel size.")
    return issues


def _validate_port(cfg: ProjectConfig, inlet: InletSpec, p: str, add: Any) -> None:
    W, H = cfg.channel.width, cfg.channel.height
    if inlet.diameter is None or not inlet.diameter > 0:
        add("error", f"{p}.diameter_m", "A port needs a positive diameter.")
        return
    if inlet.station < 0:
        add("error", f"{p}.station_m", "Port station must be at x >= 0.")
    if cfg.channel.length is not None and inlet.station > cfg.channel.length:
        add("error", f"{p}.station_m", "Port station lies beyond the channel length.")
    span = W if inlet.wall in ("top", "bottom") else H
    radius = 0.5 * inlet.diameter
    if inlet.position - radius < -1e-12 * span or inlet.position + radius > span * (1 + 1e-12):
        add("error", f"{p}.position_m", f"Port opening extends beyond the {inlet.wall} wall (span {span:.3g} m).")
    if inlet.station - radius < 0:
        add("error", f"{p}.station_m", "Port opening extends upstream of x = 0.")


def _validate_pattern(cfg: ProjectConfig, add: Any) -> None:
    pattern = cfg.pattern
    inlets = cfg.pattern_inlets()
    if not inlets:
        add("error", "pattern", "At least one pattern inlet must fill the channel at x = 0.")
        return
    cells = [(i.pattern_row, i.pattern_col) for i in inlets]
    if pattern.layout == "single":
        if len(inlets) != 1:
            add("error", "pattern.layout", "Layout 'single' requires exactly one pattern inlet.")
        return
    if pattern.layout == "layers_y":
        expected = {(0, c) for c in range(len(inlets))}
        if set((0, c) for _, c in cells) != expected or any(r != 0 for r, _ in cells):
            add("error", "pattern", "layers_y needs pattern_col = 0..n-1 (left to right) and pattern_row = 0.")
    elif pattern.layout == "layers_z":
        expected = {(r, 0) for r in range(len(inlets))}
        if set((r, 0) for r, _ in cells) != expected or any(c != 0 for _, c in cells):
            add("error", "pattern", "layers_z needs pattern_row = 0..n-1 (bottom to top) and pattern_col = 0.")
    elif pattern.layout == "grid":
        if pattern.rows < 1 or pattern.cols < 1:
            add("error", "pattern", "Grid needs positive rows and cols.")
            return
        expected = {(r, c) for r in range(pattern.rows) for c in range(pattern.cols)}
        if sorted(cells) != sorted(expected):
            add("error", "pattern", f"Grid {pattern.rows}x{pattern.cols} needs exactly one inlet per cell (row, col).")


def _validate_ports_together(cfg: ProjectConfig, add: Any) -> None:
    ports = cfg.port_inlets()
    for a_index, a in enumerate(ports):
        for b in ports[a_index + 1:]:
            if a.wall != b.wall or a.diameter is None or b.diameter is None:
                continue
            distance = math.hypot(a.station - b.station, a.position - b.position)
            if distance < 0.5 * (a.diameter + b.diameter):
                add("error", f"inlets[{cfg.inlets.index(b)}]", f"Port {b.id!r} overlaps port {a.id!r} on the {a.wall} wall.")


def _validate_pillars(cfg: ProjectConfig, add: Any) -> None:
    W, H = cfg.channel.width, cfg.channel.height
    pillars = cfg.active_pillars()
    footprints: dict[str, np.ndarray] = {}
    for index, pillar in enumerate(cfg.pillars):
        p = f"pillars[{index}]"
        if not pillar.enabled:
            continue
        problems = geometry.validate_polygon(pillar.vertices)
        for issue in problems:
            add(issue.level, f"{p}.vertices_m", issue.message)
        if any(i.level == "error" for i in problems):
            continue
        if not (pillar.height > 0 and math.isfinite(pillar.height)):
            add("error", f"{p}.height_m", "Pillar height must be positive.")
            continue
        if pillar.height > H * (1 + 1e-12):
            add("error", f"{p}.height_m", "Pillar is taller than the channel.")
        elif pillar.height >= H * (1 - 1e-9):
            add("info", f"{p}.height_m", "Full-height pillar: represented by obstacle cells in the Stokes model.")
        fp = pillar.footprint()
        footprints[pillar.id] = fp
        if fp[:, 1].min() < -1e-12 * W or fp[:, 1].max() > W * (1 + 1e-12):
            add("error", f"{p}.vertices_m", "Footprint extends beyond the side walls.")
        if fp[:, 0].min() < 0:
            add("error", f"{p}.vertices_m", "Footprint extends upstream of x = 0.")
        if cfg.channel.length is not None and fp[:, 0].max() > cfg.channel.length:
            add("error", f"{p}.vertices_m", "Footprint extends beyond the channel length.")
        if fp[:, 1].min() <= 1e-9 * W or fp[:, 1].max() >= W * (1 - 1e-9):
            add("info", f"{p}.vertices_m", "Footprint touches a side wall.")
        for port in cfg.port_inlets():
            if port.wall != pillar.mount or port.diameter is None:
                continue
            centre = np.array([port.station, port.position])
            inside = geometry.points_in_polygon(np.array([centre[0]]), np.array([centre[1]]), fp)[0]
            gap = 0.0 if inside else min(
                geometry._point_segment_distance(centre, fp[i], fp[(i + 1) % len(fp)]) for i in range(len(fp))
            )
            if inside or gap < 0.5 * port.diameter:
                add("error", f"{p}", f"Pillar {pillar.id!r} covers port {port.id!r} on the {port.wall} wall.")
    ids = [p.id for p in cfg.pillars]
    for index, pillar in enumerate(cfg.pillars):
        if ids.count(pillar.id) > 1:
            add("error", f"pillars[{index}].id", f"Duplicate pillar id {pillar.id!r}.")
    for a_index, a in enumerate(pillars):
        for b in pillars[a_index + 1:]:
            if a.id not in footprints or b.id not in footprints:
                continue
            distance = geometry.polygon_distance(footprints[a.id], footprints[b.id])
            if distance <= 0 and (a.group is None or a.group != b.group):
                add("error", "pillars", f"Pillars {a.id!r} and {b.id!r} touch or overlap. Give them the same group id "
                    "to model them as one composite solid, or separate them.")
    _check_blockage(cfg, footprints, add)


def _check_blockage(cfg: ProjectConfig, footprints: dict[str, np.ndarray], add: Any) -> None:
    """Reject geometries that close the whole cross-section at some station."""
    W, H = cfg.channel.width, cfg.channel.height
    full = [p for p in cfg.active_pillars() if p.id in footprints and p.height >= H * (1 - 1e-9)]
    if not full:
        return
    xs = np.unique(np.concatenate([footprints[p.id][:, 0] for p in full]))
    samples = np.linspace(0.0, W, 257)
    for x in np.concatenate([xs, 0.5 * (xs[1:] + xs[:-1])]):
        blocked = np.zeros_like(samples, dtype=bool)
        for p in full:
            blocked |= geometry.points_in_polygon(np.full_like(samples, x), samples, footprints[p.id])
        if blocked[1:-1].all():
            add("error", "pillars", f"Full-height pillars close the channel completely near x = {x:.4g} m.")
            return


def ensure_valid(cfg: ProjectConfig) -> list[Issue]:
    """Raise :class:`ConfigError` on errors; return warnings/info otherwise."""
    issues = validate_config(cfg)
    errors = [i for i in issues if i.level == "error"]
    if errors:
        raise ConfigError(errors)
    return issues
