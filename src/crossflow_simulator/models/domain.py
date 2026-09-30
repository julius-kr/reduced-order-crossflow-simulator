"""Operation planning and Stokes-domain construction for feature groups.

Features (ports and pillars) are ordered by their physical streamwise
extent. Features closer than the interaction distance, or pillars sharing a
group ID, form one *group* that is solved in a single coupled domain; only
well-separated groups are composed independently. Each group domain spans
the features plus upstream/downstream margins in which the flow is assumed
fully developed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import geometry
from ..config import InletSpec, PillarSpec, ProjectConfig
from ..solvers.stokes import RectilinearGrid, WallOpening, graded_faces, snap_faces
from ..transport import OpeningShape

# Flow cells across the smaller channel dimension (the app shows these as Accuracy Low / Medium / High).
FIDELITY_CELLS = {"preview": 12, "medium": 18, "reference": 20}


@dataclass
class FlowGroup:
    index: int
    x_start: float
    x_end: float
    feature_start: float
    feature_end: float
    ports: list[InletSpec] = field(default_factory=list)
    pillars: list[PillarSpec] = field(default_factory=list)

    @property
    def label(self) -> str:
        names = [p.id for p in self.ports] + [p.id for p in self.pillars]
        return f"group {self.index + 1}: " + ", ".join(names)


@dataclass
class GroupDomain:
    group: FlowGroup
    grid: RectilinearGrid
    solid: np.ndarray
    openings: list[WallOpening]
    opening_shapes: list[OpeningShape]
    port_ids: list[str]
    resolution: dict
    warnings: list[str]

    def key(self) -> str:
        return geometry.stable_hash({
            "grid": self.grid.key(),
            "solid": geometry.stable_hash(self.solid.astype(np.uint8).tobytes().hex()),
            "openings": [
                {"wall": o.wall, "profile": geometry.stable_hash(np.round(o.profile, 14).tobytes().hex())}
                for o in self.openings
            ],
        })


def _feature_extent(cfg: ProjectConfig) -> list[tuple[float, float, str, object]]:
    features: list[tuple[float, float, str, object]] = []
    for port in cfg.port_inlets():
        radius = 0.5 * float(port.diameter or 0.0)
        features.append((port.station - radius, port.station + radius, "port", port))
    for pillar in cfg.active_pillars():
        fp = pillar.footprint()
        features.append((float(fp[:, 0].min()), float(fp[:, 0].max()), "pillar", pillar))
    return sorted(features, key=lambda f: (f[0], f[1]))


def channel_extent(cfg: ProjectConfig) -> float:
    """Streamwise extent shown in the views and traced by pathlines.

    The channel length when it is given; otherwise the last feature plus two channel sizes.
    """
    if cfg.channel.length:
        return float(cfg.channel.length)
    ends = [0.0] + [end for _, end, _, _ in _feature_extent(cfg)]
    return max(ends) + 2.0 * max(cfg.channel.width, cfg.channel.height)


def plan_groups(cfg: ProjectConfig) -> list[FlowGroup]:
    L = max(cfg.channel.width, cfg.channel.height)
    up = cfg.model.upstream_margin * L
    down = cfg.model.downstream_margin * L
    interaction = cfg.model.interaction_distance if cfg.model.interaction_distance is not None else up + down
    groups: list[FlowGroup] = []
    for start, end, kind, spec in _feature_extent(cfg):
        current = groups[-1] if groups else None
        joins = current is not None and start - current.feature_end < interaction
        if current is not None and kind == "pillar" and getattr(spec, "group", None):
            joins = joins or any(p.group == spec.group for p in current.pillars)  # type: ignore[attr-defined]
        if not joins:
            current = FlowGroup(len(groups), 0.0, 0.0, start, end)
            groups.append(current)
        current.feature_end = max(current.feature_end, end)
        (current.ports if kind == "port" else current.pillars).append(spec)  # type: ignore[arg-type]
    for group in groups:
        group.x_start = group.feature_start - up
        group.x_end = group.feature_end + down
    return groups


def section_solid(cfg: ProjectConfig, x: float, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Output cells (centres ``y``, ``z``) inside an enabled pillar at streamwise station ``x``, ``[z, y]``."""
    H = cfg.channel.height
    solid = np.zeros((len(z), len(y)), dtype=bool)
    for pillar in cfg.active_pillars():
        inside = geometry.points_in_polygon(np.full(len(y), float(x)), np.asarray(y, dtype=float),
                                            pillar.footprint())
        if not inside.any():
            continue
        low, high = (0.0, pillar.height) if pillar.mount == "bottom" else (H - pillar.height, H)
        solid |= ((z >= low) & (z <= high))[:, None] & inside[None, :]
    return solid


def flow_cells(cfg: ProjectConfig) -> tuple[int, int]:
    W, H = cfg.channel.width, cfg.channel.height
    n_small = FIDELITY_CELLS[cfg.model.fidelity]
    ny = cfg.grid.flow_cells_y or (n_small if W <= H else int(round(n_small * W / H)))
    nz = cfg.grid.flow_cells_z or (n_small if H <= W else int(round(n_small * H / W)))
    return max(ny, 4), max(nz, 4)


def build_domain(cfg: ProjectConfig, group: FlowGroup) -> GroupDomain:
    W, H = cfg.channel.width, cfg.channel.height
    ny, nz = flow_cells(cfg)
    yf = np.linspace(0.0, W, ny + 1)
    heights = []
    for pillar in group.pillars:
        heights.append(pillar.height if pillar.mount == "bottom" else H - pillar.height)
    zf = snap_faces(np.linspace(0.0, H, nz + 1), heights)
    fine = min(W / ny, H / nz)
    intervals = []
    for pillar in group.pillars:
        fp = pillar.footprint()
        intervals.append((float(fp[:, 0].min()), float(fp[:, 0].max())))
    for port in group.ports:
        r = 0.5 * float(port.diameter or 0.0)
        intervals.append((port.station - r, port.station + r))
    xf = graded_faces(group.x_start, group.x_end, target=2.0 * fine, refine=intervals, fine=fine)
    grid = RectilinearGrid(xf, yf, zf)
    xc, yc, zc = grid.centers
    solid = np.zeros(grid.shape, dtype=bool)
    warnings: list[str] = []
    resolution: dict = {"flow_cells": list(grid.shape), "fine_spacing_m": fine}
    min_gap_cells = math.inf
    for pillar in group.pillars:
        fp = pillar.footprint()
        column = geometry.raster_fraction(fp, xf, yf, subsamples=4) >= 0.5
        if pillar.mount == "bottom":
            layers = zc < pillar.height
            gap = H - pillar.height
        else:
            layers = zc > H - pillar.height
            gap = H - pillar.height
        solid |= column[:, :, None] & layers[None, None, :]
        cells_across = min(np.ptp(fp[:, 0]), np.ptp(fp[:, 1])) / fine
        resolution[f"{pillar.id}_footprint_cells"] = float(cells_across)
        if not column.any():
            warnings.append(f"Pillar {pillar.id!r} is smaller than one flow cell and disappears from the grid.")
        elif cells_across < 3:
            warnings.append(f"Pillar {pillar.id!r} spans only {cells_across:.1f} flow cells; refine the grid.")
        if gap > 0:
            min_gap_cells = min(min_gap_cells, gap / (H / nz))
    if math.isfinite(min_gap_cells):
        resolution["min_gap_cells"] = float(min_gap_cells)
        if min_gap_cells < 1.0:
            warnings.append("A pillar gap is thinner than one flow cell; the gap is unresolved.")
        elif min_gap_cells < 3.0:
            warnings.append(f"A pillar gap spans only {min_gap_cells:.1f} flow cells; refine for quantitative use.")

    openings: list[WallOpening] = []
    shapes: list[OpeningShape] = []
    for port in group.ports:
        opening, shape, faces = port_opening(grid, port)
        openings.append(opening)
        shapes.append(shape)
        resolution[f"{port.id}_opening_faces"] = int(faces)
        if faces < 12:
            warnings.append(f"Port {port.id!r} is covered by only {faces} wall faces; refine the grid.")
        wall_axis = 2 if port.wall in ("top", "bottom") else 1
        side = 0 if port.wall in ("bottom", "left") else -1
        adjacent = np.take(solid, [side], axis=wall_axis).squeeze(wall_axis)
        if np.any(adjacent & (opening.profile != 0)):
            raise ValueError(f"Port {port.id!r} opens into a solid obstacle.")
    return GroupDomain(group, grid, solid, openings, shapes, [p.id for p in group.ports], resolution, warnings)


def port_opening(grid: RectilinearGrid, port: InletSpec, subsamples: int = 5) -> tuple[WallOpening, OpeningShape, int]:
    """Poiseuille inflow through a circular port, as unit-flux face velocities on the wall."""
    radius = 0.5 * float(port.diameter)
    xf = grid.faces[0]
    tf = grid.faces[1] if port.wall in ("top", "bottom") else grid.faces[2]
    offsets = (np.arange(subsamples) + 0.5) / subsamples
    xs = xf[:-1, None] + np.diff(xf)[:, None] * offsets[None, :]
    ts = tf[:-1, None] + np.diff(tf)[:, None] * offsets[None, :]
    r2 = ((xs[:, None, :, None] - port.station) ** 2 + (ts[None, :, None, :] - port.position) ** 2) / radius**2
    speed = np.where(r2 < 1.0, 1.0 - r2, 0.0)
    area = np.diff(xf)[:, None] * np.diff(tf)[None, :]
    flux = speed.mean(axis=(2, 3)) * area
    total = flux.sum()
    if total <= 0:
        raise ValueError(f"Port {port.id!r} does not overlap any wall face.")
    profile = flux / total / area  # unit volumetric flow rate
    shape = OpeningShape(port.wall, (port.station, port.position), radius)
    return WallOpening(port.wall, profile, port.id), shape, int(np.count_nonzero(flux))
