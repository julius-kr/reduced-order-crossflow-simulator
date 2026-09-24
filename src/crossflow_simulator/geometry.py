"""Polygon validation, placement, rasterization, and geometry hashing.

Footprints are x-y polygons in physical units (x streamwise, y lateral).
Rasterization support is deliberately separate from physical validity: a
polygon that rasterizes is not thereby a validated prediction geometry.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class GeometryIssue:
    level: str  # "error" | "warning" | "info"
    code: str
    message: str


def as_vertices(vertices: Iterable[Sequence[float]]) -> np.ndarray:
    array = np.asarray(list(vertices), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("vertices must be a sequence of (x, y) pairs")
    if len(array) > 1 and np.allclose(array[0], array[-1]):
        array = array[:-1]
    return array


def signed_area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    return vertices if signed_area(vertices) >= 0 else vertices[::-1].copy()


def _orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def _on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray, tol: float) -> bool:
    return (
        min(p[0], r[0]) - tol <= q[0] <= max(p[0], r[0]) + tol
        and min(p[1], r[1]) - tol <= q[1] <= max(p[1], r[1]) + tol
    )


def segments_intersect(p1, p2, q1, q2, tol: float = 0.0) -> bool:
    """True when closed segments p1-p2 and q1-q2 touch or cross."""
    p1, p2, q1, q2 = (np.asarray(v, dtype=float) for v in (p1, p2, q1, q2))
    scale = max(np.ptp(np.array([p1, p2, q1, q2]), axis=0).max(), 1e-300)
    eps = tol * scale
    o1, o2 = _orientation(p1, p2, q1), _orientation(p1, p2, q2)
    o3, o4 = _orientation(q1, q2, p1), _orientation(q1, q2, p2)
    area_eps = eps * scale
    if ((o1 > area_eps and o2 < -area_eps) or (o1 < -area_eps and o2 > area_eps)) and (
        (o3 > area_eps and o4 < -area_eps) or (o3 < -area_eps and o4 > area_eps)
    ):
        return True
    return (
        (abs(o1) <= area_eps and _on_segment(p1, q1, p2, eps))
        or (abs(o2) <= area_eps and _on_segment(p1, q2, p2, eps))
        or (abs(o3) <= area_eps and _on_segment(q1, p1, q2, eps))
        or (abs(o4) <= area_eps and _on_segment(q1, p2, q2, eps))
    )


def is_simple(vertices: np.ndarray, tol: float = 1e-12) -> bool:
    n = len(vertices)
    for i in range(n):
        a1, a2 = vertices[i], vertices[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue
            if segments_intersect(a1, a2, vertices[j], vertices[(j + 1) % n], tol):
                return False
    return True


def validate_polygon(vertices: Iterable[Sequence[float]], tol: float = 1e-9) -> list[GeometryIssue]:
    """Geometric checks independent of the channel: finiteness, degeneracy, simplicity."""
    try:
        v = as_vertices(vertices)
    except ValueError as exc:
        return [GeometryIssue("error", "shape", str(exc))]
    issues: list[GeometryIssue] = []
    if len(v) < 3:
        return [GeometryIssue("error", "too_few_vertices", "A footprint needs at least 3 distinct vertices.")]
    if not np.all(np.isfinite(v)):
        return [GeometryIssue("error", "non_finite", "Footprint coordinates must be finite.")]
    scale = float(np.ptp(v, axis=0).max())
    if scale <= 0:
        return [GeometryIssue("error", "zero_area", "Footprint has zero extent.")]
    edge_lengths = np.linalg.norm(np.roll(v, -1, axis=0) - v, axis=1)
    if np.any(edge_lengths <= tol * scale):
        issues.append(GeometryIssue("error", "repeated_point", "Footprint contains repeated consecutive vertices."))
        return issues
    collinear = abs(signed_area(v)) <= tol * scale * scale and np.all(
        [abs(_orientation(v[0], v[1], w)) <= tol * scale * scale for w in v[2:]])
    if collinear:
        issues.append(GeometryIssue("error", "zero_area", "Footprint area is zero (collinear vertices)."))
    elif not is_simple(v, tol):
        issues.append(GeometryIssue("error", "self_intersection", "Footprint edges intersect; the polygon is not simple."))
    elif abs(signed_area(v)) <= tol * scale * scale:
        issues.append(GeometryIssue("error", "zero_area", "Footprint area is zero."))
    return issues


def points_in_polygon(px: np.ndarray, py: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test, vectorized over points."""
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    inside = np.zeros(np.broadcast(px, py).shape, dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        crosses = (y0 > py) != (y1 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (px < x_cross)
        x0, y0 = x1, y1
    return inside


def polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum distance between two polygons (zero when they touch or overlap)."""
    if points_in_polygon(a[:, 0], a[:, 1], b).any() or points_in_polygon(b[:, 0], b[:, 1], a).any():
        return 0.0
    best = math.inf
    for p, q in ((a, b), (b, a)):
        for i in range(len(p)):
            for j in range(len(q)):
                if segments_intersect(p[i], p[(i + 1) % len(p)], q[j], q[(j + 1) % len(q)]):
                    return 0.0
                best = min(best, _point_segment_distance(p[i], q[j], q[(j + 1) % len(q)]))
    return best


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom == 0 else min(1.0, max(0.0, float(np.dot(p - a, ab)) / denom))
    return float(np.linalg.norm(p - (a + t * ab)))


def transform(
    local: np.ndarray, x: float, y: float, rotation_deg: float = 0.0, mirror_y: bool = False
) -> np.ndarray:
    """Place local footprint vertices: optional mirror in y, rotation about the local origin, translation."""
    v = np.array(local, dtype=float)
    if mirror_y:
        v[:, 1] = -v[:, 1]
        v = v[::-1]
    angle = math.radians(rotation_deg)
    c, s = math.cos(angle), math.sin(angle)
    rotated = v @ np.array([[c, s], [-s, c]])
    return rotated + np.array([x, y])


def directed_edges(vertices: np.ndarray) -> list[dict[str, object]]:
    """Edges of a counter-clockwise polygon with outward unit normals and lengths."""
    v = ensure_ccw(as_vertices(vertices))
    edges = []
    for i in range(len(v)):
        start, end = v[i], v[(i + 1) % len(v)]
        d = end - start
        length = float(np.hypot(*d))
        normal = np.array([d[1], -d[0]]) / length
        edges.append({"start": start.tolist(), "end": end.tolist(), "normal": normal.tolist(), "length": length})
    return edges


def raster_fraction(
    vertices: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, subsamples: int = 4
) -> np.ndarray:
    """Fraction of each (x, y) cell covered by the polygon, shape (nx, ny), by supersampling."""
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    fraction = np.zeros((nx, ny))
    offsets = (np.arange(subsamples) + 0.5) / subsamples
    xs = x_edges[:-1, None] + np.diff(x_edges)[:, None] * offsets[None, :]
    ys = y_edges[:-1, None] + np.diff(y_edges)[:, None] * offsets[None, :]
    xmin, ymin = vertices.min(axis=0)
    xmax, ymax = vertices.max(axis=0)
    ix = np.nonzero((x_edges[1:] >= xmin) & (x_edges[:-1] <= xmax))[0]
    iy = np.nonzero((y_edges[1:] >= ymin) & (y_edges[:-1] <= ymax))[0]
    if len(ix) == 0 or len(iy) == 0:
        return fraction
    px = xs[ix][:, None, :, None]
    py = ys[iy][None, :, None, :]
    inside = points_in_polygon(np.broadcast_to(px, (len(ix), len(iy), subsamples, subsamples)),
                               np.broadcast_to(py, (len(ix), len(iy), subsamples, subsamples)), vertices)
    fraction[np.ix_(ix, iy)] = inside.mean(axis=(2, 3))
    return fraction


def stable_hash(obj: object, digits: int = 12) -> str:
    """SHA-256 of canonical JSON with floats rounded to ``digits`` significant digits."""

    def canon(value: object) -> object:
        if isinstance(value, float):
            return float(f"{value:.{digits}g}")
        if isinstance(value, np.ndarray):
            return canon(value.tolist())
        if isinstance(value, np.generic):
            return canon(value.item())
        if isinstance(value, dict):
            return {str(k): canon(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, (list, tuple)):
            return [canon(v) for v in value]
        return value

    payload = json.dumps(canon(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
