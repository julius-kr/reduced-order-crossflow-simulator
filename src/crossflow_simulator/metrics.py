"""Cross-section metrics: flux fractions, IoU, interface distance, curvature.

All distances and curvatures use physical coordinates. IoU alone is
misleading for thin regions, so interface distance, centroid displacement,
and flux fractions are reported alongside it.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .results import Section


def flux_fractions(section: Section, codes: list[int] | None = None) -> dict[int, float]:
    """Axial flux fraction carried by each stream code (invalid cells excluded and reported separately)."""
    if section.fractions is not None and codes is not None and section.axial_velocity is not None:
        weights = section.axial_velocity * section.cell_area()
        total = float(np.sum(weights))
        return {int(c): float(np.sum(section.fractions[i] * weights)) / total for i, c in enumerate(codes)}
    velocity = section.axial_velocity if section.axial_velocity is not None else np.ones(section.codes.shape)
    weights = velocity * section.cell_area()
    total = float(np.sum(weights))
    present = np.unique(section.codes) if codes is None else np.asarray(codes)
    out = {int(c): float(np.sum(weights[section.codes == c])) / total for c in present if c >= 0}
    out[-1] = float(np.sum(weights[section.codes < 0])) / total
    return out


def area_fractions(section: Section) -> dict[int, float]:
    area = section.cell_area()
    total = float(area.sum())
    return {int(c): float(area[section.codes == c].sum()) / total for c in np.unique(section.codes)}


def regrid_nearest(section: Section, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resampling of categorical codes onto physical (y, z) points (display/metrics only)."""
    sy, sz = section.y, section.z
    j = np.clip(np.searchsorted(0.5 * (sy[1:] + sy[:-1]), y), 0, len(sy) - 1)
    k = np.clip(np.searchsorted(0.5 * (sz[1:] + sz[:-1]), z), 0, len(sz) - 1)
    return section.codes[np.ix_(k, j)]


def iou(a: np.ndarray, b: np.ndarray, code: int) -> float:
    ma, mb = a == code, b == code
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else float("nan")


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, border_value=1)


def interface_distance(a: np.ndarray, b: np.ndarray, code: int, spacing: tuple[float, float]) -> float:
    """Mean symmetric distance between the boundaries of ``code`` regions (physical units)."""
    ba, bb = _boundary(a == code), _boundary(b == code)
    if not ba.any() or not bb.any():
        return float("nan")
    da = ndimage.distance_transform_edt(~bb, sampling=spacing)
    db = ndimage.distance_transform_edt(~ba, sampling=spacing)
    return float(0.5 * (da[ba].mean() + db[bb].mean()))


def centroid(codes: np.ndarray, code: int, y: np.ndarray, z: np.ndarray, weights: np.ndarray | None = None):
    mask = codes == code
    w = mask.astype(float) if weights is None else np.where(mask, weights, 0.0)
    total = w.sum()
    if total == 0:
        return (float("nan"), float("nan"))
    return (float((w.sum(axis=0) * y).sum() / total), float((w.sum(axis=1) * z).sum() / total))


def interface_height(codes: np.ndarray, z: np.ndarray, lower: set[int]) -> np.ndarray:
    """Per column, the height of the first transition out of the ``lower`` codes (NaN if none)."""
    below = np.isin(codes, list(lower))
    heights = np.full(codes.shape[1], np.nan)
    for j in range(codes.shape[1]):
        column = below[:, j]
        if column[0] and not column.all():
            k = int(np.argmin(column))
            heights[j] = 0.5 * (z[k - 1] + z[k]) if k > 0 else z[0]
    return heights


def signed_curvature(y: np.ndarray, height: np.ndarray, smoothing: float, at: float | None = None) -> float:
    """Signed curvature of an interface ``z(y)`` at ``at`` (default: channel centre), physical units (1/m).

    The curve is smoothed with a Gaussian of standard deviation ``smoothing``
    (same units as ``y``) and a local quadratic is fitted within one smoothing
    length. Positive curvature is concave up.
    """
    valid = np.isfinite(height)
    if valid.sum() < 5:
        return float("nan")
    yv, hv = y[valid], height[valid]
    at = 0.5 * (y[0] + y[-1]) if at is None else at
    dy = float(np.median(np.diff(yv)))
    sigma = max(smoothing / dy, 0.5)
    smooth = ndimage.gaussian_filter1d(hv, sigma, mode="nearest")
    window = np.abs(yv - at) <= max(smoothing, 2 * dy)
    if window.sum() < 3:
        return float("nan")
    a, b, _ = np.polyfit(yv[window] - at, smooth[window], 2)
    return float(2 * a / (1 + b * b) ** 1.5)


def compare_sections(a: np.ndarray, b: np.ndarray, codes: list[int], spacing: tuple[float, float]) -> dict:
    """Per-code IoU and interface distance between two categorical fields on the same grid."""
    return {
        int(c): {"iou": iou(a, b, c), "interface_distance_m": interface_distance(a, b, c, spacing)}
        for c in codes
    }
