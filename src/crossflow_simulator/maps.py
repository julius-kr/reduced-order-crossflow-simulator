"""Transport maps: chained donor maps, adaptive sections, composition, conservative remap.

Categorical sections are computed by tracing output points backward through
every group down to the inlet plane (exact composition of the continuous
maps) and evaluating the analytic inlet pattern once at the final donor
coordinates. To keep this affordable, points are traced on a coarse lattice
first; cells whose corners disagree in origin, or whose donor map bends more
than a quarter output cell, are traced point by point, and everything else is
interpolated. A random sample of interpolated points is traced directly as a
check and reported.

Quantitative (conservative) transport uses per-group sparse transfer weights
acting on cell flux: every source cell's flux is distributed over the target
cells its fluid reaches, so per-material flux is conserved by construction.
Source cells that reach no target cell (lost thin regions) are reported.
Numerical stream codes are never interpolated as concentrations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp

from .models.stokes_map import GroupFlow
from .transport import REACHED_OPENING, REACHED_PLANE, STATUS_NAMES, trace

KIND_INVALID, KIND_PATTERN, KIND_PORT = -1, 0, 1


@dataclass
class ChainTrace:
    code: np.ndarray  # stream code (-1 invalid)
    kind: np.ndarray  # KIND_PATTERN / KIND_PORT / KIND_INVALID
    status: np.ndarray  # last trajectory status
    donor_y: np.ndarray  # y at the inlet plane (NaN unless kind == pattern)
    donor_z: np.ndarray
    residence: np.ndarray
    port_group: np.ndarray  # index of the group whose port supplied the fluid (-1 otherwise)


@dataclass
class ChainLink:
    flow: GroupFlow
    port_codes: list[int]

    @property
    def x_start(self) -> float:
        return float(self.flow.domain.grid.faces[0][0])

    @property
    def x_end(self) -> float:
        return float(self.flow.domain.grid.faces[0][-1])


class FlowChain:
    """Ordered groups plus the inlet pattern; straight developed segments between groups are identity maps."""

    def __init__(
        self,
        links: Sequence[ChainLink],
        pattern: Callable[[np.ndarray, np.ndarray], np.ndarray],
        step_fraction: float = 0.5,
        integrator: str = "rk2",
        cancel: Callable[[], bool] | None = None,
        tolerance: float = 1e-3,
    ):
        self.links = list(links)
        self.pattern = pattern
        self.step_fraction = step_fraction
        self.integrator = integrator
        self.tolerance = tolerance
        self.cancel = cancel
        self.traced_points = 0
        self.looked_up_points = 0
        # Stored donor maps of already computed sections: {group index: (SectionMap, y centres, z centres)}.
        self.section_maps: dict[int, tuple["SectionMap", np.ndarray, np.ndarray]] = {}

    def trace(self, y: np.ndarray, z: np.ndarray, upto: int) -> ChainTrace:
        """Trace seeds on the outlet plane of group ``upto`` (``-1``: the inlet plane itself)."""
        y = np.asarray(y, dtype=float).ravel().copy()
        z = np.asarray(z, dtype=float).ravel().copy()
        n = len(y)
        code = np.full(n, -1, dtype=np.int32)
        kind = np.full(n, KIND_INVALID, dtype=np.int8)
        status = np.full(n, REACHED_PLANE, dtype=np.int8)
        residence = np.zeros(n)
        port_group = np.full(n, -1, dtype=np.int16)
        to_pattern = np.zeros(n, dtype=bool)
        active = np.arange(n)
        for g in range(upto, -1, -1):
            if not len(active):
                break
            if g < upto and g in self.section_maps:
                # Compose with the stored map of the plane after group g where it is locally smooth.
                smap, yc, zc = self.section_maps[g]
                found, kind_l, code_l, dy_l, dz_l, res_l = smap.lookup(y[active], z[active], yc, zc)
                residence[active[found]] += res_l[found]
                pattern_hit = found & (kind_l == KIND_PATTERN)
                port_hit = found & (kind_l == KIND_PORT)
                y[active[pattern_hit]] = dy_l[pattern_hit]
                z[active[pattern_hit]] = dz_l[pattern_hit]
                to_pattern[active[pattern_hit]] = True
                status[active[pattern_hit]] = REACHED_PLANE
                code[active[port_hit]] = code_l[port_hit]
                kind[active[port_hit]] = KIND_PORT
                status[active[port_hit]] = REACHED_OPENING
                self.looked_up_points += int(found.sum())
                active = active[~found]
                if not len(active):
                    break
            link = self.links[g]
            grid = link.flow.domain.grid
            seeds = np.column_stack([np.full(len(active), link.x_end), y[active], z[active]])
            step = self.step_fraction * min(float(w.min()) for w in grid.widths)
            result = trace(link.flow.interpolator, seeds, stop_x=link.x_start, direction=-1,
                           openings=link.flow.domain.opening_shapes, step=step, integrator=self.integrator,
                           cancel=self.cancel, error_tolerance=self.tolerance)
            self.traced_points += len(active)
            residence[active] += result.residence_time
            status[active] = result.status
            port = result.status == REACHED_OPENING
            if port.any():
                idx = active[port]
                code[idx] = np.asarray(link.port_codes)[result.origin[port] - 1]
                kind[idx] = KIND_PORT
                port_group[idx] = g
            plane = result.status == REACHED_PLANE
            y[active[plane]] = result.end[plane, 1]
            z[active[plane]] = result.end[plane, 2]
            active = active[plane]
        to_pattern[active] = True
        if to_pattern.any():
            code[to_pattern] = self.pattern(y[to_pattern], z[to_pattern])
            kind[to_pattern] = KIND_PATTERN
        donor_y = np.where(kind == KIND_PATTERN, y, np.nan)
        donor_z = np.where(kind == KIND_PATTERN, z, np.nan)
        return ChainTrace(code, kind, status, donor_y, donor_z, residence, port_group)


@dataclass
class SectionMap:
    code: np.ndarray  # (nz, ny)
    kind: np.ndarray
    status: np.ndarray
    donor_y: np.ndarray
    donor_z: np.ndarray
    residence: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def lookup(self, y: np.ndarray, z: np.ndarray, yc: np.ndarray, zc: np.ndarray, max_spread_cells: float = 2.0):
        """Interpolate this map at arbitrary points where it is locally smooth.

        A point is resolved when its four surrounding cell centres all trace to
        the inlet plane with donors within ``max_spread_cells`` cells of each other
        (bilinear interpolation of the donor coordinates), or all come from the
        same port. Other points, and points outside the lattice of cell centres,
        are left for direct tracing.
        """
        n = len(y)
        found = np.zeros(n, dtype=bool)
        kind = np.full(n, KIND_INVALID, dtype=np.int8)
        code = np.full(n, -1, dtype=np.int32)
        dy = np.full(n, np.nan)
        dz = np.full(n, np.nan)
        res = np.zeros(n)
        inside = (y >= yc[0]) & (y <= yc[-1]) & (z >= zc[0]) & (z <= zc[-1])
        if not inside.any() or len(yc) < 2 or len(zc) < 2:
            return found, kind, code, dy, dz, res
        yi, zi = y[inside], z[inside]
        fy = np.interp(yi, yc, np.arange(len(yc), dtype=float))
        fz = np.interp(zi, zc, np.arange(len(zc), dtype=float))
        j0 = np.clip(np.floor(fy).astype(int), 0, len(yc) - 2)
        k0 = np.clip(np.floor(fz).astype(int), 0, len(zc) - 2)
        s, t = fy - j0, fz - k0
        corners = [(k0, j0), (k0 + 1, j0), (k0, j0 + 1), (k0 + 1, j0 + 1)]
        kinds = [self.kind[c] for c in corners]
        codes = [self.code[c] for c in corners]
        all_pattern = np.logical_and.reduce([k == KIND_PATTERN for k in kinds])
        spacing = max_spread_cells * max(float(np.max(np.diff(yc))), float(np.max(np.diff(zc))))
        ys = np.array([np.nan_to_num(self.donor_y[c]) for c in corners])
        zs = np.array([np.nan_to_num(self.donor_z[c]) for c in corners])
        smooth = all_pattern & (np.ptp(ys, axis=0) <= spacing) & (np.ptp(zs, axis=0) <= spacing)
        same_port = (np.logical_and.reduce([k == KIND_PORT for k in kinds])
                     & (codes[0] == codes[1]) & (codes[0] == codes[2]) & (codes[0] == codes[3]))
        w = [(1 - t) * (1 - s), t * (1 - s), (1 - t) * s, t * s]
        local_found = smooth | same_port
        idx = np.flatnonzero(inside)
        found[idx] = local_found
        kind[idx] = np.where(smooth, KIND_PATTERN, np.where(same_port, KIND_PORT, KIND_INVALID))
        code[idx] = np.where(same_port, codes[0], -1)
        dy[idx] = np.where(smooth, sum(wi * a for wi, a in zip(w, ys)), np.nan)
        dz[idx] = np.where(smooth, sum(wi * a for wi, a in zip(w, zs)), np.nan)
        res[idx] = sum(wi * self.residence[c] for wi, c in zip(w, corners))
        return found, kind, code, dy, dz, res


def adaptive_section(
    chain: FlowChain,
    upto: int,
    y: np.ndarray,
    z: np.ndarray,
    stride: int = 4,
    tolerance: float | None = None,
    check_samples: int = 64,
    seed: int = 0,
) -> SectionMap:
    """Donor map of the section after group ``upto`` on the (z, y) output grid."""
    ny, nz = len(y), len(z)
    tolerance = 0.25 * min(float(np.min(np.diff(y))), float(np.min(np.diff(z)))) if tolerance is None else tolerance
    stride = max(1, int(stride))
    jn = np.unique(np.r_[np.arange(0, ny, stride), ny - 1])
    kn = np.unique(np.r_[np.arange(0, nz, stride), nz - 1])
    Yc, Zc = np.meshgrid(y[jn], z[kn])
    coarse = chain.trace(Yc, Zc, upto)
    shape_c = (len(kn), len(jn))
    c_code, c_kind = coarse.code.reshape(shape_c), coarse.kind.reshape(shape_c)
    c_dy, c_dz = coarse.donor_y.reshape(shape_c), coarse.donor_z.reshape(shape_c)
    c_res = coarse.residence.reshape(shape_c)
    c_status = coarse.status.reshape(shape_c)

    # Interpolation error estimate from second differences of the donor map.
    error = np.zeros(shape_c)
    for axis in (0, 1):
        for d in (c_dy, c_dz):
            second = np.full(shape_c, 0.0)
            if shape_c[axis] >= 3:
                lo = [slice(None)] * 2
                mid = [slice(None)] * 2
                hi = [slice(None)] * 2
                lo[axis], mid[axis], hi[axis] = slice(0, -2), slice(1, -1), slice(2, None)
                val = np.abs(d[tuple(hi)] - 2 * d[tuple(mid)] + d[tuple(lo)]) / 8.0
                val = np.where(np.isfinite(val), val, np.inf)
                target = [slice(None)] * 2
                target[axis] = slice(1, -1)
                second[tuple(target)] = np.maximum(second[tuple(target)], val)
            error = np.maximum(error, second)

    # Cell classification (cells between neighbouring coarse nodes).
    corners = lambda a: (a[:-1, :-1], a[1:, :-1], a[:-1, 1:], a[1:, 1:])  # noqa: E731
    k00, k10, k01, k11 = corners(c_kind)
    code00, code10, code01, code11 = corners(c_code)
    e_cell = np.maximum.reduce(corners(error))
    all_pattern = (k00 == KIND_PATTERN) & (k10 == KIND_PATTERN) & (k01 == KIND_PATTERN) & (k11 == KIND_PATTERN)
    same_port = ((k00 == KIND_PORT) & (k10 == KIND_PORT) & (k01 == KIND_PORT) & (k11 == KIND_PORT)
                 & (code00 == code10) & (code00 == code01) & (code00 == code11))
    interpolable = (all_pattern & (e_cell <= tolerance)) | same_port
    if len(kn) == 1 or len(jn) == 1:
        interpolable = np.zeros((max(len(kn) - 1, 1), max(len(jn) - 1, 1)), dtype=bool)

    # Map every fine point to its cell and interpolation weights.
    kk, jj = np.meshgrid(np.arange(nz), np.arange(ny), indexing="ij")
    a = np.clip(np.searchsorted(kn, kk, side="right") - 1, 0, max(len(kn) - 2, 0))
    b = np.clip(np.searchsorted(jn, jj, side="right") - 1, 0, max(len(jn) - 2, 0))
    refine = ~interpolable[a, b] if interpolable.size else np.ones((nz, ny), dtype=bool)
    # Points on a shared cell edge are refined if either neighbouring cell is refined.
    on_k = np.isin(kk, kn) & (a > 0)
    on_j = np.isin(jj, jn) & (b > 0)
    if interpolable.size:
        refine |= on_k & ~interpolable[np.maximum(a - 1, 0), b]
        refine |= on_j & ~interpolable[a, np.maximum(b - 1, 0)]
        refine |= on_k & on_j & ~interpolable[np.maximum(a - 1, 0), np.maximum(b - 1, 0)]
    is_node = np.isin(kk, kn) & np.isin(jj, jn)

    code = np.full((nz, ny), -1, dtype=np.int32)
    kind = np.full((nz, ny), KIND_INVALID, dtype=np.int8)
    status = np.full((nz, ny), REACHED_PLANE, dtype=np.int8)
    donor_y = np.full((nz, ny), np.nan)
    donor_z = np.full((nz, ny), np.nan)
    residence = np.zeros((nz, ny))
    node_k = np.searchsorted(kn, kk[is_node])
    node_j = np.searchsorted(jn, jj[is_node])
    code[is_node] = c_code[node_k, node_j]
    kind[is_node] = c_kind[node_k, node_j]
    status[is_node] = c_status[node_k, node_j]
    donor_y[is_node] = c_dy[node_k, node_j]
    donor_z[is_node] = c_dz[node_k, node_j]
    residence[is_node] = c_res[node_k, node_j]

    interp = ~refine & ~is_node
    if interp.any() and len(kn) > 1 and len(jn) > 1:
        ai, bi = a[interp], b[interp]
        t = (kk[interp] - kn[ai]) / (kn[ai + 1] - kn[ai])
        s = (jj[interp] - jn[bi]) / (jn[bi + 1] - jn[bi])

        def bilinear(field_: np.ndarray) -> np.ndarray:
            return ((1 - t) * (1 - s) * field_[ai, bi] + t * (1 - s) * field_[ai + 1, bi]
                    + (1 - t) * s * field_[ai, bi + 1] + t * s * field_[ai + 1, bi + 1])

        port_cell = c_kind[ai, bi] == KIND_PORT
        iy = np.where(port_cell, np.nan, bilinear(np.nan_to_num(c_dy)))
        iz = np.where(port_cell, np.nan, bilinear(np.nan_to_num(c_dz)))
        values = np.where(port_cell, c_code[ai, bi], 0)
        if (~port_cell).any():
            values[~port_cell] = chain.pattern(iy[~port_cell], iz[~port_cell])
        code[interp] = values
        kind[interp] = np.where(port_cell, KIND_PORT, KIND_PATTERN)
        status[interp] = np.where(port_cell, REACHED_OPENING, REACHED_PLANE)
        donor_y[interp] = iy
        donor_z[interp] = iz
        residence[interp] = bilinear(c_res)

    todo = refine & ~is_node
    if todo.any():
        direct = chain.trace(np.broadcast_to(y, (nz, ny))[todo], np.broadcast_to(z[:, None], (nz, ny))[todo], upto)
        code[todo] = direct.code
        kind[todo] = direct.kind
        status[todo] = direct.status
        donor_y[todo] = direct.donor_y
        donor_z[todo] = direct.donor_z
        residence[todo] = direct.residence

    # Spot-check interpolated points against direct tracing.
    check = {"samples": 0, "code_mismatch": 0, "max_donor_error_m": 0.0}
    candidates = np.flatnonzero(interp.ravel())
    if len(candidates) and check_samples > 0:
        rng = np.random.default_rng(seed)
        pick = rng.choice(candidates, size=min(check_samples, len(candidates)), replace=False)
        kp, jp = np.unravel_index(pick, (nz, ny))
        direct = chain.trace(y[jp], z[kp], upto)
        mismatch = direct.code != code[kp, jp]
        errors = np.hypot(direct.donor_y - donor_y[kp, jp], direct.donor_z - donor_z[kp, jp])
        errors = errors[np.isfinite(errors)]
        check = {
            "samples": int(len(pick)),
            "code_mismatch": int(mismatch.sum()),
            "max_donor_error_m": float(errors.max()) if len(errors) else 0.0,
        }
    counts = {name: int(np.sum(status == c)) for c, name in STATUS_NAMES.items()}
    diagnostics = {
        "points": int(nz * ny),
        "coarse_points": int(len(kn) * len(jn)),
        "directly_traced": int(todo.sum()),
        "interpolated": int(interp.sum()),
        "interpolation_tolerance_m": tolerance,
        "interpolation_check": check,
        "trajectory_status": counts,
        "invalid_fraction": float(np.mean(kind == KIND_INVALID)),
    }
    return SectionMap(code, kind, status, donor_y, donor_z, residence, diagnostics)


# --------------------------------------------------------------------------- group maps and composition


@dataclass
class GroupMap:
    """Backward map of one group on the output grid: donor (y, z) on its inflow plane, or a port."""

    y: np.ndarray
    z: np.ndarray
    kind: np.ndarray  # KIND_PATTERN means "from the inflow plane" here
    port: np.ndarray  # port index (0-based within the group) or -1
    donor_y: np.ndarray
    donor_z: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def evaluate(self, yq: np.ndarray, zq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Donor at arbitrary points: bilinear where the 4 neighbours share the inflow origin, else nearest."""
        yq = np.asarray(yq, dtype=float)
        zq = np.asarray(zq, dtype=float)
        fy = np.interp(yq, self.y, np.arange(len(self.y)))
        fz = np.interp(zq, self.z, np.arange(len(self.z)))
        j0 = np.clip(np.floor(fy).astype(int), 0, len(self.y) - 2)
        k0 = np.clip(np.floor(fz).astype(int), 0, len(self.z) - 2)
        s, t = fy - j0, fz - k0
        kinds = [self.kind[k0, j0], self.kind[k0 + 1, j0], self.kind[k0, j0 + 1], self.kind[k0 + 1, j0 + 1]]
        smooth = np.logical_and.reduce([k == KIND_PATTERN for k in kinds])

        def bilinear(f: np.ndarray) -> np.ndarray:
            return ((1 - t) * (1 - s) * f[k0, j0] + t * (1 - s) * f[k0 + 1, j0]
                    + (1 - t) * s * f[k0, j0 + 1] + t * s * f[k0 + 1, j0 + 1])

        kn = np.clip(np.rint(fz).astype(int), 0, len(self.z) - 1)
        jn = np.clip(np.rint(fy).astype(int), 0, len(self.y) - 1)
        kind = np.where(smooth, KIND_PATTERN, self.kind[kn, jn])
        port = np.where(smooth, -1, self.port[kn, jn])
        dy = np.where(smooth, bilinear(np.nan_to_num(self.donor_y)), self.donor_y[kn, jn])
        dz = np.where(smooth, bilinear(np.nan_to_num(self.donor_z)), self.donor_z[kn, jn])
        return kind, port, dy, dz


def group_map(flow: GroupFlow, y: np.ndarray, z: np.ndarray, stride: int = 4, step_fraction: float = 0.5,
              integrator: str = "rk2", cancel: Callable[[], bool] | None = None,
              tolerance: float = 1e-3) -> GroupMap:
    """Single-group backward map on the output grid (pattern replaced by the identity)."""
    n_ports = len(flow.domain.openings)
    link = ChainLink(flow, port_codes=[-(i + 2) for i in range(n_ports)])  # negative codes tag ports
    chain = FlowChain([link], pattern=lambda yy, zz: np.zeros(len(yy), dtype=np.int32),
                      step_fraction=step_fraction, integrator=integrator, cancel=cancel, tolerance=tolerance)
    section = adaptive_section(chain, 0, y, z, stride=stride, check_samples=0)
    port = np.where(section.kind == KIND_PORT, -section.code - 2, -1)
    return GroupMap(y, z, section.kind, port, section.donor_y, section.donor_z, section.diagnostics)


def compose(first: GroupMap, second: GroupMap) -> GroupMap:
    """Map of ``first`` then ``second`` (``second`` is downstream): donor_first(donor_second(a))."""
    kind2, port2 = second.kind, second.port
    from_plane = kind2 == KIND_PATTERN
    kind = kind2.copy()
    port = np.where(kind2 == KIND_PORT, port2 + 1000, -1)  # ports of the second map are tagged +1000
    dy = np.full(kind.shape, np.nan)
    dz = np.full(kind.shape, np.nan)
    k1, p1, y1, z1 = first.evaluate(second.donor_y[from_plane], second.donor_z[from_plane])
    kind[from_plane] = k1
    port[from_plane] = p1
    dy[from_plane] = y1
    dz[from_plane] = z1
    return GroupMap(second.y, second.z, kind, port, dy, dz, {"composed": True})


# --------------------------------------------------------------------------- conservative remap


class ConservativeRemap:
    """Flux-conservative sparse transfer from a source section (+ ports) to a target section."""

    def __init__(
        self,
        gmap: GroupMap,
        source_edges: tuple[np.ndarray, np.ndarray],
        source_flux: np.ndarray,
        target_flux: np.ndarray,
        port_rates: Sequence[float],
        subsamples: int = 3,
    ):
        y_edges, z_edges = source_edges
        nz, ny = target_flux.shape
        n_source = source_flux.size
        n_ports = len(port_rates)
        offsets = (np.arange(subsamples) + 0.5) / subsamples
        ty_edges = np.concatenate([[gmap.y[0] - 0.5 * (gmap.y[1] - gmap.y[0])], 0.5 * (gmap.y[1:] + gmap.y[:-1]),
                                   [gmap.y[-1] + 0.5 * (gmap.y[-1] - gmap.y[-2])]])
        tz_edges = np.concatenate([[gmap.z[0] - 0.5 * (gmap.z[1] - gmap.z[0])], 0.5 * (gmap.z[1:] + gmap.z[:-1]),
                                   [gmap.z[-1] + 0.5 * (gmap.z[-1] - gmap.z[-2])]])
        sy = (ty_edges[:-1, None] + np.diff(ty_edges)[:, None] * offsets[None, :]).ravel()
        sz = (tz_edges[:-1, None] + np.diff(tz_edges)[:, None] * offsets[None, :]).ravel()
        SY, SZ = np.meshgrid(sy, sz)
        kind, port, dy, dz = gmap.evaluate(SY.ravel(), SZ.ravel())
        target_row = (np.repeat(np.arange(nz), subsamples)[:, None] * ny
                      + np.repeat(np.arange(ny), subsamples)[None, :]).ravel()
        weight = (target_flux.ravel() / subsamples**2)[target_row]
        plane = kind == KIND_PATTERN
        j = np.clip(np.searchsorted(y_edges, dy[plane]) - 1, 0, len(y_edges) - 2)
        k = np.clip(np.searchsorted(z_edges, dz[plane]) - 1, 0, len(z_edges) - 2)
        cols = [k * (len(y_edges) - 1) + j]
        rows = [target_row[plane]]
        vals = [weight[plane]]
        ported = kind == KIND_PORT
        cols.append(n_source + port[ported])
        rows.append(target_row[ported])
        vals.append(weight[ported])
        raw = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(nz * ny, n_source + n_ports))
        column_sum = np.asarray(raw.sum(axis=0)).ravel()
        supply = np.concatenate([source_flux.ravel(), np.asarray(port_rates, dtype=float)])
        scale = np.divide(supply, column_sum, out=np.zeros_like(supply), where=column_sum > 0)
        self.matrix = (raw @ sp.diags(scale)).tocsr()  # every source's flux is fully distributed
        self.lost = supply * (column_sum == 0)
        self.supply = supply
        self.n_source = n_source
        row_flux = np.asarray(self.matrix.sum(axis=1)).ravel()
        target = target_flux.ravel()
        self.row_consistency = float(np.max(np.abs(row_flux - target)) / max(target.max(), 1e-300))
        self.shape = (nz, ny)

    def apply(self, source_fractions: np.ndarray, port_streams: Sequence[int]) -> tuple[np.ndarray, dict]:
        """Push per-stream flux fractions (n_streams, nz_s, ny_s) through the map."""
        n_streams = source_fractions.shape[0]
        port_part = np.zeros((n_streams, len(port_streams)))
        for p, s in enumerate(port_streams):
            port_part[s, p] = 1.0
        amounts = np.concatenate([source_fractions.reshape(n_streams, -1), port_part], axis=1)  # fraction per source
        flux = self.matrix @ amounts.T  # (targets, streams) absolute flux
        total = flux.sum(axis=1, keepdims=True)
        fractions = np.divide(flux, total, out=np.zeros_like(flux), where=total > 0).T.reshape(n_streams, *self.shape)
        lost_by_stream = (amounts * self.lost[None, :]).sum(axis=1)
        return fractions, {
            "lost_flux_by_stream_m3_s": lost_by_stream.tolist(),
            "lost_flux_fraction": float(self.lost.sum() / max(self.supply.sum(), 1e-300)),
            "row_consistency": self.row_consistency,
        }


# --------------------------------------------------------------------------- sub-cell stream fractions


def subcell_fractions(
    code: np.ndarray,
    kind: np.ndarray,
    donor_y: np.ndarray,
    donor_z: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    pattern: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_streams: int,
    samples: int = 4,
) -> np.ndarray:
    """Area fraction of every stream inside each output cell, shape ``(n_streams, nz, ny)``.

    Each cell is supersampled ``samples x samples``. Where the four surrounding
    cell centres all trace back to the inlet plane, the continuous donor
    coordinates are interpolated to the sub-points and the analytic inlet
    pattern is evaluated there; elsewhere (ports, origin boundaries, invalid
    trajectories) the cell's own code is used. This removes the half-cell
    quantization of interfaces from flux diagnostics without re-tracing.
    Cells without a valid trajectory contribute to no stream.
    """
    nz, ny = code.shape
    yc, zc = 0.5 * (y_edges[1:] + y_edges[:-1]), 0.5 * (z_edges[1:] + z_edges[:-1])
    offsets = (np.arange(samples) + 0.5) / samples
    sub_y = (y_edges[:-1, None] + np.diff(y_edges)[:, None] * offsets).ravel()
    sub_z = (z_edges[:-1, None] + np.diff(z_edges)[:, None] * offsets).ravel()
    fy = np.interp(sub_y, yc, np.arange(ny, dtype=float))
    fz = np.interp(sub_z, zc, np.arange(nz, dtype=float))
    j0 = np.clip(np.floor(fy).astype(int), 0, max(ny - 2, 0))
    k0 = np.clip(np.floor(fz).astype(int), 0, max(nz - 2, 0))
    ty, tz = fy - j0, fz - k0
    J0, K0 = np.meshgrid(j0, k0)
    TY, TZ = np.meshgrid(ty, tz)
    j1, k1 = np.minimum(J0 + 1, ny - 1), np.minimum(K0 + 1, nz - 1)
    smooth = ((kind[K0, J0] == KIND_PATTERN) & (kind[K0, j1] == KIND_PATTERN)
              & (kind[k1, J0] == KIND_PATTERN) & (kind[k1, j1] == KIND_PATTERN))
    owner = code[np.repeat(np.arange(nz), samples)[:, None], np.repeat(np.arange(ny), samples)[None, :]]
    sub_code = owner.copy()
    if smooth.any():
        def bilinear(f: np.ndarray) -> np.ndarray:
            f = np.nan_to_num(f)
            return ((1 - TZ) * ((1 - TY) * f[K0, J0] + TY * f[K0, j1]) + TZ * ((1 - TY) * f[k1, J0] + TY * f[k1, j1]))

        sub_code[smooth] = pattern(bilinear(donor_y)[smooth], bilinear(donor_z)[smooth])
    fractions = np.zeros((n_streams, nz, ny))
    for c in np.unique(sub_code):
        if c < 1 or c > n_streams:
            continue
        fractions[c - 1] = (sub_code == c).reshape(nz, samples, ny, samples).mean(axis=(1, 3))
    return fractions


def pattern_fractions(y_edges: np.ndarray, z_edges: np.ndarray,
                      pattern: Callable[[np.ndarray, np.ndarray], np.ndarray], n_streams: int,
                      samples: int = 4) -> np.ndarray:
    """Exact-by-supersampling stream area fractions of the analytic inlet pattern."""
    offsets = (np.arange(samples) + 0.5) / samples
    sy = (y_edges[:-1, None] + np.diff(y_edges)[:, None] * offsets).ravel()
    sz = (z_edges[:-1, None] + np.diff(z_edges)[:, None] * offsets).ravel()
    SY, SZ = np.meshgrid(sy, sz)
    codes = pattern(SY, SZ)
    nz, ny = len(z_edges) - 1, len(y_edges) - 1
    fractions = np.zeros((n_streams, nz, ny))
    for c in np.unique(codes):
        if 1 <= c <= n_streams:
            fractions[c - 1] = (codes == c).reshape(nz, samples, ny, samples).mean(axis=(1, 3))
    return fractions
