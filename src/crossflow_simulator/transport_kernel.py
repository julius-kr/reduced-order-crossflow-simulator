"""Optional compiled streamline kernel (Numba).

This narrow compiled kernel was added after a missed performance budget:
profiling showed the per-step Python overhead of the vectorized tracer
dominating multi-group runs. The kernel implements exactly the algorithm of
:func:`crossflow_simulator.transport.trace` for the adaptive midpoint
integrator (same interpolation, wall and solid limiters, opening
classification, status codes), one trajectory per loop iteration, in
parallel over trajectories. The NumPy implementation remains the reference;
``tests/test_transport_kernel.py`` requires both to agree.

Numba is optional. Set ``CROSSFLOW_DISABLE_NUMBA=1`` to force the NumPy path.
"""
from __future__ import annotations

import os

import numpy as np

try:  # pragma: no cover - availability depends on the environment
    if os.environ.get("CROSSFLOW_DISABLE_NUMBA"):
        raise ImportError("disabled")
    from numba import njit, prange

    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False

REACHED_PLANE, REACHED_OPENING, WALL, SOLID, STAGNATION, MAX_LENGTH = 0, 1, 2, 3, 4, 5

if AVAILABLE:

    @njit(cache=True, inline="always")
    def _index(x, coords, uniform, origin, inverse):
        n = coords.shape[0]
        if uniform:
            f = (x - origin) * inverse
        else:
            if x <= coords[0]:
                f = 0.0
            elif x >= coords[n - 1]:
                f = n - 1.0
            else:
                i = np.searchsorted(coords, x) - 1
                f = i + (x - coords[i]) / (coords[i + 1] - coords[i])
        if f < 0.0:
            f = 0.0
        if f > n - 1.0:
            f = n - 1.0
        return f

    @njit(cache=True, inline="always")
    def _trilinear(data, f0, f1, f2):
        n0, n1, n2 = data.shape
        i0 = min(int(f0), n0 - 2) if n0 > 1 else 0
        i1 = min(int(f1), n1 - 2) if n1 > 1 else 0
        i2 = min(int(f2), n2 - 2) if n2 > 1 else 0
        t0, t1, t2 = f0 - i0, f1 - i1, f2 - i2
        j0, j1, j2 = min(i0 + 1, n0 - 1), min(i1 + 1, n1 - 1), min(i2 + 1, n2 - 1)
        c00 = data[i0, i1, i2] * (1 - t0) + data[j0, i1, i2] * t0
        c01 = data[i0, i1, j2] * (1 - t0) + data[j0, i1, j2] * t0
        c10 = data[i0, j1, i2] * (1 - t0) + data[j0, j1, i2] * t0
        c11 = data[i0, j1, j2] * (1 - t0) + data[j0, j1, j2] * t0
        c0 = c00 * (1 - t1) + c10 * t1
        c1 = c01 * (1 - t1) + c11 * t1
        return c0 * (1 - t2) + c1 * t2

    @njit(cache=True)
    def _velocity(p, data0, data1, data2, flags0, flags1, flags2, faces, centers, uniform, origin, inverse, out):
        # faces/centers: 3 axis coordinate arrays each (centers padded by one ghost layer)
        fi = np.empty(3)
        ci = np.empty(3)
        for d in range(3):
            fi[d] = _index(p[d], faces[d], uniform[0, d], origin[0, d], inverse[0, d])
            ci[d] = _index(p[d], centers[d], uniform[1, d], origin[1, d], inverse[1, d])
        for c in range(3):
            if c == 0:
                data, flags = data0, flags0
            elif c == 1:
                data, flags = data1, flags1
            else:
                data, flags = data2, flags2
            f0 = fi[0] if c == 0 else ci[0]
            f1 = fi[1] if c == 1 else ci[1]
            f2 = fi[2] if c == 2 else ci[2]
            value = _trilinear(data, f0, f1, f2)
            # quadratic decay of the wall-normal component next to no-penetration faces
            shape = flags.shape
            pos = fi[c]
            i0 = min(int(pos), shape[c] - 2)
            t = pos - i0
            idx = np.empty(3, dtype=np.int64)
            for d in range(3):
                if d != c:
                    k = int(np.rint(ci[d])) - 1
                    if k < 0:
                        k = 0
                    if k > shape[d] - 1:
                        k = shape[d] - 1
                    idx[d] = k
            idx[c] = i0
            lower = flags[idx[0], idx[1], idx[2]]
            idx[c] = i0 + 1
            upper = flags[idx[0], idx[1], idx[2]]
            if lower and upper:
                value *= 4.0 * t * (1.0 - t)
            elif lower:
                value *= t
            elif upper:
                value *= 1.0 - t
            out[c] = value

    @njit(cache=True, inline="always")
    def _in_solid(p, solid, faces):
        idx = np.empty(3, dtype=np.int64)
        for d in range(3):
            n = solid.shape[d]
            k = np.searchsorted(faces[d], p[d], side="right") - 1  # same as floor of the face index
            if k < 0:
                k = 0
            if k > n - 1:
                k = n - 1
            idx[d] = k
        return solid[idx[0], idx[1], idx[2]], idx

    @njit(cache=True)
    def _direction(p, lo, hi, direction, stagnation, data0, data1, data2, flags0, flags1, flags2, faces, centers,
                   uniform, origin, inverse, out):
        q = np.empty(3)
        for d in range(3):
            q[d] = min(max(p[d], lo[d]), hi[d])
        vel = np.empty(3)
        _velocity(q, data0, data1, data2, flags0, flags1, flags2, faces, centers, uniform, origin, inverse, vel)
        speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        safe = speed if speed > stagnation else 1.0
        for d in range(3):
            out[d] = direction * vel[d] / safe
        return speed

    @njit(cache=True)
    def _contains(openings, wall, x, t, tolerance):
        for k in range(openings.shape[0]):
            if int(openings[k, 0]) == wall:
                r = openings[k, 3] + tolerance
                if (x - openings[k, 1]) ** 2 + (t - openings[k, 2]) ** 2 <= r * r:
                    return k + 1
        return 0

    @njit(cache=True, parallel=True)
    def trace_kernel(seeds, data0, data1, data2, flags0, flags1, flags2, faces, centers, uniform, origin, inverse,
                     solid, any_solid, openings, bounds, lo, hi, widths_min, params):
        """Trace every seed; returns (end, status, origin, residence, length, wall_limited)."""
        stop_x, direction, ds, ds_min, ds_max, tol, max_length, stagnation, opening_tol = (
            params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8])
        max_steps = int(params[9])
        n = seeds.shape[0]
        end = seeds.copy()
        status = np.full(n, MAX_LENGTH, dtype=np.int8)
        origin_out = np.full(n, -1, dtype=np.int16)
        residence = np.zeros(n)
        length = np.zeros(n)
        limited_out = np.zeros(n, dtype=np.bool_)
        for i in prange(n):
            p = seeds[i].copy()
            h = ds
            res = 0.0
            ln = 0.0
            limited = False
            done = False
            d1 = np.empty(3)
            d2 = np.empty(3)
            pm = np.empty(3)
            new = np.empty(3)
            for _ in range(max_steps):
                s1 = _direction(p, lo, hi, direction, stagnation, data0, data1, data2, flags0, flags1, flags2,
                                faces, centers, uniform, origin, inverse, d1)
                for d in range(3):
                    pm[d] = p[d] + 0.5 * h * d1[d]
                s2 = _direction(pm, lo, hi, direction, stagnation, data0, data1, data2, flags0, flags1, flags2,
                                faces, centers, uniform, origin, inverse, d2)
                err = 0.5 * h * np.sqrt((d2[0] - d1[0]) ** 2 + (d2[1] - d1[1]) ** 2 + (d2[2] - d1[2]) ** 2)
                accept = err <= tol or h <= ds_min * (1 + 1e-9)
                grow = 0.9 * np.sqrt(tol / max(err, 1e-300))
                grow = min(max(grow, 0.3), 2.0)
                h_next = min(max(h * grow, ds_min), ds_max)
                if not accept:
                    h = h_next
                    continue
                for d in range(3):
                    new[d] = p[d] + h * d2[d]
                # wall-normal step limiter (openings exempt at the crossing point)
                for w in range(4):
                    axis = 1 if w < 2 else 2
                    lower = w == 0 or w == 2
                    plane = bounds[axis, 0] if lower else bounds[axis, 1]
                    d_old = (p[axis] - plane) if lower else (plane - p[axis])
                    d_new = (new[axis] - plane) if lower else (plane - new[axis])
                    target = max(0.5 * d_old, 1e-3 * widths_min[axis])
                    if d_new < target:
                        theta = 1.0
                        if d_new < 0:
                            delta = d_old - d_new
                            theta = min(max(d_old / (delta if delta != 0 else 1.0), 0.0), 1.0)
                        px = p[0] + theta * (new[0] - p[0])
                        pt = (p[1] + theta * (new[1] - p[1])) if axis == 2 else (p[2] + theta * (new[2] - p[2]))
                        if _contains(openings, w, px, pt, opening_tol) == 0:
                            new[axis] = plane + target if lower else plane - target
                            limited = True
                speed = min(s1, s2)
                stalled = speed <= stagnation
                if not stalled:
                    res += h / max(s2, stagnation)
                ln += h
                h = h_next
                # stop plane
                crossed = new[0] <= stop_x if direction < 0 else new[0] >= stop_x
                if crossed:
                    denom = new[0] - p[0]
                    theta = (stop_x - p[0]) / denom if denom != 0 else 1.0
                    for d in range(3):
                        new[d] = p[d] + theta * (new[d] - p[d])
                    status[i] = REACHED_PLANE
                    origin_out[i] = 0
                    done = True
                if not done:
                    for w in range(4):
                        axis = 1 if w < 2 else 2
                        lower = w == 0 or w == 2
                        plane = bounds[axis, 0] if lower else bounds[axis, 1]
                        beyond = new[axis] < plane if lower else new[axis] > plane
                        if beyond:
                            delta = new[axis] - p[axis]
                            theta = min(max((plane - p[axis]) / (delta if delta != 0 else 1.0), 0.0), 1.0)
                            for d in range(3):
                                new[d] = p[d] + theta * (new[d] - p[d])
                            new[axis] = plane
                            pt = new[1] if axis == 2 else new[2]
                            k = _contains(openings, w, new[0], pt, opening_tol)
                            if k > 0:
                                status[i] = REACHED_OPENING
                                origin_out[i] = k
                            else:
                                status[i] = WALL
                            done = True
                            break
                if not done and any_solid:
                    inside, n_cell = _in_solid(new, solid, faces)
                    if inside:
                        _, p_cell = _in_solid(p, solid, faces)
                        for axis in range(3):
                            if n_cell[axis] != p_cell[axis]:
                                up = n_cell[axis] > p_cell[axis]
                                f = faces[axis]
                                face = f[min(p_cell[axis] + 1, f.shape[0] - 1)] if up else f[p_cell[axis]]
                                d_old = abs(p[axis] - face)
                                target = max(0.5 * d_old, 1e-3 * widths_min[axis])
                                new[axis] = face - target if up else face + target
                        limited = True
                        still, _ = _in_solid(new, solid, faces)
                        if still:
                            status[i] = SOLID
                            done = True
                if not done and stalled:
                    status[i] = STAGNATION
                    done = True
                if not done and ln > max_length:
                    status[i] = MAX_LENGTH
                    done = True
                for d in range(3):
                    p[d] = new[d]
                if done:
                    break
            for d in range(3):
                end[i, d] = p[d]
            residence[i] = res
            length[i] = ln
            limited_out[i] = limited
        return end, status, origin_out, residence, length, limited_out


def kernel_inputs(velocity) -> dict | None:
    """Flatten a :class:`StaggeredInterpolator` into kernel arrays (cached on the interpolator)."""
    if not AVAILABLE:
        return None
    cached = getattr(velocity, "_kernel_inputs", None)
    if cached is not None:
        return cached
    faces = tuple(np.ascontiguousarray(a.coords) for a in velocity.face_axes)
    centers = tuple(np.ascontiguousarray(a.coords) for a in velocity.center_axes)
    uniform = np.array([[a.uniform for a in velocity.face_axes], [a.uniform for a in velocity.center_axes]])
    origin = np.array([[a.origin for a in velocity.face_axes], [a.origin for a in velocity.center_axes]])
    inverse = np.array([[a.inverse for a in velocity.face_axes], [a.inverse for a in velocity.center_axes]])
    flags = [f.reshape(s) for f, s in zip(velocity.flags, velocity.face_shapes)]
    inputs = {
        "data": tuple(np.ascontiguousarray(d) for d in velocity.data),
        "flags": tuple(np.ascontiguousarray(f) for f in flags),
        "faces": faces,
        "centers": centers,
        "uniform": uniform,
        "origin": origin,
        "inverse": inverse,
        "solid": np.ascontiguousarray(velocity.solid),
        "any_solid": bool(velocity.any_solid),
    }
    velocity._kernel_inputs = inputs
    return inputs
