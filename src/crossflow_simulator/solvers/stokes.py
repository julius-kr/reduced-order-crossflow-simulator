"""Staggered (MAC) finite-volume Stokes solver on rectilinear grids.

Solves ``-div(2 mu e(u)) + grad p = 0``, ``div u = 0`` in a channel segment
``[x0, x1] x [0, W] x [0, H]`` with

* obstacle (solid) cells for pillars, including full-height obstacles;
* prescribed normal velocity on the inflow plane, the outflow plane, and on
  wall openings (ports); no-slip everywhere else;
* a cell-wise viscosity field (harmonic averaging at cell edges).

The viscous operator is assembled in symmetric energy form
``A = 2 sum Dn^T (mu V) Dn + 4 sum Ds^T (mu_e V_e) Ds`` from discrete normal
and shear strain rates, so constant and variable viscosity share one code
path and ``A`` is symmetric positive definite. No-slip for tangential
components uses ghost reflection about the wall or solid surface.

Arrays are indexed ``[x, y, z]``: ``u`` has shape ``(nx+1, ny, nz)``, ``v``
``(nx, ny+1, nz)``, ``w`` ``(nx, ny, nz+1)``, ``p`` ``(nx, ny, nz)``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Literal, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, minres, splu

UNKNOWN, DIRICHLET, GHOST = 0, 1, 2
WALL_AXIS = {"left": (1, 0), "right": (1, -1), "bottom": (2, 0), "top": (2, -1)}


class SolveCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class RectilinearGrid:
    xf: np.ndarray
    yf: np.ndarray
    zf: np.ndarray

    @cached_property
    def faces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.xf, self.yf, self.zf)

    @cached_property
    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(0.5 * (f[1:] + f[:-1]) for f in self.faces)  # type: ignore[return-value]

    @cached_property
    def widths(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(np.diff(f) for f in self.faces)  # type: ignore[return-value]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.xf) - 1, len(self.yf) - 1, len(self.zf) - 1)

    @cached_property
    def cell_volume(self) -> np.ndarray:
        hx, hy, hz = self.widths
        return hx[:, None, None] * hy[None, :, None] * hz[None, None, :]

    def face_shape(self, component: int) -> tuple[int, int, int]:
        shape = list(self.shape)
        shape[component] += 1
        return tuple(shape)  # type: ignore[return-value]

    def face_area(self, component: int) -> np.ndarray:
        """Area of each face of ``component`` (broadcastable to the face shape)."""
        widths = list(self.widths)
        widths[component] = np.ones(len(self.faces[component]))
        return widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]

    def face_coordinates(self, component: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = list(self.centers)
        coords[component] = self.faces[component]
        return tuple(coords)  # type: ignore[return-value]

    def key(self) -> dict[str, list[float]]:
        return {"xf": self.xf.tolist(), "yf": self.yf.tolist(), "zf": self.zf.tolist()}


@dataclass
class WallOpening:
    """A port: prescribed inflow through part of a wall.

    ``profile`` has the shape of the wall's face array (the face array with the
    wall-normal axis removed) and holds the inward normal velocity for a unit
    volumetric flow rate; it is zero outside the opening.
    """

    wall: Literal["left", "right", "bottom", "top"]
    profile: np.ndarray
    label: str = ""


@dataclass
class StokesSolution:
    grid: RectilinearGrid
    solid: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    p: np.ndarray
    info: dict = field(default_factory=dict)

    def combine(self, other: "StokesSolution", a: float = 1.0, b: float = 1.0) -> "StokesSolution":
        return StokesSolution(
            self.grid,
            self.solid,
            a * self.u + b * other.u,
            a * self.v + b * other.v,
            a * self.w + b * other.w,
            a * self.p + b * other.p,
            {"combined": True},
        )

    def scaled(self, factor: float) -> "StokesSolution":
        return StokesSolution(self.grid, self.solid, factor * self.u, factor * self.v, factor * self.w,
                              factor * self.p, {"scaled": factor})

    def plane_flux(self, index: int) -> float:
        area = self.grid.face_area(0)[0]
        return float(np.sum(self.u[index] * area))

    def divergence(self) -> np.ndarray:
        hx, hy, hz = self.grid.widths
        ax = hy[None, :, None] * hz[None, None, :]
        ay = hx[:, None, None] * hz[None, None, :]
        az = hx[:, None, None] * hy[None, :, None]
        return (
            (self.u[1:] - self.u[:-1]) * ax
            + (self.v[:, 1:] - self.v[:, :-1]) * ay
            + (self.w[:, :, 1:] - self.w[:, :, :-1]) * az
        )

    def pressure_drop(self) -> float:
        """Mean pressure in the first fluid cell layer minus the last."""
        first, last = self.p[0], self.p[-1]
        return float(np.nanmean(first) - np.nanmean(last))

    def arrays(self) -> dict[str, np.ndarray]:
        return {"u": self.u, "v": self.v, "w": self.w, "p": self.p}


def _face_status(grid: RectilinearGrid, fluid: np.ndarray, component: int) -> np.ndarray:
    """Classify faces: UNKNOWN (interior fluid), DIRICHLET (boundary/solid surface), GHOST."""
    n = grid.shape[component]
    pad_width = [(0, 0)] * 3
    pad_width[component] = (1, 1)
    padded = np.pad(fluid, pad_width, constant_values=False)
    lower = np.take(padded, np.arange(0, n + 1), axis=component)
    upper = np.take(padded, np.arange(1, n + 2), axis=component)
    status = np.full(grid.face_shape(component), GHOST, dtype=np.int8)
    status[lower | upper] = DIRICHLET
    status[lower & upper] = UNKNOWN
    return status


class StokesProblem:
    """Assembled discrete Stokes operator for one geometry and viscosity field."""

    def __init__(
        self,
        grid: RectilinearGrid,
        solid: np.ndarray,
        viscosity: float | np.ndarray = 1.0,
        openings: Sequence[WallOpening] = (),
    ):
        started = time.perf_counter()
        self.grid = grid
        self.solid = np.asarray(solid, dtype=bool)
        if self.solid.shape != grid.shape:
            raise ValueError("solid mask must match the grid shape")
        self.fluid = ~self.solid
        mu = np.broadcast_to(np.asarray(viscosity, dtype=float), grid.shape).copy()
        if np.any(~(mu[self.fluid] > 0)):
            raise ValueError("viscosity must be positive in fluid cells")
        mu[self.solid] = 1.0
        self.viscosity = mu
        self.openings = list(openings)
        # Operators are assembled in nondimensional form (lengths / L, viscosity / mu_ref); SI scaling
        # (cell volumes ~1e-15 m^3) otherwise defeats the iterative solver's stopping tests.
        self.length_scale = float(max(grid.yf[-1] - grid.yf[0], grid.zf[-1] - grid.zf[0]))
        self.viscosity_scale = float(np.median(mu[self.fluid])) if self.fluid.any() else 1.0
        L = self.length_scale
        self.op_grid = RectilinearGrid(grid.xf / L, grid.yf / L, grid.zf / L)
        self.op_mu = mu / self.viscosity_scale

        # ---- face bookkeeping
        self.status = [_face_status(grid, self.fluid, c) for c in range(3)]
        sizes = [s.size for s in self.status]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        self.n_faces = int(self.offsets[-1])
        self.face_index = [
            (self.offsets[c] + np.arange(sizes[c])).reshape(self.status[c].shape) for c in range(3)
        ]
        flat_status = np.concatenate([s.ravel() for s in self.status])
        self.unknown = np.nonzero(flat_status == UNKNOWN)[0]
        self.known = np.nonzero(flat_status == DIRICHLET)[0]
        for opening in self.openings:
            axis, side = WALL_AXIS[opening.wall]
            wall_status = np.take(self.status[axis], [side], axis=axis).squeeze(axis)
            if np.any((opening.profile != 0) & (wall_status != DIRICHLET)):
                raise ValueError(f"Opening {opening.label!r} covers faces that are blocked by a solid.")

        # ---- operators on the full face vector
        cells = np.nonzero(self.fluid.ravel())[0]
        self.pressure_cells = cells
        normal = self._normal_strain_operator(cells)
        shear = self._shear_strain_operator()
        full = (normal[0].T @ sp.diags(normal[1]) @ normal[0]) * 2.0 + (shear[0].T @ sp.diags(shear[1]) @ shear[0]) * 4.0
        full = full.tocsr()
        self.A = full[self.unknown][:, self.unknown].tocsr()
        self.A_known = full[self.unknown][:, self.known].tocsr()
        divergence = self._divergence_operator(cells).tocsc()
        self.B = divergence[:, self.unknown].tocsr()
        self.B_known = divergence[:, self.known].tocsr()
        self.divergence_full = divergence.tocsr()
        # pin the pressure of the last fluid cell (removes the constant null space)
        self.B_pinned = self.B[:-1]
        self.B_known_pinned = self.B_known[:-1]
        n_u, n_p = self.A.shape[0], self.B_pinned.shape[0]
        self.n_velocity, self.n_pressure = n_u, n_p
        self.K = sp.bmat([[self.A, -self.B_pinned.T], [-self.B_pinned, None]], format="csc")
        mu_cells = self.op_mu.ravel()[cells[:-1]]
        volumes = self.op_grid.cell_volume.ravel()[cells[:-1]]
        self.pressure_scaling = volumes / mu_cells
        self._direct = None
        self._precond = None
        self.timings = {"assembly": time.perf_counter() - started}

    # ------------------------------------------------------------------ operators
    def _normal_strain_operator(self, cells: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
        grid = self.op_grid
        nx, ny, nz = grid.shape
        ci, cj, ck = np.unravel_index(cells, grid.shape)
        blocks = []
        for c in range(3):
            width = grid.widths[c][(ci, cj, ck)[c]]
            lower = [ci, cj, ck]
            upper = [ci.copy(), cj.copy(), ck.copy()]
            upper[c] = upper[c] + 1
            low_idx = self.face_index[c][tuple(lower)]
            up_idx = self.face_index[c][tuple(upper)]
            rows = np.arange(len(cells))
            blocks.append(
                sp.csr_matrix(
                    (np.concatenate([-1.0 / width, 1.0 / width]), (np.concatenate([rows, rows]),
                                                                   np.concatenate([low_idx, up_idx]))),
                    shape=(len(cells), self.n_faces),
                )
            )
        weight = self.op_mu.ravel()[cells] * grid.cell_volume.ravel()[cells]
        return sp.vstack(blocks).tocsr(), np.tile(weight, 3)

    def _shear_strain_operator(self) -> tuple[sp.csr_matrix, np.ndarray]:
        grid = self.op_grid
        rows_all, cols_all, vals_all, weights_all = [], [], [], []
        row_offset = 0
        for a, b in ((0, 1), (0, 2), (1, 2)):
            t = 3 - a - b
            perm = (a, b, t)
            Na, Nb, Nt = (grid.shape[i] for i in perm)
            fluid = np.transpose(self.fluid, perm)
            mu = np.transpose(self.op_mu, perm)
            ha, hb, ht = (grid.widths[i] for i in perm)
            ca, cb = grid.centers[a], grid.centers[b]
            fa_c, fb_c = grid.faces[a], grid.faces[b]
            # face indices/status of component a (shape Na+1, Nb, Nt) and b (Na, Nb+1, Nt)
            idx_a = np.transpose(self.face_index[a], perm)
            st_a = np.transpose(self.status[a], perm)
            idx_b = np.transpose(self.face_index[b], perm)
            st_b = np.transpose(self.status[b], perm)
            # quarter volumes of the 4 cells around each edge (Na+1, Nb+1, Nt)
            fluid_pad = np.pad(fluid, ((1, 1), (1, 1), (0, 0)), constant_values=False)
            mu_pad = np.pad(mu, ((1, 1), (1, 1), (0, 0)), constant_values=1.0)
            ha_pad = np.concatenate([[0.0], ha, [0.0]])
            hb_pad = np.concatenate([[0.0], hb, [0.0]])
            vol = np.zeros((Na + 1, Nb + 1, Nt))
            inv_mu = np.zeros((Na + 1, Nb + 1, Nt))
            for da in (0, 1):
                for db in (0, 1):
                    f = fluid_pad[da:da + Na + 1, db:db + Nb + 1]
                    q = 0.25 * ha_pad[da:da + Na + 1][:, None, None] * hb_pad[db:db + Nb + 1][None, :, None] * ht[None, None, :]
                    q = np.where(f, q, 0.0)
                    vol += q
                    inv_mu += q / mu_pad[da:da + Na + 1, db:db + Nb + 1]
            edge_ok = vol > 0
            mu_edge = np.where(edge_ok, vol / np.where(edge_ok, inv_mu, 1.0), 0.0)

            coeffs: list[tuple[np.ndarray, np.ndarray]] = []  # (face index, coefficient) per edge
            # d f_a / d b : faces a at (ia, ib-1) and (ia, ib)
            pad_idx = np.pad(idx_a, ((0, 0), (1, 1), (0, 0)), constant_values=-1)
            pad_st = np.pad(st_a, ((0, 0), (1, 1), (0, 0)), constant_values=GHOST)
            coeffs.extend(self._tangential_derivative(pad_idx, pad_st, axis=1, centers=cb, faces=fb_c, n_other=Na + 1, Nt=Nt))
            # d f_b / d a : faces b at (ia-1, ib) and (ia, ib)
            pad_idx = np.pad(idx_b, ((1, 1), (0, 0), (0, 0)), constant_values=-1)
            pad_st = np.pad(st_b, ((1, 1), (0, 0), (0, 0)), constant_values=GHOST)
            coeffs.extend(self._tangential_derivative(pad_idx, pad_st, axis=0, centers=ca, faces=fa_c, n_other=Nb + 1, Nt=Nt))

            edge_ids = np.nonzero(edge_ok.ravel())[0]
            local_row = -np.ones(edge_ok.size, dtype=np.int64)
            local_row[edge_ids] = np.arange(len(edge_ids)) + row_offset
            for face_idx, coefficient in coeffs:
                face_flat = face_idx.ravel()
                coef_flat = 0.5 * coefficient.ravel()
                mask = (local_row >= 0) & (face_flat >= 0) & (coef_flat != 0)
                rows_all.append(local_row[mask])
                cols_all.append(face_flat[mask])
                vals_all.append(coef_flat[mask])
            weights_all.append((mu_edge * vol).ravel()[edge_ids])
            row_offset += len(edge_ids)
        matrix = sp.csr_matrix(
            (np.concatenate(vals_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
            shape=(row_offset, self.n_faces),
        )
        return matrix, np.concatenate(weights_all)

    @staticmethod
    def _tangential_derivative(pad_idx, pad_st, axis, centers, faces, n_other, Nt):
        """Coefficients of d f / d s at edges, s along ``axis`` of the padded (a, b, t) array.

        Returns two (face index, coefficient) pairs of edge shape. Ghost neighbours
        (outside the domain or inside a solid) are eliminated by reflection about
        the wall, which lies at the edge position.
        """
        n_edges_s = len(faces)
        lower = np.take(pad_idx, np.arange(0, n_edges_s), axis=axis)
        upper = np.take(pad_idx, np.arange(1, n_edges_s + 1), axis=axis)
        lower_st = np.take(pad_st, np.arange(0, n_edges_s), axis=axis)
        upper_st = np.take(pad_st, np.arange(1, n_edges_s + 1), axis=axis)
        c_pad = np.concatenate([[np.nan], centers, [np.nan]])
        full_dist = c_pad[1:n_edges_s + 1] - c_pad[0:n_edges_s]
        up_half = c_pad[1:n_edges_s + 1] - faces
        low_half = faces - c_pad[0:n_edges_s]
        shape = [1, 1, 1]
        shape[axis] = n_edges_s
        full_dist, up_half, low_half = (d.reshape(shape) for d in (full_dist, up_half, low_half))
        lo_ok = lower_st != GHOST
        up_ok = upper_st != GHOST
        both = lo_ok & up_ok
        with np.errstate(divide="ignore", invalid="ignore"):
            coef_up = np.where(both, 1.0 / full_dist, np.where(up_ok & ~lo_ok, 1.0 / up_half, 0.0))
            coef_lo = np.where(both, -1.0 / full_dist, np.where(lo_ok & ~up_ok, -1.0 / low_half, 0.0))
        coef_up = np.nan_to_num(coef_up)
        coef_lo = np.nan_to_num(coef_lo)
        return [(np.where(up_ok, upper, -1), coef_up), (np.where(lo_ok, lower, -1), coef_lo)]

    def _divergence_operator(self, cells: np.ndarray) -> sp.csr_matrix:
        grid = self.op_grid
        ci, cj, ck = np.unravel_index(cells, grid.shape)
        rows, cols, vals = [], [], []
        hx, hy, hz = grid.widths
        areas = (hy[cj] * hz[ck], hx[ci] * hz[ck], hx[ci] * hy[cj])
        r = np.arange(len(cells))
        for c in range(3):
            lower = [ci, cj, ck]
            upper = [ci.copy(), cj.copy(), ck.copy()]
            upper[c] = upper[c] + 1
            rows.extend([r, r])
            cols.extend([self.face_index[c][tuple(lower)], self.face_index[c][tuple(upper)]])
            vals.extend([-areas[c], areas[c]])
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(cells), self.n_faces)
        )

    # ------------------------------------------------------------------ boundary data
    def boundary_values(
        self,
        inflow: np.ndarray | None = None,
        outflow: np.ndarray | None = None,
        opening_rates: Sequence[float] = (),
    ) -> np.ndarray:
        """Full face vector of prescribed values (zero on no-slip faces).

        ``inflow``/``outflow`` are normal velocities of shape ``(ny, nz)`` on the
        x = x0 and x = x1 planes; ``opening_rates`` scale each opening's
        unit-flux profile.
        """
        values = np.zeros(self.n_faces)
        if inflow is not None:
            values[self.face_index[0][0].ravel()] = np.where(self.status[0][0] == DIRICHLET, inflow, 0.0).ravel()
        if outflow is not None:
            values[self.face_index[0][-1].ravel()] = np.where(self.status[0][-1] == DIRICHLET, outflow, 0.0).ravel()
        for opening, rate in zip(self.openings, opening_rates):
            axis, side = WALL_AXIS[opening.wall]
            sign = 1.0 if side == 0 else -1.0  # inward normal velocity -> signed component
            idx = np.take(self.face_index[axis], [side], axis=axis).squeeze(axis)
            values[idx.ravel()] += (sign * rate * opening.profile).ravel()  # several openings may share a wall
        return values

    # ------------------------------------------------------------------ solving
    def _choose(self, method: str) -> str:
        if method != "auto":
            return method
        # SuperLU fill-in on 3D saddle-point systems grows quickly; measured
        # ~0.7 s at 12k unknowns but ~270 s at 95k, so only small systems go direct.
        return "direct" if self.K.shape[0] <= 15_000 else "minres"

    def solve(
        self,
        known_values: np.ndarray,
        method: Literal["auto", "direct", "minres"] = "auto",
        tolerance: float = 1e-12,
        cancel: Callable[[], bool] | None = None,
        force: np.ndarray | None = None,
        initial: StokesSolution | None = None,
        velocity_scale: float | None = None,
        max_residual: float | None = None,
    ) -> StokesSolution:
        """Solve for the prescribed boundary values.

        ``force`` is a full face vector of body forces integrated over the momentum control
        volumes (N), for example the lagged convective term of a Picard iteration. ``initial``
        warm-starts the iterative solver. ``velocity_scale`` sets the nondimensional velocity
        unit when all boundary values vanish (a force-driven correction). ``max_residual`` is the
        accepted true relative residual (default ``max(1e-5, 1e3 * tolerance)``); corrections that
        only need accuracy relative to their own size may accept more.
        """
        started = time.perf_counter()
        velocity_scale = velocity_scale or float(np.max(np.abs(known_values))) or 1.0
        known_nd = known_values / velocity_scale
        x_known = known_nd[self.known]
        f = -(self.A_known @ x_known)
        if force is not None:
            f = f + np.asarray(force, dtype=float)[self.unknown] / (
                self.viscosity_scale * self.length_scale * velocity_scale)
        g = self.B_known_pinned @ x_known
        rhs = np.concatenate([f, g])
        method = self._choose(method)
        iterations = 0
        if method == "direct":
            if self._direct is None:
                t0 = time.perf_counter()
                self._direct = splu(self.K, permc_spec="COLAMD", diag_pivot_thresh=1.0)
                self.timings["factorization"] = time.perf_counter() - t0
            solution = self._direct.solve(rhs)
        elif method == "minres":
            M = self._preconditioner()
            counter = {"n": 0}

            def callback(_: np.ndarray) -> None:
                counter["n"] += 1
                if cancel is not None and counter["n"] % 10 == 0 and cancel():
                    raise SolveCancelled()

            x0 = None if initial is None else self._initial_vector(initial, velocity_scale)
            solution, status = minres(self.K, rhs, x0=x0, M=M, rtol=tolerance, maxiter=5000, callback=callback)
            iterations = counter["n"]
            if status != 0:
                raise RuntimeError(f"MINRES did not converge (status {status})")
        else:
            raise ValueError(f"Unknown linear solver {method!r}")
        residual = float(np.linalg.norm(self.K @ solution - rhs) / max(np.linalg.norm(rhs), 1e-300))
        if residual > (max_residual if max_residual is not None else max(1e-5, 1e3 * tolerance)):
            raise RuntimeError(f"Stokes solve failed: relative residual {residual:.2e} after {iterations} iterations")
        full = known_nd.copy()
        full[self.unknown] = solution[: self.n_velocity]
        divergence = self.divergence_full @ full
        flux_scale = max(float(np.max(np.abs(full))) * float(np.max(self.op_grid.face_area(0))), 1e-300)
        full *= velocity_scale
        pressure_scale = self.viscosity_scale * velocity_scale / self.length_scale
        pressure = np.full(self.grid.shape, np.nan)
        p_flat = np.concatenate([solution[self.n_velocity:], [0.0]]) * pressure_scale
        pressure.ravel()[self.pressure_cells] = p_flat
        outlet = pressure[-1][self.fluid[-1]]
        if outlet.size:
            pressure -= float(np.mean(outlet))
        u = full[self.offsets[0]:self.offsets[1]].reshape(self.grid.face_shape(0))
        v = full[self.offsets[1]:self.offsets[2]].reshape(self.grid.face_shape(1))
        w = full[self.offsets[2]:self.offsets[3]].reshape(self.grid.face_shape(2))
        info = {
            "method": method,
            "iterations": iterations,
            "relative_residual": residual,
            "max_cell_divergence": float(np.max(np.abs(divergence))) / flux_scale,
            "unknowns": int(self.K.shape[0]),
            "solve_seconds": time.perf_counter() - started,
            "assembly_seconds": self.timings.get("assembly", 0.0),
            "factorization_seconds": self.timings.get("factorization", 0.0),
        }
        return StokesSolution(self.grid, self.solid, u, v, w, pressure, info)

    def _initial_vector(self, initial: StokesSolution, velocity_scale: float) -> np.ndarray:
        """Nondimensional unknown vector (velocities, pinned pressures) of a solution on this grid."""
        full = np.concatenate([initial.u.ravel(), initial.v.ravel(), initial.w.ravel()]) / velocity_scale
        pressure = initial.p.ravel()[self.pressure_cells]
        pressure = (pressure - pressure[-1]) / (self.viscosity_scale * velocity_scale / self.length_scale)
        return np.concatenate([full[self.unknown], pressure[:-1]])

    def _preconditioner(self) -> LinearOperator:
        """Block-diagonal preconditioner: velocity blocks (AMG V-cycle or exact LU) and scaled pressure mass."""
        if self._precond is None:
            t0 = time.perf_counter()
            component = np.concatenate([np.full(s.size, c) for c, s in enumerate(self.status)])[self.unknown]
            blocks = []
            for c in range(3):
                sel = np.nonzero(component == c)[0]
                if len(sel):
                    blocks.append((sel, _block_solver(self.A[sel][:, sel])))
            inv_scaling = 1.0 / self.pressure_scaling
            n_u = self.n_velocity

            def apply(r: np.ndarray) -> np.ndarray:
                out = np.empty_like(r)
                for sel, solve_block in blocks:
                    out[sel] = solve_block(r[sel])
                out[n_u:] = r[n_u:] * inv_scaling
                return out

            self._precond = LinearOperator(self.K.shape, matvec=apply, dtype=float)
            self.timings["preconditioner"] = time.perf_counter() - t0
            self.timings["preconditioner_kind"] = "amg" if _pyamg() is not None else "lu"
        return self._precond


def _pyamg():
    try:
        import pyamg  # optional accelerator, measured 4-5x faster setup+solve than block LU
    except ImportError:  # pragma: no cover - exercised only without pyamg
        return None
    return pyamg


def _block_solver(block: sp.spmatrix) -> Callable[[np.ndarray], np.ndarray]:
    pyamg = _pyamg()
    if pyamg is not None and block.shape[0] > 2000:
        hierarchy = pyamg.ruge_stuben_solver(block.tocsr())
        return lambda r: hierarchy.solve(r, maxiter=1, cycle="V", tol=1e-30)
    lu = splu(block.tocsc(), permc_spec="COLAMD")
    return lu.solve


def uniform_faces(start: float, stop: float, n: int) -> np.ndarray:
    return np.linspace(start, stop, n + 1)


def graded_faces(start: float, stop: float, target: float, refine: Sequence[tuple[float, float]] = (),
                 fine: float | None = None, growth: float = 1.25) -> np.ndarray:
    """1D faces with spacing ``fine`` inside the ``refine`` intervals, growing geometrically to ``target``.

    Exact interval end points (e.g. pillar tops, port edges) can be inserted by
    listing zero-length intervals ``(a, a)``.
    """
    fine = target if fine is None else fine
    if not refine:
        n = max(1, int(round((stop - start) / target)))
        return np.linspace(start, stop, n + 1)
    # desired local spacing function sampled finely, then integrated
    s = np.linspace(start, stop, 4001)
    distance = np.full_like(s, np.inf)
    for a, b in refine:
        distance = np.minimum(distance, np.where((s >= a) & (s <= b), 0.0, np.minimum(np.abs(s - a), np.abs(s - b))))
    # spacing grows geometrically with distance measured in fine cells
    steps = distance / fine
    spacing = np.minimum(fine * growth ** np.maximum(steps / 2.0, 0.0), target)
    density = 1.0 / spacing
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(s))])
    n = max(1, int(np.ceil(cumulative[-1])))
    targets = np.linspace(0.0, cumulative[-1], n + 1)
    return np.interp(targets, cumulative, s)


def snap_faces(faces: np.ndarray, points: Sequence[float], min_fraction: float = 0.3) -> np.ndarray:
    """Move the nearest interior face onto each point (so heights align with faces)."""
    faces = faces.copy()
    for point in points:
        if point <= faces[0] or point >= faces[-1]:
            continue
        i = int(np.argmin(np.abs(faces - point)))
        if i in (0, len(faces) - 1):
            continue
        faces[i] = point
    faces = np.unique(faces)
    widths = np.diff(faces)
    keep = np.concatenate([[True], widths > min_fraction * np.median(widths)])
    keep[-1] = True
    return faces[keep] if keep.sum() >= 2 else faces
