"""Generate the small analytical reference fixtures in ``reference_data/analytical/``.

These verify the reference pipeline and the flux/viscosity partition against
closed-form solutions. They are *not* external validation data.

1. ``layered_viscosity_wide_*``: two layers of different viscosity in a wide
   channel (W = 12 H). Away from the side walls the flow is plane Poiseuille
   flow; the interface height that splits the flux in the prescribed ratio
   follows from the layered solution and is compared along the central half
   of the width.
2. ``lateral_layers_equal``: three side-by-side equal-viscosity layers in a
   straight channel; the partition is evaluated with the exact duct series.

Run from the repository root: ``python tools/make_reference_fixtures.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crossflow_simulator import physics  # noqa: E402
from crossflow_simulator.calibration import ReferenceWriter  # noqa: E402
from crossflow_simulator.config import (  # noqa: E402
    ChannelGeometry, FluidProperties, GridSpec, InitialPattern, InletSpec, ProjectConfig,
)

UM, UL_MIN = 1e-6, 1e-9 / 60


def plane_two_layer_flux(h1: float, H: float, mu1: float, mu2: float) -> tuple[float, float]:
    """Fluxes per unit width (G = 1) of a two-layer plane Poiseuille flow with interface at h1."""
    M = np.array([[h1, -h1, -1.0], [mu1, -mu2, 0.0], [0.0, H, 1.0]])
    rhs = np.array([h1**2 / (2 * mu1) - h1**2 / (2 * mu2), 0.0, H**2 / (2 * mu2)])
    c1, c2, d2 = np.linalg.solve(M, rhs)
    q1 = -h1**3 / (6 * mu1) + c1 * h1**2 / 2
    q2 = (-(H**3 - h1**3) / (6 * mu2) + c2 * (H**2 - h1**2) / 2 + d2 * (H - h1))
    return q1, q2


def main() -> int:
    writer = ReferenceWriter(ROOT / "reference_data" / "analytical")
    orientation = {"x": "streamwise", "y": "lateral from the left wall looking downstream", "z": "from the bottom"}
    W, H = 1200 * UM, 100 * UM
    for ratio, split in ((1.0, 0.5), (4.0, 0.5), (0.25, 0.3)):
        mu2 = 0.01
        mu1 = ratio * mu2
        fraction = split
        h1 = brentq(lambda h: plane_two_layer_flux(h, H, mu1, mu2)[0] / sum(plane_two_layer_flux(h, H, mu1, mu2))
                    - fraction, 1e-9 * H, H * (1 - 1e-9))
        cfg = ProjectConfig(
            name=f"two layers, viscosity ratio {ratio}", channel=ChannelGeometry(W, H),
            materials=[FluidProperties("A", mu1), FluidProperties("B", mu2)],
            inlets=[InletSpec("in1", "A", "pattern", flow_rate=fraction * 2 * UL_MIN, pattern_row=0),
                    InletSpec("in2", "B", "pattern", flow_rate=(1 - fraction) * 2 * UL_MIN, pattern_row=1)],
            pattern=InitialPattern(layout="layers_z"), grid=GridSpec(output_ny=240, output_nz=80),
        )
        y = np.linspace(0.25 * W, 0.75 * W, 41)
        writer.write(f"layered_viscosity_wide_{str(ratio).replace('.', 'p')}", {
            "source": {"kind": "analytical", "tool": "closed-form two-layer plane Poiseuille flow",
                       "citation": "textbook layered channel flow", "license": "n/a", "reuse_permitted": True},
            "config": cfg.to_dict(), "plane": {"x_m": 0.0, "description": "inlet pattern (developed flow)"},
            "quantity": "interface", "lower_codes": [1], "orientation": orientation,
            "family": f"layered_viscosity_ratio_{ratio:g}", "split": "holdout" if ratio == 0.25 else "train",
            "numerics": {"note": "exact in the wide-channel limit; side walls excluded (central half of W)"},
            "uncertainty": {"interface_m": 0.5 * UM},
        }, {"y_m": y, "z_interface_m": np.full_like(y, h1)})

    cfg = ProjectConfig(
        name="three lateral layers", channel=ChannelGeometry(300 * UM, 150 * UM),
        materials=[FluidProperties("A", 0.02), FluidProperties("B", 0.02), FluidProperties("C", 0.02)],
        inlets=[InletSpec(f"in{i + 1}", "ABC"[i], "pattern", flow_rate=r * UL_MIN, pattern_col=i)
                for i, r in enumerate((1.0, 2.0, 3.0))],
        pattern=InitialPattern(layout="layers_y"), grid=GridSpec(output_ny=150),
    )
    ny, nz = 150, 75
    y_edges, z_edges = np.linspace(0, 300 * UM, ny + 1), np.linspace(0, 150 * UM, nz + 1)
    fine = 3000
    yf = (np.arange(fine) + 0.5) * 300 * UM / fine
    zf = (np.arange(fine // 2) + 0.5) * 150 * UM / (fine // 2)
    u = physics.rectangular_duct_profile(yf, zf, 300 * UM, 150 * UM, normalization="none")
    cumulative = np.cumsum(u.sum(axis=0))
    cumulative /= cumulative[-1]
    bounds = np.interp([1 / 6, 3 / 6], cumulative, yf)
    yc = 0.5 * (y_edges[1:] + y_edges[:-1])
    labels = np.where(yc < bounds[0], 1, np.where(yc < bounds[1], 2, 3))[None, :].repeat(nz, axis=0)
    writer.write("lateral_layers_equal", {
        "source": {"kind": "analytical", "tool": "rectangular duct series, 3000-point quadrature",
                   "citation": "fully developed laminar duct flow", "license": "n/a", "reuse_permitted": True},
        "config": cfg.to_dict(), "plane": {"x_m": 0.0, "description": "inlet pattern"}, "quantity": "labels",
        "orientation": orientation, "family": "lateral_layers", "numerics": {"quadrature_points": fine},
    }, {"labels": labels.astype(np.int32), "y_edges_m": y_edges, "z_edges_m": z_edges})
    print(f"Wrote {writer.written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
