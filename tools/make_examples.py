"""Generate the representative example projects in ``examples/``.

Representative cases from work package A: empty channel, three serial inlets,
unequal parallel inlets, unequal 2x2 array, off-center bottom-mounted right
triangle, its reflected and top-mounted variants, and two nearby pillars.
New-solver-only examples cover a viscosity contrast and pressure-driven flow. Five square-channel
examples use pillars of the legacy CFD map library, so all four solvers can run them.

Run from the repository root: ``python tools/make_examples.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crossflow_simulator.config import (  # noqa: E402
    ChannelGeometry, FluidProperties, InitialPattern, InletSpec, OperatingConditions, PillarSpec, ProjectConfig,
)

UM = 1e-6
UL_MIN = 1e-9 / 60.0
MPAS = 1e-3
CHANNEL = ChannelGeometry(width=300 * UM, height=150 * UM, length=2500 * UM)
TRIANGLE = [(0.0, 0.0), (150 * UM, 0.0), (0.0, 120 * UM)]  # right angle at the local origin


def fluids(n: int, viscosities: list[float] | None = None) -> list[FluidProperties]:
    names = ["resin A", "resin B", "resin C", "resin D"]
    viscosities = viscosities or [20 * MPAS] * n
    return [FluidProperties(id=chr(ord("A") + i), name=names[i], viscosity=viscosities[i], density=1100.0)
            for i in range(n)]


def two_layers(rates=(2.0, 2.0)) -> tuple[list[InletSpec], InitialPattern]:
    inlets = [InletSpec(f"in{i + 1}", "AB"[i], "pattern", flow_rate=r * UL_MIN, pattern_row=i)
              for i, r in enumerate(rates)]
    return inlets, InitialPattern(layout="layers_z")


def examples() -> dict[str, tuple[ProjectConfig, dict]]:
    out: dict[str, tuple[ProjectConfig, dict]] = {}

    out["empty_channel"] = (ProjectConfig(
        name="Empty rectangular channel", channel=CHANNEL, materials=fluids(1),
        inlets=[InletSpec("in1", "A", "pattern", flow_rate=5 * UL_MIN)],
        notes="Single inlet filling a straight channel: the outlet equals the inlet (identity map)."),
        {"shared_capability": True, "family": "baseline"})

    out["serial_three_inlets"] = (ProjectConfig(
        name="Three serial inlets", channel=CHANNEL, materials=fluids(3),
        inlets=[
            InletSpec("in1", "A", "pattern", flow_rate=2 * UL_MIN, diameter=150 * UM),
            InletSpec("in2", "B", "port", flow_rate=2 * UL_MIN, station=600 * UM, wall="top", position=150 * UM,
                      diameter=150 * UM),
            InletSpec("in3", "C", "port", flow_rate=2 * UL_MIN, station=1400 * UM, wall="top", position=150 * UM,
                      diameter=150 * UM),
        ],
        notes="Serial top-wall ports narrower than the channel: curved, nested layers."),
        {"shared_capability": True, "family": "serial_inlets"})

    out["parallel_unequal"] = (ProjectConfig(
        name="Unequal parallel inlets", channel=CHANNEL, materials=fluids(3),
        inlets=[InletSpec(f"in{i + 1}", "ABC"[i], "pattern", flow_rate=r * UL_MIN, pattern_col=i)
                for i, r in enumerate((1.0, 2.0, 3.0))],
        pattern=InitialPattern(layout="layers_y"),
        notes="Side-by-side streams with flow ratio 1:2:3."),
        {"shared_capability": True, "family": "patterns"})

    out["array_2x2_unequal"] = (ProjectConfig(
        name="Unequal 2x2 inlet array", channel=CHANNEL, materials=fluids(4),
        inlets=[InletSpec(f"in{k + 1}", "ABCD"[k], "pattern", flow_rate=r * UL_MIN, pattern_row=k // 2,
                          pattern_col=k % 2) for k, r in enumerate((1.0, 2.0, 3.0, 4.0))],
        pattern=InitialPattern(layout="grid", rows=2, cols=2, partition="auto"),
        notes="Non-separable per-inlet flow rates; the legacy array operator ignores them."),
        {"shared_capability": True, "family": "patterns"})

    inlets, pattern = two_layers()
    out["right_triangle_bottom"] = (ProjectConfig(
        name="Off-center bottom-mounted right triangle", channel=CHANNEL, materials=fluids(2),
        inlets=inlets, pattern=pattern,
        pillars=[PillarSpec("tri", vertices=TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM, mount="bottom")],
        notes="Asymmetric partial-height pillar (h/H = 0.6) off the channel centreline."),
        {"shared_capability": True, "family": "partial_pillars"})

    inlets, pattern = two_layers()
    out["right_triangle_reflected"] = (ProjectConfig(
        name="Reflected right triangle", channel=CHANNEL, materials=fluids(2), inlets=inlets, pattern=pattern,
        pillars=[PillarSpec("tri", vertices=TRIANGLE, height=90 * UM, x=300 * UM, y=260 * UM, mount="bottom",
                            mirror_y=True)],
        notes="Mirror image of right_triangle_bottom about the channel centreline (y -> W - y)."),
        {"shared_capability": True, "family": "partial_pillars"})

    inlets, pattern = two_layers()
    out["right_triangle_top"] = (ProjectConfig(
        name="Top-mounted right triangle", channel=CHANNEL, materials=fluids(2), inlets=inlets, pattern=pattern,
        pillars=[PillarSpec("tri", vertices=TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM, mount="top")],
        notes="Top-mounted variant; the legacy lubrication-divergence model has no top mount."),
        {"shared_capability": False, "family": "partial_pillars"})

    inlets, pattern = two_layers()
    out["two_nearby_pillars"] = (ProjectConfig(
        name="Two nearby pillars", channel=CHANNEL, materials=fluids(2), inlets=inlets, pattern=pattern,
        pillars=[
            PillarSpec("tri1", vertices=TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM),
            PillarSpec("tri2", vertices=TRIANGLE, height=90 * UM, x=600 * UM, y=140 * UM),
        ],
        notes="Pillars 150 um apart interact; they are solved as one coupled group."),
        {"shared_capability": True, "family": "partial_pillars"})

    inlets = [InletSpec("in1", "A", "pattern", flow_rate=2 * UL_MIN, pattern_row=0),
              InletSpec("in2", "B", "pattern", flow_rate=2 * UL_MIN, pattern_row=1)]
    out["viscosity_contrast_layers"] = (ProjectConfig(
        name="Viscosity contrast (experimental)", channel=CHANNEL, materials=fluids(2, [40 * MPAS, 10 * MPAS]),
        inlets=inlets, pattern=InitialPattern(layout="layers_z"),
        pillars=[PillarSpec("tri", vertices=TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM)],
        notes="4:1 viscosity ratio: flux-constrained layers and viscosity-coupled Stokes iteration (experimental)."),
        {"shared_capability": False, "family": "fluid_properties"})

    out["pressure_driven_serial"] = (ProjectConfig(
        name="Pressure-driven serial inlets", channel=CHANNEL, materials=fluids(2),
        operating=OperatingConditions(drive="pressure", outlet_pressure=0.0),
        inlets=[
            InletSpec("in1", "A", "pattern", pressure=3000.0, tubing_resistance=6.0e13),
            InletSpec("in2", "B", "port", pressure=3000.0, tubing_resistance=6.0e13, station=600 * UM,
                      wall="top", position=150 * UM, diameter=150 * UM),
        ],
        notes="Inlet pressures behind tubing resistances; flow rates come from the hydraulic network."),
        {"shared_capability": False, "family": "driving_conditions"})

    # A pillar entry of the published legacy CFD library (ph 1.125, A-B at y 0-0.75, tip 0.75 downstream at
    # y 0, in a 1.5 x 1.5 channel), scaled to 200 um per library unit; the mirrored copy is an entry as well.
    corner = [(0.0, 0.0), (0.0, 150 * UM), (150 * UM, 0.0)]
    out["square_library_pillars"] = (ProjectConfig(
        name="Library pillars in a square channel (legacy CFD)",
        channel=ChannelGeometry(width=300 * UM, height=300 * UM, length=2000 * UM), materials=fluids(3),
        inlets=[InletSpec(f"in{i + 1}", "ABC"[i], "pattern", flow_rate=r * UL_MIN, pattern_col=i)
                for i, r in enumerate((0.5, 5.0, 0.5))],
        pattern=InitialPattern(layout="layers_y"),
        pillars=[PillarSpec("wall", vertices=corner, height=225 * UM, x=600 * UM, y=0.0),
                 PillarSpec("mirrored", vertices=corner, height=225 * UM, x=1300 * UM, y=300 * UM, mirror_y=True)],
        notes="Square channel with a pillar of the legacy CFD map library (three quarters of the height), once at "
              "each side wall; all four solvers can run it."),
        {"shared_capability": True, "family": "partial_pillars"})

    # More library entries (all bottom-mounted, at most W long, so the legacy solver runs them as well).
    square = ChannelGeometry(width=300 * UM, height=300 * UM, length=2000 * UM)
    side_by_side = [InletSpec(f"in{i + 1}", "ABC"[i], "pattern", flow_rate=2 * UL_MIN, pattern_col=i) for i in range(3)]
    out["square_centred_pillar"] = (ProjectConfig(
        name="Centred pillar, three side-by-side layers (legacy CFD)", channel=square, materials=fluids(3),
        inlets=side_by_side, pattern=InitialPattern(layout="layers_y"),
        # library entry ph 0.75, A-B at y 0.375-1.125, tip (4.75, 0.75)
        pillars=[PillarSpec("centre", vertices=[(0.0, 75 * UM), (0.0, 225 * UM), (150 * UM, 150 * UM)],
                            height=150 * UM, x=700 * UM, y=0.0)],
        notes="Square channel; a library pillar half the height, pointing downstream in the middle of the channel, "
              "splits the middle stream. All four solvers can run it."),
        {"shared_capability": True, "family": "partial_pillars"})

    out["square_full_width_wedge"] = (ProjectConfig(
        name="Full-width wedge under two stacked layers (legacy CFD)", channel=square, materials=fluids(2),
        inlets=[InletSpec(f"in{i + 1}", "AB"[i], "pattern", flow_rate=3 * UL_MIN, pattern_row=i) for i in range(2)],
        pattern=InitialPattern(layout="layers_z"),
        # library entry ph 0.5, A-B across the whole width, tip (4.75, 0.75)
        pillars=[PillarSpec("wedge", vertices=[(0.0, 0.0), (0.0, 300 * UM), (150 * UM, 150 * UM)],
                            height=100 * UM, x=700 * UM, y=0.0)],
        notes="Square channel; a library pillar a third of the height spans the whole width as a wedge pointing "
              "downstream, and lifts the lower of two stacked layers. All four solvers can run it."),
        {"shared_capability": True, "family": "partial_pillars"})

    # library entry ph 0.75, A-B at y 0-0.75, tip (4.75, 0); the mirrored copies are entries as well
    wall = [(0.0, 0.0), (0.0, 150 * UM), (150 * UM, 0.0)]
    out["square_pillar_sequence"] = (ProjectConfig(
        name="Pillar sequence on alternating walls (legacy CFD)",
        channel=ChannelGeometry(width=300 * UM, height=300 * UM, length=2800 * UM), materials=fluids(3),
        inlets=side_by_side, pattern=InitialPattern(layout="layers_y"),
        pillars=[PillarSpec(f"p{k + 1}", vertices=wall, height=150 * UM, x=(500 + 500 * k) * UM,
                            y=300 * UM if k % 2 else 0.0, mirror_y=bool(k % 2)) for k in range(4)],
        notes="Square channel; four library pillars, half the height, alternate between the side walls: a "
              "flow-sculpting sequence. The legacy CFD chains four maps; the CFD reference resolves their "
              "interaction. All four solvers can run it."),
        {"shared_capability": True, "family": "partial_pillars"})

    out["square_upstream_pillar_array"] = (ProjectConfig(
        name="Upstream-pointing pillar, 2x2 inlet array (legacy CFD)", channel=square, materials=fluids(4),
        inlets=[InletSpec(f"in{k + 1}", "ABCD"[k], "pattern", flow_rate=1.5 * UL_MIN, pattern_row=k // 2,
                          pattern_col=k % 2) for k in range(4)],
        pattern=InitialPattern(layout="grid", rows=2, cols=2, partition="auto"),
        # library entry ph 1.0, A-B at y 0-1.125, tip (3.4375, 0.5625)
        pillars=[PillarSpec("upstream", vertices=[(0.0, 0.0), (0.0, 225 * UM), (-112.5 * UM, 112.5 * UM)],
                            height=200 * UM, x=800 * UM, y=0.0)],
        notes="Square channel; a library pillar two thirds of the height, pointing upstream across three quarters "
              "of the width, deforms four equal streams from a 2x2 inlet array. All four solvers can run it."),
        {"shared_capability": True, "family": "partial_pillars"})
    return out


def main() -> int:
    target = ROOT / "examples"
    target.mkdir(exist_ok=True)
    index = []
    for name, (cfg, info) in examples().items():
        cfg.save(target / f"{name}.json")
        index.append({"file": f"{name}.json", "name": cfg.name, **info, "notes": cfg.notes})
    (target / "index.json").write_text(json.dumps({"examples": index}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(index)} examples to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
