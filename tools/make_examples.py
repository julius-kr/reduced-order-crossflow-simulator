"""Generate the representative example projects in ``examples/``.

Representative cases from work package A: empty channel, three serial inlets,
unequal parallel inlets, unequal 2x2 array, off-center bottom-mounted right
triangle, its reflected and top-mounted variants, and two nearby pillars.
New-solver-only examples cover a viscosity contrast and pressure-driven flow.

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
