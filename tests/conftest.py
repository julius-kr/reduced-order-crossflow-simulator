"""Shared test fixtures for the new implementation.

The tests exercise the package under ``src/``. The legacy snapshot is only
ever executed in isolated subprocesses (see ``test_backends.py`` and
``test_legacy_defects.py``); it is never imported here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("CROSSFLOW_DISABLE_DISK_CACHE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

UM = 1e-6
UL_MIN = 1e-9 / 60.0


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    return ROOT / "examples"


def make_config(**overrides):
    """A small two-layer channel, optionally with pillars/ports, for fast tests."""
    from crossflow_simulator.config import (
        ChannelGeometry, FluidProperties, GridSpec, InitialPattern, InletSpec, ProjectConfig,
    )

    rates = overrides.pop("rates", (2.0, 2.0))
    layout = overrides.pop("layout", "layers_z")
    materials = overrides.pop("materials", None) or [FluidProperties("A", 20e-3), FluidProperties("B", 20e-3),
                                                     FluidProperties("C", 20e-3), FluidProperties("D", 20e-3)]
    inlets = overrides.pop("inlets", None)
    if inlets is None:
        inlets = []
        for i, r in enumerate(rates):
            inlets.append(InletSpec(
                f"in{i + 1}", materials[i].id, "pattern", flow_rate=r * UL_MIN,
                pattern_row=i if layout == "layers_z" else 0, pattern_col=i if layout == "layers_y" else 0,
            ))
    cfg = ProjectConfig(
        channel=overrides.pop("channel", ChannelGeometry(width=300 * UM, height=150 * UM)),
        materials=materials,
        inlets=inlets,
        pattern=overrides.pop("pattern", InitialPattern(layout=layout)),
        pillars=overrides.pop("pillars", []),
        grid=overrides.pop("grid", GridSpec(output_ny=96)),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def config_factory():
    return make_config
