#this module defines the Inlet class, which represents an inlet stream in a crossflow simulation. Each inlet has normalized coordinates (x, y, z), a volumetric flow rate, and a label to identify the stream in the cross-section. The module also provides utility functions to infer the layout of inlets based on their coordinates and to obtain sorted unique values from a list with tolerance-based grouping.
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LayoutKind = Literal["vertical", "horizontal", "array"]


@dataclass
class Inlet:
    """
    Represents one inlet stream.

    x, y, z are normalized coordinates.
    flow_rate is the volumetric flow rate assigned to this inlet.
    label identifies the stream in the cross-section.
    """

    x: float
    y: float
    z: float
    flow_rate: float
    label: int


def _varies(values: list[float], tolerance: float = 1e-9) -> bool:
    """
    Return True if the values vary by more than the given tolerance.
    """
    return max(values) - min(values) > tolerance


def infer_layout_kind(inlets: list[Inlet]) -> LayoutKind:
    """
    Infer the inlet layout from inlet coordinates.

    Rules:
    - x varies, y and z fixed  -> series along x -> vertical stacking
    - y varies, x and z fixed  -> parallel across y -> horizontal stacking
    - y and z both vary        -> array configuration
    """

    if len(inlets) == 0:
        raise ValueError("At least one inlet is required.")

    x_values = [inlet.x for inlet in inlets]
    y_values = [inlet.y for inlet in inlets]
    z_values = [inlet.z for inlet in inlets]

    x_varies = _varies(x_values)
    y_varies = _varies(y_values)
    z_varies = _varies(z_values)

    if x_varies and not y_varies and not z_varies:
        return "vertical"

    if y_varies and not x_varies and not z_varies:
        return "horizontal"

    if y_varies and z_varies:
        return "array"

    raise ValueError(
        "Could not infer layout. Expected series along x, "
        "parallel along y, or array in the y-z plane."
    )


def sorted_unique(values: list[float], tolerance: float = 1e-9) -> list[float]:
    """
    Return sorted unique values using tolerance-based grouping.
    """
    sorted_values = sorted(values)

    unique_values: list[float] = []

    for value in sorted_values:
        if not unique_values or abs(value - unique_values[-1]) > tolerance:
            unique_values.append(value)

    return unique_values