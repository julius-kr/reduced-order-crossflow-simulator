"""Unit conversion at the input/output boundary.

All numerical code works in SI units. Project files store SI numbers under
unit-suffixed keys (``width_m``); for convenience the loader also accepts
strings such as ``"300 um"`` or ``"10 uL/min"``.
"""
from __future__ import annotations

import math
import re

UNITS: dict[str, dict[str, float]] = {
    "length": {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9},
    "flow_rate": {
        "m3/s": 1.0,
        "mL/min": 1e-6 / 60.0,
        "mL/h": 1e-6 / 3600.0,
        "uL/min": 1e-9 / 60.0,
        "µL/min": 1e-9 / 60.0,
        "uL/h": 1e-9 / 3600.0,
        "µL/h": 1e-9 / 3600.0,
        "nL/min": 1e-12 / 60.0,
    },
    "viscosity": {"Pa s": 1.0, "Pa.s": 1.0, "mPa s": 1e-3, "mPa.s": 1e-3, "cP": 1e-3},
    "density": {"kg/m3": 1.0, "g/cm3": 1000.0, "g/mL": 1000.0},
    "pressure": {"Pa": 1.0, "kPa": 1e3, "mbar": 100.0, "bar": 1e5, "psi": 6894.757293168},
    "diffusivity": {"m2/s": 1.0, "um2/s": 1e-12, "µm2/s": 1e-12},
    "resistance": {"Pa s/m3": 1.0, "mbar/(uL/min)": 100.0 / (1e-9 / 60.0)},
    "temperature": {"K": 1.0},
    "angle": {"deg": 1.0},
}

SI_UNIT = {
    "length": "m",
    "flow_rate": "m3/s",
    "viscosity": "Pa s",
    "density": "kg/m3",
    "pressure": "Pa",
    "diffusivity": "m2/s",
    "resistance": "Pa s/m3",
    "temperature": "K",
    "angle": "deg",
}

# JSON key suffix used for each quantity kind.
KEY_SUFFIX = {
    "length": "m",
    "flow_rate": "m3_s",
    "viscosity": "Pa_s",
    "density": "kg_m3",
    "pressure": "Pa",
    "diffusivity": "m2_s",
    "resistance": "Pa_s_m3",
    "temperature": "K",
    "angle": "deg",
}

_QUANTITY = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


def to_si(value: float, unit: str, kind: str) -> float:
    try:
        factor = UNITS[kind][unit]
    except KeyError:
        known = ", ".join(UNITS.get(kind, {}))
        raise ValueError(f"Unknown {kind} unit {unit!r}; expected one of: {known}") from None
    return float(value) * factor


def from_si(value: float, unit: str, kind: str) -> float:
    return float(value) / UNITS[kind][unit]


def parse_quantity(text: str | float | int, kind: str) -> float:
    """Parse ``"300 um"`` (or a bare SI number) into an SI float."""
    if isinstance(text, (int, float)):
        value = float(text)
    else:
        match = _QUANTITY.match(text)
        if not match:
            raise ValueError(f"Cannot parse {kind} quantity {text!r}")
        number, unit = match.groups()
        value = to_si(float(number), unit or SI_UNIT[kind], kind)
    if not math.isfinite(value):
        raise ValueError(f"{kind} must be finite, got {text!r}")
    return value


def format_quantity(value_si: float | None, unit: str, kind: str, digits: int = 4) -> str:
    if value_si is None:
        return "—"
    return f"{from_si(value_si, unit, kind):.{digits}g} {unit}"
