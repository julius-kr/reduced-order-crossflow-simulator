"""Result view: final section, CFD differences, and run status."""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..materials import colors_for
from ..plotter import stream_labels
from ..results import SimulationResult
from .canvas import SectionCanvas
from .qt import QtWidgets, Signal


# The new solver's fidelity presets as the app names them (Accuracy), from fastest to most accurate.
ACCURACY = {"preview": "Low", "medium": "Medium", "reference": "High"}
# The results the new solver's sections can be differenced against; the comparison controller supplies their codes.
DIFFERENCES = {"cfd": ("Difference to CFD", "CFD reference", "Compare with CFD"),
               "legacy_cfd": ("Difference to legacy CFD", "legacy CFD", "Compare with legacy CFD")}


class ResultsView(QtWidgets.QGroupBox):
    export_requested = Signal()
    visibility_changed = Signal(object)  # set of hidden stream codes
    difference_changed = Signal(str)  # "cfd", "legacy_cfd", or "" for none

    def __init__(self, parent=None):
        super().__init__("New solver", parent)
        self.result: SimulationResult | None = None
        # (name, callable station index -> codes or None) of the result the difference is shown against
        self.reference: tuple[str, Callable[[int], np.ndarray | None]] | None = None
        self.hidden: set[int] = set()
        self.length_unit = "um"
        self.color_by = "material"
        self.stale = False
        self.station: int | None = None
        left = QtWidgets.QVBoxLayout(self)
        left.setContentsMargins(4, 4, 4, 4)
        self.status = QtWidgets.QLabel("Not run")
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        differences = QtWidgets.QHBoxLayout()
        self.differences: dict[str, QtWidgets.QCheckBox] = {}
        for name, (label, _, _) in DIFFERENCES.items():
            box = self.differences[name] = QtWidgets.QCheckBox(label)
            box.toggled.connect(lambda checked, n=name: self._difference_toggled(n, checked))
            self.set_difference_available(name, False)
            differences.addWidget(box)
        left.addLayout(differences)
        self.canvas = SectionCanvas(self)
        self.canvas.setMinimumSize(200, 150)
        left.addWidget(self.canvas, 1)
        exports = QtWidgets.QHBoxLayout()
        self.export_button = QtWidgets.QPushButton("Export...")
        self.export_button.clicked.connect(self.export_requested.emit)
        exports.addWidget(self.export_button)
        exports.addStretch(1)
        left.addLayout(exports)

    # ------------------------------------------------------------------ data
    def show_result(self, result: SimulationResult, length_unit: str, color_by: str) -> None:
        self.result = result
        self.length_unit = length_unit
        self.color_by = color_by
        self.stale = False
        self.station = None
        self.status.setText(f"{result.status}: {result.message}")
        self.status.setStyleSheet(f"color: {'#1B5E20' if result.ok else '#B00020'}")
        self.redraw()

    def mark_stale(self, stale: bool) -> None:
        if stale != self.stale:
            self.stale = stale
            if self.result is not None:
                prefix = "STALE (inputs changed): " if stale else ""
                self.status.setText(f"{prefix}{self.result.status}: {self.result.message}")
                self.status.setStyleSheet(f"color: {'#B00020' if stale or not self.result.ok else '#1B5E20'}")
            self.redraw()

    def show_message(self, text: str) -> None:
        self.status.setText(text)
        self.status.setStyleSheet("")
        self.canvas.clear_message(text)

    # ------------------------------------------------------------------ differences
    def _difference_toggled(self, name: str, checked: bool) -> None:
        if checked:  # one difference at a time
            for other, box in self.differences.items():
                if other != name and box.isChecked():
                    box.blockSignals(True)
                    box.setChecked(False)
                    box.blockSignals(False)
        self.difference_changed.emit(self.difference_choice())

    def difference_choice(self) -> str:
        """The result the difference is asked against ("cfd", "legacy_cfd"), or "" for none."""
        return next((name for name, box in self.differences.items() if box.isChecked() and box.isEnabled()), "")

    def set_difference_available(self, name: str, available: bool) -> None:
        """Enable a difference box while its comparison is on; switching the comparison off unchecks it."""
        label, result, action = DIFFERENCES[name]
        box = self.differences[name]
        box.setEnabled(available)
        if not available and box.isChecked():
            box.blockSignals(True)
            box.setChecked(False)
            box.blockSignals(False)
        box.setToolTip(f"Mark the cells where the new result differs from the {result} at this station"
                       if available else f"Turn on {action} in the toolbar to show the difference to the {result}")

    def use_reference(self, name: str, codes: Callable[[int], np.ndarray | None]) -> None:
        """Mark where the new result differs from ``codes(station index)`` (None: nothing to compare there)."""
        self.reference = (name, codes)
        self.redraw()

    def reference_codes(self, index: int) -> np.ndarray | None:
        if self.reference is None:
            return None
        codes = self.reference[1](index)
        return None if codes is None else np.asarray(codes)

    def clear_reference(self) -> None:
        self.reference = None
        self.redraw()

    def set_station(self, index: int) -> None:
        self.station = index
        self.redraw()

    def redraw(self) -> None:
        if self.result is None:
            return
        if not self.result.ok or not self.result.sections:
            self.canvas.clear_message(f"{self.result.status}: {self.result.message}")
            return
        index = len(self.result.sections) - 1 if self.station is None else min(
            max(self.station, 0), len(self.result.sections) - 1)
        section = self.result.sections[index]
        reference = self.reference_codes(index)
        self.canvas.show_section(section, colors_for(self.result.streams, self.color_by), stream_labels(self.result),
                                 self.length_unit, self.hidden, True, reference,
                                 title=section.name, stale=self.stale)
