"""Result view: intermediate sections, overlays, material visibility, differences to a CFD, diagnostics."""
from __future__ import annotations

import json
from typing import Callable

import numpy as np

from .. import metrics
from ..materials import colors_for
from ..models.domain import FIDELITY_CELLS
from ..plotter import stream_labels
from ..results import SimulationResult
from ..units import UNITS, format_quantity
from .canvas import SectionCanvas
from .qt import QtCore, QtGui, QtWidgets, Signal


# The new solver's fidelity presets as the app names them (Accuracy), from fastest to most accurate.
ACCURACY = {"preview": "Low", "medium": "Medium", "reference": "High"}
# The results the new solver's sections can be differenced against; the comparison controller supplies their codes.
DIFFERENCES = {"cfd": ("Difference to CFD", "CFD reference", "Compare with CFD"),
               "legacy_cfd": ("Difference to legacy CFD", "legacy CFD", "Compare with legacy CFD")}


class ResultsView(QtWidgets.QWidget):
    export_requested = Signal()
    station_changed = Signal(int)
    visibility_changed = Signal(object)  # set of hidden stream codes
    difference_changed = Signal(str)  # "cfd", "legacy_cfd", or "" for none

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: SimulationResult | None = None
        # (name, callable station index -> codes or None) of the result the difference is shown against
        self.reference: tuple[str, Callable[[int], np.ndarray | None]] | None = None
        self.hidden: set[int] = set()
        self.length_unit = "um"
        self.color_by = "material"
        self.stale = False
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(self.splitter)
        left_box = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_box)
        self.splitter.addWidget(left_box)
        controls = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.redraw)
        self.slider.valueChanged.connect(self.station_changed.emit)
        self.station = QtWidgets.QLabel("—")
        controls.addWidget(QtWidgets.QLabel("Station"))
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.station)
        left.addLayout(controls)
        self.canvas = SectionCanvas(self)
        left.addWidget(self.canvas, 1)
        toggles = QtWidgets.QHBoxLayout()
        self.interfaces = QtWidgets.QCheckBox("Interfaces")
        self.interfaces.setChecked(True)
        self.interfaces.toggled.connect(self.redraw)
        self.differences: dict[str, QtWidgets.QCheckBox] = {}
        for name, (label, _, _) in DIFFERENCES.items():
            box = self.differences[name] = QtWidgets.QCheckBox(label)
            box.toggled.connect(lambda checked, n=name: self._difference_toggled(n, checked))
            self.set_difference_available(name, False)
        export = QtWidgets.QPushButton("Export...")
        export.clicked.connect(self.export_requested.emit)
        for w in (self.interfaces, *self.differences.values(), export):
            toggles.addWidget(w)
        toggles.addStretch(1)
        left.addLayout(toggles)
        right_box = QtWidgets.QWidget()
        right_box.setMinimumWidth(160)
        right = QtWidgets.QVBoxLayout(right_box)
        self.splitter.addWidget(right_box)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        right.addWidget(QtWidgets.QLabel("Materials (uncheck to hide)"))
        self.streams = QtWidgets.QListWidget()
        self.streams.setMaximumHeight(140)
        self.streams.itemChanged.connect(self._visibility_changed)
        right.addWidget(self.streams)
        right.addWidget(QtWidgets.QLabel("Diagnostics"))
        self.diagnostics = QtWidgets.QTreeWidget()
        self.diagnostics.setHeaderLabels(["Item", "Value"])
        self.diagnostics.setColumnWidth(0, 210)
        right.addWidget(self.diagnostics, 1)

    # ------------------------------------------------------------------ data
    def show_result(self, result: SimulationResult, length_unit: str, color_by: str) -> None:
        self.result = result
        self.length_unit = length_unit
        self.color_by = color_by
        self.stale = False
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(len(result.sections) - 1, 0))
        self.slider.setValue(max(len(result.sections) - 1, 0))
        self.slider.blockSignals(False)
        self.streams.blockSignals(True)
        self.streams.clear()
        colors = colors_for(result.streams, color_by)
        for stream in result.streams:
            item = QtWidgets.QListWidgetItem(f"{stream['inlet_id']} ({stream['material_id']})")
            item.setData(QtCore.Qt.UserRole, int(stream["code"]))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked if int(stream["code"]) in self.hidden else QtCore.Qt.Checked)
            pixmap = QtGui.QPixmap(12, 12)
            pixmap.fill(QtGui.QColor(colors[int(stream["code"])]))
            item.setIcon(QtGui.QIcon(pixmap))
            self.streams.addItem(item)
        self.streams.blockSignals(False)
        self.fill_diagnostics()
        self.redraw()

    def mark_stale(self, stale: bool) -> None:
        if stale != self.stale:
            self.stale = stale
            self.redraw()

    def show_message(self, text: str) -> None:
        self.canvas.clear_message(text)

    def _visibility_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        code = item.data(QtCore.Qt.UserRole)
        if item.checkState() == QtCore.Qt.Checked:
            self.hidden.discard(code)
        else:
            self.hidden.add(code)
        self.visibility_changed.emit(set(self.hidden))
        self.redraw()

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
                       if available else f"Turn on Compare > {action} to show the difference to the {result}")

    def use_reference(self, name: str, codes: Callable[[int], np.ndarray | None]) -> None:
        """Mark where the new result differs from ``codes(station index)`` (None: nothing to compare there)."""
        self.reference = (name, codes)
        self.fill_diagnostics()
        self.redraw()

    def reference_codes(self, index: int) -> np.ndarray | None:
        if self.reference is None:
            return None
        codes = self.reference[1](index)
        return None if codes is None else np.asarray(codes)

    def clear_reference(self) -> None:
        self.reference = None
        self.fill_diagnostics()
        self.redraw()

    def redraw(self) -> None:
        if self.result is None:
            return
        if not self.result.ok or not self.result.sections:
            self.canvas.clear_message(f"{self.result.status}: {self.result.message}")
            return
        index = self.slider.value()
        section = self.result.sections[index]
        self.station.setText(f"{index + 1}/{len(self.result.sections)}: x = "
                             f"{section.x / UNITS['length'][self.length_unit]:.4g} {self.length_unit}")
        reference = self.reference_codes(index)
        self.canvas.show_section(section, colors_for(self.result.streams, self.color_by), stream_labels(self.result),
                                 self.length_unit, self.hidden, self.interfaces.isChecked(), reference,
                                 title=section.name, stale=self.stale)
        self.fill_reference_metrics(section)

    # ------------------------------------------------------------------ diagnostics
    def fill_diagnostics(self) -> None:
        tree = self.diagnostics
        tree.clear()
        r = self.result
        if r is None:
            return

        def add(parent, key, value="") -> QtWidgets.QTreeWidgetItem:
            item = QtWidgets.QTreeWidgetItem([str(key), str(value)])
            (parent.addChild(item) if parent is not None else tree.addTopLevelItem(item))
            return item

        status = add(None, "Status", r.status)
        status.setForeground(1, QtGui.QColor("#1B5E20" if r.ok else "#B00020"))
        add(None, "Message", r.message)
        fidelity = r.provenance.get("fidelity")
        if fidelity in FIDELITY_CELLS:
            add(None, "Accuracy", f"{ACCURACY[fidelity]} ({FIDELITY_CELLS[fidelity]} flow cells across the smaller "
                                  "channel dimension)")
        validity = add(None, "Validity", r.validity.get("status", ""))
        for check in r.validity.get("checks", []):
            item = add(validity, check["check"], ("ok  " if check["ok"] else "FAIL  ") + check["detail"])
            item.setForeground(1, QtGui.QColor("#1B5E20" if check["ok"] else "#B00020"))
        if r.warnings:
            warn = add(None, "Warnings", len(r.warnings))
            for w in r.warnings:
                add(warn, "•", w)
        timings = add(None, "Timings", f"{r.timings.get('total', 0):.3f} s total")
        for k, v in r.timings.items():
            if k != "total":
                add(timings, k, f"{v:.3f} s")
        if r.ok:
            flux = add(None, "Flux fractions (final)", "")
            final = r.final
            exp = final.diagnostics.get("expected_flux_m3_s", {})
            total = sum(exp.values()) or 1.0
            for code, value in final.diagnostics.get("flux_fraction", {}).items():
                add(flux, f"code {code}", f"{100 * value:.2f}% (expected {100 * exp.get(code, 0.0) / total:.2f}%)")
            d = r.diagnostics
            add(None, "Max flux-fraction error", f"{100 * d.get('max_flux_fraction_error', 0):.3f} pp")
            add(None, "Reynolds number", f"{d.get('reynolds_number', float('nan')):.3g}")
            for group in d.get("groups", []):
                g = add(None, group.get("group", "group"), group.get("cache", ""))
                add(g, "pressure drop", format_quantity(group.get("pressure_drop_Pa"), "mbar", "pressure"))
                add(g, "flux in / out", f"{group.get('flux_in_m3_s', 0):.4g} / {group.get('flux_out_m3_s', 0):.4g} m3/s")
                add(g, "flow cells", str(group.get("resolution", {}).get("flow_cells")))
                for solve in group.get("solves", []):
                    add(g, "solve", f"{solve['method']}, {solve['iterations']} it, residual "
                                    f"{solve['relative_residual']:.1e}, {solve['solve_seconds']:.2f} s")
                if "coupling" in group:
                    add(g, "viscosity coupling", json.dumps(group["coupling"]["converged"]))
            status_counts = final.diagnostics.get("trajectory_status")
            if status_counts:
                traj = add(None, "Trajectories (final)", "")
                for k, v in status_counts.items():
                    if k != "filled_from_neighbours":
                        add(traj, k, v)
                if "filled_from_neighbours" in status_counts:
                    fill = final.diagnostics.get("trajectory_fill") or {}
                    n = status_counts["filled_from_neighbours"]
                    share = 100 * n / max(fill.get("flow_cells") or final.codes.size, 1)
                    add(traj, "Filled from neighbours", f"{n} ({share:.3g} %)")
                    if fill.get("unfilled"):
                        add(traj, "Not filled (grey)", str(fill["unfilled"]))
            add(None, "Provenance", json.dumps(r.provenance))
            add(None, "Solver", f"{r.solver.get('name')} {r.solver.get('version')}")
        elif "traceback" in r.diagnostics:
            add(None, "Traceback", (r.diagnostics.get("traceback") or "")[-400:])
        tree.expandToDepth(0)

    def fill_reference_metrics(self, section) -> None:
        for i in range(self.diagnostics.topLevelItemCount()):
            if self.diagnostics.topLevelItem(i).text(0) == "Difference":
                self.diagnostics.takeTopLevelItem(i)
                break
        codes = self.reference_codes(self.slider.value())
        if codes is None or codes.shape != section.codes.shape:
            return
        top = QtWidgets.QTreeWidgetItem(["Difference", f"to the {self.reference[0]}"])
        self.diagnostics.addTopLevelItem(top)
        spacing = (float(np.diff(section.z_edges).mean()), float(np.diff(section.y_edges).mean()))
        for code in sorted(c for c in np.unique(section.codes) if c >= 0):
            iou = metrics.iou(section.codes, codes, code)
            shift = metrics.interface_shift(section.codes, codes, code, spacing)
            scale = 1.0 / UNITS["length"][self.length_unit]
            top.addChild(QtWidgets.QTreeWidgetItem([f"code {code}", f"IoU {iou:.3f}, interface shift "
                                                                     f"{shift * scale:.3g} {self.length_unit}"]))
        top.setExpanded(True)
