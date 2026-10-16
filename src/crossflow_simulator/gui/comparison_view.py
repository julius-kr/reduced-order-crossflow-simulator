"""Optional comparisons: the frozen legacy solver, the CFD reference, and the legacy CFD.

"Compare with legacy solver" shows the legacy result to the left of the new solver's results view;
"Compare with CFD" shows the CFD reference to the right, and "Compare with legacy CFD" (the vendored
3D-hydrodynamic-flow-sculpting map library) right of that. Any combination can be on. The strip
below them lists the measured performance of every shown solver and an agreement table: the accuracy
of the new solver, legacy, and legacy CFD against the CFD reference (with the CFD's own mesh
uncertainty) and the behavioural agreement of legacy and legacy CFD with new. "Compare new with"
chooses whether, and against which CFD (the CFD reference or the legacy CFD), the new solver is
measured. The strip also offers the legacy benchmark and the export. The new side is always the main results
view: a comparison run hands its new-solver result over through ``new_run``, and its "Difference to CFD" and
"Difference to legacy CFD" boxes mark where the new result differs from those results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import cfd, legacy_cfd, metrics
from ..comparison import (
    BackendRun, ComparisonResult, accuracy_against, build_comparison, cfd_compute_seconds,
    backend_end_to_end_seconds, export_comparison, speed_against_cfd,
)
from ..config import ProjectConfig
from ..materials import colors_for
from ..plotter import result_figure, stream_labels
from ..results import SimulationResult
from ..units import UNITS, format_quantity
from .bridge import ComparisonRunner
from .canvas import SectionCanvas
from .results_view import ResultsView
from .runtime_view import RuntimeChart
from .qt import QtCore, QtGui, QtWidgets, Signal
from .state import ProjectState

GOOD, BAD, WARN, INFO = "#1B5E20", "#B00020", "#8A6D00", "#0D47A1"
PRESET_LABELS = {cells: f"{name.capitalize()} ({cells} cells)" for name, cells in cfd.PRESETS.items()}


def _fmt_seconds(value) -> str:
    return "—" if value is None else (f"{value * 1000:.0f} ms" if value < 1 else f"{value:.2f} s")




def _fmt_length(value: float | None, unit: str) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value / UNITS['length'][unit]:.3g} {unit}"


class BackendPane(QtWidgets.QGroupBox):
    """One comparison result: status line, section canvas, and export button."""

    show_station_x = False  # legacy sections have no streamwise position

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.result: SimulationResult | None = None
        self.length_unit = "um"
        self.color_by = "material"
        self.layout_ = QtWidgets.QVBoxLayout(self)
        self.layout_.setContentsMargins(4, 4, 4, 4)
        self.status = QtWidgets.QLabel("Not run")
        self.status.setWordWrap(True)
        self.layout_.addWidget(self.status)
        plot = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = SectionCanvas(plot)
        self.canvas.setMinimumSize(200, 150)
        plot_layout.addWidget(self.canvas, 1)
        exports = QtWidgets.QHBoxLayout()
        self.export_button = QtWidgets.QPushButton("Export...")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_result)
        exports.addWidget(self.export_button)
        exports.addStretch(1)
        plot_layout.addLayout(exports)
        self.layout_.addWidget(plot, 1)

    def set_status(self, text: str, color: str = "black") -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color}")

    def export_result(self) -> None:
        """Export this backend's native result, project snapshot, figure, and diagnostics."""
        result = self.result
        if result is None or not result.ok or result.final is None:
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Export result")
        if not directory:
            return
        out = Path(directory)
        result.save(out / "result.npz")
        ProjectConfig.from_dict(result.config).save(out / "config.json")
        from matplotlib import pyplot as plt

        figure = result_figure(result, colors_for(result.streams, self.color_by), self.length_unit)
        try:
            figure.savefig(out / "sections.png", dpi=200)
        finally:
            plt.close(figure)
        (out / "diagnostics.json").write_text(json.dumps({
            "status": result.status, "message": result.message, "diagnostics": result.diagnostics,
            "validity": result.validity, "timings": result.timings, "provenance": result.provenance,
            "solver": result.solver, "warnings": result.warnings,
        }, indent=2, default=str), encoding="utf-8")



    def show_run(self, run: BackendRun, length_unit: str, color_by: str, limits, reference=None,
                 stale: bool = False, station: int | None = None, hidden: set[int] | None = None) -> None:
        self.result = run.result
        self.length_unit = length_unit
        self.color_by = color_by
        self.export_button.setEnabled(run.status == "success" and run.result is not None
                                      and run.result.ok and run.result.final is not None)
        if run.status == "unsupported":
            self.set_status(run.message, WARN)
            self.canvas.clear_message("Not supported\n\n" + "\n".join(run.unsupported))
            return
        if run.result is None or not run.result.ok or run.result.final is None:
            self.set_status(f"{run.status}: {run.message}", BAD)
            self.canvas.clear_message(f"{run.status}\n{run.message[:200]}")
            return
        result = run.result
        section = result.final
        if station is not None and 0 <= station < len(result.sections):
            section = result.sections[station]
        prefix = "STALE (inputs changed): " if stale else ""
        self.set_status(f"{prefix}{run.status}: {run.message}", BAD if stale else GOOD)
        title = section.name
        if self.show_station_x:
            title += f" (x = {section.x / UNITS['length'][length_unit]:.4g} {length_unit})"
        self.canvas.show_section(section, colors_for(result.streams, color_by), stream_labels(result), length_unit,
                                 hidden=hidden, interfaces=True, reference=reference, title=title, limits=limits,
                                 stale=stale)


class CFDPane(BackendPane):
    """The CFD reference: grid choice, own run button, progress, plot, and export."""

    grid_chosen = Signal(int)
    run_requested = Signal()
    recompute_requested = Signal()
    show_station_x = True

    def __init__(self, parent=None):
        super().__init__("CFD reference", parent)
        bar = QtWidgets.QGridLayout()
        self.grid = QtWidgets.QComboBox()
        for cells, label in PRESET_LABELS.items():
            self.grid.addItem(label, cells)
        self.grid.setToolTip("Cells across the smaller channel dimension on the reference grid. A second grid with "
                             "2/3 of the cells estimates the mesh uncertainty.")
        self.grid.activated.connect(lambda index: self.grid_chosen.emit(int(self.grid.itemData(index))))
        self.run_button = QtWidgets.QPushButton("Run CFD")
        self.run_button.setToolTip("Compute (or load the stored) CFD reference for the current inputs")
        self.run_button.clicked.connect(self.run_requested.emit)
        self.recompute_button = QtWidgets.QPushButton("Recompute")
        self.recompute_button.setToolTip("Delete the stored CFD result for these inputs and compute it again, so the "
                                         "CFD compute time (and the new solver's speed against it) is measured now")
        self.recompute_button.clicked.connect(self.recompute_requested.emit)
        self.estimate = QtWidgets.QLabel()
        self.estimate.setStyleSheet("color: #555555")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setMaximumWidth(120)
        self.progress.setVisible(False)
        bar.addWidget(QtWidgets.QLabel("Grid"), 0, 0)
        bar.addWidget(self.grid, 0, 1)
        bar.addWidget(self.run_button, 1, 0)
        bar.addWidget(self.recompute_button, 1, 1)
        bar.addWidget(self.estimate, 2, 0, 1, 2)
        bar.addWidget(self.progress, 3, 0, 1, 2)
        self.layout_.insertLayout(0, bar)

    def set_grid(self, cells: int, seconds: float) -> None:
        index = self.grid.findData(cells)
        if index < 0:
            self.grid.addItem(f"Custom ({cells} cells)", cells)
            index = self.grid.count() - 1
        self.grid.setCurrentIndex(index)
        self.estimate.setText(f"about {seconds:.0f} s" if seconds < 90 else f"about {seconds / 60:.0f} min")

    def set_progress(self, fraction: float | None) -> None:
        self.progress.setVisible(fraction is not None)
        if fraction is not None:
            self.progress.setValue(int(100 * fraction))



class LegacyCFDPane(BackendPane):
    """The legacy CFD result and export."""

    show_station_x = True

    def __init__(self, parent=None):
        super().__init__("Legacy CFD", parent)



PERFORMANCE_ROWS = ("Status", "Solve time (in worker)", "End-to-end backend time", "Worker startup (separate)",
                    "Cache state", "Peak worker memory", "Speed vs CFD")
COLUMNS = (("legacy", "Legacy"), ("new", "New"), ("cfd", "CFD"), ("legacy_cfd", "Legacy CFD"))
REFERENCES = {"cfd": "CFD", "legacy_cfd": "legacy CFD"}  # what the new solver can be measured against




def _performance_values(name: str, run: BackendRun | None, reference: BackendRun | None) -> list[str]:
    if run is None:
        return ["—"] * len(PERFORMANCE_ROWS)
    peak = run.memory.get("peak_bytes") if run.memory else None
    # Legacy CFD applies stored maps; the worker's duration is not the original
    # CFD solve time, which is unavailable in the vendored result data.
    solve = _fmt_seconds(None if name == "legacy_cfd" else
                         cfd_compute_seconds(run) if name == "cfd" else run.solve_seconds)
    speed = "—"
    if name == "cfd":
        speed = "reference"
    elif run.cache_state == "map reused":
        speed = "n/a (map reused)"  # a cached lookup is not compared with a computed reference
    elif reference is not None and reference.result is not None and reference.result.ok:
        value = speed_against_cfd(run, reference).get("value")
        if value:
            speed = f"{value:.0f}x faster" if value >= 1 else f"{1 / value:.1f}x slower"
    return [run.status, solve, _fmt_seconds(backend_end_to_end_seconds(run)), _fmt_seconds(run.startup_seconds),
            run.cache_state, "unavailable" if not peak else f"{peak / 2**20:.0f} MiB", speed]


class ComparisonController(QtCore.QObject):
    """Legacy pane (left of the new results), CFD and legacy CFD panes (right of them), and the comparison strip."""

    new_run = Signal(object)  # BackendRun of the new solver from a comparison run
    busy_changed = Signal(bool)
    cfd_changed = Signal(object, bool)  # CFD result (or None) and whether it is stale
    message = Signal(str)

    def __init__(self, state: ProjectState, results: ResultsView, parent=None):
        super().__init__(parent)
        self.state = state
        self.results = results
        self.runner = ComparisonRunner()
        self.runner.update.connect(self._update)
        self.runner.done.connect(self._done)
        self.runner.error.connect(self._error)
        self.comparison: ComparisonResult | None = None
        self.legacy_comparison: ComparisonResult | None = None  # the last run that included legacy (speedup)
        self.runs: dict[str, BackendRun] = {}
        self.new_result: SimulationResult | None = None
        self.enabled = {"legacy": False, "cfd": False, "legacy_cfd": False}
        self.station: int | None = None
        self.busy = False
        self._syncing = False
        self._accuracy: dict[str, dict] = {}
        self._difference_key: tuple | None = None
        self.runtime_chart = RuntimeChart()
        self.pane = BackendPane("Legacy solver")
        self.cfd_pane = CFDPane()
        self.cfd_pane.grid_chosen.connect(self._grid_chosen)
        self.cfd_pane.run_requested.connect(lambda: self.run(self.state.config, ("cfd",)))
        self.cfd_pane.recompute_requested.connect(lambda: self.run(self.state.config, ("cfd",), recompute_cfd=True))
        self.legacy_cfd_pane = LegacyCFDPane()
        self.panes = {"legacy": self.pane, "cfd": self.cfd_pane, "legacy_cfd": self.legacy_cfd_pane}

        self.strip = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.strip)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.trials = QtWidgets.QSpinBox()
        self.trials.setRange(1, 20)
        self.trials.setValue(3)
        self.bench_button = QtWidgets.QPushButton("Benchmark legacy/new")
        self.bench_button.setToolTip("Repeatable protocol: sequential runs, alternating order, cold/warm/cached scopes")
        self.bench_button.clicked.connect(lambda: self.benchmark(self.state.config))
        self.export_button = QtWidgets.QPushButton("Export comparison...")
        self.export_button.clicked.connect(self.export)
        self.compare_new = QtWidgets.QCheckBox("Compare new with")
        self.compare_new.setChecked(True)
        self.compare_new.setToolTip("Measure the new solver against a CFD result in the agreement table")
        self.compare_new.toggled.connect(self._reference_changed)
        self.reference = QtWidgets.QComboBox()
        self.reference.setToolTip("The CFD the new solver is measured against: the CFD reference (full-channel 3D "
                                  "Navier-Stokes) or the legacy CFD (precomputed COMSOL map library)")
        self.reference.currentIndexChanged.connect(lambda _: self._reference_changed())
        self.bench_widgets = [QtWidgets.QLabel("Trials"), self.trials, self.bench_button]
        self.reference_widgets = [self.compare_new, self.reference]
        for widget in (*self.bench_widgets, *self.reference_widgets, self.export_button):
            bar.addWidget(widget)
        bar.addStretch(1)
        layout.addLayout(bar)
        tables = QtWidgets.QHBoxLayout()
        self.performance = QtWidgets.QTableWidget(len(PERFORMANCE_ROWS), len(COLUMNS))
        self.performance.setHorizontalHeaderLabels([label for _, label in COLUMNS])
        self.performance.setVerticalHeaderLabels(list(PERFORMANCE_ROWS))
        self.performance.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.metrics = QtWidgets.QTableWidget(0, 7)
        self.metrics.setHorizontalHeaderLabels(["Compared", "Stream", "IoU", "Interface shift", "Flux misassigned",
                                                "Centroid shift", "Verdict"])
        self.metrics.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.metrics.setToolTip("Interface shift: area where a stream differs / its interface length. Flux "
                                "misassigned: share of the flow (CFD velocity) placed in the wrong stream. Targets: "
                                "IoU >= 0.90, shift <= 2% of H plus the CFD mesh uncertainty, flux <= 0.5 pp.")
        for table in (self.performance, self.metrics):
            table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            table.verticalHeader().setDefaultSectionSize(20)
            table.setMaximumHeight(165)
        tables.addWidget(self.performance, 2)
        tables.addWidget(self.metrics, 4)
        layout.addLayout(tables)
        canvases = [self.pane.canvas, self.results.canvas, self.cfd_pane.canvas, self.legacy_cfd_pane.canvas]
        self.canvases = canvases
        for source in canvases:
            source.limit_listeners.append(lambda s=source: self._sync(s, canvases))
            source.size_listeners.append(lambda: QtCore.QTimer.singleShot(0, self._size_plots))
        self.results.difference_changed.connect(lambda _: self._apply_difference())
        state.changed.connect(self._state_changed)
        self._state_changed()

    # ------------------------------------------------------------------ helpers
    def _size_plots(self) -> None:
        visible = [canvas for canvas in self.canvases if canvas.isVisible()]
        if not visible:
            return
        # Controls above the canvases can have different heights. Fit each plot in
        # their shared vertical interval so z=0 lines up in window coordinates.
        tops = [c.mapToGlobal(QtCore.QPoint(0, 0)).y() for c in visible]
        shared_top = max(tops) + 40
        shared_bottom = min(top + c.height() for top, c in zip(tops, visible)) - 50
        width = max(20.0, min(c.width() for c in visible) - 150)
        available_height = max(20.0, shared_bottom - shared_top)
        y0, y1, z0, z1 = self.limits()
        ratio = (y1 - y0) / max(z1 - z0, 1e-30)
        width = min(width, available_height * ratio)
        height = width / ratio
        baseline = (shared_top + shared_bottom + height) / 2
        for canvas, top in zip(visible, tops):
            scale = canvas.figure.bbox.height / max(canvas.height(), 1)
            canvas.set_plot_size(width * scale, height * scale,
                                 (top + canvas.height() - baseline) * scale)

    def limits(self) -> tuple[float, float, float, float]:
        cfg = self.state.config
        scale = 1.0 / UNITS["length"][cfg.display.length_unit]
        return (0.0, cfg.channel.width * scale, 0.0, cfg.channel.height * scale)

    def _new_run(self) -> BackendRun | None:
        """The new solver's result shown in the results view (from a comparison run or a preview)."""
        result = self.new_result
        run = self.runs.get("new")
        if run is not None and (result is None or run.result is result):
            return run
        if result is None:
            return None
        return BackendRun("new", result.status, result.message, solve_seconds=result.timings.get("total"),
                          cache_state="map reused" if result.diagnostics.get("map_reused") else "preview worker",
                          result=result)

    def _other_inputs(self, run: BackendRun | None) -> bool:
        """Whether a legacy or legacy CFD result belongs to other inputs (the new solver's accuracy is not one)."""
        if not (run and run.result):
            return False
        cfg = self.state.config.copy()
        cfg.model.fidelity = (run.result.config or {}).get("model", {}).get("fidelity", cfg.model.fidelity)
        return run.result.input_hash != cfg.input_hash()

    def _legacy_stale(self) -> bool:
        return self._other_inputs(self.runs.get("legacy"))

    def _legacy_cfd_stale(self) -> bool:
        return self._other_inputs(self.runs.get("legacy_cfd"))

    def _cfd_stale(self) -> bool:
        run = self.runs.get("cfd")
        return bool(run and run.result and run.result.ok and not cfd.is_current(run.result, self.state.config))

    def _new_stale(self) -> bool:
        result = self.new_result
        if result is None:
            return False
        return result.input_hash != self.state.config.input_hash()  # the accuracy is an input of the new solver

    # ------------------------------------------------------------------ enabling and capability
    def set_enabled(self, legacy: bool, cfd_on: bool, legacy_cfd_on: bool = False) -> None:
        self.enabled = {"legacy": legacy, "cfd": cfd_on, "legacy_cfd": legacy_cfd_on}
        for widget in self.bench_widgets:
            widget.setVisible(legacy)
        for col, (name, _) in enumerate(COLUMNS):
            self.performance.setColumnHidden(col, name != "new" and not self.enabled[name])
        self.results.set_difference_available("cfd", cfd_on)
        self.results.set_difference_available("legacy_cfd", legacy_cfd_on)
        self.reference.blockSignals(True)
        current = self.reference.currentData()
        self.reference.clear()
        for name, flag in (("cfd", cfd_on), ("legacy_cfd", legacy_cfd_on)):
            if flag:
                self.reference.addItem("CFD reference" if name == "cfd" else "legacy CFD", name)
        self.reference.setCurrentIndex(max(self.reference.findData(current), 0))
        self.reference.blockSignals(False)
        for widget in self.reference_widgets:
            widget.setVisible(cfd_on or legacy_cfd_on)
        self._state_changed()

    def new_reference(self) -> str | None:
        """The CFD the new solver is measured against ("cfd" or "legacy_cfd"), or None when that is off."""
        choice = self.reference.currentData()
        if not self.compare_new.isChecked() or not choice or not self.enabled.get(choice):
            return None
        return choice

    def _reference_changed(self, *_) -> None:
        self.reference.setEnabled(self.compare_new.isChecked())
        self.redraw()

    def show_capability(self) -> None:
        """Before a run: say whether legacy, the CFD, and legacy CFD can compute the current case, and why not."""
        cfg = self.state.config
        if self.enabled["legacy"] and "legacy" not in self.runs and not self.busy:
            from ..backends.legacy import build_plan

            reasons = build_plan(cfg)["unsupported"]
            if reasons:
                self.pane.set_status("Not supported by legacy", WARN)
                self.pane.canvas.clear_message("Not supported by legacy\n\n" + "\n".join(reasons))
            else:
                self.pane.set_status("Ready: Run (F5) runs the new solver, then legacy", "#555555")
                self.pane.canvas.clear_message("Press Run (F5) to compare")
        if self.enabled["cfd"]:
            try:
                self.cfd_pane.set_grid(cfg.cfd.cells, cfd.estimate_seconds(cfg))
            except Exception:  # noqa: BLE001 - invalid geometry: the setup panel reports it
                pass
            if "cfd" not in self.runs and not self.busy:
                reasons = cfd.unsupported(cfg)
                if reasons:
                    self.cfd_pane.set_status("Not supported by the CFD reference", WARN)
                    self.cfd_pane.canvas.clear_message("Not supported by the CFD reference\n\n" + "\n".join(reasons))
                else:
                    self.cfd_pane.set_status("Ready: Run (F5) or Run CFD computes the reference", "#555555")
                    self.cfd_pane.canvas.clear_message("Press Run (F5) or Run CFD\nto compute the CFD reference")
        if self.enabled["legacy_cfd"] and "legacy_cfd" not in self.runs and not self.busy:
            try:
                reasons = legacy_cfd.build_plan(cfg)["unsupported"]
            except Exception as exc:  # noqa: BLE001 - invalid geometry: the setup panel reports it
                reasons = [str(exc)]
            if reasons:
                self.legacy_cfd_pane.set_status("Not supported by legacy CFD", WARN)
                self.legacy_cfd_pane.canvas.clear_message("Not supported by legacy CFD\n\n" + "\n".join(reasons))
            else:
                self.legacy_cfd_pane.set_status("Ready: Run (F5) runs the new solver, then legacy CFD", "#555555")
                self.legacy_cfd_pane.canvas.clear_message("Press Run (F5) to compare")

    def load_cached_cfd(self) -> bool:
        """Show the stored CFD reference for the current inputs, if one exists (no computation)."""
        run = self.runs.get("cfd")
        if run is not None and run.result is not None and run.result.ok and not self._cfd_stale():
            return True
        try:
            result = cfd.cached_result(self.state.config)
        except Exception:  # noqa: BLE001
            result = None
        if result is None:
            return False
        from ..backends.cfd import CFDBackend

        self.runs["cfd"] = BackendRun("cfd", result.status, result.message, cache_state=result.diagnostics["cache"],
                                      solve_seconds=result.diagnostics.get("computed_seconds", result.timings.get("total")),
                                      end_to_end_seconds=result.diagnostics.get("computed_backend_seconds"),
                                      effective_settings=CFDBackend().effective_settings(self.state.config),
                                      result=result)
        return True

    def _grid_chosen(self, cells: int) -> None:
        if cells != self.state.config.cfd.cells:
            self.state.apply(lambda c: setattr(c.cfd, "cells", cells), "CFD grid")

    # ------------------------------------------------------------------ actions
    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.bench_button.setEnabled(not busy)
        self.cfd_pane.run_button.setEnabled(not busy)
        self.cfd_pane.recompute_button.setEnabled(not busy)
        if not busy:
            self.cfd_pane.set_progress(None)
        self.busy_changed.emit(busy)

    def run(self, cfg: ProjectConfig, names: tuple[str, ...] | None = None, recompute_cfd: bool = False) -> None:
        """Run the new solver and the enabled comparisons (or only ``names``) in isolated workers.

        ``recompute_cfd`` discards the stored CFD result of these inputs and computes it again.
        """
        if self.busy:
            return
        if names is None:  # legacy CFD takes milliseconds, so it runs before the CFD reference
            names = ("new",) + tuple(n for n in ("legacy", "legacy_cfd", "cfd") if self.enabled[n])
        for name in names:
            if name != "new":
                self.runs.pop(name, None)
        if "legacy" in names:
            self.legacy_comparison = None
            self.pane.set_status("Queued (runs after the new solver)" if "new" in names else "Queued", "#555555")
        if "legacy_cfd" in names:
            self.legacy_cfd_pane.set_status("Queued (runs after the new solver)" if "new" in names else "Queued",
                                            "#555555")
        if "cfd" in names:
            self.cfd_pane.set_status(("Recomputing (stored result cleared)" if recompute_cfd else "Queued")
                                     + (" (runs after the other solvers)" if len(names) > 1 else ""), "#555555")
            self.cfd_pane.set_progress(0.0)
        self.comparison = None
        self.fill_performance()
        self._set_busy(True)
        self.runner.run_set(cfg, names, recompute_cfd=recompute_cfd)

    def run_pair(self, cfg: ProjectConfig) -> None:
        self.run(cfg, ("new", "legacy"))

    def benchmark(self, cfg: ProjectConfig) -> None:
        if self.busy or self.state.errors():
            return
        self.runs.pop("legacy", None)
        self.legacy_comparison = None
        self.pane.set_status("Benchmark running (sequential, alternating order)...", "#555555")
        self.message.emit("Benchmark running...")
        self._set_busy(True)
        self.runner.benchmark(cfg, self.trials.value())

    def cancel(self) -> None:
        if not self.busy:
            return
        self.runner.cancel()
        for name, pane in self.panes.items():
            if self.enabled[name] and name not in self.runs:
                pane.set_status("Cancelled; worker restarted", WARN)
        self._set_busy(False)

    def current_comparison(self) -> ComparisonResult | None:
        runs = {name: run for name, run in self.runs.items() if name != "new"}
        new = self._new_run()
        if new is not None:
            runs["new"] = new
        if len(runs) < 2:
            return None
        return build_comparison(self.state.config, runs, self.comparison.comparison_id if self.comparison else None)

    def export(self) -> None:
        comparison = self.current_comparison()
        if comparison is None:
            QtWidgets.QMessageBox.information(self.strip, "Export", "Run a comparison first.")
            return
        comparison.benchmark = self.legacy_comparison.benchmark if self.legacy_comparison else None
        directory = QtWidgets.QFileDialog.getExistingDirectory(self.strip, "Export comparison")
        if directory:
            out = export_comparison(comparison, Path(directory) / f"comparison_{comparison.comparison_id}",
                                    self.state.config.display.length_unit)
            self.message.emit(f"Exported to {out}")

    def shutdown(self) -> None:
        self.runner.shutdown()

    # ------------------------------------------------------------------ inputs from the main window
    def set_new_result(self, result: SimulationResult | None) -> None:
        self.new_result = result
        if any(self.enabled.values()):
            self.redraw()
        else:
            self.fill_performance()

    def set_station(self, index: int | None) -> None:
        self.station = index
        if self.enabled["cfd"] or self.enabled["legacy_cfd"]:
            self.redraw()

    # ------------------------------------------------------------------ updates from the runner
    def _update(self, kind: str, data: dict) -> None:
        backend = data.get("backend") or data.get("worker")
        pane = self.panes.get(backend)
        if pane is not None:
            if kind == "pending":
                pane.set_status("Running...", INFO)
            elif kind == "message" and data.get("type") == "progress":
                pane.set_status(f"Running: {data.get('message', '')}", INFO)
                if backend == "cfd":
                    self.cfd_pane.set_progress(float(data.get("fraction", 0.0)))
        if kind == "run" and "run" in data:
            run: BackendRun = data["run"]
            self.runs[backend] = run
            if backend == "new":
                self.new_run.emit(run)
            else:
                self.redraw()

    def _done(self, comparison: ComparisonResult) -> None:
        self.comparison = comparison
        if "legacy" in comparison.runs:
            self.legacy_comparison = comparison
        self.runs.update(comparison.runs)
        self._set_busy(False)
        self.redraw()

    def _error(self, message: str) -> None:
        self._set_busy(False)
        self.message.emit(f"Comparison failed: {message}")

    def _state_changed(self) -> None:
        if self.enabled["cfd"] and not self.busy:
            self.load_cached_cfd()
        self.show_capability()
        self.redraw()

    # ------------------------------------------------------------------ drawing
    def redraw(self) -> None:
        cfg = self.state.config
        unit, color_by = cfg.display.length_unit, cfg.display.color_by
        hidden = set(self.results.hidden)
        legacy = self.runs.get("legacy")
        if legacy and self.enabled["legacy"]:
            self.pane.show_run(legacy, unit, color_by, self.limits(), stale=self._legacy_stale(), hidden=hidden)
        reference = self.runs.get("cfd")
        cfd_result = reference.result if reference is not None and reference.result is not None else None
        if reference and self.enabled["cfd"]:
            station = self.station if (self.new_result is not None and cfd_result is not None
                                       and len(self.new_result.sections) == len(cfd_result.sections)) else None
            self.cfd_pane.show_run(reference, unit, color_by, self.limits(), stale=self._cfd_stale(),
                                   station=station, hidden=hidden)
        mapped = self.runs.get("legacy_cfd")
        if mapped and self.enabled["legacy_cfd"]:
            mapped_result = mapped.result
            station = self.station if (self.new_result is not None and mapped_result is not None
                                       and len(self.new_result.sections) == len(mapped_result.sections)) else None
            self.legacy_cfd_pane.show_run(mapped, unit, color_by, self.limits(), stale=self._legacy_cfd_stale(),
                                          station=station, hidden=hidden)
        shown = cfd_result if (self.enabled["cfd"] and cfd_result is not None and cfd_result.ok) else None
        self.cfd_changed.emit(shown, self._cfd_stale())
        self._accuracy = self.accuracy()
        self.fill_performance()
        self.fill_metrics()
        self._apply_difference()
        QtCore.QTimer.singleShot(0, self._size_plots)

    def fill_performance(self) -> None:
        reference = self.runs.get("cfd")
        runs = {"legacy": self.runs.get("legacy"), "new": self._new_run(), "cfd": reference,
                "legacy_cfd": self.runs.get("legacy_cfd")}
        for col, (name, _) in enumerate(COLUMNS):
            for row, value in enumerate(_performance_values(name, runs[name], reference)):
                item = QtWidgets.QTableWidgetItem(value)
                if name == "legacy_cfd" and row == 1:
                    item.setToolTip("Original CFD solve time unavailable: this solver applies precomputed maps.")
                self.performance.setItem(row, col, item)
        stale = {"new": self._new_stale(), "legacy": self._legacy_stale(),
                 "cfd": self._cfd_stale(), "legacy_cfd": self._legacy_cfd_stale()}
        records = []
        for name, label in (("new", "New solver"), ("legacy", "Legacy solver"),
                            ("cfd", "CFD reference")):
            if name != "new" and not self.enabled[name]:
                continue
            run = runs[name]
            seconds = None
            status = "not run" if run is None else run.status
            if run is not None and run.status == "success":
                seconds = cfd_compute_seconds(run) if name == "cfd" else run.solve_seconds
                if stale[name]:
                    status = "stale"
            records.append((name, label, seconds, status))
        self.runtime_chart.show_runtimes(records)

    def accuracy(self) -> dict[str, dict]:
        """Accuracy of new against the chosen CFD, and of legacy and legacy CFD against the CFD reference.

        New and legacy CFD are measured at the selected station, legacy at its final section.
        """
        out = {}
        new = self.new_result
        choice = self.new_reference()
        chosen = self.runs.get(choice) if choice else None
        if (chosen is not None and chosen.result is not None and chosen.result.ok and new is not None and new.ok
                and new.sections):
            out["new"] = {**accuracy_against(chosen.result, new, self.station), "reference": choice}
        reference = self.runs.get("cfd")
        if not self.enabled["cfd"] or reference is None or reference.result is None or not reference.result.ok:
            return out
        legacy = self.runs.get("legacy")
        if legacy is not None and legacy.result is not None and legacy.result.ok and legacy.result.final is not None:
            out["legacy"] = accuracy_against(reference.result, legacy.result)
        mapped = self.runs.get("legacy_cfd")
        if (self.enabled["legacy_cfd"] and mapped is not None and mapped.result is not None and mapped.result.ok
                and mapped.result.final is not None):
            out["legacy_cfd"] = accuracy_against(reference.result, mapped.result, self.station)
        return out

    def fill_metrics(self) -> None:
        self.metrics.setRowCount(0)
        cfg = self.state.config
        unit = cfg.display.length_unit
        H = cfg.channel.height
        rows = []
        accuracy = self._accuracy
        new_reference = accuracy.get("new", {}).get("reference")
        for name, label in (("new", f"New vs {REFERENCES.get(new_reference, 'CFD')}"), ("legacy", "Legacy vs CFD"),
                            ("legacy_cfd", "Legacy CFD vs CFD")):
            entry = accuracy.get(name)
            if entry is None or (name != "new" and not self.enabled[name]):
                continue
            stale = self._legacy_cfd_stale() if entry.get("reference") == "legacy_cfd" else self._cfd_stale()
            target = entry["targets"]
            for code, m in entry["per_stream"].items():
                ok = (m["iou"] >= target["iou_min"] and not (m["interface_shift_m"] > target["interface_shift_m"])
                      and abs(m["flux_misassigned"]) <= target["flux_misassigned_max"])
                shift = m["interface_shift_m"]
                within = entry["cfd_uncertainty_m"] > 0 and np.isfinite(shift) and shift <= entry["cfd_uncertainty_m"]
                verdict = ("within CFD uncertainty" if within else "within target") if ok else "outside target"
                if stale:
                    verdict = "STALE: " + verdict
                rows.append((label, code, m["iou"], shift, m["flux_misassigned"], m["centroid_displacement_m"],
                             verdict, (WARN if stale else GOOD if ok else BAD)))
        legacy = self.runs.get("legacy")
        new = self.new_result
        if (self.enabled["legacy"] and legacy is not None and legacy.result is not None and legacy.result.ok
                and new is not None and new.ok and new.final is not None):
            entry = accuracy_against(new, legacy.result)
            for code, m in entry["per_stream"].items():
                rows.append(("Legacy vs New", code, m["iou"], m["interface_shift_m"], m["flux_misassigned"],
                             m["centroid_displacement_m"], "behavioural only", "#555555"))
        mapped = self.runs.get("legacy_cfd")
        if (self.enabled["legacy_cfd"] and mapped is not None and mapped.result is not None and mapped.result.ok
                and new is not None and new.ok and new.final is not None and new_reference != "legacy_cfd"):
            entry = accuracy_against(new, mapped.result, self.station)
            for code, m in entry["per_stream"].items():
                rows.append(("Legacy CFD vs New", code, m["iou"], m["interface_shift_m"], m["flux_misassigned"],
                             m["centroid_displacement_m"], "behavioural only", "#555555"))
        for label, code, iou, shift, flux, centroid, verdict, color in rows:
            row = self.metrics.rowCount()
            self.metrics.insertRow(row)
            shift_text = _fmt_length(shift, unit) + (f" ({100 * shift / H:.2g}% H)" if np.isfinite(shift) else "")
            values = [label, str(code), f"{iou:.3f}", shift_text, f"{100 * flux:+.2f} pp", _fmt_length(centroid, unit),
                      verdict]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col == 6:
                    item.setForeground(QtGui.QColor(color))
                self.metrics.setItem(row, col, item)


    def _apply_difference(self) -> None:
        """Hand the results view the codes of the result its difference box asks for (CFD or legacy CFD)."""
        choice = self.results.difference_choice()
        new = self.new_result
        source = self.runs.get(choice) if choice else None
        key = (choice, id(new), id(source.result) if source is not None else None)
        if key == self._difference_key and (not choice or self.results.reference is not None):
            return
        self._difference_key = key
        if choice == "cfd":
            run = self.runs.get("cfd")
            if run and run.result and run.result.ok:
                result = run.result

                def codes(index: int, _result=result):
                    sections = _result.sections
                    if new is not None and len(new.sections) == len(sections) and 0 <= index < len(sections):
                        return sections[index].codes
                    return sections[-1].codes if index == len(sections) - 1 else None

                self.results.use_reference("CFD reference", codes)
                return
        elif choice == "legacy_cfd":
            run = self.runs.get("legacy_cfd")
            if run and run.result and run.result.ok and new is not None and new.ok:
                result = run.result

                def mapped_codes(index: int, _result=result):
                    sections = _result.sections
                    if len(new.sections) == len(sections) and 0 <= index < len(sections):
                        source = sections[index]
                    elif index == len(new.sections) - 1:
                        source = sections[-1]
                    else:
                        return None
                    target = new.sections[index]
                    return metrics.regrid_nearest(source, target.y, target.z)

                self.results.use_reference("legacy CFD (regridded, nearest)", mapped_codes)
                return
        if self.results.reference is not None:  # none asked for, or its result is not there (yet)
            self.results.clear_reference()

    def _sync(self, source: SectionCanvas, canvases: list[SectionCanvas]) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            for target in canvases:
                if target is source or not target.isVisible():
                    continue
                target.axes.set_xlim(source.axes.get_xlim())
                target.axes.set_ylim(source.axes.get_ylim())
                target.draw_idle()
        finally:
            self._syncing = False


__all__ = ["BackendPane", "CFDPane", "ComparisonController", "LegacyCFDPane"]
