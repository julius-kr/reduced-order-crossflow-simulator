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
    export_comparison, realistic_seconds, speed_against_cfd,
)
from ..config import ProjectConfig
from ..materials import colors_for
from ..plotter import stream_labels
from ..results import SimulationResult
from ..units import UNITS, format_quantity
from .bridge import ComparisonRunner
from .canvas import SectionCanvas
from .results_view import ResultsView
from .qt import NavigationToolbar2QT, QtCore, QtGui, QtWidgets, Signal
from .state import ProjectState

GOOD, BAD, WARN, INFO = "#1B5E20", "#B00020", "#8A6D00", "#0D47A1"
PRESET_LABELS = {cells: f"{name.capitalize()} ({cells} cells)" for name, cells in cfd.PRESETS.items()}


def _fmt_seconds(value) -> str:
    return "—" if value is None else (f"{value * 1000:.0f} ms" if value < 1 else f"{value:.2f} s")


def _fmt_duration(value) -> str:
    """Seconds as ms, s, min or h (for times that range from milliseconds to days)."""
    if value is None:
        return "—"
    if value < 120:
        return _fmt_seconds(value)
    return f"{value / 60:.1f} min" if value < 7200 else f"{value / 3600:.1f} h"


def _fmt_length(value: float | None, unit: str) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value / UNITS['length'][unit]:.3g} {unit}"


class BackendPane(QtWidgets.QGroupBox):
    """One comparison result: status line, section canvas, and a tree of measurements (resizable split)."""

    show_station_x = False  # legacy sections have no streamwise position

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
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
        self.toolbar = NavigationToolbar2QT(self.canvas, plot)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        self.performance = QtWidgets.QTreeWidget()
        self.performance.setHeaderLabels(["Measurement", "Value"])
        self.performance.setColumnWidth(0, 190)
        self.performance.setMinimumHeight(60)
        self.split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.split.addWidget(plot)
        self.split.addWidget(self.performance)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 1)
        self.split.setChildrenCollapsible(False)
        self.layout_.addWidget(self.split, 1)

    def set_status(self, text: str, color: str = "black") -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color}")

    def _add(self, parent, key, value="", color: str | None = None) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem([str(key), str(value)])
        (parent.addChild(item) if parent is not None else self.performance.addTopLevelItem(item))
        if color:
            item.setForeground(1, QtGui.QColor(color))
        return item

    def fill_tree(self, run: BackendRun) -> None:
        add = self._add
        add(None, "Status", run.status)
        add(None, "Solve time (in worker)", _fmt_seconds(run.solve_seconds))
        add(None, "End-to-end backend time", _fmt_seconds(run.end_to_end_seconds))
        add(None, "Worker startup (separate)", _fmt_seconds(run.startup_seconds))
        add(None, "Cache state", run.cache_state)
        peak = run.memory.get("peak_bytes") if run.memory else None
        add(None, "Peak worker memory", "unavailable" if not peak else f"{peak / 2**20:.0f} MiB ({run.memory['method']})")
        if run.result is not None and run.result.timings:
            stages = add(None, "Stage timings", "")
            for k, v in run.result.timings.items():
                add(stages, k, _fmt_seconds(v))
        settings = add(None, "Effective settings", "")
        for k, v in run.effective_settings.items():
            add(settings, k, json.dumps(v) if not isinstance(v, str) else v)
        if run.result is not None:
            diag = add(None, "Diagnostics", "")
            d = run.result.diagnostics
            for key in ("max_flux_fraction_error", "reynolds_number", "unmatched_labels", "pressure_solver_messages",
                        "unavailable"):
                if key in d:
                    add(diag, key, json.dumps(d[key]) if not isinstance(d[key], float) else f"{d[key]:.4g}")
            for w in run.result.warnings:
                add(diag, "warning", w)

    def show_run(self, run: BackendRun, length_unit: str, color_by: str, limits, reference=None,
                 stale: bool = False, station: int | None = None, hidden: set[int] | None = None) -> None:
        self.performance.clear()
        self.fill_tree(run)
        self.performance.expandToDepth(0)
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
    """The CFD reference: grid choice, own run button and progress, and every CFD statistic."""

    grid_chosen = Signal(int)
    run_requested = Signal()
    recompute_requested = Signal()
    show_station_x = True

    def __init__(self, parent=None):
        super().__init__("CFD reference (full-channel 3D Navier-Stokes)", parent)
        bar = QtWidgets.QHBoxLayout()
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
        for widget in (QtWidgets.QLabel("Grid"), self.grid, self.run_button, self.recompute_button, self.progress,
                       self.estimate):
            bar.addWidget(widget)
        bar.addStretch(1)
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

    def fill_tree(self, run: BackendRun) -> None:
        add = self._add
        result = run.result
        add(None, "Status", run.status, GOOD if run.status == "success" else BAD)
        computed = cfd_compute_seconds(run)
        add(None, "CFD compute time", _fmt_seconds(computed))
        add(None, "This run", f"{_fmt_seconds(run.solve_seconds)} ({run.cache_state})")
        peak = run.memory.get("peak_bytes") if run.memory else None
        add(None, "Peak worker memory", "unavailable" if not peak else f"{peak / 2**20:.0f} MiB")
        if result is None or not result.ok:
            if result is not None and result.diagnostics.get("traceback"):
                add(None, "Traceback", result.diagnostics["traceback"][-400:])
            return
        info = result.diagnostics.get("cfd") or {}
        H = float(result.config["channel"]["height_m"])
        unit = result.config.get("display", {}).get("length_unit", "um")
        mesh = info.get("mesh_convergence")
        if mesh:
            ok = mesh["uncertainty_m"] <= cfd.TARGET_FRACTION_OF_H * H
            item = add(None, "Mesh uncertainty (final)", f"{_fmt_length(mesh['uncertainty_m'], unit)} "
                       f"({100 * mesh['uncertainty_fraction_of_H']:.2f}% of H)", GOOD if ok else WARN)
            add(item, "grids compared", f"{mesh['cells'][0]} vs {mesh['cells'][1]} cells across")
            add(item, "interface shift between grids", _fmt_length(mesh["interface_shift_m"], unit))
            add(item, "min IoU between grids", f"{mesh['min_iou']:.4f}")
            add(item, "estimate", mesh["method"])
            for station in mesh["stations"]:
                add(item, station["station"], f"shift {_fmt_length(station['interface_shift_m'], unit)}, "
                                              f"uncertainty {_fmt_length(station['uncertainty_m'], unit)}")
        else:
            add(None, "Mesh uncertainty", "not estimated (convergence check off)", WARN)
        checks = add(None, "Numerical checks", result.validity.get("status", ""))
        for check in result.validity.get("checks", []):
            add(checks, check["check"], ("ok  " if check["ok"] else "FAIL  ") + check["detail"],
                GOOD if check["ok"] else BAD)
        add(None, "Reynolds number", f"{result.diagnostics.get('reynolds_number', float('nan')):.3g}")
        for level in info.get("levels", []):
            label = "reference grid" if level is info["levels"][0] else "convergence grid"
            nx, ny, nz = level["grid"]
            item = add(None, f"{label}: {level['cells_across']} cells", f"{nx} x {ny} x {nz} = {level['cells']:,} "
                       f"cells, {level['unknowns']:,} unknowns")
            stokes = level.get("stokes") or {}
            add(item, "Stokes solve", f"{stokes.get('method')}, {stokes.get('iterations')} iterations, residual "
                                      f"{stokes.get('relative_residual', float('nan')):.1e}")
            picard = level.get("picard")
            if picard:
                add(item, "inertia (Picard)", f"{picard['iterations']} steps, converged {picard['converged']}, "
                                              f"velocity change {100 * picard['inertia_effect']:.3g}%")
            coupling = level.get("coupling")
            if coupling:
                add(item, "viscosity coupling", f"{len(coupling['iterations'])} iterations, converged "
                                                f"{coupling['converged']}")
            add(item, "continuity (worst plane)", f"{level['mass_balance']['relative_error']:.1e} of the flow")
            add(item, "pressure drop (x = 0 to outlet)", format_quantity(level.get("pressure_drop_Pa"), "mbar",
                                                                          "pressure"))
            add(item, "field", f"{level.get('field_cache')}, {_fmt_seconds(level.get('seconds'))}")
        final = result.final
        flux = add(None, "Flux fractions (final, CFD velocity)", "")
        expected = final.diagnostics.get("expected_flux_m3_s", {})
        total = sum(expected.values()) or 1.0
        for code, value in final.diagnostics.get("flux_fraction", {}).items():
            add(flux, f"code {code}", f"{100 * value:.2f}% (inlet {100 * expected.get(code, 0.0) / total:.2f}%)")
        if result.timings:
            stages = add(None, "Stage timings", "")
            for k, v in result.timings.items():
                add(stages, k, _fmt_seconds(v))
        settings = add(None, "Effective settings", "")
        for k, v in run.effective_settings.items():
            add(settings, k, json.dumps(v) if not isinstance(v, str) else v)
        for w in result.warnings:
            add(None, "warning", w, WARN)


class LegacyCFDPane(BackendPane):
    """The legacy CFD: the upstream map library, which entries were used, and the adapter's conventions."""

    show_station_x = True

    def __init__(self, parent=None):
        super().__init__("Legacy CFD (3D-hydrodynamic-flow-sculpting map library)", parent)

    def fill_tree(self, run: BackendRun) -> None:
        add = self._add
        result = run.result
        add(None, "Status", run.status, GOOD if run.status == "success" else (WARN if run.status == "unsupported"
                                                                              else BAD))
        if run.result is not None and run.result.ok:
            maps = len(run.result.diagnostics.get("operation_log", []))
            total = add(None, "Time including COMSOL (estimated)", _legacy_cfd_time(run))
            add(total, "COMSOL simulations", f"{maps} x {legacy_cfd.COMSOL_PILLAR['seconds_per_map'] / 60:.2f} min"
                                             f" = {_fmt_duration(legacy_cfd.comsol_seconds(maps))}")
            add(total, "basis", f"{legacy_cfd.COMSOL_PILLAR['basis']} ({legacy_cfd.COMSOL_PILLAR['maps']} files, "
                                f"{legacy_cfd.COMSOL_PILLAR['first_export']} to "
                                f"{legacy_cfd.COMSOL_PILLAR['last_export']})")
            add(total, "whole pillar library", _fmt_duration(legacy_cfd.comsol_seconds(
                legacy_cfd.COMSOL_PILLAR["maps"])))
        add(None, "Map application (in worker)", _fmt_seconds(run.solve_seconds))
        add(None, "End-to-end backend time", _fmt_seconds(run.end_to_end_seconds))
        add(None, "Worker startup (separate)", _fmt_seconds(run.startup_seconds))
        add(None, "Map library", run.cache_state)
        peak = run.memory.get("peak_bytes") if run.memory else None
        add(None, "Peak worker memory", "unavailable" if not peak else f"{peak / 2**20:.0f} MiB")
        upstream = legacy_cfd.UPSTREAM
        item = add(None, "Upstream tool", f"{upstream['name']} @ {upstream['revision'][:7]}")
        add(item, "source", upstream["url"])
        add(item, "license", upstream["license"])
        add(item, "libraries", upstream["libraries_url"])
        if result is not None and result.ok:
            d = result.diagnostics
            library = d.get("library")
            if library:
                entry = add(None, "Library file", Path(library["file"]).name)
                add(entry, "entries", str(library["entries"]))
                add(entry, "size", f"{library['bytes'] / 2**20:.0f} MiB")
                add(entry, "sha256", library["sha256"][:16] + "...")
                add(entry, "load time", _fmt_seconds(library["load_seconds"]))
            log = d.get("operation_log", [])
            maps = add(None, "Library maps applied", str(len(log)))
            for op in log:
                flags = [n for n, on in (("z-reflected", op["flip_z"]), ("y-reflected", op["mirror_y"])) if on]
                note = f" ({', '.join(flags)})" if flags else ""
                add(maps, op["pillar"], f"{op['key']}{note}; changed {100 * op['changed_fraction']:.1f}% of the section")
            final = result.final
            flux = add(None, "Flux fractions (final, duct profile)", "")
            expected = final.diagnostics.get("expected_flux_m3_s", {})
            total = sum(expected.values()) or 1.0
            for code, value in final.diagnostics.get("flux_fraction", {}).items():
                if int(code) >= 0:
                    add(flux, f"code {code}", f"{100 * value:.2f}% (inlet {100 * expected.get(code, 0.0) / total:.2f}%)")
            if result.timings:
                stages = add(None, "Stage timings", "")
                for k, v in result.timings.items():
                    add(stages, k, _fmt_seconds(v))
            for w in result.warnings:
                add(None, "warning", w, WARN)
        elif result is not None and result.diagnostics.get("traceback"):
            add(None, "Traceback", result.diagnostics["traceback"][-400:])
        settings = add(None, "Effective settings", "")
        for k, v in run.effective_settings.items():
            add(settings, k, json.dumps(v) if not isinstance(v, str) else v)


PERFORMANCE_ROWS = ("Status", "Solve time (in worker)", "End-to-end backend time", "Worker startup (separate)",
                    "Cache state", "Peak worker memory", "Speed vs CFD")
COLUMNS = (("legacy", "Legacy"), ("new", "New"), ("cfd", "CFD"), ("legacy_cfd", "Legacy CFD"))
REFERENCES = {"cfd": "CFD", "legacy_cfd": "legacy CFD"}  # what the new solver can be measured against


def _legacy_cfd_time(run: BackendRun) -> str:
    """The legacy CFD's realistic time: the COMSOL simulations behind its maps plus applying them."""
    comsol = run.result.diagnostics.get("comsol_seconds") if run.result is not None else None
    if comsol is None:
        return _fmt_seconds(run.solve_seconds)
    if not comsol:
        return f"{_fmt_seconds(run.solve_seconds)} (no COMSOL map used)"
    return f"~{_fmt_duration(realistic_seconds(run))} (COMSOL maps; applied in {_fmt_seconds(run.solve_seconds)})"


def _performance_values(name: str, run: BackendRun | None, reference: BackendRun | None) -> list[str]:
    if run is None:
        return ["—"] * len(PERFORMANCE_ROWS)
    peak = run.memory.get("peak_bytes") if run.memory else None
    solve = _fmt_seconds(run.solve_seconds)
    speed = "—"
    if name == "legacy_cfd":
        solve = _legacy_cfd_time(run)
    if name == "cfd":
        computed = cfd_compute_seconds(run)
        if computed and run.solve_seconds is not None and computed > 2 * run.solve_seconds:
            solve = f"{solve} (computed in {_fmt_seconds(computed)})"
        speed = "reference"
    elif run.cache_state == "map reused":
        speed = "n/a (map reused)"  # a cached lookup is not compared with a computed reference
    elif reference is not None and reference.result is not None and reference.result.ok:
        value = speed_against_cfd(run, reference).get("value")
        if value:
            speed = f"{value:.0f}x faster" if value >= 1 else f"{1 / value:.1f}x slower"
    return [run.status, solve, _fmt_seconds(run.end_to_end_seconds), _fmt_seconds(run.startup_seconds),
            run.cache_state, "unavailable" if not peak else f"{peak / 2**20:.0f} MiB", speed]


class ComparisonController(QtCore.QObject):
    """Legacy pane (left of the new results), CFD and legacy CFD panes (right of them), and the comparison strip."""

    new_run = Signal(object)  # BackendRun of the new solver from a comparison run
    busy_changed = Signal(bool)
    cfd_changed = Signal(object, bool)  # CFD result (or None) and whether it is stale

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
        self.pane = BackendPane("Legacy solver (frozen snapshot)")
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
        self.compare_new.setToolTip("Measure the new solver against a CFD result in the agreement table and summary")
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
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.summary)
        canvases = [self.pane.canvas, self.results.canvas, self.cfd_pane.canvas, self.legacy_cfd_pane.canvas]
        for source in canvases:
            source.limit_listeners.append(lambda s=source: self._sync(s, canvases))
        self.results.difference_changed.connect(lambda _: self._apply_difference())
        state.changed.connect(self._state_changed)
        self._state_changed()

    # ------------------------------------------------------------------ helpers
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
        self.summary.setText("Benchmark running...")
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
            self.summary.setText(f"Exported to {out}")

    def shutdown(self) -> None:
        self.runner.shutdown()

    # ------------------------------------------------------------------ inputs from the main window
    def set_new_result(self, result: SimulationResult | None) -> None:
        self.new_result = result
        if any(self.enabled.values()):
            self.redraw()

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
        self.summary.setText(f"Comparison failed: {message}")

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
        self._fill_summary()

    def fill_performance(self) -> None:
        reference = self.runs.get("cfd")
        runs = {"legacy": self.runs.get("legacy"), "new": self._new_run(), "cfd": reference,
                "legacy_cfd": self.runs.get("legacy_cfd")}
        for col, (name, _) in enumerate(COLUMNS):
            for row, value in enumerate(_performance_values(name, runs[name], reference)):
                self.performance.setItem(row, col, QtWidgets.QTableWidgetItem(value))

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

    def _fill_summary(self) -> None:
        cfg = self.state.config
        unit = cfg.display.length_unit
        parts = []
        comparison = self.legacy_comparison
        if self.enabled["legacy"] and comparison is not None:
            speed = comparison.speedup
            if speed.get("value"):
                parts.append(f"Speedup legacy/new = {speed['value']:.1f}x ({speed['scope']}; legacy "
                             f"{_fmt_seconds(speed['legacy_seconds'])}, new {_fmt_seconds(speed['new_seconds'])}).")
            elif "legacy" in comparison.runs:
                parts.append(f"No legacy/new speedup: {speed.get('reason', 'n/a')}.")
            if comparison.benchmark:
                stats = comparison.benchmark["statistics"]
                items = [f"{name} {scope}: median {_fmt_seconds(s['median'])} (n={s['n']}, IQR {_fmt_seconds(s['iqr'])})"
                         for name, scopes in stats.items() for scope, s in scopes.items()]
                parts.append("Benchmark: " + "; ".join(items))
        reference = self.runs.get("cfd")
        if self.enabled["cfd"] and reference is not None and reference.result is not None and reference.result.ok:
            result = reference.result
            mesh = (result.diagnostics.get("cfd") or {}).get("mesh_convergence")
            text = f"CFD reference computed in {_fmt_seconds(cfd_compute_seconds(reference))}"
            if mesh:
                text += (f"; its own mesh uncertainty is {_fmt_length(mesh['uncertainty_m'], unit)} "
                         f"({100 * mesh['uncertainty_fraction_of_H']:.2g}% of H, {mesh['cells'][0]} vs "
                         f"{mesh['cells'][1]} cells)")
            parts.append(text + ".")
            if self._cfd_stale():
                parts.insert(0, "STALE: the CFD reference belongs to other inputs; Run (F5) or Run CFD updates it.")
        elif self.enabled["cfd"]:
            parts.append("CFD reference not computed for these inputs yet (Run or Run CFD).")
        choice = self.new_reference()
        if choice == "legacy_cfd" and "new" not in self._accuracy:
            mapped = self.runs.get("legacy_cfd")
            parts.append("Legacy CFD cannot run this case (see its pane)." if mapped and mapped.status == "unsupported"
                         else "Legacy CFD not run for these inputs yet (Run, F5).")
        elif choice == "legacy_cfd" and self._legacy_cfd_stale():
            parts.insert(0, "STALE: the legacy CFD result belongs to other inputs; Run (F5) updates it.")
        for name, label in (("new", "New solver"), ("legacy", "Legacy"), ("legacy_cfd", "Legacy CFD")):
            entry = self._accuracy.get(name)
            if entry is None or (name != "new" and not self.enabled[name]):
                continue
            against = REFERENCES[entry.get("reference", "cfd")]
            verdict = "within the targets" if entry["within_targets"] else "outside the targets"
            if entry["within_cfd_uncertainty"] and entry["cfd_uncertainty_m"] > 0:
                verdict += " and within the CFD's own uncertainty"
            parts.append(f"{label} vs {against} at {entry['station']}: interface shift up to "
                         f"{_fmt_length(entry['max_interface_shift_m'], unit)}, IoU >= {entry['min_iou']:.3f}, "
                         f"flux misassigned <= {100 * entry['max_flux_misassigned']:.2f} pp: {verdict}.")
        notes = []
        if choice == "legacy_cfd":
            notes.append("the legacy CFD has no mesh-uncertainty estimate (its maps are quantized to 250 x 250 "
                         "nodes), so the plain targets apply")
        if self.enabled["legacy"]:
            notes.append("agreement with legacy is behavioural, not validation")
        if self.enabled["cfd"]:
            notes.append("the CFD reference solves the whole channel without the new solver's reduced-order "
                         "assumptions; it is verified numerically but not externally validated")
        if self.enabled["legacy_cfd"]:
            notes.append("legacy CFD chains precomputed COMSOL streamline maps and covers only geometries in its "
                         "libraries")
        if comparison is not None and not comparison.is_current(cfg) and self.enabled["legacy"]:
            parts.insert(0, "STALE: the project changed after this comparison; run again to update it.")
        self.summary.setText(" ".join(parts) + ("\nNote: " + "; ".join(notes) + "." if notes else ""))

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
