"""Desktop application: one window with setup, geometry, results, and optional comparisons.

The solver never runs on the GUI event loop: the new solver runs in a
persistent isolated worker (latest job wins), and the comparisons (frozen
legacy solver, CFD reference) drive their own isolated workers from a
background thread.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .. import __version__
from ..backends.base import bundle_root
from ..config import ProjectConfig
from ..materials import colors_for
from ..models.domain import FIDELITY_CELLS, plan_groups
from ..models.validity import MODEL_CARDS
from ..plotter import result_figure
from ..results import SimulationResult
from ..units import UNITS
from .bridge import SolverRunner
from .comparison_view import ComparisonController
from .geometry_view import GeometryEditor
from .panels import SetupPanel
from .qt import QtCore, QtGui, QtWidgets, QTimer
from .results_view import ACCURACY, ResultsView
from .state import ProjectState, autosave_path


def symbol_icon(kind: str) -> QtGui.QIcon:
    """A run (green triangle) or stop (red square) symbol, independent of the platform's icon theme."""
    pixmap = QtGui.QPixmap(64, 64)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    if kind == "run":
        painter.setBrush(QtGui.QColor("#2e9d4a"))
        painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(14, 6), QtCore.QPointF(58, 32), QtCore.QPointF(14, 58)]))
    else:
        painter.setBrush(QtGui.QColor("#c0392b"))
        painter.drawRoundedRect(QtCore.QRectF(12, 12, 40, 40), 5, 5)
    painter.end()
    return QtGui.QIcon(pixmap)


# Run times and CFD agreement measured on the examples (docs/validation_and_performance.md, section 3).
ACCURACY_TIPS = {fidelity: f"{text} ({FIDELITY_CELLS[fidelity]} flow cells across the smaller channel dimension)"
                 for fidelity, text in (
                     ("preview", "Low: fastest, 1-2 s on the examples; the default"),
                     ("medium", "Medium: typically 2.4x Low's run time; within the CFD targets on every example"),
                     ("reference", "High: typically 3.5x Low's run time; closest to the CFD on most examples"))}


def examples_dir() -> Path:
    return bundle_root() / "examples"


def example_entries() -> list[dict]:
    index = examples_dir() / "index.json"
    if not index.exists():
        return []
    return json.loads(index.read_text(encoding="utf-8"))["examples"]


def default_project() -> ProjectConfig:
    path = examples_dir() / "right_triangle_bottom.json"
    if path.exists():
        return ProjectConfig.load(path)
    from ..config import ChannelGeometry, FluidProperties, InletSpec

    return ProjectConfig(channel=ChannelGeometry(300e-6, 150e-6), materials=[FluidProperties("A", 0.02)],
                         inlets=[InletSpec("in1", "A", flow_rate=2e-9 / 60)])


class SequencePanel(QtWidgets.QWidget):
    """Physical operation sequence (ordered by station); toggle and reorder by swapping stations."""

    def __init__(self, state: ProjectState, on_select, parent=None):
        super().__init__(parent)
        self.state = state
        self.on_select = on_select
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QtWidgets.QLabel("<b>Operation sequence</b> (physical station order; features closer than the "
                                   "interaction distance form one coupled group)")
        heading.setWordWrap(True)
        layout.addWidget(heading)
        self.list = QtWidgets.QListWidget()
        self.list.itemChanged.connect(self._toggled)
        self.list.currentRowChanged.connect(self._selected)
        layout.addWidget(self.list, 1)
        buttons = QtWidgets.QHBoxLayout()
        for label, step in (("Move upstream", -1), ("Move downstream", 1)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda _=False, s=step: self._move(s))
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        state.changed.connect(self.refresh)
        self.refresh()

    def entries(self) -> list[tuple[str, int, float]]:
        cfg = self.state.config
        out = []
        for i, p in enumerate(cfg.pillars):
            out.append(("pillar", i, float(p.footprint()[:, 0].mean())))
        for i, inlet in enumerate(cfg.inlets):
            if inlet.placement == "port":
                out.append(("inlet", i, inlet.station))
        return sorted(out, key=lambda e: e[2])

    def refresh(self) -> None:
        cfg = self.state.config
        unit = cfg.display.length_unit
        scale = 1.0 / UNITS["length"][unit]
        self.list.blockSignals(True)
        self.list.clear()
        try:
            groups = plan_groups(cfg)
        except Exception:  # noqa: BLE001 - invalid geometry is reported by the setup panel
            groups = []
        group_of = {}
        for g in groups:
            for p in g.pillars:
                group_of[("pillar", p.id)] = g.index + 1
            for port in g.ports:
                group_of[("inlet", port.id)] = g.index + 1
        for kind, index, x in self.entries():
            spec = cfg.pillars[index] if kind == "pillar" else cfg.inlets[index]
            group = group_of.get((kind, spec.id))
            label = (f"x = {x * scale:8.1f} {unit}   {kind:6s} {spec.id}"
                     + (f"   (group {group})" if group else "   (disabled)"))
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, (kind, index))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if spec.enabled else QtCore.Qt.Unchecked)
            self.list.addItem(item)
        self.list.blockSignals(False)

    def _toggled(self, item: QtWidgets.QListWidgetItem) -> None:
        kind, index = item.data(QtCore.Qt.UserRole)
        enabled = item.checkState() == QtCore.Qt.Checked
        self.state.apply(lambda c: setattr((c.pillars if kind == "pillar" else c.inlets)[index], "enabled", enabled))

    def _selected(self, row: int) -> None:
        if row >= 0:
            self.on_select(*self.list.item(row).data(QtCore.Qt.UserRole))

    def _move(self, step: int) -> None:
        row = self.list.currentRow()
        entries = self.entries()
        other = row + step
        if row < 0 or not 0 <= other < len(entries):
            return
        (k1, i1, x1), (k2, i2, x2) = entries[row], entries[other]

        def shift(c: ProjectConfig, kind: str, index: int, dx: float) -> None:
            if kind == "pillar":
                c.pillars[index].x += dx
            else:
                c.inlets[index].station += dx

        def mutate(c: ProjectConfig) -> None:
            shift(c, k1, i1, x2 - x1)
            shift(c, k2, i2, x1 - x2)

        self.state.apply(mutate)
        self.list.setCurrentRow(other)


class MainWindow(QtWidgets.QMainWindow):
    """One combined window: setup on the left; geometry and results on the right.

    "Compare with legacy solver" adds the legacy result to the left of the new result,
    "Compare with CFD" adds the CFD reference to its right, and "Compare with legacy CFD" adds
    the vendored 3D-hydrodynamic-flow-sculpting map library right of that; any combination can
    be on. A strip below them shows every shown solver's performance and the agreement
    (accuracy against the CFD).
    Without comparisons the new result uses the whole width.
    """

    def __init__(self, config: ProjectConfig | None = None, path: Path | None = None):
        super().__init__()
        self.setWindowTitle(f"Crossflow simulator {__version__}")
        self.resize(1600, 1000)
        self.state = ProjectState(config or default_project(), path)
        self.runner = SolverRunner()
        self.runner.progress.connect(self._progress)
        self.runner.finished.connect(self._finished)
        self.runner.failed.connect(lambda job, msg: self._set_status(f"Job {job} failed: {msg}"))
        self.last_result: SimulationResult | None = None
        self._explicit_jobs: set[int] = set()
        self.live_timer = QTimer(self)
        self.live_timer.setSingleShot(True)
        self.live_timer.timeout.connect(lambda: self.run(live=True))

        self.setup = SetupPanel(self.state)
        self.sequence = SequencePanel(self.state, self._select_feature)
        self.geometry = GeometryEditor(self.state)
        self.results = ResultsView()
        self.results.export_requested.connect(self.export_result)
        self.comparison = ComparisonController(self.state, self.results, self)
        self.comparison.new_run.connect(self._comparison_new_run)
        self.comparison.busy_changed.connect(self._comparison_busy)
        self.results.station_changed.connect(self.comparison.set_station)
        self.results.visibility_changed.connect(lambda _: self._comparing() and self.comparison.redraw())

        top = self.geometry
        self.result_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.result_split.addWidget(self.comparison.pane)
        self.result_split.addWidget(self.results)
        self.result_split.addWidget(self.comparison.cfd_pane)
        self.result_split.addWidget(self.comparison.legacy_cfd_pane)
        self.result_split.setSizes([500, 700, 500, 500])
        results_area = QtWidgets.QWidget()
        area = QtWidgets.QVBoxLayout(results_area)
        area.setContentsMargins(0, 0, 0, 0)
        area.addWidget(self.result_split, 1)
        area.addWidget(self.comparison.strip)
        self.work = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.work.addWidget(top)
        self.work.addWidget(results_area)
        self.work.setSizes([400, 560])
        left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left.addWidget(self.setup)
        left.addWidget(self.sequence)
        left.setSizes([720, 240])
        main = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setup.setMinimumWidth(420)
        main.addWidget(left)
        main.addWidget(self.work)
        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)
        main.setSizes([460, 1140])
        self.setCentralWidget(main)
        self.setup.selection_changed.connect(self._table_selected)
        self.geometry.item_selected.connect(lambda kind, row: self.setup.select(kind, row))

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setRange(0, 100)
        self.estimate = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.estimate)
        self.statusBar().addPermanentWidget(self.progress)
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._update_comparisons()
        self.state.changed.connect(self._state_changed)
        self._state_changed()
        QTimer.singleShot(0, self._offer_recovery)

    # ------------------------------------------------------------------ actions, menus, toolbar
    def _build_actions(self) -> None:
        self.run_action = self._make_action("&Run", lambda: self.run(), "F5")
        self.run_action.setIcon(symbol_icon("run"))
        self.run_action.setToolTip("Run the simulation (F5): the new solver, then every comparison that is on")
        self.cancel_action = self._make_action("&Cancel", self.cancel, "Esc")
        self.cancel_action.setIcon(symbol_icon("stop"))
        self.cancel_action.setToolTip("Cancel the running calculation (Esc); the last result stays visible")
        self.live = QtGui.QAction("&Auto-update on edit", self, checkable=True)
        self.live.setToolTip("Run the new solver automatically after every edit (latest edit wins)")
        self.live.setChecked(True)
        self.live.toggled.connect(lambda on: on and self.live_timer.start(300))
        self.compare = QtGui.QAction("Compare with &legacy solver", self, checkable=True)
        self.compare.setToolTip("Show the frozen legacy solver's result left of the new result; Run then runs both")
        self.compare.toggled.connect(lambda _: self._update_comparisons())
        self.compare_cfd = QtGui.QAction("Compare with &CFD", self, checkable=True)
        self.compare_cfd.setToolTip("Show the CFD reference (full-channel 3D Navier-Stokes, the ground truth for the "
                                    "reduced-order solver) right of the new result, with accuracy and performance; "
                                    "Run then also computes it (or loads the stored one)")
        self.compare_cfd.toggled.connect(lambda _: self._update_comparisons())
        self.compare_legacy_cfd = QtGui.QAction("Compare with le&gacy CFD", self, checkable=True)
        self.compare_legacy_cfd.setToolTip("Show the vendored 3D-hydrodynamic-flow-sculpting tool (precomputed COMSOL "
                                           "streamline maps) right of the other results; Run then also runs it. It "
                                           "covers only geometries in its libraries.")
        self.compare_legacy_cfd.toggled.connect(lambda _: self._update_comparisons())
        self.accuracy_group = QtGui.QActionGroup(self)  # exclusive
        self.accuracy_actions: dict[str, QtGui.QAction] = {}
        for fidelity, label in ACCURACY.items():
            action = QtGui.QAction(label, self, checkable=True)
            action.setToolTip(ACCURACY_TIPS[fidelity])
            action.triggered.connect(lambda _=False, f=fidelity: self.set_accuracy(f))
            self.accuracy_group.addAction(action)
            self.accuracy_actions[fidelity] = action

    def _make_action(self, text, slot, shortcut=None) -> QtGui.QAction:
        action = QtGui.QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        action.triggered.connect(slot)
        return action

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file = bar.addMenu("&File")
        self._action(file, "&New from template...", self.open_example, "Ctrl+N")
        self._action(file, "&Open...", self.open_project, QtGui.QKeySequence.Open)
        self._action(file, "&Save", self.save_project, QtGui.QKeySequence.Save)
        self._action(file, "Save &as...", self.save_project_as, QtGui.QKeySequence.SaveAs)
        file.addSeparator()
        self._action(file, "&Export result...", self.export_result, "Ctrl+E")
        file.addSeparator()
        self._action(file, "&Quit", self.close, QtGui.QKeySequence.Quit)
        edit = bar.addMenu("&Edit")
        self.undo_action = self._action(edit, "&Undo", self.state.undo, QtGui.QKeySequence.Undo)
        self.redo_action = self._action(edit, "&Redo", self.state.redo, QtGui.QKeySequence.Redo)
        bar.addAction(self.run_action)  # a button, not a menu: one click runs
        compare = bar.addMenu("&Compare")
        for action in (self.compare, self.compare_cfd, self.compare_legacy_cfd):
            compare.addAction(action)
        solver = bar.addMenu("&Solver")
        solver.addAction(self.cancel_action)
        solver.addAction(self.live)
        accuracy = solver.addMenu("&Accuracy")
        accuracy.setToolTipsVisible(True)
        for action in self.accuracy_actions.values():
            accuracy.addAction(action)
        solver.addSeparator()
        self._action(solver, "Restart solver worker", self.runner.restart)
        help_menu = bar.addMenu("&Help")
        self._action(help_menu, "Model card and validity", self.show_model_card)
        self._action(help_menu, "&About", self.about)

    def _build_toolbar(self) -> None:
        tools = self.addToolBar("Main")
        tools.setMovable(False)
        tools.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tools.addWidget(QtWidgets.QLabel(" Example "))
        self.examples = QtWidgets.QComboBox()
        self.examples.setMinimumWidth(260)
        for entry in example_entries():
            suffix = "" if entry["shared_capability"] else "  (new solver only)"
            self.examples.addItem(entry["name"] + suffix, entry["file"])
        tools.addWidget(self.examples)
        load = QtWidgets.QPushButton("Load")
        load.clicked.connect(self.load_selected_example)
        tools.addWidget(load)
        tools.addSeparator()
        tools.addAction(self.run_action)
        tools.addAction(self.cancel_action)
        tools.addWidget(QtWidgets.QLabel(" Accuracy "))
        self.accuracy = QtWidgets.QComboBox()
        for fidelity, label in ACCURACY.items():
            self.accuracy.addItem(label, fidelity)
            self.accuracy.setItemData(self.accuracy.count() - 1, ACCURACY_TIPS[fidelity],
                                      QtCore.Qt.ItemDataRole.ToolTipRole)
        self.accuracy.setToolTip("The new solver's accuracy: a finer flow grid, closer to the CFD but slower. Saved "
                                 "with the project (the CLI's --fidelity preview / medium / reference).")
        self.accuracy.currentIndexChanged.connect(lambda _: self.set_accuracy(self.accuracy.currentData()))
        tools.addWidget(self.accuracy)
        tools.addSeparator()
        tools.addAction(self.live)
        tools.addAction(self.compare)
        tools.addAction(self.compare_cfd)
        tools.addAction(self.compare_legacy_cfd)

    def _action(self, menu, text, slot, shortcut=None) -> QtGui.QAction:
        action = self._make_action(text, slot, shortcut)
        menu.addAction(action)
        return action

    # ------------------------------------------------------------------ project files
    def open_example(self) -> None:
        entries = example_entries()
        if not entries:
            return
        names = [f"{e['name']}" + ("" if e["shared_capability"] else "  (new solver only)") for e in entries]
        choice, ok = QtWidgets.QInputDialog.getItem(self, "New from template", "Template", names, 0, False)
        if ok:
            entry = entries[names.index(choice)]
            self.state.replace(ProjectConfig.load(examples_dir() / entry["file"]), None)

    def load_selected_example(self) -> None:
        file = self.examples.currentData()
        if file:
            self.state.replace(ProjectConfig.load(examples_dir() / file), None)

    def open_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", "", "Projects (*.json)")
        if path:
            try:
                self.state.load(path)
            except Exception as exc:  # noqa: BLE001 - shown to the user
                QtWidgets.QMessageBox.critical(self, "Cannot open project", str(exc))

    def save_project(self) -> None:
        if self.state.path is None:
            self.save_project_as()
        else:
            self.state.save()
            self._set_status(f"Saved {self.state.path}")

    def save_project_as(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", "project.json", "Projects (*.json)")
        if path:
            self.state.save(path)
            self._set_status(f"Saved {path}")

    def export_result(self) -> None:
        if self.last_result is None:
            QtWidgets.QMessageBox.information(self, "Export", "Run a calculation first.")
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Export result")
        if not directory:
            return
        out = Path(directory)
        self.last_result.save(out / "result.npz")
        ProjectConfig.from_dict(self.last_result.config).save(out / "config.json")
        cfg = self.state.config
        result_figure(self.last_result, colors_for(self.last_result.streams, cfg.display.color_by),
                      cfg.display.length_unit).savefig(out / "sections.png", dpi=200)
        (out / "diagnostics.json").write_text(json.dumps({
            "status": self.last_result.status, "validity": self.last_result.validity,
            "timings": self.last_result.timings, "provenance": self.last_result.provenance,
            "solver": self.last_result.solver}, indent=2, default=str), encoding="utf-8")
        self._set_status(f"Exported to {out}")

    def _offer_recovery(self) -> None:
        payload = ProjectState.recoverable(self.state.path)
        if payload is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Recover unsaved work", "An autosaved version of this project is newer than the saved file. "
                                          "Recover it?")
        if answer == QtWidgets.QMessageBox.Yes:
            self.state.replace(ProjectConfig.from_dict(payload["config"]), self.state.path)
        else:
            autosave_path(self.state.path).unlink(missing_ok=True)

    # ------------------------------------------------------------------ running
    def set_accuracy(self, fidelity: str) -> None:
        """Choose the new solver's accuracy (a project input: a result at another accuracy is stale)."""
        if fidelity in ACCURACY and self.state.config.model.fidelity != fidelity:
            self.state.apply(lambda c: setattr(c.model, "fidelity", fidelity))

    def _show_accuracy(self) -> None:
        fidelity = self.state.config.model.fidelity
        self.accuracy.blockSignals(True)
        self.accuracy.setCurrentIndex(max(self.accuracy.findData(fidelity), 0))
        self.accuracy.blockSignals(False)
        if fidelity in self.accuracy_actions:
            self.accuracy_actions[fidelity].setChecked(True)

    def run(self, live: bool = False) -> None:
        errors = self.state.errors()
        if errors:
            if not live:
                QtWidgets.QMessageBox.warning(self, "Invalid project", "\n".join(str(e) for e in errors))
            return
        cfg = self.state.config.copy()
        accuracy = ACCURACY.get(cfg.model.fidelity, cfg.model.fidelity)
        if self._comparing() and not live:
            if self.comparison.busy:
                self._set_status("A comparison is still running; cancel it first (Esc)")
                return
            self.comparison.run(cfg)
            others = [n for n, a in (("legacy", self.compare), ("legacy CFD", self.compare_legacy_cfd),
                                     ("the CFD reference", self.compare_cfd)) if a.isChecked()]
            self._set_status(f"Comparison: new solver ({accuracy} accuracy) first, then {' and '.join(others)}, "
                             "in isolated workers")
            return
        estimate = self.runner.estimate(cfg.model.fidelity)
        self.estimate.setText(f"~{estimate:.1f} s" if estimate else "no time estimate yet")
        try:
            job = self.runner.submit(cfg)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Could not start the solver worker: {exc}")
            return
        if not live:
            self._explicit_jobs.add(job)
        self.progress.setValue(0)
        self._set_status(f"Job {job}: calculation started ({accuracy} accuracy)")

    def cancel(self) -> None:
        self.runner.cancel()
        self.comparison.cancel()
        self._set_status("Cancel requested; the last successful result stays visible")

    def _progress(self, job: int, fraction: float, message: str) -> None:
        self.progress.setValue(int(100 * fraction))
        self._set_status(f"Job {job}: {message}")

    def _finished(self, job: int, result: SimulationResult) -> None:
        self.progress.setValue(100)
        explicit = job in self._explicit_jobs
        self._explicit_jobs.discard(job)
        if result.status == "cancelled":
            self._set_status(f"Job {job} cancelled; showing the last successful result")
            return
        if result.ok:
            self.show_result(result)
            self._set_status(f"Job {job}: {result.message} ({result.timings.get('total', 0):.2f} s)")
        else:
            self._set_status(f"Job {job} {result.status}: {result.message}")
            if explicit:
                QtWidgets.QMessageBox.warning(self, f"Calculation {result.status}", result.message[:800])

    def _comparison_new_run(self, run) -> None:
        if run.result is not None and run.result.ok:
            self.show_result(run.result)
            self._set_status(f"New solver: {run.result.message}; comparisons running...")
        else:
            self._set_status(f"New solver {run.status}: {run.message}")

    def _comparison_busy(self, busy: bool) -> None:
        if not busy and self.comparison.comparison is not None:
            self._set_status("Comparison finished")

    def show_result(self, result: SimulationResult) -> None:
        """Show a new-solver result (never replacing a current one with an outdated one)."""
        current = self._is_current(result)
        if self.last_result is not None and self._is_current(self.last_result) and not current:
            return
        self.last_result = result
        cfg = self.state.config
        self.results.show_result(result, cfg.display.length_unit, cfg.display.color_by)
        self.results.mark_stale(not current)
        self.comparison.set_new_result(result)

    def _is_current(self, result: SimulationResult) -> bool:
        return result.input_hash == self.state.config.input_hash()  # the accuracy is one of the inputs

    def _comparing(self) -> bool:
        return self.compare.isChecked() or self.compare_cfd.isChecked() or self.compare_legacy_cfd.isChecked()

    def _update_comparisons(self) -> None:
        legacy, cfd_on = self.compare.isChecked(), self.compare_cfd.isChecked()
        mapped = self.compare_legacy_cfd.isChecked()
        self.comparison.pane.setVisible(legacy)
        self.comparison.cfd_pane.setVisible(cfd_on)
        self.comparison.legacy_cfd_pane.setVisible(mapped)
        shown = legacy + cfd_on + mapped
        self.comparison.strip.setVisible(shown > 0)
        self.comparison.set_enabled(legacy, cfd_on, mapped)
        height = max(self.work.height(), 2)
        top = height * (30 if shown else 42) // 100  # comparisons need room for the strip
        self.work.setSizes([top, height - top])
        if shown:
            width = max(self.result_split.width(), 4)
            side = width * {1: 40, 2: 30, 3: 24}[shown] // 100  # the new result keeps the largest share
            self.result_split.setSizes([side if legacy else 0, width - shown * side, side if cfd_on else 0,
                                        side if mapped else 0])

    def _state_changed(self) -> None:
        name = self.state.path.name if self.state.path else "untitled"
        self.setWindowTitle(f"{name}{' *' if self.state.dirty else ''} — Crossflow simulator {__version__}")
        self.undo_action.setEnabled(self.state.can_undo())
        self.redo_action.setEnabled(self.state.can_redo())
        self._show_accuracy()
        if self.last_result is not None:
            self.results.mark_stale(not self._is_current(self.last_result))
        if self.live.isChecked():
            self.live_timer.start(700)  # debounce rapid edits

    def _select_feature(self, kind: str, index: int) -> None:
        self.setup.select(kind, index)

    def _table_selected(self, kind: str, row: int) -> None:
        for item in self.geometry.top_scene.items():
            if getattr(item, "kind", None) == kind and getattr(item, "index", None) == row:
                self.geometry.top_scene.clearSelection()
                item.setSelected(True)

    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def show_model_card(self) -> None:
        text = "\n\n".join(
            f"{card['title']}\n{card['description']}\nExternally validated: {card['externally_validated']}\n"
            f"Envelope: {json.dumps(card['envelope'])}" for card in MODEL_CARDS.values())
        QtWidgets.QMessageBox.information(self, "Models and validity", text)

    def about(self) -> None:
        QtWidgets.QMessageBox.about(self, "About", f"Crossflow simulator {__version__}\n"
                                                   "New solver: 3D Stokes flow maps with continuous transport.\n"
                                                   "Legacy solver: frozen snapshot run in an isolated worker.\n"
                                                   "CFD reference: full-channel 3D Navier-Stokes in its own worker.")

    def closeEvent(self, event):  # noqa: N802 - Qt API
        if self.state.dirty:
            answer = QtWidgets.QMessageBox.question(self, "Unsaved changes", "Save changes before closing?",
                                                    QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard
                                                    | QtWidgets.QMessageBox.Cancel)
            if answer == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QtWidgets.QMessageBox.Save:
                self.save_project()
        self.runner.shutdown()
        self.comparison.shutdown()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])
    app.setApplicationName("Crossflow simulator")
    config, path = None, None
    if argv:
        path = Path(argv[0])
        config = ProjectConfig.load(path)
    window = MainWindow(config, path)
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
