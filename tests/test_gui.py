"""Offscreen GUI acceptance checks (PySide6)."""
import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from crossflow_simulator.config import ProjectConfig  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def app():
    from crossflow_simulator.gui.qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(["tests"])


@pytest.fixture
def window(app, examples_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("CROSSFLOW_CACHE_DIR", str(tmp_path / "cache"))
    from crossflow_simulator.gui.app import MainWindow

    w = MainWindow(ProjectConfig.load(examples_dir / "parallel_unequal.json"), None)
    w.live.setChecked(False)  # tests start runs explicitly; auto-update has its own test
    w.live_timer.stop()
    w.show()
    yield w
    w.view3d.timer.stop()
    w.runner.shutdown()
    w.comparison.shutdown()
    w.state.dirty = False
    w.close()


def wait(app, condition, timeout):
    started = time.time()
    while not condition() and time.time() - started < timeout:
        app.processEvents()
        time.sleep(0.02)
    return condition()


def test_launch_run_preview_and_mark_stale(app, window):
    window.run()
    assert wait(app, lambda: window.last_result is not None, 120)
    assert window.last_result.ok
    assert not window.results.stale
    window.state.apply(lambda c: setattr(c.inlets[0], "flow_rate", c.inlets[0].flow_rate * 2))
    app.processEvents()
    assert window.results.stale
    window.state.undo()
    app.processEvents()
    assert not window.results.stale


def test_one_click_run_button_in_the_menu_bar(app, window):
    from PySide6.QtTest import QTest

    from crossflow_simulator.gui.qt import QtCore, QtWidgets

    bar = window.menuBar()
    entries = {a.text().replace("&", ""): a for a in bar.actions()}
    assert list(entries) == ["File", "Edit", "Run", "Compare", "Solver", "Help"]
    run = entries["Run"]
    assert run is window.run_action and run.menu() is None and not run.icon().isNull()  # a button, not a menu
    assert entries["Compare"].menu().actions() == [window.compare, window.compare_cfd, window.compare_legacy_cfd]
    assert {window.cancel_action, window.live} <= set(entries["Solver"].menu().actions())
    # no separate reference run of the new solver: the reference is the CFD comparison (Accuracy is Low/Medium/High)
    labels = [a.text() for menu in bar.findChildren(QtWidgets.QMenu) for a in menu.actions()]
    assert not any("reference" in label.lower() for label in labels) and not hasattr(window.setup, "fidelity")
    QTest.mouseClick(bar, QtCore.Qt.MouseButton.LeftButton, pos=bar.actionGeometry(run).center())
    assert wait(app, lambda: window.last_result is not None, 120)
    assert window.last_result.ok and window.last_result.provenance["fidelity"] == "preview"


def test_accuracy_low_medium_high_chooses_the_new_solvers_flow_grid(app, window):
    combo = window.accuracy
    assert [combo.itemText(i) for i in range(combo.count())] == ["Low", "Medium", "High"]
    assert combo.currentText() == "Low" and window.accuracy_actions["preview"].isChecked()
    solver = next(a.menu() for a in window.menuBar().actions() if a.text().replace("&", "") == "Solver")
    accuracy = next(a.menu() for a in solver.actions() if a.menu() and a.text().replace("&", "") == "Accuracy")
    assert [a.text() for a in accuracy.actions()] == ["Low", "Medium", "High"]
    window.run()
    assert wait(app, lambda: window.last_result is not None, 120)
    low = window.last_result
    # the accuracy is a project input: saved and undoable, and a result at another accuracy is stale
    combo.setCurrentIndex(combo.findData("medium"))
    app.processEvents()
    assert window.state.config.model.fidelity == "medium" and window.accuracy_actions["medium"].isChecked()
    assert window.results.stale
    window.accuracy_actions["reference"].trigger()
    app.processEvents()
    assert window.state.config.model.fidelity == "reference" and combo.currentText() == "High"
    window.run()
    assert wait(app, lambda: window.last_result is not low and window._is_current(window.last_result), 180)
    assert window.last_result.provenance["fidelity"] == "reference" and not window.results.stale
    tree = window.results.diagnostics
    rows = {tree.topLevelItem(i).text(0): tree.topLevelItem(i).text(1) for i in range(tree.topLevelItemCount())}
    assert rows["Accuracy"].startswith("High (20 flow cells")
    window.state.undo()
    app.processEvents()
    assert combo.currentText() == "Medium" and window.results.stale


def test_latest_job_wins(app, window):
    first = window.runner.submit(window.state.config)
    second = window.runner.submit(window.state.config)
    assert second > first
    assert wait(app, lambda: window.last_result is not None, 120)
    assert window.runner.latest_job == second


def test_save_load_round_trip_keeps_parameters_and_colours(app, window, tmp_path):
    window.state.apply(lambda c: setattr(c.materials[0], "color", "#123456"))
    path = tmp_path / "project.json"
    window.state.save(path)
    loaded = ProjectConfig.load(path)
    assert loaded == window.state.config
    assert loaded.materials[0].color == "#123456"


def test_setup_table_edit_converts_display_units(app, window):
    table = window.setup.inlets
    column = [c.attr for c in table.columns].index("flow_rate")
    table.item(0, column).setText("7")  # uL/min
    app.processEvents()
    assert window.state.config.inlets[0].flow_rate == pytest.approx(7e-9 / 60)


def test_invalid_edit_is_flagged_inline(app, window):
    window.state.apply(lambda c: setattr(c.inlets[0], "flow_rate", -1e-12))
    app.processEvents()
    table = window.setup.inlets
    column = [c.attr for c in table.columns].index("flow_rate")
    assert table.item(0, column).toolTip()
    window.state.undo()
    app.processEvents()
    assert not table.item(0, column).toolTip()  # cells update in place, so the flag clears


@pytest.mark.parametrize("name", ["materials", "inlets"])
def test_double_click_edits_every_time_after_a_commit(app, window, name):
    # a commit refreshes the table; that used to destroy the closing editor and leave the view in its
    # editing state, so later double-clicks opened no editor
    from PySide6.QtTest import QTest

    from crossflow_simulator.gui.qt import QtCore, QtWidgets

    table = getattr(window.setup, name)
    window.setup.toolbox.setCurrentWidget(table.parentWidget() if name == "materials"
                                          else table.parentWidget().parentWidget())
    attr = "viscosity" if name == "materials" else "flow_rate"
    column = [c.attr for c in table.columns].index(attr)
    app.processEvents()

    def edit(row, text, key=None):
        pos = table.visualRect(table.model().index(row, column)).center()
        QTest.mouseClick(table.viewport(), QtCore.Qt.LeftButton, pos=pos)
        QTest.mouseDClick(table.viewport(), QtCore.Qt.LeftButton, pos=pos)
        app.processEvents()
        editor = QtWidgets.QApplication.focusWidget()
        assert isinstance(editor, QtWidgets.QLineEdit), f"no editor on row {row}"
        editor.setText(text)
        if key is not None:
            QTest.keyClick(editor, key)
            app.processEvents()
            assert QtWidgets.QAbstractItemView.state(table) == QtWidgets.QAbstractItemView.NoState

    edit(0, "3", QtCore.Qt.Key_Return)
    edit(1, "4")  # committed by the next double-click taking the focus
    edit(0, "5", QtCore.Qt.Key_Return)
    values = [getattr(x, attr) for x in getattr(window.state.config, name)[:2]]
    unit = 1e-3 if name == "materials" else 1e-9 / 60  # mPa s, uL/min
    assert values == [pytest.approx(5 * unit), pytest.approx(4 * unit)]


def test_add_pillar_offers_a_non_isosceles_trapezoid(app, window):
    from crossflow_simulator import legacy_cfd
    from crossflow_simulator.backends.legacy import build_plan
    from crossflow_simulator.gui.qt import QtWidgets

    button = next(b for b in window.setup.findChildren(QtWidgets.QPushButton) if b.text() == "Add pillar")
    shapes = {a.text(): a for a in button.menu().actions()}
    assert list(shapes) == ["Right triangle", "Non-isosceles trapezoid"]
    before = len(window.state.config.pillars)
    shapes["Non-isosceles trapezoid"].trigger()
    app.processEvents()
    pillars = window.state.config.pillars
    assert len(pillars) == before + 1 and window.setup.pillars.rowCount() == before + 1
    v = np.asarray(pillars[-1].vertices)
    assert len(v) == 4
    sides = np.roll(v, -1, axis=0) - v
    assert sides[0][1] == 0 and sides[2][1] == 0 and abs(sides[0][0]) != abs(sides[2][0])  # one pair parallel
    assert not np.isclose(np.hypot(*sides[1]), np.hypot(*sides[3]))  # legs of different length
    # the new solver and legacy run it; the legacy CFD library has only triangles and says so
    assert not [r for r in build_plan(window.state.config)["unsupported"] if pillars[-1].id in r]
    assert any("4-vertex footprint" in r for r in legacy_cfd.build_plan(window.state.config)["unsupported"])
    window.run()
    assert wait(app, lambda: window.last_result is not None, 120)
    assert window.last_result.ok and window._is_current(window.last_result)


def test_one_window_without_tabs_and_optional_comparison_panes(app, window):
    from crossflow_simulator.gui.qt import QtWidgets

    controller = window.comparison
    assert not window.findChildren(QtWidgets.QTabWidget)
    for widget in (window.setup, window.geometry, window.view3d, window.results, window.sequence):
        assert widget.isVisibleTo(window)
    assert not any(w.isVisibleTo(window) for w in (controller.pane, controller.cfd_pane, controller.legacy_cfd_pane,
                                                   controller.strip))
    window.compare.setChecked(True)
    app.processEvents()
    assert controller.pane.isVisibleTo(window) and controller.strip.isVisibleTo(window)
    assert "Ready" in controller.pane.status.text()
    window.compare_cfd.setChecked(True)  # all three results side by side
    app.processEvents()
    assert controller.pane.isVisibleTo(window) and controller.cfd_pane.isVisibleTo(window)
    assert "Ready" in controller.cfd_pane.status.text()
    sizes = window.result_split.sizes()
    assert sizes[0] > 0 and sizes[2] > 0 and sizes[1] > sizes[0]
    window.compare_legacy_cfd.setChecked(True)  # all four results side by side
    app.processEvents()
    assert controller.legacy_cfd_pane.isVisibleTo(window) and "Ready" in controller.legacy_cfd_pane.status.text()
    sizes = window.result_split.sizes()
    assert all(s > 0 for s in sizes) and sizes[1] >= max(sizes[0], sizes[2], sizes[3])
    assert not controller.performance.isColumnHidden(3)
    # the only differences on offer are to the CFD and the legacy CFD, once their comparison is on
    labels = [w.text() for w in window.results.findChildren(QtWidgets.QAbstractButton)]
    assert "Difference to CFD" in labels and "Difference to legacy CFD" in labels
    assert not any("reference" in label.lower() for label in labels)
    assert all(box.isEnabled() for box in window.results.differences.values())
    window.compare.setChecked(False)
    window.compare_legacy_cfd.setChecked(False)
    app.processEvents()
    assert not controller.pane.isVisibleTo(window) and controller.cfd_pane.isVisibleTo(window)
    assert not controller.legacy_cfd_pane.isVisibleTo(window) and controller.performance.isColumnHidden(3)
    assert not window.results.differences["legacy_cfd"].isEnabled()
    window.compare_cfd.setChecked(False)
    app.processEvents()
    assert not controller.strip.isVisibleTo(window)


def test_compare_mode_explains_cases_legacy_cannot_run(app, window, examples_dir):
    window.compare.setChecked(True)
    window.state.replace(ProjectConfig.load(examples_dir / "viscosity_contrast_layers.json"), None)
    app.processEvents()
    assert "Not supported by legacy" in window.comparison.pane.status.text()
    window.compare_legacy_cfd.setChecked(True)
    window.state.replace(ProjectConfig.load(examples_dir / "right_triangle_bottom.json"), None)
    app.processEvents()
    assert "Not supported by legacy CFD" in window.comparison.legacy_cfd_pane.status.text()


def test_3d_view_animates_the_computed_flow_and_hides_it_when_stale(app, window, examples_dir):
    window.state.replace(ProjectConfig.load(examples_dir / "right_triangle_bottom.json"), None)
    view = window.view3d
    assert not view.flow_shown and "run a calculation" in view.message
    window.run()
    assert wait(app, lambda: window.last_result is not None and view.flow_shown, 120)
    cfg = window.state.config
    before = view.particle_positions()
    view.advance(0.5)
    after = view.particle_positions()
    assert len(before) > 100 and not np.allclose(before, after)
    # cells without a valid trajectory were filled from their neighbours, and the results view says how many
    tree = window.results.diagnostics
    trajectories = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                        if tree.topLevelItem(i).text(0) == "Trajectories (final)")
    rows = {trajectories.child(i).text(0): trajectories.child(i).text(1) for i in range(trajectories.childCount())}
    filled = int(window.last_result.final.filled.sum())
    assert filled > 0 and rows["Filled from neighbours"] == f"{filled} ({100 * filled / 12800:.3g} %)"
    assert np.all((after[:, 1] >= -1e-9) & (after[:, 1] <= cfg.channel.width + 1e-9))
    assert np.all((after[:, 2] >= -1e-9) & (after[:, 2] <= cfg.channel.height + 1e-9))
    window.results.slider.setValue(0)  # the selected station is shown in 3D
    assert view.station == 0
    window.state.apply(lambda c: setattr(c.pillars[0], "x", c.pillars[0].x + 50e-6))
    app.processEvents()
    assert not view.flow_shown and len(view.particle_positions()) == 0
    assert "Inputs changed" in view.message


def test_auto_update_runs_a_preview_after_an_edit(app, window):
    window.live.setChecked(True)
    window.state.apply(lambda c: setattr(c.inlets[0], "flow_rate", c.inlets[0].flow_rate * 1.5))
    assert wait(app, lambda: window.last_result is not None and window._is_current(window.last_result), 120)
    assert window.view3d.flow_shown


@pytest.mark.legacy
def test_compare_with_legacy_runs_both_and_shows_performance(app, window, tmp_path, examples_dir):
    window.state.replace(ProjectConfig.load(examples_dir / "parallel_unequal.json"), None)
    window.compare.setChecked(True)
    window.run()
    controller = window.comparison
    assert wait(app, lambda: controller.comparison is not None, 300)
    runs = controller.comparison.runs
    assert runs["legacy"].status == "success" and runs["new"].status == "success"
    assert window.last_result is runs["new"].result  # the new side is the main results view
    assert controller.metrics.rowCount() == 3  # legacy vs new, per stream
    assert "Speedup" in controller.summary.text()
    assert controller.performance.item(0, 0).text() == "success" and controller.performance.item(0, 1).text() == "success"
    assert not any(box.isEnabled() for box in window.results.differences.values())  # no difference to legacy
    from crossflow_simulator.comparison import export_comparison

    out = export_comparison(controller.comparison, tmp_path / "report")
    assert (out / "side_by_side.png").exists()


def test_compare_with_cfd_shows_the_reference_accuracy_and_3d_flow(app, window, examples_dir):
    cfg = ProjectConfig.load(examples_dir / "right_triangle_bottom.json")
    cfg.cfd.cells = 12
    cfg.grid.output_ny = 64
    window.state.replace(cfg, None)
    window.compare_cfd.setChecked(True)
    controller = window.comparison
    window.run()
    assert wait(app, lambda: not controller.busy and "cfd" in controller.runs, 600)
    run = controller.runs["cfd"]
    assert run.status == "success", run.message
    assert window.last_result is controller.runs["new"].result
    compared = [controller.metrics.item(r, 0).text() for r in range(controller.metrics.rowCount())]
    assert compared == ["New vs CFD", "New vs CFD"]
    assert "mesh uncertainty" in controller.summary.text() and "New solver vs CFD" in controller.summary.text()
    assert controller.performance.item(0, 2).text() == "success"
    assert "faster" in controller.performance.item(6, 1).text()  # speed of the new solver vs the CFD
    assert "Mesh uncertainty (final)" in [controller.cfd_pane.performance.topLevelItem(i).text(0)
                                  for i in range(controller.cfd_pane.performance.topLevelItemCount())]
    # the CFD pane follows the station slider; the overlay marks where the new result differs from the CFD
    window.results.slider.setValue(0)
    app.processEvents()
    assert "inlet plane" in controller.cfd_pane.canvas.axes.get_title()
    window.results.differences["cfd"].setChecked(True)
    assert window.results.reference[0] == "CFD reference" and window.results.slider.value() == 0
    diagnostics = window.results.diagnostics
    assert "Difference" in [diagnostics.topLevelItem(i).text(0) for i in range(diagnostics.topLevelItemCount())]
    # the 3D view can animate the CFD flow
    view = window.view3d
    assert view.flow_source.isVisibleTo(window)
    view.flow_source.setCurrentIndex(view.flow_source.findData("cfd"))
    assert view.flow_shown and "CFD reference" in view.message
    # Recompute discards the stored CFD result and computes it again, measuring its compute time now
    assert controller.cfd_pane.recompute_button.isEnabled()
    controller.cfd_pane.recompute_button.click()
    assert wait(app, lambda: not controller.busy and "cfd" in controller.runs, 600)
    recomputed = controller.runs["cfd"]
    assert recomputed.status == "success" and recomputed.cache_state.startswith("recomputed")
    assert recomputed.result.diagnostics["computed_seconds"] > 0
    # choosing a grid is a CFD setting only: the new result stays current, the CFD becomes stale
    controller.cfd_pane.grid_chosen.emit(16)
    app.processEvents()
    assert window.state.config.cfd.cells == 16 and not window.results.stale
    assert "STALE" in controller.cfd_pane.status.text() and not view.flow_shown


@pytest.mark.legacy
def test_all_four_results_side_by_side_with_accuracy_against_the_cfd(app, window, examples_dir):
    cfg = ProjectConfig.load(examples_dir / "parallel_unequal.json")
    cfg.cfd.cells = 12
    cfg.grid.output_ny = 64
    window.state.replace(cfg, None)
    for action in (window.compare, window.compare_cfd, window.compare_legacy_cfd):
        action.setChecked(True)
    controller = window.comparison
    window.run()
    assert wait(app, lambda: not controller.busy and {"legacy", "cfd", "legacy_cfd"} <= set(controller.runs), 600)
    assert all(controller.runs[n].status == "success" for n in ("new", "legacy", "cfd", "legacy_cfd"))
    compared = {controller.metrics.item(r, 0).text() for r in range(controller.metrics.rowCount())}
    assert compared == {"New vs CFD", "Legacy vs CFD", "Legacy CFD vs CFD", "Legacy vs New", "Legacy CFD vs New"}
    assert [controller.performance.item(0, c).text() for c in range(4)] == ["success"] * 4
    assert "Legacy CFD vs CFD" in controller.summary.text()
    assert "Upstream tool" in [controller.legacy_cfd_pane.performance.topLevelItem(i).text(0)
                               for i in range(controller.legacy_cfd_pane.performance.topLevelItemCount())]
    differences = window.results.differences
    differences["cfd"].setChecked(True)
    assert window.results.reference[0] == "CFD reference"
    differences["legacy_cfd"].setChecked(True)  # one difference at a time
    assert window.results.reference[0].startswith("legacy CFD") and not differences["cfd"].isChecked()
    differences["legacy_cfd"].setChecked(False)
    assert window.results.reference is None
    # the new solver can be measured against either CFD, or against none
    assert [controller.reference.itemData(i) for i in range(controller.reference.count())] == ["cfd", "legacy_cfd"]
    controller.reference.setCurrentIndex(controller.reference.findData("legacy_cfd"))
    compared = [controller.metrics.item(r, 0).text() for r in range(controller.metrics.rowCount())]
    assert "New vs legacy CFD" in compared and "New vs CFD" not in compared and "Legacy CFD vs New" not in compared
    assert "Legacy CFD vs CFD" in compared  # the other solvers stay measured against the CFD reference
    assert "New solver vs legacy CFD" in controller.summary.text()
    controller.compare_new.setChecked(False)
    compared = [controller.metrics.item(r, 0).text() for r in range(controller.metrics.rowCount())]
    assert not any(c.startswith("New vs") for c in compared) and not controller.reference.isEnabled()
    assert "New solver vs" not in controller.summary.text()


def test_new_solver_measured_against_the_legacy_cfd_alone(app, window):
    controller = window.comparison
    assert not controller.compare_new.isVisibleTo(window)
    window.compare_legacy_cfd.setChecked(True)
    app.processEvents()
    assert controller.compare_new.isVisibleTo(window) and controller.compare_new.isChecked()
    assert [controller.reference.itemData(i) for i in range(controller.reference.count())] == ["legacy_cfd"]
    window.run()
    assert wait(app, lambda: not controller.busy and "legacy_cfd" in controller.runs, 300)
    assert controller.runs["legacy_cfd"].status == "success", controller.runs["legacy_cfd"].message
    compared = [controller.metrics.item(r, 0).text() for r in range(controller.metrics.rowCount())]
    assert compared == ["New vs legacy CFD"] * 3  # one row per stream
    assert [controller.metrics.item(r, 6).text() for r in range(3)] == ["within target"] * 3
    assert "New solver vs legacy CFD" in controller.summary.text()
    assert "no mesh-uncertainty estimate" in controller.summary.text()
    # the legacy CFD's time includes the COMSOL simulations behind its maps (none in a straight channel)
    column = [controller.performance.horizontalHeaderItem(c).text() for c in range(4)].index("Legacy CFD")
    assert "no COMSOL map used" in controller.performance.item(1, column).text()
    tree = controller.legacy_cfd_pane.performance
    assert "Time including COMSOL (estimated)" in [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    window.set_accuracy("medium")  # the new solver's accuracy is not an input of the legacy CFD
    app.processEvents()
    assert window.results.stale and not controller._legacy_cfd_stale()
