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


def test_simplified_menu_bar_and_one_click_run_in_toolbar(app, window):
    from PySide6.QtTest import QTest

    from crossflow_simulator.gui.qt import QtCore, QtWidgets

    bar = window.menuBar()
    entries = {a.text().replace("&", ""): a for a in bar.actions()}
    assert list(entries) == ["File", "Edit"]
    toolbar = window.findChild(QtWidgets.QToolBar)
    assert {window.run_action, window.cancel_action, window.live, window.compare,
            window.compare_cfd, window.compare_legacy_cfd} <= set(toolbar.actions())
    run_button = toolbar.widgetForAction(window.run_action)
    QTest.mouseClick(run_button, QtCore.Qt.MouseButton.LeftButton)
    assert wait(app, lambda: window.last_result is not None, 120)
    assert window.last_result.ok and window.last_result.provenance["fidelity"] == "preview"


def test_accuracy_low_medium_high_chooses_the_new_solvers_flow_grid(app, window):
    combo = window.accuracy
    assert [combo.itemText(i) for i in range(combo.count())] == ["Low", "Medium", "High"]
    assert combo.currentText() == "Low" and window.accuracy_actions["preview"].isChecked()
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
    assert window.last_result.message in window.results.status.text()
    window.state.undo()
    app.processEvents()
    assert combo.currentText() == "Medium" and window.results.stale


def test_comparison_panes_export_their_own_results_and_keep_run_text(app, window, tmp_path, monkeypatch):
    import json
    from dataclasses import replace

    from crossflow_simulator.comparison import BackendRun
    from crossflow_simulator.gui.qt import NavigationToolbar2QT, QtWidgets
    from crossflow_simulator.results import SimulationResult
    from crossflow_simulator.simulator import simulate

    result = simulate(window.state.config)
    assert result.ok
    panes = (("legacy", window.comparison.pane), ("cfd", window.comparison.cfd_pane),
             ("legacy_cfd", window.comparison.legacy_cfd_pane))
    for backend, pane in panes:
        assert not pane.findChildren(NavigationToolbar2QT)
        assert not pane.export_button.isEnabled()
        native = replace(result, backend=backend, message=f"{backend} run complete")
        run = BackendRun(backend, "success", native.message, result=native)
        pane.show_run(run, "um", "material", None)
        status = pane.status.text()
        assert native.message in status
        assert not pane.findChildren(QtWidgets.QTreeWidget)
        assert pane.export_button.isEnabled()
        output = tmp_path / backend
        output.mkdir()
        monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *_, p=output: str(p))
        pane.export_button.click()
        exported = SimulationResult.load(output / "result.npz")
        assert exported.backend == backend and exported.message == native.message
        np.testing.assert_array_equal(exported.final.codes, native.final.codes)
        assert ProjectConfig.load(output / "config.json") == window.state.config
        assert (output / "sections.png").stat().st_size > 0
        assert json.loads((output / "diagnostics.json").read_text())["message"] == native.message
        assert pane.status.text() == status
        pane.show_run(BackendRun(backend, "unsupported", "Unavailable", unsupported=["Unavailable"]),
                      "um", "material", None)
        assert not pane.export_button.isEnabled()


@pytest.mark.parametrize("enabled", [(True, False, False), (False, True, False),
                                    (True, True, False), (True, True, True)])
def test_visible_solver_images_have_equal_size(app, window, enabled):
    from dataclasses import replace

    from crossflow_simulator.comparison import BackendRun
    from crossflow_simulator.gui.qt import QtCore, QtWidgets
    from crossflow_simulator.simulator import simulate

    result = simulate(window.state.config)
    window.show_result(result)
    for action, on in zip((window.compare, window.compare_cfd, window.compare_legacy_cfd), enabled):
        action.setChecked(on)
    for backend, pane in window.comparison.panes.items():
        if window.comparison.enabled[backend]:
            native = replace(result, backend=backend)
            pane.show_run(BackendRun(backend, "success", native.message, result=native), "um", "material", None)
    for size in ((1600, 1000), (1850, 1100)):
        window.resize(*size)
        for _ in range(4):
            app.processEvents()
        window.comparison._size_plots()
        canvases = [c for c in window.comparison.canvases if c.isVisibleTo(window)]
        bounds = []
        baselines = []
        for canvas in canvases:
            canvas.draw()
            assert len(canvas.axes.images) == 1
            bounds.append(canvas.axes.get_window_extent().size)
            scale = canvas.figure.bbox.height / canvas.height()
            baseline = canvas.axes.transData.transform((0, 0))[1] / scale
            baselines.append(canvas.mapToGlobal(QtCore.QPoint(0, 0)).y() + canvas.height() - baseline)
        assert len(bounds) == 1 + sum(enabled)
        for bound in bounds[1:]:
            np.testing.assert_allclose(bound, bounds[0], atol=0.5, rtol=0)
        np.testing.assert_allclose(baselines, baselines[0], atol=0.5, rtol=0)
        assert not window.results.findChildren(QtWidgets.QTreeWidget)
        assert window.comparison.runtime_chart.isVisibleTo(window)
        assert not hasattr(window.comparison, "summary")


def test_new_solver_controls_status_and_locked_panes(app, window):
    from crossflow_simulator.gui.qt import QtWidgets
    from crossflow_simulator.simulator import simulate

    window.show_result(simulate(window.state.config))
    app.processEvents()
    view = window.results
    assert "#1B5E20" in view.status.styleSheet()
    assert not view.findChildren(QtWidgets.QToolButton)
    assert not view.findChildren(QtWidgets.QSlider)
    assert view.canvas.axes.get_title() == view.result.final.name
    assert {box.text() for box in view.findChildren(QtWidgets.QCheckBox)} == {
        "Difference to CFD", "Difference to legacy CFD"}
    boxes = list(view.differences.values())
    assert boxes[0].y() == boxes[1].y()
    assert all(box.y() + box.height() <= view.canvas.y() for box in boxes)
    view.mark_stale(True)
    assert "#B00020" in view.status.styleSheet()
    view.mark_stale(False)
    assert "#1B5E20" in view.status.styleSheet()
    assert window.minimumSize() != window.maximumSize()
    for splitter in window.findChildren(QtWidgets.QSplitter):
        for index in range(1, splitter.count()):
            assert not splitter.handle(index).isEnabled()


def test_cached_cfd_performance_shows_computation_timings(app, window, monkeypatch):
    from crossflow_simulator.gui.comparison_view import _performance_values
    from crossflow_simulator.simulator import simulate

    result = simulate(window.state.config)
    result.backend = "cfd"
    result.diagnostics.update(computed_seconds=87.5, computed_backend_seconds=88.25,
                              cache="disk cache (computed in 87.5 s)")
    monkeypatch.setattr("crossflow_simulator.gui.comparison_view.cfd.cached_result", lambda cfg: result)
    assert window.comparison.load_cached_cfd()
    run = window.comparison.runs["cfd"]
    # Loading the same result again must not replace its compute time with lookup time.
    run.solve_seconds = 0.002
    run.end_to_end_seconds = 0.005
    values = _performance_values("cfd", run, run)
    assert values[1:3] == ["87.50 s", "88.25 s"]


def test_benchmark_keeps_uncached_result_in_runtime_display(app, window, examples_dir, tmp_path):
    from crossflow_simulator.comparison import ComparisonService
    from crossflow_simulator.gui.comparison_view import _fmt_seconds

    cfg = ProjectConfig.load(examples_dir / "serial_three_inlets.json")
    window.state.replace(cfg.copy())
    controller = window.comparison
    published = []
    controller.new_run.connect(published.append)
    service = ComparisonService(tmp_path / "benchmark", on_update=controller._update)
    try:
        comparison = service.benchmark(cfg, trials=3, include_cold=False)
    finally:
        service.close()
    controller._done(comparison)

    measured = comparison.runs["new"]
    assert measured.status == "success", measured.message
    assert comparison.benchmark["statistics"]["new"]["cached"]["n"] == 3
    assert len(published) == 3
    assert window.last_result is measured.result
    assert controller._new_run() is measured
    mean = np.mean(comparison.benchmark["samples"]["new"]["warm_uncached"])
    assert controller.performance.item(1, 1).text() == _fmt_seconds(mean)
    assert next(row[2] for row in controller.runtime_chart.records if row[0] == "new") == pytest.approx(mean)
    assert "uncached mean, n=3" in controller.performance.horizontalHeaderItem(1).text()


@pytest.mark.parametrize("failed_middle", [False, True])
def test_benchmark_tables_summarize_all_uncached_trials(app, window, failed_middle):
    import json
    from dataclasses import replace

    from crossflow_simulator.comparison import BackendRun, accuracy_against, build_comparison, describe
    from crossflow_simulator.simulator import simulate

    controller = window.comparison
    window.compare.setChecked(True)
    window.compare_cfd.setChecked(True)
    base = simulate(window.state.config)
    assert base.ok
    runs = {"new": [], "legacy": []}
    for name, times in (("new", (0.2, 0.4, 0.9)), ("legacy", (1.0, 2.0, 6.0))):
        for index, seconds in enumerate(times):
            codes = base.final.codes.copy()
            # Distinct trial results make averaging metrics distinguishable from using the last result.
            codes[:, : (index + 1) * (2 if name == "new" else 3)] = 2
            result = replace(base, backend=name,
                             sections=[*base.sections[:-1], replace(base.final, codes=codes)])
            runs[name].append(BackendRun(name, "success", solve_seconds=seconds,
                                        end_to_end_seconds=seconds + (index + 1) * 0.1,
                                        cache_state="cleared", result=result))
    if failed_middle:
        runs["legacy"][1] = BackendRun("legacy", "failed", solve_seconds=0.001)
    comparison = build_comparison(window.state.config, {name: trials[-1] for name, trials in runs.items()})
    comparison.benchmark_runs = runs
    comparison.benchmark = {
        "protocol": {"trials": 3}, "cold": {},
        "statistics": {name: {"warm_uncached": describe([run.solve_seconds for run in trials
                                                        if run.status == "success"])}
                       for name, trials in runs.items()},
        "samples": {"new": {"cached": [0.001, 0.002, 0.003]}},
    }
    controller.runs["cfd"] = BackendRun("cfd", "success", solve_seconds=10.0,
                                        result=replace(base, backend="cfd"))
    window.show_result(runs["new"][-1].result)
    controller._done(comparison)
    assert controller.performance.item(1, 1).text() == "500 ms"
    assert controller.performance.item(2, 1).text() == "700 ms"
    assert "median 400 ms" in controller.performance.item(1, 1).toolTip()
    expected_legacy_time = 3.5 if failed_middle else 3.0
    assert controller.performance.item(1, 0).text() == f"{expected_legacy_time:.2f} s"
    assert next(row[2] for row in controller.runtime_chart.records if row[0] == "new") == pytest.approx(0.5)
    for label, reference_name, result_name in (("New vs CFD", "cfd", "new"),
                                                ("Legacy vs CFD", "cfd", "legacy"),
                                                ("Legacy vs New", "new", "legacy")):
        pairs = ([(controller.runs["cfd"], trial) for trial in runs[result_name]] if reference_name == "cfd"
                 else list(zip(runs[reference_name], runs[result_name])))
        entries = [accuracy_against(ref.result, run.result) for ref, run in pairs
                   if ref.status == run.status == "success"]
        expected_count = len(entries)
        rows = [r for r in range(controller.metrics.rowCount())
                if controller.metrics.item(r, 0).text() == f"{label} (uncached mean, n={expected_count})"]
        assert len(rows) == len(entries[0]["per_stream"])
        for row in rows:
            code = int(controller.metrics.item(row, 1).text())
            expected = np.mean([entry["per_stream"][code]["iou"] for entry in entries])
            assert controller.metrics.item(row, 2).text() == f"{expected:.3f}"
            assert "Range:" in controller.metrics.item(row, 2).toolTip()
    exported = controller.current_comparison()
    summary = json.loads(json.dumps(exported.summary()))
    assert summary["benchmark"]["agreement"]["new"]["benchmark"]["n"] == 3
    assert len(summary["uncached_benchmark_runs"]["new"]) == 3
    assert summary["speedup"]["scope"] == "median warm-worker uncached solve time"
    # A later manual result at the same inputs must return both tables to single-run mode.
    window.show_result(replace(base, timings={"total": 0.123}))
    assert controller.performance.horizontalHeaderItem(1).text() == "New"
    assert controller.performance.item(1, 1).text() == "123 ms"
    assert all("uncached mean" not in controller.metrics.item(r, 0).text()
               for r in range(controller.metrics.rowCount()))


def test_runtime_chart_tracks_selected_solvers_and_original_cfd_compute_time(app, window):
    from dataclasses import replace

    from crossflow_simulator.comparison import BackendRun
    from crossflow_simulator.simulator import simulate

    controller = window.comparison
    chart = controller.runtime_chart
    assert chart.records == [("new", "New solver", None, "not run")]
    result = simulate(window.state.config)
    window.show_result(result)
    assert chart.records == [("new", "New solver", result.timings["total"], "success")]
    window.compare.setChecked(True)
    window.compare_cfd.setChecked(True)
    window.compare_legacy_cfd.setChecked(True)
    native = replace(result, backend="cfd", diagnostics={**result.diagnostics, "computed_seconds": 87.5})
    controller.runs.update({
        "legacy": BackendRun("legacy", "success", "Done", solve_seconds=2.5, result=result),
        "cfd": BackendRun("cfd", "success", "Cached", solve_seconds=0.002, result=native),
        "legacy_cfd": BackendRun("legacy_cfd", "success", "Precomputed", solve_seconds=0.02, result=result),
    })
    controller.fill_performance()
    rows = {row[0]: row for row in chart.records}
    assert set(rows) == {"new", "legacy", "cfd"}
    assert rows["new"][2] == result.timings["total"]
    assert rows["legacy"][2] == 2.5
    assert rows["cfd"][2] == 87.5
    assert controller.performance.item(1, 3).text() == "\u2014"
    np.testing.assert_allclose([bar.get_width() for bar in chart.axes.patches],
                               [result.timings["total"], 2.5, 87.5])
    window.compare.setChecked(False)
    assert "legacy" not in {row[0] for row in chart.records}
    window.state.apply(lambda cfg: setattr(cfg.inlets[0], "flow_rate", cfg.inlets[0].flow_rate * 2))
    assert next(row for row in chart.records if row[0] == "new")[3] == "stale"


def test_new_export_button_is_bottom_left_and_still_emits(app, window):
    from PySide6.QtTest import QSignalSpy

    view = window.results
    app.processEvents()
    button = view.export_button
    assert button.x() == view.canvas.x()
    assert button.y() >= view.canvas.y() + view.canvas.height()
    assert button.width() < view.width() / 2
    spy = QSignalSpy(view.export_requested)
    # Avoid opening a native export dialog through the main window connection.
    view.export_requested.disconnect(window.export_result)
    button.click()
    assert spy.count() == 1


def test_maximizing_and_fullscreen_preserve_locked_panes(app, window):
    from PySide6.QtTest import QTest

    from crossflow_simulator.gui.qt import QtCore

    window.showMaximized()
    app.processEvents()
    assert window.isMaximized()
    QTest.keyClick(window, QtCore.Qt.Key_F11)
    app.processEvents()
    assert window.isFullScreen()
    QTest.keyClick(window, QtCore.Qt.Key_F11)
    app.processEvents()
    assert window.isMaximized() and not window.isFullScreen()
    window.showNormal()
    window.toggle_fullscreen()
    assert window.isFullScreen()
    window.toggle_fullscreen()
    assert not window.isFullScreen() and not window.isMaximized()


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
    for widget in (window.setup, window.geometry, window.view3d, window.results, controller.runtime_chart):
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
    assert all(s > 0 for s in sizes[:3])
    window.compare_legacy_cfd.setChecked(True)  # all four results side by side
    app.processEvents()
    assert controller.legacy_cfd_pane.isVisibleTo(window) and "Ready" in controller.legacy_cfd_pane.status.text()
    sizes = window.result_split.sizes()
    # The fixed window accommodates each pane's controls; image size/alignment
    # is checked independently of the containing widgets' minimum widths.
    assert all(s > 0 for s in sizes)
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
    # Removing the diagnostics table does not remove the data from the result.
    filled = int(window.last_result.final.filled.sum())
    assert filled > 0
    assert window.last_result.final.diagnostics["trajectory_status"]["filled_from_neighbours"] == filled
    assert np.all((after[:, 1] >= -1e-9) & (after[:, 1] <= cfg.channel.width + 1e-9))
    assert np.all((after[:, 2] >= -1e-9) & (after[:, 2] <= cfg.channel.height + 1e-9))
    assert view.station is None  # the 3D view defaults to the final calculated section
    window.state.apply(lambda c: setattr(c.pillars[0], "x", c.pillars[0].x + 50e-6))
    app.processEvents()
    assert not view.flow_shown and len(view.particle_positions()) == 0
    assert "Inputs changed" in view.message


@pytest.mark.parametrize("proportions", ["true proportions", "compressed length"])
def test_3d_preview_keeps_channel_corners_and_axis_labels_inside_canvas(app, examples_dir, proportions):
    from crossflow_simulator.gui.state import ProjectState
    from crossflow_simulator.gui.view3d import FlowView3D

    view = FlowView3D(ProjectState(ProjectConfig.load(examples_dir / "parallel_unequal.json")))
    view.proportions.setCurrentText(proportions)
    view.show()
    try:
        for size in ((1168, 350), (900, 280), (1500, 420)):
            view.resize(*size)
            app.processEvents()
            for elevation, azimuth in ((24, -58), (32, -35)):
                view.ax.view_init(elev=elevation, azim=azimuth)
                for _ in range(8):
                    view.canvas.draw()
                    app.processEvents()
                bounds = view._scene_bounds(view.canvas.get_renderer())
                width, height = view.figure.bbox.size
                assert bounds.x0 >= 15 and bounds.x1 <= width - 15
                assert bounds.y0 >= 29 and bounds.y1 <= height - 11
                assert not any(collection.get_clip_on() for collection in view.ax.collections)
    finally:
        view.timer.stop()
        view.close()


def test_3d_station_slider_drags_and_updates_sections_without_recalculating(app, window, examples_dir):
    from PySide6.QtTest import QTest

    from crossflow_simulator.gui.qt import QtCore, QtWidgets
    from crossflow_simulator.simulator import simulate

    view = window.view3d
    assert not view.station_slider.isEnabled()
    window.state.replace(ProjectConfig.load(examples_dir / "square_library_pillars.json"), None)
    result = simulate(window.state.config)
    assert result.ok and len(result.sections) == 3
    window.show_result(result)
    app.processEvents()
    slider = view.station_slider
    assert slider.isEnabled() and slider.value() == 2
    assert not window.results.findChildren(QtWidgets.QSlider)
    view.canvas.draw()
    final_collections = len(view.ax.collections)
    option = QtWidgets.QStyleOptionSlider()
    slider.initStyleOption(option)
    handle = slider.style().subControlRect(QtWidgets.QStyle.CC_Slider, option,
                                          QtWidgets.QStyle.SC_SliderHandle, slider)
    QTest.mousePress(slider, QtCore.Qt.LeftButton, pos=handle.center())
    QTest.mouseMove(slider, QtCore.QPoint(2, slider.height() // 2))
    app.processEvents()
    # Tracking redraws the preview while the mouse is still held down.
    assert slider.value() == view.station == 0
    assert len(view.ax.collections) == final_collections - 1
    QTest.mouseRelease(slider, QtCore.Qt.LeftButton, pos=QtCore.QPoint(2, slider.height() // 2))
    app.processEvents()
    assert slider.value() == view.station == window.comparison.station == 0
    assert result.sections[0].name in view.station_label.text()
    assert window.results.canvas.axes.get_title() == result.sections[0].name
    assert len(view.ax.collections) == final_collections - 1
    QTest.keyClick(slider, QtCore.Qt.Key_Right)
    app.processEvents()
    assert slider.value() == view.station == 1
    assert window.results.canvas.axes.get_title() == result.sections[1].name
    assert result.sections[1].name in view.station_label.text()
    assert window.last_result is result
    window.show_result(result)
    assert slider.value() == 2 and window.results.station is None and window.comparison.station is None
    view.mark_stale(True)
    assert not slider.isEnabled()
    view.mark_stale(False)
    assert slider.isEnabled()


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
    assert controller.performance.item(0, 2).text() == "success"
    assert "faster" in controller.performance.item(6, 1).text()  # speed of the new solver vs the CFD
    assert run.result.diagnostics["cfd"]["mesh_convergence"] is not None
    # Both panes show the final section; the overlay compares their final material labels.
    app.processEvents()
    assert run.result.final.name in controller.cfd_pane.canvas.axes.get_title()
    window.results.differences["cfd"].setChecked(True)
    assert window.results.reference[0] == "CFD reference"
    assert window.results.reference_codes(len(window.last_result.sections) - 1) is not None
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
    controller.compare_new.setChecked(False)
    compared = [controller.metrics.item(r, 0).text() for r in range(controller.metrics.rowCount())]
    assert not any(c.startswith("New vs") for c in compared) and not controller.reference.isEnabled()


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
    # Applying stored maps does not measure the original CFD computation time.
    column = [controller.performance.horizontalHeaderItem(c).text() for c in range(4)].index("Legacy CFD")
    assert controller.performance.item(1, column).text() == "\u2014"
    window.set_accuracy("medium")  # the new solver's accuracy is not an input of the legacy CFD
    app.processEvents()
    assert window.results.stale and not controller._legacy_cfd_stale()
