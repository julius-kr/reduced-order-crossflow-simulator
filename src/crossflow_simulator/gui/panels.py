"""Project setup panels: channel, materials, inlets, pattern, pillars, model settings, issues."""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..config import FluidProperties, InletSpec, PillarSpec, ProjectConfig
from ..materials import material_colors, palette_color
from ..units import KEY_SUFFIX, UNITS, from_si, to_si
from .qt import QtCore, QtGui, QtWidgets, Signal
from .state import ProjectState


@dataclass
class Column:
    attr: str
    header: str
    kind: str = "float"  # text | float | int | bool | choice | vertices
    quantity: str | None = None  # unit kind for conversion
    choices: tuple = ()
    width: int = 90


def display_unit(cfg: ProjectConfig, quantity: str) -> str:
    return {
        "length": cfg.display.length_unit,
        "flow_rate": cfg.display.flow_unit,
        "viscosity": cfg.display.viscosity_unit,
        "pressure": cfg.display.pressure_unit,
        "resistance": "mbar/(uL/min)",
        "density": "kg/m3",
        "diffusivity": "um2/s",
        "temperature": "K",
        "angle": "deg",
    }[quantity]


def _field_kind(item: Any, attr: str) -> str | None:
    for f in dataclasses.fields(item):
        if f.name == attr:
            return f.metadata.get("kind")
    return None


class DataclassTable(QtWidgets.QTableWidget):
    """Editable table over a list of dataclasses with unit conversion and inline issue highlighting."""

    edited = Signal(int, str, object)  # row, attribute, SI value

    def __init__(self, columns: list[Column], state: ProjectState, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.project = state  # not ``self.state``: that would hide QAbstractItemView.state()
        self._filling = False
        self.setColumnCount(len(columns))
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.cellChanged.connect(self._cell_changed)
        for i, col in enumerate(columns):
            self.setColumnWidth(i, col.width)

    def headers(self, cfg: ProjectConfig) -> None:
        labels = []
        for col in self.columns:
            unit = f" [{display_unit(cfg, col.quantity)}]" if col.quantity else ""
            labels.append(col.header + unit)
        self.setHorizontalHeaderLabels(labels)

    def populate(self, items: list, cfg: ProjectConfig, issues: dict[tuple[int, str], str]) -> None:
        """Show ``items``, updating the existing cells in place.

        This runs while an edit is being committed (the commit changes the project, which refreshes
        the table). Replacing cells or removing their widgets then would destroy the closing editor
        and leave the view stuck in its editing state, so later double-clicks open no editor.
        """
        self._filling = True
        try:
            self.headers(cfg)
            self.setRowCount(len(items))
            for row, item in enumerate(items):
                for c, col in enumerate(self.columns):
                    value = getattr(item, col.attr)
                    if col.kind == "choice":
                        self._show_choice(row, c, col, value, cfg)
                        continue
                    cell = self.item(row, c)
                    if cell is None:
                        cell = QtWidgets.QTableWidgetItem()
                        if col.kind == "bool":
                            cell.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled
                                          | QtCore.Qt.ItemIsSelectable)
                        self.setItem(row, c, cell)
                    if col.kind == "bool":
                        cell.setCheckState(QtCore.Qt.Checked if value else QtCore.Qt.Unchecked)
                    else:
                        cell.setText(self._format(value, col, cfg))
                    message = issues.get((row, self._json_key(item, col.attr)))
                    cell.setBackground(QtGui.QColor("#FFD6D6") if message else QtGui.QBrush())
                    cell.setToolTip(message or "")
        finally:
            self._filling = False

    def _show_choice(self, row: int, c: int, col: Column, value: Any, cfg: ProjectConfig) -> None:
        choices = [str(x) for x in (col.choices if col.choices else tuple(m.id for m in cfg.materials))]
        combo = self.cellWidget(row, c)
        if not isinstance(combo, QtWidgets.QComboBox):
            combo = QtWidgets.QComboBox()
            combo.currentTextChanged.connect(
                lambda text, r=row, a=col.attr: None if self._filling else self.edited.emit(r, a, text))
            self.setItem(row, c, QtWidgets.QTableWidgetItem(""))
            self.setCellWidget(row, c, combo)
        if [combo.itemText(i) for i in range(combo.count())] != choices:
            combo.clear()
            combo.addItems(choices)
        combo.setCurrentText(str(value))

    @staticmethod
    def _json_key(item: Any, attr: str) -> str:
        kind = _field_kind(item, attr)
        return f"{attr}_{KEY_SUFFIX[kind]}" if kind else attr

    def _format(self, value: Any, col: Column, cfg: ProjectConfig) -> str:
        if value is None:
            return ""
        if col.kind == "vertices":
            unit = display_unit(cfg, "length")
            return "; ".join(f"{from_si(x, unit, 'length'):.4g}, {from_si(y, unit, 'length'):.4g}" for x, y in value)
        if col.quantity:
            return f"{from_si(float(value), display_unit(cfg, col.quantity), col.quantity):.6g}"
        return str(value)

    def _parse(self, text: str, col: Column, cfg: ProjectConfig) -> Any:
        text = text.strip()
        if col.kind == "text":
            return text or None if col.attr == "group" else text
        if col.kind == "int":
            return int(float(text))
        if col.kind == "vertices":
            unit = display_unit(cfg, "length")
            pairs = [p for p in re.split(r"[;\n]", text) if p.strip()]
            return [tuple(to_si(float(v), unit, "length") for v in pair.split(",")) for pair in pairs]
        if text == "":
            return None
        value = float(text)
        return to_si(value, display_unit(cfg, col.quantity), col.quantity) if col.quantity else value

    def _cell_changed(self, row: int, column: int) -> None:
        if self._filling:
            return
        col = self.columns[column]
        cell = self.item(row, column)
        try:
            if col.kind == "bool":
                value = cell.checkState() == QtCore.Qt.Checked
            else:
                value = self._parse(cell.text(), col, self.project.config)
        except (ValueError, TypeError):
            cell.setBackground(QtGui.QColor("#FFD6D6"))
            cell.setToolTip("Could not parse this value")
            return
        self.edited.emit(row, col.attr, value)


MATERIAL_COLUMNS = [
    Column("id", "ID", "text", width=60), Column("name", "Name", "text", width=110),
    Column("viscosity", "Viscosity", quantity="viscosity"), Column("density", "Density", quantity="density"),
    Column("diffusivity", "Diffusivity", quantity="diffusivity"),
]
INLET_COLUMNS = [
    Column("id", "ID", "text", width=60), Column("material", "Material", "choice", width=80),
    Column("placement", "Placement", "choice", choices=("pattern", "port"), width=80),
    Column("flow_rate", "Flow rate", quantity="flow_rate"), Column("pressure", "Pressure", quantity="pressure"),
    Column("tubing_resistance", "Tubing R", quantity="resistance"),
    Column("pattern_row", "Row", "int", width=45), Column("pattern_col", "Col", "int", width=45),
    Column("station", "Station x", quantity="length"),
    Column("wall", "Wall", "choice", choices=("top", "bottom", "left", "right"), width=75),
    Column("position", "Position", quantity="length"), Column("diameter", "Diameter", quantity="length"),
    Column("enabled", "On", "bool", width=40),
]
PILLAR_COLUMNS = [
    Column("id", "ID", "text", width=60), Column("vertices", "Footprint vertices (local x, y)", "vertices", width=260),
    Column("height", "Height", quantity="length"), Column("x", "x", quantity="length"),
    Column("y", "y", quantity="length"), Column("rotation", "Rotation", quantity="angle", width=70),
    Column("mirror_y", "Mirror", "bool", width=50),
    Column("mount", "Mount", "choice", choices=("bottom", "top"), width=75),
    Column("group", "Group", "text", width=70), Column("enabled", "On", "bool", width=40),
]
# Footprints offered by "Add pillar", in units of the channel width W (local x streamwise, y spanwise).
PILLAR_SHAPES = {
    "Right triangle": [(0.0, 0.0), (0.5, 0.0), (0.0, 0.4)],
    # parallel sides 0.6 W and 0.3 W along the flow; the legs slope differently (1:4 upstream, 1:2 downstream)
    "Non-isosceles trapezoid": [(0.0, 0.0), (0.6, 0.0), (0.4, 0.4), (0.1, 0.4)],
}


class SetupPanel(QtWidgets.QWidget):
    """All project inputs, bound to :class:`ProjectState`."""

    selection_changed = Signal(str, int)  # "pillar" | "inlet", row

    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.toolbox = QtWidgets.QToolBox()
        layout.addWidget(self.toolbox)

        # Channel and operating conditions
        channel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(channel)
        self.width_edit, self.height_edit, self.length_edit = (QtWidgets.QLineEdit() for _ in range(3))
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(["um", "mm"])
        self.flow_unit_combo = QtWidgets.QComboBox()
        self.flow_unit_combo.addItems(["uL/min", "uL/h", "mL/h", "nL/min"])
        self.drive_combo = QtWidgets.QComboBox()
        self.drive_combo.addItems(["flow_rate", "pressure"])
        self.outlet_edit = QtWidgets.QLineEdit()
        self.color_by = QtWidgets.QComboBox()
        self.color_by.addItems(["material", "inlet"])
        form.addRow("Width W", self.width_edit)
        form.addRow("Height H", self.height_edit)
        form.addRow("Length (optional)", self.length_edit)
        form.addRow("Length unit", self.unit_combo)
        form.addRow("Flow unit", self.flow_unit_combo)
        form.addRow("Driving condition", self.drive_combo)
        form.addRow("Outlet pressure", self.outlet_edit)
        form.addRow("Colour by", self.color_by)
        self.toolbox.addItem(channel, "Channel and driving conditions")
        for widget in (self.width_edit, self.height_edit, self.length_edit, self.outlet_edit):
            widget.editingFinished.connect(self._channel_edited)
        self.unit_combo.currentTextChanged.connect(lambda u: self._set(lambda c: setattr(c.display, "length_unit", u)))
        self.flow_unit_combo.currentTextChanged.connect(lambda u: self._set(lambda c: setattr(c.display, "flow_unit", u)))
        self.drive_combo.currentTextChanged.connect(lambda d: self._set(lambda c: setattr(c.operating, "drive", d)))
        self.color_by.currentTextChanged.connect(lambda v: self._set(lambda c: setattr(c.display, "color_by", v)))

        # Materials
        self.materials = DataclassTable(MATERIAL_COLUMNS, state)
        self.materials.edited.connect(lambda r, a, v: self._edit_list("materials", r, a, v))
        self.toolbox.addItem(self._with_buttons(self.materials, [
            ("Add material", self._add_material), ("Remove", lambda: self._remove("materials", self.materials)),
            ("Colour...", self._pick_color),
        ]), "Materials")

        # Inlets and pattern
        inlet_box = QtWidgets.QWidget()
        inlet_layout = QtWidgets.QVBoxLayout(inlet_box)
        pattern_row = QtWidgets.QHBoxLayout()
        self.layout_combo = QtWidgets.QComboBox()
        self.layout_combo.addItems(["single", "layers_y", "layers_z", "grid"])
        self.rows_spin, self.cols_spin = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for spin in (self.rows_spin, self.cols_spin):
            spin.setRange(1, 8)
        self.partition_combo = QtWidgets.QComboBox()
        self.partition_combo.addItems(["auto", "aligned", "staggered_rows", "staggered_columns"])
        for label, widget in (("Inlet pattern", self.layout_combo), ("rows", self.rows_spin),
                              ("cols", self.cols_spin), ("partition", self.partition_combo)):
            pattern_row.addWidget(QtWidgets.QLabel(label))
            pattern_row.addWidget(widget)
        pattern_row.addStretch(1)
        inlet_layout.addLayout(pattern_row)
        self.layout_combo.currentTextChanged.connect(lambda v: self._set(lambda c: setattr(c.pattern, "layout", v)))
        self.rows_spin.valueChanged.connect(lambda v: self._set(lambda c: setattr(c.pattern, "rows", v)))
        self.cols_spin.valueChanged.connect(lambda v: self._set(lambda c: setattr(c.pattern, "cols", v)))
        self.partition_combo.currentTextChanged.connect(lambda v: self._set(lambda c: setattr(c.pattern, "partition", v)))
        self.inlets = DataclassTable(INLET_COLUMNS, state)
        self.inlets.edited.connect(lambda r, a, v: self._edit_list("inlets", r, a, v))
        self.inlets.itemSelectionChanged.connect(lambda: self._emit_selection("inlet", self.inlets))
        inlet_layout.addWidget(self._with_buttons(self.inlets, [
            ("Add pattern inlet", lambda: self._add_inlet("pattern")), ("Add port", lambda: self._add_inlet("port")),
            ("Remove", lambda: self._remove("inlets", self.inlets)), ("Move up", lambda: self._move_inlet(-1)),
            ("Move down", lambda: self._move_inlet(1)),
        ]))
        self.flow_summary = QtWidgets.QLabel()
        inlet_layout.addWidget(self.flow_summary)
        self.toolbox.addItem(inlet_box, "Inlets (explicit order = stable IDs)")

        # Pillars
        self.pillars = DataclassTable(PILLAR_COLUMNS, state)
        self.pillars.edited.connect(lambda r, a, v: self._edit_list("pillars", r, a, v))
        self.pillars.itemSelectionChanged.connect(lambda: self._emit_selection("pillar", self.pillars))
        self.toolbox.addItem(self._with_buttons(self.pillars, [
            ("Add pillar", [(shape, lambda s=shape: self._add_pillar(s)) for shape in PILLAR_SHAPES]),
            ("Duplicate", self._duplicate_pillar),
            ("Rotate 90°", lambda: self._transform_pillar(rotate=90.0)),
            ("Reflect", lambda: self._transform_pillar(reflect=True)),
            ("Remove", lambda: self._remove("pillars", self.pillars)),
        ]), "Pillars (partial or full height)")

        # Model settings
        model = QtWidgets.QWidget()
        mform = QtWidgets.QFormLayout(model)
        self.coupling = QtWidgets.QComboBox()
        self.coupling.addItems(["auto", "off", "iterate"])
        self.transport = QtWidgets.QComboBox()
        self.transport.addItems(["categorical", "conservative"])
        self.output_ny = QtWidgets.QSpinBox()
        self.output_ny.setRange(16, 600)
        mform.addRow("Viscosity coupling", self.coupling)
        mform.addRow("Transport", self.transport)
        mform.addRow("Output cells across W", self.output_ny)
        self.coupling.currentTextChanged.connect(lambda v: self._set(lambda c: setattr(c.model, "viscosity_coupling", v)))
        self.transport.currentTextChanged.connect(lambda v: self._set(lambda c: setattr(c.model, "transport", v)))
        self.output_ny.editingFinished.connect(
            lambda: self._set(lambda c: setattr(c.grid, "output_ny", self.output_ny.value())))
        self.toolbox.addItem(model, "Model and numerics (advanced)")

        self.issues = QtWidgets.QListWidget()
        self.issues.setMaximumHeight(110)
        layout.addWidget(QtWidgets.QLabel("Validation"))
        layout.addWidget(self.issues)
        state.changed.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------ helpers
    def _with_buttons(self, table: QtWidgets.QWidget,
                      buttons: list[tuple[str, Callable | list[tuple[str, Callable]]]]) -> QtWidgets.QWidget:
        """Buttons below a table; a button given a list of (label, callback) opens a menu of those choices."""
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(table)
        row = QtWidgets.QHBoxLayout()
        for label, callback in buttons:
            button = QtWidgets.QPushButton(label)
            if isinstance(callback, list):
                menu = QtWidgets.QMenu(button)
                for text, choice in callback:
                    menu.addAction(text).triggered.connect(lambda _=False, c=choice: c())
                button.setMenu(menu)
            else:
                button.clicked.connect(callback)
            row.addWidget(button)
        row.addStretch(1)
        v.addLayout(row)
        return box

    def _set(self, mutate: Callable[[ProjectConfig], None]) -> None:
        if not getattr(self, "_refreshing", False):
            self.state.apply(mutate)

    def _edit_list(self, name: str, row: int, attr: str, value: Any) -> None:
        self._set(lambda c: setattr(getattr(c, name)[row], attr, value))

    def _emit_selection(self, kind: str, table: QtWidgets.QTableWidget) -> None:
        rows = {i.row() for i in table.selectedIndexes()}
        if rows:
            self.selection_changed.emit(kind, min(rows))

    def selected_row(self, table: QtWidgets.QTableWidget) -> int | None:
        rows = {i.row() for i in table.selectedIndexes()}
        return min(rows) if rows else None

    def select(self, kind: str, row: int) -> None:
        table = self.pillars if kind == "pillar" else self.inlets
        table.selectRow(row)

    def _channel_edited(self) -> None:
        cfg = self.state.config
        unit = cfg.display.length_unit
        pressure_unit = cfg.display.pressure_unit
        try:
            width = to_si(float(self.width_edit.text()), unit, "length")
            height = to_si(float(self.height_edit.text()), unit, "length")
            length = to_si(float(self.length_edit.text()), unit, "length") if self.length_edit.text().strip() else None
            outlet = to_si(float(self.outlet_edit.text() or 0.0), pressure_unit, "pressure")
        except ValueError:
            return

        def mutate(c: ProjectConfig) -> None:
            c.channel.width, c.channel.height, c.channel.length = width, height, length
            c.operating.outlet_pressure = outlet

        self._set(mutate)

    def _add_material(self) -> None:
        def mutate(c: ProjectConfig) -> None:
            n = len(c.materials)
            ids = {m.id for m in c.materials}
            new_id = next(f"M{i}" for i in range(n + 1, n + 100) if f"M{i}" not in ids)
            c.materials.append(FluidProperties(id=new_id, viscosity=c.materials[-1].viscosity if c.materials else 1e-3,
                                               name=f"material {n + 1}"))
        self._set(mutate)

    def _pick_color(self) -> None:
        row = self.selected_row(self.materials)
        if row is None:
            return
        current = material_colors(self.state.config)[self.state.config.materials[row].id]
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, "Material colour")
        if color.isValid():
            self._set(lambda c: setattr(c.materials[row], "color", color.name().upper()))

    def _add_inlet(self, placement: str) -> None:
        def mutate(c: ProjectConfig) -> None:
            ids = {i.id for i in c.inlets}
            new_id = next(f"in{k}" for k in range(1, 200) if f"in{k}" not in ids)
            material = c.materials[min(len(c.inlets), len(c.materials) - 1)].id
            rate = 2e-9 / 60 if c.operating.drive == "flow_rate" else None
            spec = InletSpec(new_id, material, placement, flow_rate=rate)
            if c.operating.drive == "pressure":
                spec.pressure, spec.tubing_resistance = 3000.0, 6e13
            if placement == "port":
                spec.station = max([i.station for i in c.port_inlets()] + [0.0]) + 2 * c.channel.width
                spec.position = 0.5 * c.channel.width
                spec.diameter = 0.5 * c.channel.width
            else:
                spec.pattern_row = len(c.pattern_inlets()) if c.pattern.layout == "layers_z" else 0
                spec.pattern_col = len(c.pattern_inlets()) if c.pattern.layout == "layers_y" else 0
            c.inlets.append(spec)
        self._set(mutate)

    def _move_inlet(self, step: int) -> None:
        row = self.selected_row(self.inlets)
        if row is None:
            return
        target = row + step

        def mutate(c: ProjectConfig) -> None:
            if 0 <= target < len(c.inlets):
                c.inlets[row], c.inlets[target] = c.inlets[target], c.inlets[row]
        self._set(mutate)
        self.inlets.selectRow(max(0, min(target, self.inlets.rowCount() - 1)))

    def _add_pillar(self, shape: str = "Right triangle") -> None:
        def mutate(c: ProjectConfig) -> None:
            ids = {p.id for p in c.pillars}
            new_id = next(f"p{k}" for k in range(1, 200) if f"p{k}" not in ids)
            W, H = c.channel.width, c.channel.height
            x = max([float(p.footprint()[:, 0].max()) for p in c.pillars] + [0.0]) + 1.5 * W
            c.pillars.append(PillarSpec(new_id, [(u * W, v * W) for u, v in PILLAR_SHAPES[shape]], height=0.6 * H,
                                        x=x, y=0.13 * W))
        self._set(mutate)

    def _duplicate_pillar(self) -> None:
        row = self.selected_row(self.pillars)
        if row is None:
            return

        def mutate(c: ProjectConfig) -> None:
            source = c.pillars[row]
            ids = {p.id for p in c.pillars}
            copy = dataclasses.replace(source, id=next(f"{source.id}_{k}" for k in range(2, 200)
                                                        if f"{source.id}_{k}" not in ids))
            copy.x = source.x + 1.5 * c.channel.width
            c.pillars.append(copy)
        self._set(mutate)

    def _transform_pillar(self, rotate: float = 0.0, reflect: bool = False) -> None:
        row = self.selected_row(self.pillars)
        if row is None:
            return

        def mutate(c: ProjectConfig) -> None:
            p = c.pillars[row]
            p.rotation = (p.rotation + rotate) % 360.0
            if reflect:
                p.mirror_y = not p.mirror_y
        self._set(mutate)

    def _remove(self, name: str, table: QtWidgets.QTableWidget) -> None:
        row = self.selected_row(table)
        if row is not None:
            self._set(lambda c: getattr(c, name).pop(row))

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        self._refreshing = True
        try:
            cfg = self.state.config
            unit = cfg.display.length_unit
            issues: dict[str, dict[tuple[int, str], str]] = {"materials": {}, "inlets": {}, "pillars": {}}
            self.issues.clear()
            for issue in self.state.issues():
                item = QtWidgets.QListWidgetItem(issue)
                if issue.startswith("[error]"):
                    item.setForeground(QtGui.QColor("#B00020"))
                self.issues.addItem(item)
                match = re.match(r"\[\w+\] (materials|inlets|pillars)\[(\d+)\](?:\.(\w+))?: (.*)", issue)
                if match:
                    issues[match.group(1)][(int(match.group(2)), match.group(3) or "id")] = match.group(4)
            if not self.issues.count():
                self.issues.addItem("No issues.")
            for widget, value in ((self.width_edit, cfg.channel.width), (self.height_edit, cfg.channel.height),
                                  (self.length_edit, cfg.channel.length)):
                widget.setText("" if value is None else f"{from_si(value, unit, 'length'):.6g}")
            self.outlet_edit.setText(f"{from_si(cfg.operating.outlet_pressure, cfg.display.pressure_unit, 'pressure'):.6g}")
            for combo, value in ((self.unit_combo, unit), (self.flow_unit_combo, cfg.display.flow_unit),
                                 (self.drive_combo, cfg.operating.drive), (self.color_by, cfg.display.color_by),
                                 (self.layout_combo, cfg.pattern.layout), (self.partition_combo, cfg.pattern.partition),
                                 (self.coupling, cfg.model.viscosity_coupling),
                                 (self.transport, cfg.model.transport)):
                combo.blockSignals(True)
                combo.setCurrentText(value)
                combo.blockSignals(False)
            for spin, value in ((self.rows_spin, cfg.pattern.rows), (self.cols_spin, cfg.pattern.cols),
                                (self.output_ny, cfg.grid.output_ny)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
            self.materials.populate(cfg.materials, cfg, issues["materials"])
            colors = material_colors(cfg)
            for row, m in enumerate(cfg.materials):
                item = self.materials.item(row, 0)
                if item is not None:
                    item.setIcon(_swatch(colors[m.id]))
            self.inlets.populate(cfg.inlets, cfg, issues["inlets"])
            self.pillars.populate(cfg.pillars, cfg, issues["pillars"])
            active = cfg.active_inlets()
            if cfg.operating.drive == "flow_rate" and active:
                rates = {i.id: (i.flow_rate or 0.0) for i in active}
                total = sum(rates.values())
                share = ", ".join(f"{k}: {100 * v / total:.1f}%" for k, v in rates.items()) if total else "-"
                self.flow_summary.setText(
                    f"Total flow {from_si(total, cfg.display.flow_unit, 'flow_rate'):.4g} {cfg.display.flow_unit}; "
                    f"ratios {share}")
            else:
                self.flow_summary.setText("Flow rates follow from inlet pressures and resistances (hydraulic network).")
        finally:
            self._refreshing = False


def _swatch(color: str) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(12, 12)
    pixmap.fill(QtGui.QColor(color))
    return QtGui.QIcon(pixmap)


__all__ = ["SetupPanel", "display_unit", "palette_color", "UNITS"]
