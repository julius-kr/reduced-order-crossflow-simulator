"""Interactive geometry editor: top view (x-y) with draggable pillars/ports, side view (x-z)."""
from __future__ import annotations

from ..config import ProjectConfig
from ..models.domain import channel_extent
from ..units import UNITS
from .qt import QtCore, QtGui, QtWidgets, Signal
from .state import ProjectState

BOTTOM_BRUSH = QtGui.QColor("#5A5A5A")
TOP_BRUSH = QtGui.QColor("#B8B8B8")
PORT_PEN = QtGui.QColor("#0072B2")


class _Draggable:
    """Mixin reporting the drag delta (scene units) to the editor on release."""

    def setup(self, editor: "GeometryEditor", kind: str, index: int) -> None:
        self.editor, self.kind, self.index = editor, kind, index
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable | QtWidgets.QGraphicsItem.ItemIsSelectable
                      | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self.setToolTip(f"{kind} {index + 1}: drag to move (snaps to the grid)")

    def itemChange(self, change, value):  # noqa: N802 - Qt API
        if change == QtWidgets.QGraphicsItem.ItemPositionChange and self.editor.snap > 0:
            step = self.editor.snap
            return QtCore.QPointF(round(value.x() / step) * step, round(value.y() / step) * step)
        if change == QtWidgets.QGraphicsItem.ItemSelectedHasChanged and value:
            self.editor.item_selected.emit(self.kind, self.index)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt API
        super().mouseReleaseEvent(event)
        delta = self.pos()
        if delta.x() or delta.y():
            self.editor.moved(self.kind, self.index, delta.x(), delta.y())


class PillarItem(_Draggable, QtWidgets.QGraphicsPolygonItem):
    pass


class PortItem(_Draggable, QtWidgets.QGraphicsEllipseItem):
    pass


class GeometryEditor(QtWidgets.QWidget):
    item_selected = Signal(str, int)

    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self.snap = 0.0
        layout = QtWidgets.QVBoxLayout(self)
        tools = QtWidgets.QHBoxLayout()
        self.snap_box = QtWidgets.QComboBox()
        self.snap_box.addItems(["no snapping", "1", "5", "10", "25"])
        self.snap_box.setCurrentText("5")
        self.snap_box.currentTextChanged.connect(self._snap_changed)
        tools.addWidget(QtWidgets.QLabel("Grid snap"))
        tools.addWidget(self.snap_box)
        self.unit_label = QtWidgets.QLabel()
        tools.addWidget(self.unit_label)
        tools.addStretch(1)
        fit = QtWidgets.QPushButton("Fit view")
        fit.clicked.connect(self.fit)
        tools.addWidget(fit)
        layout.addLayout(tools)
        self.top_scene = QtWidgets.QGraphicsScene(self)
        self.side_scene = QtWidgets.QGraphicsScene(self)
        self.top = QtWidgets.QGraphicsView(self.top_scene)
        self.side = QtWidgets.QGraphicsView(self.side_scene)
        for view in (self.top, self.side):
            view.setRenderHint(QtGui.QPainter.Antialiasing)
            view.scale(1, -1)  # y and z point up
            view.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        layout.addWidget(QtWidgets.QLabel("Top view (x streamwise, y lateral) — drag pillars and ports"))
        layout.addWidget(self.top, 3)
        layout.addWidget(QtWidgets.QLabel("Side view (x streamwise, z vertical)"))
        layout.addWidget(self.side, 2)
        self._snap_changed(self.snap_box.currentText())
        state.changed.connect(self.rebuild)
        self.rebuild()

    def _snap_changed(self, text: str) -> None:
        self.snap = 0.0 if not text[0].isdigit() else float(text)

    def scale(self) -> float:
        return 1.0 / UNITS["length"][self.state.config.display.length_unit]

    def channel_length(self, cfg: ProjectConfig) -> float:
        return channel_extent(cfg)

    def rebuild(self) -> None:
        cfg = self.state.config
        s = self.scale()
        W, H = cfg.channel.width * s, cfg.channel.height * s
        L = self.channel_length(cfg) * s
        unit = cfg.display.length_unit
        self.unit_label.setText(f"units: {unit}")
        self.top_scene.clear()
        self.side_scene.clear()
        wall = QtGui.QPen(QtGui.QColor("black"), 0)
        wall.setCosmetic(True)
        self.top_scene.addRect(QtCore.QRectF(0, 0, L, W), wall, QtGui.QBrush(QtGui.QColor("#F4F8FF")))
        self.side_scene.addRect(QtCore.QRectF(0, 0, L, H), wall, QtGui.QBrush(QtGui.QColor("#F4F8FF")))
        inlet_text = self.top_scene.addText("pattern inlets →")
        inlet_text.setTransform(QtGui.QTransform.fromScale(W / 250.0, -W / 250.0))
        inlet_text.setPos(0, W * 0.6)
        for index, pillar in enumerate(cfg.pillars):
            if not pillar.enabled:
                continue
            fp = pillar.footprint() * s
            polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in fp])
            item = PillarItem(polygon)
            item.setup(self, "pillar", index)
            brush = BOTTOM_BRUSH if pillar.mount == "bottom" else TOP_BRUSH
            item.setBrush(QtGui.QBrush(brush))
            pen = QtGui.QPen(QtGui.QColor("black"), 0)
            pen.setCosmetic(True)
            if pillar.mount == "top":
                pen.setStyle(QtCore.Qt.DashLine)
            item.setPen(pen)
            item.setToolTip(f"{pillar.id}: {pillar.mount}-mounted, height {pillar.height * s:.4g} {unit}")
            self.top_scene.addItem(item)
            x0, x1 = float(fp[:, 0].min()), float(fp[:, 0].max())
            h = pillar.height * s
            z0 = 0.0 if pillar.mount == "bottom" else H - h
            self.side_scene.addRect(QtCore.QRectF(x0, z0, x1 - x0, h), pen, QtGui.QBrush(brush))
        for index, inlet in enumerate(cfg.inlets):
            if not inlet.enabled or inlet.placement != "port" or not inlet.diameter:
                continue
            d = inlet.diameter * s
            pen = QtGui.QPen(PORT_PEN, 0)
            pen.setCosmetic(True)
            pen.setWidth(2)
            if inlet.wall in ("top", "bottom"):
                item = PortItem(QtCore.QRectF(inlet.station * s - d / 2, inlet.position * s - d / 2, d, d))
                if inlet.wall == "bottom":
                    pen.setStyle(QtCore.Qt.DashLine)
                z = H if inlet.wall == "top" else 0.0
                self.side_scene.addLine(inlet.station * s - d / 2, z, inlet.station * s + d / 2, z, pen)
            else:
                y = 0.0 if inlet.wall == "left" else W
                item = PortItem(QtCore.QRectF(inlet.station * s - d / 2, y - 0.05 * W, d, 0.1 * W))
            item.setup(self, "inlet", index)
            item.setPen(pen)
            item.setBrush(QtGui.QBrush(QtGui.QColor(0, 114, 178, 60)))
            item.setToolTip(f"{inlet.id}: {inlet.wall}-wall port, D = {d:.4g} {unit}")
            self.top_scene.addItem(item)
        self.fit()

    def fit(self) -> None:
        for view, scene in ((self.top, self.top_scene), (self.side, self.side_scene)):
            rect = scene.itemsBoundingRect()
            view.fitInView(rect.adjusted(-rect.width() * 0.02, -rect.height() * 0.1, rect.width() * 0.02,
                                         rect.height() * 0.1), QtCore.Qt.KeepAspectRatio)

    def moved(self, kind: str, index: int, dx: float, dy: float) -> None:
        factor = 1.0 / self.scale()

        def mutate(c: ProjectConfig) -> None:
            if kind == "pillar":
                c.pillars[index].x += dx * factor
                c.pillars[index].y += dy * factor
            else:
                inlet = c.inlets[index]
                inlet.station += dx * factor
                if inlet.wall in ("top", "bottom"):
                    inlet.position += dy * factor

        messages = self.state.apply(mutate)
        if messages and any(m.startswith("[error]") for m in messages):
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "\n".join(m for m in messages if m.startswith("[error]")))

    def resizeEvent(self, event):  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self.fit()
