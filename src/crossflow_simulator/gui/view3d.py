"""3D channel view: pillars, ports, inlet and outlet, computed sections, and animated pathlines.

The geometry always follows the current project. The flow (particles moving
from the inlets and ports to the outlet, faint streamlines, and coloured
sections) comes from the latest new-solver result or, when the CFD comparison
has a result, from the CFD reference (chosen in the view). It is hidden while
the chosen result is stale, so the animation never shows a flow that belongs
to other inputs.

Particles are animated by blitting: rotating or zooming redraws the whole
scene once, and each animation frame only redraws the particles over the
cached background.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from ..config import ProjectConfig
from ..materials import INVALID_COLOR, colors_for, stream_table
from ..models.domain import channel_extent
from ..results import Section, SimulationResult
from ..units import UNITS
from .qt import FigureCanvasQTAgg, Figure, QtCore, QtWidgets, QTimer
from .state import ProjectState

FRAME_MS = 40
TRANSIT_SECONDS = 8.0  # at 1x, a typical particle crosses the shown channel in this many seconds
PARTICLES_PER_LINE = 2
STREAMLINE_STRIDE = 4
SECTION_CELLS = (40, 20)  # at most this many displayed cells (y, z) per section slice
COMPRESSED_LENGTH = 3.0  # compressed view: shown length at most this many channel widths
ZOOM = 1.1
SPEEDS = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
SOURCES = {"new": "New solver", "cfd": "CFD reference"}
PILLAR_COLORS = {"bottom": "#6E6E6E", "top": "#A8A8A8"}


class ParticleTracks:
    """Pathlines packed for fast interpolation: valid samples first, padded with the end point."""

    def __init__(self, result: SimulationResult, seed: int = 0):
        lines = result.pathlines
        times = lines.time.astype(float)
        points = lines.points.astype(float)
        valid = np.isfinite(times) & np.isfinite(points).all(axis=2)
        count = valid.sum(axis=1)
        keep = count >= 2
        order = np.argsort(~valid[keep], axis=1, kind="stable")
        times = np.take_along_axis(times[keep], order, axis=1)
        points = np.take_along_axis(points[keep], order[:, :, None], axis=1)
        count = count[keep]
        rows = np.arange(len(count))
        last = count - 1
        pad = np.arange(times.shape[1])[None, :] > last[:, None]
        times[pad] = np.inf
        points[pad] = np.broadcast_to(points[rows, last][:, None, :], points.shape)[pad]
        self.times, self.points, self.count = times, points, count
        self.duration = np.maximum(times[rows, last], 1e-12)
        self.code = lines.code[keep].astype(int)
        self.outlet_x = float(lines.outlet_x)
        reached = lines.status[keep] == 0
        self.typical_transit = float(np.median(self.duration[reached] if reached.any() else self.duration))
        rng = np.random.default_rng(seed)
        self.line = np.repeat(np.arange(len(count)), PARTICLES_PER_LINE)
        slots = np.tile(np.arange(PARTICLES_PER_LINE), len(count))
        self.phase = (slots + rng.random(len(self.line))) / PARTICLES_PER_LINE * self.duration[self.line]

    def positions(self, physical_time: float) -> np.ndarray:
        """Particle positions (N, 3) at ``physical_time``; each particle loops over its line."""
        line = self.line
        t = np.mod(physical_time + self.phase, self.duration[line])
        times = self.times[line]
        j = np.clip((times <= t[:, None]).sum(axis=1), 1, self.count[line] - 1)
        rows = np.arange(len(line))
        t0, t1 = times[rows, j - 1], times[rows, j]
        w = np.clip((t - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0, 1.0)
        p0, p1 = self.points[line, j - 1], self.points[line, j]
        return p0 + w[:, None] * (p1 - p0)


class FlowView3D(QtWidgets.QWidget):
    """Interactive 3D view of the channel with the animated flow of the latest result."""

    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self.result: SimulationResult | None = None
        self.stale = False
        self.sources: dict[str, tuple[SimulationResult | None, bool]] = {"new": (None, False), "cfd": (None, False)}
        self.station: int | None = None
        self.hidden: set[int] = set()
        self.tracks: ParticleTracks | None = None
        self.clock = 0.0
        self.background = None
        self.scatter = None
        self.particle_colors = np.zeros((0, 4))
        self.message = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.play = QtWidgets.QPushButton("Pause")
        self.play.setCheckable(True)
        self.play.setChecked(True)
        self.play.toggled.connect(self._play_toggled)
        self.speed = QtWidgets.QComboBox()
        self.speed.addItems(list(SPEEDS))
        self.speed.setCurrentText("1x")
        self.proportions = QtWidgets.QComboBox()
        self.proportions.addItems(["compressed length", "true proportions"])
        self.proportions.currentTextChanged.connect(lambda _: self.rebuild())
        self.streamlines = QtWidgets.QCheckBox("Streamlines")
        self.streamlines.setChecked(True)
        self.streamlines.toggled.connect(lambda _: self.rebuild())
        self.sections = QtWidgets.QCheckBox("Sections")
        self.sections.setChecked(True)
        self.sections.toggled.connect(lambda _: self.rebuild())
        reset = QtWidgets.QPushButton("Reset view")
        reset.clicked.connect(self.reset_view)
        self.flow_source = QtWidgets.QComboBox()
        for key, label in SOURCES.items():
            self.flow_source.addItem(label, key)
        self.flow_source.setToolTip("Which result drives the particles, streamlines and sections")
        self.flow_source.currentIndexChanged.connect(lambda _: self._activate())
        self.flow_source.setVisible(False)
        for widget in (QtWidgets.QLabel("3D flow"), self.flow_source, self.play, QtWidgets.QLabel("speed"), self.speed,
                       self.proportions, self.streamlines, self.sections, reset):
            bar.addWidget(widget)
        bar.addStretch(1)
        layout.addLayout(bar)
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(320, 220)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        layout.addWidget(self.canvas, 1)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("resize_event", lambda event: self._fit_axes())
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance)
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.timeout.connect(self.rebuild)
        state.changed.connect(lambda: self._rebuild_timer.start(120))
        self.reset_view(redraw=False)
        self.rebuild()

    # ------------------------------------------------------------------ inputs
    @property
    def source(self) -> str:
        return self.flow_source.currentData() or "new"

    def set_result(self, result: SimulationResult | None, stale: bool = False) -> None:
        """The new solver's latest result."""
        self.sources["new"] = (result, stale)
        if self.source == "new":
            self.station = None
            self._activate()

    def set_cfd_result(self, result: SimulationResult | None, stale: bool = False) -> None:
        """The CFD reference (``None`` when the CFD comparison is off or has no result)."""
        previous = self.sources["cfd"]
        if previous[0] is result and previous[1] == stale:
            return
        self.sources["cfd"] = (result, stale)
        available = result is not None and result.pathlines is not None
        self.flow_source.setVisible(available)
        if not available and self.source == "cfd":
            self.flow_source.setCurrentIndex(self.flow_source.findData("new"))  # activates "new"
        elif self.source == "cfd":
            self._activate()

    def _activate(self) -> None:
        result, stale = self.sources.get(self.source, (None, False))
        if result is not self.result:
            self.tracks = None
            if result is not None and result.ok and result.pathlines is not None and len(result.pathlines.code):
                self.tracks = ParticleTracks(result)
        self.result = result
        self.stale = stale
        self.rebuild()

    def mark_stale(self, stale: bool) -> None:
        """Staleness of the new solver's result."""
        result, previous = self.sources["new"]
        if stale != previous:
            self.sources["new"] = (result, stale)
            if self.source == "new":
                self.stale = stale
                self.rebuild()

    def set_station(self, index: int | None) -> None:
        if index != self.station:
            self.station = index
            self.rebuild()

    def set_hidden(self, hidden: set[int]) -> None:
        self.hidden = set(hidden)
        self.rebuild()

    def reset_view(self, redraw: bool = True) -> None:
        self.ax.view_init(elev=24, azim=-58)
        if redraw:
            self.canvas.draw_idle()

    def _fit_axes(self) -> None:
        """Use the panel's width: 3D axes draw into a square, which may extend above and below the
        canvas because the flat channel box needs only part of its height."""
        width, height = self.canvas.get_width_height()
        if not width or not height:
            return
        flat = 1.7 if self.proportions.currentText().startswith("compressed") else 2.6
        side = min(width, flat * height)
        fx, fy = side / width, side / height
        self.ax.set_position([0.5 - 0.5 * fx, 0.5 - 0.5 * fy, fx, fy])

    @property
    def flow_shown(self) -> bool:
        return self.tracks is not None and not self.stale

    # ------------------------------------------------------------------ scene
    def _scale(self, cfg: ProjectConfig) -> float:
        return 1.0 / UNITS["length"][cfg.display.length_unit]

    def _colors(self, cfg: ProjectConfig) -> dict[int, str]:
        streams = self.result.streams if (self.result is not None and self.flow_shown) else stream_table(cfg)
        return colors_for(streams, cfg.display.color_by)

    def rebuild(self) -> None:
        cfg = self.state.config
        s = self._scale(cfg)
        unit = cfg.display.length_unit
        W, H = cfg.channel.width * s, cfg.channel.height * s
        L = channel_extent(cfg) * s
        if self.flow_shown:
            L = max(L, self.tracks.outlet_x * s)
        colors = self._colors(cfg)
        ax = self.ax
        elev, azim = ax.elev, ax.azim
        ax.clear()
        ax.view_init(elev=elev, azim=azim)
        self._fit_axes()
        compressed = self.proportions.currentText().startswith("compressed")
        shown_length = min(L, COMPRESSED_LENGTH * max(W, H)) if compressed else L
        ax.set_box_aspect((shown_length, W, H), zoom=ZOOM)
        ax.set_xlim(0, L)
        ax.set_ylim(0, W)
        ax.set_zlim(0, H)
        ax.grid(False)
        for axis, ticks in ((ax.xaxis, 6), (ax.yaxis, 3), (ax.zaxis, 3)):
            axis.pane.set_alpha(0.0)
            axis.set_tick_params(labelsize=7, pad=0)
            axis.set_major_locator(MaxNLocator(ticks))
        ax.set_xlabel(f"x [{unit}]", fontsize=8, labelpad=-4)
        ax.set_ylabel(f"y [{unit}]", fontsize=8, labelpad=-6)
        ax.set_zlabel(f"z [{unit}]", fontsize=8, labelpad=-6)
        self._draw_channel(L, W, H)
        self._draw_pillars(cfg, s)
        self._draw_ports(cfg, s, W, H, colors)
        if self.flow_shown and self.sections.isChecked():
            self._draw_sections(s, L, colors)
        if self.flow_shown and self.streamlines.isChecked():
            self._draw_streamlines(s, colors)
        self.scatter = ax.scatter([], [], [], s=7, depthshade=False, animated=True, linewidths=0)
        if self.flow_shown:
            self.particle_colors = np.array([to_rgba(colors.get(int(c), INVALID_COLOR)) for c in
                                             self.tracks.code[self.tracks.line]])
            self.particle_colors[np.isin(self.tracks.code[self.tracks.line], list(self.hidden)), 3] = 0.0
            transit = self.tracks.typical_transit
            origin = "the CFD reference" if self.source == "cfd" else "the latest result"
            self.message = (f"Flow from {origin}: {len(self.tracks.duration)} pathlines; a typical "
                            f"particle takes {transit:.3g} s to the outlet (shown in about "
                            f"{TRANSIT_SECONDS / SPEEDS[self.speed.currentText()]:.0f} s).")
        elif self.tracks is not None:
            self.message = "Inputs changed: the flow is shown again after the next calculation."
        else:
            self.message = "Geometry only: run a calculation to animate the flow."
        if compressed and shown_length < L:
            self.message += f"  Length shown compressed {L / shown_length:.1f}x."
        for text in list(self.figure.texts):
            text.remove()
        self.figure.text(0.01, 0.01, self.message, fontsize=7, color="#444444")
        self._update_timer()
        self.background = None
        self.canvas.draw_idle()

    def _draw_channel(self, L: float, W: float, H: float) -> None:
        corners = np.array([[x, y, z] for x in (0, L) for y in (0, W) for z in (0, H)])
        edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        self.ax.add_collection3d(Line3DCollection([corners[list(e)] for e in edges], colors="#303030",
                                                  linewidths=0.8))
        floor = [[(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0)]]
        self.ax.add_collection3d(Poly3DCollection(floor, facecolors=(0.2, 0.4, 0.8, 0.05), edgecolors="none"))
        for x, text, direction in ((0.0, "inlet", 1), (L, "outlet", 1)):
            self.ax.quiver(x - (0.12 * L if x == 0 else 0.0), W / 2, H / 2, 0.1 * L * direction, 0, 0,
                           color="#1B5E20", linewidth=1.5, arrow_length_ratio=0.35)
            self.ax.text(x, W / 2, 1.18 * H, text, fontsize=8, ha="center", color="#1B5E20")

    def _draw_pillars(self, cfg: ProjectConfig, s: float) -> None:
        H = cfg.channel.height * s
        for pillar in cfg.active_pillars():
            fp = pillar.footprint() * s
            h = pillar.height * s
            z0, z1 = (0.0, h) if pillar.mount == "bottom" else (H - h, H)
            low = [(x, y, z0) for x, y in fp]
            high = [(x, y, z1) for x, y in fp]
            faces = [low, high] + [[low[i], low[(i + 1) % len(fp)], high[(i + 1) % len(fp)], high[i]]
                                   for i in range(len(fp))]
            self.ax.add_collection3d(Poly3DCollection(faces, facecolors=PILLAR_COLORS.get(pillar.mount, "#777777"),
                                                      edgecolors="#202020", linewidths=0.5, alpha=0.8))
            self.ax.text(float(fp[:, 0].mean()), float(fp[:, 1].mean()), z1 + 0.08 * H, pillar.id, fontsize=7)

    def _draw_ports(self, cfg: ProjectConfig, s: float, W: float, H: float, colors: dict[int, str]) -> None:
        codes = cfg.stream_codes()
        angle = np.linspace(0, 2 * np.pi, 25)[:-1]
        for inlet in cfg.port_inlets():
            if not inlet.diameter:
                continue
            r = 0.5 * inlet.diameter * s
            x0, t0 = inlet.station * s, inlet.position * s
            a, b = x0 + r * np.cos(angle), t0 + r * np.sin(angle)
            color = colors.get(codes[inlet.id], INVALID_COLOR)
            if inlet.wall in ("top", "bottom"):
                z = H if inlet.wall == "top" else 0.0
                disc = np.column_stack([a, b, np.full_like(a, z)])
                start = (x0, t0, z + (0.45 * H if inlet.wall == "top" else -0.45 * H))
                vector = (0, 0, -0.4 * H) if inlet.wall == "top" else (0, 0, 0.4 * H)
            else:
                y = 0.0 if inlet.wall == "left" else W
                disc = np.column_stack([a, np.full_like(a, y), b])
                start = (x0, y + (-0.45 * W if inlet.wall == "left" else 0.45 * W), t0)
                vector = (0, 0.4 * W, 0) if inlet.wall == "left" else (0, -0.4 * W, 0)
            self.ax.add_collection3d(Poly3DCollection([disc], facecolors=color, edgecolors="#0072B2",
                                                      linewidths=0.8, alpha=0.9))
            self.ax.quiver(*start, *vector, color=color, linewidth=1.5, arrow_length_ratio=0.35)
            self.ax.text(start[0], start[1], start[2], f" {inlet.id}", fontsize=7)

    def _section_polys(self, section: Section, s: float, x: float, colors: dict[int, str]):
        codes = section.codes
        ny, nz = codes.shape[1], codes.shape[0]
        step_y, step_z = max(1, int(np.ceil(ny / SECTION_CELLS[0]))), max(1, int(np.ceil(nz / SECTION_CELLS[1])))
        y_edges = section.y_edges[::step_y] * s
        z_edges = section.z_edges[::step_z] * s
        if y_edges[-1] != section.y_edges[-1] * s:
            y_edges = np.append(y_edges, section.y_edges[-1] * s)
        if z_edges[-1] != section.z_edges[-1] * s:
            z_edges = np.append(z_edges, section.z_edges[-1] * s)
        sample = codes[::step_z, ::step_y]
        polys, faces = [], []
        for i in range(len(z_edges) - 1):
            for j in range(len(y_edges) - 1):
                code = int(sample[min(i, sample.shape[0] - 1), min(j, sample.shape[1] - 1)])
                if code in self.hidden:
                    continue
                y0, y1, z0, z1 = y_edges[j], y_edges[j + 1], z_edges[i], z_edges[i + 1]
                polys.append([(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)])
                faces.append(to_rgba(colors.get(code, INVALID_COLOR), 0.85))
        return polys, faces

    def _draw_sections(self, s: float, L: float, colors: dict[int, str]) -> None:
        sections = self.result.sections
        chosen = [0]
        station = len(sections) - 1 if self.station is None else self.station
        if station > 0:
            chosen.append(station)
        for index in chosen:
            section = sections[index]
            if section.grid_kind != "cell_centred":
                continue
            x = min(section.x * s, L) if index else 0.0
            polys, faces = self._section_polys(section, s, x, colors)
            if polys:
                self.ax.add_collection3d(Poly3DCollection(polys, facecolors=faces, edgecolors="none",
                                                          linewidths=0.0, antialiased=False))

    def _draw_streamlines(self, s: float, colors: dict[int, str]) -> None:
        tracks = self.tracks
        segments, line_colors = [], []
        for i in range(0, len(tracks.count), STREAMLINE_STRIDE):
            code = int(tracks.code[i])
            if code in self.hidden:
                continue
            segments.append(tracks.points[i, :tracks.count[i]] * s)
            line_colors.append(to_rgba(colors.get(code, INVALID_COLOR), 0.35))
        if segments:
            self.ax.add_collection3d(Line3DCollection(segments, colors=line_colors, linewidths=0.6))

    # ------------------------------------------------------------------ animation
    def _play_toggled(self, playing: bool) -> None:
        self.play.setText("Pause" if playing else "Play")
        self._update_timer()

    def _update_timer(self) -> None:
        running = self.flow_shown and self.play.isChecked() and self.isVisible()
        if running and not self.timer.isActive():
            self.timer.start(FRAME_MS)
        elif not running and self.timer.isActive():
            self.timer.stop()

    def particle_positions(self) -> np.ndarray:
        """Current particle positions in metres (empty when no flow is shown)."""
        if not self.flow_shown:
            return np.zeros((0, 3))
        rate = self.tracks.typical_transit / TRANSIT_SECONDS
        return self.tracks.positions(self.clock * rate)

    def advance(self, seconds: float | None = None) -> None:
        """Advance the animation clock (default: one frame at the chosen speed) and redraw the particles."""
        step = FRAME_MS / 1000.0 if seconds is None else seconds
        self.clock += step * SPEEDS[self.speed.currentText()]
        self._draw_particles(blit=True)

    def _on_draw(self, event) -> None:
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        self._draw_particles(blit=False)

    def _draw_particles(self, blit: bool) -> None:
        if self.scatter is None:
            return
        points = self.particle_positions()
        if not len(points):
            return
        s = self._scale(self.state.config)
        if blit:
            if self.background is None:
                return
            self.canvas.restore_region(self.background)
        self.scatter._offsets3d = (points[:, 0] * s, points[:, 1] * s, points[:, 2] * s)
        self.scatter.set_facecolors(self.particle_colors)
        self.scatter.do_3d_projection()
        self.ax.draw_artist(self.scatter)
        if blit:
            self.canvas.blit(self.figure.bbox)

    def showEvent(self, event):  # noqa: N802 - Qt API
        super().showEvent(event)
        self._update_timer()

    def hideEvent(self, event):  # noqa: N802 - Qt API
        super().hideEvent(event)
        self._update_timer()
