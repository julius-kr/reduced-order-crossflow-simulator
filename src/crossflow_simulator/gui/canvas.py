"""Matplotlib canvases for cross-sections and geometry, embedded in Qt."""
from __future__ import annotations

import numpy as np

from ..materials import INVALID_COLOR
from ..plotter import code_image
from ..results import Section
from ..units import UNITS
from .qt import FigureCanvasQTAgg, Figure, QtWidgets


class SectionCanvas(FigureCanvasQTAgg):
    """One cross-section on physical axes with optional overlays."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, title: str = ""):
        self.figure = Figure(figsize=(5, 3), layout="constrained")
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.title = title
        # Called after the user zooms or pans (for synchronized views); Axes.clear() drops the
        # Matplotlib callbacks, so they are reconnected after every redraw.
        self.limit_listeners: list = []
        self.size_listeners: list = []
        self.setMinimumSize(280, 180)
        self.clear_message("No result yet")

    def clear_message(self, text: str) -> None:
        self.axes.clear()
        self.axes.set_axis_off()
        self.axes.text(0.5, 0.5, text, ha="center", va="center", transform=self.axes.transAxes, wrap=True)
        self.draw_idle()

    def show_section(
        self,
        section: Section,
        colors: dict[int, str],
        labels: dict[int, str],
        length_unit: str = "um",
        hidden: set[int] | None = None,
        interfaces: bool = True,
        reference: np.ndarray | None = None,
        title: str | None = None,
        limits: tuple[float, float, float, float] | None = None,
        stale: bool = False,
    ) -> None:
        scale = 1.0 / UNITS["length"][length_unit]
        codes = section.codes.copy()
        hidden = hidden or set()
        shown = codes.copy()
        for code in hidden:
            shown[codes == code] = -2
        palette = dict(colors)
        palette[-2] = "#FFFFFF"
        palette.setdefault(-1, INVALID_COLOR)
        index, cmap, legend = code_image(shown, palette)
        y0, y1, z0, z1 = (v * scale for v in section.extent)
        ax = self.axes
        ax.clear()
        ax.set_axis_on()
        ax.imshow(index, origin="lower", extent=(y0, y1, z0, z1), cmap=cmap, interpolation="nearest",
                  vmin=-0.5, vmax=len(cmap.colors) - 0.5, aspect="equal")
        if interfaces and codes.size:
            y, z = section.y * scale, section.z * scale
            for code in np.unique(codes):
                if code < 0 or code in hidden:
                    continue
                mask = (codes == code).astype(float)
                if 0 < mask.mean() < 1 and len(y) > 1 and len(z) > 1:
                    ax.contour(y, z, mask, levels=[0.5], colors="k", linewidths=0.6)
        if reference is not None and reference.shape == codes.shape:
            differs = np.ma.masked_where(reference == codes, np.ones_like(codes, dtype=float))
            ax.imshow(differs, origin="lower", extent=(y0, y1, z0, z1), cmap="Reds", alpha=0.45, vmin=0, vmax=1,
                      interpolation="nearest", aspect="equal")
        ax.set_xlabel(f"y [{length_unit}]")
        ax.set_ylabel(f"z [{length_unit}]")
        heading = title if title is not None else self.title
        if stale:
            heading = f"{heading}  [STALE: inputs changed]"
        ax.set_title(heading, fontsize=9, color="#B00020" if stale else "black")
        handles = []
        from matplotlib.patches import Patch

        for code, color in legend.items():
            if code == -2:
                continue
            name = "no valid trajectory" if code == -1 else labels.get(code, f"unmatched label {code}")
            handles.append(Patch(facecolor=color, edgecolor="0.3", label=name))
        if handles:
            ax.legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
        if limits is not None:
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])
        for event in ("xlim_changed", "ylim_changed"):
            ax.callbacks.connect(event, self._limits_changed)
        self.draw_idle()

    def _limits_changed(self, _axes) -> None:
        for listener in list(self.limit_listeners):
            listener()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        for listener in getattr(self, "size_listeners", []):
            listener()

    def set_plot_size(self, width: float, height: float, bottom: float | None = None) -> None:
        """Use a shared image size and an optional baseline in canvas pixels."""
        self.figure.set_layout_engine(None)
        fw, fh = self.figure.bbox.width, self.figure.bbox.height
        if fw <= 0 or fh <= 0:
            return
        bottom = (fh - height) / 2 if bottom is None else bottom
        position = [55 / fw, bottom / fh, width / fw, height / fh]
        if not np.allclose(self.axes.get_position().bounds, position, atol=1e-6):
            self.axes.set_position(position)
            self.draw_idle()


class GeometryCanvas(FigureCanvasQTAgg):
    """Read-only top view used in reports; the interactive editor lives in geometry_view."""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(6, 2), layout="constrained")
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
