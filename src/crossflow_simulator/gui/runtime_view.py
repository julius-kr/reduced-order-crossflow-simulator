"""Measured solve times for the solvers selected in the toolbar."""
from __future__ import annotations

import math

from .qt import Figure, FigureCanvasQTAgg, QtWidgets


class RuntimeChart(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Solver runtimes", parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.figure = Figure(figsize=(4, 2.4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(200, 180)
        layout.addWidget(self.canvas)
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.27, right=0.97, top=0.95, bottom=0.23)
        self.records: list[tuple[str, str, float | None, str]] = []

    def show_runtimes(self, records: list[tuple[str, str, float | None, str]]) -> None:
        """Rows contain backend, display label, measured seconds, and result state."""
        if records == self.records:
            return
        self.records = records
        ax = self.axes
        ax.clear()
        colors = {"new": "#23956A", "legacy": "#397EB5", "cfd": "#D89325", "legacy_cfd": "#8064AD"}
        maximum = max((seconds for _, _, seconds, _ in records if seconds is not None), default=1)
        maximum = max(maximum, 0.001)
        for row, (backend, _, seconds, status) in enumerate(records):
            if seconds is not None and math.isfinite(seconds) and seconds >= 0:
                bar = ax.barh(row, seconds, height=0.55, color=colors[backend])[0]
                if status == "stale":
                    bar.set_alpha(0.45)
                    bar.set_hatch("//")
                text = f"{seconds:.3g} s" + (" (stale)" if status == "stale" else "")
                x = seconds + maximum * 0.025
            else:
                text = status.capitalize() if status != "success" else "Time unavailable"
                x = maximum * 0.025
            ax.text(x, row, text, va="center", fontsize=8)
        ax.set_yticks(range(len(records)), [label for _, label, _, _ in records], fontsize=8)
        ax.set_ylim(len(records) - 0.5, -0.5)
        ax.set_xlim(0, maximum * 1.6)
        ax.set_xlabel("Solve time [s]", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.xaxis.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
        for edge in ("top", "right", "left"):
            ax.spines[edge].set_visible(False)
        self.canvas.draw_idle()
