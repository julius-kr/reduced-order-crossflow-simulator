"""Physical-axis plotting of cross-sections, geometry, and paired comparisons.

Colours come from stable material IDs (see ``materials``), never from the
numerical value of a code, so more than eight materials stay distinct and a
legacy result with unexpected raw labels is shown as such rather than being
silently recoloured.
"""
from __future__ import annotations

from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

from .materials import INVALID_COLOR, UNMATCHED_COLORS  # noqa: E402
from .results import Section, SimulationResult  # noqa: E402
from .units import UNITS  # noqa: E402


def code_image(codes: np.ndarray, colors: dict[int, str]) -> tuple[np.ndarray, ListedColormap, dict[int, str]]:
    """Index image plus colormap; codes without a colour get distinct 'unmatched' colours."""
    present = [int(c) for c in np.unique(codes)]
    palette: list[str] = []
    lookup: dict[int, int] = {}
    legend: dict[int, str] = {}
    unmatched = iter(UNMATCHED_COLORS * 8)
    for c in present:
        color = colors.get(c) or next(unmatched)
        lookup[c] = len(palette)
        palette.append(color)
        legend[c] = color
    index = np.vectorize(lookup.get)(codes) if codes.size else codes
    return index, ListedColormap(palette if palette else [INVALID_COLOR]), legend


def draw_section(ax, section: Section, colors: dict[int, str], length_unit: str = "um",
                 labels: dict[int, str] | None = None, title: str | None = None, legend: bool = True) -> None:
    scale = 1.0 / UNITS["length"][length_unit]
    index, cmap, legend_colors = code_image(section.codes, colors)
    y0, y1, z0, z1 = (v * scale for v in section.extent)
    ax.imshow(index, origin="lower", extent=(y0, y1, z0, z1), cmap=cmap, interpolation="nearest",
              vmin=-0.5, vmax=len(cmap.colors) - 0.5, aspect="equal")
    ax.set_xlabel(f"y [{length_unit}]")
    ax.set_ylabel(f"z [{length_unit}]")
    if title:
        ax.set_title(title, fontsize=9)
    if legend:
        handles = []
        for code, color in legend_colors.items():
            name = "no valid trajectory" if code < 0 else (labels or {}).get(code, f"code {code}")
            handles.append(Patch(facecolor=color, edgecolor="0.3", label=name))
        ax.legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)


def stream_labels(result: SimulationResult) -> dict[int, str]:
    return {int(s["code"]): f"{s['inlet_id']} ({s['material_id']})" for s in result.streams}


def result_figure(result: SimulationResult, colors: dict[int, str], length_unit: str = "um",
                  sections: Iterable[int] | None = None):
    chosen = list(range(len(result.sections))) if sections is None else list(sections)
    fig, axes = plt.subplots(1, len(chosen), figsize=(4.2 * len(chosen) + 1.5, 3.2), squeeze=False)
    labels = stream_labels(result)
    for ax, index in zip(axes[0], chosen):
        section = result.sections[index]
        draw_section(ax, section, colors, length_unit, labels, section.name, legend=index == chosen[-1])
    fig.tight_layout()
    return fig


def draw_top_view(ax, config: dict, length_unit: str = "um") -> None:
    """Channel footprint with pillars and ports (x-y plane) from a config snapshot."""
    from .config import ProjectConfig

    cfg = ProjectConfig.from_dict(config)
    scale = 1.0 / UNITS["length"][length_unit]
    W = cfg.channel.width * scale
    xs = [0.0]
    for p in cfg.active_pillars():
        fp = p.footprint() * scale
        xs.extend(fp[:, 0])
        ax.add_patch(Polygon(fp, closed=True, facecolor="#555555" if p.mount == "bottom" else "#999999",
                             edgecolor="k", alpha=0.8, label=f"{p.id} ({p.mount})"))
    for port in cfg.port_inlets():
        if port.wall in ("top", "bottom"):
            circle = matplotlib.patches.Circle((port.station * scale, port.position * scale),
                                               0.5 * port.diameter * scale, fill=False,
                                               linestyle="--" if port.wall == "top" else "-")
            ax.add_patch(circle)
            ax.annotate(port.id, (port.station * scale, port.position * scale), fontsize=7, ha="center")
        xs.append(port.station * scale)
    length = (cfg.channel.length * scale) if cfg.channel.length else max(xs) + W
    ax.plot([0, length], [0, 0], "k-", lw=1.5)
    ax.plot([0, length], [W, W], "k-", lw=1.5)
    ax.set_xlim(-0.05 * length, 1.05 * length)
    ax.set_ylim(-0.1 * W, 1.1 * W)
    ax.set_aspect("equal")
    ax.set_xlabel(f"x [{length_unit}]")
    ax.set_ylabel(f"y [{length_unit}]")


def comparison_figure(legacy: SimulationResult | None, new: SimulationResult | None, colors_legacy: dict[int, str],
                      colors_new: dict[int, str], length_unit: str = "um", timings: dict | None = None):
    """Side-by-side final sections: legacy on the left, new on the right, same physical axes."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True, sharey=True)
    for ax, result, colors, name in ((axes[0], legacy, colors_legacy, "Legacy solver"),
                                     (axes[1], new, colors_new, "New solver")):
        if result is None or not result.ok or result.final is None:
            status = "not run" if result is None else f"{result.status}: {result.message[:80]}"
            ax.text(0.5, 0.5, f"{name}\n{status}", ha="center", va="center", transform=ax.transAxes, wrap=True)
            ax.set_axis_off()
            continue
        extra = ""
        if timings and name in timings:
            extra = f"\n{timings[name]}"
        draw_section(ax, result.final, colors, length_unit, stream_labels(result), f"{name}{extra}",
                     legend=name == "New solver")
    fig.tight_layout()
    return fig
