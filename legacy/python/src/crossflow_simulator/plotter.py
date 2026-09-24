from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from crossflow_simulator.cross_section import CrossSection


DEFAULT_COLORS = [  # a list of colors to use for plotting different labels in the cross-section
    "white",        #0->white
    "tab:blue",     #1->blue
    "tab:orange",   #2->orange
    "tab:green",    #3->green
    "tab:red",      #4->red
    "tab:purple",   #5->purple
    "tab:brown",    #6->brown
    "tab:pink",     #7->pink
]


def plot_cross_section(            #plotting function, takes three inputs
    cross_section: CrossSection,   #cross_section: an instance of the CrossSection class, which contains the labels array to be plotted
    title: str | None = None,      #optional title
    show_axes: bool = True,        #true to show axes labels, false to hide them
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a labeled 2D cross-section.

    Each integer label is shown as a different color.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """

    cmap = ListedColormap(DEFAULT_COLORS)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(
        cross_section.labels,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=len(DEFAULT_COLORS) - 1,
    )

    if title is not None:
        ax.set_title(title)

    if show_axes:
        ax.set_xlabel("y")
        ax.set_ylabel("z")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_aspect("equal")
    plt.tight_layout()

    return fig, ax