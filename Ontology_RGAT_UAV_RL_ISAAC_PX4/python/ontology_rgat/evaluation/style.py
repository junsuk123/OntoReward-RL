"""One visual system for every figure this workspace produces.

The palette is the validated categorical default: slot 1 blue, slot 2 orange,
slot 3 aqua. Two-policy comparisons -- which is almost every figure here -- use
slots 1 and 2, a pair that clears the colour-vision-deficiency and
normal-vision separation floors on an all-pairs check against the light chart
surface. Aqua only ever appears as a third series alongside both, which is the
three-slot set the same check clears.

Two rules the retired MATLAB figures broke and these do not:

* **No dual axis.** ``makePlots.m`` put episode energy and depletion rate on one
  plot with two y-scales, and hover seconds against power on another. The
  alignment of two scales is arbitrary, so such a plot invents a relationship
  that is not in the data. Each measure gets its own panel here.
* **Identity is never colour alone.** Every multi-series panel carries a legend,
  and every bar carries its value, so the figure survives greyscale printing.
"""
from __future__ import annotations

from typing import Any

__all__ = ["SERIES", "POLICY_COLORS", "INK", "MUTED", "GRID", "SURFACE",
           "apply_style", "annotate_bars", "figure"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8dce3"
# Validated categorical slots 1-3.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
POLICY_COLORS = {"Manual": SERIES[0], "Ontology-RGAT": SERIES[1]}
# Reserved for state, never for a series.
STATUS = {"good": "#1baf7a", "bad": "#e34948", "reference": "#52514e"}


def apply_style() -> None:
    """Set the shared rcParams. Agg, so a figure never needs a display."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.size": 9,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def figure(nrows: int = 1, ncols: int = 1, size: tuple[float, float] = (7.0, 4.0),
           **kwargs: Any):
    import matplotlib.pyplot as plt
    apply_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=size, constrained_layout=True,
                             **kwargs)
    return fig, axes


def annotate_bars(ax, bars, values, fmt: str = "{:.2f}") -> None:
    """Put the value on every bar.

    Three of the palette's light-mode slots sit below 3:1 against the surface,
    which obliges visible labels rather than colour alone -- and a reader of a
    printed figure wants the number anyway.
    """
    for bar, value in zip(bars, values):
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2,
                                        bar.get_height()),
                    ha="center", va="bottom", fontsize=8, color=INK,
                    xytext=(0, 2), textcoords="offset points")
