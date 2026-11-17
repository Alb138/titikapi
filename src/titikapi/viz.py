"""Shared figure style: one palette and one set of mark specs for every chart in the project.

Values follow the reference data-viz palette (light mode, for static PNGs in the README and
notebooks). Categorical slots are assigned in fixed order; fire magnitude uses a single-hue
orange ramp (light = little/none, dark = a lot).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE = SERIES[0], SERIES[1]

# Single-hue sequential ramp for fire activity / risk: near-surface -> slot-2 orange -> dark.
FIRE_CMAP = LinearSegmentedColormap.from_list(
    "titikapi_fire", ["#fbeee6", "#f5b896", "#eb6834", "#b8461a", "#6e2a0e"]
)


def use_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
        }
    )


def titles(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    """Title, plus an optional secondary line (units, period, source) between it and the plot."""
    ax.set_title(title, pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def degree_ticks(ax: plt.Axes) -> None:
    """Map axes: 95°E / 2°S style tick labels, no axis titles, no spines or tick marks."""
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}°E"))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: "0°" if v == 0 else f"{abs(v):g}°{'N' if v > 0 else 'S'}")
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def map_extent(grid) -> list[float]:
    r = grid.region
    return [r.lon_min, r.lon_max, r.lat_min, r.lat_max]


def masked(values: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~mask, values)


def label_ends(ax: plt.Axes, x: float, items: list[tuple[float, str]], min_gap: float) -> None:
    """Direct labels at the right end of lines, nudged apart so none overlap.

    `items` are (y, text); `min_gap` is the smallest vertical distance in data units."""
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    placed: list[float] = []
    for i in order:
        y = items[i][0] if not placed else max(items[i][0], placed[-1] + min_gap)
        placed.append(y)
        ax.annotate(
            items[i][1],
            xy=(x, items[i][0]),
            xytext=(x, y),
            textcoords="data",
            xycoords="data",
            ha="left",
            va="center",
            fontsize=9,
            color=INK_2,
            annotation_clip=False,
        )
