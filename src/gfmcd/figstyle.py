"""Figure style. Colours come from a palette validated with the dataviz validator.

    node scripts/validate_palette.js "#2a78d6,#e34948" --mode light
    -> ALL CHECKS PASS   worst adjacent CVD dE 74.6 (protan)

Colour encodes the MECHANISM, never the arm index. That is the entire point of
`fig_lambda_grok`: every curve reaches 1.0000 and only the colour separates them.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"

# Categorical slots (light mode), from the validated reference palette.
BLUE = "#2a78d6"      # slot 1
AQUA = "#1baf7a"      # slot 2
YELLOW = "#eda100"    # slot 3
VIOLET = "#4a3aa7"    # slot 5
RED = "#e34948"       # slot 6
ORANGE = "#eb6834"    # slot 8

# Roles. Validated as a 4-set: #2a78d6,#eb6834,#4a3aa7,#e34948 -> all six checks PASS,
# worst adjacent CVD dE 78.8.
ANCHORED = BLUE       # the model built an anchored counter
ANCHOR_FREE = RED     # the model built an anchor-free estimator (bag)
COUNTER = BLUE
MIXED = ORANGE
RELAY = VIOLET
BAG = RED
TRUNCATED = ORANGE    # a run stopped mid-transition


def apply_figure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": "#d8d7d2",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e8e7e2",
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "text.color": INK,
        }
    )


def annotate(ax, text, xy, xytext, color=INK_2, **kw):
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        color=color,
        fontsize=8.5,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.7, shrinkA=0, shrinkB=3),
        **kw,
    )
