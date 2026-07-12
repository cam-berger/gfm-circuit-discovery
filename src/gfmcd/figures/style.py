"""Shared visual language.  One place to change, every study stays consistent."""
from __future__ import annotations

# Colour-blind safe (Okabe-Ito).  SIGNAL/CONTROL are semantic, not decorative:
# anything that is the claim gets SIGNAL, anything that is a null/control gets MUTED.
COLORS = {
    "signal": "#0072B2",     # the effect / intact model
    "ablated": "#D55E00",    # the intervention
    "muted": "#999999",      # controls, nulls, non-significant
    "accent": "#009E73",
    "warn": "#CC79A7",
    "grid": "#E6E6E6",
    "text": "#222222",
}

RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "lines.linewidth": 1.4,
}


def use_style():
    import matplotlib as mpl
    mpl.rcParams.update(RC)


def panel_label(ax, letter: str, dx: float = -0.12, dy: float = 1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left", color=COLORS["text"])
