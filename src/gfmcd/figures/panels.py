"""Reusable panel primitives.

Each is a *shape of argument*, not a chart type:
  grouped_bars_ci  -- "the effect is present in condition X and collapses under Y"
  ranked_dots      -- "one component dominates a full scan" (localisation)
  interaction_plot -- "the effect of A depends on the state of B" (a 2x2 factorial)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from gfmcd.figures.style import COLORS


def grouped_bars_ci(ax, categories: Sequence[str], series: dict, *,
                    ylabel: str = "", colors: dict | None = None, hline: float | None = 0.0):
    """series: {label: (values, lo, hi)} -- one bar per category, per series."""
    colors = colors or {}
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(categories))
    for i, (label, (v, lo, hi)) in enumerate(series.items()):
        v, lo, hi = np.asarray(v, float), np.asarray(lo, float), np.asarray(hi, float)
        pos = x - 0.4 + w * (i + 0.5)
        c = colors.get(label, COLORS["signal"] if i == 0 else COLORS["ablated"])
        ax.bar(pos, v, width=w * 0.88, color=c, label=label, zorder=3)
        ax.errorbar(pos, v, yerr=[v - lo, hi - v], fmt="none", ecolor=COLORS["text"],
                    elinewidth=0.8, capsize=2, zorder=4)
    if hline is not None:
        ax.axhline(hline, color=COLORS["text"], lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def ranked_dots(ax, values: Sequence[float], *, lo=None, hi=None, labels=None,
                highlight: int | None = None, ylabel: str = "", top_k: int = 20):
    """Ranked scan: one component highlighted against the whole population."""
    v = np.asarray(values, float)
    order = np.argsort(-v)[:top_k]
    xs = np.arange(len(order))
    cols = [COLORS["signal"] if (highlight is not None and i == highlight)
            else COLORS["muted"] for i in order]
    if lo is not None and hi is not None:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        ax.errorbar(xs, v[order], yerr=[v[order] - lo[order], hi[order] - v[order]],
                    fmt="none", ecolor=COLORS["muted"], elinewidth=0.8, capsize=1.5, zorder=2)
    ax.scatter(xs, v[order], c=cols, s=22, zorder=3)
    ax.axhline(0, color=COLORS["text"], lw=0.8)
    if labels is not None:
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[i] for i in order], rotation=90)
    ax.set_ylabel(ylabel)
    return ax, order


def interaction_plot(ax, x_labels: Sequence[str], lines: dict, *, ylabel: str = "",
                     annotate: dict | None = None):
    """2x2 factorial. lines: {label: [y_at_x0, y_at_x1]}.

    Non-parallel lines ARE the finding: the effect of one factor depends on the other.
    """
    x = np.arange(len(x_labels))
    for i, (label, ys) in enumerate(lines.items()):
        c = COLORS["signal"] if i == 0 else COLORS["ablated"]
        ax.plot(x, ys, "o-", color=c, label=label, markersize=5, zorder=3)
    if annotate:
        for xi, (txt, y0, y1) in annotate.items():
            ax.annotate("", xy=(xi, y0), xytext=(xi, y1),
                        arrowprops=dict(arrowstyle="<->", color=COLORS["text"], lw=0.8))
            ax.text(xi + 0.05, (y0 + y1) / 2, txt, fontsize=7, va="center",
                    color=COLORS["text"])
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.35, len(x_labels) - 0.4)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    return ax
