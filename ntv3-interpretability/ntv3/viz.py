"""Shared plotting style for the NTv3 study.

Deliberately different from the DNABERT-2 notebooks' plots (default
matplotlib rcParams, white background, tab10 categorical colors, no shared
style module). This uses the validated categorical/sequential/diverging
palette from the repo's dataviz skill (references/palette.md) so results
across all phases read as one consistent, distinct visual system: warm
off-white chart surface, muted ink, a fixed-order 8-hue categorical ramp
(blue/orange/aqua/yellow/magenta/green/violet/red), and a single-hue blue
sequential ramp for magnitude/heatmaps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

DIVERGING = ("#2a78d6", "#f0efec", "#e34948")  # blue - neutral gray - red

GOOD = "#0ca30c"
CRITICAL = "#d03b3b"


def _base_style(ax, *, title: str | None = None, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5)


def new_figure(figsize=(7, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, path: Path, *, tight: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def errorbar_plot(
    labels, means, ci_low, ci_high, *, title="", xlabel="", ylabel="", figsize=(7, 4.5),
    color=None, horizontal=False,
):
    fig, ax = new_figure(figsize)
    color = color or CATEGORICAL[0]
    x = np.arange(len(labels))
    yerr = np.array([np.array(means) - np.array(ci_low), np.array(ci_high) - np.array(means)])
    if horizontal:
        ax.errorbar(means, x, xerr=yerr, fmt="o", color=color, ecolor=color,
                     elinewidth=2, capsize=3, markersize=6, markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.axvline(0, color=AXIS, linewidth=1, linestyle="--", zorder=1)
    else:
        ax.errorbar(x, means, yerr=yerr, fmt="o", color=color, ecolor=color,
                     elinewidth=2, capsize=3, markersize=6, markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.axhline(0, color=AXIS, linewidth=1, linestyle="--", zorder=1)
    _base_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def bar_plot(labels, values, *, title="", xlabel="", ylabel="", figsize=(7, 4.5),
             colors=None, horizontal=False, errors=None):
    fig, ax = new_figure(figsize)
    colors = colors or [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(labels))]
    x = np.arange(len(labels))
    if horizontal:
        ax.barh(x, values, color=colors, xerr=errors, error_kw=dict(ecolor=INK_SECONDARY, capsize=3),
                height=0.62, zorder=2)
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
    else:
        ax.bar(x, values, color=colors, yerr=errors, error_kw=dict(ecolor=INK_SECONDARY, capsize=3),
               width=0.62, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
    _base_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def line_plot(x, series: dict, *, title="", xlabel="", ylabel="", figsize=(7, 4.5), bands: dict | None = None):
    fig, ax = new_figure(figsize)
    for i, (name, y) in enumerate(series.items()):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=name, zorder=3)
        if bands and name in bands:
            lo, hi = bands[name]
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=2)
    if len(series) > 1:
        leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="best")
    _base_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def heatmap(matrix, *, row_labels, col_labels, title="", xlabel="", ylabel="",
            figsize=(7, 5), cmap_colors=None, center=None):
    fig, ax = new_figure(figsize)
    from matplotlib.colors import LinearSegmentedColormap

    if center is not None:
        colors = [DIVERGING[2], DIVERGING[1], DIVERGING[0]]  # red - gray - blue (low->high)
    else:
        colors = ["#fcfcfb"] + SEQUENTIAL_BLUE
    cmap = LinearSegmentedColormap.from_list("ntv3", colors)
    vmax = np.nanmax(np.abs(matrix)) if center is not None else np.nanmax(matrix)
    vmin = -vmax if center is not None else np.nanmin(matrix)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color=INK_SECONDARY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def histogram(values, *, title="", xlabel="", ylabel="count", figsize=(6.5, 4.2),
              color=None, vlines: dict | None = None, bins=30):
    fig, ax = new_figure(figsize)
    color = color or CATEGORICAL[0]
    ax.hist(values, bins=bins, color=color, alpha=0.85, zorder=2, edgecolor=SURFACE, linewidth=0.5)
    if vlines:
        for i, (label, x) in enumerate(vlines.items()):
            c = CATEGORICAL[(i + 1) % len(CATEGORICAL)]
            ax.axvline(x, color=c, linestyle="--", linewidth=1.8, label=label, zorder=3)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _base_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def scatter_plot(x, y, *, title="", xlabel="", ylabel="", figsize=(6.5, 5), trendline=True, color=None):
    fig, ax = new_figure(figsize)
    color = color or CATEGORICAL[0]
    ax.scatter(x, y, color=color, alpha=0.55, s=28, edgecolor="none", zorder=3)
    if trendline and len(x) > 2:
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, np.polyval(coeffs, xs), color=CATEGORICAL[7], linewidth=2, zorder=4)
    _base_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax
