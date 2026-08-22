#!/usr/bin/env python
"""Figure 1: partner-specific coupling and load-bearing attention heads.

Every plotted value is read from the manifest-backed pair-category, corrected
development-scan, and sealed-test artifacts. The figure reports raw pre-softmax
logit differences as logit units.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from figstyle import (  # noqa: E402
    COUPLING,
    DELETED,
    INK,
    MUT,
    RULE,
    SEC,
    panel_header,
    save_figure,
    use_style,
)

CATEGORIES = HERE / "data" / "pair_categories.json"
SCAN = REPO / "results" / "splicebert_coupling" / "development" / "ablation_scan.json"
SEALED = REPO / "results" / "splicebert_coupling" / "sealed_test.json"
OUT = HERE / "fig1_coupling"

def _open_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=RULE, linewidth=0.65)
    ax.grid(axis="x", visible=False)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def main() -> None:
    cats = json.loads(CATEGORIES.read_text())["categories"]
    scan = json.loads(SCAN.read_text())
    sealed = json.loads(SEALED.read_text())

    use_style()
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.8,
        }
    )

    fig = plt.figure(figsize=(10.9, 3.25), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.34, 0.88, 1.08), wspace=0.30)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    # A — paired dot-and-interval display avoids bar-area decoration.
    ax = axes[0]
    keys = list(cats)
    labels = [
        "Same\nintron",
        "Same\nexon",
        "No\npair",
        "Other\ndonor",
        "A–A\ncontrol",
        "Random\nmatch",
    ]
    x = np.arange(len(keys), dtype=float)
    for offset, condition, label, color, marker in (
        (-0.12, "intact", "Intact model", COUPLING, "o"),
        (0.12, "L3H1 ablated", "L3H1 ablated", DELETED, "s"),
    ):
        means = np.asarray([cats[key][condition]["mean"] for key in keys])
        lows = np.asarray([cats[key][condition]["lo"] for key in keys])
        highs = np.asarray([cats[key][condition]["hi"] for key in keys])
        ax.errorbar(
            x + offset,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=5.2,
            capsize=2.4,
            elinewidth=1.25,
            label=label,
            zorder=3,
        )
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Causal coupling (logit units)")
    ax.set_ylim(-0.12, 1.10)
    ax.legend(loc="upper right", frameon=False, handletextpad=0.45)
    panel_header(ax, "A", "L3H1: Intron Coupling")
    _open_axis(ax)

    # B — horizontal ranking is readable at manuscript scale.
    ax = axes[1]
    scan_top = scan["heads"][:8]
    names = [f"L{row['layer']}H{row['head']}" for row in scan_top][::-1]
    values = np.asarray([row["specificity"] for row in scan_top][::-1])
    colors = [COUPLING if name == "L3H1" else MUT for name in names]
    y = np.arange(len(names))
    ax.hlines(y, 0, values, color=colors, linewidth=1.5, zorder=1)
    ax.scatter(
        values,
        y,
        s=[43 if name == "L3H1" else 26 for name in names],
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(y, names)
    ax.set_xlabel("Partner-specific coupling loss\n(logit units)")
    ax.set_xlim(-0.02, 0.88)
    ax.text(values[-1] + 0.025, y[-1], f"{values[-1]:.3f}", va="center", color=COUPLING)
    panel_header(ax, "B", "Corrected Mean-Ablation Scan")
    _open_axis(ax)

    # C — sealed necessity estimates for the two nominated heads and their pair.
    ax = axes[2]
    head_records = sealed["necessity"]["heads"]
    keys = ["L3H1", "L2H15", "circuit C"]
    labels = ["L3H1", "L2H15", "L3H1 + L2H15"]
    means = np.asarray([head_records[key]["lost"] for key in keys])
    lows = np.asarray([head_records[key]["lost_ci"][0] for key in keys])
    highs = np.asarray([head_records[key]["lost_ci"][1] for key in keys])
    yy = np.arange(len(keys))[::-1]
    ax.errorbar(
        means,
        yy,
        xerr=np.vstack((means - lows, highs - means)),
        fmt="o",
        color=COUPLING,
        markerfacecolor=COUPLING,
        markeredgecolor="white",
        markeredgewidth=0.7,
        markersize=5.5,
        capsize=2.4,
        elinewidth=1.25,
        zorder=3,
    )
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(yy, labels)
    ax.set_xlabel("Coupling removed (logit units)")
    ax.set_xlim(0, 1.18)
    pair = head_records["circuit C"]
    ax.text(
        0.03,
        0.05,
        f"Pair removes {100 * pair['frac_lost']:.1f}% · health retained "
        f"{100 * pair['health_retained']:.1f}%",
        transform=ax.transAxes,
        color=SEC,
        fontsize=7.2,
        va="bottom",
    )
    panel_header(ax, "C", "Sealed Necessity")
    _open_axis(ax)

    save_figure(
        fig,
        OUT,
        sources=[CATEGORIES, SCAN, SEALED],
        formats=("png", "pdf"),
        dpi=450,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
