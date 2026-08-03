#!/usr/bin/env python
"""Figure 1: partner-specific coupling and causal localization to L3H1.

Every plotted value is read from the recorded pair-category and confirmation
artifacts. The figure reports raw logit differences as logit units.
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
CONFIRMATION = REPO / "results" / "splicebert_coupling" / "confirmation.json"
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
    conf = json.loads(CONFIRMATION.read_text())

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
    scan = conf["scan_top"]
    names = [f"L{row['layer']}H{row['head']}" for row in scan][::-1]
    values = np.asarray([row["recovery"] for row in scan][::-1])
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
    ax.set_xlabel("Recovered coupling (logit units)")
    ax.set_xlim(-0.02, 0.88)
    ax.text(values[-1] + 0.025, y[-1], f"{values[-1]:.3f}", va="center", color=COUPLING)
    panel_header(ax, "B", "Coupling Strength")
    _open_axis(ax)

    # C — the factorial is the mechanistic sign test.
    ax = axes[2]
    factorial = conf["factorial"]
    xx = np.asarray([0.0, 1.0])
    intact = np.asarray([factorial["M_CI"], factorial["M_XI"]])
    ablated = np.asarray([factorial["M_CA"], factorial["M_XA"]])
    ax.plot(
        xx,
        intact,
        marker="o",
        color=COUPLING,
        markersize=5.5,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label="L3H1 intact",
    )
    ax.plot(
        xx,
        ablated,
        marker="s",
        color=DELETED,
        markersize=5.2,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label="L3H1 ablated",
    )
    ax.set_xticks(xx, ["Intact\nacceptor", "Corrupted\nacceptor"])
    ax.set_ylabel("Donor margin (logit units)")
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(3.78, 5.48)
    ax.legend(loc="upper right", frameon=False, handletextpad=0.45)
    ax.annotate(
        f"+{factorial['M_CI'] - factorial['M_CA']:.3f}",
        xy=(0, np.mean([factorial["M_CI"], factorial["M_CA"]])),
        xytext=(0.09, 4.86),
        fontsize=7.3,
        color=INK,
    )
    ax.annotate(
        f"{factorial['M_XI'] - factorial['M_XA']:.3f}",
        xy=(1, np.mean([factorial["M_XI"], factorial["M_XA"]])),
        xytext=(0.78, 4.02),
        fontsize=7.3,
        color=INK,
    )
    ci = factorial["did_ci"]
    ax.text(
        0.02,
        0.04,
        f"Interaction {factorial['did']:+.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
        transform=ax.transAxes,
        color=SEC,
        fontsize=7.2,
        va="bottom",
    )
    panel_header(ax, "C", "Partner Dependence")
    _open_axis(ax)

    save_figure(
        fig,
        OUT,
        sources=[CATEGORIES, CONFIRMATION],
        formats=("png", "pdf"),
        dpi=450,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
