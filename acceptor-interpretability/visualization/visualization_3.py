"""visualization_3: fixed-offset vs. per-sequence real-coordinate placement.

Reads results/phase0_results.json (fixed BP=[160,183)/PPT=[183,198) for
every sequence) and results/phase3_results.json (per-sequence branch point
located by a yUnAy-consensus scan, tract derived from that location) and
plots the three disruption effects side by side under both coordinate
schemes - does a better-located branch-point window change the small,
marginal BP-only effect?

Run:
    python visualization/visualization_3.py
Writes:
    results/figures/visualization_3.png
"""

import json
import textwrap
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures" / "visualization_3.png"

COLOR_FIXED = "#2a78d6"
COLOR_REAL = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

GROUPS = [
    ("BP-only\ndisruption", "delta_ppt_bp_only", "bp_real"),
    ("PPT-only\ndisruption", "ppt_only_alone", "ppt_real"),
    ("BP+PPT\nboth disrupted", "both_disrupted", "both_real"),
]


def marker(ax, y, stat, color):
    mean, lo, hi = stat["mean"], stat["ci_lo"], stat["ci_hi"]
    ax.errorbar(
        mean, y, xerr=[[mean - lo], [hi - mean]],
        fmt="o", color=color, ecolor=color, elinewidth=2.2, capsize=4,
        markersize=9, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3,
    )
    ax.annotate(
        f"{mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
        xy=(hi, y), xytext=(8, 0), textcoords="offset points",
        va="center", ha="left", fontsize=8.8, color=INK_SECONDARY,
    )


def main():
    with open(ROOT / "results" / "phase0_results.json") as f:
        phase0 = json.load(f)
    with open(ROOT / "results" / "phase3_results.json") as f:
        phase3 = json.load(f)

    available = [fam.name for fam in fm.fontManager.ttflist]
    font = next((f for f in ("Segoe UI", "DejaVu Sans", "Arial") if f in available), "sans-serif")
    plt.rcParams["font.family"] = font

    fig, ax = plt.subplots(figsize=(9.5, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    group_positions = []
    y = 0.0
    for _ in GROUPS:
        group_positions.append((y + 0.28, y - 0.28))
        y += 1.3

    for (label, fixed_key, real_key), (y_top, y_bot) in zip(GROUPS, group_positions):
        marker(ax, y_top, phase0[fixed_key], COLOR_FIXED)
        marker(ax, y_bot, phase3[real_key], COLOR_REAL)

    ax.axvline(0, color=BASELINE, linewidth=1.1, zorder=1)
    mid_ticks = [(t + b) / 2 for t, b in group_positions]
    ax.set_yticks(mid_ticks)
    ax.set_yticklabels([label for label, *_ in GROUPS], fontsize=10, color=INK_PRIMARY)
    all_ys = [y for pair in group_positions for y in pair]
    ax.set_ylim(min(all_ys) - 0.7, max(all_ys) + 0.7)

    ax.set_xlabel("Δ m_acc  (masked-AG logit margin)", fontsize=10, color=INK_SECONDARY)
    subtitle = textwrap.fill(
        f"located BP: {phase3['mean_bp_distance_from_ag']:.0f}nt upstream of AG on average "
        f"({phase3['n_fallback']}/200 fallback to fixed offset); "
        f"tract length: {phase3['mean_ppt_len']:.0f}nt average (fixed design used a constant 15nt)",
        width=88,
    )
    ax.set_title(
        f"Phase 3 — fixed-offset vs. per-sequence real-coordinate placement\n{subtitle}",
        fontsize=10.5, color=INK_PRIMARY, loc="left", pad=14,
    )

    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", colors=INK_MUTED)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([0], [0], marker="o", color=COLOR_FIXED, linestyle="", markersize=8, label="Fixed offset (Phase 0)"),
        plt.Line2D([0], [0], marker="o", color=COLOR_REAL, linestyle="", markersize=8, label="Per-sequence real coordinate (Phase 3)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.4)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
