"""visualization_2: absolute m_acc per condition vs. the random-AG floor.

Reads results/phase2_results.json (experiments/phase2_absolute_floor.py) and
plots each condition's absolute m_acc as a point + 95% CI against a shaded
floor band (the random-AG condition's CI) - separating "still above floor,
retains signal" from "collapsed to floor, no signal left to lose".

Run:
    python visualization/visualization_2.py
Writes:
    results/figures/visualization_2.png
"""

import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "phase2_results.json"
OUT = ROOT / "results" / "figures" / "visualization_2.png"

COLOR_ABOVE = "#2a78d6"     # clearly above floor - retains signal (blue, categorical slot 1)
COLOR_AT_FLOOR = "#eb6834"  # at floor - no signal left (orange, categorical slot 2)
COLOR_FLOOR_LINE = "#898781"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
FLOOR_BAND = "#eb683422"  # translucent orange band for the floor CI

ORDER = [
    ("clean", "Clean\n(intact BP + PPT)"),
    ("bp_only", "BP disrupted"),
    ("ppt_only", "PPT disrupted"),
    ("both", "BP + PPT\nboth disrupted"),
]


def main():
    with open(RESULTS) as f:
        summary = json.load(f)

    available = [fam.name for fam in fm.fontManager.ttflist]
    font = next((f for f in ("Segoe UI", "DejaVu Sans", "Arial") if f in available), "sans-serif")
    plt.rcParams["font.family"] = font

    floor = summary["rand_ag"]
    verdicts = summary["verdicts"]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhspan(floor["ci_lo"], floor["ci_hi"], color=FLOOR_BAND, zorder=0, label="_nolegend_")
    ax.axhline(floor["mean"], color=COLOR_FLOOR_LINE, linewidth=1.4, linestyle="--", zorder=1)
    ax.annotate(
        f"random-AG floor: {floor['mean']:+.3f}  [{floor['ci_lo']:+.3f}, {floor['ci_hi']:+.3f}]",
        xy=(len(ORDER) - 1 + 0.42, floor["mean"]), xytext=(0, 6), textcoords="offset points",
        ha="right", va="bottom", fontsize=9, color=INK_SECONDARY,
    )

    xs = list(range(len(ORDER)))
    for x, (key, label) in zip(xs, ORDER):
        stat = summary[key]
        mean, lo, hi = stat["mean"], stat["ci_lo"], stat["ci_hi"]
        if key == "clean":
            color = COLOR_ABOVE
        else:
            above = verdicts[key]["clearly_above_floor"]
            color = COLOR_ABOVE if above else COLOR_AT_FLOOR
        ax.errorbar(
            x, mean, yerr=[[mean - lo], [hi - mean]],
            fmt="o", color=color, ecolor=color, elinewidth=2.5, capsize=5,
            markersize=10, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
        )
        ax.annotate(
            f"{mean:+.3f}", xy=(x, hi), xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=9.5, color=INK_PRIMARY,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in ORDER], fontsize=10, color=INK_PRIMARY)
    ax.set_xlim(-0.6, len(ORDER) - 1 + 0.6)

    ax.set_ylabel("absolute m_acc  (masked-AG logit margin)", fontsize=10, color=INK_SECONDARY)
    ax.set_title(
        "Phase 2 — absolute m_acc vs. the random-AG floor\n"
        "(n=200 acceptor windows, 95% bootstrap CI; shaded band = floor CI)",
        fontsize=11, color=INK_PRIMARY, loc="left", pad=14,
    )

    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="y", colors=INK_MUTED)
    ax.tick_params(axis="x", length=0)

    handles = [
        plt.Line2D([0], [0], marker="o", color=COLOR_ABOVE, linestyle="", markersize=9, label="Clearly above floor (retains signal)"),
        plt.Line2D([0], [0], marker="o", color=COLOR_AT_FLOOR, linestyle="", markersize=9, label="Indistinguishable from floor"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
