"""visualization_0: forest plot of the Phase 0 effect sizes.

Reads results/phase0_results.json (written by experiments/phase0_pilot.py)
and renders every reported Delta as a mean +/- 95% bootstrap CI, grouped by
acceptor-window vs matched non-splice-control where a control exists.

Run:
    python visualization/visualization_0.py
Writes:
    results/figures/visualization_0.png
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "phase0_results.json"
OUT = ROOT / "results" / "figures" / "visualization_0.png"

# validated 2-slot categorical pair (dataviz skill palette, light mode):
# slot 1 blue = acceptor window, slot 2 orange = non-splice control.
COLOR_ACCEPTOR = "#2a78d6"
COLOR_CONTROL = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# (group label, acceptor-side key, control-side key or None, control n-key)
ROWS = [
    ("Context required\n(clean − random-AG)", "context_required", None),
    ("AG-scanning\n(true site − downstream decoy)", "ag_scanning", None),
    ("Branch point + tract\nboth disrupted", "both_disrupted", "both_control_nonsplice"),
    ("Polypyrimidine tract\ndisrupted alone", "ppt_only_alone", "ppt_only_control_nonsplice"),
    ("Branch point\ndisrupted alone (Δppt)", "delta_ppt_bp_only", "delta_ppt_control_nonsplice"),
]


def build_layout(summary):
    """Assign a y-coordinate to every (group, series) row, groups spaced apart,
    paired series stacked tightly within a group."""
    entries = []
    y = 0.0
    group_gap = 1.4
    series_gap = 0.62
    for label, acc_key, ctrl_key in reversed(ROWS):
        if ctrl_key is None:
            entries.append((label, "acceptor", summary[acc_key], y))
            y += group_gap
        else:
            entries.append((label, "control", summary[ctrl_key], y))
            y += series_gap
            entries.append((label, "acceptor", summary[acc_key], y))
            y += group_gap
    return entries


def main():
    with open(RESULTS) as f:
        summary = json.load(f)

    available = [fam.name for fam in fm.fontManager.ttflist]
    font = next((f for f in ("Segoe UI", "DejaVu Sans", "Arial") if f in available), "sans-serif")
    plt.rcParams["font.family"] = font

    entries = build_layout(summary)

    fig, ax = plt.subplots(figsize=(10.5, 6.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    seen_group_labels = {}
    for label, series, stat, y in entries:
        color = COLOR_ACCEPTOR if series == "acceptor" else COLOR_CONTROL
        mean, lo, hi = stat["mean"], stat["ci_lo"], stat["ci_hi"]
        ax.errorbar(
            mean, y, xerr=[[mean - lo], [hi - mean]],
            fmt="o", color=color, ecolor=color, elinewidth=2, capsize=4,
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3,
        )
        ax.annotate(
            f"{mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
            xy=(hi, y), xytext=(8, 0), textcoords="offset points",
            va="center", ha="left", fontsize=9, color=INK_SECONDARY,
        )
        seen_group_labels.setdefault(label, []).append(y)

    ax.axvline(0, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)

    group_ticks = [sum(ys) / len(ys) for ys in seen_group_labels.values()]
    group_labels = list(seen_group_labels.keys())
    ax.set_yticks(group_ticks)
    ax.set_yticklabels(group_labels, fontsize=10, color=INK_PRIMARY)

    ax.set_xlabel("Δ m_acc  (masked-AG logit margin)", fontsize=10, color=INK_SECONDARY)
    subtitle = textwrap.fill(
        "SpliceBERT on real GUE splice_reconstructed windows "
        "(n=200 acceptor / n=200 non-splice, 95% bootstrap CI)",
        width=78,
    )
    ax.set_title(
        f"Phase 0 — Acceptor-Recognition Circuit: effect sizes\n{subtitle}",
        fontsize=11, color=INK_PRIMARY, loc="left", pad=14,
    )

    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", colors=INK_MUTED)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([0], [0], marker="o", color=COLOR_ACCEPTOR, linestyle="", markersize=8, label="Acceptor window"),
        plt.Line2D([0], [0], marker="o", color=COLOR_CONTROL, linestyle="", markersize=8, label="Non-splice control"),
    ]
    ax.legend(
        handles=handles, loc="lower right", frameon=False,
        fontsize=9, labelcolor=INK_SECONDARY,
    )

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.42)  # room for direct labels

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
