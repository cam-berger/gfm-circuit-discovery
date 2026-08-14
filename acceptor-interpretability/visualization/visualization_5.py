"""visualization_5: the intervention ladder - necessity, mediation, routing vs. content.

Reads results/phase5_results.json. Three panels:
  A - mean-ablation deltas (necessity): 5 individual heads + 2 joint groups.
  B - restoration-patching recovery fraction (mediation): 2 primary heads x
      {corrupt, shuffle} recipient.
  C - Q/V-split recovery fraction (routing vs. content): 2 primary heads x
      {Q-patch, V-patch}.

All three panels share the finding: every single-head and most joint
interventions are statistically indistinguishable from zero - a clean null
on individual/small-group causal necessity, in contrast to Phase 4's real
attention-level content-sensitivity for these same heads.

Run:
    python visualization/visualization_5.py
Writes:
    results/figures/visualization_5.png
"""

import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures" / "visualization_5.png"

COLOR_A = "#2a78d6"
COLOR_B = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def marker(ax, y, mean, lo, hi, color, filled, label=None):
    ax.errorbar(
        mean, y, xerr=[[mean - lo], [hi - mean]],
        fmt="o", color=color, ecolor=color, elinewidth=2, capsize=4,
        markersize=8, markerfacecolor=color if filled else SURFACE,
        markeredgecolor=color, markeredgewidth=1.6, zorder=3,
    )
    if label:
        ax.annotate(label, xy=(hi, y), xytext=(8, 0),
                    textcoords="offset points", va="center",
                    ha="left", fontsize=8.3, color=INK_SECONDARY)


def style_axis(ax, xlabel):
    ax.axvline(0, color=BASELINE, linewidth=1.1, zorder=1)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8.3)
    ax.tick_params(axis="y", length=0)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin - (xmax - xmin) * 0.05, xmax + (xmax - xmin) * 0.35)


def main():
    with open(ROOT / "results" / "phase5_results.json") as f:
        r = json.load(f)

    available = [fam.name for fam in fm.fontManager.ttflist]
    font = next((f for f in ("Segoe UI", "DejaVu Sans", "Arial") if f in available), "sans-serif")
    plt.rcParams["font.family"] = font

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 5.4), facecolor=SURFACE)
    for ax in (axA, axB, axC):
        ax.set_facecolor(SURFACE)

    # Panel A - ablation
    rows_a = [
        ("L5H8 (primary)", r["ablation"]["L5H8"], COLOR_A),
        ("L4H12 (primary)", r["ablation"]["L4H12"], COLOR_A),
        ("L3H5 (secondary)", r["ablation"]["L3H5"], COLOR_A),
        ("L0H13 (secondary)", r["ablation"]["L0H13"], COLOR_A),
        ("L0H2 (control)", r["ablation"]["L0H2"], INK_MUTED),
        ("joint: 4 readers", r["ablation"]["joint_real"], COLOR_B),
        ("joint: all 5", r["ablation"]["joint_all"], COLOR_B),
    ]
    ys = list(range(len(rows_a)))[::-1]
    for y, (label, stat, color) in zip(ys, rows_a):
        significant = stat["ci_lo"] > 0 or stat["ci_hi"] < 0
        marker(axA, y, stat["mean"], stat["ci_lo"], stat["ci_hi"], color, significant,
               f"{stat['mean']:+.3f}")
    axA.set_yticks(ys)
    axA.set_yticklabels([l for l, *_ in rows_a], fontsize=9.5, color=INK_PRIMARY)
    axA.set_ylim(-0.8, max(ys) + 0.8)
    axA.set_title("A - mean-ablation (necessity)\nfilled = CI excludes 0", fontsize=10, color=INK_PRIMARY, loc="left")
    style_axis(axA, "Δ m_acc (clean − ablated)")

    # Panel B - mediation (recovery fraction)
    rows_b = [
        ("L5H8 → corrupt", r["mediation"]["L5H8"]["corrupt"]),
        ("L5H8 → shuffle", r["mediation"]["L5H8"]["shuffle"]),
        ("L4H12 → corrupt", r["mediation"]["L4H12"]["corrupt"]),
        ("L4H12 → shuffle", r["mediation"]["L4H12"]["shuffle"]),
    ]
    ys_b = list(range(len(rows_b)))[::-1]
    for y, (label, stat) in zip(ys_b, rows_b):
        significant = stat["ci_lo"] > 0 or stat["ci_hi"] < 0
        marker(axB, y, stat["mean"], stat["ci_lo"], stat["ci_hi"], COLOR_A, significant,
               f"{stat['mean']:+.1%}")
    axB.set_yticks(ys_b)
    axB.set_yticklabels([l for l, _ in rows_b], fontsize=9.5, color=INK_PRIMARY)
    axB.set_ylim(-0.8, max(ys_b) + 0.8)
    axB.set_title("B - restoration patching (mediation)\nrecovery fraction of clean−degraded gap",
                  fontsize=10, color=INK_PRIMARY, loc="left")
    style_axis(axB, "recovery fraction")

    # Panel C - Q/V split
    rows_c = [
        ("L5H8 Q-patch (routing)", r["qv_split"]["L5H8"]["q_patch"], COLOR_A),
        ("L5H8 V-patch (content)", r["qv_split"]["L5H8"]["v_patch"], COLOR_B),
        ("L4H12 Q-patch (routing)", r["qv_split"]["L4H12"]["q_patch"], COLOR_A),
        ("L4H12 V-patch (content)", r["qv_split"]["L4H12"]["v_patch"], COLOR_B),
    ]
    ys_c = list(range(len(rows_c)))[::-1]
    for y, (label, stat, color) in zip(ys_c, rows_c):
        significant = stat["ci_lo"] > 0 or stat["ci_hi"] < 0
        marker(axC, y, stat["mean"], stat["ci_lo"], stat["ci_hi"], color, significant,
               f"{stat['mean']:+.1%}")
    axC.set_yticks(ys_c)
    axC.set_yticklabels([l for l, *_ in rows_c], fontsize=9.5, color=INK_PRIMARY)
    axC.set_ylim(-0.8, max(ys_c) + 0.8)
    axC.set_title("C - Q/V split (routing vs. content)\nrecovery fraction, corrupted recipient",
                  fontsize=10, color=INK_PRIMARY, loc="left")
    style_axis(axC, "recovery fraction")

    fig.suptitle(
        "Phase 5 - intervention ladder on Phase 4's discovered tract-readers: "
        "a clean null on single/small-group causal necessity",
        fontsize=12, color=INK_PRIMARY, x=0.01, ha="left", y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
