"""visualization_4: head content-control, retargeted to tract-readers.

Reads results/phase4_results.json. Two panels (PPT, BP), one point per
tested head: x = clean-condition attention mass from the masked-AG query
into the region (log scale - clean masses span six orders of magnitude),
y = delta_corrupt (clean - corrupted attention mass; positive = attention
drops when the region's content is destroyed, the content-control signature).
Filled = passes content-control ("content reader"), hollow = does not
(positional-locality artifact or no attention to the region at all).

Run:
    python visualization/visualization_4.py
Writes:
    results/figures/visualization_4.png
"""

import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures" / "visualization_4.png"

COLOR_A = "#2a78d6"   # passes content-control
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

ORIGIN_MARKER = {"donor_circuit": "o", "top_ppt_attender": "^", "top_bp_attender": "s"}
ORIGIN_LABEL = {
    "donor_circuit": "donor-circuit candidate",
    "top_ppt_attender": "top clean-PPT attender (discovered)",
    "top_bp_attender": "top clean-BP attender (discovered)",
}


def panel(ax, results, region, title):
    heads = results["candidate_heads"]
    for key, head in heads.items():
        d = head[region]
        origin = head["origin"][0]  # primary origin for marker shape
        passes = d["verdict"].startswith("content reader")
        x = max(d["clean_mass"], 1e-6)
        y = d["delta_corrupt"]["mean"]
        lo, hi = d["delta_corrupt"]["ci_lo"], d["delta_corrupt"]["ci_hi"]
        color = COLOR_A if passes else INK_MUTED
        marker = ORIGIN_MARKER[origin]
        ax.errorbar(
            x, y, yerr=[[y - lo], [hi - y]], fmt=marker, color=color, ecolor=color,
            elinewidth=1.1, capsize=2.5, markersize=7.5,
            markerfacecolor=color if passes else SURFACE, markeredgecolor=color,
            markeredgewidth=1.4, zorder=3,
        )
        if passes:
            ax.annotate(
                key, xy=(x, y), xytext=(6, 5), textcoords="offset points",
                fontsize=8.3, color=INK_PRIMARY, fontweight="bold",
            )
    ax.axhline(0, color=BASELINE, linewidth=1.1, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("clean attention mass into region (log scale)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("Δ attention mass\n(clean − corrupted)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title(title, fontsize=10.5, color=INK_PRIMARY, loc="left", pad=8)
    ax.grid(axis="both", color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)


def main():
    with open(ROOT / "results" / "phase4_results.json") as f:
        results = json.load(f)

    available = [fam.name for fam in fm.fontManager.ttflist]
    font = next((f for f in ("Segoe UI", "DejaVu Sans", "Arial") if f in available), "sans-serif")
    plt.rcParams["font.family"] = font

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.6), facecolor=SURFACE)
    for ax in (axA, axB):
        ax.set_facecolor(SURFACE)

    panel(axA, results, "ppt", "A — attention into the polypyrimidine tract")
    panel(axB, results, "bp", "B — attention into the branch-point region")

    handles = [
        plt.Line2D([0], [0], marker=m, color=INK_MUTED, linestyle="", markersize=8,
                   markerfacecolor=SURFACE, markeredgewidth=1.4, label=ORIGIN_LABEL[k])
        for k, m in ORIGIN_MARKER.items()
    ] + [
        plt.Line2D([0], [0], marker="o", color=COLOR_A, linestyle="", markersize=8,
                   markerfacecolor=COLOR_A, label="filled = passes content-control (genuine reader)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.6,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Phase 4 — head content-control: 21 heads (6 donor-circuit candidates + 15 discovered "
        "top attenders) tested for genuine content reading vs. positional locality",
        fontsize=11.5, color=INK_PRIMARY, x=0.01, ha="left", y=1.01,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
