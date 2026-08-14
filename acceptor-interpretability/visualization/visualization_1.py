"""visualization_1: Phase-0 effects vs. the Phase-1 controls, two panels.

Reads results/phase0_results.json and results/phase1_results.json.

Panel A - effects vs. the far-intronic noise floor (the one true null
control here: content-destroying corruption of a region with no known
functional role). Marker fill marks SURVIVES vs MARGINAL (CI overlaps the
floor's CI).

Panel B - order vs. composition: content-destroying corruption vs.
composition-preserving shuffle-in-place, for BP and PPT. This is a
*different* comparison, not folded into panel A, because PPT shuffle-in-place
came back large (+1.207) - a real finding (the tract signal is largely
order/identity-driven, not aggregate pyrimidine density), not a null result.

Run:
    python visualization/visualization_1.py
Writes:
    results/figures/visualization_1.png
"""

import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures" / "visualization_1.png"

COLOR_A = "#2a78d6"   # categorical slot 1 - content-destroying corruption
COLOR_B = "#eb6834"   # categorical slot 2 - composition-preserving shuffle-in-place
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
FLOOR_BAND = "#89878122"


def marker(ax, x, y, stat, color, filled):
    mean, lo, hi = stat["mean"], stat["ci_lo"], stat["ci_hi"]
    ax.errorbar(
        mean, y, xerr=[[mean - lo], [hi - mean]],
        fmt="o", color=color, ecolor=color, elinewidth=2, capsize=4,
        markersize=8, markerfacecolor=color if filled else SURFACE,
        markeredgecolor=color, markeredgewidth=1.6, zorder=3,
    )
    ax.annotate(
        f"{mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
        xy=(hi if mean >= 0 else lo, y), xytext=(8 if mean >= 0 else -8, 0),
        textcoords="offset points", va="center",
        ha="left" if mean >= 0 else "right", fontsize=8.8, color=INK_SECONDARY,
    )


def main():
    with open(ROOT / "results" / "phase0_results.json") as f:
        phase0 = json.load(f)
    with open(ROOT / "results" / "phase1_results.json") as f:
        phase1 = json.load(f)
    verdicts = phase1.get("verdicts_vs_control_floor", {})

    available = [fam.name for fam in fm.fontManager.ttflist]
    font = next((f for f in ("Segoe UI", "DejaVu Sans", "Arial") if f in available), "sans-serif")
    plt.rcParams["font.family"] = font

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(9.5, 8.2), facecolor=SURFACE,
        gridspec_kw={"height_ratios": [5, 3]},
    )
    for ax in (axA, axB):
        ax.set_facecolor(SURFACE)

    # --- Panel A: effects vs. far-intronic floor -----------------------------
    rows_a = [
        ("Context required", phase0["context_required"], "context_required"),
        ("AG-scanning", phase0["ag_scanning"], "ag_scanning"),
        ("PPT-only disruption", phase0["ppt_only_alone"], "ppt_only_alone"),
        ("BP-only disruption (Δppt)", phase0["delta_ppt_bp_only"], "delta_ppt_bp_only"),
    ]
    floor = phase1["far_intronic_control"]
    ys = list(range(len(rows_a)))[::-1]
    axA.axvspan(floor["ci_lo"], floor["ci_hi"], color=FLOOR_BAND, zorder=0)
    axA.axvline(floor["mean"], color=INK_MUTED, linewidth=1.3, linestyle="--", zorder=1)
    axA.annotate(
        f"far-intronic noise floor: {floor['mean']:+.3f}  [{floor['ci_lo']:+.3f}, {floor['ci_hi']:+.3f}]",
        xy=(floor["ci_hi"], max(ys) + 0.55), xytext=(4, 0), textcoords="offset points",
        fontsize=8.8, color=INK_MUTED, ha="left",
    )
    for y, (label, stat, key) in zip(ys, rows_a):
        filled = verdicts.get(key, {}).get("clears_control_floor", True)
        marker(axA, 0, y, stat, COLOR_A, filled)
    axA.axvline(0, color=BASELINE, linewidth=1.1, linestyle="-", zorder=1)
    axA.set_yticks(ys)
    axA.set_yticklabels([label for label, *_ in rows_a], fontsize=10, color=INK_PRIMARY)
    axA.set_title(
        "A — Phase-0 effects vs. the far-intronic noise floor\n"
        "filled = clears the floor, hollow = CI overlaps it (marginal)",
        fontsize=10.5, color=INK_PRIMARY, loc="left", pad=10,
    )
    axA.set_ylim(-0.7, max(ys) + 1.0)

    # --- Panel B: order vs. composition ---------------------------------------
    groups = [
        ("Polypyrimidine\ntract", phase0["ppt_only_alone"], phase1["ppt_shuffle_in_place"]),
        ("Branch point", phase0["delta_ppt_bp_only"], phase1["bp_shuffle_in_place"]),
    ]
    ys_b = []
    y = 0.0
    group_positions = []
    for _, _, _ in groups:
        group_positions.append((y + 0.31, y - 0.31))
        y += 1.4
    for (label, content, shuffle), (y_top, y_bot) in zip(groups, group_positions):
        marker(axB, 0, y_top, content, COLOR_A, True)
        marker(axB, 0, y_bot, shuffle, COLOR_B, True)
    axB.axvline(0, color=BASELINE, linewidth=1.1, linestyle="-", zorder=1)
    mid_ticks = [(t + b) / 2 for t, b in group_positions]
    axB.set_yticks(mid_ticks)
    axB.set_yticklabels([label for label, _, _ in groups], fontsize=10, color=INK_PRIMARY)
    all_ys = [y for pair in group_positions for y in pair]
    axB.set_ylim(min(all_ys) - 0.7, max(all_ys) + 0.7)
    axB.set_title(
        "B — order vs. composition: content-destroying vs. shuffle-in-place",
        fontsize=10.5, color=INK_PRIMARY, loc="left", pad=10,
    )

    for ax in (axA, axB):
        ax.set_xlabel("Δ m_acc  (masked-AG logit margin)", fontsize=9.5, color=INK_SECONDARY)
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(axis="x", colors=INK_MUTED)
        ax.tick_params(axis="y", length=0)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin - (xmax - xmin) * 0.05, xmax + (xmax - xmin) * 0.5)

    handles = [
        plt.Line2D([0], [0], marker="o", color=COLOR_A, linestyle="", markersize=8, label="Content-destroying corruption"),
        plt.Line2D([0], [0], marker="o", color=COLOR_B, linestyle="", markersize=8, label="Composition-preserving shuffle-in-place"),
    ]
    axB.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.8, labelcolor=INK_SECONDARY)

    fig.suptitle(
        "Phase 1 — control fix: is the tract/BP signal real, and is it composition or order?",
        fontsize=12, color=INK_PRIMARY, x=0.01, ha="left", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
