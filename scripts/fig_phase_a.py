"""Phase A figures: both from history JSONs, no checkpoint loaded.

fig_lambda_grok      every arm reaches 1.0000; only colour separates the mechanisms.
fig_cellB_cutoff     the same cell, stopped at 8k vs run to 24k. Why we do not trust an
                     accuracy verdict read mid-grok (D31).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from gfmcd.figstyle import ANCHOR_FREE, ANCHORED, INK, INK_2, INK_MUTED, TRUNCATED, apply_figure_style
from gfmcd.paths import project_paths

CKPT = project_paths().checkpoints
OUT = project_paths().results / "reading-frame/figures"
DATA = project_paths().results / "reading-frame/fig_data"
GUESS = 0.3768  # alignment-free ceiling on the uniform-codon task
GROK = 0.60

# Mechanism from the causal verdict (D29), NOT from accuracy.
ARMS = [
    ("nano_uniform", "λ=0 uniform", ANCHORED),
    ("nano_lam0.25", "λ=0.25", ANCHORED),
    ("nano_lam0.5", "λ=0.5", ANCHORED),
    ("nano_lam0.75", "λ=0.75", ANCHORED),
    ("nano_lam1", "λ=1", ANCHOR_FREE),
    ("nano_lam1.5", "λ=1.5", ANCHOR_FREE),
    ("nano_lam2", "λ=2", ANCHOR_FREE),
]
SENSE = ("nano_sense", "sense codons", ANCHORED)


def history(tag: str):
    return json.loads((CKPT / f"{tag}_history.json").read_text())


def grok_step(h) -> int | None:
    for r in h:
        if r["aa_acc"] > GROK:
            return r["step"]
    return None


def write_csv(path: Path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def fig_lambda_grok():
    fig = plt.figure(figsize=(8.6, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 1.0], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    side = fig.add_subplot(gs[0, 1])
    side.axis("off")
    rows, groks = [], []

    ax.axhline(GUESS, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)

    for tag, label, colour in ARMS + [SENSE]:
        h = history(tag)
        xs = [r["step"] for r in h]
        ys = [r["aa_acc"] for r in h]
        ax.plot(xs, ys, color=colour, lw=1.9, alpha=0.9,
                ls=(0, (5, 2)) if tag == "nano_sense" else "-", zorder=3)
        g = grok_step(h)
        mech = "anchored" if colour == ANCHORED else "anchor-free"
        for x, y in zip(xs, ys):
            rows.append([tag, label, x, y, mech, g or ""])
        if g:
            gy = next(r["aa_acc"] for r in h if r["step"] == g)
            ax.plot([g], [gy], "o", ms=4.5, color=colour, mec="white", mew=1.0, zorder=4)
        groks.append((label, g, colour))

    ax.set_xlim(0, 4300)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("training step")
    ax.set_ylabel("amino-acid accuracy")

    ax.text(4250, GUESS - 0.045, "guessing ceiling 0.3768", color=INK_MUTED, fontsize=8, ha="right")
    ax.text(4250, 0.90, "all arms hold 1.0000\nto the end of training", color=INK_MUTED, fontsize=8,
            ha="right", va="top")

    handles = [plt.Line2D([], [], color=ANCHORED, lw=2.2, label="anchored counter"),
               plt.Line2D([], [], color=ANCHOR_FREE, lw=2.2, label="anchor-free estimator")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.02),
              fontsize=8.5, labelcolor=INK_2)

    # Grok-step table lives outside the data area, so no curve can collide with it.
    side.text(0.0, 0.86, "grok step", fontsize=9, color=INK, transform=side.transAxes)
    side.text(0.0, 0.815, "first step with aa > 0.60", fontsize=7.8, color=INK_MUTED, transform=side.transAxes)
    for i, (label, g, colour) in enumerate(groks):
        y = 0.745 - i * 0.062
        side.text(0.0, y, label, fontsize=8.5, color=colour, transform=side.transAxes)
        side.text(0.98, y, f"{g:,}", fontsize=8.5, color=colour, ha="right",
                  transform=side.transAxes, family="DejaVu Sans Mono")
    side.text(0.0, 0.745 - 8 * 0.062 - 0.02,
              "Among anchored arms the grok\nstep falls as the shortcut\nstrengthens: a partial shortcut\nscaffolds the exact mechanism.",
              fontsize=7.8, color=INK_2, transform=side.transAxes, va="top")

    ax.set_title("Every arm reaches 1.0000. Only the causal assay separates the mechanisms.",
                 color=INK, loc="left", pad=22)
    ax.text(0, 1.035, "Colour = mechanism from cue conflict + start-codon ablation (D29), never from accuracy. "
                      "Sense-codon control dashed.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_lambda_grok.png")
    fig.savefig(OUT / "fig_lambda_grok.svg")
    write_csv(DATA / "fig_lambda_grok.csv", rows, ["arm", "label", "step", "aa_acc", "mechanism", "grok_step"])
    plt.close(fig)

    print("fig_lambda_grok: grok steps")
    for label, g, colour in groks:
        print(f"  {label:<16}{'anchored' if colour == ANCHORED else 'anchor-free':<14}{g}")


def fig_cellB_cutoff():
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    rows = []

    ax.axhline(GUESS, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)

    for tag, label, colour in [("cellB_s1", "seed 1 — 24,000 steps", ANCHORED),
                               ("cellB_s0", "seed 0 — stopped at 8,000", TRUNCATED)]:
        h = history(tag)
        xs = [r["step"] for r in h]
        ys = [r["aa_acc"] for r in h]
        ax.plot(xs, ys, color=colour, lw=2.1, zorder=3, label=label)
        for x, y in zip(xs, ys):
            rows.append([tag, label, x, y])

    ax.axvline(8000, color=TRUNCATED, lw=1.0, ls=(0, (3, 3)), zorder=2)
    ax.annotate("seed-0 budget ends here\nverdict withdrawn (D31)",
                xy=(8000, 0.66), xytext=(9000, 0.85), color=TRUNCATED, fontsize=8.5, va="center",
                arrowprops=dict(arrowstyle="-", color=TRUNCATED, lw=0.8, shrinkA=0, shrinkB=4))
    ax.annotate("transition\nstep 18–20k", xy=(18900, 0.78), xytext=(20600, 0.66),
                color=ANCHORED, fontsize=8.5, va="center",
                arrowprops=dict(arrowstyle="-", color=ANCHORED, lw=0.8, shrinkA=0, shrinkB=4))
    ax.text(24300, 0.9735, "  0.9735", color=ANCHORED, fontsize=8.5, va="center")
    ax.text(27000, GUESS - 0.05, "guessing ceiling 0.3768", color=INK_MUTED, fontsize=8, ha="right")

    ax.set_xlim(0, 27000)
    ax.set_ylim(0, 1.06)
    ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("training step")
    ax.set_ylabel("amino-acid accuracy")
    ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.99), fontsize=8.5, labelcolor=INK_2)

    ax.set_title("The same cell, stopped early and run long.", color=INK, loc="left", pad=22)
    ax.text(0, 1.035, "Cell B (no shortcut, hard detector). Read at 8k it looks like an absent circuit. "
                      "It was mid-transition.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")

    fig.savefig(OUT / "fig_cellB_cutoff.png")
    fig.savefig(OUT / "fig_cellB_cutoff.svg")
    write_csv(DATA / "fig_cellB_cutoff.csv", rows, ["run", "label", "step", "aa_acc"])
    plt.close(fig)

    h0, h1 = history("cellB_s0"), history("cellB_s1")
    print(f"\nfig_cellB_cutoff: seed 0 final aa {h0[-1]['aa_acc']:.4f} @ step {h0[-1]['step']}")
    print(f"                  seed 1 final aa {h1[-1]['aa_acc']:.4f} @ step {h1[-1]['step']}  grok @ {grok_step(h1)}")


if __name__ == "__main__":
    apply_figure_style()
    fig_lambda_grok()
    fig_cellB_cutoff()
    print(f"\nwrote {OUT}/*.png|svg  and  {DATA}/*.csv")
