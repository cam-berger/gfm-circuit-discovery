"""Phase B figures. Read the dumps; never re-derive a number by hand."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from gfmcd.figstyle import BAG, COUNTER, INK, INK_2, INK_MUTED, MIXED, RELAY, apply_figure_style

DATA = Path("results/fig_data")
OUT = Path("results/figures")
MECH = {"counter": COUNTER, "mixed": MIXED, "relay": RELAY, "bag": BAG}
SHORT = {
    "A · uniform, easy detector": "A  uniform, easy",
    "B · uniform, hard detector": "B  uniform, hard",
    "C · real CDS, easy detector": "C  real CDS, easy",
    "D · real CDS, natural": "D  real CDS, natural",
    "μ=0 · real CDS, LM objective": "μ=0  real CDS, LM",
}


def fig_signature():
    sigs = json.loads((DATA / "signature.json").read_text())
    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    ax.axvline(0.05, color=INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.axhline(0.50, color=INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)

    # Region labels, placed in empty space rather than over the points.
    ax.text(0.30, 0.72, "COUNTER\ncounted from the anchor", color=COUNTER, fontsize=8.5, va="center")
    ax.text(0.44, 0.27, "RELAY\nanchor used, not counted from", color=RELAY, fontsize=8.5, va="center")
    ax.text(0.005, 0.30, "BAG\nanchor unused", color=BAG, fontsize=8.5, va="center")

    # (label dx, dy, ha) per model, chosen so nothing overlaps.
    place = {
        "A · uniform, easy detector": (-10, -34, "right"),
        "B · uniform, hard detector": (10, 14, "left"),
        "C · real CDS, easy detector": (0, 14, "center"),
        "D · real CDS, natural": (0, 16, "center"),
        "μ=0 · real CDS, LM objective": (10, -24, "left"),
    }

    for s in sigs:
        c = MECH[s["mechanism"]]
        ax.plot([s["x"]], [s["y"]], "o", ms=10, color=c, mec="white", mew=1.4, zorder=4)
        ctl = max(abs(s["per_layer"][s["best_layer"]]["control"]), 1.5e-3)
        ax.plot([s["x"] - ctl, s["x"] + ctl], [s["y"], s["y"]], color=c, lw=1.6, alpha=0.6, zorder=3)

        if s["mechanism"] == "bag":
            note = "no layer shows\nanchor dependence"
        else:
            note = f"anchor enters {s['best_layer'].replace('.resid_post','')}"
        dx, dy, ha = place[s["model"]]
        ax.annotate(f"{SHORT[s['model']]}\n{note}", (s["x"], s["y"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8.2, color=c, va="center")

    ax.set_xlim(-0.035, 0.74)
    ax.set_ylim(-0.12, 1.14)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("peak anchor dependence   (max over depth of far-field probe drop, minus decoy-Met control)")
    ax.set_ylabel("anchor share under cue conflict  (≥64 nt)")
    ax.set_title("Three mechanisms, one architecture, indistinguishable by accuracy.", color=INK, loc="left", pad=22)
    ax.text(0, 1.035, "All five models score 0.96–1.00 on their own objective. Both axes are causal and "
                      "objective-independent. Bars = decoy-Met control.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")

    handles = [plt.Line2D([], [], ls="", marker="o", ms=8, color=MECH[m], label=m)
               for m in ("counter", "mixed", "relay", "bag")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.99), fontsize=8.5, labelcolor=INK_2)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_2x2_signature.png")
    fig.savefig(OUT / "fig_2x2_signature.svg")
    plt.close(fig)


def fig_decay():
    curves = json.loads((DATA / "cue_conflict_decay.json").read_text())
    sigs = {s["model"]: s["mechanism"] for s in json.loads((DATA / "signature.json").read_text())}
    fig, ax = plt.subplots(figsize=(7.4, 4.5))

    # Four of the five curves converge below 0.15; nudge the end-labels apart.
    dy = {
        "A · uniform, easy detector": 0,
        "B · uniform, hard detector": 0,
        "D · real CDS, natural": 10,
        "μ=0 · real CDS, LM objective": -3,
        "C · real CDS, easy detector": -15,
    }

    for name, rows in curves.items():
        c = MECH[sigs[name]]
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        ax.plot(xs, ys, "-o", ms=5, color=c, lw=2.0, mec="white", mew=0.9)
        ax.annotate(SHORT[name], (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, dy[name]),
                    fontsize=8.2, color=c, va="center")

    ax.axhline(1 / 3, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(9, 1 / 3 + 0.02, "chance 0.333", color=INK_MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([9.5, 23.5, 47.5, 95.5, 191.5])
    ax.set_xticklabels(["4–15", "16–31", "32–63", "64–127", "128–255"])
    ax.set_xlim(7, 560)
    ax.set_ylim(-0.07, 1.06)
    ax.set_xlabel("distance from the inserted nucleotide (nt)")
    ax.set_ylabel("follows the ANCHOR")
    ax.set_title("Under cue conflict, only the counter keeps following the anchor.", color=INK, loc="left", pad=22)
    ax.text(0, 1.035, "One nucleotide inserted after the ATG: counting says one phase, local codon structure says another.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")

    fig.savefig(OUT / "fig_cue_conflict_decay.png")
    fig.savefig(OUT / "fig_cue_conflict_decay.svg")
    plt.close(fig)


def fig_layers():
    layers = json.loads((DATA / "layer_probe.json").read_text())
    sigs = {s["model"]: s["mechanism"] for s in json.loads((DATA / "signature.json").read_text())}
    names = ["resid.embed"] + [f"L{i}.resid_post" for i in range(6)]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))

    # The two relays land within 0.03 of the counter at L5; separate the end-labels.
    dy = {
        "A · uniform, easy detector": 5,
        "D · real CDS, natural": -5,
        "C · real CDS, easy detector": -18,
        "μ=0 · real CDS, LM objective": 6,
        "B · uniform, hard detector": -6,
    }

    for name, vals in layers.items():
        c = INK_MUTED if "untrained" in name else MECH[sigs[name]]
        ys = [vals[n] for n in names]
        ax.plot(range(len(names)), ys, "-o", ms=4.5, color=c, lw=2.0 if c != INK_MUTED else 1.3,
                ls="-" if c != INK_MUTED else (0, (4, 3)), mec="white", mew=0.8)
        label = "untrained (random init)" if "untrained" in name else SHORT[name]
        ax.annotate(label, (len(names) - 1, ys[-1]), textcoords="offset points",
                    xytext=(8, dy.get(name, 0)), fontsize=8.2, color=c, va="center")

    ax.axhline(1 / 3, color=INK_MUTED, lw=0.8, ls=(0, (2, 3)))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["embed", "L0", "L1", "L2", "L3", "L4", "L5"])
    ax.set_xlim(-0.3, 8.9)
    ax.set_ylim(0.25, 1.06)
    ax.set_xlabel("residual stream, by depth")
    ax.set_ylabel("phase decodability (linear probe)")
    ax.set_title("The bag pools at L0 and stops. The relay pools, then corrects. The counter builds.",
                 color=INK, loc="left", pad=22)
    ax.text(0, 1.035, "Real CDS anchor-free ceiling = 0.9509. The relay reaches it by L2, then exceeds it at L4 using the anchor.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    # Draw the ceiling only across the data, so it does not run under the right-margin labels.
    ax.plot([0, len(names) - 1], [0.9509, 0.9509], color=RELAY, lw=0.8, ls=(0, (5, 3)), alpha=0.7, zorder=1)
    ax.text(0.05, 0.962, "anchor-free ceiling 0.9509", color=RELAY, fontsize=8)

    fig.savefig(OUT / "fig_layer_probe.png")
    fig.savefig(OUT / "fig_layer_probe.svg")
    plt.close(fig)


if __name__ == "__main__":
    apply_figure_style()
    fig_signature()
    fig_decay()
    fig_layers()
    print("wrote results/figures/fig_2x2_signature, fig_cue_conflict_decay, fig_layer_probe (.png/.svg)")
