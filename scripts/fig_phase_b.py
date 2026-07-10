"""Phase B figures. Read the dumps; never re-derive a number by hand.

RETRACTED (D35): `fig_signature` and `fig_decay` both plotted a cue-conflict anchor share read from a
probe at `L2.resid_post`, a layer that does not carry the anchor in cells B, C, or D. They reported the
wrong mechanism for three of five models. Both are removed; their replacements, scored against the
model's own output, are `fig_2x2_corrected` and `fig_decay_output` in `scripts/fig_phase_c.py`.
`fig_layer_probe` is unaffected -- it plots phase decodability by depth, which is a claim about a
representation and is correctly measured by a probe.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from gfmcd.figstyle import BAG, COUNTER, INK, INK_2, INK_MUTED, RELAY, apply_figure_style
from gfmcd.paths import project_paths

DATA = project_paths().results / "reading-frame/fig_data"
OUT = project_paths().results / "reading-frame/figures"
MECH = {"counter": COUNTER, "relay": RELAY, "bag": BAG}
SHORT = {
    "A · uniform, easy detector": "A  uniform, easy",
    "B · uniform, hard detector": "B  uniform, hard",
    "C · real CDS, easy detector": "C  real CDS, easy",
    "D · real CDS, natural": "D  real CDS, natural",
    "μ=0 · real CDS, LM objective": "μ=0  real CDS, LM",
}



def fig_layers():
    layers = json.loads((DATA / "layer_probe.json").read_text())
    sigs = {s["model"]: s["mechanism"] for s in json.loads((DATA / "signature_corrected.json").read_text())}
    names = ["resid.embed"] + [f"L{i}.resid_post" for i in range(6)]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))

    # C and D land within 0.03 of the counter at L5; separate the end-labels.
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
    ax.set_title("On real DNA the pooled route is built first, and then overruled.",
                 color=INK, loc="left", pad=22)
    ax.text(0, 1.035, "Cells C and D reach the anchor-free ceiling by L2 without the anchor, which enters at L4. "
                      "The output follows the anchor (D35).",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    # Draw the ceiling only across the data, so it does not run under the right-margin labels.
    ax.plot([0, len(names) - 1], [0.9509, 0.9509], color=RELAY, lw=0.8, ls=(0, (5, 3)), alpha=0.7, zorder=1)
    ax.text(0.05, 0.962, "anchor-free ceiling 0.9509", color=RELAY, fontsize=8)

    fig.savefig(OUT / "fig_layer_probe.png")
    fig.savefig(OUT / "fig_layer_probe.svg")
    plt.close(fig)


if __name__ == "__main__":
    apply_figure_style()
    fig_layers()
    print("wrote results/reading-frame/figures/fig_layer_probe (.png/.svg)")
