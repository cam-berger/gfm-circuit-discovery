"""Phase C figures. The headline is the retraction: the readout decides the answer.

    fig_conflict_depth   anchor share vs depth, ending at the model's output. D33 read at L2.
    fig_2x2_corrected    the signature scatter with a behavioural y-axis.

    python scripts/fig_phase_c.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from gfmcd.figstyle import BAG, COUNTER, INK, INK_2, INK_MUTED, RELAY, apply_figure_style

DATA = Path("results/reading-frame/fig_data")
OUT = Path("results/reading-frame/figures")
MECH = {"counter": COUNTER, "relay": RELAY, "bag": BAG}
SHORT = {
    "A · uniform, easy detector": "A  uniform, easy",
    "B · uniform, hard detector": "B  uniform, hard",
    "C · real CDS, easy detector": "C  real CDS, easy",
    "D · real CDS, natural": "D  real CDS, natural",
    "μ=0 · real CDS, LM objective": "μ=0  real CDS, LM",
}
LAYERS = ["L0.resid_post", "L1.resid_post", "L2.resid_post", "L3.resid_post", "L4.resid_post", "L5.resid_post"]


def fig_conflict_depth():
    conf = json.loads((DATA / "phase_c_conflict_layers.json").read_text())
    sigs = {s["model"]: s["mechanism"] for s in json.loads((DATA / "signature_corrected.json").read_text())}
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    x = list(range(len(LAYERS)))
    xo = len(LAYERS) + 0.6  # the output, set apart from the residual-stream columns

    ax.axvspan(1.6, 2.4, color=INK_MUTED, alpha=0.09, zorder=0, lw=0)
    ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.axvline((len(LAYERS) - 1 + xo) / 2, color=INK_MUTED, lw=0.8, zorder=1)

    dy = {"A · uniform, easy detector": 5, "B · uniform, hard detector": -1,
          "C · real CDS, easy detector": -6, "D · real CDS, natural": 4,
          "μ=0 · real CDS, LM objective": -6}

    for name, r in conf.items():
        c = MECH[sigs[name]]
        ys = [r["layers"][ln]["anchor"] for ln in LAYERS]
        ax.plot(x, ys, "-o", ms=4.5, color=c, lw=2.0, mec="white", mew=0.8, zorder=3)
        if r["output"]:
            ax.plot([x[-1], xo], [ys[-1], r["output"]["anchor"]], color=c, lw=2.0, ls=(0, (1, 2)), zorder=3)
            ax.plot([xo], [r["output"]["anchor"]], "o", ms=8, color=c, mec="white", mew=1.2, zorder=4)
            end = r["output"]["anchor"]
        else:
            end = ys[-1]
            ax.annotate("no aa head", (x[-1], ys[-1]), textcoords="offset points", xytext=(8, -12),
                        fontsize=7.5, color=INK_MUTED)
        ax.annotate(SHORT[name], (xo, end), textcoords="offset points", xytext=(10, dy[name]),
                    fontsize=8.2, color=c, va="center")

    ax.text(2.0, 0.72, "D33 read\nthe y-axis here", ha="center", va="center", fontsize=7.8, color=INK_MUTED)

    handles = [plt.Line2D([], [], marker="o", ms=7, lw=2, color=MECH[m], label=lbl)
               for m, lbl in (("counter", "counter"), ("relay", "relay"), ("bag", "bag"))]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.90), fontsize=8.2,
              labelcolor=INK_2, framealpha=0)

    ax.set_xticks(x + [xo])
    ax.set_xticklabels(["L0", "L1", "L2", "L3", "L4", "L5", "OUTPUT"])
    ax.set_xlim(-0.4, 9.6)
    ax.set_ylim(-0.04, 1.09)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("where the cue conflict is read")
    ax.set_ylabel("follows the ANCHOR  (far field, ≥64 nt)")
    ax.set_title("Where you read the model decides which mechanism you report.", color=INK, loc="left", pad=24)
    ax.text(0, 1.045, "Same intervention, same models. Cells B and C look anchor-free at L2 and are anchored counters at the output.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig_conflict_depth.{ext}")
    plt.close(fig)


def fig_2x2_corrected():
    sigs = json.loads((DATA / "signature_corrected.json").read_text())
    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    ax.axhline(0.50, color=INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.text(0.735, 0.515, "frozen threshold: counted from the anchor", ha="right", va="bottom",
            fontsize=7.8, color=INK_MUTED)

    place = {
        "A · uniform, easy detector": (-12, -4, "right"),
        "B · uniform, hard detector": (-12, -4, "right"),
        "C · real CDS, easy detector": (-12, -4, "right"),
        "D · real CDS, natural": (12, 0, "left"),
        "μ=0 · real CDS, LM objective": (12, 0, "left"),
    }

    for s in sigs:
        c = MECH[s["mechanism"]]
        ax.plot([s["x"]], [s["y"]], "o", ms=10, color=c, mec="white", mew=1.4, zorder=4)
        ctl = max(s["ctl"], 1.5e-3)
        ax.plot([s["x"] - ctl, s["x"] + ctl], [s["y"], s["y"]], color=c, lw=1.6, alpha=0.6, zorder=3)
        note = "" if s["y_is_output"] else "\nprobe max (no aa head)"
        dx, dy, ha = place[s["model"]]
        ax.annotate(f"{SHORT[s['model']]}{note}", (s["x"], s["y"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8.2, color=c, va="center")

    ax.set_xlim(-0.045, 0.75)
    ax.set_ylim(-0.08, 1.12)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("peak anchor dependence   (max over depth of far-field probe drop, minus decoy-Met control)")
    ax.set_ylabel("anchor share under cue conflict  (≥64 nt)")
    ax.set_title("Two causal axes, one ordering. Only cell D stops counting.", color=INK, loc="left", pad=22)
    ax.text(0, 1.035, "Bars = decoy-Met control. The circuit survives an unreliable anchor, and survives a pooled "
                      "shortcut — but not both.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")

    handles = [plt.Line2D([], [], ls="", marker="o", ms=8, color=MECH[m], label=lbl)
               for m, lbl in (("counter", "counter — counted from the anchor"),
                              ("relay", "relay — anchor used, not counted from"),
                              ("bag", "bag — anchor unused"))]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.995, 0.02), fontsize=8.2, labelcolor=INK_2)

    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig_2x2_corrected.{ext}")
    plt.close(fig)


def fig_decay_output():
    """Replaces Phase B's fig_cue_conflict_decay, which read the same conflict off an L2 probe."""
    conf = json.loads((DATA / "phase_c_conflict_layers.json").read_text())
    sigs = {s["model"]: s["mechanism"] for s in json.loads((DATA / "signature_corrected.json").read_text())}
    fig, ax = plt.subplots(figsize=(7.4, 4.5))

    dy = {"A · uniform, easy detector": 0, "B · uniform, hard detector": 0,
          "C · real CDS, easy detector": 0, "D · real CDS, natural": 0}

    for name, r in conf.items():
        if not r["decay"]:
            continue
        c = MECH[sigs[name]]
        xs = [row[0] for row in r["decay"]]
        ys = [row[1] for row in r["decay"]]
        ax.plot(xs, ys, "-o", ms=5, color=c, lw=2.0, mec="white", mew=0.9)
        ax.annotate(SHORT[name], (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, dy[name]),
                    fontsize=8.2, color=c, va="center")

    ax.axhline(1 / 3, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(9, 1 / 3 + 0.02, "chance 0.333", color=INK_MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([9.5, 23.5, 47.5, 95.5, 191.5])
    ax.set_xticklabels(["4–15", "16–31", "32–63", "64–127", "128–255"])
    ax.set_xlim(7, 560)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel("distance from the inserted nucleotide (nt)")
    ax.set_ylabel("follows the ANCHOR  (model output)")
    ax.set_title("The anchor holds at 255 nt. Distance is not what separates the mechanisms.",
                 color=INK, loc="left", pad=22)
    ax.text(0, 1.035, "Scored against the model's own predictions, not a probe. Each curve is flat: "
                      "whatever the model does, it does at every distance.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")

    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig_decay_output.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    apply_figure_style()
    fig_conflict_depth()
    fig_2x2_corrected()
    fig_decay_output()
    print("wrote results/reading-frame/figures/fig_conflict_depth, fig_2x2_corrected, fig_decay_output (.png/.svg)")
