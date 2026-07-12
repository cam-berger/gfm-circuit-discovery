#!/usr/bin/env python
"""Figure 2 -- the mechanism of same-intron boundary coupling.

  A  Selection collapses under acceptor corruption.  L3H1's attention to the partner falls
     95%; a distance-matched control position never had any.

  B  The read is CONJUNCTIVE.  Attention restored AND coupling restored, side by side --
     because the verbal claim ("K restores selection but not the conclusion") must be SHOWN,
     not asserted.  Key restores 80% of attention but only 0.035 of the coupling; value
     restores 0% of attention; only together (0.711) does the head both FIND the partner and
     READ it.  Interaction I = +0.573 [+0.522, +0.624].

  C  Address and content are SEPARABLE -- the binding result.  Swap L3H1's key/value between
     the true partner and a distance-matched decoy AG.  Key-swap moves attention entirely to
     the decoy; value-swap leaves attention put but still destroys the margin; BOTH-swap
     leaves attention on the decoy yet FULLY RESTORES the margin -- the head reads whatever
     the key routes it to, and the conclusion follows the VALUE.

  D  L2H15 is an upstream ADDRESSING head.  Restoring L3H1's key (not its value) to its
     L2H15-intact value rescues the coupling L2H15 supplies.

Computes nothing; reads manifest-stamped artifacts and hashes them into provenance.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.figures import COLORS, panel_label, save_figure, use_style  # noqa: E402

HERE = Path(__file__).parent
RES = REPO / "results" / "splicebert_coupling"
QKV, COMP, RESC = RES / "qkv.json", RES / "competing_acceptors.json", RES / "l2_rescue.json"


def main() -> None:
    for p in (QKV, COMP, RESC):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    qkv, comp, resc = (json.loads(p.read_text()) for p in (QKV, COMP, RESC))

    use_style()
    fig, ax = plt.subplots(1, 4, figsize=(14.0, 3.4))

    # ---- A: selection collapses under corruption --------------------------------
    a = qkv["attention"]
    names = ["intact", "acceptor\ncorrupted", "L2H15\nablated", "control\nposition"]
    vals = [a["partner_intact"], a["partner_corrupted"], a["partner_ablated"],
            a["control_intact"]]
    cols = [COLORS["signal"], COLORS["ablated"], COLORS["warn"], COLORS["muted"]]
    ax[0].bar(np.arange(4), vals, color=cols, zorder=3, width=0.68)
    ax[0].set_xticks(np.arange(4))
    ax[0].set_xticklabels(names)
    ax[0].set_ylabel("L3H1 attention\n(donor → partner acceptor)")
    ax[0].annotate(f"{100*(1-vals[1]/vals[0]):.0f}% collapse", xy=(1, vals[1]),
                   xytext=(1.35, vals[0] * 0.62), fontsize=7, color=COLORS["ablated"],
                   arrowprops=dict(arrowstyle="->", color=COLORS["ablated"], lw=0.8))
    ax[0].set_title("Corruption destroys SELECTION", pad=8)
    panel_label(ax[0], "A")

    # ---- B: conjunctive read -- attention AND coupling ---------------------------
    kv = qkv["kv"]
    order = ["key@acceptor", "value@acceptor", "key+value@acceptor"]
    lab = ["key\nonly", "value\nonly", "key +\nvalue"]
    cpl = [kv[k]["recovery"] for k in order]
    clo = [kv[k]["lo"] for k in order]
    chi = [kv[k]["hi"] for k in order]
    att = [kv[k]["attention_restored"] for k in order]
    x = np.arange(3)
    w = 0.36
    ax[1].bar(x - w / 2, att, width=w, color=COLORS["accent"], label="attention restored",
              zorder=3)
    ax[1].bar(x + w / 2, cpl, width=w, color=COLORS["signal"], label="coupling restored",
              zorder=3)
    ax[1].errorbar(x + w / 2, cpl, yerr=[np.array(cpl) - clo, np.array(chi) - np.array(cpl)],
                   fmt="none", ecolor=COLORS["text"], elinewidth=0.8, capsize=2, zorder=4)
    ax[1].axhline(0, color=COLORS["text"], lw=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(lab)
    ax[1].set_ylabel("fraction restored")
    ax[1].set_ylim(-0.05, 1.30)
    ax[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=6.5)
    I = qkv["interaction"]
    ax[1].text(0.98, 0.98, f"K×V interaction\nI = {I['I']:+.3f}\n[{I['ci'][0]:.3f}, {I['ci'][1]:.3f}]",
               transform=ax[1].transAxes, fontsize=6.5, ha="right", va="top",
               bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COLORS["muted"], lw=0.6))
    ax[1].set_title("The read is CONJUNCTIVE", pad=8)
    panel_label(ax[1], "B")

    # ---- C: competing acceptors -- address vs content ---------------------------
    C = comp["conditions"]
    conds = ["baseline", "key-swap", "value-swap", "both-swap"]
    short = ["base", "key\nswap", "value\nswap", "both\nswap"]
    at = [C[c]["attn_partner"] for c in conds]
    ad_ = [C[c]["attn_decoy"] for c in conds]
    mg = [C[c]["margin"] for c in conds]
    x = np.arange(4)
    ax[2].bar(x - 0.19, at, width=0.36, color=COLORS["signal"], label="attn → partner", zorder=3)
    ax[2].bar(x + 0.19, ad_, width=0.36, color=COLORS["warn"], label="attn → decoy", zorder=3)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(short)
    ax[2].set_ylabel("L3H1 attention")
    ax[2].set_ylim(0, 0.82)
    ax[2].legend(loc="upper left", fontsize=6, ncol=2, columnspacing=0.8,
                 bbox_to_anchor=(-0.02, 1.02))
    ax2 = ax[2].twinx()
    ax2.plot(x, mg, "o--", color=COLORS["text"], markersize=5, lw=1.4, zorder=5)
    ax2.axhline(comp["corrupt_margin"], color=COLORS["muted"], lw=0.8, ls=":")
    ax2.text(3.42, comp["corrupt_margin"], "partner\ndestroyed", fontsize=5.5,
             color=COLORS["muted"], va="center")
    # label the line directly -- a second legend collides with the first
    ax2.annotate("donor margin", xy=(0, mg[0]), xytext=(0.18, mg[0] + 0.20),
                 fontsize=6.5, color=COLORS["text"],
                 arrowprops=dict(arrowstyle="-", color=COLORS["text"], lw=0.7))
    ax2.text(3.0, mg[3] + 0.10, "fully\nrestored", fontsize=6, ha="center",
             color=COLORS["accent"], fontweight="bold")
    ax2.set_ylabel("donor logit margin", color=COLORS["text"])
    ax2.grid(False)
    ax2.set_ylim(comp["corrupt_margin"] - 0.12, comp["clean_margin"] + 0.55)
    ax[2].set_title("ADDRESS vs CONTENT are separable", pad=8)
    panel_label(ax[2], "C")

    # ---- D: L2H15 rescue ---------------------------------------------------------
    r = resc["rescue"]
    keys = ["K only", "V only", "K+V", "MLP2 write", "K+V @control-pos"]
    lab = ["key", "value", "key+\nvalue", "MLP2\nwrite", "control\npos"]
    v = [r[k]["rescue"] for k in keys]
    lo = [r[k]["lo"] for k in keys]
    hi = [r[k]["hi"] for k in keys]
    cols = [COLORS["signal"], COLORS["muted"], COLORS["signal"], COLORS["accent"],
            COLORS["muted"]]
    x = np.arange(len(keys))
    ax[3].bar(x, v, color=cols, zorder=3, width=0.66)
    ax[3].errorbar(x, v, yerr=[np.array(v) - lo, np.array(hi) - np.array(v)], fmt="none",
                   ecolor=COLORS["text"], elinewidth=0.8, capsize=2, zorder=4)
    ax[3].axhline(0, color=COLORS["text"], lw=0.8)
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(lab)
    ax[3].set_ylabel("coupling rescued\n(fraction L2H15 supplies)")
    ax[3].set_title("L2H15 builds the ADDRESS", pad=8)
    panel_label(ax[3], "D")

    fig.tight_layout(w_pad=2.4)
    out = save_figure(fig, HERE / "fig2_mechanism", sources=[QKV, COMP, RESC])
    print(f"wrote {out}.png / .pdf  (+ provenance)")


if __name__ == "__main__":
    main()
