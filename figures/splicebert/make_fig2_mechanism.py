#!/usr/bin/env python
"""Figure 2 -- the mechanism of same-intron boundary coupling.

  A  Selection collapses under acceptor corruption.  L3H1's attention to the partner falls
     95%; a distance-matched control position never had any.

  B  The read is CONJUNCTIVE.  Attention restored AND coupling restored, side by side --
     because the verbal claim ("K restores selection but not the conclusion") must be SHOWN,
     not asserted.  Key restores 80% of attention but only 0.035 of the coupling; value
     restores 0% of attention; only together (0.711) does the head both FIND the partner and
     READ it.  Interaction I = +0.573 [+0.522, +0.624].

  C  The address is not a COORDINATE -- the RELOCATED read.
     (This panel previously showed a key/value SWAP between the partner and a decoy.  That was
     an algebraic identity: transposing (k,v) PAIRS relabels the summation index of
     out_i = sum_j softmax(q_i.k_j) v_j, so the head's output is unchanged EXACTLY -- it
     reproduced on a randomly-initialised model.  Retracted as R7.)
     The replacement is a ONE-WAY COPY, which can fail: destroy the partner's AG, then copy
     its clean key and/or value somewhere else.  Content with no address is worth 0.004.
     Key+value works at a decoy AG (0.669), at a NON-SITE (0.656), and in place (0.686) --
     relocation costs 2%.  The head finds its partner by CONTENT, wherever the key is put.

  D  L2H15 is an upstream ADDRESSING head, and its edge into L3H1 is CLOSED: restoring L3H1's
     whole output rescues 1.05 of what L2H15 supplies -- nothing of it goes anywhere else.

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
# competing_acceptors.json is RETRACTED (R7 -- the swap is an identity) and is NOT read here.
SEAL, ACCT = RES / "sealed_val.json", RES / "l2_accounting.json"
QKV = RES / "qkv.json"


def main() -> None:
    for p in (SEAL, ACCT, QKV):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    seal, acct, qkv = (json.loads(p.read_text()) for p in (SEAL, ACCT, QKV))

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

    # ---- C: the RELOCATED read -- the address is not a coordinate -----------------
    B = seal["binding"]["conditions"]
    conds = ["value@decoy", "key@control-pos (CONTROL)", "key@decoy",
             "key+value@decoy", "key+value@control-pos", "key+value@partner (reference)"]
    short = ["value\n@decoy", "key\n@non-site\n(CONTROL)", "key\n@decoy",
             "key+value\n@decoy", "key+value\n@non-site", "key+value\nin place\n(ref)"]
    v = [B[c]["recovery"] for c in conds]
    lo = [B[c]["lo"] for c in conds]
    hi = [B[c]["hi"] for c in conds]
    cols = [COLORS["muted"], COLORS["muted"], COLORS["warn"],
            COLORS["signal"], COLORS["signal"], COLORS["accent"]]
    x = np.arange(len(conds))
    ax[2].bar(x, v, color=cols, zorder=3, width=0.68)
    ax[2].errorbar(x, v, yerr=[np.array(v) - lo, np.array(hi) - np.array(v)], fmt="none",
                   ecolor=COLORS["text"], elinewidth=0.8, capsize=2, zorder=4)
    ax[2].axhline(0, color=COLORS["text"], lw=0.8)
    ax[2].axhline(v[-1], color=COLORS["accent"], ls="--", lw=0.9, zorder=2)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(short, fontsize=5.5)
    ax[2].set_ylabel("coupling restored\n(partner's AG destroyed)")
    ax[2].set_ylim(-0.05, 0.85)
    ax[2].annotate("content with no\naddress: worthless", xy=(0, v[0]), xytext=(0.05, 0.30),
                   fontsize=6, color=COLORS["muted"],
                   arrowprops=dict(arrowstyle="->", color=COLORS["muted"], lw=0.8))
    ax[2].annotate("works ANYWHERE\nyou put it", xy=(4, v[4]), xytext=(2.5, 0.79),
                   fontsize=6, color=COLORS["signal"], ha="center",
                   arrowprops=dict(arrowstyle="->", color=COLORS["signal"], lw=0.8))
    ax[2].set_title("The address is not a COORDINATE", pad=8)
    panel_label(ax[2], "C")

    # ---- D: L2H15 -> L3H1 is a CLOSED edge ---------------------------------------
    r = acct["rescue"]
    keys = ["L3H1 key @acceptor", "L3H1 query @donor", "MLP2 write @all-pos",
            "L3H1 head output @all-pos", "ALL layer-3 heads @all-pos"]
    lab = ["L3H1 key\n@acceptor", "L3H1 query\n@donor", "MLP2\nwrite",
           "L3H1 WHOLE\noutput", "all L3\nheads"]
    v = [r[k]["rescue"] for k in keys]
    lo = [r[k]["lo"] for k in keys]
    hi = [r[k]["hi"] for k in keys]
    cols = [COLORS["muted"], COLORS["muted"], COLORS["warn"], COLORS["signal"],
            COLORS["signal"]]
    x = np.arange(len(keys))
    ax[3].bar(x, v, color=cols, zorder=3, width=0.66)
    ax[3].errorbar(x, v, yerr=[np.array(v) - lo, np.array(hi) - np.array(v)], fmt="none",
                   ecolor=COLORS["text"], elinewidth=0.8, capsize=2, zorder=4)
    ax[3].axhline(1.0, color=COLORS["text"], ls=":", lw=0.9, zorder=2)
    ax[3].text(4.45, 1.03, "100%", fontsize=6, ha="right", color=COLORS["text"])
    ax[3].axhline(0, color=COLORS["text"], lw=0.8)
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(lab, fontsize=6)
    ax[3].set_ylabel("fraction of L2H15's effect\nrescued")
    ax[3].set_ylim(0, 1.35)
    ax[3].annotate("nothing of L2H15\nbypasses L3H1", xy=(3, v[3]), xytext=(1.3, 1.22),
                   fontsize=6, color=COLORS["signal"],
                   arrowprops=dict(arrowstyle="->", color=COLORS["signal"], lw=0.8))
    ax[3].set_title("L2H15 → L3H1 is a CLOSED edge", pad=8)
    panel_label(ax[3], "D")

    fig.tight_layout(w_pad=2.4)
    out = save_figure(fig, HERE / "fig2_mechanism", sources=[QKV, SEAL, ACCT])
    print(f"wrote {out}.png / .pdf  (+ provenance)")


if __name__ == "__main__":
    main()
