#!/usr/bin/env python
"""Figure 3 -- what the circuit actually is, and where it stops.

  A  WHERE L3H1 WRITES.  Its causal contribution lives in 11 of 1026 tokens -- the donor and
     its +/-5 flank.  The acceptor, the intron body, the flanking exons: all zero.  The head
     READS at the acceptor (Fig 2) and WRITES at the 5' splice site.  That is what a
     content-addressed read looks like when you measure it instead of assuming it.

  B  HOW IT REACHES THE READOUT.  Freeze every downstream block and L3H1's write is worth
     0.022 -- it is not a logit, it is a signal.  No single block mediates it; the recovery
     accumulates only as the whole stack is released.  The readout is a distributed chain,
     not a head we failed to find.

  C  L2H15 -> L3H1 IS A CLOSED EDGE.  Restoring L3H1's output rescues 1.05 [0.97, 1.16] of
     L2H15's effect: nothing of L2H15 reaches the behaviour except through L3H1.  The earlier
     "~60% unaccounted" was an artefact of only ever patching a 4-token window.

  D  THE LIMIT.  Ablating everything outside the circuit destroys the coupling -- but so does
     ablating the same NUMBER of random heads.  Against a size- and health-matched null the
     test has no power: at matched health the circuit-preserving ablation is GENTLER than
     random.  This panel exists so the completeness claim is not made.

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
POS = RES / "position_profile.json"
PATHJ = RES / "readout_path.json"
ACCT = RES / "l2_accounting.json"
NULLJ = RES / "completeness_null.json"


def main() -> None:
    for p in (POS, PATHJ, ACCT, NULLJ):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    pos, pth, acct, nul = (json.loads(p.read_text()) for p in (POS, PATHJ, ACCT, NULLJ))

    use_style()
    fig, ax = plt.subplots(1, 4, figsize=(15.0, 3.5))

    # ---- A: where L3H1 writes -------------------------------------------------------
    order = ["donor", "donor flank", "acceptor", "intron body", "upstream exon", "downstream"]
    lab = ["donor\n(1 nt)", "donor\nflank\n(10)", "acceptor\n(11)", "intron\nbody\n(87)",
           "upstream\nexon\n(309)", "down-\nstream\n(607)"]
    R = pos["regions"]
    v = [R[k]["recovery"] for k in order]
    lo = [R[k]["lo"] for k in order]
    hi = [R[k]["hi"] for k in order]
    cols = [COLORS["signal"], COLORS["signal"]] + [COLORS["muted"]] * 4
    x = np.arange(len(order))
    ax[0].bar(x, v, color=cols, zorder=3, width=0.7)
    ax[0].errorbar(x, v, yerr=[np.array(v) - lo, np.array(hi) - np.array(v)], fmt="none",
                   ecolor=COLORS["text"], elinewidth=0.8, capsize=2, zorder=4)
    ax[0].axhline(R["ALL positions"]["recovery"], color=COLORS["accent"], ls="--", lw=1.0,
                  zorder=2)
    ax[0].text(5.4, R["ALL positions"]["recovery"] + 0.02, "all 1026", fontsize=6,
               color=COLORS["accent"], ha="right")
    ax[0].axhline(0, color=COLORS["text"], lw=0.8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(lab, fontsize=6)
    ax[0].set_ylabel("coupling restored\n(patch L3H1 here only)")
    ax[0].set_title("L3H1 WRITES at the 5′ splice site", pad=8)
    ax[0].annotate("11 of 1026 tokens\ncarry the whole effect", xy=(0.5, v[1]),
                   xytext=(1.9, 0.60), fontsize=6.5, color=COLORS["signal"],
                   arrowprops=dict(arrowstyle="->", color=COLORS["signal"], lw=0.8))
    panel_label(ax[0], "A")

    # ---- B: the downstream chain ----------------------------------------------------
    cum = pth["mediation_cumulative"]
    keys = list(cum)
    short = ["direct\n(all frozen)", "+MLP3", "+L4\nattn", "+MLP4", "+L5\nattn", "all live"]
    y = [cum[k] for k in keys]
    x = np.arange(len(y))
    ax[1].plot(x, y, "o-", color=COLORS["signal"], markersize=5, zorder=4)
    ax[1].fill_between(x, 0, y, color=COLORS["signal"], alpha=0.10, zorder=2)
    ax[1].axhline(0, color=COLORS["text"], lw=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(short[:len(y)], fontsize=6)
    ax[1].set_ylabel("coupling restored\n(patch L3H1, free the stack)")
    ax[1].set_ylim(-0.03, 0.95)
    ax[1].annotate(f"direct effect\n{y[0]:.3f}", xy=(0, y[0]), xytext=(0.35, 0.28),
                   fontsize=6.5, color=COLORS["ablated"],
                   arrowprops=dict(arrowstyle="->", color=COLORS["ablated"], lw=0.8))
    ax[1].set_title("The readout is a CHAIN, not a head", pad=8)
    panel_label(ax[1], "B")

    # ---- C: L2H15 -> L3H1 closed ----------------------------------------------------
    rk = ["L3H1 key @acceptor", "L3H1 key+value @all-pos", "L3H1 query @donor",
          "L3H1 head output @all-pos", "ALL layer-3 heads @all-pos",
          "layer-2 output (CALIBRATION)"]
    rlab = ["key\n@acceptor", "key+value\n@all-pos", "query\n@donor", "L3H1 whole\noutput",
            "all L3\nheads", "layer-2 out\n(calibration)"]
    A = acct["rescue"]
    v = [A[k]["rescue"] for k in rk]
    lo = [A[k]["lo"] for k in rk]
    hi = [A[k]["hi"] for k in rk]
    cols = [COLORS["muted"]] * 3 + [COLORS["signal"], COLORS["signal"], COLORS["accent"]]
    x = np.arange(len(rk))
    ax[2].bar(x, v, color=cols, zorder=3, width=0.7)
    ax[2].errorbar(x, v, yerr=[np.array(v) - lo, np.array(hi) - np.array(v)], fmt="none",
                   ecolor=COLORS["text"], elinewidth=0.8, capsize=2, zorder=4)
    ax[2].axhline(1.0, color=COLORS["text"], ls=":", lw=0.9, zorder=2)
    ax[2].text(5.45, 1.03, "100%", fontsize=6, ha="right", color=COLORS["text"])
    ax[2].axhline(0, color=COLORS["text"], lw=0.8)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(rlab, fontsize=6)
    ax[2].set_ylabel("fraction of L2H15's effect\nrescued")
    ax[2].set_ylim(0, 1.35)
    ax[2].set_title("L2H15 → L3H1 is a CLOSED edge", pad=8)
    panel_label(ax[2], "C")

    # ---- D: the completeness test has no power --------------------------------------
    S = nul["sets"]
    names = list(S)
    short = [n.split(" (")[0].replace("L3-others + ", "L3o+") for n in names]
    obs = [S[n]["observed"] for n in names]
    nm = [S[n]["null_mean"] for n in names]
    ns = [S[n]["null_std"] for n in names]
    hm = [S[n]["health_matched_coupling"] or 0.0 for n in names]
    x = np.arange(len(names))
    w = 0.26
    ax[3].bar(x - w, obs, width=w, color=COLORS["signal"], label="ablate outside C", zorder=3)
    ax[3].bar(x, nm, width=w, color=COLORS["muted"], label="size-matched random", zorder=3)
    ax[3].errorbar(x, nm, yerr=ns, fmt="none", ecolor=COLORS["text"], elinewidth=0.8,
                   capsize=2, zorder=4)
    ax[3].bar(x + w, hm, width=w, color=COLORS["warn"], label="health-matched random",
              zorder=3)
    ax[3].axhline(nul["baseline_coupling"], color=COLORS["accent"], ls="--", lw=1.0)
    ax[3].text(2.45, nul["baseline_coupling"] + 0.03, "intact", fontsize=6, ha="right",
               color=COLORS["accent"])
    ax[3].axhline(0, color=COLORS["text"], lw=0.8)
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(short, fontsize=6)
    ax[3].set_ylabel("coupling retained")
    ax[3].legend(loc="upper right", fontsize=6)
    for i, n in enumerate(names):
        ax[3].text(i - w, obs[i] + 0.04, f"p={S[n]['p_add_one']:.2f}", fontsize=5.5,
                   ha="center", color=COLORS["text"])
    ax[3].set_title("Completeness: NO POWER", pad=8)
    panel_label(ax[3], "D")

    fig.tight_layout(w_pad=2.2)
    out = save_figure(fig, HERE / "fig3_circuit", sources=[POS, PATHJ, ACCT, NULLJ])
    print(f"wrote {out}.png / .pdf  (+ provenance)")


if __name__ == "__main__":
    main()
