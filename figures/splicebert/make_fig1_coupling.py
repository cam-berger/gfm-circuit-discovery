#!/usr/bin/env python
"""Figure 1 -- same-intron boundary coupling and its causal localisation to L3H1.

  A  Causal analogue of the SpliceBERT attention figure.  Same pair categories, but the
     y-axis is CAUSAL COUPLING, not attention.  Intact vs L3H1-ablated.
     Message: same-intron coupling collapses under L3H1 ablation; same-exon coupling and
     every control are untouched.  The attention analysis showed the association exists;
     this shows which association the model actually USES, and that one head carries it.

  B  Mechanistic localisation.  Held-out 96-head denoise scan, ranked.
     Message: L3H1 is #1 of 96, ~4x the next head, and its held-out recovery matches its
     discovery-set recovery -- so it is not a cherry-pick.

  C  The computation.  2x2 factorial as an interaction plot.
     Message: the lines are NOT parallel.  L3H1's contribution CHANGES SIGN with partner
     status: +0.910 when the partner is intact, -0.200 when it is corrupted.  The head
     supplies signed evidence for or against a donor given its same-intron partner.

This script computes NOTHING.  It reads manifest-stamped artifacts and hashes them into
<name>.provenance.json.  If a number here is not in an artifact, that is a bug.
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

from gfmcd.figures import (  # noqa: E402
    COLORS, grouped_bars_ci, interaction_plot, panel_label, ranked_dots, save_figure,
    use_style,
)

HERE = Path(__file__).parent
CATS = HERE / "data" / "pair_categories.json"
CONF = REPO / "results" / "splicebert_coupling" / "confirmation.json"


def main() -> None:
    if not CONF.exists():
        raise SystemExit(f"missing {CONF} -- run scripts/splicebert/run_confirm.py first")
    cats = json.loads(CATS.read_text())["categories"]
    conf = json.loads(CONF.read_text())

    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))

    # ---- A: causal coupling by pair category ------------------------------------
    names = list(cats)
    intact = [cats[c]["intact"] for c in names]
    abl = [cats[c]["L3H1 ablated"] for c in names]
    grouped_bars_ci(
        axes[0], [n.replace("\n", "\n") for n in names],
        {"SpliceBERT intact": ([r["mean"] for r in intact],
                               [r["lo"] for r in intact], [r["hi"] for r in intact]),
         "L3H1 ablated": ([r["mean"] for r in abl],
                          [r["lo"] for r in abl], [r["hi"] for r in abl])},
        ylabel="causal coupling\n(Δ donor logit margin)",
        colors={"SpliceBERT intact": COLORS["signal"], "L3H1 ablated": COLORS["ablated"]},
    )
    axes[0].set_title("Which association does the model USE?", pad=8)
    panel_label(axes[0], "A")

    # ---- B: held-out 96-head scan -------------------------------------------------
    scan = conf["scan_top"]
    # the artifact stores the top-k; pad the rest of the population at ~0 for context
    vals = [s["recovery"] for s in scan]
    labels = [f"L{s['layer']}H{s['head']}" for s in scan]
    ax = axes[1]
    xs = np.arange(len(vals))
    cols = [COLORS["signal"] if labels[i] == "L3H1" else COLORS["muted"]
            for i in range(len(vals))]
    ax.scatter(xs, vals, c=cols, s=30, zorder=3)
    ax.axhline(0, color=COLORS["text"], lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("effect recovered by\nsingle-head denoising")
    top = vals[0]
    ax.annotate(f"#1 of 96\n{top:.3f}", xy=(0, top), xytext=(1.2, top * 0.92),
                fontsize=7, color=COLORS["signal"],
                arrowprops=dict(arrowstyle="->", color=COLORS["signal"], lw=0.8))
    ax.set_title(f"Localisation (held-out, rank #{conf['scan_rank_of_L3H1']}/96)", pad=8)
    panel_label(ax, "B")

    # ---- C: the 2x2 factorial as an interaction plot -------------------------------
    f = conf["factorial"]
    ax = axes[2]
    interaction_plot(
        ax, ["partner\nintact", "partner\ncorrupted"],
        {"L3H1 intact": [f["M_CI"], f["M_XI"]],
         "L3H1 ablated": [f["M_CA"], f["M_XA"]]},
        ylabel="donor logit margin",
    )
    ax.annotate("", xy=(0, f["M_CA"]), xytext=(0, f["M_CI"]),
                arrowprops=dict(arrowstyle="<->", color=COLORS["text"], lw=0.8))
    ax.text(0.07, (f["M_CI"] + f["M_CA"]) / 2,
            f"+{f['M_CI']-f['M_CA']:.3f}\nhead SUPPORTS\nthe donor",
            fontsize=6.5, va="center", ha="left")
    ax.annotate("", xy=(1, f["M_XA"]), xytext=(1, f["M_XI"]),
                arrowprops=dict(arrowstyle="<->", color=COLORS["text"], lw=0.8))
    ax.text(0.93, (f["M_XI"] + f["M_XA"]) / 2,
            f"{f['M_XI']-f['M_XA']:.3f}\nhead OPPOSES\nthe donor",
            fontsize=6.5, va="center", ha="right")
    # headroom so the annotations never collide with the axis labels
    ys = [f["M_CI"], f["M_XI"], f["M_CA"], f["M_XA"]]
    ax.set_ylim(min(ys) - 0.18, max(ys) + 0.18)
    did, ci = f["did"], f["did_ci"]
    ax.set_title(f"Signed partner-consistency\nDiD = {did:+.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", pad=8)
    panel_label(ax, "C")

    fig.tight_layout(w_pad=2.2)
    out = save_figure(fig, HERE / "fig1_coupling", sources=[CATS, CONF])
    print(f"wrote {out}.png / .pdf  (+ provenance)")


if __name__ == "__main__":
    main()
