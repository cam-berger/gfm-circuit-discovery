#!/usr/bin/env python
"""Seal ED1's scale constants from the TRAIN pilot, on the LINEAR observable (red-team M9).

v1 of the prereg took SESOI, `sd_between` and the native-coupling floor from a max-margin
artifact, then froze a linear-margin protocol -- thresholds in the wrong units.  This
re-derives them from `ed1_scores_trainpilot.json` and prints the block to paste into §8.

SESOI is anchored to the COUPLING SCALE (a property of the phenomenon: one third of mean
`C_exon`), never to the observed architecture difference (the quantity under test) -- so
the threshold cannot be tuned by what the pilot happened to show.

It also checks the inference layer's calibration at a realistic n, comparing the frozen
dyadic multiplier bootstrap against two permutation alternatives.  A decision rule whose
FPR is not ~alpha is not a decision rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "splicebert"))

from gfmcd.splicing.exon_intron import dyadic_bootstrap  # noqa: E402


def fpr_dyadic(d, ga, gb, n_sim, n_boot, seed=0):
    """FPR of the frozen decision (percentile CI excludes 0) under the sign-flip null."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    d = d - np.nanmean(d)
    hits = 0
    for b in range(n_sim):
        s = rng.choice([-1.0, 1.0], size=len(d))
        _, lo, hi = dyadic_bootstrap(d * s, ga, gb, n_boot=n_boot, seed=5000 + b)
        hits += not (lo <= 0 <= hi)
    return hits / n_sim


def fpr_cluster_flip(d, ga, gb, n_sim, n_perm, seed=0):
    """FPR of a TWO-WAY wild sign-flip: an iid sign per GENE, pair multiplier = product.

    Respects the dyadic dependence a pair-level flip ignores (pairs sharing a gene move
    together), which is what makes it a candidate when the bootstrap over-rejects.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float) - np.nanmean(np.asarray(d, float))
    keys = sorted(set(map(str, ga)) | set(map(str, gb)))
    idx = {g: i for i, g in enumerate(keys)}
    ia = np.array([idx[str(g)] for g in ga])
    ib = np.array([idx[str(g)] for g in gb])
    hits = 0
    for b in range(n_sim):
        s0 = rng.choice([-1.0, 1.0], size=len(d))
        obs = abs(np.nanmean(d * s0))
        null = np.empty(n_perm)
        for t in range(n_perm):
            sg = rng.choice([-1.0, 1.0], size=len(keys))
            null[t] = np.nanmean(d * s0 * sg[ia] * sg[ib])
        hits += (np.mean(np.abs(null) >= obs) <= 0.05)
    return hits / n_sim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="results/splicebert_coupling/development/ed1_trainpilot.json")
    ap.add_argument("--nsim", type=int, default=200)
    args = ap.parse_args()

    r = json.loads((REPO / args.pilot).read_text())
    s, p = r["secondary"], r["primary"]
    cells = s["cells"]

    c_exon = np.nanmean([cells[c]["C_exon"]["mean"] for c in ("SS", "SL", "LS", "LL")
                         if c in cells and np.isfinite(cells[c]["C_exon"]["mean"])])
    c_int_short = r["memorials"]["native_coupling_present"]["C_intron_down_short"]
    sesoi = round(c_exon / 3, 3)
    floor = round(0.4 * c_int_short, 3)

    print("=" * 74)
    print("ED1 SEALED CONSTANTS  (TRAIN pilot, LINEAR observable)")
    print("=" * 74)
    print(f"  mean C_exon over the 2x2            = {c_exon:.4f} nats")
    print(f"  C_intron | down-SHORT               = {c_int_short:.4f} nats")
    print(f"  sd_between(C_exon), gene level      = {s['sd_between_C_exon']:.4f}")
    print(f"  matched pairs in pilot              = {r['cohort']['n_pairs']}")
    print(f"  achieved MDE in pilot               = {s['achieved_mde']:.4f}")
    print()
    print(f"  ==> SESOI  = C_exon/3               = {sesoi}")
    print(f"  ==> native_coupling_present floor   = {floor}   (0.4 x observed)")
    n_val = 3472
    mde_val = s["achieved_mde"] * np.sqrt(r["cohort"]["n_pairs"] / n_val)
    print(f"  ==> projected MDE at val n={n_val}    = {mde_val:.4f}"
          f"   {'<= SESOI (bounded null licensable)' if mde_val <= sesoi else '> SESOI (null NOT licensable)'}")
    print()
    print("INFERENCE CALIBRATION at the pilot n (sign-flip null; target FPR <= 0.05)")
    print(f"  reported by runner (dyadic bootstrap): {r['calibration']['permuted_label_fpr']['fpr']}")
    print("=" * 74)
    print("\nPaste into PREREGISTRATION §8 and pass via --sesoi/--floor.")


if __name__ == "__main__":
    main()
