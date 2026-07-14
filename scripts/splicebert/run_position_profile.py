#!/usr/bin/env python
"""WHERE does L3H1 write the partner information that matters?

The path decomposition split L3H1's contribution by position and found something the
"the donor asks about its partner" story does not predict:

    patch L3H1 at the DONOR only          0.359
    patch L3H1 at every OTHER position    0.537     <- LARGER
    patch L3H1 everywhere                 0.823

and destroying the donor's query collapses donor->partner attention by 95% while costing
only 29% of the coupling.  So the donor's own read of its partner is real, but it is at most
a third of the mechanism.  The majority of L3H1's causal contribution is written at positions
that are NOT the readout position -- and can therefore only matter by being picked up later.

This script asks WHICH positions, by patching L3H1's clean output into the corrupt run inside
one region at a time.  Regions are defined relative to the two sites, so they mean the same
thing in every example:

    donor          [d]                      the masked readout position itself
    donor flank    [d-5, d+5] \ {d}         the 5' splice-site neighbourhood
    acceptor       [a-5, a+5]               the partner AG and its neighbourhood
    intron body    (d+5, a-5)               everything between them
    upstream exon  [0, d-5)                 5' of the donor
    downstream     (a+5, end]               3' of the acceptor

The regions partition the sequence, so the recoveries are directly comparable and their
sum can be checked against the all-positions total (super-additivity is itself a result).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask, capture_head_outputs,
    gboot_ratio, load_adapter, load_examples, patch_heads, pick_device, ratio_of_means,
)

REGIONS = ["donor", "donor flank", "acceptor", "intron body", "upstream exon",
           "downstream", "ALL positions"]


def region_masks(ex, off: int, S: int, device):
    """Boolean [S] mask per region.  The six regions PARTITION the sequence."""
    d, a = ex.donor_pos + off, ex.next_acceptor_pos + off
    idx = torch.arange(S, device=device)
    m = {}
    m["donor"] = idx == d
    m["donor flank"] = (idx >= d - 5) & (idx <= d + 5) & (idx != d)
    m["acceptor"] = (idx >= a - 5) & (idx <= a + 5)
    m["intron body"] = (idx > d + 5) & (idx < a - 5)
    m["upstream exon"] = (idx >= off) & (idx < d - 5)
    m["downstream"] = (idx > a + 5) & (idx < S)
    m["ALL positions"] = torch.ones(S, dtype=torch.bool, device=device)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/position_profile.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev)
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                         tag=f"posprof-{args.split}", min_effect=0.3)
    genes = [x.gene_id for x in X]
    print(f"{len(X)} {args.split} exons   device={dev}")

    num = {r: [] for r in REGIONS}
    dnm: list[float] = []
    sizes = {r: [] for r in REGIONS}

    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            cl, co, dn = task.pair(ch)
            S = cl["input_ids"].shape[1]
            cc = capture_head_outputs(ad, lambda: ad.model(**cl))
            mc = task.margin(ad.model(**cl).logits.float(), dn)
            mk = task.margin(ad.model(**co).logits.float(), dn)
            dnm += (mc - mk).tolist()

            per = [region_masks(x, OFF, S, ad.device) for x in ch]
            for r in REGIONS:
                mask = torch.stack([p[r] for p in per])          # [B,S]
                sizes[r] += mask.sum(-1).tolist()
                with patch_heads(ad, cc, [READER], token_mask=mask):
                    m = task.margin(ad.model(**co).logits.float(), dn)
                num[r] += (m - mk).tolist()
            print(f"  {min(i+task.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    print("\n" + "=" * 76)
    print(f"WHERE L3H1's OUTPUT MATTERS  ({args.split}, n={len(X)})")
    print(f"  {'region':16s} {'tokens':>7s} {'recovery':>9s} {'95% CI':>18s}")
    R: dict = {}
    part = 0.0
    for r in REGIONS:
        v = ratio_of_means(num[r], dnm)
        lo, hi = gboot_ratio(num[r], dnm, genes)
        R[r] = dict(recovery=v, lo=lo, hi=hi, mean_tokens=float(np.mean(sizes[r])))
        if r != "ALL positions":
            part += v
        print(f"  {r:16s} {np.mean(sizes[r]):7.0f} {v:9.3f}  [{lo:+.3f},{hi:+.3f}]")
    tot = R["ALL positions"]["recovery"]
    print(f"\n  sum of the six disjoint regions {part:.3f}   vs   all-at-once {tot:.3f}")
    if part < tot - 0.05:
        print("  SUB-additive: the regions must be patched TOGETHER to work -- the downstream")
        print("  stack integrates L3H1's writes across positions rather than reading any one.")
    elif part > tot + 0.05:
        print("  SUPER-additive: the regions are partly REDUNDANT -- several carry the same")
        print("  partner information and any one of them suffices to restore some of it.")
    else:
        print("  ADDITIVE: each region contributes independently.")
    R["_sum_of_regions"] = float(part)
    R["_all_positions"] = float(tot)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "n": len(X), "regions": R}, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:position_profile", sys.argv,
        config={"split": args.split, "n": len(X), "reader": list(READER)},
        notes=["The six regions partition the sequence, so recoveries are comparable and "
               "their sum is a meaningful check against the all-positions total."],
    ).write(out.parent / "position_profile.manifest.json")
    print("=" * 76)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
