#!/usr/bin/env python
"""IS THE ADDRESS CONTENT, OR DISTANCE?  (the experiment that replaces the retracted D9/D16)

Two binding experiments have now failed for the same underlying reason: they were
mathematically incapable of failing.

  R7  SWAPPING L3H1's (key,value) pair between the partner and a decoy is an algebraic
      identity -- transposing (k,v) PAIRS permutes the summation index of
      out_i = sum_j softmax(q_i.k_j) v_j, so the output is unchanged EXACTLY.
  D16 COPYING the pair to another slot is barely better: BERT's positions are absolute, the
      attention score q_i.k_j has no term in the slot index j, so a (k,v) pair carried to a
      new slot attracts EXACTLY the attention it attracted at home (verified: 9.9e-07).
      "Relocation costs nothing" is architecture, not biology.  RETRACTED.

The flaw both share: they move the KEY, and the key already contains the position embedding of
its origin.  You cannot dissociate content from position by transporting an object that
carries both.

THIS EXPERIMENT DOES NOT MOVE A KEY.  It moves the SEQUENCE, and lets the model compute a
fresh key at the new location -- so the transplanted acceptor carries the position code of
where it now IS, not of where it came from.

  1. destroy the true partner's AG (the standard corruption) -> the coupling collapses
  2. TRANSPLANT the partner's real acceptor context (a `--span`-nt window ending in its AG)
     into the sequence at a range of DISTANCES from the donor
  3. measure how much of the donor's margin comes back, as a function of distance

    recovery FLAT in distance      -> the address is CONTENT.  The head finds an acceptor
                                      wherever it is.
    recovery PEAKED near the true  -> the address is CONTENT *and* DISTANCE.  The head looks
    partner distance                 for an acceptor in a window, i.e. it scans.
    recovery ~0 everywhere         -> the head needs the ORIGINAL site, not a copy of it.

CONTROL: transplant a SHUFFLED version of the same window (same bases, same length, no AG) at
the same distances.  If that also "recovers", the effect is the insertion, not the acceptor.

Every cell can be zero.  That is the point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot_ratio,
    load_adapter, load_examples, pick_device, ratio_of_means,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402

DISTANCES = [40, 60, 80, 100, 120, 150, 200, 300, 450]   # nt, donor -> transplanted AG


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--span", type=int, default=25, help="acceptor context transplanted, nt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/address_transplant.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev)
    task = CouplingTask(ad, batch=args.batch)
    rng = random.Random(0)

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                         tag=f"transplant-{args.split}", min_effect=0.3)
    genes = [x.gene_id for x in X]
    print(f"{len(X)} {args.split} exons   span={args.span}nt   device={dev}\n")

    def acceptor_context(x):
        """The real acceptor context: `span` nt ending at the partner's AG (inclusive)."""
        a = x.next_acceptor_pos
        return x.sequence[a - args.span:a]          # ends with ...AG

    def transplant(seq, ctx, at):
        """Write `ctx` so that its final AG lands at [at-2, at-1]."""
        lo = at - len(ctx)
        if lo < 4 or at + 2 >= len(seq):
            return None
        return seq[:lo] + ctx + seq[at:]

    num = {(d, k): [] for d in DISTANCES for k in ("acceptor", "shuffled")}
    dnm: list[float] = []
    keep_idx: dict = {(d, k): [] for d in DISTANCES for k in ("acceptor", "shuffled")}

    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            dn = [x.donor_pos for x in ch]
            clean = [x.sequence for x in ch]
            corrupt = [substitute(x.sequence, x.next_acceptor_pos - 2, CORRUPT_ALT) for x in ch]

            m_cl = task.margin(ad.model(**task.encode(clean, dn)).logits.float(), dn)
            m_co = task.margin(ad.model(**task.encode(corrupt, dn)).logits.float(), dn)
            dnm += (m_cl - m_co).tolist()

            for d in DISTANCES:
                for kind in ("acceptor", "shuffled"):
                    seqs, idx = [], []
                    for j, x in enumerate(ch):
                        ctx = acceptor_context(x)
                        if kind == "shuffled":
                            c = list(ctx)
                            rng.shuffle(c)
                            ctx = "".join(c)
                            if ctx[-2:] == "AG":            # must NOT be an acceptor
                                ctx = ctx[:-2] + "CC"
                        at = x.donor_pos + d
                        # never land on, or adjacent to, the true partner site
                        if abs(at - x.next_acceptor_pos) <= args.span:
                            continue
                        s = transplant(corrupt[j], ctx, at)
                        if s is None or len(s) != len(x.sequence):
                            continue
                        seqs.append(s)
                        idx.append(i + j)
                    if not seqs:
                        continue
                    sub_dn = [X[k].donor_pos for k in idx]
                    m = task.margin(ad.model(**task.encode(seqs, sub_dn)).logits.float(), sub_dn)
                    base_co = [m_co[k - i].item() for k in idx]
                    num[(d, kind)] += (m.cpu().numpy() - np.array(base_co)).tolist()
                    keep_idx[(d, kind)] += idx
            print(f"  {min(i+task.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    dnm = np.asarray(dnm)
    print("=" * 82)
    print("ACCEPTOR TRANSPLANT -- recovery of the donor margin vs the transplant's DISTANCE")
    print(f"  (the true partner sits at {np.mean([x.next_acceptor_pos - x.donor_pos for x in X]):.0f}nt"
          f"; its AG is destroyed in every row below)")
    print(f"\n  {'distance':>9s} {'real acceptor':>26s} {'shuffled control':>26s}   n")
    R: dict = {"span": args.span, "n": len(X),
               "true_distance_mean": float(np.mean([x.next_acceptor_pos - x.donor_pos for x in X])),
               "rows": []}
    for d in DISTANCES:
        row = {"distance": d}
        cells = []
        for kind in ("acceptor", "shuffled"):
            idx = keep_idx[(d, kind)]
            if not idx:
                cells.append("        --        ")
                continue
            den = dnm[idx]
            g = [genes[k] for k in idx]
            r = ratio_of_means(num[(d, kind)], den)
            lo, hi = gboot_ratio(num[(d, kind)], den, g)
            row[kind] = dict(recovery=r, lo=lo, hi=hi, n=len(idx))
            cells.append(f"{r:7.3f} [{lo:+.3f},{hi:+.3f}]")
        n = len(keep_idx[(d, 'acceptor')])
        print(f"  {d:7d}nt {cells[0]:>26s} {cells[1]:>26s}  {n:3d}")
        R["rows"].append(row)

    ok = [r for r in R["rows"] if "acceptor" in r]
    if ok:
        best = max(ok, key=lambda r: r["acceptor"]["recovery"])
        near = [r for r in ok if abs(r["distance"] - R["true_distance_mean"]) <= 40]
        far = [r for r in ok if r["distance"] >= 300]
        mn = np.mean([r["acceptor"]["recovery"] for r in near]) if near else float("nan")
        mf = np.mean([r["acceptor"]["recovery"] for r in far]) if far else float("nan")
        print(f"\n  peak recovery at {best['distance']}nt: {best['acceptor']['recovery']:.3f}")
        print(f"  near the true partner distance: {mn:.3f}    far (>=300nt): {mf:.3f}")
        R["near_recovery"], R["far_recovery"] = float(mn), float(mf)
        if mn > 0.15 and mn > 2 * max(mf, 0.01):
            print("  => the address is CONTENT *and* DISTANCE.  A transplanted acceptor is read")
            print("     when it sits where a partner should be, and ignored when it does not.")
            print("     The head SCANS a window; it is not a position-free content lookup.")
        elif mn > 0.15 and mf > 0.5 * mn:
            print("  => the address is CONTENT.  A transplanted acceptor is read wherever it is.")
        else:
            print("  => a transplanted acceptor is NOT read.  The head needs the original site,")
            print("     not a copy of its sequence -- the address is not reconstructible from")
            print("     this window alone.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:address_transplant", sys.argv,
        config={"split": args.split, "n": len(X), "span": args.span, "distances": DISTANCES},
        notes=["Replaces the retracted D9 (swap = identity) and D16 (copy = architecture).",
               "This moves the SEQUENCE, not the key, so the transplant carries the position "
               "code of where it now is -- which is what makes content/position separable.",
               "Shuffled-context control at every distance; every cell can be zero."],
    ).write(out.parent / "address_transplant.manifest.json")
    print("=" * 82)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
