#!/usr/bin/env python
"""THE DECISIVE TEST: is the coupling exon-specific, or just distance decay?

The same donor has TWO partners:
    - its EXON-mate acceptor,   upstream, separated by the EXON   (exon definition)
    - its INTRON-mate acceptor, downstream, separated by the INTRON (intron definition)

Same donor, same model, same MLM readout.  Only the intervening sequence type and the
separation differ.  So we can plot the coupling strength against DISTANCE for three
curves and read the answer straight off:

    (a) true EXON partner      vs distance (= exon length)
    (b) true INTRON partner    vs distance (= intron length)
    (c) NON-partner AG motifs  vs distance   <- the generic distance-decay baseline

  If (a),(b) >> (c) at every distance      -> the coupling is PARTNER-SPECIFIC, not generic.
  If (a) > (b) at MATCHED distance         -> the EXON is a privileged unit  => exon definition.
  If (a) ~ (b) at matched distance         -> coupling is symmetric; the exon is NOT special,
                                              and the length trend is distance decay.

Also fixes the transplant control: the donor acceptor is now STRENGTH-MATCHED on its PWM
score (the earlier version grabbed a random exon, so "coupling survives transplantation"
was confounded with "I installed a weaker acceptor").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.coupling import (  # noqa: E402
    CouplingConfig, CouplingScorer, build_coupling_dataset, destroy_acceptor,
    stable_rng, transplant_acceptor,
)
from gfmcd.splicing.baselines import site_window  # noqa: E402
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402


def gene_bootstrap(vals, genes, n=2000, seed=0):
    """95% CI resampling GENES -- exons of one gene are not independent."""
    rng = np.random.default_rng(seed)
    by: dict = {}
    for v, g in zip(vals, genes):
        if not np.isnan(v):
            by.setdefault(g, []).append(v)
    keys = list(by)
    if not keys:
        return (np.nan, np.nan)
    m = np.empty(n)
    for i in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        m[i] = np.mean([v for j in pick for v in by[keys[j]]])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chroms", nargs="*", default=None)
    ap.add_argument("--gtf", default="/Users/cberg/Research/data/genomes/gencode.v50.basic.annotation.gtf.gz")
    ap.add_argument("--fasta", default="/Users/cberg/Research/data/genomes/hg38.fa")
    ap.add_argument("--model", default="data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
    ap.add_argument("--out", default="results/splicebert_coupling/development/exon_vs_intron.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    mp = args.model if Path(args.model).is_absolute() else str(REPO / args.model)
    scorer = CouplingScorer(AutoModelForMaskedLM.from_pretrained(mp).to(dev),
                            AutoTokenizer.from_pretrained(mp), device=dev, batch_size=args.batch)

    pwm = pickle.load(open(REPO / "colab_results/splicebert_run/baselines/local_pair_baseline.pkl", "rb"))
    acc_strength = lambda s, p: pwm.acceptor_pwm.score(site_window(s, p, "acceptor"))  # noqa: E731

    cfg = CouplingConfig()
    X = build_coupling_dataset(args.gtf, PyfaidxGenome(args.fasta), cfg,
                               allowed_chromosomes=set(args.chroms) if args.chroms else None)
    X = [x for x in X
         if x.split == "train" and x.next_acceptor_pos is not None
         and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"]
    rng = stable_rng("sel", cfg.seed)
    rng.shuffle(X)
    X = X[: args.n]
    print(f"{len(X):,} exons with BOTH partners in-window   device={dev}")

    # index acceptors by PWM strength for the strength-matched transplant
    pool = [(acc_strength(x.sequence, x.acceptor_pos), x) for x in X]
    pool.sort(key=lambda t: t[0])
    strengths = np.array([t[0] for t in pool])

    rows = []
    for i, x in enumerate(X):
        if i % 50 == 0:
            print(f"  {i}/{len(X)}", end="\r", flush=True)
        r = stable_rng(x.example_id, cfg.seed)
        seq, a, d, na = x.sequence, x.acceptor_pos, x.donor_pos, x.next_acceptor_pos
        d_exon, d_intron = d - a, na - d
        clean = scorer._margin([seq], [d], "G")[0]

        def drop(variants):
            if not variants:
                return np.nan
            return float(clean - scorer._margin(variants, [d] * len(variants), "G").mean())

        e_exon = drop(destroy_acceptor(seq, a))          # (a) across the EXON
        e_intron = drop(destroy_acceptor(seq, na))       # (b) across the INTRON

        # (c) generic baseline: non-partner AGs, recorded WITH their distance
        ags = [p for p in x.other_ag if abs(p - a) > 15 and abs(p - na) > 15]
        r.shuffle(ags)
        null_pts = []
        for p in ags[:8]:
            null_pts.append((abs(p - d), drop([destroy_acceptor(seq, p)[0]])))

        # strength-matched transplant of the exon-mate acceptor
        s0 = acc_strength(seq, a)
        j = int(np.argmin(np.abs(strengths - s0)))
        cands = [k for k in range(max(0, j - 5), min(len(pool), j + 6))
                 if pool[k][1].example_id != x.example_id]
        e_tp = np.nan
        if cands:
            src = pool[r.choice(cands)][1]
            try:
                e_tp = drop([transplant_acceptor(seq, a, src.sequence, src.acceptor_pos)])
            except ValueError:
                pass

        rows.append(dict(gene=x.gene_id, exon_len=x.exon_length, d_exon=d_exon,
                         d_intron=d_intron, clean=float(clean), e_exon=e_exon,
                         e_intron=e_intron, e_transplant=float(e_tp),
                         null=null_pts))
    print()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    report(rows)
    print(f"\nsaved -> {out}")


def report(rows) -> None:
    g = [r["gene"] for r in rows]
    ee = np.array([r["e_exon"] for r in rows], float)
    ei = np.array([r["e_intron"] for r in rows], float)
    et = np.array([r["e_transplant"] for r in rows], float)
    null_d = np.array([d for r in rows for d, _ in r["null"]], float)
    null_e = np.array([e for r in rows for _, e in r["null"]], float)

    lo, hi = gene_bootstrap(ee, g)
    print("\n" + "=" * 74)
    print("OVERALL (gene-grouped bootstrap 95% CI)")
    print(f"  couple across the EXON   (true partner) : {np.nanmean(ee):+.3f}  [{lo:+.3f},{hi:+.3f}]")
    lo, hi = gene_bootstrap(ei, g)
    print(f"  couple across the INTRON (true partner) : {np.nanmean(ei):+.3f}  [{lo:+.3f},{hi:+.3f}]")
    lo, hi = gene_bootstrap(et, g)
    print(f"  STRENGTH-MATCHED transplant             : {np.nanmean(et):+.3f}  [{lo:+.3f},{hi:+.3f}]")
    print(f"  non-partner AG (generic baseline)       : {np.nanmean(null_e):+.3f}")

    print("\nEFFECT vs DISTANCE  -- the decisive comparison")
    print(f"  {'distance':12s} {'EXON partner':>16s} {'INTRON partner':>16s} {'non-partner AG':>16s}")
    BINS = [(20, 60), (60, 100), (100, 150), (150, 220), (220, 320), (320, 500)]
    table = []
    for lo_, hi_ in BINS:
        me = ee[(np.array([r["d_exon"] for r in rows]) >= lo_) &
                (np.array([r["d_exon"] for r in rows]) < hi_)]
        mi = ei[(np.array([r["d_intron"] for r in rows]) >= lo_) &
                (np.array([r["d_intron"] for r in rows]) < hi_)]
        mn = null_e[(null_d >= lo_) & (null_d < hi_)]
        f = lambda v: f"{np.nanmean(v):+.3f} (n={len(v)})" if len(v) >= 8 else "  --"  # noqa: E731
        print(f"  {lo_:>4d}-{hi_:<7d} {f(me):>16s} {f(mi):>16s} {f(mn):>16s}")
        if len(me) >= 8 and len(mi) >= 8:
            table.append((lo_, hi_, float(np.nanmean(me)), float(np.nanmean(mi))))

    if table:
        de = np.mean([t[2] for t in table])
        di = np.mean([t[3] for t in table])
        print(f"\n  averaged over DISTANCE-MATCHED bins:  EXON {de:+.3f}   INTRON {di:+.3f}"
              f"   ratio {de/max(di,1e-6):.2f}x")
        print("  ratio >> 1  => the EXON is a privileged unit (exon definition)")
        print("  ratio ~ 1   => symmetric coupling; the length trend is distance decay")
    print("=" * 74)


if __name__ == "__main__":
    main()
