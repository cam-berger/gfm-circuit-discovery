#!/usr/bin/env python
"""ED1 cohort pilot -- FEASIBILITY & IDENTIFIABILITY only.  No model, no outcome.

Answers the questions that decide whether the ED1 confirmatory design is even runnable,
BEFORE any GPU time is spent:

  1. How many eligible donors are there per split, and how do they fall into the 2x2
     flanking-intron architecture (upstream intron S/L) x (downstream intron S/L)?
  2. Is the CROSS-EXON separation (= exon length, the only distance that enters the
     distance-clean primary estimand) on COMMON SUPPORT across the downstream-intron
     strata?  If not, the primary is not matchable and the design is dead.
  3. What are the pre-matching standardized mean differences on the exogenous genomic
     covariates, and do they survive caliper matching?
  4. Are distance-matched non-partner AG shams available on both sides (the specificity
     control), and at what realized distance mismatch?

Everything here is computed from the annotation + genome only.  ``--split train`` is the
design/pilot split (already contaminated by ``results/splicebert_coupling/development/
exon_vs_intron.json``); ``--split val`` is the confirmatory split and NOTHING outcome-like
is computed for it here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.splicing.coupling import CouplingConfig, build_coupling_dataset  # noqa: E402
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402
from gfmcd.splicing.exon_intron import (  # noqa: E402
    ARCH_LONG_MIN, ARCH_SHORT_MAX, EXACT_KEYS, ShamPolicy, architecture_cell, axis_spec,
    covariate_row, eligible, gc_fraction, match_pairs, sham_candidates,
)

PWM_PKL = "colab_results/splicebert_run/baselines/local_pair_baseline.pkl"


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference -- the covariate-balance diagnostic."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--gtf", default=None)
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--pwm", default=str(REPO.parent.parent.parent / PWM_PKL))
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite-record", action="store_true",
                    help="replace an existing artifact of record (results are immutable by default)")
    args = ap.parse_args()

    cfg = CouplingConfig()
    chroms = {"train": None, "val": list(cfg.val_chromosomes),
              "test": list(cfg.test_chromosomes)}[args.split]
    X = build_coupling_dataset(args.gtf, PyfaidxGenome(args.fasta), cfg,
                               allowed_chromosomes=set(chroms) if chroms else None)
    X = [x for x in X if x.split == args.split]
    print(f"[{args.split}] built {len(X):,} internal-exon windows")

    keep = [x for x in X if eligible(x)]
    print(f"[{args.split}] eligible (both partners in-window, canonical GT donor, "
          f"upstream intron annotated): {len(keep):,}  genes={len({x.gene_id for x in keep}):,}")

    pwm = pickle.load(open(args.pwm, "rb"))
    from gfmcd.splicing.baselines import site_window
    a_score = lambda s, p: pwm.acceptor_pwm.score(site_window(s, p, "acceptor"))  # noqa: E731
    d_score = lambda s, p: pwm.donor_pwm.score(site_window(s, p, "donor"))  # noqa: E731

    pol = ShamPolicy()
    rows = []
    for x in keep:
        cell = architecture_cell(x)
        n_up, dm_up = sham_candidates(x, "exon", pol)
        n_dn, dm_dn = sham_candidates(x, "intron", pol)
        rows.append(dict(
            example_id=x.example_id, gene=x.gene_id, chrom=x.chrom, cell=cell,
            L_exon=x.exon_length, L_down=x.intron_length_down, L_up=x.intron_length_up,
            donor_pos=x.donor_pos, acc_pos=x.acceptor_pos, next_acc=x.next_acceptor_pos,
            s_acc=a_score(x.sequence, x.acceptor_pos),
            s_next=a_score(x.sequence, x.next_acceptor_pos),
            s_don=d_score(x.sequence, x.donor_pos),
            gc_exon=gc_fraction(x.sequence[x.acceptor_pos:x.donor_pos]),
            gc_intron=gc_fraction(x.sequence[x.donor_pos:x.next_acceptor_pos]),
            n_iso_don=x.n_isoforms_donor, n_iso_acc=x.n_isoforms_acceptor,
            gene_type=x.gene_type, strand=x.strand,
            # red-team M4: in-window competing REAL sites -- the site-density alternative
            n_real_acc=len(x.real_acc), n_real_don=len(x.real_don),
            n_sham_up=n_up, n_sham_dn=n_dn,
            dmis_up=dm_up, dmis_dn=dm_dn,
        ))

    print(f"\n=== 2x2 flanking-intron architecture  (S = <={ARCH_SHORT_MAX}nt, "
          f"L = >={ARCH_LONG_MIN}nt; buffer excluded) ===")
    cc = Counter(r["cell"] for r in rows)
    for c in ("SS", "SL", "LS", "LL", None):
        lbl = c or "buffer/undefined"
        gn = len({r["gene"] for r in rows if r["cell"] == c})
        print(f"  up-{lbl[0] if c else '-'} / down-{lbl[1] if c else '-'}"
              f"   {lbl:18s} n={cc[c]:6d}  genes={gn:5d}")

    dn_s = [r for r in rows if r["cell"] in ("SS", "LS")]
    dn_l = [r for r in rows if r["cell"] in ("SL", "LL")]
    print(f"\n=== PRIMARY axis: downstream intron  SHORT n={len(dn_s)}  LONG n={len(dn_l)} ===")
    print(f"    genes: SHORT={len({r['gene'] for r in dn_s})}  LONG={len({r['gene'] for r in dn_l})}")

    print("\n=== covariate balance BEFORE matching (SMD; |SMD|<0.1 = balanced) ===")
    cov = ["L_exon", "s_acc", "s_next", "s_don", "gc_exon", "gc_intron",
           "n_iso_don", "n_iso_acc", "donor_pos", "acc_pos", "L_up",
           "n_real_acc", "n_real_don"]
    for k in cov:
        a = np.array([r[k] for r in dn_s], float)
        b = np.array([r[k] for r in dn_l], float)
        flag = "" if abs(smd(a, b)) < 0.1 else "   <-- IMBALANCED"
        print(f"  {k:12s} SHORT={np.nanmean(a):9.3f}  LONG={np.nanmean(b):9.3f}  SMD={smd(a, b):+.3f}{flag}")

    print("\n=== common support of the CROSS-EXON separation (exon length) ===")
    for lo, hi in [(20, 60), (60, 100), (100, 140), (140, 200), (200, 300), (300, 500)]:
        ns = sum(1 for r in dn_s if lo <= r["L_exon"] < hi)
        nl = sum(1 for r in dn_l if lo <= r["L_exon"] < hi)
        print(f"  exon {lo:3d}-{hi:<4d}  SHORT={ns:5d}  LONG={nl:5d}  min={min(ns, nl):5d}")

    print(f"\n=== sham availability (policy {pol}) ===")
    for side, kn, kd in [("exon(up)", "n_sham_up", "dmis_up"), ("intron(down)", "n_sham_dn", "dmis_dn")]:
        n = np.array([r[kn] for r in rows])
        dm = np.array([r[kd] for r in rows if r[kn] > 0], float)
        print(f"  {side:13s} >=1 sham: {(n > 0).mean():.3f}   median |log2 dist ratio| = "
              f"{np.median(np.abs(dm)) if len(dm) else float('nan'):.3f}")
    both = np.array([(r["n_sham_up"] > 0) and (r["n_sham_dn"] > 0) for r in rows])
    print(f"  BOTH sides available: {both.mean():.3f}  (n={both.sum()})")

    # --- the identifiability question: can the strata be balanced at all? ---------------
    # (headline check only; run_ed1_matching_pilot.py does both axes from this artifact)
    arch = [dict(r, up_band=r["cell"][0], dn_band=r["cell"][1])
            for r in rows if r["cell"] is not None]
    cov, cal = axis_spec("down")
    pairs = match_pairs(arch, [r["dn_band"] == "L" for r in arch],
                        covariates=cov, calipers=cal, exact=EXACT_KEYS["down"])
    mt = [arch[i] for i, _ in pairs]
    mc = [arch[j] for _, j in pairs]
    print(f"\n=== outcome-blind 1:1 matching, PRIMARY (downstream) axis; calipers {cal} ===")
    print(f"  matched pairs: {len(pairs)}   genes: {len({r['gene'] for r in mt + mc})}")
    worst = max(abs(smd([covariate_row(r)[k] for r in mc],
                        [covariate_row(r)[k] for r in mt])) for k in cov)
    print(f"  worst post-match |SMD| = {worst:.3f}  ->  "
          f"{'BALANCED' if worst < 0.1 else 'NOT BALANCED'}")
    print(f"  matched LONG cells: {dict(Counter(r['cell'] for r in mt))}")

    out = Path(args.out) if args.out else (
        REPO / f"results/splicebert_coupling/development/ed1_cohort_pilot_{args.split}.json")
    out = guard_output_path(out, overwrite=args.overwrite_record)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        split=args.split, n_windows=len(X), n_eligible=len(keep),
        arch_counts={str(k): v for k, v in cc.items()},
        sham_policy=pol.__dict__, rows=rows), indent=1))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
