#!/usr/bin/env python
"""Cross-site coupling protocol (steps 1-7).

    Q: does 5'SS (donor) recognition causally depend on the PARTNER 3'SS (acceptor)?

Native MLM readout; perturbations SUBSTITUTE real bases (never [MASK]).

  1-3  dependence      : destroy the partner acceptor -> drop in donor margin
  4    matched controls: distance-matched random position
                         motif-matched non-partner AG
                         another REAL annotated splice site
                         transplanted acceptor of similar local strength
  5    specificity     : does destroying acceptor A hurt ITS donor more than OTHER donors
                         in the same window?  (within-sequence contrast -- divides out
                         "the perturbation perturbed everything")
  6    both directions : acceptor->donor across the exon; donor->acceptor across the exon;
                         donor->acceptor across the intron (short introns only)
  7    stratify        : exon length, donor strength, constitutive vs alternative, gene type

A drop alone proves only NON-LOCAL dependence.  The claim needs dependence AND
partner-selectivity AND (later) mechanistic localisation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.coupling import (  # noqa: E402
    CouplingConfig, CouplingScorer, build_coupling_dataset, destroy_acceptor,
    destroy_donor, perturb_random, stable_rng, transplant_acceptor,
)
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402


def mean_drop(scorer, clean, variants, pos, base="G"):
    """clean - mean(perturbed).  Positive = recognition of the target was hurt."""
    if not variants:
        return np.nan
    p = scorer._margin(variants, [pos] * len(variants), base)
    return float(clean - p.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-null", type=int, default=6, help="samples per null condition")
    ap.add_argument("--gtf", default=None)
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--model", default="data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
    ap.add_argument("--chroms", nargs="*", default=None)
    ap.add_argument("--out", default="results/splicebert_coupling/coupling.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    model_path = args.model if Path(args.model).is_absolute() else str(REPO / args.model)
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForMaskedLM.from_pretrained(model_path).to(dev)
    scorer = CouplingScorer(mdl, tok, device=dev, batch_size=args.batch)

    cfg = CouplingConfig()
    X = build_coupling_dataset(
        args.gtf, PyfaidxGenome(args.fasta), cfg,
        allowed_chromosomes=set(args.chroms) if args.chroms else None,
    )
    X = [x for x in X if x.split == args.split and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"]
    rng = stable_rng("select", cfg.seed)
    rng.shuffle(X)
    X = X[: args.n]
    print(f"{len(X):,} internal exons ({args.split}, canonical GT donors)  device={dev}")

    rows = []
    for i, x in enumerate(X):
        if i % 50 == 0:
            print(f"  {i}/{len(X)}", end="\r", flush=True)
        r = stable_rng(x.example_id, cfg.seed)
        seq, a, d, N = x.sequence, x.acceptor_pos, x.donor_pos, len(x.sequence)
        L = d - a

        clean_d = scorer._margin([seq], [d], "G")[0]
        clean_a = scorer._margin([seq], [a - 1], "G")[0]

        # ---- step 1-3: destroy the PARTNER acceptor, measure the donor -------------
        acc_variants = destroy_acceptor(seq, a)
        e_partner = mean_drop(scorer, clean_d, acc_variants, d)

        # ---- step 4: matched controls ---------------------------------------------
        # (a) distance-matched random position (same offset from the donor, other side)
        dm = d + L
        e_dist = (mean_drop(scorer, clean_d, [perturb_random(seq, dm)], d)
                  if 10 < dm < N - 10 else np.nan)
        # (b) motif-matched non-partner AGs
        ags = list(x.other_ag)
        r.shuffle(ags)
        ags = ags[: args.k_null]
        e_ag = mean_drop(scorer, clean_d, [destroy_acceptor(seq, p)[0] for p in ags], d)
        # (c) another REAL annotated splice site
        reals = list(x.real_acc)
        r.shuffle(reals)
        reals = reals[: args.k_null]
        e_real = mean_drop(scorer, clean_d, [destroy_acceptor(seq, p)[0] for p in reals], d)
        # (d) transplanted acceptor of similar local strength (another exon's acceptor context)
        donor_ex = X[r.randrange(len(X))]
        try:
            tp = transplant_acceptor(seq, a, donor_ex.sequence, donor_ex.acceptor_pos)
            e_transplant = mean_drop(scorer, clean_d, [tp], d)
        except ValueError:
            e_transplant = np.nan

        # ---- step 5: SPECIFICITY -- effect on OTHER donors in the same window -------
        gts = list(x.other_gt)
        r.shuffle(gts)
        gts = gts[: args.k_null]
        if gts:
            base = scorer._margin([seq] * len(gts), gts, "G")
            pert = scorer._margin([acc_variants[0]] * len(gts), gts, "G")
            e_other_donors = float((base - pert).mean())
        else:
            e_other_donors = np.nan

        # ---- step 6: reverse direction -- destroy the DONOR, measure the ACCEPTOR ---
        don_variants = destroy_donor(seq, d)
        e_rev = mean_drop(scorer, clean_a, don_variants, a - 1)
        # matched null for the reverse direction
        gts2 = list(x.other_gt)
        r.shuffle(gts2)
        gts2 = gts2[: args.k_null]
        e_rev_null = mean_drop(scorer, clean_a,
                               [destroy_donor(seq, p)[0] for p in gts2], a - 1) if gts2 else np.nan

        rows.append(dict(
            example_id=x.example_id, gene_id=x.gene_id, exon_length=x.exon_length,
            intron_down=x.intron_length_down, n_iso_donor=x.n_isoforms_donor,
            constitutive=bool(x.is_constitutive), gene_type=x.gene_type,
            clean_donor=float(clean_d), clean_acceptor=float(clean_a),
            e_partner=e_partner, e_dist_matched=float(e_dist), e_motif_ag=e_ag,
            e_real_site=e_real, e_transplant=float(e_transplant),
            e_other_donors=e_other_donors, e_reverse=e_rev, e_reverse_null=float(e_rev_null),
        ))
    print()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    ExperimentManifest.create(
        "splicebert_coupling:main", sys.argv,
        config={"n": len(rows), "split": args.split, "model": model_path, "k_null": args.k_null},
        seeds={"build": cfg.seed},
        notes=["Native MLM readout; perturbations substitute real bases, never [MASK].",
               "A drop proves NON-LOCAL dependence only; the claim needs partner-selectivity too."],
    ).write(out.parent / "manifest.json")
    (out.parent / "per_example.json").write_text(json.dumps(rows, indent=1))
    report(rows, out)


def report(rows, out: Path) -> None:
    def col(k):
        v = np.array([r[k] for r in rows], float)
        return v[~np.isnan(v)]

    from scipy import stats

    print("\n" + "=" * 70)
    print("STEP 1-4  DEPENDENCE + MATCHED CONTROLS")
    print("  effect = drop in the TRUE DONOR's margin when we destroy ...")
    print(f"    {'condition':38s} {'mean':>7s} {'median':>7s}")
    for k, lbl in [("e_partner", "its PARTNER acceptor"),
                   ("e_real_site", "another REAL annotated splice site"),
                   ("e_motif_ag", "a motif-matched non-partner AG"),
                   ("e_dist_matched", "a distance-matched random position"),
                   ("e_transplant", "transplanted acceptor (similar strength)")]:
        v = col(k)
        print(f"    {lbl:38s} {v.mean():+7.3f} {np.median(v):+7.3f}")

    print("\nSTEP 5  PARTNER-SELECTIVITY (within-sequence contrast)")
    p_ = np.array([r["e_partner"] for r in rows], float)
    o_ = np.array([r["e_other_donors"] for r in rows], float)
    m = ~np.isnan(p_) & ~np.isnan(o_)
    print(f"    destroy the acceptor -> drop at ITS OWN donor   : {p_[m].mean():+.3f}")
    print(f"    destroy the acceptor -> drop at OTHER donors    : {o_[m].mean():+.3f}")
    print(f"    partner-selective gap                          : {p_[m].mean()-o_[m].mean():+.3f}")
    if m.sum() > 10:
        w = stats.wilcoxon(p_[m], o_[m])
        print(f"    paired Wilcoxon p = {w.pvalue:.2e}   (n={m.sum()})")

    print("\nSTEP 6  REVERSE DIRECTION (destroy the donor, measure the acceptor)")
    rv, rn = col("e_reverse"), col("e_reverse_null")
    print(f"    destroy its PARTNER donor : {rv.mean():+.3f}")
    print(f"    destroy a non-partner GT  : {rn.mean():+.3f}")

    print("\nSTEP 7  STRATIFICATION  (partner effect on the donor)")
    def strat(name, keyfn, bins):
        print(f"  by {name}:")
        for lbl, sel in bins:
            v = np.array([r["e_partner"] for r in rows if sel(r)], float)
            v = v[~np.isnan(v)]
            if len(v) < 10:
                continue
            print(f"    {lbl:28s} n={len(v):5d}  mean {v.mean():+.3f}  median {np.median(v):+.3f}")
    strat("donor strength (clean margin)", None, [
        ("weak   (margin < 2)", lambda r: r["clean_donor"] < 2),
        ("medium (2-5)", lambda r: 2 <= r["clean_donor"] < 5),
        ("strong (>= 5)", lambda r: r["clean_donor"] >= 5)])
    strat("exon length", None, [
        ("short  (<80nt)", lambda r: r["exon_length"] < 80),
        ("medium (80-200)", lambda r: 80 <= r["exon_length"] < 200),
        ("long   (>=200)", lambda r: r["exon_length"] >= 200)])
    strat("isoform support", None, [
        ("alternative (<5 iso)", lambda r: not r["constitutive"]),
        ("constitutive (>=5)", lambda r: r["constitutive"])])
    strat("gene type", None, [
        ("protein_coding", lambda r: r["gene_type"] == "protein_coding"),
        ("other", lambda r: r["gene_type"] != "protein_coding")])
    print("=" * 70)
    print(f"saved -> {out.parent}")


if __name__ == "__main__":
    main()
