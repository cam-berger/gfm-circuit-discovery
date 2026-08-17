#!/usr/bin/env python
"""M1 -- does the masked-acceptor readout depend on the partner donor being a real donor?

Intervenes on NATIVE donors.  Real ``GC-AG`` introns (treatment) and matched real ``GT-AG``
introns (comparator) are each destroyed with the SAME operator ``GG``, so both arms land on an
identical post-intervention state and the only difference between them is the motif the model
was reading beforehand.  No synthetic construct, no motif-hygiene carve-out.

    rho = mean(Y_treatment) / mean(Y_comparator),   Y = clean - destroyed  (masked acceptor G)

    rho ~ 1  the readout depends on a GC donor as much as a GT donor -> donor identity
    rho ~ 0  destroying a real GC donor does not move the readout -> canonical GT only

Development tier, train split.  The validation reserve (79 rows / 65 genes) is NOT spent here,
and chr8/9 is never read -- it holds only 22 GC rows, too few to confirm anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, _dataset,
    load_adapter, pick_device,
)
from gfmcd.splicing.coupling import planted_donor_motifs, substitute  # noqa: E402
from gfmcd.splicing.donor_class import (  # noqa: E402
    assert_seal_intact, train_chromosomes,
    COMPARATOR_MOTIF, MATCHED_OPERATOR, TREATMENT_MOTIF, donor_motif,
    eligible_donor_motifs, flank_and_support_key, gboot_two_sample_ratio, match_comparators,
    matching_balance,
)
from gfmcd.splicing.reverse_coupling import (  # noqa: E402
    acceptor_readout, edit_size, margins_at,
)

SEALED_SPLIT = "test"
CURRENT_TAG = "m1-donor-class-v1"
SESOI = 0.15                 # on the ratio scale
SEED = 0
N_BOOT = 2000
MIN_GENES = 30
K_COMPARATORS = 3
AC2_RHO_GG_ANCHOR = 0.834    # AC2, val, GT arm: rho_GG = 0.834 [0.807, 0.860]
AC2_ANCHOR_BAND = (0.60, 1.00)   # wide: different cohort, so a wiring check, not a reproduction


def classify(lo: float, hi: float, sesoi: float = SESOI) -> str:
    """Strict classification of the donor-class ratio.  A touched boundary is unresolved."""
    if lo > 1.0 - sesoi:
        return "GC_RECOGNIZED_AS_DONOR"
    if hi < sesoi:
        return "GC_NOT_RECOGNIZED"
    if lo > sesoi and hi < 1.0 - sesoi:
        return "PARTIAL_RECOGNITION"
    return "UNRESOLVED"


def _y(task, rows, alt):
    """Masked-acceptor coupling Y = clean - destroyed for one arm under one operator."""
    pos = [acceptor_readout(x) for x in rows]
    clean = margins_at(task, pos, [x.sequence for x in rows])
    dead = margins_at(task, pos, [substitute(x.sequence, x.donor_pos, alt) for x in rows])
    return clean - dead


def build_memorials(cohort, y_t, y_c, noop_max, rho_gg_anchor) -> dict:
    """Falsifiable checks on the measurement path."""
    t, c = cohort.treatment, cohort.comparators
    post_t = {substitute(x.sequence, x.donor_pos, MATCHED_OPERATOR)
              [x.donor_pos:x.donor_pos + 2] for x in t}
    post_c = {substitute(x.sequence, x.donor_pos, MATCHED_OPERATOR)
              [x.donor_pos:x.donor_pos + 2] for x in c}
    edits_t = {edit_size(donor_motif(x), MATCHED_OPERATOR) for x in t}
    edits_c = {edit_size(donor_motif(x), MATCHED_OPERATOR) for x in c}
    bal = matching_balance(cohort)
    balanced = all(
        abs(bal["treatment"][p][b] - bal["comparator"][p][b]) < 1e-9
        for p in bal["treatment"] for b in "ACGT"
    )
    ids = [x.example_id for x in c]
    lo, hi = AC2_ANCHOR_BAND
    return {
        "identical_post_intervention_state": {
            "PASS": post_t == post_c == {MATCHED_OPERATOR},
            "observed": {"treatment": sorted(post_t), "comparator": sorted(post_c)},
            "note": "both arms must land on GG; otherwise the contrast is not motif-only",
        },
        "edit_size_matched_across_arms": {
            "PASS": edits_t == edits_c == {1},
            "observed": {"treatment": sorted(edits_t), "comparator": sorted(edits_c)},
            "note": "GG is 1 substitution on GT and on GC; unequal sizes confound the contrast",
        },
        "native_motifs_are_as_declared": {
            "PASS": ({donor_motif(x) for x in t} == {TREATMENT_MOTIF}
                     and {donor_motif(x) for x in c} == {COMPARATOR_MOTIF}),
            "note": "treatment rows are real GC-AG, comparators real GT-AG",
        },
        "noop_alt_is_exactly_zero": {
            "PASS": bool(abs(float(noop_max)) < 1e-6),
            "observed": float(noop_max),
            "note": "rewriting a donor's own motif over itself must give zero coupling",
        },
        "matching_is_exactly_balanced": {
            "PASS": bool(balanced),
            "note": "exact stratum matching must equalise -1/+3/+5 across arms",
        },
        "no_comparator_reused": {
            "PASS": len(ids) == len(set(ids)),
            "note": "reuse would duplicate rows into the gene bootstrap and shrink the interval",
        },
        "comparator_arm_matches_ac2_wiring_anchor": {
            "PASS": bool(rho_gg_anchor is not None and lo <= rho_gg_anchor <= hi),
            "observed": rho_gg_anchor,
            "band": [lo, hi],
            "ac2_val_reference": AC2_RHO_GG_ANCHOR,
            "note": ("cross-artifact WIRING anchor only -- different cohort and split, so this "
                     "is not a reproduction of AC2 and must not be reported as one"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("smoke", "development"), default="smoke")
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k", type=int, default=K_COMPARATORS)
    ap.add_argument("--limit-treatment", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=False)
    task = CouplingTask(ad, batch=args.batch)

    D = _dataset(args.gtf, args.fasta, allowed_chromosomes=train_chromosomes())
    n_checked = assert_seal_intact(D)

    def arm(motif):
        rows = eligible_donor_motifs(D, "train", donor_motifs=(motif,))
        return [x for x in rows
                if not planted_donor_motifs(x.sequence, x.donor_pos, MATCHED_OPERATOR)]

    gc_rows, gt_pool = arm(TREATMENT_MOTIF), arm(COMPARATOR_MOTIF)
    if args.stage == "smoke":
        gc_rows = gc_rows[:8]
    elif args.limit_treatment:
        gc_rows = gc_rows[:args.limit_treatment]

    # Support-aware key: GC-AG introns are markedly less supported (median 4 donor isoforms
    # against 12), and coupling rises with support, so flank+gap alone leaves the confounder
    # uncontrolled. Matching on it balances the median exactly and moves rho by only +0.02.
    cohort = match_comparators(gc_rows, gt_pool, k=args.k, key_fn=flank_and_support_key)
    t, c = cohort.treatment, cohort.comparators
    if not t or not c:
        raise SystemExit("matching produced an empty arm")

    y_t = _y(task, t, MATCHED_OPERATOR)
    y_c = _y(task, c, MATCHED_OPERATOR)

    # no-op memorial: rewrite each row's own native motif over itself
    noop = max(
        float(np.max(np.abs(_y(task, t, TREATMENT_MOTIF)))) if t else 0.0,
        float(np.max(np.abs(_y(task, c, COMPARATOR_MOTIF)))) if c else 0.0,
    )

    # AC2 wiring anchor: rho_GG on the comparator (GT) arm, relative to full AC destruction
    y_c_ac = _y(task, c, "AC")
    rho_gg_anchor = (float(y_c.sum() / y_c_ac.sum())
                     if abs(float(y_c_ac.sum())) > 1e-9 else None)

    memorials = build_memorials(cohort, y_t, y_c, noop, rho_gg_anchor)
    all_pass = all(m["PASS"] for m in memorials.values())

    gt_t = [x.gene_id for x in t]
    gt_c = [x.gene_id for x in c]
    verdict, est, lo, hi = "MEMORIALS_FAILED", None, None, None
    if all_pass:
        if args.stage == "smoke":
            verdict = "SMOKE_OK"
        else:
            est, lo, hi = gboot_two_sample_ratio(
                y_t, gt_t, y_c, gt_c, n_draws=N_BOOT, seed=SEED, min_genes=MIN_GENES
            )
            verdict = classify(lo, hi)

    result = {
        "schema": "gfmcd.m1.donor_class/1",
        "experiment": "m1_donor_class",
        "stage": args.stage,
        "tier": "development",
        "split": "train",
        "sealed_chr8_chr9_opened": False,
        "seal_enforced_by": "allowed_chromosomes=train_chromosomes() at dataset build",
        "validation_reserve_spent": False,
        "operator": MATCHED_OPERATOR,
        "sesoi": SESOI,
        "cohort": {
            "tag": CURRENT_TAG,
            "k": args.k,
            "n_treatment": len(t),
            "n_treatment_genes": len(set(gt_t)),
            "n_comparator": len(c),
            "n_comparator_genes": len(set(gt_c)),
            "n_unmatched_treatment": len(cohort.unmatched_treatment),
            "unmatched_treatment_ids": cohort.unmatched_treatment,
            "balance": matching_balance(cohort),
        },
        "arms": {
            "treatment_GC": {"mean_Y": float(y_t.mean()), "n": len(y_t)},
            "comparator_GT": {"mean_Y": float(y_c.mean()), "n": len(y_c)},
        },
        "primary": {
            "estimand": "mean(Y | native GC-AG) / mean(Y | matched native GT-AG), operator GG",
            "rho": est, "ci": None if lo is None else [lo, hi],
            "classification": None if lo is None else classify(lo, hi),
        },
        "memorials": memorials,
        "all_memorials_pass": all_pass,
        "verdict": verdict,
        "licenses": (
            "A development-tier statement about whether this checkpoint's masked-acceptor "
            "readout distinguishes a real GC-AG donor from a matched GT-AG donor. Quasi-"
            "observational: donor class is a property of the intron, not assigned. No component, "
            "route, mediation, or biological claim."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:m1_donor_class", sys.argv,
        config=result["cohort"], seeds=dict(bootstrap=SEED), status="DEVELOPMENT",
        notes=["Train split only; sealed chr8/9 not opened; validation reserve not spent.",
               "Development tier; licenses no component, route, or biological claim."])
    for f in (args.model, args.gtf, args.fasta):
        try:
            man.add_input(f)
        except Exception:
            man.notes.append(f"input not hashable in this checkout: {f}")
    man.add_output(out)
    man.write(out.with_suffix(".manifest.json"))
    print(f"verdict={verdict} rho={est} ci={None if lo is None else [round(lo,4), round(hi,4)]}"
          f" -> {out}")


if __name__ == "__main__":
    main()
