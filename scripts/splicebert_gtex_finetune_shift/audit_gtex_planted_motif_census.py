#!/usr/bin/env python
"""Model-free planted-motif census for the GTEx cohorts (R44 evidence base).

``ACCEPTOR_ALTS`` / ``DONOR_ALTS`` guarantee only that the bases a substitution WRITES are
not a splice motif.  They say nothing about the two dinucleotides that STRADDLE the edit.
Under a MASKED readout that is harmless -- the substituted base at the masked index is
invisible.  Under this study's unmasked ``NativeClassifierTask`` it is not.

For a substitution writing ``XY`` at ``[q-2, q)`` the three overlapping registers are

    [q-3, q-1) = seq[q-3] + X        [q-2, q) = XY        [q-1, q+1) = Y + seq[q]

A spurious ``AG`` is a planted acceptor; a spurious ``GT``/``GC`` is a planted donor
(``GC-AG`` introns are real, ~0.8% of human introns, so ``GC`` counts).

NO MODEL IS LOADED.  Genome + annotation only, so this runs without the GTEx checkpoints.
Cohorts are rebuilt with the study's own deterministic loader at the recorded (split, n,
tag), and every rebuilt example is structurally validated before anything is counted.

Emits `results/splicebert_gtex_finetune_shift/gtex_planted_motif_census_v1.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.coupling import ACCEPTOR_ALTS, DONOR_ALTS  # noqa: E402
from gfmcd.splicing.native_classifier import load_annotation_cohort  # noqa: E402

BASES = "ACGT"
ALL = [a + b for a in BASES for b in BASES]
PLANTED_DONOR = ("GT", "GC")
PLANTED_ACCEPTOR = ("AG",)

# (label, split, n, tag, consuming artifact).  Args read from each runner's source and its
# committed manifest config -- not guessed.
COHORTS = [
    ("classifier_preflight_v1", "val", 64, "sealed-unc-val",
     "native_classifier_preflight_v1.json"),
    ("mechanism_confirmation_v1", "val", 400, "sealed-unc-val",
     "native_mechanism_confirmation_v1.json"),
    ("context_preflight_v1", "train", 16, "gtex-native-context-localization-preflight-v1",
     "native_context_preflight_v1.json"),
]

# Cohorts deliberately NOT covered here, and why -- silence would read as full coverage.
NOT_COVERED = {
    "native_biological_features_preflight_v1":
        "built by load_fresh_cds_cohort (CDS-filtered), not load_annotation_cohort",
    "positional_band_surgery_v1":
        "built by an in-runner load_fresh_cohort with prior-gene exclusion",
    "positional_native_margin_gate_v1/v2":
        "replays a cohort_source JSON via _load_replayed_cohort",
}


def registers(seq: str, q: int, alt: str) -> dict[str, str]:
    return {"upstream": seq[q - 3] + alt[0], "written": alt, "downstream": alt[1] + seq[q]}


def census(examples, q_of, alts) -> dict:
    n = len(examples)
    out: dict = {}
    for alt in alts:
        pd = pa = clean = 0
        where: Counter = Counter()
        for x in examples:
            r = registers(x.sequence, q_of(x), alt)
            d = {k: v for k, v in r.items() if v in PLANTED_DONOR}
            a = {k: v for k, v in r.items() if v in PLANTED_ACCEPTOR}
            pd += bool(d)
            pa += bool(a)
            clean += not (d or a)
            for k, v in {**d, **a}.items():
                where[f"{k}:{v}"] += 1
        out[alt] = {
            "plants_donor_frac": pd / n, "plants_acceptor_frac": pa / n,
            "motif_clean_frac": clean / n, "n": n,
            "where": {k: v / n for k, v in sorted(where.items())},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default="/Users/cberg/Research/data/genomes/"
                                     "gencode.v50.basic.annotation.gtf.gz")
    ap.add_argument("--fasta", default="/Users/cberg/Research/data/genomes/hg38.fa")
    ap.add_argument("--out", default="results/splicebert_gtex_finetune_shift/"
                                     "gtex_planted_motif_census_v1.json")
    ap.add_argument("--overwrite-record", action="store_true")
    args = ap.parse_args()

    res: dict = {"cohorts": {}, "not_covered": NOT_COVERED,
                 "definitions": {
                     "planted_donor": list(PLANTED_DONOR),
                     "planted_acceptor": list(PLANTED_ACCEPTOR),
                     "registers": "[q-3,q-1)=seq[q-3]+X ; [q-2,q)=XY ; [q-1,q+1)=Y+seq[q]",
                     "readout_note": "only bites where the substituted index is VISIBLE to "
                                     "the model (unmasked NativeClassifierTask); a "
                                     "masked-readout caller is unaffected"}}
    union: dict[str, object] = {}

    for label, split, n, tag, artifact in COHORTS:
        X = load_annotation_cohort(args.gtf, args.fasta, split=split, n=n, tag=tag)
        # Structural validation -- if these fail the GTF version is wrong; stop, do not count.
        bad_a = [x.example_id for x in X
                 if x.sequence[x.next_acceptor_pos - 2:x.next_acceptor_pos] != "AG"]
        bad_d = [x.example_id for x in X
                 if x.sequence[x.donor_pos:x.donor_pos + 2] != "GT"]
        if bad_a or bad_d:
            raise SystemExit(f"{label}: cohort rebuild failed structural validation "
                             f"({len(bad_a)} bad AG, {len(bad_d)} bad GT) -- wrong GTF?")
        for x in X:
            union[x.example_id] = x

        res["cohorts"][label] = {
            "split": split, "n": len(X), "tag": tag, "consuming_artifact": artifact,
            "n_genes": len({x.gene_id for x in X}),
            "validation": {"all_acceptor_AG": True, "all_donor_GT": True},
            "context": {
                "donor_upstream_base_seq_d_minus_1":
                    {b: sum(x.sequence[x.donor_pos - 1] == b for x in X) / len(X)
                     for b in BASES},
                "acceptor_upstream_base_seq_a_minus_3":
                    {b: sum(x.sequence[x.next_acceptor_pos - 3] == b for x in X) / len(X)
                     for b in BASES}},
            "donor_side": census(X, lambda x: x.donor_pos + 2, DONOR_ALTS + ("AC", "AT")),
            "acceptor_side": census(X, lambda x: x.next_acceptor_pos,
                                    ACCEPTOR_ALTS + ("AC", "AT")),
        }

    U = list(union.values())
    res["union"] = {
        "n": len(U), "n_genes": len({x.gene_id for x in U}),
        "note": "deduplicated by example_id across the three load_annotation_cohort cohorts",
        "context": {
            "donor_upstream_base_seq_d_minus_1":
                {b: sum(x.sequence[x.donor_pos - 1] == b for x in U) / len(U) for b in BASES},
            "acceptor_upstream_base_seq_a_minus_3":
                {b: sum(x.sequence[x.next_acceptor_pos - 3] == b for x in U) / len(U)
                 for b in BASES}},
        "donor_side": census(U, lambda x: x.donor_pos + 2, tuple(ALL)),
        "acceptor_side": census(U, lambda x: x.next_acceptor_pos, tuple(ALL)),
    }

    u = res["union"]
    print(f"UNION cohort: n={u['n']} across {u['n_genes']} genes "
          f"(3 load_annotation_cohort cohorts; {len(NOT_COVERED)} cohorts NOT covered)")
    print(f"  seq[d-1]  : {{{', '.join(f'{b} {u['context']['donor_upstream_base_seq_d_minus_1'][b]:.1%}' for b in BASES)}}}")
    print(f"  seq[a-3]  : {{{', '.join(f'{b} {u['context']['acceptor_upstream_base_seq_a_minus_3'][b]:.1%}' for b in BASES)}}}")
    for side, alts in (("donor_side", DONOR_ALTS + ("AC", "AT")),
                       ("acceptor_side", ACCEPTOR_ALTS + ("AC", "AT"))):
        print(f"\n  --- {side} ---   {'alt':>4} {'plantsDON':>10} {'plantsACC':>10} {'clean':>8}")
        for alt in alts:
            r = u[side][alt]
            shipped = alt in (DONOR_ALTS if side == "donor_side" else ACCEPTOR_ALTS)
            print(f"{'':>18} {alt:>4} {r['plants_donor_frac']:>9.1%} "
                  f"{r['plants_acceptor_frac']:>9.1%} {r['motif_clean_frac']:>7.1%}"
                  f"{'   <- shipped' if shipped else ''}")

    p = guard_output_path(REPO / args.out, overwrite=args.overwrite_record)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=float))
    ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:planted_motif_census", sys.argv,
        config={"gtf": args.gtf, "fasta": args.fasta,
                "cohorts": [c[0] for c in COHORTS], "not_covered": list(NOT_COVERED)},
        status="ANNOTATION_ONLY_NO_MODEL",
    ).write(p.with_suffix(".manifest.json"))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
