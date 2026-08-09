#!/usr/bin/env python
"""Model-free planted-motif census for the Head-Context Atlas cohort (R44 evidence base).

The atlas destroys a site with ``(destroy_donor if is_donor else destroy_acceptor)(seq, pos)``.
Those helpers guarantee only that the bases WRITTEN are not a splice motif; they say nothing
about the two dinucleotides that STRADDLE the edit.  ``gate_destroy_real_bases``
(``run_all_head_screen.py:95``) checks length, footprint and alphabet -- never what the edit
CREATES -- so a planted motif passes it silently.

Why the atlas is the exposed study: its readout is ``target_base_position(w)`` while the
corruption lands on the PARTNER / COMPETITOR / DECOY.  The corrupted site is never the
masked site, so every substituted base is visible to the model.  (In the donor-circuit arc
the donor substitution sits at the masked index and is invisible.)

NO MODEL IS LOADED.  The cohort assembly mirrors ``run_all_head_screen.main`` exactly:
same GTF (sha256-pinned), same CompetitionConfig, same discovery gate, same sha256 ordering,
same per-gene cap, same matched-decoy selector.

Emits `results/splicebert_head_context_atlas/atlas_planted_motif_census_v1.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.competition import (  # noqa: E402
    CompetitionConfig, build_competition_index, enumerate_competition_examples,
)
from gfmcd.splicing.competition_seq import build_window  # noqa: E402
from gfmcd.splicing.coupling import (  # noqa: E402
    ACCEPTOR_ALTS, DONOR_ALTS, planted_acceptor_motifs, planted_donor_motifs,
)
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402
from gfmcd.splicing.gtf import load_transcripts  # noqa: E402
from gfmcd.splicing.specificity import find_matched_decoy  # noqa: E402

SEALED = {"chr8", "chr9"}
VALIDATION = {"chr11", "chr16", "chr19"}

# (label, position attr, is-donor attr) -- verbatim from run_all_head_screen.CONDITIONS
CONDITIONS = [("B", "partner", "partner_is_donor"),
              ("C", "competitor", "competitor_is_donor"),
              ("D", "decoy", "decoy_is_donor")]


def planted(seq: str, pos: int, is_donor: bool, alt: str) -> dict[str, str]:
    return (planted_donor_motifs(seq, pos, alt) if is_donor
            else planted_acceptor_motifs(seq, pos, alt))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=None)
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--per-gene-cap", type=int, default=3)
    ap.add_argument("--min-separation", type=int, default=30)
    ap.add_argument("--out", default="results/splicebert_head_context_atlas/"
                                     "atlas_planted_motif_census_v1.json")
    ap.add_argument("--overwrite-record", action="store_true")
    args = ap.parse_args()

    from maxentpy.maxent import load_matrix3, load_matrix5
    m5, m3 = load_matrix5(), load_matrix3()

    ccfg = CompetitionConfig(min_separation=args.min_separation)
    txs = load_transcripts(args.gtf, transcript_types=("protein_coding",),
                           allowed_chromosomes={f"chr{i}" for i in range(1, 23)}
                           | {"chrX", "chrY"})
    exs = [e for e in enumerate_competition_examples(build_competition_index(txs), ccfg)
           if e.usable]
    exs = [e for e in exs if e.split == "train" and e.chrom not in (SEALED | VALIDATION)]
    exs.sort(key=lambda e: int.from_bytes(
        hashlib.sha256(e.example_id.encode()).digest()[:8], "big"))

    genome = PyfaidxGenome(args.fasta)
    windows, decoys, per_gene = [], [], defaultdict(int)
    for e in exs:
        if args.n and len(windows) >= args.n:
            break
        if per_gene[e.gene_id] >= args.per_gene_cap:
            continue
        w = build_window(e, genome, ccfg)
        if w is None:
            continue
        dq, _ = find_matched_decoy(w, m5, m3)
        if dq is None:
            continue
        windows.append(w)
        decoys.append(dq)
        per_gene[e.gene_id] += 1
    print(f"cohort: {len(windows)} windows across {len(per_gene)} genes "
          f"(discovery/train, sealed+val excluded)")

    res: dict = {"cohort": {"n_windows": len(windows), "n_genes": len(per_gene),
                            "gtf": args.gtf, "per_gene_cap": args.per_gene_cap,
                            "min_separation": args.min_separation,
                            "policy": "train_only_excluding_val_and_sealed"},
                 "conditions": {},
                 "note": "bites only because the corrupted site is never the masked readout "
                         "site in this study (readout = target_base_position)"}

    for cond, pos_attr, don_attr in CONDITIONS:
        sites = []
        for w, dq in zip(windows, decoys):
            pos = dq if pos_attr == "decoy" else getattr(w, f"{pos_attr}_win")
            isdon = (w.competitor_is_donor if don_attr == "decoy_is_donor"
                     else getattr(w, don_attr))
            sites.append((w.seq, pos, bool(isdon)))
        n = len(sites)
        n_don = sum(s[2] for s in sites)
        entry: dict = {"n": n, "n_donor_sites": n_don, "n_acceptor_sites": n - n_don,
                       "donor_site_frac": n_don / n if n else 0.0, "alts": {}}
        # variants[0] -- what the head screen and group-MLP closure actually use
        # (DONOR_ALTS[0] and ACCEPTOR_ALTS[0] are both "CC")
        for alt_label, picker in (("variants[0]", lambda d: (DONOR_ALTS if d else ACCEPTOR_ALTS)[0]),):
            hit = sum(bool(planted(s, p, d, picker(d))) for s, p, d in sites)
            hit_don = sum(bool(planted(s, p, d, picker(d))) for s, p, d in sites if d)
            entry["alts"][alt_label] = {
                "alt_for_donor": DONOR_ALTS[0], "alt_for_acceptor": ACCEPTOR_ALTS[0],
                "plants_any_frac": hit / n if n else 0.0,
                "plants_on_donor_sites_frac": hit_don / n_don if n_don else None}
        # the 4-alt mean paths (behavioral_pilot, specificity_pilot)
        for i in range(4):
            hit_don = sum(bool(planted(s, p, d, DONOR_ALTS[i])) for s, p, d in sites if d)
            hit_acc = sum(bool(planted(s, p, d, ACCEPTOR_ALTS[i])) for s, p, d in sites if not d)
            entry["alts"][f"variant[{i}]"] = {
                "donor_alt": DONOR_ALTS[i], "acceptor_alt": ACCEPTOR_ALTS[i],
                "plants_on_donor_sites_frac": hit_don / n_don if n_don else None,
                "plants_on_acceptor_sites_frac": hit_acc / (n - n_don) if n - n_don else None}
        # what the clean alt would do
        clean_don = sum(bool(planted(s, p, d, "AC")) for s, p, d in sites if d)
        clean_acc = sum(bool(planted(s, p, d, "AC")) for s, p, d in sites if not d)
        entry["alts"]["AC (clean)"] = {
            "plants_on_donor_sites_frac": clean_don / n_don if n_don else None,
            "plants_on_acceptor_sites_frac": clean_acc / (n - n_don) if n - n_don else None}
        res["conditions"][cond] = entry

        d0 = entry["alts"]["variants[0]"]
        print(f"  {cond}: n={n} ({n_don} donor / {n - n_don} acceptor sites)  "
              f"variants[0] plants on {d0['plants_any_frac']:.1%} of sites "
              f"(donor sites: "
              f"{d0['plants_on_donor_sites_frac'] if d0['plants_on_donor_sites_frac'] is None else format(d0['plants_on_donor_sites_frac'], '.1%')})")
        for i in range(4):
            v = entry["alts"][f"variant[{i}]"]
            fd = v["plants_on_donor_sites_frac"]
            print(f"      variant[{i}] donor={v['donor_alt']} -> "
                  f"{'n/a' if fd is None else format(fd, '.1%')} of donor sites planted")

    p = guard_output_path(REPO / args.out, overwrite=args.overwrite_record)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=float))
    ExperimentManifest.create(
        "splicebert_head_context_atlas:planted_motif_census", sys.argv,
        config={"gtf": args.gtf, "fasta": args.fasta, "n": args.n,
                "per_gene_cap": args.per_gene_cap, "min_separation": args.min_separation},
        status="ANNOTATION_ONLY_NO_MODEL",
    ).write(p.with_suffix(".manifest.json"))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
