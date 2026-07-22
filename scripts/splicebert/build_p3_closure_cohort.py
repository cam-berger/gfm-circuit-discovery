#!/usr/bin/env python
"""Build the outcome-blind P3 closure cohort and construction audit.

This command never loads SpliceBERT.  Confirmation-role sequences may be constructed and hashed,
but no model-derived quantity is read.  A development-only ``--max-genes`` mode exists for fast
mechanical pilots; partial confirmation pools are deliberately forbidden.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.circuit import DEFAULT_FASTA, DEFAULT_GTF, _dataset, eligible  # noqa: E402
from gfmcd.splicing.p3_annotation import load_annotation_safety_index  # noqa: E402
from gfmcd.splicing.p3_closure import stable_gene_role  # noqa: E402
from gfmcd.splicing.p3_cohort import (  # noqa: E402
    P3CohortConfig,
    construct_example_package,
    construction_failure_reason,
)
from gfmcd.splicing.phase import example_phase, load_cds_frames  # noqa: E402
from gfmcd.splicing.splice_landscape import maxent_scorers  # noqa: E402


DEFAULT_COMPREHENSIVE_GTF = "/Users/cberg/Research/data/genomes/gencode.v50.annotation.gtf.gz"
NAMESPACE = "p3-closure-v1"
ROLE_WEIGHTS = (("development", 2), ("confirmation", 2), ("reserve", 1))
EXCLUDED_CHROMOSOMES = frozenset(("chr8", "chr9", "chr11", "chr16", "chr19"))
AUTOSOMES = frozenset(f"chr{i}" for i in range(1, 23))


def normalized_chrom(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def stable_example_rank(example_id: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}\0{example_id}".encode()).hexdigest()


def parse_roles(text: str) -> tuple[str, ...]:
    roles = tuple(part.strip() for part in text.split(",") if part.strip())
    valid = {name for name, _weight in ROLE_WEIGHTS}
    if not roles or set(roles) - valid:
        raise argparse.ArgumentTypeError(f"roles must be a comma-list drawn from {sorted(valid)}")
    return roles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--comprehensive-gtf", default=DEFAULT_COMPREHENSIVE_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--include-roles", type=parse_roles, default=("development",))
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument("--max-examples-per-gene", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=60)
    parser.add_argument("--max-abs-delta-score3", type=float, default=2.0)
    parser.add_argument("--max-abs-delta-score5", type=float, default=2.0)
    parser.add_argument("--max-hamming-imbalance", type=float, default=0.10)
    parser.add_argument("--max-trimer-imbalance", type=float, default=0.10)
    parser.add_argument(
        "--calibration-relaxed-landscape",
        action="store_true",
        help="Development-only: record MaxEnt/motif metrics but gate only structural and stop-free invariants",
    )
    parser.add_argument(
        "--out", default="results/phase-lattice/p3_closure_development_cohort.json"
    )
    args = parser.parse_args()
    if args.max_genes is not None and args.max_genes <= 0:
        parser.error("--max-genes must be positive")
    if args.max_examples_per_gene <= 0:
        parser.error("--max-examples-per-gene must be positive")
    if args.max_genes is not None and args.include_roles != ("development",):
        parser.error("a limited census is allowed only for the development role")
    if args.calibration_relaxed_landscape and args.include_roles != ("development",):
        parser.error("relaxed construction calibration is development-only")

    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite cohort artifact: {out}")

    config = P3CohortConfig(
        block_size=args.block_size,
        max_abs_delta_score3=args.max_abs_delta_score3,
        max_abs_delta_score5=args.max_abs_delta_score5,
        max_hamming_arm_imbalance=args.max_hamming_imbalance,
        max_trimer_arm_imbalance=args.max_trimer_imbalance,
        require_canonical_motif_nonincrease=not args.calibration_relaxed_landscape,
        require_per_variant_maxent_delta=not args.calibration_relaxed_landscape,
        require_no_new_competitive_site=not args.calibration_relaxed_landscape,
        require_strong_site_count_preserved=not args.calibration_relaxed_landscape,
    )
    config.validate()
    allowed = AUTOSOMES - EXCLUDED_CHROMOSOMES
    phase_lookup = load_cds_frames(args.gtf, allowed_chromosomes=allowed)
    safety = load_annotation_safety_index(args.comprehensive_gtf, allowed_chromosomes=allowed)
    score3, score5 = maxent_scorers()

    examples = [
        example for example in eligible(_dataset(args.gtf, args.fasta), "train")
        if normalized_chrom(example.chrom) in allowed
    ]
    by_gene: dict[str, list] = defaultdict(list)
    for example in examples:
        role = stable_gene_role(example.gene_id, namespace=NAMESPACE, roles=ROLE_WEIGHTS)
        if role in args.include_roles:
            by_gene[example.gene_id].append(example)

    ordered_genes = sorted(
        by_gene,
        key=lambda gene: (
            stable_gene_role(gene, namespace=NAMESPACE, roles=ROLE_WEIGHTS),
            hashlib.sha256(f"{NAMESPACE}\0gene\0{gene}".encode()).hexdigest(),
            gene,
        ),
    )
    if args.max_genes is not None:
        ordered_genes = ordered_genes[:args.max_genes]

    records = []
    exclusions: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    for gene_index, gene_id in enumerate(ordered_genes, 1):
        role = stable_gene_role(gene_id, namespace=NAMESPACE, roles=ROLE_WEIGHTS)
        # Physical feasibility and clean-CDS annotation are sequence-only filters.  Apply them
        # before the stable rank so a short/UTR exon cannot consume a gene's cassette quota while
        # a valid internal coding exon from the same gene is silently ignored.
        structurally_feasible = []
        structural_failures = []
        for example in by_gene[gene_id]:
            phase = example_phase(example, phase_lookup)
            if phase is None:
                structural_failures.append((example, "not_wholly_unambiguous_basic_cds"))
                continue
            if example.exon_length < config.block_size + config.donor_clearance + 3:
                structural_failures.append((example, "cds_too_short_for_two_cassettes"))
                continue
            if example.acceptor_pos < config.block_size + config.acceptor_clearance + 1:
                structural_failures.append((example, "upstream_intron_window_too_short"))
                continue
            structurally_feasible.append((example, phase))
        structurally_feasible.sort(
            key=lambda item: (stable_example_rank(item[0].example_id), item[0].example_id)
        )

        kept_for_gene = 0
        for example, phase in structurally_feasible:
            if kept_for_gene >= args.max_examples_per_gene:
                break
            candidate_counts[role] += 1
            try:
                record = construct_example_package(
                    example,
                    donor_phase=phase,
                    safety=safety,
                    score3=score3,
                    score5=score5,
                    config=config,
                )
            except Exception as exc:
                reason = construction_failure_reason(exc)
                exclusions[reason] += 1
                records.append({
                    "example_id": example.example_id,
                    "gene_id": example.gene_id,
                    "chrom": example.chrom,
                    "role": role,
                    "eligible": False,
                    "exclusion_reason": reason,
                    "exception": str(exc),
                })
                continue
            record["role"] = role
            if not record["eligible"]:
                failures = Counter()
                for variant in record["variants"]:
                    if not variant["passed"]:
                        for name, passed in variant["pure_checks"].items():
                            if not passed:
                                failures[name] += 1
                        for name, passed in variant["landscape_checks"].items():
                            if not passed:
                                failures[name] += 1
                for arm in record["balance"].values():
                    for name, passed in arm["checks"].items():
                        if not passed:
                            failures[name] += 1
                record["exclusion_reason"] = "sequence_gate_failure"
                record["failed_gate_counts"] = dict(sorted(failures.items()))
                exclusions["sequence_gate_failure"] += 1
            else:
                kept_for_gene += 1
            records.append(record)
        if not structurally_feasible:
            # Preserve one representative failure per gene so yield accounting remains auditable
            # without materialising thousands of irrelevant short/UTR exon rows.
            if structural_failures:
                example, reason = min(
                    structural_failures,
                    key=lambda item: (stable_example_rank(item[0].example_id), item[0].example_id),
                )
                exclusions[reason] += 1
                records.append({
                    "example_id": example.example_id,
                    "gene_id": example.gene_id,
                    "chrom": example.chrom,
                    "role": role,
                    "eligible": False,
                    "exclusion_reason": reason,
                })
        print(f"constructed {gene_index}/{len(ordered_genes)} genes", flush=True)

    eligible_records = [row for row in records if row.get("eligible")]
    eligible_gene_counts = Counter(row["role"] for row in eligible_records)
    eligible_unique_genes = Counter()
    for role in args.include_roles:
        eligible_unique_genes[role] = len({
            row["gene_id"] for row in eligible_records if row["role"] == role
        })
    package = {
        "schema": "gfmcd.p3_closure.cohort.v1",
        "status": "OUTCOME_BLIND_CONSTRUCTION_CENSUS",
        "model_outcomes_read": False,
        "namespace": NAMESPACE,
        "role_weights": [list(item) for item in ROLE_WEIGHTS],
        "included_roles": list(args.include_roles),
        "excluded_chromosomes": sorted(EXCLUDED_CHROMOSOMES),
        "allowed_chromosomes": sorted(allowed),
        "max_examples_per_gene": args.max_examples_per_gene,
        "max_genes_development_only": args.max_genes,
        "config": asdict(config),
        "inputs": {
            "basic_gtf": file_record(args.gtf),
            "comprehensive_gtf": file_record(args.comprehensive_gtf),
            "fasta": file_record(args.fasta),
        },
        "counts": {
            "source_train_eligible_examples": len(examples),
            "genes_considered": len(ordered_genes),
            "selected_candidates_by_role": dict(sorted(candidate_counts.items())),
            "eligible_examples_by_role": dict(sorted(eligible_gene_counts.items())),
            "eligible_genes_by_role": dict(sorted(eligible_unique_genes.items())),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        "p3-closure-cohort-v1",
        sys.argv,
        repo=REPO,
        status="OUTCOME_BLIND_COHORT",
        config={
            "namespace": NAMESPACE,
            "role_weights": ROLE_WEIGHTS,
            "included_roles": args.include_roles,
            "p3_cohort": asdict(config),
        },
        notes=[
            "No model was loaded and no model outcome was read.",
            "Limited --max-genes mode is development-only.",
            "Confirmation-role construction is annotation/sequence only and is not an exposure.",
        ],
    )
    for path in (
        args.gtf,
        args.comprehensive_gtf,
        args.fasta,
        REPO / "src/gfmcd/splicing/p3_closure.py",
        REPO / "src/gfmcd/splicing/p3_cohort.py",
        REPO / "src/gfmcd/splicing/p3_annotation.py",
        REPO / "src/gfmcd/splicing/splice_landscape.py",
        REPO / "studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_P3_CLOSURE.md",
    ):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps(package["counts"], indent=2, sort_keys=True))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
