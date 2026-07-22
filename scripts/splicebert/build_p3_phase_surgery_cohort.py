#!/usr/bin/env python
"""Build the annotation-only cohort for nucleotide-invariant P3 phase surgery."""
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
from gfmcd.splicing.p3_closure import select_spread_blocks, stable_gene_role  # noqa: E402
from gfmcd.splicing.p3_cohort import (  # noqa: E402
    P3CohortConfig,
    annotation_safe_candidate_pools,
    sequence_sha256,
)
from gfmcd.splicing.phase import example_phase, load_cds_frames  # noqa: E402


DEFAULT_COMPREHENSIVE_GTF = "/Users/cberg/Research/data/genomes/gencode.v50.annotation.gtf.gz"
NAMESPACE = "p3-closure-v1"
ROLE_WEIGHTS = (("development", 2), ("confirmation", 2), ("reserve", 1))
EXCLUDED_CHROMOSOMES = frozenset(("chr8", "chr9", "chr11", "chr16", "chr19"))
AUTOSOMES = frozenset(f"chr{i}" for i in range(1, 23))


def stable_rank(text: str) -> str:
    return hashlib.sha256(f"p3-phase-surgery-v1\0{text}".encode()).hexdigest()


def parse_roles(text: str) -> tuple[str, ...]:
    roles = tuple(part.strip() for part in text.split(",") if part.strip())
    valid = {name for name, _weight in ROLE_WEIGHTS}
    if not roles or set(roles) - valid:
        raise argparse.ArgumentTypeError(f"roles must be drawn from {sorted(valid)}")
    return roles


def choose_shift_safe_blocks(candidates, *, n: int, maximum_shift: int, table_length: int):
    safe = tuple(
        block for block in candidates
        if block.start + 1 - maximum_shift >= 0
        and block.end + 1 + maximum_shift <= table_length
    )
    selected = select_spread_blocks(safe, n=n)
    if n > 1 and min(
        abs(left.start - right.start)
        for index, left in enumerate(selected)
        for right in selected[index + 1:]
    ) < 12:
        raise ValueError("selected phase-surgery blocks are not spatially distinct")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--comprehensive-gtf", default=DEFAULT_COMPREHENSIVE_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--include-roles", type=parse_roles, default=("development",))
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument("--max-examples-per-gene", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=36)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--maximum-shift", type=int, default=10)
    parser.add_argument("--position-table-length", type=int, default=1026)
    parser.add_argument(
        "--out", default="results/phase-lattice/p3_phase_surgery_development_cohort.json"
    )
    args = parser.parse_args()
    if args.max_genes is not None and args.max_genes <= 0:
        parser.error("--max-genes must be positive")
    if args.max_examples_per_gene <= 0 or args.n_blocks < 2:
        parser.error("example and block counts must be positive; at least two blocks are required")
    if args.max_genes is not None and args.include_roles != ("development",):
        parser.error("limited cohort builds are development-only")

    out = Path(args.out)
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite phase-surgery cohort: {out}")

    config = P3CohortConfig(block_size=args.block_size, n_blocks=args.n_blocks)
    config.validate()
    allowed = AUTOSOMES - EXCLUDED_CHROMOSOMES
    phase_lookup = load_cds_frames(args.gtf, allowed_chromosomes=allowed)
    safety = load_annotation_safety_index(args.comprehensive_gtf, allowed_chromosomes=allowed)
    examples = [
        example for example in eligible(_dataset(args.gtf, args.fasta), "train")
        if (example.chrom if example.chrom.startswith("chr") else f"chr{example.chrom}") in allowed
    ]
    by_gene = defaultdict(list)
    for example in examples:
        role = stable_gene_role(example.gene_id, namespace=NAMESPACE, roles=ROLE_WEIGHTS)
        if role in args.include_roles:
            by_gene[example.gene_id].append(example)
    genes = sorted(by_gene, key=lambda gene: (stable_rank(gene), gene))
    if args.max_genes is not None:
        genes = genes[:args.max_genes]

    records = []
    exclusions: Counter[str] = Counter()
    for index, gene in enumerate(genes, 1):
        role = stable_gene_role(gene, namespace=NAMESPACE, roles=ROLE_WEIGHTS)
        candidates = sorted(by_gene[gene], key=lambda example: (stable_rank(example.example_id),
                                                                 example.example_id))
        kept = 0
        for example in candidates:
            if kept >= args.max_examples_per_gene:
                break
            phase = example_phase(example, phase_lookup)
            if phase is None:
                continue
            try:
                pools, annotation_audit = annotation_safe_candidate_pools(
                    example, donor_phase=phase, safety=safety, config=config
                )
                blocks = {
                    context: choose_shift_safe_blocks(
                        pool,
                        n=args.n_blocks,
                        maximum_shift=args.maximum_shift,
                        table_length=args.position_table_length,
                    )
                    for context, pool in pools.items()
                }
            except ValueError:
                continue
            records.append({
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "chrom": example.chrom,
                "strand": example.strand,
                "role": role,
                "eligible": True,
                "donor_phase": phase,
                "target_context": "cds",
                "acceptor_pos": example.acceptor_pos,
                "donor_pos": example.donor_pos,
                "next_acceptor_pos": example.next_acceptor_pos,
                "native_sequence_sha256": sequence_sha256(example.sequence),
                "blocks": {
                    context: [[block.start, block.end] for block in selected]
                    for context, selected in blocks.items()
                },
                "annotation_counts": {
                    key: value for key, value in annotation_audit.items() if key.startswith("n_")
                },
                "exon_length": example.exon_length,
                "donor_partner_distance": int(example.next_acceptor_pos - example.donor_pos),
            })
            kept += 1
        if kept == 0:
            exclusions["no_annotation_safe_two-block_example"] += 1
        print(f"constructed {index}/{len(genes)} genes", flush=True)

    role_counts = Counter(row["role"] for row in records)
    package = {
        "schema": "gfmcd.p3_phase_surgery.cohort.v1",
        "status": "OUTCOME_BLIND_ANNOTATION_COHORT",
        "model_outcomes_read": False,
        "namespace": NAMESPACE,
        "role_weights": [list(item) for item in ROLE_WEIGHTS],
        "included_roles": list(args.include_roles),
        "excluded_chromosomes": sorted(EXCLUDED_CHROMOSOMES),
        "config": {
            "target_context": "cds",
            "block_size": args.block_size,
            "n_blocks": args.n_blocks,
            "maximum_shift": args.maximum_shift,
            "position_table_length": args.position_table_length,
            "prefix_tokens": 1,
        },
        "inputs": {
            "basic_gtf": file_record(args.gtf),
            "comprehensive_gtf": file_record(args.comprehensive_gtf),
            "fasta": file_record(args.fasta),
        },
        "counts": {
            "source_train_eligible_examples": len(examples),
            "genes_considered": len(genes),
            "eligible_examples_by_role": dict(sorted(role_counts.items())),
            "eligible_unique_genes": len({row["gene_id"] for row in records}),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "p3-phase-surgery-cohort-v1",
        sys.argv,
        repo=REPO,
        status="OUTCOME_BLIND_COHORT",
        config=package["config"],
        notes=["No model loaded; every nucleotide is unchanged by the planned intervention."],
    )
    for path in (
        args.gtf, args.comprehensive_gtf, args.fasta,
        REPO / "src/gfmcd/splicing/p3_phase_surgery.py",
        REPO / "scripts/splicebert/build_p3_phase_surgery_cohort.py",
    ):
        manifest.add_input(path)
    manifest.add_output(out)
    manifest.write(manifest_path)
    print(json.dumps(package["counts"], indent=2, sort_keys=True))
    print(f"wrote {out} and {manifest_path}")


if __name__ == "__main__":
    main()
