#!/usr/bin/env python
"""Build the GENCODE-backed slice of the reconstruction atlas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gfmcd.adapters.dnabert2 import DNABERT2_HF_ID, DNABERT2_REVISION
from gfmcd.reconstruction_atlas.artifacts import write_jsonl_bundle
from gfmcd.reconstruction_atlas.assembly import (
    assemble_annotated_windows,
    gencode_features_from_gtf,
    sample_gencode_features,
    select_gencode_gene_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--assembly", default="GRCh38")
    parser.add_argument("--upstream-nt", type=int, default=512)
    parser.add_argument("--downstream-nt", type=int, default=512)
    parser.add_argument("--immediate-flank-nt", type=int, default=25)
    parser.add_argument("--development-percent", type=int, default=30)
    parser.add_argument("--baseline-training-percent", type=int, default=20)
    parser.add_argument("--sampled-genes-per-split", type=int, default=768)
    parser.add_argument("--clusters-per-site-class", type=int, default=512)
    parser.add_argument("--gene-types", default="protein_coding")
    parser.add_argument(
        "--chromosomes",
        default=",".join([f"chr{x}" for x in range(1, 23)] + ["chrX", "chrY"]),
    )
    parser.add_argument(
        "--include-splits",
        default="development,confirmation",
        help="comma-separated development, baseline_training, and/or confirmation",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        from pyfaidx import Fasta
    except ImportError as exc:
        raise SystemExit(
            "pyfaidx is required; run with the study's pinned requirements"
        ) from exc

    include_splits = frozenset(
        item.strip() for item in args.include_splits.split(",") if item.strip()
    )
    valid_splits = {"development", "baseline_training", "confirmation"}
    if not include_splits or include_splits - valid_splits:
        raise SystemExit(f"invalid --include-splits: {sorted(include_splits)}")
    gene_types = frozenset(
        item.strip() for item in args.gene_types.split(",") if item.strip()
    )
    chromosomes = frozenset(
        item.strip() for item in args.chromosomes.split(",") if item.strip()
    )
    selected_by_split = select_gencode_gene_ids(
        args.gtf,
        seed=args.seed,
        maximum_per_split=args.sampled_genes_per_split,
        development_percent=args.development_percent,
        baseline_training_percent=args.baseline_training_percent,
        gene_types=gene_types,
        chromosomes=chromosomes,
    )
    selected_gene_ids = frozenset(
        gene_id
        for split, gene_ids in selected_by_split.items()
        if split in include_splits
        for gene_id in gene_ids
    )
    annotation_features = gencode_features_from_gtf(
        args.gtf,
        source_version=args.source_version,
        include_gene_ids=selected_gene_ids,
    )
    features = sample_gencode_features(
        annotation_features,
        seed=args.seed,
        maximum_clusters_per_site_class=args.clusters_per_site_class,
        development_percent=args.development_percent,
        baseline_training_percent=args.baseline_training_percent,
    )
    fasta = Fasta(args.fasta, as_raw=True, sequence_always_upper=True)
    records = assemble_annotated_windows(
        features,
        fasta,
        annotation_features=annotation_features,
        assembly=args.assembly,
        upstream_nt=args.upstream_nt,
        downstream_nt=args.downstream_nt,
        split_seed=args.seed,
        development_percent=args.development_percent,
        baseline_training_percent=args.baseline_training_percent,
        immediate_flank_nt=args.immediate_flank_nt,
    )
    records = [record for record in records if record.split in include_splits]
    if not records:
        raise SystemExit("no records remain in the requested splits")
    write_jsonl_bundle(
        args.out,
        [record.to_dict() for record in records],
        experiment_id=f"gencode_atlas_collection_{args.source_version}",
        status="DEVELOPMENT",
        model_identity={
            "hf_id": DNABERT2_HF_ID,
            "revision": DNABERT2_REVISION,
            "used_for_collection_build": False,
        },
        input_paths=(args.gtf, args.fasta),
        parameters={
            "assembly": args.assembly,
            "source_version": args.source_version,
            "upstream_nt": args.upstream_nt,
            "downstream_nt": args.downstream_nt,
            "immediate_flank_nt": args.immediate_flank_nt,
            "development_percent": args.development_percent,
            "baseline_training_percent": args.baseline_training_percent,
            "sampled_genes_per_split": args.sampled_genes_per_split,
            "clusters_per_site_class": args.clusters_per_site_class,
            "gene_types": sorted(gene_types),
            "chromosomes": sorted(chromosomes),
            "include_splits": sorted(include_splits),
            "selected_gene_counts": {
                split: len(gene_ids)
                for split, gene_ids in sorted(selected_by_split.items())
            },
            "n_features": len(features),
            "n_annotation_features": len(annotation_features),
            "n_examples": len(records),
        },
        seeds={"split": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )


if __name__ == "__main__":
    main()
