#!/usr/bin/env python
"""Build an atlas collection from the normalized evidence-bearing BED contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gfmcd.adapters.dnabert2 import DNABERT2_HF_ID, DNABERT2_REVISION
from gfmcd.reconstruction_atlas.artifacts import write_jsonl_bundle
from gfmcd.reconstruction_atlas.assembly import (
    assemble_annotated_windows,
    read_normalized_bed,
)
from gfmcd.reconstruction_atlas.schema import EvidenceType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument(
        "--evidence-type",
        choices=tuple(item.value for item in EvidenceType),
        required=True,
    )
    parser.add_argument("--assembly", default="GRCh38")
    parser.add_argument("--upstream-nt", type=int, default=512)
    parser.add_argument("--downstream-nt", type=int, default=512)
    parser.add_argument("--immediate-flank-nt", type=int, default=25)
    parser.add_argument("--development-percent", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        from pyfaidx import Fasta
    except ImportError as exc:
        raise SystemExit(
            "pyfaidx is required; run with the study's pinned requirements"
        ) from exc

    features = read_normalized_bed(
        args.bed,
        source_id=args.source_id,
        source_version=args.source_version,
        evidence_type=EvidenceType(args.evidence_type),
    )
    fasta = Fasta(args.fasta, as_raw=True, sequence_always_upper=True)
    records = assemble_annotated_windows(
        features,
        fasta,
        assembly=args.assembly,
        upstream_nt=args.upstream_nt,
        downstream_nt=args.downstream_nt,
        split_seed=args.seed,
        development_percent=args.development_percent,
        immediate_flank_nt=args.immediate_flank_nt,
    )
    write_jsonl_bundle(
        args.out,
        [record.to_dict() for record in records],
        experiment_id=f"{args.source_id}_atlas_collection_{args.source_version}",
        status="DEVELOPMENT",
        model_identity={
            "hf_id": DNABERT2_HF_ID,
            "revision": DNABERT2_REVISION,
            "used_for_collection_build": False,
        },
        input_paths=(args.bed, args.fasta),
        parameters={
            "assembly": args.assembly,
            "source_id": args.source_id,
            "source_version": args.source_version,
            "evidence_type": args.evidence_type,
            "upstream_nt": args.upstream_nt,
            "downstream_nt": args.downstream_nt,
            "immediate_flank_nt": args.immediate_flank_nt,
            "development_percent": args.development_percent,
            "n_features": len(features),
            "n_examples": len(records),
        },
        seeds={"split": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )


if __name__ == "__main__":
    main()
