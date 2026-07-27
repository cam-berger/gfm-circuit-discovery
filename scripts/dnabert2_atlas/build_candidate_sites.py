#!/usr/bin/env python
"""Build outcome-blind empirical-control candidates inside admitted atlas windows."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from gfmcd.adapters.dnabert2 import DNABERT2_HF_ID, DNABERT2_REVISION
from gfmcd.reconstruction_atlas.artifacts import write_jsonl_bundle
from gfmcd.reconstruction_atlas.assembly import (
    gencode_annotation_intervals_for_windows,
)
from gfmcd.reconstruction_atlas.controls import generate_candidate_sites
from gfmcd.reconstruction_atlas.schema import load_collection
from gfmcd.reconstruction_atlas.sources import (
    annotation_intervals_from_bed_for_windows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--gencode-gtf", required=True)
    parser.add_argument(
        "--annotation-bed",
        help="optional multi-source GRCh38 BED(.gz) for conservative exclusions",
    )
    parser.add_argument("--positions-per-example", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = load_collection(args.collection)
    annotation_intervals = gencode_annotation_intervals_for_windows(
        args.gencode_gtf,
        records,
    )
    annotation_sources = ["GENCODE exon boundaries and start/stop codons"]
    if args.annotation_bed:
        external = annotation_intervals_from_bed_for_windows(
            args.annotation_bed,
            records,
        )
        merged = defaultdict(set)
        for source in (annotation_intervals, external):
            for example_id, intervals in source.items():
                merged[example_id].update(intervals)
        annotation_intervals = {
            example_id: tuple(sorted(intervals))
            for example_id, intervals in merged.items()
        }
        annotation_sources.append("frozen multi-source annotation BED")
    candidates = generate_candidate_sites(
        records,
        positions_per_example=args.positions_per_example,
        seed=args.seed,
        annotation_intervals_by_example=annotation_intervals,
    )
    write_jsonl_bundle(
        args.out,
        [site.to_dict() for site in candidates],
        experiment_id="dnabert2_atlas_candidate_sites",
        status="DEVELOPMENT",
        model_identity={
            "hf_id": DNABERT2_HF_ID,
            "revision": DNABERT2_REVISION,
            "used_for_candidate_generation": False,
        },
        input_paths=tuple(
            path
            for path in (args.collection, args.gencode_gtf, args.annotation_bed)
            if path is not None
        ),
        parameters={
            "positions_per_example": args.positions_per_example,
            "n_examples": len(records),
            "n_candidates": len(candidates),
            "annotation_exclusion_sources": annotation_sources,
            "n_windows_with_source_annotations": len(annotation_intervals),
            "outcome_blind": True,
        },
        seeds={"candidate_sampling": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )


if __name__ == "__main__":
    main()
