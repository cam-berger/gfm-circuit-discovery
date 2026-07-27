#!/usr/bin/env python
"""Build token/composition-qualified empirical controls from a declared candidate pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gfmcd.adapters.dnabert2 import load_dnabert2_tokenizer
from gfmcd.reconstruction_atlas.artifacts import write_jsonl_bundle
from gfmcd.reconstruction_atlas.controls import (
    MatchingSpec,
    build_matched_sets,
    compute_site_features,
)
from gfmcd.reconstruction_atlas.schema import (
    ControlFamily,
    SiteReference,
    load_collection,
)
from gfmcd.reconstruction_atlas.tokenization import build_mask_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--candidate-sites",
        required=True,
        help="JSONL SiteReference pool produced without model outcomes",
    )
    parser.add_argument("--split", choices=("development",), default="development")
    parser.add_argument("--controls-per-family", type=int, default=1)
    parser.add_argument("--gc-caliper", type=float, default=0.05)
    parser.add_argument("--cpg-caliper", type=float, default=0.05)
    parser.add_argument("--dinucleotide-l1-caliper", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = {
        record.example_id: record
        for record in load_collection(args.collection)
        if record.split == args.split
    }
    if not records:
        raise SystemExit(f"collection contains no {args.split} records")
    target_sites = []
    for record in records.values():
        target_id = record.metadata.get("target_element_id")
        if target_id is None:
            raise SystemExit(f"{record.example_id} has no metadata.target_element_id")
        element = record.element(str(target_id))
        target_sites.append(
            SiteReference(
                example_id=record.example_id,
                interval=element.interval,
                site_class=element.site_class,
                element_id=element.element_id,
            )
        )
    candidates = _load_candidate_sites(args.candidate_sites)
    absent = sorted({site.example_id for site in candidates} - set(records))
    if absent:
        raise SystemExit(
            "candidate sites reference examples outside the selected collection/split: "
            + ", ".join(absent[:10])
        )
    tokenizer_bundle = load_dnabert2_tokenizer()
    tokenizer = tokenizer_bundle.tokenizer

    def encode(sequence: str) -> tuple[list[int], list[tuple[int, int]]]:
        value = tokenizer(
            sequence,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        return (
            [int(x) for x in value["input_ids"]],
            [tuple(int(y) for y in offset) for offset in value["offset_mapping"]],
        )

    encoded = {
        example_id: encode(record.sequence) for example_id, record in records.items()
    }

    def features(site: SiteReference):
        record = records[site.example_id]
        ids, offsets = encoded[site.example_id]
        plan = build_mask_plan(
            record.sequence,
            site.interval,
            input_ids=ids,
            offsets=offsets,
        )
        return compute_site_features(
            site,
            record.sequence,
            plan,
            annotated_intervals=(element.interval for element in record.elements),
        )

    target_features = [features(site) for site in target_sites]
    candidate_features = [features(site) for site in candidates]
    candidate_pools = {
        family: candidate_features
        for family in (
            ControlFamily.SAME_MOTIF_UNANNOTATED,
            ControlFamily.COMPOSITION_TOKEN_MATCHED,
            ControlFamily.SAME_SEQUENCE_NON_SITE,
        )
    }
    spec = MatchingSpec(
        controls_per_family=args.controls_per_family,
        gc_caliper=args.gc_caliper,
        cpg_caliper=args.cpg_caliper,
        dinucleotide_l1_caliper=args.dinucleotide_l1_caliper,
        seed=args.seed,
    )
    matched = build_matched_sets(target_features, candidate_pools, spec)
    write_jsonl_bundle(
        args.out,
        [item.to_dict() for item in matched],
        experiment_id=f"dnabert2_atlas_control_matches_{args.split}",
        status="DEVELOPMENT",
        model_identity=tokenizer_bundle.identity(),
        input_paths=(args.collection, args.candidate_sites),
        parameters={
            "split": args.split,
            "controls_per_family": args.controls_per_family,
            "gc_caliper": args.gc_caliper,
            "cpg_caliper": args.cpg_caliper,
            "dinucleotide_l1_caliper": args.dinucleotide_l1_caliper,
            "n_targets": len(target_features),
            "n_candidates": len(candidate_features),
        },
        seeds={"matching": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )


def _load_candidate_sites(path: str | Path) -> list[SiteReference]:
    sites = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sites.append(SiteReference.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not sites:
        raise ValueError("candidate-site pool is empty")
    return sites


if __name__ == "__main__":
    main()
