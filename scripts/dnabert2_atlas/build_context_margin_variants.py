#!/usr/bin/env python
"""Build outcome-blind, target-preserving context variants for the atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    DNABERT2_REVISION,
    load_dnabert2_tokenizer,
)
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.context import (
    ContextWindowSpec,
    build_target_preserving_context_variants,
)
from gfmcd.reconstruction_atlas.schema import load_collection


class TokenizerEncoder:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def encode(self, sequence: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self.tokenizer(
            sequence,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        return (
            [int(value) for value in encoded["input_ids"]],
            [tuple(int(x) for x in value) for value in encoded["offset_mapping"]],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants-out", required=True)
    parser.add_argument("--audit-out", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    seed = int(args.seed if args.seed is not None else config["seed"])
    class_configs = config["context_intervention"]["classes"]
    candidate_shuffles = int(config["context_intervention"]["candidate_shuffles"])
    records = load_collection(args.collection)
    tokenizer_bundle = load_dnabert2_tokenizer()
    encoder = TokenizerEncoder(tokenizer_bundle.tokenizer)

    rows: list[dict[str, Any]] = []
    offered: Counter[tuple[str, str]] = Counter()
    eligible: Counter[tuple[str, str]] = Counter()
    failures: Counter[tuple[str, str]] = Counter()
    alternative_counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    changed_counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    overhang_counts: dict[tuple[str, str], list[int]] = defaultdict(list)

    for record in records:
        site_class = str(record.metadata.get("target_site_class", ""))
        if site_class not in class_configs:
            continue
        offered[(record.split, site_class)] += 1
        element_id = str(record.metadata["target_element_id"])
        element = record.element(element_id)
        item = class_configs[site_class]
        spec = ContextWindowSpec(
            upstream_nt=int(item["upstream_nt"]),
            downstream_nt=int(item["downstream_nt"]),
            irrelevant_gap_nt=int(item["irrelevant_gap_nt"]),
        )
        variants = build_target_preserving_context_variants(
            record.sequence,
            element.interval,
            spec,
            encoder=encoder.encode,
            seed=_example_seed(seed, record.example_id),
            candidate_shuffles=candidate_shuffles,
        )
        if variants is None:
            failures[(record.split, site_class)] += 1
            continue
        key = (record.split, site_class)
        eligible[key] += 1
        alternative_counts[key].append(len(variants.alternatives))
        changed_counts[key].append(variants.structured_changed_nucleotides)
        overhang_counts[key].append(variants.mask_plan.overhang_nucleotides)
        rows.append(
            {
                "example_id": record.example_id,
                "split": record.split,
                "homology_cluster": record.homology_cluster,
                "site_class": site_class,
                "element_id": element_id,
                "target": {
                    "start": element.interval.start,
                    "end": element.interval.end,
                    "sequence": element.sequence,
                },
                "structured_sequence": variants.structured_shuffle,
                "irrelevant_sequence": variants.irrelevant_shuffle,
                "alternatives": variants.alternatives,
                "audit": variants.to_dict(),
            }
        )

    variant_path, variant_manifest = write_jsonl_bundle(
        args.variants_out,
        rows,
        experiment_id="dnabert2_atlas_context_margin_variant_supply",
        status="PREFLIGHT",
        model_identity=tokenizer_bundle.identity(),
        input_paths=(args.collection, args.config),
        parameters={
            "outcome_blind": True,
            "candidate_shuffles": candidate_shuffles,
            "context_operation": "joint_mononucleotide_shuffle",
            "dose_matching": "exact_changed_nucleotide_count",
            "target_geometry": "exact",
            "checkpoint_revision": DNABERT2_REVISION,
        },
        seeds={"variant_construction": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    all_keys = sorted(set(offered) | set(eligible))
    class_rows = {}
    for split, site_class in all_keys:
        key = (split, site_class)
        alt = alternative_counts[key]
        changed = changed_counts[key]
        overhang = overhang_counts[key]
        class_rows[f"{split}|{site_class}"] = {
            "split": split,
            "site_class": site_class,
            "offered": offered[key],
            "eligible": eligible[key],
            "ineligible": failures[key],
            "eligibility_fraction": (
                eligible[key] / offered[key] if offered[key] else 0.0
            ),
            "eligible_alternatives": _range_summary(alt),
            "changed_nucleotides": _range_summary(changed),
            "BPE_overhang_nucleotides": _range_summary(overhang),
        }
    split_clusters: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split_clusters[record.split].add(record.homology_cluster)
    leakage = sorted(
        (left, right, len(split_clusters[left] & split_clusters[right]))
        for index, left in enumerate(sorted(split_clusters))
        for right in sorted(split_clusters)[index + 1 :]
        if split_clusters[left] & split_clusters[right]
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "geometry_qualified_margin_class_specific_context",
        "outcome_blind": True,
        "checkpoint_revision": DNABERT2_REVISION,
        "n_variants": len(rows),
        "by_split_and_class": class_rows,
        "invariants": {
            "target_or_scored_cover_changes": 0,
            "target_geometry_violations": 0,
            "structured_irrelevant_dose_mismatches": 0,
            "split_leakage": leakage,
        },
        "variant_supply": {
            "path": str(variant_path),
            "manifest": str(variant_manifest),
        },
    }
    write_result_bundle(
        args.audit_out,
        payload,
        experiment_id="dnabert2_atlas_context_margin_supply_audit",
        status="PREFLIGHT",
        model_identity=tokenizer_bundle.identity(),
        input_paths=(
            args.collection,
            args.config,
            variant_path,
            variant_manifest,
        ),
        parameters={
            "outcome_blind": True,
            "candidate_shuffles": candidate_shuffles,
        },
        seeds={"variant_construction": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )


def _example_seed(seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{seed}|{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _range_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {"min": ordered[0], "median": median, "max": ordered[-1]}


if __name__ == "__main__":
    main()
