#!/usr/bin/env python
"""Bind fresh promoter-decomposition loci and fixed Q/K/V token dose."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    load_dnabert2_promoter_classifier,
    tokenize_promoter_pairs,
)
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.promoter_head_analysis import select_locus_rows
from gfmcd.reconstruction_atlas.promoter_head_scan import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-cohort", required=True)
    parser.add_argument("--original-cohort", required=True)
    parser.add_argument("--cohort-audit", required=True)
    parser.add_argument("--cohort-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-token-dose", type=int, required=True)
    parser.add_argument("--pairs-per-split", type=int, default=192)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = json.loads(Path(args.cohort_config).read_text())
    expanded = read_jsonl(args.expanded_cohort)
    original = read_jsonl(args.original_cohort)
    original_ids = {str(row["example_id"]) for row in original}
    original_clusters = {str(row["homology_cluster"]) for row in original}
    selected: dict[str, list[dict[str, Any]]] = {}
    for split in ("development", "confirmation"):
        fresh = [
            row
            for row in expanded
            if row["split"] == split
            and str(row["example_id"]) not in original_ids
            and str(row["homology_cluster"]) not in original_clusters
        ]
        selected[split] = select_locus_rows(
            fresh,
            split=split,
            maximum=args.pairs_per_split,
            seed=int(config["seed"]),
            tag="promoter-address-payload-v1",
        )
    if {str(row["homology_cluster"]) for row in selected["development"]} & {
        str(row["homology_cluster"]) for row in selected["confirmation"]
    }:
        raise RuntimeError("fresh promoter decomposition splits share a locus block")

    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    token_counts: list[int] = []
    genomic_counts: list[int] = []
    maximum_tokens = 0
    truncated = 0
    for split in ("development", "confirmation"):
        rows = selected[split]
        for start in range(0, len(rows), 64):
            chunk = rows[start : start + 64]
            pairs = [
                (
                    str(row["example_id"]),
                    str(row["target"]["clean_sequence"]),
                    str(row["control"]["clean_sequence"]),
                )
                for row in chunk
            ]
            batch = tokenize_promoter_pairs(bundle, pairs)
            maximum_tokens = max(maximum_tokens, max(batch.token_counts))
            for sequence_index, count in enumerate(batch.token_counts):
                token_counts.append(count)
                offsets = batch.offset_mapping[sequence_index, :count].tolist()
                genomic_counts.append(
                    sum(int(end) > int(begin) for begin, end in offsets)
                )
                truncated += int(count > 256)
    checks = {
        "fresh_example_ids": all(
            str(row["example_id"]) not in original_ids
            for rows in selected.values()
            for row in rows
        ),
        "fresh_locus_blocks": all(
            str(row["homology_cluster"]) not in original_clusters
            for rows in selected.values()
            for row in rows
        ),
        "unique_locus_blocks": all(
            len({str(row["homology_cluster"]) for row in rows}) == len(rows)
            for rows in selected.values()
        ),
        "split_disjoint": not (
            {str(row["homology_cluster"]) for row in selected["development"]}
            & {str(row["homology_cluster"]) for row in selected["confirmation"]}
        ),
        "fixed_source_token_dose_supported": min(genomic_counts)
        >= args.source_token_dose,
        "maximum_token_geometry": maximum_tokens <= 256,
        "zero_truncation": truncated == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"promoter decomposition supply failed: {checks}")
    selected_ids = {
        split: [str(row["example_id"]) for row in rows]
        for split, rows in selected.items()
    }
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "phase": "PROMOTER_ADDRESS_PAYLOAD_OUTCOME_BLIND_SUPPLY",
        "status": "QUALIFIED",
        "model_outcomes_computed": False,
        "selection": {
            "tag": "promoter-address-payload-v1",
            "pairs_per_split": args.pairs_per_split,
            "selected_example_ids": selected_ids,
            "selected_id_sha256": {
                split: hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()
                for split, values in selected_ids.items()
            },
            "unique_locus_blocks": {
                split: len({str(row["homology_cluster"]) for row in rows})
                for split, rows in selected.items()
            },
        },
        "token_geometry": {
            "fixed_source_token_dose": args.source_token_dose,
            "genomic_tokens": _summary(genomic_counts),
            "all_valid_tokens": _summary(token_counts),
            "maximum_valid_tokens": maximum_tokens,
            "truncated_inputs": truncated,
        },
        "checks": checks,
        "boundary": (
            "This artifact binds outcome-blind loci and token geometry only. "
            "No classifier margin or internal intervention outcome was computed."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_decomposition_supply_audit_v1",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=(
            args.expanded_cohort,
            args.original_cohort,
            args.cohort_audit,
            args.cohort_config,
        ),
        parameters={
            "selection": payload["selection"],
            "source_token_dose": args.source_token_dose,
            "model_outcomes_computed": False,
        },
        seeds={"selection": int(config["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _summary(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


if __name__ == "__main__":
    main()
