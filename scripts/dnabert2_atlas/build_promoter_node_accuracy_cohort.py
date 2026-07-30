#!/usr/bin/env python
"""Bind the final target-outcome-unused loci for a CLS node-accuracy atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-cohort", required=True)
    parser.add_argument("--original-cohort", required=True)
    parser.add_argument("--D7-supply", required=True)
    parser.add_argument("--D8-supply", required=True)
    parser.add_argument("--D9-supply", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohort-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    if config.get("status") != "DRAFT":
        raise RuntimeError("outcome-blind cohort build requires the DRAFT config")
    expanded = _read_jsonl(args.expanded_cohort)
    original = _read_jsonl(args.original_cohort)
    prior_audits = [
        json.loads(Path(path).read_text())
        for path in (args.D7_supply, args.D8_supply, args.D9_supply)
    ]
    expanded_by_id = {str(row["example_id"]): row for row in expanded}
    excluded_ids = {str(row["example_id"]) for row in original}
    excluded_clusters = {str(row["homology_cluster"]) for row in original}
    for audit in prior_audits:
        for values in audit["selection"]["selected_example_ids"].values():
            for value in values:
                example_id = str(value)
                excluded_ids.add(example_id)
                excluded_clusters.add(
                    str(expanded_by_id[example_id]["homology_cluster"])
                )
    eligible = [
        dict(row)
        for row in expanded
        if str(row["example_id"]) not in excluded_ids
        and str(row["homology_cluster"]) not in excluded_clusters
    ]
    if len({str(row["homology_cluster"]) for row in eligible}) != len(eligible):
        raise RuntimeError("eligible node-atlas supply contains duplicate blocks")
    required = 2 * int(config["cohort"]["pairs_per_split"])
    if len(eligible) < required:
        raise RuntimeError(
            f"only {len(eligible)} untouched blocks remain; {required} required"
        )
    tag = str(config["cohort"]["new_split_tag"])
    seed = int(config["inference"]["seed"])
    selected = sorted(
        eligible,
        key=lambda row: hashlib.sha256(
            f"{seed}|{tag}|{row['homology_cluster']}|{row['example_id']}".encode()
        ).hexdigest(),
    )[:required]
    per_split = int(config["cohort"]["pairs_per_split"])
    rows = []
    selected_ids: dict[str, list[str]] = {
        "development": [],
        "confirmation": [],
    }
    for index, source_row in enumerate(selected):
        split = "development" if index < per_split else "confirmation"
        row = dict(source_row)
        row["source_split"] = str(row["split"])
        row["split"] = split
        row["node_accuracy_selection"] = {
            "tag": tag,
            "rank": index,
            "target_outcome_used_for_selection": False,
        }
        rows.append(row)
        selected_ids[split].append(str(row["example_id"]))
    split_clusters = {
        split: {str(row["homology_cluster"]) for row in rows if row["split"] == split}
        for split in ("development", "confirmation")
    }
    checks = {
        "exact_pairs_per_split": all(
            len(selected_ids[split]) == per_split
            for split in ("development", "confirmation")
        ),
        "unique_locus_blocks": len({str(row["homology_cluster"]) for row in rows})
        == len(rows),
        "split_disjoint": not (
            split_clusters["development"] & split_clusters["confirmation"]
        ),
        "fresh_example_ids": all(
            str(row["example_id"]) not in excluded_ids for row in rows
        ),
        "fresh_1Mb_blocks": all(
            str(row["homology_cluster"]) not in excluded_clusters for row in rows
        ),
        "outcome_blind_selection": all(
            not row["node_accuracy_selection"]["target_outcome_used_for_selection"]
            for row in rows
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"node-accuracy cohort failed: {checks}")
    inputs = (
        args.expanded_cohort,
        args.original_cohort,
        args.D7_supply,
        args.D8_supply,
        args.D9_supply,
        args.config,
    )
    model_identity = {
        "candidate": config["checkpoint"]["candidate"],
        "revision": config["checkpoint"]["revision"],
    }
    cohort_path, cohort_manifest = write_jsonl_bundle(
        args.cohort_out,
        rows,
        experiment_id="dnabert2_promoter_node_accuracy_cohort_v1",
        status="PREFLIGHT",
        model_identity=model_identity,
        input_paths=inputs,
        parameters={
            "selection_tag": tag,
            "pairs_per_split": per_split,
            "model_outcomes_computed": False,
        },
        seeds={"selection": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "phase": "PROMOTER_NODE_ACCURACY_OUTCOME_BLIND_SUPPLY",
        "status": "QUALIFIED",
        "model_outcomes_computed": False,
        "offered_untouched_blocks": len(eligible),
        "selected_pairs_per_split": per_split,
        "selection": {
            "tag": tag,
            "selected_example_ids": selected_ids,
            "selected_id_sha256": {
                split: hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()
                for split, values in selected_ids.items()
            },
        },
        "source_split_counts": {
            source_split: sum(row["source_split"] == source_split for row in rows)
            for source_split in ("development", "confirmation")
        },
        "checks": checks,
        "cohort": {
            "path": str(cohort_path),
            "manifest": str(cohort_manifest),
            "rows": len(rows),
        },
        "boundary": (
            "Selection uses identities, 1-Mb blocks, and a stable hash only. "
            "No classifier margin or internal activation was computed."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_node_accuracy_supply_v1",
        status="PREFLIGHT",
        model_identity=model_identity,
        input_paths=(*inputs, cohort_path, cohort_manifest),
        parameters={"cohort": config["cohort"]},
        seeds={"selection": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
