#!/usr/bin/env python
"""Benchmark compatible promoter heads and audit leakage into the existing EPD cohort."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    PROMOTER_CHECKPOINTS,
    PROMOTER_GUE_DATASET_ID,
    PROMOTER_GUE_DATASET_REVISION,
    Dnabert2PromoterClassifier,
    load_dnabert2_promoter_classifier,
)
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.promoter_assay import (
    binary_classification_metrics,
    read_gue_sequences,
)
from gfmcd.reconstruction_atlas.schema import load_collection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gue-root", required=True)
    parser.add_argument("--epd-collection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    gue_root = Path(args.gue_root)
    collection = [
        record
        for record in load_collection(args.epd_collection)
        if record.metadata.get("target_site_class") == "promoter.epd_tss_core"
    ]
    candidate_results: dict[str, Any] = {}
    model_identities: dict[str, Any] = {}
    input_paths: list[str | Path] = [args.epd_collection]
    for candidate_name in ("prom_core_all", "prom_300_all"):
        test_path = gue_root / candidate_name / "test.csv"
        input_paths.append(test_path)
        rows = list(csv.DictReader(test_path.open(newline="", encoding="utf-8")))
        bundle = load_dnabert2_promoter_classifier(candidate_name, device=args.device)
        scorer = Dnabert2PromoterClassifier(bundle)
        scores = scorer.score(
            (
                (f"{candidate_name}|test|{index}", row["sequence"])
                for index, row in enumerate(rows)
            ),
            batch_size=args.batch_size,
        )
        labels = [int(row["label"]) for row in rows]
        margins = [score.margin for score in scores]
        metrics = binary_classification_metrics(labels, margins)
        singleton = scorer.score(
            (
                (f"{candidate_name}|health|{index}", row["sequence"])
                for index, row in enumerate(rows[:16])
            ),
            batch_size=1,
        )
        batched = scorer.score(
            (
                (f"{candidate_name}|health|{index}", row["sequence"])
                for index, row in enumerate(rows[:16])
            ),
            batch_size=16,
        )
        repeated = scorer.score(
            (
                (f"{candidate_name}|health|{index}", row["sequence"])
                for index, row in enumerate(rows[:16])
            ),
            batch_size=16,
        )
        trainer_state_path = bundle.snapshot / "trainer_state.json"
        input_paths.append(trainer_state_path)
        trainer_state = json.loads(trainer_state_path.read_text())
        selected_step = int(
            str(trainer_state["best_model_checkpoint"]).rsplit("-", 1)[1]
        )
        selected_log = next(
            row
            for row in trainer_state["log_history"]
            if int(row.get("step", -1)) == selected_step and "eval_loss" in row
        )
        overlap = _epd_overlap(
            collection,
            gue_root / candidate_name,
            candidate_name=candidate_name,
        )
        candidate_results[candidate_name] = {
            "checkpoint": bundle.identity(),
            "gue_test": {
                **metrics,
                "path": str(test_path),
                "sha256": sha256_file(test_path),
                "token_count": {
                    "min": min(score.token_count for score in scores),
                    "median": statistics.median(score.token_count for score in scores),
                    "max": max(score.token_count for score in scores),
                    "above_frozen_256_token_ceiling": sum(
                        score.token_count > 256 for score in scores
                    ),
                },
            },
            "trainer_selected_development_checkpoint": {
                "selection_metric": "eval_loss",
                "selected_step": selected_step,
                "eval_loss": selected_log["eval_loss"],
                "eval_matthews_correlation": selected_log["eval_matthews_correlation"],
                "eval_accuracy": selected_log["eval_accuracy"],
                "trainer_state_sha256": sha256_file(trainer_state_path),
            },
            "model_health": {
                "maximum_batch_invariance_absolute_margin_difference": max(
                    abs(left.margin - right.margin)
                    for left, right in zip(singleton, batched)
                ),
                "maximum_repeat_absolute_margin_difference": max(
                    abs(left.margin - right.margin)
                    for left, right in zip(batched, repeated)
                ),
                "nonfinite_scores": sum(not _finite(score.margin) for score in scores),
            },
            "epdnew_exact_overlap": overlap,
        }
        model_identities[candidate_name] = bundle.identity()
        del scorer, bundle
        gc.collect()

    selected = "prom_300_all"
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "phase": "PROMOTER_CLASSIFIER_CHECKPOINT_AUDIT",
        "status": "PREFLIGHT",
        "outcome_independence": (
            "checkpoint selection used only each checkpoint's native frozen GUE test task; "
            "no external-atlas classifier outcome was inspected"
        ),
        "dataset": {
            "hf_id": PROMOTER_GUE_DATASET_ID,
            "revision": PROMOTER_GUE_DATASET_REVISION,
        },
        "candidate_results": candidate_results,
        "selection": {
            "selected_candidate": selected,
            "rule": (
                "among compatible human all-promoter DNABERT-2 heads, select the exact "
                "300-nt task with the greater held-out MCC, subject to AUROC >= 0.95, "
                "MCC >= 0.80, complete head, deterministic finite scores, and no truncation"
            ),
            "observed_test_mcc": {
                name: result["gue_test"]["mcc"]
                for name, result in candidate_results.items()
            },
            "observed_test_auroc": {
                name: result["gue_test"]["auroc"]
                for name, result in candidate_results.items()
            },
            "selected": True,
        },
        "current_epd_cohort_decision": {
            "eligible_for_target_inference": False,
            "reason": (
                "the selected 300-nt EPDnew windows all occur verbatim in the GUE "
                "train/dev/test corpus; classifier behavior there is training-source "
                "contaminated"
            ),
            "required_replacement": (
                "a GRCh38 TSS cohort with exact forward/reverse GUE exclusion before "
                "development or confirmation scoring"
            ),
        },
        "rejected_alternatives": [
            {
                "hf_id": "zhangtaolab/dnabert2-promoter",
                "reason": "plant active-core-promoter task and modified 8k tokenizer",
            },
            {
                "hf_id": "rahuls472/DNABERT-Promoter-Classifier",
                "reason": "DNABERT-1 fixed 6-mer backbone, not DNABERT-2 BPE",
            },
            {
                "hf_id": "rongkuanjiang/dnabert-ag-promoter",
                "reason": "undocumented DNABERT-AG/LSTM head with no auditable training dataset",
            },
            {
                "hf_id": PROMOTER_CHECKPOINTS["prom_core_all"].hf_id,
                "reason": (
                    "compatible secondary candidate, but lower native held-out MCC and "
                    "768/768 best-aligned exact EPD overlaps"
                ),
            },
        ],
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_classifier_checkpoint_audit_v1",
        status="PREFLIGHT",
        model_identity={
            "selected": model_identities[selected],
            "audited_candidates": model_identities,
        },
        input_paths=input_paths,
        parameters={
            "candidate_family": ["prom_core_all", "prom_300_all"],
            "selection_metric": "native_GUE_test_MCC",
            "selection_floor_mcc": 0.80,
            "selection_floor_auroc": 0.95,
            "epd_alignment_offsets_scanned_nt": [-2, -1, 0, 1, 2],
            "batch_size": args.batch_size,
        },
        seeds={"none": 0},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "metrics": {
                    name: {
                        key: result["gue_test"][key]
                        for key in ("accuracy", "mcc", "auroc")
                    }
                    for name, result in candidate_results.items()
                },
                "epd_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _epd_overlap(
    collection: list[Any],
    task_root: Path,
    *,
    candidate_name: str,
) -> dict[str, Any]:
    paths = [task_root / f"{split}.csv" for split in ("train", "dev", "test")]
    index = read_gue_sequences(paths)
    geometry = PROMOTER_CHECKPOINTS[candidate_name]
    offsets = range(-2, 3)
    offset_counts: dict[int, int] = {}
    rows_by_offset: dict[int, list[tuple[str, list[tuple[str, int]]]]] = {}
    for offset in offsets:
        matches = []
        for record in collection:
            element = record.element(str(record.metadata["target_element_id"]))
            center = (element.interval.start + element.interval.end) // 2 + offset
            start = center - geometry.tss_upstream_nt
            end = center + geometry.tss_downstream_nt + 1
            sequence = record.sequence[start:end]
            hits = index.get(sequence, [])
            if hits:
                matches.append((record.split, hits))
        offset_counts[offset] = len(matches)
        rows_by_offset[offset] = matches
    best_offset = max(offsets, key=lambda offset: (offset_counts[offset], -abs(offset)))
    matched = rows_by_offset[best_offset]
    split_counts = Counter(split for split, _ in matched)
    gue_membership = Counter(
        f"{atlas_split}|{gue_split}|label{label}"
        for atlas_split, hits in matched
        for gue_split, label in hits
    )
    return {
        "n_epd_examples": len(collection),
        "alignment_offset_counts": {
            str(offset): offset_counts[offset] for offset in offsets
        },
        "best_alignment_offset_nt": best_offset,
        "best_alignment_exact_matches": len(matched),
        "best_alignment_exact_match_fraction": len(matched) / len(collection),
        "atlas_split_counts": dict(sorted(split_counts.items())),
        "gue_membership_counts": dict(sorted(gue_membership.items())),
    }


def _finite(value: float) -> bool:
    import math

    return math.isfinite(value)


if __name__ == "__main__":
    main()
