#!/usr/bin/env python
"""Calibrate promoter CLS-head attribution and patching on a disjoint GUE slice."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    all_promoter_heads,
    head_label,
    load_dnabert2_promoter_classifier,
)
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.promoter_head_analysis import spearman_correlation
from gfmcd.reconstruction_atlas.promoter_head_scan import (
    evaluate_head_set,
    native_pair_rows,
    scan_promoter_heads,
    select_gue_pairs,
    within_class_gue_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patch-batch-pairs", type=int, default=64)
    parser.add_argument("--gradient-batch-pairs", type=int, default=8)
    args = parser.parse_args()

    pairs = select_gue_pairs(
        args.gue_test,
        count=args.pairs,
        offset=args.offset,
        seed=args.seed,
    )
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    heads = all_promoter_heads(bundle)
    scan = scan_promoter_heads(
        bundle,
        pairs,
        patch_batch_pairs=args.patch_batch_pairs,
        gradient_batch_pairs=args.gradient_batch_pairs,
        include_attention=False,
        progress=lambda message: print(message, flush=True),
    )
    summaries = _head_summaries(scan["head_pair_rows"])
    correlations = {
        endpoint: spearman_correlation(
            [row[f"attribution_{endpoint}_mean"] for row in summaries],
            [row[f"{endpoint}_mean"] for row in summaries],
        )
        for endpoint in ("necessity", "sufficiency")
    }
    full_set_effects = {
        direction: evaluate_head_set(
            bundle,
            pairs,
            heads,
            direction=direction,
            batch_pairs=args.patch_batch_pairs,
        )
        for direction in ("target_from_control", "control_from_target")
    }
    within_class = within_class_gue_pairs(pairs)
    within_class_effects = evaluate_head_set(
        bundle,
        within_class,
        heads,
        direction="target_from_control",
        batch_pairs=args.patch_batch_pairs,
    )
    noop_effects = evaluate_head_set(
        bundle,
        pairs,
        heads,
        direction="target_from_self",
        batch_pairs=args.patch_batch_pairs,
    )
    singleton_native = native_pair_rows(bundle, pairs, batch_pairs=1)
    batched_native = native_pair_rows(bundle, pairs, batch_pairs=args.patch_batch_pairs)
    maximum_batch_difference = max(
        abs(float(left[key]) - float(right[key]))
        for left, right in zip(singleton_native, batched_native)
        for key in ("target_margin", "control_margin")
    )
    raw_rows = [
        *({"row_type": "pair", **row} for row in scan["pair_rows"]),
        *({"row_type": "head_pair", **row} for row in scan["head_pair_rows"]),
    ]
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id="dnabert2_promoter_head_atlas_calibration_raw_v1",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=(args.gue_test,),
        parameters={
            "target_outcomes_computed": False,
            "pairs": args.pairs,
            "offset": args.offset,
            "measurement": "paired_CLS_head_activation_patch_and_first_order_attribution",
        },
        seeds={"calibration": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "phase": "PROMOTER_HEAD_ATLAS_CALIBRATION",
        "status": "PREFLIGHT",
        "target_GENCODE_outcomes_computed": False,
        "checkpoint": bundle.identity(),
        "GUE_slice": {
            "pairs": args.pairs,
            "offset": args.offset,
            "pair_ids": [pair[0] for pair in pairs],
        },
        "native": _native_summary(scan["pair_rows"]),
        "full_144_head_CLS_patch": {
            "necessity": _summary(full_set_effects["target_from_control"]),
            "sufficiency": _summary(full_set_effects["control_from_target"]),
        },
        "same_class_full_patch_negative_control": {
            "n": len(within_class_effects),
            "signed": _summary(within_class_effects),
            "mean_absolute_effect": statistics.mean(
                abs(value) for value in within_class_effects
            ),
        },
        "no_op_and_numerical": {
            "maximum_absolute_self_patch_effect": max(
                abs(value) for value in noop_effects
            ),
            "maximum_batch_invariance_absolute_margin_difference": (
                maximum_batch_difference
            ),
            "all_values_finite": _all_finite(
                scan["pair_rows"],
                scan["head_pair_rows"],
                full_set_effects,
                within_class_effects,
                noop_effects,
            ),
        },
        "attribution_exact_patch_spearman": correlations,
        "head_summaries": summaries,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(raw_rows),
        },
        "boundary": (
            "Calibration uses only a reserved GUE slice. Attribution ranks candidates; "
            "the exact paired patch is the intervention."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_head_atlas_calibration_v1",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=(args.gue_test, raw_path, raw_manifest),
        parameters={
            "pairs": args.pairs,
            "offset": args.offset,
            "patch_batch_pairs": args.patch_batch_pairs,
            "gradient_batch_pairs": args.gradient_batch_pairs,
        },
        seeds={"calibration": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "native": payload["native"],
                "full_patch": payload["full_144_head_CLS_patch"],
                "negative_control": payload["same_class_full_patch_negative_control"],
                "spearman": correlations,
                "numerical": payload["no_op_and_numerical"],
                "top_heads": [row["head_label"] for row in summaries[:8]],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _head_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), int(row["head"]))].append(row)
    output = []
    for head, head_rows in grouped.items():
        output.append(
            {
                "layer": head[0],
                "head": head[1],
                "head_label": head_label(head),
                **{
                    f"{key}_mean": statistics.mean(float(row[key]) for row in head_rows)
                    for key in (
                        "necessity",
                        "sufficiency",
                        "attribution_necessity",
                        "attribution_sufficiency",
                    )
                },
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -statistics.mean(
                (
                    row["necessity_mean"],
                    row["sufficiency_mean"],
                    row["attribution_necessity_mean"],
                    row["attribution_sufficiency_mean"],
                )
            ),
            row["layer"],
            row["head"],
        ),
    )


def _native_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "mean_target_margin": statistics.mean(
            float(row["target_margin"]) for row in rows
        ),
        "mean_control_margin": statistics.mean(
            float(row["control_margin"]) for row in rows
        ),
        "mean_clean_enrichment": statistics.mean(
            float(row["clean_enrichment"]) for row in rows
        ),
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _all_finite(*objects: Any) -> bool:
    import math

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            return all(walk(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(walk(item) for item in value)
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        return True

    return all(walk(value) for value in objects)


if __name__ == "__main__":
    main()
