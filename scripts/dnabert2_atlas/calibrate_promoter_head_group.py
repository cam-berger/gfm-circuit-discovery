#!/usr/bin/env python
"""Calibrate the frozen candidate-set/null design without GENCODE outcomes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from statistics import NormalDist

from gfmcd.adapters.dnabert2 import head_label, load_dnabert2_promoter_classifier
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.promoter_head_analysis import (
    layer_matched_random_head_sets,
    select_head_candidates,
)
from gfmcd.reconstruction_atlas.promoter_head_scan import (
    evaluate_head_set,
    select_gue_pairs,
)
from gfmcd.reconstruction_atlas.statistics import add_one_permutation_pvalue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--maximum-per-layer", type=int, default=2)
    parser.add_argument("--null-sets", type=int, default=127)
    parser.add_argument("--target-pairs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-pairs", type=int, default=64)
    args = parser.parse_args()

    calibration = json.loads(Path(args.calibration).read_text())
    gue_slice = calibration["GUE_slice"]
    pairs = select_gue_pairs(
        args.gue_test,
        count=int(gue_slice["pairs"]),
        offset=int(gue_slice["offset"]),
        seed=args.seed,
    )
    candidate_rows = select_head_candidates(
        calibration["head_summaries"],
        candidate_count=args.candidate_count,
        maximum_per_layer=args.maximum_per_layer,
    )
    if len(candidate_rows) != args.candidate_count:
        raise RuntimeError(
            f"calibration supplied only {len(candidate_rows)} sign-consistent heads"
        )
    candidates = [(int(row["layer"]), int(row["head"])) for row in candidate_rows]
    null_sets = layer_matched_random_head_sets(
        candidates,
        num_layers=12,
        num_heads=12,
        draws=args.null_sets,
        seed=args.seed ^ 0x5151,
    )
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    candidate_effects = {
        endpoint: evaluate_head_set(
            bundle,
            pairs,
            candidates,
            direction=direction,
            batch_pairs=args.batch_pairs,
        )
        for endpoint, direction in (
            ("necessity", "target_from_control"),
            ("sufficiency", "control_from_target"),
        )
    }
    null_means = {"necessity": [], "sufficiency": []}
    for index, null_set in enumerate(null_sets, start=1):
        for endpoint, direction in (
            ("necessity", "target_from_control"),
            ("sufficiency", "control_from_target"),
        ):
            null_means[endpoint].append(
                statistics.mean(
                    evaluate_head_set(
                        bundle,
                        pairs,
                        null_set,
                        direction=direction,
                        batch_pairs=args.batch_pairs,
                    )
                )
            )
        if index % 12 == 0:
            print(f"calibration null set {index}/{len(null_sets)}", flush=True)
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "phase": "PROMOTER_HEAD_ATLAS_GROUP_CALIBRATION",
        "status": "PREFLIGHT",
        "target_GENCODE_outcomes_computed": False,
        "calibration": args.calibration,
        "candidate_rule": {
            "candidate_count": args.candidate_count,
            "maximum_per_layer": args.maximum_per_layer,
            "four_way_positive_sign_required": True,
            "rank_aggregation": (
                "mean fractional rank over exact necessity, exact sufficiency, "
                "necessity attribution, and sufficiency attribution"
            ),
        },
        "calibration_candidates": [head_label(head) for head in candidates],
        "layer_profile": {
            str(layer): sum(candidate[0] == layer for candidate in candidates)
            for layer in sorted({head[0] for head in candidates})
        },
        "matched_null": {
            "requested": args.null_sets,
            "distinct_evaluated": len(null_sets),
            "candidate_excluded": True,
            "same_layer_profile": True,
        },
        "endpoints": {
            endpoint: {
                "candidate": _summary(values),
                "prospective_MDE_at_target_n": _prospective_mde(
                    values, target_n=args.target_pairs
                ),
                "null_set_mean": _summary(null_means[endpoint]),
                "p_add_one_vs_layer_matched_sets": add_one_permutation_pvalue(
                    statistics.mean(values),
                    null_means[endpoint],
                    alternative="greater",
                ),
            }
            for endpoint, values in candidate_effects.items()
        },
        "target_pairs_for_power_projection": args.target_pairs,
        "boundary": (
            "Candidates here are selected and evaluated only on the reserved GUE "
            "calibration slice. They set instrument thresholds, not GENCODE head claims."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_head_group_calibration_v1",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=(args.gue_test, args.calibration),
        parameters={
            "candidate_count": args.candidate_count,
            "maximum_per_layer": args.maximum_per_layer,
            "null_sets": len(null_sets),
            "target_pairs": args.target_pairs,
        },
        seeds={"calibration_and_null_sets": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _prospective_mde(values: list[float], *, target_n: int) -> float:
    if len(values) < 2 or target_n < 2:
        raise ValueError(
            "MDE requires at least two calibration and target observations"
        )
    z = NormalDist()
    return (
        (z.inv_cdf(0.975) + z.inv_cdf(0.8)) * statistics.stdev(values) / target_n**0.5
    )


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


if __name__ == "__main__":
    main()
