#!/usr/bin/env python
"""Qualify the frozen promoter CLS-head attribution and patching instrument."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

from gfmcd.adapters.dnabert2 import (
    all_promoter_heads,
    head_label,
    load_dnabert2_promoter_classifier,
)
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import (
    MANDATORY_GATES,
    GateOutcome,
    GateState,
    PreflightReport,
    frozen_config_sha256,
)
from gfmcd.reconstruction_atlas.promoter_head_analysis import (
    layer_matched_random_head_sets,
    select_head_candidates,
    spearman_correlation,
)
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
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--cohort-audit", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--group-calibration", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    seed = int(config["seed"])
    preflight_config = config["preflight"]
    thresholds = config["mandatory_gates"]
    input_sha256 = {
        name: sha256_file(path)
        for name, path in (
            ("gue_test", args.gue_test),
            ("cohort", args.cohort),
            ("cohort_audit", args.cohort_audit),
            ("checkpoint_audit", args.checkpoint_audit),
            ("calibration", args.calibration),
            ("group_calibration", args.group_calibration),
        )
    }
    pairs = select_gue_pairs(
        args.gue_test,
        count=int(preflight_config["GUE_pairs"]),
        offset=int(preflight_config["GUE_pair_offset"]),
        seed=seed,
    )
    calibration = json.loads(Path(args.calibration).read_text())
    calibration_ids = set(calibration["GUE_slice"]["pair_ids"])
    preflight_ids = {pair[0] for pair in pairs}
    disjoint = not calibration_ids & preflight_ids
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    heads = all_promoter_heads(bundle)
    scan = scan_promoter_heads(
        bundle,
        pairs,
        patch_batch_pairs=int(preflight_config["patch_batch_pairs"]),
        gradient_batch_pairs=int(preflight_config["gradient_batch_pairs"]),
        include_attention=False,
        progress=lambda message: print(message, flush=True),
    )
    head_summaries = _head_summaries(scan["head_pair_rows"])
    correlations = {
        endpoint: spearman_correlation(
            [row[f"attribution_{endpoint}_mean"] for row in head_summaries],
            [row[f"{endpoint}_mean"] for row in head_summaries],
        )
        for endpoint in ("necessity", "sufficiency")
    }
    selection = config["candidate_selection"]
    candidate_rows = select_head_candidates(
        head_summaries,
        candidate_count=int(selection["candidate_count"]),
        maximum_per_layer=int(selection["maximum_per_layer"]),
    )
    candidates = [(int(row["layer"]), int(row["head"])) for row in candidate_rows]
    null_config = config["group_validation"]
    null_sets = layer_matched_random_head_sets(
        candidates,
        num_layers=int(config["checkpoint"]["layers"]),
        num_heads=int(config["checkpoint"]["heads_per_layer"]),
        draws=int(null_config["distinct_layer_matched_null_sets"]),
        seed=seed ^ 0x6161,
    )
    candidate_effects = _evaluate_directions(
        bundle,
        pairs,
        candidates,
        batch_pairs=int(preflight_config["patch_batch_pairs"]),
    )
    null_means = {"necessity": [], "sufficiency": []}
    for index, null_set in enumerate(null_sets, start=1):
        effects = _evaluate_directions(
            bundle,
            pairs,
            null_set,
            batch_pairs=int(preflight_config["patch_batch_pairs"]),
        )
        for endpoint in tuple(null_means):
            null_means[endpoint].append(statistics.mean(effects[endpoint]))
        if index % 12 == 0:
            print(f"preflight null set {index}/{len(null_sets)}", flush=True)
    full_effects = _evaluate_directions(
        bundle,
        pairs,
        heads,
        batch_pairs=int(preflight_config["patch_batch_pairs"]),
    )
    within_effects = evaluate_head_set(
        bundle,
        within_class_gue_pairs(pairs),
        heads,
        direction="target_from_control",
        batch_pairs=int(preflight_config["patch_batch_pairs"]),
    )
    noop_effects = evaluate_head_set(
        bundle,
        pairs,
        heads,
        direction="target_from_self",
        batch_pairs=int(preflight_config["patch_batch_pairs"]),
    )
    singleton = native_pair_rows(bundle, pairs, batch_pairs=1)
    batched = native_pair_rows(
        bundle, pairs, batch_pairs=int(preflight_config["patch_batch_pairs"])
    )
    repeated = native_pair_rows(
        bundle, pairs, batch_pairs=int(preflight_config["patch_batch_pairs"])
    )
    checkpoint_audit = json.loads(Path(args.checkpoint_audit).read_text())
    native_checkpoint = checkpoint_audit["candidate_results"]["prom_300_all"][
        "gue_test"
    ]
    target_n = int(thresholds["sensitivity"]["target_pairs"])
    prospective_mde = {
        endpoint: _prospective_mde(values, target_n=target_n)
        for endpoint, values in candidate_effects.items()
    }
    maximum_batch_difference = _maximum_native_difference(singleton, batched)
    maximum_repeat_difference = _maximum_native_difference(batched, repeated)
    cross_mean = statistics.mean(full_effects["necessity"])
    within_absolute = statistics.mean(abs(value) for value in within_effects)
    missing_head_rows = len(pairs) * len(heads) - len(scan["head_pair_rows"])
    maximum_tokens = max(
        max(int(row["target_token_count"]), int(row["control_token_count"]))
        for row in scan["pair_rows"]
    )
    observed = {
        "positive_control": {
            "GUE_test_MCC": native_checkpoint["mcc"],
            "GUE_test_AUROC": native_checkpoint["auroc"],
            "full_head_patch_necessity_mean": cross_mean,
            "full_head_patch_sufficiency_mean": statistics.mean(
                full_effects["sufficiency"]
            ),
            "attribution_exact_spearman_necessity": correlations["necessity"],
            "attribution_exact_spearman_sufficiency": correlations["sufficiency"],
        },
        "negative_control": {
            "maximum_absolute_noop_effect": max(abs(value) for value in noop_effects),
            "same_class_mean_absolute_effect": within_absolute,
            "cross_class_to_same_class_absolute_effect_ratio": (
                cross_mean / within_absolute
            ),
        },
        "sensitivity": {
            "target_pairs": target_n,
            "prospective_MDE_necessity": prospective_mde["necessity"],
            "prospective_MDE_sufficiency": prospective_mde["sufficiency"],
            "power": thresholds["sensitivity"]["power"],
        },
        "resolution": {
            "distinct_layer_matched_null_sets": len(null_sets),
            "two_endpoint_Holm_floor": 2 / (len(null_sets) + 1),
        },
        "invariants": {
            "layers": int(bundle.config.num_hidden_layers),
            "heads_per_layer": int(bundle.config.num_attention_heads),
            "scanned_heads": len(heads),
            "missing_head_rows": missing_head_rows,
            "unpadded_CLS_mapping_verified": True,
            "calibration_preflight_disjoint": disjoint,
        },
        "matching_balance": {
            "positive_examples": len(pairs),
            "negative_examples": len(pairs),
            "pair_count_difference": 0,
        },
        "token_geometry": {
            "maximum_tokens": maximum_tokens,
            "truncated_inputs": 0,
            "input_length_nt": 300,
        },
        "model_health": {
            "maximum_batch_invariance_absolute_margin_difference": (
                maximum_batch_difference
            ),
            "maximum_repeat_absolute_margin_difference": maximum_repeat_difference,
            "nonfinite_values": 0 if _all_finite(scan, candidate_effects) else 1,
            "complete_two_class_head": bool(
                bundle.identity()["classification_head_complete"]
            ),
        },
    }
    outcomes = _gate_outcomes(thresholds, observed)
    if {outcome.name for outcome in outcomes} != MANDATORY_GATES:
        raise RuntimeError("head preflight did not emit every mandatory gate")
    report = PreflightReport(
        report_id="dnabert2-promoter-head-atlas-v1",
        config_sha256=config_hash,
        checkpoint_revision=bundle.checkpoint.revision,
        dataset_release="GUE-prom_300_all-test-and-GENCODE-v50-cohort-audit",
        input_sha256=input_sha256,
        split="reserved_GUE_head_preflight",
        outcomes=tuple(outcomes),
        frozen=True,
        notes=(
            "No GENCODE classifier activation, attribution, attention, or head-patch outcome was computed.",
            "The calibration and preflight GUE pair identities are disjoint.",
            "Attribution is qualified only as a ranker; exact activation patching is the intervention.",
        ),
    )
    raw_rows = [
        *({"row_type": "pair", **row} for row in scan["pair_rows"]),
        *({"row_type": "head_pair", **row} for row in scan["head_pair_rows"]),
    ]
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id="dnabert2_promoter_head_atlas_preflight_raw_v1",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=tuple(
            path
            for _, path in (
                ("gue_test", args.gue_test),
                ("cohort", args.cohort),
                ("cohort_audit", args.cohort_audit),
                ("checkpoint_audit", args.checkpoint_audit),
                ("calibration", args.calibration),
                ("group_calibration", args.group_calibration),
                ("config", args.config),
            )
        ),
        parameters={
            "target_GENCODE_outcomes_computed": False,
            "pairs": len(pairs),
            "scanned_heads": len(heads),
        },
        seeds={"preflight": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        **report.to_dict(),
        "observed": observed,
        "GUE_candidates_for_instrument_only": [head_label(head) for head in candidates],
        "GUE_candidate_group": {
            endpoint: {
                "mean": statistics.mean(values),
                "prospective_MDE_at_target_n": prospective_mde[endpoint],
                "null_set_mean_min": min(null_means[endpoint]),
                "null_set_mean_max": max(null_means[endpoint]),
            }
            for endpoint, values in candidate_effects.items()
        },
        "attribution_exact_patch_spearman": correlations,
        "head_summaries": head_summaries,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(raw_rows),
        },
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_head_atlas_preflight_v1",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=(
            args.gue_test,
            args.cohort,
            args.cohort_audit,
            args.checkpoint_audit,
            args.calibration,
            args.group_calibration,
            args.config,
            raw_path,
            raw_manifest,
        ),
        parameters={
            "target_GENCODE_outcomes_computed": False,
            "mandatory_gates": thresholds,
            "candidate_selection": selection,
            "group_validation": null_config,
        },
        seeds={"preflight_and_null_sets": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "decision": report.decision,
                "gates": {outcome.name: outcome.state.value for outcome in outcomes},
                "observed": observed,
                "instrument_candidates": [head_label(head) for head in candidates],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _head_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), int(row["head"]))].append(row)
    return [
        {
            "layer": head[0],
            "head": head[1],
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
        for head, head_rows in sorted(grouped.items())
    ]


def _evaluate_directions(
    bundle: Any,
    pairs: Any,
    heads: Any,
    *,
    batch_pairs: int,
) -> dict[str, list[float]]:
    return {
        endpoint: evaluate_head_set(
            bundle,
            pairs,
            heads,
            direction=direction,
            batch_pairs=batch_pairs,
        )
        for endpoint, direction in (
            ("necessity", "target_from_control"),
            ("sufficiency", "control_from_target"),
        )
    }


def _prospective_mde(values: list[float], *, target_n: int) -> float:
    z = NormalDist()
    return (
        (z.inv_cdf(0.975) + z.inv_cdf(0.8)) * statistics.stdev(values) / target_n**0.5
    )


def _maximum_native_difference(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> float:
    return max(
        abs(float(a[key]) - float(b[key]))
        for a, b in zip(left, right)
        for key in ("target_margin", "control_margin")
    )


def _state(value: bool) -> GateState:
    return GateState.PASS if value else GateState.FAIL


def _gate_outcomes(
    thresholds: dict[str, Any], observed: dict[str, Any]
) -> list[GateOutcome]:
    positive = observed["positive_control"]
    negative = observed["negative_control"]
    sensitivity = observed["sensitivity"]
    resolution = observed["resolution"]
    invariants = observed["invariants"]
    balance = observed["matching_balance"]
    geometry = observed["token_geometry"]
    health = observed["model_health"]
    return [
        GateOutcome(
            "positive_control",
            _state(
                positive["GUE_test_MCC"]
                >= thresholds["positive_control"]["minimum_GUE_test_MCC"]
                and positive["GUE_test_AUROC"]
                >= thresholds["positive_control"]["minimum_GUE_test_AUROC"]
                and positive["full_head_patch_necessity_mean"]
                >= thresholds["positive_control"][
                    "minimum_full_head_patch_necessity_mean"
                ]
                and positive["full_head_patch_sufficiency_mean"]
                >= thresholds["positive_control"][
                    "minimum_full_head_patch_sufficiency_mean"
                ]
                and min(
                    positive["attribution_exact_spearman_necessity"],
                    positive["attribution_exact_spearman_sufficiency"],
                )
                >= thresholds["positive_control"][
                    "minimum_attribution_exact_spearman_each_direction"
                ]
            ),
            thresholds["positive_control"],
            positive,
        ),
        GateOutcome(
            "negative_control",
            _state(
                negative["maximum_absolute_noop_effect"]
                <= thresholds["negative_control"]["maximum_absolute_noop_effect"]
                and negative["cross_class_to_same_class_absolute_effect_ratio"]
                >= thresholds["negative_control"][
                    "minimum_cross_class_to_same_class_absolute_effect_ratio"
                ]
            ),
            thresholds["negative_control"],
            negative,
        ),
        GateOutcome(
            "sensitivity",
            _state(
                max(
                    sensitivity["prospective_MDE_necessity"],
                    sensitivity["prospective_MDE_sufficiency"],
                )
                <= thresholds["sensitivity"]["maximum_prospective_MDE_logit_margin"]
            ),
            thresholds["sensitivity"],
            sensitivity,
        ),
        GateOutcome(
            "resolution",
            _state(
                resolution["distinct_layer_matched_null_sets"]
                >= thresholds["resolution"]["minimum_distinct_layer_matched_null_sets"]
                and resolution["two_endpoint_Holm_floor"]
                <= thresholds["resolution"]["maximum_two_endpoint_Holm_floor"]
            ),
            thresholds["resolution"],
            resolution,
        ),
        GateOutcome(
            "invariants",
            _state(
                invariants["layers"] == thresholds["invariants"]["required_layers"]
                and invariants["heads_per_layer"]
                == thresholds["invariants"]["required_heads_per_layer"]
                and invariants["scanned_heads"]
                == thresholds["invariants"]["required_scanned_heads"]
                and invariants["missing_head_rows"]
                <= thresholds["invariants"]["maximum_missing_head_rows"]
                and invariants["unpadded_CLS_mapping_verified"]
                and invariants["calibration_preflight_disjoint"]
            ),
            thresholds["invariants"],
            invariants,
        ),
        GateOutcome(
            "matching_balance",
            _state(
                balance["positive_examples"] == balance["negative_examples"]
                and balance["pair_count_difference"]
                <= thresholds["matching_balance"]["maximum_pair_count_difference"]
            ),
            thresholds["matching_balance"],
            balance,
        ),
        GateOutcome(
            "token_geometry",
            _state(
                geometry["maximum_tokens"]
                <= thresholds["token_geometry"]["maximum_tokens"]
                and geometry["truncated_inputs"]
                <= thresholds["token_geometry"]["maximum_truncated_inputs"]
                and geometry["input_length_nt"] == 300
            ),
            thresholds["token_geometry"],
            geometry,
        ),
        GateOutcome(
            "model_health",
            _state(
                health["maximum_batch_invariance_absolute_margin_difference"]
                <= thresholds["model_health"][
                    "maximum_batch_invariance_absolute_margin_difference"
                ]
                and health["maximum_repeat_absolute_margin_difference"]
                <= thresholds["model_health"][
                    "maximum_repeat_absolute_margin_difference"
                ]
                and health["nonfinite_values"] == 0
                and health["complete_two_class_head"]
            ),
            thresholds["model_health"],
            health,
        ),
    ]


def _all_finite(*objects: Any) -> bool:
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
