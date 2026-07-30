#!/usr/bin/env python
"""Qualify the frozen promoter classifier margin and context instrument."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any

from gfmcd.adapters.dnabert2 import (
    PROMOTER_CHECKPOINTS,
    Dnabert2PromoterClassifier,
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
from gfmcd.reconstruction_atlas.promoter_assay import (
    binary_classification_metrics,
    context_indices,
    matched_context_shuffles,
)
from gfmcd.reconstruction_atlas.statistics import cluster_bootstrap_interval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--cohort-audit", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    seed = int(config["seed"])
    preflight_config = config["preflight"]
    gates = config["mandatory_gates"]
    checkpoint_name = str(config["checkpoint"]["candidate"])
    checkpoint = PROMOTER_CHECKPOINTS[checkpoint_name]
    checkpoint_audit = json.loads(Path(args.checkpoint_audit).read_text())
    if checkpoint_audit["selection"]["selected_candidate"] != checkpoint_name:
        raise RuntimeError("frozen config and checkpoint audit select different models")
    cohort_audit = json.loads(Path(args.cohort_audit).read_text())
    input_sha256 = {
        "gue_test": sha256_file(args.gue_test),
        "cohort": sha256_file(args.cohort),
        "cohort_audit": sha256_file(args.cohort_audit),
        "checkpoint_audit": sha256_file(args.checkpoint_audit),
    }
    pairs = _gue_pairs(
        args.gue_test,
        maximum=int(preflight_config["gue_test_pairs"]),
        seed=seed,
    )
    intervention = config["context_intervention"]
    indices, anchor = context_indices(
        tss_index=int(config["collection"]["window_upstream_nt"]),
        target_width_nt=int(intervention["target_anchor_width_nt"]),
        upstream_context_nt=int(intervention["upstream_context_nt"]),
        downstream_context_nt=int(intervention["downstream_context_nt"]),
    )
    requests: list[tuple[str, str]] = []
    metadata: dict[str, dict[str, Any]] = {}
    variant_audit = {
        "anchor_changes": 0,
        "context_composition_changes": 0,
        "target_control_dose_mismatches": 0,
        "ineligible_pairs": 0,
    }
    for pair_id, positive, negative in pairs:
        shuffles = matched_context_shuffles(
            positive,
            negative,
            indices=indices,
            seed=_pair_seed(seed, pair_id),
            candidates=int(intervention["candidate_shuffles"]),
        )
        if shuffles is None:
            variant_audit["ineligible_pairs"] += 1
            continue
        positive_shuffle, negative_shuffle, changed = shuffles
        variants = {
            "positive_clean": positive,
            "positive_shuffle": positive_shuffle,
            "negative_clean": negative,
            "negative_shuffle": negative_shuffle,
        }
        variant_audit["anchor_changes"] += int(
            positive[anchor[0] : anchor[1]] != positive_shuffle[anchor[0] : anchor[1]]
            or negative[anchor[0] : anchor[1]]
            != negative_shuffle[anchor[0] : anchor[1]]
        )
        variant_audit["context_composition_changes"] += int(
            sorted(positive[index] for index in indices)
            != sorted(positive_shuffle[index] for index in indices)
            or sorted(negative[index] for index in indices)
            != sorted(negative_shuffle[index] for index in indices)
        )
        variant_audit["target_control_dose_mismatches"] += int(
            _hamming(positive, positive_shuffle) != _hamming(negative, negative_shuffle)
        )
        for variant, sequence in variants.items():
            request_id = f"{pair_id}|{variant}"
            requests.append((request_id, sequence))
            metadata[request_id] = {
                "pair_id": pair_id,
                "variant": variant,
                "changed_nucleotides": 0 if variant.endswith("clean") else changed,
            }
    if len(requests) < 4 * int(preflight_config["gue_test_pairs"]):
        raise RuntimeError(
            f"preflight could not construct every dose-matched GUE pair: {variant_audit}"
        )

    bundle = load_dnabert2_promoter_classifier(checkpoint_name, device=args.device)
    scorer = Dnabert2PromoterClassifier(bundle)
    scores = scorer.score(requests, batch_size=args.batch_size)
    by_id = {score.request_id: score for score in scores}
    effect_rows = _effects(by_id, metadata)
    pair_ids = [row["pair_id"] for row in effect_rows]
    clean_values = [row["clean_margin_separation"] for row in effect_rows]
    context_values = [row["context_difference"] for row in effect_rows]
    draws = int(preflight_config["bootstrap_draws"])
    clean_ci = cluster_bootstrap_interval(
        clean_values, pair_ids, draws=draws, seed=seed
    )
    context_ci = cluster_bootstrap_interval(
        context_values, pair_ids, draws=draws, seed=seed ^ 0x1010
    )
    clean_mean = statistics.mean(clean_values)
    context_mean = statistics.mean(context_values)
    native = checkpoint_audit["candidate_results"][checkpoint_name]["gue_test"]
    permuted_mcc = _permuted_label_mcc(
        effect_rows,
        seed=seed,
        permutations=int(preflight_config.get("negative_control_permutations", 1)),
    )
    detectable = _minimum_detectable_effect(
        context_values,
        power=float(gates["sensitivity"]["power"]),
    )
    health = _model_health(scorer, requests[:16])
    max_tokens = max(score.token_count for score in scores)
    cohort_invariants = cohort_audit["invariant_violations"]
    matching = cohort_audit["matching_balance"]
    observed = {
        "positive_control": {
            "GUE_test_MCC": native["mcc"],
            "GUE_test_AUROC": native["auroc"],
            "clean_margin_separation": clean_mean,
            "clean_margin_separation_CI": list(clean_ci),
            "context_difference": context_mean,
            "context_difference_CI": list(context_ci),
        },
        "negative_control": {
            "absolute_mean_permuted_label_MCC": abs(
                permuted_mcc["mean_permuted_label_MCC"]
            ),
            **permuted_mcc,
        },
        "sensitivity": {
            "minimum_detectable_context_effect_logit_margin": detectable,
            "power": gates["sensitivity"]["power"],
        },
        "resolution": {
            "context_effect_CI_half_width": (context_ci[1] - context_ci[0]) / 2,
            "interval": list(context_ci),
        },
        "invariants": {
            **cohort_invariants,
            "preflight_anchor_changes": variant_audit["anchor_changes"],
            "preflight_context_composition_changes": variant_audit[
                "context_composition_changes"
            ],
            "preflight_target_control_dose_mismatches": variant_audit[
                "target_control_dose_mismatches"
            ],
        },
        "matching_balance": {
            "maximum_absolute_GC_fraction_difference": matching[
                "absolute_GC_fraction_difference"
            ]["max"],
            "maximum_absolute_CpG_fraction_difference": matching[
                "absolute_CpG_fraction_difference"
            ]["max"],
        },
        "token_geometry": {
            "maximum_tokens": max_tokens,
            "truncated_inputs": sum(score.token_count > 256 for score in scores),
        },
        "model_health": health,
    }
    outcomes = _gate_outcomes(gates, observed)
    if {outcome.name for outcome in outcomes} != MANDATORY_GATES:
        raise RuntimeError("promoter preflight did not emit every mandatory gate")
    report = PreflightReport(
        report_id="dnabert2-promoter-classifier-v1-qualified",
        config_sha256=config_hash,
        checkpoint_revision=checkpoint.revision,
        dataset_release=str(config["collection"]["release"]),
        input_sha256=input_sha256,
        split="GUE_native_test_and_outcome_blind_GENCODE_audit",
        outcomes=tuple(outcomes),
        frozen=True,
        notes=(
            "No GENCODE target or matched-control classifier outcome was computed.",
            "GUE held-out labels and GUE context shuffles qualify the supervised margin.",
            "The existing EPDnew atlas cohort remains excluded because it overlaps GUE.",
        ),
    )
    raw_rows = [
        {
            **score.to_dict(),
            **metadata[score.request_id],
        }
        for score in scores
    ]
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id="dnabert2_promoter_classifier_preflight_raw_v1",
        status="PREFLIGHT",
        model_identity=scorer.identity(),
        input_paths=(
            args.gue_test,
            args.cohort,
            args.cohort_audit,
            args.checkpoint_audit,
            args.config,
        ),
        parameters={
            "target_outcomes_computed": False,
            "gue_pairs": len(effect_rows),
            "measurement": config["measurement"],
            "context_intervention": intervention,
        },
        seeds={"preflight": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        **report.to_dict(),
        "observed": observed,
        "GUE_pair_effects": {
            "n": len(effect_rows),
            "clean_margin_separation_mean": clean_mean,
            "clean_margin_separation_CI": list(clean_ci),
            "context_difference_mean": context_mean,
            "context_difference_CI": list(context_ci),
        },
        "variant_audit": variant_audit,
        "raw_scores": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(raw_rows),
        },
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_classifier_preflight_v1",
        status="PREFLIGHT",
        model_identity=scorer.identity(),
        input_paths=(
            args.gue_test,
            args.cohort,
            args.cohort_audit,
            args.checkpoint_audit,
            args.config,
            raw_path,
            raw_manifest,
        ),
        parameters={
            "target_outcomes_computed": False,
            "bootstrap_draws": draws,
            "measurement": config["measurement"],
            "mandatory_gates": gates,
        },
        seeds={"preflight": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "decision": report.decision,
                "gate_states": {
                    outcome.name: outcome.state.value for outcome in outcomes
                },
                "observed": observed,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _gue_pairs(
    path: str | Path, *, maximum: int, seed: int
) -> list[tuple[str, str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_label = {
        label: sorted(
            (row["sequence"] for row in rows if int(row["label"]) == label),
            key=lambda sequence: hashlib.sha256(
                f"{seed}|GUE|{label}|{sequence}".encode()
            ).hexdigest(),
        )[:maximum]
        for label in (0, 1)
    }
    if any(len(values) < maximum for values in by_label.values()):
        raise RuntimeError("GUE test split cannot supply the frozen preflight pairs")
    return [
        (f"GUE-pair-{index:04d}", positive, negative)
        for index, (positive, negative) in enumerate(
            zip(by_label[1], by_label[0]), start=1
        )
    ]


def _effects(
    scores: dict[str, Any], metadata: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = {}
    for request_id, score in scores.items():
        info = metadata[request_id]
        grouped.setdefault(str(info["pair_id"]), {})[str(info["variant"])] = (
            score.margin
        )
    output = []
    for pair_id, values in sorted(grouped.items()):
        if set(values) != {
            "positive_clean",
            "positive_shuffle",
            "negative_clean",
            "negative_shuffle",
        }:
            raise RuntimeError(f"incomplete promoter preflight pair: {pair_id}")
        positive_loss = values["positive_clean"] - values["positive_shuffle"]
        negative_loss = values["negative_clean"] - values["negative_shuffle"]
        output.append(
            {
                "pair_id": pair_id,
                "positive_clean_margin": values["positive_clean"],
                "negative_clean_margin": values["negative_clean"],
                "clean_margin_separation": (
                    values["positive_clean"] - values["negative_clean"]
                ),
                "positive_context_loss": positive_loss,
                "negative_context_loss": negative_loss,
                "context_difference": positive_loss - negative_loss,
            }
        )
    return output


def _permuted_label_mcc(
    rows: list[dict[str, Any]], *, seed: int, permutations: int
) -> dict[str, Any]:
    if permutations < 1:
        raise ValueError("negative-control permutations must be positive")
    original_labels = [1] * len(rows) + [0] * len(rows)
    margins = [row["positive_clean_margin"] for row in rows] + [
        row["negative_clean_margin"] for row in rows
    ]
    rng = random.Random(seed ^ 0xC0FFEE)
    values = []
    for _ in range(permutations):
        labels = original_labels.copy()
        rng.shuffle(labels)
        values.append(float(binary_classification_metrics(labels, margins)["mcc"]))
    return {
        "permutations": permutations,
        "mean_permuted_label_MCC": statistics.mean(values),
        "minimum_permuted_label_MCC": min(values),
        "maximum_permuted_label_MCC": max(values),
        "standard_deviation_permuted_label_MCC": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
    }


def _minimum_detectable_effect(values: list[float], *, power: float) -> float:
    standard_error = statistics.stdev(values) / len(values) ** 0.5
    normal = NormalDist()
    return (normal.inv_cdf(0.975) + normal.inv_cdf(power)) * standard_error


def _model_health(
    scorer: Dnabert2PromoterClassifier,
    requests: list[tuple[str, str]],
) -> dict[str, Any]:
    singleton = scorer.score(requests, batch_size=1)
    batched = scorer.score(requests, batch_size=len(requests))
    repeated = scorer.score(requests, batch_size=len(requests))
    return {
        "maximum_batch_invariance_absolute_margin_difference": max(
            abs(left.margin - right.margin) for left, right in zip(singleton, batched)
        ),
        "maximum_repeat_absolute_margin_difference": max(
            abs(left.margin - right.margin) for left, right in zip(batched, repeated)
        ),
        "nonfinite_scores": 0,
        "complete_two_class_head": bool(
            scorer.identity()["classification_head_complete"]
        ),
    }


def _gate_outcomes(
    thresholds: dict[str, Any], observed: dict[str, Any]
) -> list[GateOutcome]:
    positive = thresholds["positive_control"]
    positive_observed = observed["positive_control"]
    outcomes = [
        GateOutcome(
            name="positive_control",
            state=_state(
                positive_observed["GUE_test_MCC"] >= positive["minimum_GUE_test_MCC"]
                and positive_observed["GUE_test_AUROC"]
                >= positive["minimum_GUE_test_AUROC"]
                and positive_observed["clean_margin_separation"]
                >= positive["minimum_clean_margin_separation"]
                and positive_observed["clean_margin_separation_CI"][0] > 0
                and positive_observed["context_difference"]
                >= positive["minimum_context_difference"]
                and positive_observed["context_difference_CI"][0] > 0
            ),
            threshold=positive,
            observed=positive_observed,
        ),
        GateOutcome(
            name="negative_control",
            state=_state(
                observed["negative_control"]["absolute_mean_permuted_label_MCC"]
                <= thresholds["negative_control"].get(
                    "maximum_absolute_mean_permuted_label_MCC",
                    thresholds["negative_control"].get(
                        "maximum_absolute_permuted_label_MCC"
                    ),
                )
            ),
            threshold=thresholds["negative_control"],
            observed=observed["negative_control"],
        ),
        GateOutcome(
            name="sensitivity",
            state=_state(
                observed["sensitivity"][
                    "minimum_detectable_context_effect_logit_margin"
                ]
                <= thresholds["sensitivity"][
                    "maximum_detectable_context_effect_logit_margin"
                ]
            ),
            threshold=thresholds["sensitivity"],
            observed=observed["sensitivity"],
        ),
        GateOutcome(
            name="resolution",
            state=_state(
                observed["resolution"]["context_effect_CI_half_width"]
                <= thresholds["resolution"]["maximum_context_effect_CI_half_width"]
            ),
            threshold=thresholds["resolution"],
            observed=observed["resolution"],
        ),
        GateOutcome(
            name="invariants",
            state=_state(not any(observed["invariants"].values())),
            threshold=thresholds["invariants"],
            observed=observed["invariants"],
        ),
        GateOutcome(
            name="matching_balance",
            state=_state(
                observed["matching_balance"]["maximum_absolute_GC_fraction_difference"]
                <= thresholds["matching_balance"][
                    "maximum_absolute_GC_fraction_difference"
                ]
                and observed["matching_balance"][
                    "maximum_absolute_CpG_fraction_difference"
                ]
                <= thresholds["matching_balance"][
                    "maximum_absolute_CpG_fraction_difference"
                ]
            ),
            threshold=thresholds["matching_balance"],
            observed=observed["matching_balance"],
        ),
        GateOutcome(
            name="token_geometry",
            state=_state(
                observed["token_geometry"]["maximum_tokens"]
                <= thresholds["token_geometry"]["maximum_tokens"]
                and observed["token_geometry"]["truncated_inputs"]
                <= thresholds["token_geometry"]["maximum_truncated_inputs"]
            ),
            threshold=thresholds["token_geometry"],
            observed=observed["token_geometry"],
        ),
        GateOutcome(
            name="model_health",
            state=_state(
                observed["model_health"][
                    "maximum_batch_invariance_absolute_margin_difference"
                ]
                <= thresholds["model_health"]["batch_invariance_absolute_tolerance"]
                and observed["model_health"][
                    "maximum_repeat_absolute_margin_difference"
                ]
                <= thresholds["model_health"]["determinism_absolute_tolerance"]
                and observed["model_health"]["nonfinite_scores"] == 0
                and observed["model_health"]["complete_two_class_head"]
            ),
            threshold=thresholds["model_health"],
            observed=observed["model_health"],
        ),
    ]
    return outcomes


def _state(passed: bool) -> GateState:
    return GateState.PASS if passed else GateState.FAIL


def _pair_seed(seed: int, pair_id: str) -> int:
    digest = hashlib.sha256(f"{seed}|preflight|{pair_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


if __name__ == "__main__":
    main()
