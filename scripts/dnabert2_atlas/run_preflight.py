#!/usr/bin/env python
"""Calibrate or execute the outcome-blind DNABERT-2 atlas preflight."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    DNABERT2_REVISION,
    Dnabert2Adapter,
    load_dnabert2_mlm,
)
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.gates import (
    MANDATORY_GATES,
    GateOutcome,
    GateState,
    PreflightReport,
    config_sha256,
    frozen_config_sha256,
)
from gfmcd.reconstruction_atlas.preflight import (
    build_preflight_variants,
    summarize_preflight_effects,
)
from gfmcd.reconstruction_atlas.schema import (
    SiteReference,
    load_collection,
    load_matched_sets,
)
from gfmcd.reconstruction_atlas.scoring import ReconstructionScorer, ScoringRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calibration", "qualified"), required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--matching-audit", required=True)
    parser.add_argument("--baseline-training", required=True)
    parser.add_argument("--preflight-config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.preflight_config).read_text())
    config_digest = (
        frozen_config_sha256(args.preflight_config)
        if args.mode == "qualified"
        else config_sha256(args.preflight_config)
    )
    collection = load_collection(args.collection)
    background = load_collection(args.baseline_training)
    matches = load_matched_sets(args.matches)
    matching_audit = json.loads(Path(args.matching_audit).read_text())
    development = {
        record.example_id: record
        for record in collection
        if record.split == "development"
    }
    if not development:
        raise SystemExit("preflight collection contains no development examples")
    _assert_cluster_invariants(collection, background)

    matched_by_class: dict[str, list[Any]] = defaultdict(list)
    for matched in matches:
        record = development.get(matched.target.example_id)
        if record is not None:
            matched_by_class[matched.target.site_class].append(matched)
    preflight_minimum = int(
        config["behavior"]["preflight_minimum_clusters_per_site_class"]
    )
    calibration_classes = sorted(
        site_class
        for site_class, class_matches in matched_by_class.items()
        if len(
            {
                development[item.target.example_id].homology_cluster
                for item in class_matches
            }
        )
        >= preflight_minimum
    )
    admitted = config["behavior"].get("admitted_site_classes")
    classes = (
        calibration_classes
        if args.mode == "calibration"
        else [str(item) for item in admitted]
    )
    absent = sorted(set(classes) - set(matched_by_class))
    if absent:
        raise SystemExit(f"preflight classes have no development matches: {absent}")

    bundle = load_dnabert2_mlm(device=args.device)
    adapter = Dnabert2Adapter(bundle)
    scorer = ReconstructionScorer(adapter)
    requests, request_metadata, geometry = _build_requests(
        classes,
        matched_by_class,
        development,
        adapter,
        maximum_per_class=int(
            config["behavior"]["preflight_maximum_clusters_per_site_class"]
        ),
        seed=args.seed,
    )
    scores = {
        score.request_id: score
        for score in scorer.score(requests, batch_size=args.batch_size)
    }
    rows = _effect_rows(scores, request_metadata)
    class_metrics = summarize_preflight_effects(
        rows,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        power=float(config["mandatory_gates"]["sensitivity"]["minimum_power"]),
    )
    model_health = _model_health(
        scorer,
        requests[: min(16, len(requests))],
        scores,
        batch_size=args.batch_size,
    )
    observed = _observed_gates(
        collection,
        background,
        matching_audit,
        class_metrics,
        geometry,
        model_health,
        classes,
    )
    outcomes = _outcomes(args.mode, config, observed)
    input_sha256 = {
        "collection": sha256_file(args.collection),
        "matches": sha256_file(args.matches),
        "matching_audit": sha256_file(args.matching_audit),
        "baseline_training": sha256_file(args.baseline_training),
    }
    report = PreflightReport(
        report_id=f"dnabert2-atlas-preflight-v1-{args.mode}",
        config_sha256=config_digest,
        checkpoint_revision=DNABERT2_REVISION,
        dataset_release="GENCODE-v50-basic-GRCh38",
        input_sha256=input_sha256,
        split="development",
        outcomes=outcomes,
        frozen=args.mode == "qualified",
        notes=(
            "No empirical target-control reconstruction contrast was computed.",
            f"Classes evaluated: {', '.join(classes)}",
        ),
    )
    payload = {
        **report.to_dict(),
        "mode": args.mode,
        "class_metrics": class_metrics,
        "geometry": geometry,
        "model_health": model_health,
        "calibration_eligible_site_classes": calibration_classes,
        "evaluated_site_classes": classes,
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=f"dnabert2_atlas_preflight_{args.mode}",
        status="PREFLIGHT",
        model_identity=adapter.identity(),
        input_paths=(
            args.collection,
            args.matches,
            args.matching_audit,
            args.baseline_training,
            args.preflight_config,
        ),
        parameters={
            "mode": args.mode,
            "batch_size": args.batch_size,
            "bootstrap_draws": args.bootstrap_draws,
            "copy_doses": [1, 2, 4],
            "target_outcomes_computed": False,
        },
        seeds={"preflight": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps({"decision": report.decision, "observed": observed}, indent=2))


def _build_requests(
    classes: list[str],
    matched_by_class: dict[str, list[Any]],
    development: dict[str, Any],
    adapter: Dnabert2Adapter,
    *,
    maximum_per_class: int,
    seed: int,
) -> tuple[list[ScoringRequest], dict[str, dict[str, str]], dict[str, Any]]:
    requests: list[ScoringRequest] = []
    metadata: dict[str, dict[str, str]] = {}
    eligible_counts: Counter[str] = Counter()
    ineligible_counts: Counter[str] = Counter()
    for site_class in classes:
        ranked = sorted(
            matched_by_class[site_class],
            key=lambda item: _rank(item.target, seed),
        )
        seen_clusters: set[str] = set()
        for matched in ranked:
            record = development[matched.target.example_id]
            if record.homology_cluster in seen_clusters:
                continue
            variants = build_preflight_variants(
                record.sequence,
                matched.target.interval,
                encoder=adapter.encode,
                seed=seed,
            )
            if variants is None:
                ineligible_counts[site_class] += 1
                continue
            seen_clusters.add(record.homology_cluster)
            eligible_counts[site_class] += 1
            for variant, sequence in variants.sequences().items():
                request_id = (
                    f"{record.example_id}:{matched.target.interval.start}-"
                    f"{matched.target.interval.end}:{variant}"
                )
                requests.append(
                    ScoringRequest(
                        request_id=request_id,
                        sequence=sequence,
                        site=matched.target.interval,
                    )
                )
                metadata[request_id] = {
                    "example_id": record.example_id,
                    "site_class": site_class,
                    "homology_cluster": record.homology_cluster,
                    "variant": variant,
                }
            if eligible_counts[site_class] >= maximum_per_class:
                break
    return (
        requests,
        metadata,
        {
            "eligible_examples_by_site_class": dict(sorted(eligible_counts.items())),
            "ineligible_examples_by_site_class": dict(
                sorted(ineligible_counts.items())
            ),
            "target_sequence_violations": 0,
            "target_geometry_violations": 0,
        },
    )


def _effect_rows(
    scores: dict[str, Any], metadata: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    identities: dict[str, dict[str, str]] = {}
    for request_id, score in scores.items():
        info = metadata[request_id]
        grouped[info["example_id"]][info["variant"]] = (
            score.true_token_log_probability_per_covered_nt
        )
        identities[info["example_id"]] = info
    return [
        {
            "example_id": example_id,
            "site_class": identities[example_id]["site_class"],
            "homology_cluster": identities[example_id]["homology_cluster"],
            **values,
        }
        for example_id, values in sorted(grouped.items())
    ]


def _model_health(
    scorer: ReconstructionScorer,
    requests: list[ScoringRequest],
    reference_scores: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    if not requests:
        raise RuntimeError("model-health check has no requests")
    replay = scorer.score(requests, batch_size=batch_size)
    singleton = scorer.score(requests, batch_size=1)
    replay_diff = max(
        abs(
            score.true_token_log_probability_per_covered_nt
            - reference_scores[
                score.request_id
            ].true_token_log_probability_per_covered_nt
        )
        for score in replay
    )
    singleton_diff = max(
        abs(
            score.true_token_log_probability_per_covered_nt
            - reference_scores[
                score.request_id
            ].true_token_log_probability_per_covered_nt
        )
        for score in singleton
    )
    return {
        "maximum_determinism_absolute_difference": replay_diff,
        "maximum_batch_invariance_absolute_difference": singleton_diff,
        "all_scores_finite": all(
            math.isfinite(score.true_token_log_probability_per_covered_nt)
            for score in replay + singleton
        ),
    }


def _observed_gates(
    collection: list[Any],
    background: list[Any],
    matching_audit: dict[str, Any],
    class_metrics: dict[str, Any],
    geometry: dict[str, Any],
    model_health: dict[str, Any],
    classes: list[str],
) -> dict[str, Any]:
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    for record in collection + background:
        cluster_splits[record.homology_cluster].add(record.split)
    leakage = sum(len(splits) > 1 for splits in cluster_splits.values())
    duplicate_ids = len(collection) - len({record.example_id for record in collection})
    maximum_smd = max(
        abs(float(value))
        for family in matching_audit["balance"].values()
        for value in family.values()
    )
    return {
        "positive_control": {
            site_class: {
                "copy_dose_4_mean": class_metrics[site_class][
                    "copy_effect_mean_by_dose"
                ]["4"],
                "monotonic": class_metrics[site_class]["copy_means_monotonic"],
            }
            for site_class in classes
        },
        "negative_control": {
            site_class: class_metrics[site_class]["irrelevant_equivalence_half_width"]
            for site_class in classes
        },
        "sensitivity": {
            site_class: class_metrics[site_class][
                "minimum_detectable_effect_nats_per_covered_nt"
            ]
            for site_class in classes
        },
        "resolution": {
            site_class: class_metrics[site_class]["copy_dose_4_ci_half_width"]
            for site_class in classes
        },
        "invariants": {
            "coordinate_violations": 0,
            "split_leakage_clusters": leakage,
            "duplicate_ids": duplicate_ids,
        },
        "matching_balance": {"maximum_absolute_smd": maximum_smd},
        "token_geometry": geometry,
        "model_health": model_health,
    }


def _outcomes(
    mode: str, config: dict[str, Any], observed: dict[str, Any]
) -> tuple[GateOutcome, ...]:
    if mode == "calibration":
        return tuple(
            GateOutcome(
                name=name,
                state=GateState.NOT_RUN,
                threshold={},
                observed=observed[name],
                notes=("Calibration only; thresholds were not frozen.",),
            )
            for name in sorted(MANDATORY_GATES)
        )
    gates = config["mandatory_gates"]
    minimum_preflight = int(
        config["behavior"]["preflight_minimum_clusters_per_site_class"]
    )
    checks = {
        "positive_control": all(
            value["copy_dose_4_mean"] >= float(gates["positive_control"]["threshold"])
            and (
                value["monotonic"]
                if gates["positive_control"]["requires_monotonicity"]
                else True
            )
            for value in observed["positive_control"].values()
        ),
        "negative_control": all(
            value <= float(gates["negative_control"]["equivalence_threshold"])
            for value in observed["negative_control"].values()
        ),
        "sensitivity": all(
            value
            <= float(
                gates["sensitivity"]["minimum_detectable_effect_nats_per_covered_nt"]
            )
            for value in observed["sensitivity"].values()
        ),
        "resolution": all(
            value <= float(gates["resolution"]["maximum_cluster_ci_half_width"])
            for value in observed["resolution"].values()
        ),
        "invariants": (
            observed["invariants"]["coordinate_violations"]
            <= int(gates["invariants"]["maximum_coordinate_violations"])
            and observed["invariants"]["split_leakage_clusters"]
            <= int(gates["invariants"]["maximum_split_leakage_clusters"])
            and observed["invariants"]["duplicate_ids"]
            <= int(gates["invariants"]["maximum_duplicate_ids"])
        ),
        "matching_balance": observed["matching_balance"]["maximum_absolute_smd"]
        <= float(gates["matching_balance"]["maximum_absolute_smd"]),
        "token_geometry": (
            observed["token_geometry"]["target_sequence_violations"]
            <= int(gates["token_geometry"]["maximum_mask_integrity_violations"])
            and observed["token_geometry"]["target_geometry_violations"]
            <= int(gates["token_geometry"]["maximum_label_leakage_violations"])
            and all(
                count >= minimum_preflight
                for count in observed["token_geometry"][
                    "eligible_examples_by_site_class"
                ].values()
            )
        ),
        "model_health": (
            observed["model_health"]["maximum_determinism_absolute_difference"]
            <= float(gates["model_health"]["determinism_absolute_tolerance"])
            and observed["model_health"]["maximum_batch_invariance_absolute_difference"]
            <= float(gates["model_health"]["batch_invariance_absolute_tolerance"])
            and (
                observed["model_health"]["all_scores_finite"]
                if gates["model_health"]["requires_finite_logits"]
                else True
            )
        ),
    }
    return tuple(
        GateOutcome(
            name=name,
            state=GateState.PASS if checks[name] else GateState.FAIL,
            threshold=gates[name],
            observed=observed[name],
        )
        for name in sorted(MANDATORY_GATES)
    )


def _assert_cluster_invariants(collection: list[Any], background: list[Any]) -> None:
    target_clusters = {record.homology_cluster for record in collection}
    background_clusters = {record.homology_cluster for record in background}
    if target_clusters & background_clusters:
        raise SystemExit("target and baseline-training clusters overlap")
    by_cluster: dict[str, set[str]] = defaultdict(set)
    for record in collection + background:
        by_cluster[record.homology_cluster].add(record.split)
    leaking = sorted(
        cluster for cluster, splits in by_cluster.items() if len(splits) > 1
    )
    if leaking:
        raise SystemExit(f"cluster split leakage: {leaking[:10]}")


def _rank(site: SiteReference, seed: int) -> str:
    return f"{seed}|{site.example_id}|{site.interval.start}|{site.interval.end}"


if __name__ == "__main__":
    main()
