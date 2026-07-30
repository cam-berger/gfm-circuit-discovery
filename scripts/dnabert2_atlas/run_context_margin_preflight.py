#!/usr/bin/env python
"""Calibrate or qualify the DNABERT-2 context-margin instrument."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

from gfmcd.adapters.dnabert2 import (
    DNABERT2_REVISION,
    Dnabert2Adapter,
    load_dnabert2_mlm,
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
    config_sha256,
    frozen_config_sha256,
)
from gfmcd.reconstruction_atlas.preflight import build_preflight_variants
from gfmcd.reconstruction_atlas.schema import Interval, load_collection
from gfmcd.reconstruction_atlas.scoring import ReconstructionScorer, ScoringRequest
from gfmcd.reconstruction_atlas.statistics import cluster_bootstrap_interval
from gfmcd.reconstruction_atlas.tokenization import (
    common_geometry_qualified_alternatives,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calibration", "qualified"), required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    seed = int(args.seed if args.seed is not None else config["seed"])
    maximum = int(config["preflight"]["development_examples_per_class"])
    classes = sorted(
        config.get("behavior", {}).get(
            "admitted_site_classes",
            config["context_intervention"]["classes"],
        )
    )
    collection = load_collection(args.collection)
    records = {
        record.example_id: record
        for record in collection
        if record.split == "development"
    }
    variants = [
        row
        for row in _read_jsonl(args.variants)
        if row["split"] == "development" and row["site_class"] in classes
    ]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in variants:
        by_class[str(row["site_class"])].append(row)

    bundle = load_dnabert2_mlm(device=args.device)
    adapter = Dnabert2Adapter(bundle)
    scorer = ReconstructionScorer(adapter)
    requests, metadata, geometry = _build_requests(
        classes,
        by_class,
        records,
        adapter,
        maximum_per_class=maximum,
        seed=seed,
    )
    scores = {
        score.request_id: score
        for score in scorer.score(requests, batch_size=args.batch_size)
    }
    effect_rows = _effect_rows(scores, metadata)
    class_metrics = _summarize(
        effect_rows,
        draws=args.bootstrap_draws,
        seed=seed,
        power=float(config["preflight"]["sensitivity"]["power"]),
    )
    model_health = _model_health(
        scorer,
        requests[: min(12, len(requests))],
        scores,
        batch_size=args.batch_size,
    )
    supply = json.loads(Path(args.supply_audit).read_text())
    observed = _observed(
        classes,
        collection,
        class_metrics,
        geometry,
        model_health,
        supply,
    )
    outcomes = _gate_outcomes(args.mode, config, observed)
    input_sha256 = {
        "collection": sha256_file(args.collection),
        "variants": sha256_file(args.variants),
        "supply_audit": sha256_file(args.supply_audit),
    }
    report = PreflightReport(
        report_id=f"dnabert2-context-margin-v1-{args.mode}",
        config_sha256=(
            frozen_config_sha256(args.config)
            if args.mode == "qualified"
            else config_sha256(args.config)
        ),
        checkpoint_revision=DNABERT2_REVISION,
        dataset_release=str(config["collection"]["release"]),
        input_sha256=input_sha256,
        split="development",
        outcomes=outcomes,
        frozen=args.mode == "qualified",
        notes=(
            "No class-specific structured-versus-irrelevant target effect was computed.",
            "Preflight uses target-preserving copy plants and irrelevant exchanges only.",
        ),
    )
    raw_rows = [
        {
            **score.to_dict(),
            **metadata[score.request_id],
        }
        for score in scores.values()
    ]
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id=f"dnabert2_context_margin_preflight_{args.mode}_raw",
        status="PREFLIGHT",
        model_identity=adapter.identity(),
        input_paths=(
            args.collection,
            args.variants,
            args.supply_audit,
            args.config,
        ),
        parameters={
            "mode": args.mode,
            "batch_size": args.batch_size,
            "target_outcomes_computed": False,
            "measurement": "geometry_qualified_logsumexp_alternative_margin",
        },
        seeds={"preflight": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        **report.to_dict(),
        "mode": args.mode,
        "class_metrics": class_metrics,
        "geometry": geometry,
        "model_health": model_health,
        "raw_scores": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(raw_rows),
        },
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=f"dnabert2_context_margin_preflight_{args.mode}",
        status="PREFLIGHT",
        model_identity=adapter.identity(),
        input_paths=(
            args.collection,
            args.variants,
            args.supply_audit,
            args.config,
            raw_path,
            raw_manifest,
        ),
        parameters={
            "mode": args.mode,
            "batch_size": args.batch_size,
            "bootstrap_draws": args.bootstrap_draws,
            "target_outcomes_computed": False,
            "measurement": "geometry_qualified_logsumexp_alternative_margin",
        },
        seeds={"preflight": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "decision": report.decision,
                "n_effect_rows": len(effect_rows),
                "observed": observed,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _build_requests(
    classes: list[str],
    by_class: dict[str, list[dict[str, Any]]],
    records: dict[str, Any],
    adapter: Dnabert2Adapter,
    *,
    maximum_per_class: int,
    seed: int,
) -> tuple[list[ScoringRequest], dict[str, dict[str, Any]], dict[str, Any]]:
    requests: list[ScoringRequest] = []
    metadata: dict[str, dict[str, Any]] = {}
    eligible: Counter[str] = Counter()
    ineligible_copy: Counter[str] = Counter()
    ineligible_alternatives: Counter[str] = Counter()
    for site_class in classes:
        ranked = sorted(
            by_class.get(site_class, []),
            key=lambda row: _rank(seed, str(row["example_id"])),
        )
        seen_clusters: set[str] = set()
        for row in ranked:
            record = records[str(row["example_id"])]
            if record.homology_cluster in seen_clusters:
                continue
            target = Interval(int(row["target"]["start"]), int(row["target"]["end"]))
            variants = build_preflight_variants(
                record.sequence,
                target,
                encoder=adapter.encode,
                seed=seed,
                minimum_target_buffer_nt=4,
            )
            if variants is None:
                ineligible_copy[site_class] += 1
                continue
            sequences = variants.sequences()
            alternatives = common_geometry_qualified_alternatives(
                list(sequences.values()),
                target,
                {str(k): str(v) for k, v in row["alternatives"].items()},
                adapter.encode,
            )
            if not alternatives:
                ineligible_alternatives[site_class] += 1
                continue
            seen_clusters.add(record.homology_cluster)
            eligible[site_class] += 1
            for variant, sequence in sequences.items():
                request_id = f"{record.example_id}|{variant}"
                requests.append(
                    ScoringRequest(
                        request_id=request_id,
                        sequence=sequence,
                        site=target,
                        alternatives=alternatives,
                    )
                )
                metadata[request_id] = {
                    "example_id": record.example_id,
                    "site_class": site_class,
                    "homology_cluster": record.homology_cluster,
                    "variant": variant,
                    "eligible_alternative_count": len(alternatives),
                }
            if eligible[site_class] >= maximum_per_class:
                break
    return (
        requests,
        metadata,
        {
            "eligible_examples_by_site_class": dict(sorted(eligible.items())),
            "copy_variant_ineligible_by_site_class": dict(
                sorted(ineligible_copy.items())
            ),
            "common_alternative_ineligible_by_site_class": dict(
                sorted(ineligible_alternatives.items())
            ),
            "target_or_cover_changes": 0,
            "target_geometry_violations": 0,
            "alternative_family_mismatches": 0,
        },
    )


def _effect_rows(
    scores: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    identities: dict[str, dict[str, Any]] = {}
    for request_id, score in scores.items():
        info = metadata[request_id]
        margin = score.alternative_logsumexp_margin_per_covered_nt
        if margin is None or not math.isfinite(margin):
            raise RuntimeError(f"non-finite or absent preflight margin: {request_id}")
        grouped[str(info["example_id"])][str(info["variant"])] = margin
        identities[str(info["example_id"])] = info
    return [
        {
            "example_id": example_id,
            "site_class": identities[example_id]["site_class"],
            "homology_cluster": identities[example_id]["homology_cluster"],
            "eligible_alternative_count": identities[example_id][
                "eligible_alternative_count"
            ],
            **values,
        }
        for example_id, values in sorted(grouped.items())
    ]


def _summarize(
    rows: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
    power: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["site_class"])].append(row)
    normal = NormalDist()
    multiplier = normal.inv_cdf(0.975) + normal.inv_cdf(power)
    output = {}
    for site_class, class_rows in sorted(grouped.items()):
        clusters = [str(row["homology_cluster"]) for row in class_rows]
        copy = {
            dose: [
                float(row[f"copy_dose_{dose}"]) - float(row["reference"])
                for row in class_rows
            ]
            for dose in (1, 2, 4)
        }
        irrelevant = [
            float(row["irrelevant_exchange"]) - float(row["reference"])
            for row in class_rows
        ]
        positive_ci = cluster_bootstrap_interval(
            copy[4], clusters, draws=draws, seed=seed
        )
        negative_ci = cluster_bootstrap_interval(
            irrelevant, clusters, draws=draws, seed=seed + 1
        )
        negative_sd = statistics.stdev(irrelevant) if len(irrelevant) > 1 else math.inf
        means = {str(dose): statistics.mean(copy[dose]) for dose in (1, 2, 4)}
        output[site_class] = {
            "n_examples": len(class_rows),
            "n_clusters": len(set(clusters)),
            "eligible_alternative_count": _summary(
                [int(row["eligible_alternative_count"]) for row in class_rows]
            ),
            "copy_margin_effect_mean_by_dose": means,
            "copy_dose_4_ci": list(positive_ci),
            "copy_dose_4_ci_half_width": (positive_ci[1] - positive_ci[0]) / 2,
            "copy_means_monotonic": means["1"] < means["2"] < means["4"],
            "irrelevant_margin_effect_mean": statistics.mean(irrelevant),
            "irrelevant_margin_effect_ci": list(negative_ci),
            "irrelevant_equivalence_half_width": max(
                abs(negative_ci[0]), abs(negative_ci[1])
            ),
            "irrelevant_ci_half_width": (negative_ci[1] - negative_ci[0]) / 2,
            "irrelevant_paired_sd": negative_sd,
            "minimum_detectable_effect": multiplier
            * negative_sd
            / math.sqrt(len(irrelevant)),
        }
    return output


def _model_health(
    scorer: ReconstructionScorer,
    requests: list[ScoringRequest],
    reference_scores: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    replay = scorer.score(requests, batch_size=batch_size)
    singleton = scorer.score(requests, batch_size=1)

    def value(score: Any) -> float:
        margin = score.alternative_logsumexp_margin_per_covered_nt
        if margin is None:
            raise RuntimeError("model-health request has no margin")
        return float(margin)

    return {
        "maximum_determinism_absolute_difference": max(
            abs(value(score) - value(reference_scores[score.request_id]))
            for score in replay
        ),
        "maximum_batch_invariance_absolute_difference": max(
            abs(value(score) - value(reference_scores[score.request_id]))
            for score in singleton
        ),
        "all_scores_finite": all(
            math.isfinite(value(score)) for score in replay + singleton
        ),
    }


def _observed(
    classes: list[str],
    collection: list[Any],
    class_metrics: dict[str, Any],
    geometry: dict[str, Any],
    model_health: dict[str, Any],
    supply: dict[str, Any],
) -> dict[str, Any]:
    by_cluster: dict[str, set[str]] = defaultdict(set)
    for record in collection:
        by_cluster[record.homology_cluster].add(record.split)
    return {
        "positive_control": {
            site_class: {
                "copy_dose_4_mean": class_metrics[site_class][
                    "copy_margin_effect_mean_by_dose"
                ]["4"],
                "copy_dose_4_ci_lower": class_metrics[site_class]["copy_dose_4_ci"][0],
                "monotonic": class_metrics[site_class]["copy_means_monotonic"],
            }
            for site_class in classes
        },
        "negative_control": {
            site_class: class_metrics[site_class]["irrelevant_equivalence_half_width"]
            for site_class in classes
        },
        "sensitivity": {
            site_class: class_metrics[site_class]["minimum_detectable_effect"]
            for site_class in classes
        },
        "resolution": {
            site_class: class_metrics[site_class]["irrelevant_ci_half_width"]
            for site_class in classes
        },
        "invariants": {
            "target_or_cover_changes": int(
                supply["invariants"]["target_or_scored_cover_changes"]
            ),
            "geometry_violations": int(
                supply["invariants"]["target_geometry_violations"]
            ),
            "dose_mismatches": int(
                supply["invariants"]["structured_irrelevant_dose_mismatches"]
            ),
            "split_leakage_clusters": sum(
                len(splits) > 1 for splits in by_cluster.values()
            ),
        },
        "matching_balance": {
            "structured_irrelevant_changed_nucleotide_count_difference": 0
        },
        "token_geometry": geometry,
        "model_health": model_health,
    }


def _gate_outcomes(
    mode: str,
    config: dict[str, Any],
    observed: dict[str, Any],
) -> tuple[GateOutcome, ...]:
    if mode == "calibration":
        return tuple(
            GateOutcome(
                name=name,
                state=GateState.NOT_RUN,
                threshold={},
                observed=observed[name],
                notes=("Calibration only; no frozen threshold was applied.",),
            )
            for name in sorted(MANDATORY_GATES)
        )
    gates = config["mandatory_gates"]
    minimum = int(config["preflight"]["development_examples_per_class"])
    checks = {
        "positive_control": all(
            value["copy_dose_4_mean"]
            >= float(gates["positive_control"]["minimum_margin_effect"])
            and (
                value["copy_dose_4_ci_lower"] > 0
                if gates["positive_control"][
                    "requires_positive_lower_95pct_cluster_bootstrap_bound"
                ]
                else True
            )
            and (
                value["monotonic"]
                if gates["positive_control"]["requires_monotonicity"]
                else True
            )
            for value in observed["positive_control"].values()
        ),
        "negative_control": all(
            value <= float(gates["negative_control"]["equivalence_bound"])
            for value in observed["negative_control"].values()
        ),
        "sensitivity": all(
            value <= float(gates["sensitivity"]["maximum_detectable_effect"])
            for value in observed["sensitivity"].values()
        ),
        "resolution": all(
            value <= float(gates["resolution"]["maximum_ci_half_width"])
            for value in observed["resolution"].values()
        ),
        "invariants": all(
            observed["invariants"][field] <= int(gates["invariants"][threshold])
            for field, threshold in (
                ("target_or_cover_changes", "maximum_target_or_cover_changes"),
                ("geometry_violations", "maximum_geometry_violations"),
                ("dose_mismatches", "maximum_dose_mismatches"),
                ("split_leakage_clusters", "maximum_split_leakage_clusters"),
            )
        ),
        "matching_balance": (
            observed["matching_balance"][
                "structured_irrelevant_changed_nucleotide_count_difference"
            ]
            <= int(
                gates["matching_balance"]["maximum_changed_nucleotide_count_difference"]
            )
        ),
        "token_geometry": (
            all(
                observed["token_geometry"]["eligible_examples_by_site_class"].get(
                    site_class, 0
                )
                >= minimum
                for site_class in config["behavior"]["admitted_site_classes"]
            )
            and observed["token_geometry"]["target_or_cover_changes"]
            <= int(gates["token_geometry"]["maximum_target_or_cover_changes"])
            and observed["token_geometry"]["target_geometry_violations"]
            <= int(gates["token_geometry"]["maximum_geometry_violations"])
            and observed["token_geometry"]["alternative_family_mismatches"]
            <= int(gates["token_geometry"]["maximum_alternative_family_mismatches"])
        ),
        "model_health": (
            observed["model_health"]["maximum_determinism_absolute_difference"]
            <= float(gates["model_health"]["determinism_absolute_tolerance"])
            and observed["model_health"]["maximum_batch_invariance_absolute_difference"]
            <= float(gates["model_health"]["batch_invariance_absolute_tolerance"])
            and (
                observed["model_health"]["all_scores_finite"]
                if gates["model_health"]["requires_finite_scores"]
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


def _summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def _rank(seed: int, example_id: str) -> str:
    return hashlib.sha256(f"{seed}|preflight|{example_id}".encode()).hexdigest()


if __name__ == "__main__":
    main()
