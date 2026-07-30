#!/usr/bin/env python
"""Run the frozen class-specific DNABERT-2 context-margin atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
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
    frozen_config_sha256,
    load_preflight_report,
)
from gfmcd.reconstruction_atlas.schema import Interval, load_collection
from gfmcd.reconstruction_atlas.scoring import ReconstructionScorer, ScoringRequest
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    cluster_bootstrap_interval,
    holm_adjust,
    paired_cluster_sign_permutation,
)
from gfmcd.reconstruction_atlas.tokenization import geometry_is_invariant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--development-report")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    seed = int(args.seed if args.seed is not None else config["seed"])
    input_sha256 = {
        "collection": sha256_file(args.collection),
        "variants": sha256_file(args.variants),
        "supply_audit": sha256_file(args.supply_audit),
    }
    preflight = load_preflight_report(args.preflight_report)
    preflight.assert_qualified(
        expected_config_sha256=frozen_config_sha256(args.config),
        expected_checkpoint_revision=DNABERT2_REVISION,
        expected_input_sha256=input_sha256,
    )
    admitted = frozenset(str(x) for x in config["behavior"]["admitted_site_classes"])
    screened = admitted
    development = None
    if args.split == "confirmation":
        if not args.development_report:
            raise SystemExit(
                "--development-report is required before confirmation can open"
            )
        development = json.loads(Path(args.development_report).read_text())
        if development.get("split") != "development":
            raise SystemExit("development report carries the wrong split")
        if development.get("preflight_report_id") != preflight.report_id:
            raise SystemExit("development and preflight reports disagree")
        if development.get("config_sha256") != preflight.config_sha256:
            raise SystemExit("development and frozen config disagree")
        screened = frozenset(
            site_class
            for site_class, decision in development["behavior_decisions"].items()
            if bool(decision["qualified_on_split"])
        )
        if not screened:
            raise SystemExit(
                "no development-qualified classes; confirmation remains sealed"
            )
        if not screened <= admitted:
            raise SystemExit("development report qualifies an unregistered class")

    records = {
        record.example_id: record
        for record in load_collection(args.collection)
        if record.split == args.split
    }
    variants = [
        row
        for row in _read_jsonl(args.variants)
        if row["split"] == args.split and row["site_class"] in screened
    ]
    selected = _select_rows(
        variants,
        classes=screened,
        maximum_per_class=int(
            config["behavior"]["development_maximum_targets_per_class"]
        ),
        seed=seed,
    )
    if not selected:
        raise SystemExit("no eligible target-preserving context variants")

    bundle = load_dnabert2_mlm(device=args.device)
    adapter = Dnabert2Adapter(bundle)
    scorer = ReconstructionScorer(adapter)
    requests, metadata = _requests(selected, records)
    scores = {
        score.request_id: score
        for score in scorer.score(requests, batch_size=args.batch_size)
    }
    effect_rows, audit = _effects(scores, metadata)
    if any(value for value in audit.values()):
        raise RuntimeError(f"target scoring invariant violation: {audit}")
    inference = _inference(
        effect_rows,
        permutations=int(config["behavior"]["permutation_draws"]),
        bootstrap_draws=args.bootstrap_draws,
        seed=seed,
    )
    decisions = _decisions(effect_rows, inference, config, screened)
    raw_rows = [
        {
            **score.to_dict(),
            **metadata[score.request_id],
        }
        for score in scores.values()
    ]
    status = "DEVELOPMENT" if args.split == "development" else "SEALED"
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id=f"dnabert2_context_margin_{args.split}_raw",
        status=status,
        model_identity=adapter.identity(),
        input_paths=tuple(
            path
            for path in (
                args.collection,
                args.variants,
                args.supply_audit,
                args.config,
                args.preflight_report,
                args.development_report,
            )
            if path is not None
        ),
        parameters={
            "split": args.split,
            "batch_size": args.batch_size,
            "measurement": "geometry_qualified_logsumexp_alternative_margin",
            "primary_effect": "irrelevant_margin_minus_structured_margin",
            "n_scored_sequences": len(raw_rows),
        },
        seeds={"selection_and_inference": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "geometry_qualified_margin_class_specific_context",
        "split": args.split,
        "status": status,
        "preflight_report_id": preflight.report_id,
        "preflight_decision": preflight.decision,
        "config_sha256": preflight.config_sha256,
        "preregistered_site_classes": sorted(admitted),
        "screened_site_classes": sorted(screened),
        "quarantined_site_classes": config["behavior"]["quarantined_site_classes"],
        "confirmation_selection": (
            "development-qualified classes only"
            if args.split == "confirmation"
            else "all preflight-admitted classes"
        ),
        "n_targets": len(effect_rows),
        "n_scored_sequences": len(raw_rows),
        "measurement": config["measurement"],
        "effect_definition": {
            "primary": "margin(irrelevant_shuffle)-margin(structured_shuffle)",
            "secondary": "true_score(irrelevant_shuffle)-true_score(structured_shuffle)",
            "positive_primary_interpretation": "class-specific context supports the true target over its fixed eligible alternatives more than equal-dose irrelevant context",
        },
        "site_class_summary": _class_summary(effect_rows),
        "inference": inference,
        "behavior_decisions": decisions,
        "target_scoring_audit": audit,
        "raw_scores": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(raw_rows),
        },
        "mechanistic_boundary": config["mechanistic_boundary"],
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=f"dnabert2_context_margin_atlas_{args.split}",
        status=status,
        model_identity=adapter.identity(),
        input_paths=tuple(
            path
            for path in (
                args.collection,
                args.variants,
                args.supply_audit,
                args.config,
                args.preflight_report,
                args.development_report,
                raw_path,
                raw_manifest,
            )
            if path is not None
        ),
        parameters={
            "split": args.split,
            "batch_size": args.batch_size,
            "bootstrap_draws": args.bootstrap_draws,
            "permutations": int(config["behavior"]["permutation_draws"]),
            "measurement": "geometry_qualified_logsumexp_alternative_margin",
            "primary_effect": "irrelevant_margin_minus_structured_margin",
            "multiplicity": config["behavior"]["multiple_testing"],
            "screened_site_classes": sorted(screened),
        },
        seeds={"selection_and_inference": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "n_targets": len(effect_rows),
                "qualified": sorted(
                    name
                    for name, decision in decisions.items()
                    if decision["qualified_on_split"]
                ),
                "class_effects": {
                    name: {
                        "mean": test["observed_cluster_mean"],
                        "ci": test["cluster_bootstrap_ci"],
                        "p_holm": test["p_holm"],
                    }
                    for name, test in inference["primary_tests"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    classes: frozenset[str],
    maximum_per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["site_class"])].append(row)
    selected = []
    for site_class in sorted(classes):
        seen_clusters: set[str] = set()
        ranked = sorted(
            by_class.get(site_class, []),
            key=lambda row: _rank(seed, str(row["example_id"])),
        )
        for row in ranked:
            cluster = str(row["homology_cluster"])
            if cluster in seen_clusters:
                continue
            seen_clusters.add(cluster)
            selected.append(row)
            if len(seen_clusters) >= maximum_per_class:
                break
    return selected


def _requests(
    rows: list[dict[str, Any]],
    records: dict[str, Any],
) -> tuple[list[ScoringRequest], dict[str, dict[str, Any]]]:
    requests = []
    metadata = {}
    for row in rows:
        example_id = str(row["example_id"])
        record = records[example_id]
        target = Interval(int(row["target"]["start"]), int(row["target"]["end"]))
        alternatives = {str(k): str(v) for k, v in row["alternatives"].items()}
        sequences = {
            "reference": record.sequence,
            "structured": str(row["structured_sequence"]),
            "irrelevant": str(row["irrelevant_sequence"]),
        }
        for variant, sequence in sequences.items():
            request_id = f"{example_id}|{variant}"
            requests.append(
                ScoringRequest(
                    request_id=request_id,
                    sequence=sequence,
                    site=target,
                    alternatives=alternatives,
                )
            )
            metadata[request_id] = {
                "example_id": example_id,
                "site_class": str(row["site_class"]),
                "homology_cluster": str(row["homology_cluster"]),
                "element_id": str(row["element_id"]),
                "variant": variant,
                "expected_alternative_count": len(alternatives),
                "structured_changed_nucleotides": int(
                    row["audit"]["structured_changed_nucleotides"]
                ),
                "irrelevant_changed_nucleotides": int(
                    row["audit"]["irrelevant_changed_nucleotides"]
                ),
            }
    return requests, metadata


def _effects(
    scores: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    identities: dict[str, dict[str, Any]] = {}
    audit = {
        "nonfinite_scores": 0,
        "alternative_family_mismatches": 0,
        "geometry_violations": 0,
        "dose_mismatches": 0,
    }
    for request_id, score in scores.items():
        info = metadata[request_id]
        example_id = str(info["example_id"])
        margin = score.alternative_logsumexp_margin_per_covered_nt
        if margin is None or not math.isfinite(margin):
            audit["nonfinite_scores"] += 1
        if (
            score.eligible_alternative_count != int(info["expected_alternative_count"])
            or score.ineligible_alternatives
        ):
            audit["alternative_family_mismatches"] += 1
        grouped[example_id][str(info["variant"])] = score
        identities[example_id] = info
    rows = []
    for example_id, variants in sorted(grouped.items()):
        if set(variants) != {"reference", "structured", "irrelevant"}:
            raise RuntimeError(f"incomplete target variant set: {example_id}")
        reference = variants["reference"]
        structured = variants["structured"]
        irrelevant = variants["irrelevant"]
        if not (
            geometry_is_invariant(reference.mask_plan, structured.mask_plan)
            and geometry_is_invariant(reference.mask_plan, irrelevant.mask_plan)
        ):
            audit["geometry_violations"] += 1
        info = identities[example_id]
        if int(info["structured_changed_nucleotides"]) != int(
            info["irrelevant_changed_nucleotides"]
        ):
            audit["dose_mismatches"] += 1
        reference_margin = float(reference.alternative_logsumexp_margin_per_covered_nt)
        structured_margin = float(
            structured.alternative_logsumexp_margin_per_covered_nt
        )
        irrelevant_margin = float(
            irrelevant.alternative_logsumexp_margin_per_covered_nt
        )
        rows.append(
            {
                "example_id": example_id,
                "site_class": info["site_class"],
                "homology_cluster": info["homology_cluster"],
                "element_id": info["element_id"],
                "eligible_alternative_count": reference.eligible_alternative_count,
                "covered_nucleotides": reference.mask_plan.covered_nucleotides,
                "BPE_overhang_nucleotides": (reference.mask_plan.overhang_nucleotides),
                "changed_nucleotides": int(info["structured_changed_nucleotides"]),
                "reference_margin": reference_margin,
                "structured_margin": structured_margin,
                "irrelevant_margin": irrelevant_margin,
                "structured_margin_loss_from_reference": (
                    reference_margin - structured_margin
                ),
                "irrelevant_margin_loss_from_reference": (
                    reference_margin - irrelevant_margin
                ),
                "primary_context_effect": (irrelevant_margin - structured_margin),
                "secondary_true_score_context_effect": (
                    irrelevant.true_token_log_probability_per_covered_nt
                    - structured.true_token_log_probability_per_covered_nt
                ),
            }
        )
    return rows, audit


def _inference(
    rows: list[dict[str, Any]],
    *,
    permutations: int,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["site_class"])].append(row)
    tests = {}
    raw_pvalues = {}
    for site_class, class_rows in sorted(grouped.items()):
        values = [float(row["primary_context_effect"]) for row in class_rows]
        clusters = [str(row["homology_cluster"]) for row in class_rows]
        observed, null = paired_cluster_sign_permutation(
            values,
            clusters,
            permutations=permutations,
            seed=seed,
        )
        pvalue = add_one_permutation_pvalue(
            observed,
            null,
            alternative="greater",
        )
        tests[site_class] = {
            "n_targets": len(class_rows),
            "n_clusters": len(set(clusters)),
            "observed_cluster_mean": observed,
            "cluster_bootstrap_ci": list(
                cluster_bootstrap_interval(
                    values,
                    clusters,
                    draws=bootstrap_draws,
                    seed=seed,
                )
            ),
            "p_add_one": pvalue,
            "permutations": len(null),
        }
        raw_pvalues[site_class] = pvalue
    adjusted = holm_adjust(raw_pvalues)
    for site_class, pvalue in adjusted.items():
        tests[site_class]["p_holm"] = pvalue
    return {
        "primary_tests": tests,
        "family": sorted(tests),
        "multiple_testing": "Holm",
        "pvalue": "add-one one-sided cluster sign permutation",
    }


def _decisions(
    rows: list[dict[str, Any]],
    inference: dict[str, Any],
    config: dict[str, Any],
    screened: frozenset[str],
) -> dict[str, Any]:
    minimum = int(config["behavior"]["minimum_clusters_per_class"])
    sesoi = float(config["behavior"]["primary_sesoi_nats_per_covered_nt"])
    alpha = float(config["behavior"]["familywise_alpha"])
    output = {}
    for site_class in sorted(screened):
        test = inference["primary_tests"].get(site_class)
        if test is None:
            output[site_class] = {
                "qualified_on_split": False,
                "reason": "no eligible target rows",
            }
            continue
        checks = {
            "minimum_clusters": int(test["n_clusters"]) >= minimum,
            "mean_at_least_SESOI": float(test["observed_cluster_mean"]) >= sesoi,
            "positive_lower_95pct_bound": float(test["cluster_bootstrap_ci"][0]) > 0,
            "Holm_p_at_most_alpha": float(test["p_holm"]) <= alpha,
        }
        output[site_class] = {
            "qualified_on_split": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "minimum_clusters": minimum,
                "SESOI": sesoi,
                "familywise_alpha": alpha,
            },
        }
    return output


def _class_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["site_class"])].append(row)
    output = {}
    for site_class, class_rows in sorted(grouped.items()):
        output[site_class] = {
            "n_targets": len(class_rows),
            "n_clusters": len({str(row["homology_cluster"]) for row in class_rows}),
            "primary_context_effect_mean": statistics.mean(
                float(row["primary_context_effect"]) for row in class_rows
            ),
            "secondary_true_score_context_effect_mean": statistics.mean(
                float(row["secondary_true_score_context_effect"]) for row in class_rows
            ),
            "structured_margin_loss_from_reference_mean": statistics.mean(
                float(row["structured_margin_loss_from_reference"])
                for row in class_rows
            ),
            "irrelevant_margin_loss_from_reference_mean": statistics.mean(
                float(row["irrelevant_margin_loss_from_reference"])
                for row in class_rows
            ),
            "eligible_alternative_count": _summary(
                [int(row["eligible_alternative_count"]) for row in class_rows]
            ),
            "covered_nucleotides": _summary(
                [int(row["covered_nucleotides"]) for row in class_rows]
            ),
            "BPE_overhang_nucleotides": _summary(
                [int(row["BPE_overhang_nucleotides"]) for row in class_rows]
            ),
            "changed_nucleotides": _summary(
                [int(row["changed_nucleotides"]) for row in class_rows]
            ),
        }
    return output


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
    return hashlib.sha256(f"{seed}|target|{example_id}".encode()).hexdigest()


if __name__ == "__main__":
    main()
