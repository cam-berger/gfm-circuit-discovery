#!/usr/bin/env python
"""Run the gated all-head promoter attribution and exact CLS-patching atlas."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    head_label,
    load_dnabert2_promoter_classifier,
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
from gfmcd.reconstruction_atlas.promoter_head_analysis import (
    layer_matched_random_head_sets,
    parse_head_label,
    select_head_candidates,
    select_locus_rows,
    serialize_head_set,
    spearman_correlation,
)
from gfmcd.reconstruction_atlas.promoter_head_scan import (
    evaluate_head_set,
    read_jsonl,
    scan_promoter_heads,
)
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    cluster_bootstrap_interval,
    holm_adjust,
    paired_cluster_sign_permutation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--cohort-audit", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--group-calibration", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--development-report")
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patch-batch-pairs", type=int, default=64)
    parser.add_argument("--gradient-batch-pairs", type=int, default=8)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    seed = int(config["seed"])
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
    preflight = load_preflight_report(args.preflight_report)
    preflight.assert_qualified(
        expected_config_sha256=config_hash,
        expected_checkpoint_revision=str(config["checkpoint"]["revision"]),
        expected_input_sha256=input_sha256,
    )
    development = None
    frozen_candidates = None
    if args.split == "confirmation":
        if not args.development_report:
            raise SystemExit(
                "--development-report is required before confirmation can open"
            )
        development = json.loads(Path(args.development_report).read_text())
        if (
            development.get("split") != "development"
            or development.get("preflight_report_id") != preflight.report_id
            or development.get("config_sha256") != config_hash
            or not development.get("qualified_on_split")
        ):
            raise RuntimeError("development head atlas did not authorize confirmation")
        frozen_candidates = [
            parse_head_label(label)
            for label in development["candidate_selection"]["selected_heads"]
        ]
    cohort = read_jsonl(args.cohort)
    selected_rows = select_locus_rows(
        cohort,
        split=args.split,
        maximum=int(config["cohort"]["pairs_per_split"]),
        seed=seed,
        tag=str(config["cohort"]["selection_tag"]),
    )
    clusters = {
        str(row["example_id"]): str(row["homology_cluster"]) for row in selected_rows
    }
    if len(set(clusters.values())) != len(selected_rows):
        raise RuntimeError("promoter head atlas selection is not locus-disjoint")
    pairs = [
        (
            str(row["example_id"]),
            str(row["target"]["clean_sequence"]),
            str(row["control"]["clean_sequence"]),
        )
        for row in selected_rows
    ]
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    scan = scan_promoter_heads(
        bundle,
        pairs,
        patch_batch_pairs=args.patch_batch_pairs,
        gradient_batch_pairs=args.gradient_batch_pairs,
        include_attention=True,
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
    if frozen_candidates is None:
        candidate_rows = select_head_candidates(
            head_summaries,
            candidate_count=int(selection["candidate_count"]),
            maximum_per_layer=int(selection["maximum_per_layer"]),
        )
        candidates = [(int(row["layer"]), int(row["head"])) for row in candidate_rows]
        selection_source = "development_frozen_rank_rule"
    else:
        candidates = frozen_candidates
        selection_source = "unchanged_development_candidates"
    if len(candidates) != int(selection["candidate_count"]):
        raise RuntimeError("head atlas did not supply the frozen candidate count")
    null_config = config["group_validation"]
    null_sets = layer_matched_random_head_sets(
        candidates,
        num_layers=int(config["checkpoint"]["layers"]),
        num_heads=int(config["checkpoint"]["heads_per_layer"]),
        draws=int(null_config["distinct_layer_matched_null_sets"]),
        seed=seed ^ (0x7171 if args.split == "development" else 0x8181),
    )
    candidate_effects = _evaluate_directions(
        bundle,
        pairs,
        candidates,
        batch_pairs=args.patch_batch_pairs,
    )
    null_records = []
    null_means = {"necessity": [], "sufficiency": []}
    for index, null_set in enumerate(null_sets, start=1):
        effects = _evaluate_directions(
            bundle,
            pairs,
            null_set,
            batch_pairs=args.patch_batch_pairs,
        )
        record = {
            "null_set": serialize_head_set(null_set),
            **{
                f"{endpoint}_mean": statistics.mean(values)
                for endpoint, values in effects.items()
            },
        }
        null_records.append(record)
        for endpoint in tuple(null_means):
            null_means[endpoint].append(record[f"{endpoint}_mean"])
        if index % 12 == 0:
            print(
                f"{args.split} layer-matched null set {index}/{len(null_sets)}",
                flush=True,
            )
    inference = _group_inference(
        candidate_effects,
        null_means,
        clusters=[clusters[pair_id] for pair_id, _, _ in pairs],
        config=null_config,
        seed=seed ^ (0x9191 if args.split == "development" else 0xA1A1),
    )
    individual = _individual_candidate_inference(
        scan["head_pair_rows"],
        candidates,
        clusters=clusters,
        config=config,
        seed=seed ^ (0xB1B1 if args.split == "development" else 0xC1C1),
    )
    clean_behavior = _clean_behavior_inference(
        scan["pair_rows"],
        clusters=clusters,
        config=config,
        seed=seed ^ (0xD1D1 if args.split == "development" else 0xE1E1),
    )
    correlation_gate = min(correlations.values()) >= float(
        selection["attribution_exact_minimum_spearman_each_direction"]
    )
    qualified = (
        clean_behavior["qualified"]
        and correlation_gate
        and all(test["qualified"] for test in inference["tests"].values())
    )
    candidate_attention = {
        head_label(head): next(
            row["attention"]
            for row in head_summaries
            if (int(row["layer"]), int(row["head"])) == head
        )
        for head in candidates
    }
    raw_rows = [
        *(
            {
                "row_type": "pair",
                **row,
                "homology_cluster": clusters[str(row["pair_id"])],
            }
            for row in scan["pair_rows"]
        ),
        *(
            {
                "row_type": "head_pair",
                **row,
                "homology_cluster": clusters[str(row["pair_id"])],
            }
            for row in scan["head_pair_rows"]
        ),
        *(
            {
                "row_type": "candidate_group_pair",
                "pair_id": pair_id,
                "homology_cluster": clusters[pair_id],
                "necessity": candidate_effects["necessity"][index],
                "sufficiency": candidate_effects["sufficiency"][index],
            }
            for index, (pair_id, _, _) in enumerate(pairs)
        ),
    ]
    status = "DEVELOPMENT" if args.split == "development" else "SEALED"
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id=f"dnabert2_promoter_head_atlas_{args.split}_raw_v1",
        status=status,
        model_identity=bundle.identity(),
        input_paths=tuple(
            path
            for path in (
                args.gue_test,
                args.cohort,
                args.cohort_audit,
                args.checkpoint_audit,
                args.calibration,
                args.group_calibration,
                args.config,
                args.preflight_report,
                args.development_report,
            )
            if path is not None
        ),
        parameters={
            "split": args.split,
            "pairs": len(pairs),
            "scanned_heads": len(head_summaries),
            "measurement": config["measurement"],
        },
        seeds={"selection_and_inference": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_clean_behavior_CLS_head_attribution_and_patching_v1",
        "split": args.split,
        "status": status,
        "preflight_report_id": preflight.report_id,
        "config_sha256": config_hash,
        "n_pairs": len(pairs),
        "n_unique_locus_blocks": len(set(clusters.values())),
        "scanned_heads": len(head_summaries),
        "scope": config["scope"],
        "clean_behavior": clean_behavior,
        "attribution_exact_patch_spearman": correlations,
        "candidate_selection": {
            "source": selection_source,
            "rule": selection,
            "selected_heads": serialize_head_set(candidates),
            "layer_profile": {
                str(layer): sum(head[0] == layer for head in candidates)
                for layer in sorted({head[0] for head in candidates})
            },
        },
        "candidate_group_inference": inference,
        "individual_candidate_inference": individual,
        "candidate_attention_source_bands": candidate_attention,
        "head_summaries": head_summaries,
        "layer_matched_null_sets": null_records,
        "qualified_on_split": qualified,
        "confirmation_gate": (
            "opened_from_qualified_development"
            if args.split == "confirmation"
            else (
                "confirmation_may_open" if qualified else "confirmation_remains_sealed"
            )
        ),
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(raw_rows),
        },
        "mechanistic_boundary": config["mechanistic_boundary"],
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=f"dnabert2_promoter_head_atlas_{args.split}_v1",
        status=status,
        model_identity=bundle.identity(),
        input_paths=tuple(
            path
            for path in (
                args.gue_test,
                args.cohort,
                args.cohort_audit,
                args.checkpoint_audit,
                args.calibration,
                args.group_calibration,
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
            "measurement": config["measurement"],
            "candidate_selection": selection,
            "group_validation": null_config,
            "patch_batch_pairs": args.patch_batch_pairs,
            "gradient_batch_pairs": args.gradient_batch_pairs,
        },
        seeds={"selection_null_sets_and_inference": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "qualified": qualified,
                "clean_behavior": clean_behavior,
                "spearman": correlations,
                "candidates": serialize_head_set(candidates),
                "candidate_group": inference["tests"],
                "candidate_attention": candidate_attention,
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
    for head, head_rows in sorted(grouped.items()):
        attention_names = tuple(head_rows[0]["target_attention"])
        attention = {}
        for band in attention_names:
            target = statistics.mean(
                float(row["target_attention"][band]) for row in head_rows
            )
            control = statistics.mean(
                float(row["control_attention"][band]) for row in head_rows
            )
            attention[band] = {
                "target_mean": target,
                "control_mean": control,
                "target_minus_control": target - control,
            }
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
                "attention": attention,
            }
        )
    return output


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


def _group_inference(
    effects: dict[str, list[float]],
    null_means: dict[str, list[float]],
    *,
    clusters: list[str],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    tests = {}
    raw_pvalues = {}
    for index, endpoint in enumerate(("necessity", "sufficiency")):
        values = effects[endpoint]
        observed = statistics.mean(values)
        interval = cluster_bootstrap_interval(
            values,
            clusters,
            draws=int(config["bootstrap_draws"]),
            seed=seed ^ ((index + 1) * 0x1111),
        )
        _, sign_null = paired_cluster_sign_permutation(
            values,
            clusters,
            permutations=int(config["sign_permutations"]),
            seed=seed ^ ((index + 1) * 0x2222),
        )
        sign_p = add_one_permutation_pvalue(observed, sign_null, alternative="greater")
        random_set_p = add_one_permutation_pvalue(
            observed, null_means[endpoint], alternative="greater"
        )
        raw_pvalues[endpoint] = random_set_p
        tests[endpoint] = {
            "n_pairs": len(values),
            "n_clusters": len(set(clusters)),
            "mean": observed,
            "cluster_bootstrap_CI": list(interval),
            "p_add_one_sign_diagnostic": sign_p,
            "p_add_one_layer_matched_random_sets": random_set_p,
            "null_set_mean_min": min(null_means[endpoint]),
            "null_set_mean_median": statistics.median(null_means[endpoint]),
            "null_set_mean_max": max(null_means[endpoint]),
        }
    adjusted = holm_adjust(raw_pvalues)
    for endpoint, adjusted_p in adjusted.items():
        sesoi = float(config[f"{endpoint}_SESOI_logit_margin"])
        test = tests[endpoint]
        checks = {
            "mean_at_least_SESOI": test["mean"] >= sesoi,
            "positive_lower_CI": test["cluster_bootstrap_CI"][0] > 0,
            "Holm_random_set_p_at_most_alpha": adjusted_p
            <= float(config["familywise_alpha"]),
        }
        test.update(
            {
                "SESOI_logit_margin": sesoi,
                "p_holm_layer_matched_random_sets": adjusted_p,
                "checks": checks,
                "qualified": all(checks.values()),
            }
        )
    return {
        "tests": tests,
        "candidate_set_size": int(config["candidate_set_size"]),
        "distinct_layer_matched_null_sets": len(null_means["necessity"]),
        "multiple_testing": config["multiple_testing"],
    }


def _individual_candidate_inference(
    rows: list[dict[str, Any]],
    candidates: list[tuple[int, int]],
    *,
    clusters: dict[str, str],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    by_head: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        head = int(row["layer"]), int(row["head"])
        if head in candidates:
            by_head[head].append(row)
    tests = {}
    raw_pvalues = {}
    for head_index, head in enumerate(candidates):
        head_rows = by_head[head]
        cluster_values = [clusters[str(row["pair_id"])] for row in head_rows]
        for endpoint_index, endpoint in enumerate(("necessity", "sufficiency")):
            values = [float(row[endpoint]) for row in head_rows]
            observed, null = paired_cluster_sign_permutation(
                values,
                cluster_values,
                permutations=int(config["group_validation"]["sign_permutations"]),
                seed=seed ^ ((head_index + 1) * 0x1111) ^ endpoint_index,
            )
            pvalue = add_one_permutation_pvalue(observed, null, alternative="greater")
            interval = cluster_bootstrap_interval(
                values,
                cluster_values,
                draws=int(config["group_validation"]["bootstrap_draws"]),
                seed=seed ^ ((head_index + 1) * 0x2222) ^ endpoint_index,
            )
            name = f"{head_label(head)}|{endpoint}"
            raw_pvalues[name] = pvalue
            tests[name] = {
                "head": head_label(head),
                "endpoint": endpoint,
                "mean": observed,
                "cluster_bootstrap_CI": list(interval),
                "p_add_one_sign": pvalue,
            }
    adjusted = holm_adjust(raw_pvalues)
    for name, pvalue in adjusted.items():
        tests[name]["p_holm_sign_16_tests"] = pvalue
        tests[name]["positive_sign_supported"] = tests[name]["cluster_bootstrap_CI"][
            0
        ] > 0 and pvalue <= float(config["group_validation"]["familywise_alpha"])
    return {
        "tests": tests,
        "multiple_testing": "Holm across eight candidates x two directions",
        "boundary": (
            "Positive individual signs are candidate evidence only; individual "
            "heads were not each tested against a matched-random head family."
        ),
    }


def _clean_behavior_inference(
    rows: list[dict[str, Any]],
    *,
    clusters: dict[str, str],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    values = [float(row["clean_enrichment"]) for row in rows]
    cluster_values = [clusters[str(row["pair_id"])] for row in rows]
    observed, null = paired_cluster_sign_permutation(
        values,
        cluster_values,
        permutations=int(config["group_validation"]["sign_permutations"]),
        seed=seed,
    )
    pvalue = add_one_permutation_pvalue(observed, null, alternative="greater")
    interval = cluster_bootstrap_interval(
        values,
        cluster_values,
        draws=int(config["group_validation"]["bootstrap_draws"]),
        seed=seed ^ 0x3333,
    )
    sesoi = float(
        config["confirmation_gate"]["requires_development_clean_enrichment_SESOI"]
    )
    checks = {
        "mean_at_least_SESOI": observed >= sesoi,
        "positive_lower_CI": interval[0] > 0,
        "p_add_one_at_most_alpha": pvalue
        <= float(config["group_validation"]["familywise_alpha"]),
    }
    return {
        "mean": observed,
        "cluster_bootstrap_CI": list(interval),
        "p_add_one_sign": pvalue,
        "SESOI_logit_margin": sesoi,
        "checks": checks,
        "qualified": all(checks.values()),
    }


if __name__ == "__main__":
    main()
