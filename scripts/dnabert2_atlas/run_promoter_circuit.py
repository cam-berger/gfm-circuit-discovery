#!/usr/bin/env python
"""Run the frozen promoter CLS-circuit reverse-engineering battery."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import (
    Head,
    head_label,
    load_dnabert2_promoter_classifier,
)
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_circuit import (
    exact_pair_patch_rows,
    layer_pair_nulls,
    path_rescue_rows,
    query_patch_rows,
    strict_faithfulness_rows,
)
from gfmcd.reconstruction_atlas.promoter_head_analysis import parse_head_label
from gfmcd.reconstruction_atlas.promoter_head_scan import read_jsonl
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    cluster_bootstrap_interval,
    holm_adjust,
)

Pair = tuple[str, str, str]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--head-development-raw", required=True)
    parser.add_argument("--head-confirmation-raw", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--development-report")
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-pairs", type=int, default=64)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    preflight = json.loads(Path(args.preflight_report).read_text())
    if (
        not preflight.get("qualified")
        or preflight.get("config_sha256") != config_hash
        or preflight.get("checkpoint_revision") != config["checkpoint"]["revision"]
    ):
        raise RuntimeError("qualified matching promoter-circuit preflight is required")
    development = None
    if args.split == "confirmation":
        if not args.development_report:
            raise SystemExit("--development-report is required for confirmation")
        development = json.loads(Path(args.development_report).read_text())
        if (
            development.get("split") != "development"
            or development.get("config_sha256") != config_hash
            or not development.get("qualified_for_confirmation")
        ):
            raise RuntimeError("development did not authorize circuit confirmation")

    cohort = read_jsonl(args.cohort)
    excluded_rows = [
        *read_jsonl(args.head_development_raw),
        *read_jsonl(args.head_confirmation_raw),
    ]
    selected = _select_unused_rows(
        cohort,
        excluded_rows,
        split=args.split,
        count=int(config["cohort"]["pairs_per_split"]),
    )
    clusters = {
        str(row["example_id"]): str(row["homology_cluster"]) for row in selected
    }
    pairs: list[Pair] = [
        (
            str(row["example_id"]),
            str(row["target"]["clean_sequence"]),
            str(row["control"]["clean_sequence"]),
        )
        for row in selected
    ]
    cluster_order = [clusters[pair_id] for pair_id, _, _ in pairs]

    candidates = tuple(
        parse_head_label(label) for label in config["frozen_basis"]["candidate_heads"]
    )
    late_pair = tuple(
        parse_head_label(label)
        for label in config["pair_specificity"]["candidate_pair"]
    )
    source_heads = tuple(
        parse_head_label(label) for label in config["path_rescue"]["source_heads"]
    )
    stable_c4 = ((9, 7), (10, 5), (11, 2), (11, 5))
    all_heads = tuple(
        (layer, head)
        for layer in range(int(config["checkpoint"]["layers"]))
        for head in range(int(config["checkpoint"]["heads_per_layer"]))
    )
    all_layers = tuple(range(int(config["checkpoint"]["layers"])))
    null_pairs = layer_pair_nulls(
        late_pair,
        num_heads=int(config["checkpoint"]["heads_per_layer"]),
    )
    named_pairs: dict[str, Sequence[Head]] = {
        "candidate_L11_pair": late_pair,
        **{
            f"null_{index:02d}_{head_label(pair[0])}_{head_label(pair[1])}": pair
            for index, pair in enumerate(null_pairs, start=1)
        },
    }
    conditions: dict[str, tuple[Sequence[Head], Sequence[int]]] = {
        "empty": ((), ()),
        "L11_pair": (late_pair, ()),
        "stable_C4": (stable_c4, ()),
        "candidate_C8": (candidates, ()),
        "all_MLPs_only": ((), all_layers),
        "all_heads_only": (all_heads, ()),
        "candidate_C8_plus_all_MLPs": (candidates, all_layers),
        "all_nodes": (all_heads, all_layers),
    }
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    progress = lambda message: print(f"{args.split}: {message}", flush=True)
    faithfulness_rows = strict_faithfulness_rows(
        bundle,
        pairs,
        conditions,
        batch_pairs=args.batch_pairs,
        progress=progress,
    )
    pair_rows = exact_pair_patch_rows(
        bundle,
        pairs,
        named_pairs,
        batch_pairs=args.batch_pairs,
        progress=progress,
    )
    query_rows = query_patch_rows(
        bundle,
        pairs,
        named_pairs,
        batch_pairs=args.batch_pairs,
        progress=progress,
    )
    rescue_rows = path_rescue_rows(
        bundle,
        pairs,
        corrupt_heads=source_heads,
        restore_pairs=named_pairs,
        batch_pairs=args.batch_pairs,
        progress=progress,
    )

    seed = int(config["inference"]["seed"]) ^ (
        0xC1C1 if args.split == "development" else 0xD1D1
    )
    draws = int(config["inference"]["bootstrap_draws"])
    faithfulness = _faithfulness_summary(
        faithfulness_rows,
        clusters,
        draws=draws,
        seed=seed,
    )
    pair_specificity = _pair_specificity_summary(
        pair_rows,
        clusters,
        config["pair_specificity"],
        draws=draws,
        seed=seed ^ 0x1111,
    )
    query_dependence = _single_endpoint_specificity(
        query_rows,
        clusters,
        group_key="head_pair",
        endpoint="query_necessity",
        candidate_key="candidate_L11_pair",
        sesoi=float(config["query_dependence"]["SESOI_logit_margin"]),
        draws=draws,
        seed=seed ^ 0x2222,
    )
    path_rescue = _path_rescue_summary(
        rescue_rows,
        clusters,
        config["path_rescue"],
        draws=draws,
        seed=seed ^ 0x3333,
    )
    clean = _clean_summary(
        faithfulness_rows,
        clusters,
        draws=draws,
        seed=seed ^ 0x4444,
    )
    anchors_pass = faithfulness["anchors"]["empty_max_abs_error"] <= float(
        config["preflight"]["maximum_absolute_anchor_error"]
    ) and faithfulness["anchors"]["all_nodes_max_abs_error"] <= float(
        config["preflight"]["maximum_absolute_noop_difference"]
    )
    clean_qualified = clean["mean"] >= 0.5 and clean["ci95"][0] > 0
    causal_tests = {
        "pair_specificity": pair_specificity["qualified"],
        "query_dependence": query_dependence["qualified"],
        "path_rescue": path_rescue["qualified"],
    }
    qualified_for_confirmation = (
        args.split == "development"
        and anchors_pass
        and clean_qualified
        and any(causal_tests.values())
    )
    replicated = (
        args.split == "confirmation"
        and anchors_pass
        and clean_qualified
        and any(causal_tests.values())
    )

    raw_rows = [
        *({"row_type": "faithfulness", **row} for row in faithfulness_rows),
        *({"row_type": "pair_specificity", **row} for row in pair_rows),
        *({"row_type": "query_dependence", **row} for row in query_rows),
        *({"row_type": "path_rescue", **row} for row in rescue_rows),
    ]
    input_paths = [
        args.cohort,
        args.head_development_raw,
        args.head_confirmation_raw,
        args.config,
        args.preflight_report,
        args.checkpoint_audit,
    ]
    if args.development_report:
        input_paths.append(args.development_report)
    status = "DEVELOPMENT" if args.split == "development" else "SEALED"
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id=f"dnabert2_promoter_circuit_{args.split}_raw_v1",
        status=status,
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "split": args.split,
            "node_universe": config["node_universe"],
            "faithfulness": config["faithfulness"],
            "pair_specificity": config["pair_specificity"],
            "query_dependence": config["query_dependence"],
            "path_rescue": config["path_rescue"],
        },
        seeds={"selection_and_inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_CLS_circuit_reverse_engineering_v1",
        "split": args.split,
        "status": status,
        "config_sha256": config_hash,
        "preflight_report": str(args.preflight_report),
        "preflight_report_sha256": sha256_file(args.preflight_report),
        "n_pairs": len(pairs),
        "n_unique_locus_blocks": len(set(cluster_order)),
        "excluded_head_atlas_example_count": len(
            {
                str(row["pair_id"])
                for row in excluded_rows
                if row.get("row_type") == "pair"
            }
        ),
        "candidate_heads": [head_label(head) for head in candidates],
        "clean_behavior": {**clean, "qualified": clean_qualified},
        "faithfulness": faithfulness,
        "pair_specificity": pair_specificity,
        "query_dependence": query_dependence,
        "path_rescue": path_rescue,
        "causal_test_qualification": causal_tests,
        "anchors_pass": anchors_pass,
        "qualified_for_confirmation": qualified_for_confirmation,
        "replicated_causal_signal": replicated,
        "confirmation_gate": (
            "not_applicable"
            if args.split == "confirmation"
            else (
                "confirmation_may_open"
                if qualified_for_confirmation
                else "confirmation_remains_sealed"
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
        experiment_id=f"dnabert2_promoter_circuit_{args.split}_v1",
        status=status,
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={
            "split": args.split,
            "config_sha256": config_hash,
            "batch_pairs": args.batch_pairs,
        },
        seeds={"selection_and_inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "clean_behavior": clean,
                "faithfulness": faithfulness,
                "pair_specificity": pair_specificity,
                "query_dependence": query_dependence,
                "path_rescue": path_rescue,
                "qualified_for_confirmation": qualified_for_confirmation,
                "replicated_causal_signal": replicated,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _select_unused_rows(
    cohort: Sequence[Mapping[str, Any]],
    excluded_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    count: int,
) -> list[dict[str, Any]]:
    excluded_ids = {
        str(row["pair_id"]) for row in excluded_rows if row.get("row_type") == "pair"
    }
    excluded_clusters = {
        str(row["homology_cluster"])
        for row in excluded_rows
        if row.get("row_type") == "pair"
    }
    eligible = [
        dict(row)
        for row in cohort
        if row.get("split") == split
        and str(row["example_id"]) not in excluded_ids
        and str(row["homology_cluster"]) not in excluded_clusters
    ]
    by_cluster: dict[str, dict[str, Any]] = {}
    for row in sorted(eligible, key=lambda item: str(item["example_id"])):
        by_cluster.setdefault(str(row["homology_cluster"]), row)
    selected = list(by_cluster.values())
    if len(selected) != count:
        raise ValueError(
            f"unused {split} cohort supplied {len(selected)} unique loci, expected {count}"
        )
    return selected


def _faithfulness_summary(
    rows: Sequence[Mapping[str, Any]],
    clusters: Mapping[str, str],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped = _group_rows(rows, "condition")
    summaries = {}
    for index, (condition, condition_rows) in enumerate(sorted(grouped.items())):
        numerator = [float(row["retained_enrichment"]) for row in condition_rows]
        denominator = [float(row["clean_enrichment"]) for row in condition_rows]
        order = [clusters[str(row["pair_id"])] for row in condition_rows]
        summaries[condition] = {
            "retained_enrichment_mean": statistics.mean(numerator),
            "native_enrichment_mean": statistics.mean(denominator),
            "faithfulness": statistics.mean(numerator) / statistics.mean(denominator),
            "faithfulness_ci95": _cluster_ratio_interval(
                numerator,
                denominator,
                order,
                draws=draws,
                seed=seed + index,
            ),
        }
    empty = grouped["empty"]
    all_nodes = grouped["all_nodes"]
    return {
        "definition": "mean(retained target-control enrichment) / mean(native target-control enrichment)",
        "conditions": summaries,
        "anchors": {
            "empty_max_abs_error": max(
                abs(
                    float(row["patched_target_margin"])
                    - float(row["native_control_margin"])
                )
                for row in empty
            ),
            "all_nodes_max_abs_error": max(
                abs(
                    float(row["patched_target_margin"])
                    - float(row["native_target_margin"])
                )
                for row in all_nodes
            ),
        },
    }


def _pair_specificity_summary(
    rows: Sequence[Mapping[str, Any]],
    clusters: Mapping[str, str],
    config: Mapping[str, Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped = _group_rows(rows, "head_pair")
    candidate = grouped["candidate_L11_pair"]
    tests: dict[str, dict[str, Any]] = {}
    raw_p = {}
    for index, endpoint in enumerate(("necessity", "sufficiency")):
        values = [float(row[endpoint]) for row in candidate]
        order = [clusters[str(row["pair_id"])] for row in candidate]
        observed = statistics.mean(values)
        null_means = [
            statistics.mean(float(row[endpoint]) for row in group)
            for name, group in sorted(grouped.items())
            if name != "candidate_L11_pair"
        ]
        raw_p[endpoint] = add_one_permutation_pvalue(observed, null_means)
        tests[endpoint] = {
            "mean": observed,
            "ci95": cluster_bootstrap_interval(
                values,
                order,
                draws=draws,
                seed=seed + index,
            ),
            "null_count": len(null_means),
            "null_max": max(null_means),
            "raw_add_one_p": raw_p[endpoint],
        }
    adjusted = holm_adjust(raw_p)
    sesoi = float(config["SESOI_logit_margin_each_endpoint"])
    for endpoint, test in tests.items():
        test["holm_adjusted_p"] = adjusted[endpoint]
        test["qualified"] = (
            test["mean"] >= sesoi
            and test["ci95"][0] > 0
            and test["holm_adjusted_p"] <= float(config["familywise_alpha"])
        )
    return {
        "candidate_pair": config["candidate_pair"],
        "tests": tests,
        "qualified": all(test["qualified"] for test in tests.values()),
    }


def _single_endpoint_specificity(
    rows: Sequence[Mapping[str, Any]],
    clusters: Mapping[str, str],
    *,
    group_key: str,
    endpoint: str,
    candidate_key: str,
    sesoi: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped = _group_rows(rows, group_key)
    candidate = grouped[candidate_key]
    values = [float(row[endpoint]) for row in candidate]
    order = [clusters[str(row["pair_id"])] for row in candidate]
    observed = statistics.mean(values)
    null_means = [
        statistics.mean(float(row[endpoint]) for row in group)
        for name, group in sorted(grouped.items())
        if name != candidate_key
    ]
    pvalue = add_one_permutation_pvalue(observed, null_means)
    interval = cluster_bootstrap_interval(
        values,
        order,
        draws=draws,
        seed=seed,
    )
    qualified = observed >= sesoi and interval[0] > 0 and pvalue <= 0.05
    return {
        "endpoint": endpoint,
        "mean": observed,
        "ci95": interval,
        "SESOI": sesoi,
        "null_count": len(null_means),
        "null_max": max(null_means),
        "add_one_p": pvalue,
        "qualified": qualified,
    }


def _path_rescue_summary(
    rows: Sequence[Mapping[str, Any]],
    clusters: Mapping[str, str],
    config: Mapping[str, Any],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped = _group_rows(rows, "restore_pair")
    candidate = grouped["candidate_L11_pair"]
    source = [float(row["source_loss"]) for row in candidate]
    rescue = [float(row["rescue"]) for row in candidate]
    order = [clusters[str(row["pair_id"])] for row in candidate]
    source_mean = statistics.mean(source)
    rescue_mean = statistics.mean(rescue)
    null_means = [
        statistics.mean(float(row["rescue"]) for row in group)
        for name, group in sorted(grouped.items())
        if name != "candidate_L11_pair"
    ]
    pvalue = add_one_permutation_pvalue(rescue_mean, null_means)
    source_ci = cluster_bootstrap_interval(
        source,
        order,
        draws=draws,
        seed=seed,
    )
    rescue_ci = cluster_bootstrap_interval(
        rescue,
        order,
        draws=draws,
        seed=seed + 1,
    )
    fraction_ci = _cluster_ratio_interval(
        rescue,
        source,
        order,
        draws=draws,
        seed=seed + 2,
    )
    qualified = (
        source_mean >= float(config["source_loss_SESOI_logit_margin"])
        and source_ci[0] > 0
        and rescue_ci[0] > 0
        and pvalue <= float(config["alpha"])
    )
    return {
        "source_loss_mean": source_mean,
        "source_loss_ci95": source_ci,
        "rescue_mean": rescue_mean,
        "rescue_ci95": rescue_ci,
        "rescue_fraction": rescue_mean / source_mean,
        "rescue_fraction_ci95": fraction_ci,
        "null_count": len(null_means),
        "null_max": max(null_means),
        "add_one_p": pvalue,
        "qualified": qualified,
    }


def _clean_summary(
    rows: Sequence[Mapping[str, Any]],
    clusters: Mapping[str, str],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    unique = _group_rows(rows, "condition")["all_nodes"]
    values = [float(row["clean_enrichment"]) for row in unique]
    order = [clusters[str(row["pair_id"])] for row in unique]
    return {
        "mean": statistics.mean(values),
        "ci95": cluster_bootstrap_interval(
            values,
            order,
            draws=draws,
            seed=seed,
        ),
    }


def _cluster_ratio_interval(
    numerator: Sequence[float],
    denominator: Sequence[float],
    clusters: Sequence[str],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    if not (len(numerator) == len(denominator) == len(clusters)) or not numerator:
        raise ValueError("ratio bootstrap inputs must be non-empty and aligned")
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for top, bottom, cluster in zip(numerator, denominator, clusters):
        if not math.isfinite(top) or not math.isfinite(bottom):
            raise ValueError("ratio bootstrap inputs must be finite")
        grouped[str(cluster)].append((float(top), float(bottom)))
    keys = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [rng.choice(keys) for _ in keys]
        values = [value for key in sampled for value in grouped[key]]
        denominator_mean = statistics.mean(value[1] for value in values)
        if math.isclose(denominator_mean, 0.0, abs_tol=1e-12):
            raise ValueError("ratio bootstrap denominator reached zero")
        estimates.append(
            statistics.mean(value[0] for value in values) / denominator_mean
        )
    estimates.sort()
    lo = max(0, math.floor(0.025 * draws))
    hi = min(draws - 1, math.ceil(0.975 * draws) - 1)
    return estimates[lo], estimates[hi]


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)


if __name__ == "__main__":
    main()
