#!/usr/bin/env python
"""Run the frozen L11H2/L11H5 K/V and residual-write decomposition."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gfmcd.adapters.dnabert2 import head_label, load_dnabert2_promoter_classifier
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_address_payload import (
    capture_residual_payload_rows,
    scan_qkv_operations,
)
from gfmcd.reconstruction_atlas.promoter_circuit import layer_pair_nulls
from gfmcd.reconstruction_atlas.promoter_head_analysis import (
    parse_head_label,
)
from gfmcd.reconstruction_atlas.promoter_head_scan import read_jsonl
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    cluster_bootstrap_interval,
    holm_adjust,
)

Pair = tuple[str, str, str]
PRIMARY_OPERATIONS = ("key_permute", "value_permute")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--expanded-cohort", required=True)
    parser.add_argument("--original-cohort", required=True)
    parser.add_argument("--cohort-audit", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--calibration", required=True)
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
    supply = json.loads(Path(args.supply_audit).read_text())
    preflight = json.loads(Path(args.preflight_report).read_text())
    if (
        supply.get("status") != "QUALIFIED"
        or not preflight.get("qualified")
        or preflight.get("config_sha256") != config_hash
        or preflight.get("checkpoint_revision") != config["checkpoint"]["revision"]
    ):
        raise RuntimeError("qualified matching supply and preflight are required")
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
            raise RuntimeError("development did not authorize confirmation")

    selected = _select_fresh_rows(
        read_jsonl(args.expanded_cohort),
        read_jsonl(args.original_cohort),
        supply,
        config,
        split=args.split,
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
    candidate = tuple(
        parse_head_label(label)
        for label in config["candidate_and_nulls"]["candidate_pair"]
    )
    null_pairs = layer_pair_nulls(
        candidate,
        num_heads=int(config["checkpoint"]["heads_per_layer"]),
    )
    named_pairs = {
        "candidate_L11_pair": candidate,
        **{
            f"null_{index:02d}_{head_label(pair[0])}_{head_label(pair[1])}": pair
            for index, pair in enumerate(null_pairs, start=1)
        },
    }
    dose = int(config["surgery"]["source_token_dose_per_sequence_per_head"])
    seed = int(config["surgery"]["seed"])
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    progress = lambda message: print(f"{args.split}: {message}", flush=True)
    operation_rows = scan_qkv_operations(
        bundle,
        pairs,
        named_pairs,
        (*PRIMARY_OPERATIONS, "key_value_joint_permute"),
        source_token_count=dose,
        seed=seed,
        batch_pairs=args.batch_pairs,
        progress=progress,
    )
    identity_rows = scan_qkv_operations(
        bundle,
        pairs,
        {"candidate_L11_pair": candidate},
        ("identity",),
        source_token_count=dose,
        seed=seed,
        batch_pairs=args.batch_pairs,
        progress=progress,
    )
    residual = capture_residual_payload_rows(
        bundle,
        pairs,
        layer=11,
        candidate_heads=candidate,
        batch_pairs=args.batch_pairs,
        progress=progress,
    )

    inference_seed = int(config["inference"]["seed"]) ^ (
        0xE1E1 if args.split == "development" else 0xF1F1
    )
    primary = _primary_summary(
        operation_rows,
        clusters,
        config,
        development=development,
        seed=inference_seed,
    )
    joint = _operation_summary(
        operation_rows,
        clusters,
        operation="key_value_joint_permute",
        draws=int(config["inference"]["bootstrap_draws"]),
        seed=inference_seed ^ 0x1111,
    )
    identity_max = max(
        abs(float(row[endpoint]))
        for row in identity_rows
        for endpoint in ("target_loss", "control_loss", "target_specific_loss")
    )
    clean = _clean_summary(
        residual["joint_rows"],
        clusters,
        draws=int(config["inference"]["bootstrap_draws"]),
        seed=inference_seed ^ 0x2222,
    )
    residual_summary = _residual_summary(
        residual,
        clusters,
        development=development,
        draws=int(config["inference"]["bootstrap_draws"]),
        seed=inference_seed ^ 0x3333,
    )
    identity_pass = identity_max <= float(
        config["calibration_and_preflight"][
            "maximum_identity_absolute_margin_difference"
        ]
    )
    clean_qualified = clean["mean"] >= 0.5 and clean["ci95"][0] > 0
    qualifying_roles = [
        operation
        for operation in PRIMARY_OPERATIONS
        if primary["tests"][operation]["qualified"]
    ]
    qualified_for_confirmation = (
        args.split == "development"
        and identity_pass
        and clean_qualified
        and bool(qualifying_roles)
    )
    replicated_roles = (
        [
            operation
            for operation in PRIMARY_OPERATIONS
            if primary["tests"][operation]["qualified"]
            and bool(development["primary_inference"]["tests"][operation]["qualified"])
        ]
        if development is not None
        else []
    )

    raw_rows = [
        *({"row_type": "qkv_operation", **row} for row in operation_rows),
        *({"row_type": "identity", **row} for row in identity_rows),
        *({"row_type": "residual_joint", **row} for row in residual["joint_rows"]),
        *({"row_type": "residual_head", **row} for row in residual["head_rows"]),
    ]
    input_paths = [
        args.expanded_cohort,
        args.original_cohort,
        args.cohort_audit,
        args.supply_audit,
        args.calibration,
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
        experiment_id=f"dnabert2_promoter_address_payload_{args.split}_raw_v1",
        status=status,
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "split": args.split,
            "surgery": config["surgery"],
            "primary_gates": config["primary_gates"],
            "residual_payload": config["residual_payload"],
        },
        seeds={"selection_surgery_inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_L11_address_payload_decomposition_v1",
        "split": args.split,
        "status": status,
        "config_sha256": config_hash,
        "n_pairs": len(pairs),
        "n_unique_locus_blocks": len(set(clusters.values())),
        "selected_id_sha256": hashlib.sha256(
            ("\n".join(pair_id for pair_id, _, _ in pairs) + "\n").encode()
        ).hexdigest(),
        "candidate_pair": config["candidate_and_nulls"]["candidate_pair"],
        "source_token_dose": dose,
        "clean_behavior": {**clean, "qualified": clean_qualified},
        "identity_control": {
            "maximum_absolute_margin_difference": identity_max,
            "qualified": identity_pass,
        },
        "primary_inference": primary,
        "joint_KV_position_diagnostic": joint,
        "residual_payload": residual_summary,
        "qualified_for_confirmation": qualified_for_confirmation,
        "replicated_computational_roles": replicated_roles,
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
        experiment_id=f"dnabert2_promoter_address_payload_{args.split}_v1",
        status=status,
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={
            "split": args.split,
            "config_sha256": config_hash,
            "batch_pairs": args.batch_pairs,
        },
        seeds={"selection_surgery_inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "clean_behavior": clean,
                "identity_control": payload["identity_control"],
                "primary_inference": primary,
                "joint_KV_position_diagnostic": joint,
                "residual_payload": residual_summary,
                "qualified_for_confirmation": qualified_for_confirmation,
                "replicated_computational_roles": replicated_roles,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _select_fresh_rows(
    expanded: Sequence[Mapping[str, Any]],
    original: Sequence[Mapping[str, Any]],
    supply: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    split: str,
) -> list[dict[str, Any]]:
    original_ids = {str(row["example_id"]) for row in original}
    original_clusters = {str(row["homology_cluster"]) for row in original}
    fresh = [
        row
        for row in expanded
        if row["split"] == split
        and str(row["example_id"]) not in original_ids
        and str(row["homology_cluster"]) not in original_clusters
    ]
    fresh_by_id = {str(row["example_id"]): dict(row) for row in fresh}
    frozen_ids = supply["selection"]["selected_example_ids"][split]
    selected = [fresh_by_id[example_id] for example_id in frozen_ids]
    if len(selected) != int(config["cohort"]["pairs_per_split"]):
        raise RuntimeError("frozen supply has the wrong target pair count")
    ids = [str(row["example_id"]) for row in selected]
    if ids != supply["selection"]["selected_example_ids"][split]:
        raise RuntimeError("target selection disagrees with frozen supply audit")
    digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    if digest != config["cohort"]["selected_id_sha256"][split]:
        raise RuntimeError("target selection hash disagrees with frozen config")
    return selected


def _primary_summary(
    rows: Sequence[Mapping[str, Any]],
    clusters: Mapping[str, str],
    config: Mapping[str, Any],
    *,
    development: Mapping[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    grouped = _group_rows(rows)
    tests = {}
    raw_p = {}
    gates = config["primary_gates"]
    draws = int(gates["bootstrap_draws"])
    for index, operation in enumerate(PRIMARY_OPERATIONS):
        candidate = grouped[(operation, "candidate_L11_pair")]
        values = [float(row["target_specific_loss"]) for row in candidate]
        order = [clusters[str(row["pair_id"])] for row in candidate]
        observed = statistics.mean(values)
        interval = cluster_bootstrap_interval(
            values,
            order,
            draws=draws,
            seed=seed + index,
        )
        null_means = [
            statistics.mean(float(row["target_specific_loss"]) for row in group)
            for (null_operation, pair_name), group in sorted(grouped.items())
            if null_operation == operation and pair_name != "candidate_L11_pair"
        ]
        raw_p[operation] = add_one_permutation_pvalue(
            abs(observed),
            [abs(value) for value in null_means],
        )
        tests[operation] = {
            "target_loss_mean": statistics.mean(
                float(row["target_loss"]) for row in candidate
            ),
            "control_loss_mean": statistics.mean(
                float(row["control_loss"]) for row in candidate
            ),
            "target_specific_loss_mean": observed,
            "target_specific_loss_ci95": interval,
            "absolute_SESOI": float(gates["absolute_SESOI_logit_margin"]),
            "null_count": len(null_means),
            "null_maximum_absolute_mean": max(abs(value) for value in null_means),
            "raw_add_one_p": raw_p[operation],
        }
    adjusted = holm_adjust(raw_p)
    for operation, test in tests.items():
        interval = test["target_specific_loss_ci95"]
        sign_preserved = (
            development is None
            or test["target_specific_loss_mean"]
            * float(
                development["primary_inference"]["tests"][operation][
                    "target_specific_loss_mean"
                ]
            )
            > 0
        )
        test["holm_adjusted_p"] = adjusted[operation]
        test["confirmation_sign_preserved"] = sign_preserved
        test["qualified"] = (
            abs(test["target_specific_loss_mean"]) >= test["absolute_SESOI"]
            and (interval[0] > 0 or interval[1] < 0)
            and test["holm_adjusted_p"] <= float(gates["familywise_alpha"])
            and sign_preserved
        )
    return {
        "alternative": config["measurement"]["alternative"],
        "tests": tests,
        "qualified_roles": [
            operation
            for operation in PRIMARY_OPERATIONS
            if tests[operation]["qualified"]
        ],
    }


def _operation_summary(
    rows: Sequence[Mapping[str, Any]],
    clusters: Mapping[str, str],
    *,
    operation: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped = _group_rows(rows)
    candidate = grouped[(operation, "candidate_L11_pair")]
    values = [float(row["target_specific_loss"]) for row in candidate]
    order = [clusters[str(row["pair_id"])] for row in candidate]
    observed = statistics.mean(values)
    null_means = [
        statistics.mean(float(row["target_specific_loss"]) for row in group)
        for (null_operation, pair_name), group in sorted(grouped.items())
        if null_operation == operation and pair_name != "candidate_L11_pair"
    ]
    return {
        "operation": operation,
        "status": "secondary_descriptive",
        "target_loss_mean": statistics.mean(
            float(row["target_loss"]) for row in candidate
        ),
        "control_loss_mean": statistics.mean(
            float(row["control_loss"]) for row in candidate
        ),
        "target_specific_loss_mean": observed,
        "target_specific_loss_ci95": cluster_bootstrap_interval(
            values,
            order,
            draws=draws,
            seed=seed,
        ),
        "null_count": len(null_means),
        "null_maximum_absolute_mean": max(abs(value) for value in null_means),
        "uncorrected_add_one_magnitude_p": add_one_permutation_pvalue(
            abs(observed),
            [abs(value) for value in null_means],
        ),
    }


def _clean_summary(
    rows: Sequence[Mapping[str, Any]],
    clusters: Mapping[str, str],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    values = [
        float(row["target_margin"]) - float(row["control_margin"]) for row in rows
    ]
    order = [clusters[str(row["pair_id"])] for row in rows]
    return {
        "mean": statistics.mean(values),
        "ci95": cluster_bootstrap_interval(
            values,
            order,
            draws=draws,
            seed=seed,
        ),
    }


def _residual_summary(
    residual: Mapping[str, Sequence[Mapping[str, Any]]],
    clusters: Mapping[str, str],
    *,
    development: Mapping[str, Any] | None,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    joint = residual["joint_rows"]
    deltas = np.asarray(
        [row["delta_vector"] for row in joint],
        dtype=np.float64,
    )
    mean_delta = deltas.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_delta))
    if mean_norm == 0:
        raise RuntimeError("mean candidate residual delta is exactly zero")
    split_direction = mean_delta / mean_norm
    frozen_direction = (
        split_direction
        if development is None
        else np.asarray(
            development["residual_payload"]["development_mean_direction"],
            dtype=np.float64,
        )
    )
    frozen_direction = frozen_direction / np.linalg.norm(frozen_direction)
    projections = deltas @ frozen_direction
    delta_norms = np.linalg.norm(deltas, axis=1)
    cosines = np.divide(
        projections,
        delta_norms,
        out=np.zeros_like(projections),
        where=delta_norms > 0,
    )
    order = [clusters[str(row["pair_id"])] for row in joint]
    head_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in residual["head_rows"]:
        head_groups[int(row["head"])].append(row)
    return {
        "development_mean_direction": [
            float(value)
            for value in (split_direction if development is None else frozen_direction)
        ],
        "split_mean_delta_norm": mean_norm,
        "mean_per_locus_delta_norm": float(delta_norms.mean()),
        "directional_coherence_ratio": mean_norm / float(delta_norms.mean()),
        "split_mean_direction_cosine_with_development": float(
            split_direction @ frozen_direction
        ),
        "per_locus_projection_on_development_direction": {
            "mean": float(projections.mean()),
            "ci95": cluster_bootstrap_interval(
                [float(value) for value in projections],
                order,
                draws=draws,
                seed=seed,
            ),
            "fraction_positive": float((projections > 0).mean()),
            "mean_cosine": float(cosines.mean()),
        },
        "candidate_pair_write_geometry": {
            key: statistics.mean(float(row[key]) for row in joint)
            for key in (
                "target_write_norm",
                "control_write_norm",
                "target_control_cosine",
                "delta_norm",
            )
        },
        "layer11_head_write_geometry": {
            f"L11H{head}": {
                key: statistics.mean(float(row[key]) for row in rows)
                for key in (
                    "target_write_norm",
                    "control_write_norm",
                    "target_control_cosine",
                    "delta_norm",
                )
            }
            for head, rows in sorted(head_groups.items())
        },
        "boundary": (
            "Residual direction is descriptive in v1; no semantic label or "
            "direction-only causal claim is licensed."
        ),
    }


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["operation"]), str(row["head_pair"]))].append(row)
    return dict(grouped)


if __name__ == "__main__":
    main()
