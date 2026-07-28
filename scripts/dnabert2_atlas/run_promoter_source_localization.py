#!/usr/bin/env python
"""Run the frozen promoter source-localization and biological qualification."""

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

from gfmcd.adapters.dnabert2 import (
    head_label,
    load_dnabert2_promoter_classifier,
)
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_circuit import layer_pair_nulls
from gfmcd.reconstruction_atlas.promoter_head_scan import (
    native_pair_rows,
    read_jsonl,
)
from gfmcd.reconstruction_atlas.promoter_source_execution import (
    scan_equal_dose_tiles,
    scan_head_pair_regional_values,
    score_motif_disruptions,
)
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    cluster_bootstrap_interval,
    holm_adjust,
)

Pair = tuple[str, str, str]
PRIMARY_OPERATIONS = ("mask", "candidate_value_swap")
ANNOTATIONS = {
    "CAGE_dominant_CTSS": "FANTOM5_CAGE_dominant_CTSS",
    "PLS_cCRE": "ENCODE_PLS_cCRE",
    "CpG_island": "UCSC_CpG_island",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--annotated-cohort", required=True)
    parser.add_argument("--cohort-manifest", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--development-report")
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=32)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    supply = json.loads(Path(args.supply_audit).read_text())
    preflight = json.loads(Path(args.preflight_report).read_text())
    if (
        config.get("status") != "FROZEN"
        or supply.get("status") != "QUALIFIED"
        or not preflight.get("qualified")
        or preflight.get("config_sha256") != config_hash
        or preflight.get("checkpoint_revision") != config["checkpoint"]["revision"]
    ):
        raise RuntimeError("matching frozen config, supply, and preflight required")
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

    all_rows = read_jsonl(args.annotated_cohort)
    rows = _select_rows(all_rows, supply, config, split=args.split)
    row_by_id = {str(row["example_id"]): row for row in rows}
    pairs: list[Pair] = [
        (
            str(row["example_id"]),
            str(row["target"]["clean_sequence"]),
            str(row["control"]["clean_sequence"]),
        )
        for row in rows
    ]
    clusters = {str(row["example_id"]): str(row["homology_cluster"]) for row in rows}
    qualified_development_tests = (
        []
        if development is None
        else [
            test
            for test in development["primary_localization"]["tests"].values()
            if test["qualified"]
        ]
    )
    requested_tiles = (
        None
        if development is None
        else sorted({int(test["tile_index"]) for test in qualified_development_tests})
    )
    operations_by_tile = (
        None
        if development is None
        else {
            tile: tuple(
                sorted(
                    {
                        str(test["operation"])
                        for test in qualified_development_tests
                        if int(test["tile_index"]) == tile
                    }
                )
            )
            for tile in requested_tiles
        }
    )
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    geometry = config["tile_geometry"]
    progress = lambda message: print(f"{args.split}: {message}", flush=True)
    tile_rows = scan_equal_dose_tiles(
        bundle,
        pairs,
        tile_width_nt=int(geometry["nominal_tile_width_nt"]),
        token_dose=int(geometry["fixed_BPE_token_dose_per_sequence"]),
        maximum_cover_nt=int(geometry["maximum_actual_cover_nt"]),
        maximum_cover_difference_nt=int(
            geometry["maximum_target_control_cover_length_difference_nt"]
        ),
        candidate_heads=((11, 2), (11, 5)),
        include_all_layer_heads=False,
        include_identity=True,
        batch_pairs=args.batch_pairs,
        tile_indices=requested_tiles,
        operations_by_tile=operations_by_tile,
        progress=progress,
    )
    native_rows = native_pair_rows(bundle, pairs, batch_pairs=args.batch_pairs)
    clean = _clean_summary(
        native_rows,
        clusters,
        draws=int(config["inference"]["bootstrap_draws"]),
        seed=_split_seed(config, args.split) ^ 0x1010,
    )
    identity_max = max(
        abs(float(row[endpoint]))
        for row in tile_rows
        if row["operation"] == "candidate_identity"
        for endpoint in ("target_loss", "control_loss", "target_specific_loss")
    )
    identity_qualified = identity_max <= float(
        config["calibration_and_preflight"][
            "maximum_identity_absolute_margin_difference"
        ]
    )
    clean_qualified = clean["mean"] >= 0.5 and clean["ci95"][0] > 0
    primary = _primary_localization(
        tile_rows,
        clusters,
        config,
        split=args.split,
        development=development,
        seed=_split_seed(config, args.split) ^ 0x2020,
    )

    null_rows = []
    value_tests = (
        [
            test
            for test in primary["tests"].values()
            if test["operation"] == "candidate_value_swap" and test["qualified"]
        ]
        if development is None
        else [
            test
            for test in primary["tests"].values()
            if test["operation"] == "candidate_value_swap"
        ]
    )
    if value_tests:
        candidate = ((11, 2), (11, 5))
        null_pairs = layer_pair_nulls(
            candidate, num_heads=int(config["checkpoint"]["heads_per_layer"])
        )
        named_pairs = {
            "candidate_L11_pair": candidate,
            **{
                f"null_{index:02d}_{head_label(pair[0])}_{head_label(pair[1])}": pair
                for index, pair in enumerate(null_pairs, start=1)
            },
        }
        for test in value_tests:
            null_rows.extend(
                scan_head_pair_regional_values(
                    bundle,
                    pairs,
                    tile={
                        "tile_index": int(test["tile_index"]),
                        "nominal_start": int(test["nominal_start"]),
                        "nominal_end": int(test["nominal_end"]),
                    },
                    head_pairs=named_pairs,
                    token_dose=int(geometry["fixed_BPE_token_dose_per_sequence"]),
                    maximum_cover_nt=int(geometry["maximum_actual_cover_nt"]),
                    maximum_cover_difference_nt=int(
                        geometry["maximum_target_control_cover_length_difference_nt"]
                    ),
                    batch_pairs=args.batch_pairs,
                    progress=progress,
                )
            )
        _apply_value_pair_specificity(primary, null_rows, config)

    nominated_tiles = (
        sorted(
            {
                int(test["tile_index"])
                for test in primary["tests"].values()
                if test["qualified"]
            }
        )
        if development is None
        else list(requested_tiles)
    )
    biological_regions = _biological_region_qualification(
        tile_rows,
        row_by_id,
        all_rows,
        primary,
        clusters,
        config,
        split=args.split,
        development=development,
        seed=_split_seed(config, args.split) ^ 0x3030,
    )
    motif_names = (
        (
            "TATA",
            "Inr",
            "CCAAT",
            "GC_box",
        )
        if development is None
        else tuple(
            name
            for name, test in development["motif_disruption"]["tests"].items()
            if test["qualified"]
        )
    )
    motif_rows = (
        score_motif_disruptions(
            bundle,
            rows,
            motifs=motif_names,
            nominated_tile_indices=nominated_tiles,
            batch_pairs=args.batch_pairs,
        )
        if nominated_tiles and motif_names
        else []
    )
    motif_summary = _motif_summary(
        motif_rows,
        all_rows,
        nominated_tiles,
        clusters,
        config,
        split=args.split,
        development=development,
        seed=_split_seed(config, args.split) ^ 0x4040,
    )
    qualified_tiles = [test for test in primary["tests"].values() if test["qualified"]]
    qualified_for_confirmation = (
        args.split == "development"
        and identity_qualified
        and clean_qualified
        and bool(qualified_tiles)
    )
    replicated_tiles = (
        [
            test["test_id"]
            for test in qualified_tiles
            if development is not None
            and development["primary_localization"]["tests"][test["test_id"]][
                "qualified"
            ]
        ]
        if development is not None
        else []
    )

    raw_rows = [
        *({"row_type": "tile_effect", **row} for row in tile_rows),
        *({"row_type": "regional_head_null", **row} for row in null_rows),
        *({"row_type": "motif_disruption", **row} for row in motif_rows),
    ]
    input_paths = [
        args.annotated_cohort,
        args.cohort_manifest,
        args.supply_audit,
        args.calibration,
        args.config,
        args.preflight_report,
        args.checkpoint_audit,
        args.source_manifest,
    ]
    if args.development_report:
        input_paths.append(args.development_report)
    status = "DEVELOPMENT" if args.split == "development" else "SEALED"
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id=f"dnabert2_promoter_source_localization_{args.split}_raw_v1",
        status=status,
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "split": args.split,
            "config_sha256": config_hash,
            "requested_tiles": requested_tiles,
        },
        seeds={"inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_source_localization_v1",
        "split": args.split,
        "status": status,
        "config_sha256": config_hash,
        "n_pairs": len(pairs),
        "n_unique_locus_blocks": len(set(clusters.values())),
        "selected_id_sha256": hashlib.sha256(
            ("\n".join(pair_id for pair_id, _, _ in pairs) + "\n").encode()
        ).hexdigest(),
        "clean_behavior": {**clean, "qualified": clean_qualified},
        "identity_control": {
            "maximum_absolute_margin_difference": identity_max,
            "qualified": identity_qualified,
        },
        "primary_localization": primary,
        "biological_region_qualification": biological_regions,
        "motif_disruption": motif_summary,
        "qualified_for_confirmation": qualified_for_confirmation,
        "replicated_tiles": replicated_tiles,
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
        experiment_id=f"dnabert2_promoter_source_localization_{args.split}_v1",
        status=status,
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={
            "split": args.split,
            "config_sha256": config_hash,
            "batch_pairs": args.batch_pairs,
        },
        seeds={"inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "clean_behavior": payload["clean_behavior"],
                "identity_control": payload["identity_control"],
                "primary_localization": primary,
                "biological_region_qualification": biological_regions,
                "motif_disruption": motif_summary,
                "qualified_for_confirmation": qualified_for_confirmation,
                "replicated_tiles": replicated_tiles,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _select_rows(
    all_rows: Sequence[Mapping[str, Any]],
    supply: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    split: str,
) -> list[dict[str, Any]]:
    by_id = {
        str(row["example_id"]): dict(row) for row in all_rows if row["split"] == split
    }
    ids = [str(value) for value in supply["selection"]["selected_example_ids"][split]]
    rows = [by_id[value] for value in ids]
    digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    if (
        len(rows) != int(config["cohort"]["pairs_per_split"])
        or digest != config["cohort"]["selected_id_sha256"][split]
        or digest != supply["selection"]["selected_id_sha256"][split]
    ):
        raise RuntimeError("annotated cohort selection disagrees with frozen supply")
    return rows


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
        "ci95": cluster_bootstrap_interval(values, order, draws=draws, seed=seed),
    }


def _primary_localization(
    rows: Sequence[Mapping[str, Any]],
    clusters: Mapping[str, str],
    config: Mapping[str, Any],
    *,
    split: str,
    development: Mapping[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    primary_rows = [row for row in rows if str(row["operation"]) in PRIMARY_OPERATIONS]
    by_operation_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in primary_rows:
        by_operation_pair[(str(row["operation"]), str(row["pair_id"]))].append(
            float(row["target_specific_loss"])
        )
    contrast_rows = []
    if split == "development":
        for row in primary_rows:
            values = by_operation_pair[(str(row["operation"]), str(row["pair_id"]))]
            if len(values) < 2:
                raise RuntimeError("localization contrast requires at least two tiles")
            raw = float(row["target_specific_loss"])
            other_mean = (sum(values) - raw) / (len(values) - 1)
            contrast_rows.append({**row, "localization_contrast": raw - other_mean})
    else:
        contrast_rows = [{**row, "localization_contrast": None} for row in primary_rows]
    tests: dict[str, dict[str, Any]] = {}
    for operation in PRIMARY_OPERATIONS:
        for tile in sorted(
            {
                int(row["tile_index"])
                for row in primary_rows
                if row["operation"] == operation
            }
        ):
            group = [
                row
                for row in contrast_rows
                if row["operation"] == operation and int(row["tile_index"]) == tile
            ]
            values = [float(row["target_specific_loss"]) for row in group]
            order = [clusters[str(row["pair_id"])] for row in group]
            test_id = f"{operation}|tile{tile:02d}"
            raw_sesoi = float(
                config["primary_gates"][
                    (
                        "mask_absolute_SESOI_logit_margin"
                        if operation == "mask"
                        else "candidate_value_absolute_SESOI_logit_margin"
                    )
                ]
            )
            contrast_sesoi = float(
                config["primary_gates"][
                    (
                        "mask_localization_contrast_SESOI_logit_margin"
                        if operation == "mask"
                        else "candidate_value_localization_contrast_SESOI_logit_margin"
                    )
                ]
            )
            test = {
                "test_id": test_id,
                "operation": operation,
                "tile_index": tile,
                "nominal_start": int(group[0]["nominal_start"]),
                "nominal_end": int(group[0]["nominal_end"]),
                "relative_start": int(group[0]["relative_start"]),
                "relative_end": int(group[0]["relative_end"]),
                "n_pairs": len(group),
                "target_loss_mean": statistics.mean(
                    float(row["target_loss"]) for row in group
                ),
                "control_loss_mean": statistics.mean(
                    float(row["control_loss"]) for row in group
                ),
                "target_specific_loss_mean": statistics.mean(values),
                "target_specific_loss_ci95": cluster_bootstrap_interval(
                    values,
                    order,
                    draws=int(config["primary_gates"]["bootstrap_draws"]),
                    seed=seed + tile + (0 if operation == "mask" else 100),
                ),
                "raw_SESOI": raw_sesoi,
                "localization_contrast_SESOI": contrast_sesoi,
                "regional_pair_specificity": None,
            }
            if split == "development":
                contrasts = [float(row["localization_contrast"]) for row in group]
                test.update(
                    {
                        "localization_contrast_mean": statistics.mean(contrasts),
                        "localization_contrast_ci95": cluster_bootstrap_interval(
                            contrasts,
                            order,
                            draws=int(config["primary_gates"]["bootstrap_draws"]),
                            seed=seed + tile + (1000 if operation == "mask" else 2000),
                        ),
                    }
                )
            else:
                test.update(
                    {
                        "localization_contrast_mean": development[
                            "primary_localization"
                        ]["tests"][test_id]["localization_contrast_mean"],
                        "localization_contrast_ci95": development[
                            "primary_localization"
                        ]["tests"][test_id]["localization_contrast_ci95"],
                        "localization_contrast_status": (
                            "frozen development localization evidence; "
                            "not recomputed in sealed confirmation"
                        ),
                    }
                )
            tests[test_id] = test
    if split == "development":
        max_p = _max_sign_flip_pvalues(
            {
                test_id: [
                    (
                        str(row["pair_id"]),
                        float(row["localization_contrast"]),
                    )
                    for row in contrast_rows
                    if row["operation"] == test["operation"]
                    and int(row["tile_index"]) == test["tile_index"]
                ]
                for test_id, test in tests.items()
            },
            draws=int(config["inference"]["sign_flip_draws"]),
            seed=seed ^ 0xABCD,
        )
    else:
        max_p = _max_sign_flip_pvalues(
            {
                test_id: [
                    (str(row["pair_id"]), float(row["target_specific_loss"]))
                    for row in primary_rows
                    if row["operation"] == test["operation"]
                    and int(row["tile_index"]) == test["tile_index"]
                ]
                for test_id, test in tests.items()
            },
            draws=int(config["inference"]["sign_flip_draws"]),
            seed=seed ^ 0xBCDE,
        )
    for test_id, test in tests.items():
        raw_ci = test["target_specific_loss_ci95"]
        if split == "development":
            contrast_gate = (
                test["localization_contrast_mean"]
                >= test["localization_contrast_SESOI"]
                and test["localization_contrast_ci95"][0] > 0
            )
        else:
            development_test = development["primary_localization"]["tests"][test_id]
            contrast_gate = bool(development_test["qualified"])
        sign_preserved = (
            development is None
            or test["target_specific_loss_mean"]
            * float(
                development["primary_localization"]["tests"][test_id][
                    "target_specific_loss_mean"
                ]
            )
            > 0
        )
        test["familywise_max_sign_flip_p"] = max_p[test_id]
        test["confirmation_sign_preserved"] = sign_preserved
        test["qualified_before_pair_specificity"] = (
            test["n_pairs"]
            >= int(config["tile_geometry"]["minimum_eligible_pairs_per_target_tile"])
            and test["target_specific_loss_mean"] >= test["raw_SESOI"]
            and raw_ci[0] > 0
            and contrast_gate
            and max_p[test_id] <= float(config["primary_gates"]["familywise_alpha"])
            and sign_preserved
        )
        test["qualified"] = test["qualified_before_pair_specificity"]
    return {
        "measurement": config["measurement"],
        "family": (
            "all 30 tiles x mask/value"
            if split == "development"
            else "unchanged development-nominated tiles x mask/value"
        ),
        "tests": tests,
        "qualified_test_ids": [
            test_id for test_id, test in tests.items() if test["qualified"]
        ],
    }


def _apply_value_pair_specificity(
    primary: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    by_tile_pair: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        by_tile_pair[(int(row["tile_index"]), str(row["head_pair"]))].append(
            float(row["target_specific_loss"])
        )
    raw_p = {}
    summaries = {}
    for test_id, test in primary["tests"].items():
        if test["operation"] != "candidate_value_swap" or not test["qualified"]:
            continue
        tile = int(test["tile_index"])
        candidate = statistics.mean(by_tile_pair[(tile, "candidate_L11_pair")])
        null_means = [
            statistics.mean(values)
            for (null_tile, pair_name), values in sorted(by_tile_pair.items())
            if null_tile == tile and pair_name != "candidate_L11_pair"
        ]
        raw_p[test_id] = add_one_permutation_pvalue(
            abs(candidate), [abs(value) for value in null_means]
        )
        summaries[test_id] = {
            "candidate_mean": candidate,
            "null_count": len(null_means),
            "null_maximum_absolute_mean": max(abs(value) for value in null_means),
            "candidate_exceeds_all_null_magnitudes": abs(candidate)
            > max(abs(value) for value in null_means),
            "raw_add_one_magnitude_p": raw_p[test_id],
        }
    adjusted = holm_adjust(raw_p) if raw_p else {}
    for test_id, summary in summaries.items():
        summary["holm_adjusted_p"] = adjusted[test_id]
        summary["qualified"] = summary[
            "candidate_exceeds_all_null_magnitudes"
        ] and adjusted[test_id] <= float(config["primary_gates"]["familywise_alpha"])
        primary["tests"][test_id]["regional_pair_specificity"] = summary
        primary["tests"][test_id]["qualified"] = (
            primary["tests"][test_id]["qualified"] and summary["qualified"]
        )
    primary["qualified_test_ids"] = [
        test_id for test_id, test in primary["tests"].items() if test["qualified"]
    ]


def _biological_region_qualification(
    tile_rows: Sequence[Mapping[str, Any]],
    row_by_id: Mapping[str, Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Any],
    clusters: Mapping[str, str],
    config: Mapping[str, Any],
    *,
    split: str,
    development: Mapping[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    minimum = int(
        config["biological_qualification"][
            "minimum_annotated_and_unannotated_pairs_per_split"
        ]
    )
    candidates: dict[str, dict[str, Any]] = {}
    value_rows = [row for row in tile_rows if row["operation"] in PRIMARY_OPERATIONS]
    for test_id, test in primary["tests"].items():
        if (
            split == "development"
            and not test["qualified"]
            or development is not None
            and not development["primary_localization"]["tests"][test_id]["qualified"]
        ):
            continue
        tile = int(test["tile_index"])
        operation = str(test["operation"])
        group = [
            row
            for row in value_rows
            if row["operation"] == operation and int(row["tile_index"]) == tile
        ]
        for raw_feature, label in ANNOTATIONS.items():
            annotated = [
                row
                for row in group
                if row_by_id[str(row["pair_id"])]["localization"]["tiles"][tile][
                    "annotation_overlap"
                ][raw_feature]
            ]
            unannotated = [row for row in group if row not in annotated]
            supply_counts = _annotation_supply_counts(
                all_rows,
                split="confirmation",
                tile=tile,
                feature=raw_feature,
            )
            combo_id = f"{test_id}|{label}"
            powered = (
                len(annotated) >= minimum
                and len(unannotated) >= minimum
                and supply_counts["annotated"] >= minimum
                and supply_counts["unannotated"] >= minimum
            )
            if development is not None:
                dev_combo = development["biological_region_qualification"]["tests"].get(
                    combo_id
                )
                powered = bool(dev_combo and dev_combo["qualified"])
            annotated_values = [float(row["target_specific_loss"]) for row in annotated]
            unannotated_values = [
                float(row["target_specific_loss"]) for row in unannotated
            ]
            candidates[combo_id] = {
                "test_id": combo_id,
                "primary_test_id": test_id,
                "operation": operation,
                "tile_index": tile,
                "annotation": label,
                "development_and_confirmation_supply_powered": powered,
                "n_annotated": len(annotated),
                "n_unannotated": len(unannotated),
                "annotated_mean": (
                    statistics.mean(annotated_values) if annotated_values else None
                ),
                "annotated_ci95": (
                    cluster_bootstrap_interval(
                        annotated_values,
                        [clusters[str(row["pair_id"])] for row in annotated],
                        draws=int(config["inference"]["bootstrap_draws"]),
                        seed=seed + len(candidates),
                    )
                    if annotated_values
                    else None
                ),
                "unannotated_mean": (
                    statistics.mean(unannotated_values) if unannotated_values else None
                ),
                "unannotated_ci95": (
                    cluster_bootstrap_interval(
                        unannotated_values,
                        [clusters[str(row["pair_id"])] for row in unannotated],
                        draws=int(config["inference"]["bootstrap_draws"]),
                        seed=seed + 1000 + len(candidates),
                    )
                    if unannotated_values
                    else None
                ),
                "heterogeneity_annotated_minus_unannotated": (
                    statistics.mean(annotated_values)
                    - statistics.mean(unannotated_values)
                    if annotated_values and unannotated_values
                    else None
                ),
                "_annotated_rows": annotated,
            }
    powered_family = {
        combo_id: [
            (str(row["pair_id"]), float(row["target_specific_loss"]))
            for row in test["_annotated_rows"]
        ]
        for combo_id, test in candidates.items()
        if test["development_and_confirmation_supply_powered"]
    }
    pvalues = (
        _max_sign_flip_pvalues(
            powered_family,
            draws=int(config["inference"]["sign_flip_draws"]),
            seed=seed ^ 0xCA6E,
        )
        if powered_family
        else {}
    )
    for combo_id, test in candidates.items():
        test["familywise_max_sign_flip_p"] = pvalues.get(combo_id)
        raw_sesoi = float(
            config["primary_gates"][
                (
                    "mask_absolute_SESOI_logit_margin"
                    if test["operation"] == "mask"
                    else "candidate_value_absolute_SESOI_logit_margin"
                )
            ]
        )
        test["qualified"] = (
            test["development_and_confirmation_supply_powered"]
            and test["annotated_mean"] is not None
            and test["annotated_mean"] >= raw_sesoi
            and test["annotated_ci95"][0] > 0
            and test["familywise_max_sign_flip_p"]
            <= float(config["primary_gates"]["familywise_alpha"])
        )
        test.pop("_annotated_rows")
    return {
        "tests": candidates,
        "qualified_test_ids": [
            combo_id for combo_id, test in candidates.items() if test["qualified"]
        ],
        "boundary": (
            "Qualification means the nominated regional effect persists within "
            "independently annotated tile-locus intersections. The unannotated "
            "estimate is a heterogeneity diagnostic; overlap is not specificity."
        ),
    }


def _annotation_supply_counts(
    all_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    tile: int,
    feature: str,
) -> dict[str, int]:
    eligible = [
        row
        for row in all_rows
        if row["split"] == split and row["localization"]["tiles"][tile]["eligible"]
    ]
    annotated = sum(
        bool(row["localization"]["tiles"][tile]["annotation_overlap"][feature])
        for row in eligible
    )
    return {"annotated": annotated, "unannotated": len(eligible) - annotated}


def _motif_summary(
    rows: Sequence[Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
    nominated_tiles: Sequence[int],
    clusters: Mapping[str, str],
    config: Mapping[str, Any],
    *,
    split: str,
    development: Mapping[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    motifs = ("TATA", "Inr", "CCAAT", "GC_box")
    minimum = int(
        config["biological_qualification"]["motifs"][
            "minimum_geometry_qualified_pairs_per_split"
        ]
    )
    tests: dict[str, dict[str, Any]] = {}
    raw_p = {}
    for index, motif in enumerate(motifs):
        group = [row for row in rows if row["motif"] == motif]
        values = [float(row["target_specific_loss"]) for row in group]
        confirmation_supply = _motif_supply(
            all_rows,
            split="confirmation",
            motif=motif,
            nominated_tiles=nominated_tiles,
        )
        powered = len(values) >= minimum and confirmation_supply >= minimum
        if development is not None:
            powered = bool(
                development["motif_disruption"]["tests"].get(motif, {}).get("qualified")
            )
        test = {
            "motif": motif,
            "family": (
                "core_promoter" if motif in {"TATA", "Inr"} else "promoter_associated"
            ),
            "n_pairs": len(values),
            "confirmation_supply_for_same_nominated_tiles": confirmation_supply,
            "powered": powered,
            "target_loss_mean": (
                statistics.mean(float(row["target_loss"]) for row in group)
                if group
                else None
            ),
            "control_loss_mean": (
                statistics.mean(float(row["control_loss"]) for row in group)
                if group
                else None
            ),
            "target_specific_loss_mean": (statistics.mean(values) if values else None),
            "target_specific_loss_ci95": (
                cluster_bootstrap_interval(
                    values,
                    [clusters[str(row["pair_id"])] for row in group],
                    draws=int(config["inference"]["bootstrap_draws"]),
                    seed=seed + index,
                )
                if values
                else None
            ),
            "raw_add_one_sign_flip_p": None,
            "holm_adjusted_p": None,
            "qualified": False,
        }
        if powered and values:
            raw_p[motif] = _one_sample_sign_flip_p(
                values,
                draws=int(config["inference"]["sign_flip_draws"]),
                seed=seed + 100 + index,
            )
            test["raw_add_one_sign_flip_p"] = raw_p[motif]
        tests[motif] = test
    adjusted = holm_adjust(raw_p) if raw_p else {}
    for motif, pvalue in adjusted.items():
        test = tests[motif]
        test["holm_adjusted_p"] = pvalue
        sign_preserved = (
            development is None
            or test["target_specific_loss_mean"]
            * float(
                development["motif_disruption"]["tests"][motif][
                    "target_specific_loss_mean"
                ]
            )
            > 0
        )
        test["confirmation_sign_preserved"] = sign_preserved
        test["qualified"] = (
            test["target_specific_loss_mean"]
            >= float(
                config["biological_qualification"]["motifs"][
                    "target_specific_SESOI_logit_margin"
                ]
            )
            and test["target_specific_loss_ci95"][0] > 0
            and pvalue <= float(config["primary_gates"]["familywise_alpha"])
            and sign_preserved
        )
    return {
        "nominated_tile_indices": list(nominated_tiles),
        "tests": tests,
        "qualified_motifs": [
            motif for motif, test in tests.items() if test["qualified"]
        ],
        "boundary": (
            "The paired composition-preserving shuffle is classifier-level motif "
            "necessity evidence. It is not evidence of L11H2/L11H5 mediation."
        ),
    }


def _motif_supply(
    all_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    motif: str,
    nominated_tiles: Sequence[int],
) -> int:
    nominated = {int(value) for value in nominated_tiles}
    return sum(
        any(
            disruption["motif"] == motif
            and any(
                tile["tile_index"] in nominated
                and disruption["start"] < tile["nominal_end"]
                and tile["nominal_start"] < disruption["end"]
                for tile in row["localization"]["tiles"]
            )
            for disruption in row["localization"]["motif_disruptions"]
        )
        for row in all_rows
        if row["split"] == split
    )


def _max_sign_flip_pvalues(
    family: Mapping[str, Sequence[tuple[str, float]]],
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    if not family:
        return {}
    ids = sorted({pair_id for values in family.values() for pair_id, _ in values})
    id_index = {pair_id: index for index, pair_id in enumerate(ids)}
    names = sorted(family)
    matrix = np.zeros((len(ids), len(names)), dtype=np.float64)
    counts = np.zeros(len(names), dtype=np.float64)
    observed = np.zeros(len(names), dtype=np.float64)
    for column, name in enumerate(names):
        values = list(family[name])
        counts[column] = len(values)
        observed[column] = statistics.mean(value for _, value in values)
        for pair_id, value in values:
            matrix[id_index[pair_id], column] = value
    rng = np.random.default_rng(seed)
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(draws, len(ids)),
        replace=True,
    )
    null_means = (signs @ matrix) / counts
    null_max = null_means.max(axis=1)
    return {
        name: float((1 + np.sum(null_max >= observed[column])) / (draws + 1))
        for column, name in enumerate(names)
    }


def _one_sample_sign_flip_p(values: Sequence[float], *, draws: int, seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    observed = float(array.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(array)), replace=True)
    null = (signs * array).mean(axis=1)
    return float((1 + np.sum(null >= observed)) / (draws + 1))


def _split_seed(config: Mapping[str, Any], split: str) -> int:
    return int(config["inference"]["seed"]) ^ (
        0xD8D8 if split == "development" else 0xC8C8
    )


if __name__ == "__main__":
    main()
