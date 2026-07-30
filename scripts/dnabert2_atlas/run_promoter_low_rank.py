#!/usr/bin/env python
"""Fit or transfer a frozen low-rank L11MLP promoter subspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import load_dnabert2_promoter_classifier
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_head_scan import read_jsonl
from gfmcd.reconstruction_atlas.promoter_low_rank import (
    capture_mlp_writes_for_pairs,
    decode_float32_basis,
    encode_float32_basis,
    fit_uncentered_difference_basis,
    paired_mlp_projection_rows,
    random_orthonormal_basis,
    random_subspace_add_one_pvalue,
    select_low_rank_candidate,
    summarize_projection_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--development-report")
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=50)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    supply = json.loads(Path(args.supply_audit).read_text())
    preflight = json.loads(Path(args.preflight_report).read_text())
    _validate_instrument(config, config_hash, supply, preflight)

    development = None
    if args.split == "confirmation":
        if not args.development_report:
            raise RuntimeError("--development-report is required for confirmation")
        development = json.loads(Path(args.development_report).read_text())
        if (
            development.get("split") != "development"
            or development.get("config_sha256") != config_hash
            or development.get("preflight_sha256") != sha256_file(args.preflight_report)
            or development.get("confirmation_gate") != "confirmation_may_open"
        ):
            raise RuntimeError(
                "confirmation requires unchanged qualified development selection"
            )

    cohort = read_jsonl(args.cohort)
    selected = [row for row in cohort if row.get("split") == args.split]
    expected_ids = list(supply["selection"]["selected_example_ids"][args.split])
    if (
        len(selected) != int(config["cohort"]["pairs_per_split"])
        or [str(row["example_id"]) for row in selected] != expected_ids
        or _id_hash(expected_ids)
        != str(supply["selection"]["selected_id_sha256"][args.split])
    ):
        raise RuntimeError("cohort split does not match the frozen D10 supply")
    pairs = [
        (
            str(row["example_id"]),
            str(row["target"]["clean_sequence"]),
            str(row["control"]["clean_sequence"]),
        )
        for row in selected
    ]

    bundle = load_dnabert2_promoter_classifier(
        str(config["checkpoint"]["candidate"]), device=args.device
    )
    layer = int(config["scope"]["layer"])
    basis_config = config["basis"]
    inference = config["inference"]
    seed = int(inference["seed"]) ^ (0x10A0 if args.split == "development" else 0x10A1)
    basis_record = None
    basis_diagnostics = None
    if args.split == "development":
        _pair_ids, _margins, writes = capture_mlp_writes_for_pairs(
            bundle,
            pairs,
            layer=layer,
            batch_pairs=args.batch_pairs,
        )
        basis, basis_diagnostics = fit_uncentered_difference_basis(
            writes,
            maximum_rank=int(basis_config["maximum_rank"]),
        )
        basis_record = encode_float32_basis(basis)
        rank_grid = [int(rank) for rank in basis_config["rank_grid"]]
    else:
        basis_record = dict(development["basis"])  # type: ignore[index]
        basis = decode_float32_basis(basis_record)
        rank_grid = [int(development["selection"]["selected_rank"])]  # type: ignore[index]

    raw_rows: list[dict[str, Any]] = []
    full_rows = paired_mlp_projection_rows(
        bundle,
        pairs,
        layer=layer,
        batch_pairs=args.batch_pairs,
        direction="bidirectional_swap",
        basis=None,
    )
    raw_rows.extend(_annotate(full_rows, condition="full_L11MLP"))
    full_summary = _summarize(full_rows, inference, seed ^ 0xF011)

    rank_summaries: dict[int, dict[str, Any]] = {}
    rank_rows: dict[int, list[dict[str, Any]]] = {}
    for rank in rank_grid:
        rows = paired_mlp_projection_rows(
            bundle,
            pairs,
            layer=layer,
            batch_pairs=args.batch_pairs,
            direction="bidirectional_swap",
            basis=basis[:, :rank],
        )
        rank_rows[rank] = rows
        rank_summaries[rank] = _summarize(rows, inference, seed ^ rank)
        raw_rows.extend(_annotate(rows, condition="rank_subspace", rank=rank))

    if args.split == "development":
        selected_rank, rank_checks = select_low_rank_candidate(
            rank_summaries,
            full_summary=full_summary,
            maximum_rank=int(basis_config["maximum_rank"]),
            minimum_class_margin_loss=float(inference["class_margin_loss_SESOI"]),
            minimum_capture_fraction=float(
                inference["minimum_full_effect_capture_fraction"]
            ),
            minimum_accuracy_loss=float(inference["accuracy_loss_SESOI"]),
        )
    else:
        selected_rank = rank_grid[0]
        capture_fraction = (
            float(rank_summaries[selected_rank]["class_margin_loss"])
            / float(full_summary["class_margin_loss"])
            if float(full_summary["class_margin_loss"]) > 0
            else float("-inf")
        )
        rank_checks = {
            selected_rank: {
                "capture_fraction": capture_fraction,
                "checks": _confirmation_effect_checks(
                    rank_summaries[selected_rank],
                    full_summary,
                    inference,
                ),
            }
        }

    selected_summary = (
        rank_summaries[selected_rank] if selected_rank is not None else None
    )
    complement_summary = None
    directional = None
    null_summaries: list[dict[str, Any]] = []
    null_p = None
    if selected_rank is not None:
        selected_basis = basis[:, :selected_rank]
        complement_rows = paired_mlp_projection_rows(
            bundle,
            pairs,
            layer=layer,
            batch_pairs=args.batch_pairs,
            direction="bidirectional_swap",
            basis=selected_basis,
            component="complement",
        )
        complement_summary = _summarize(complement_rows, inference, seed ^ 0xC0A1)
        raw_rows.extend(
            _annotate(
                complement_rows,
                condition="orthogonal_complement",
                rank=selected_rank,
            )
        )
        directional = {}
        for direction, label in (
            ("target_from_control", "necessity"),
            ("control_from_target", "sufficiency"),
        ):
            rows = paired_mlp_projection_rows(
                bundle,
                pairs,
                layer=layer,
                batch_pairs=args.batch_pairs,
                direction=direction,
                basis=selected_basis,
            )
            directional[label] = _summarize(
                rows,
                inference,
                seed ^ (0xA11 if label == "necessity" else 0x5AFF),
            )
            raw_rows.extend(
                _annotate(
                    rows,
                    condition=label,
                    rank=selected_rank,
                )
            )
        for null_index in range(int(inference["random_subspaces"])):
            null_basis = random_orthonormal_basis(
                hidden_size=int(config["scope"]["hidden_size"]),
                rank=selected_rank,
                seed=int(inference["seed"]) ^ (0xBAD5 + 104729 * null_index),
            )
            rows = paired_mlp_projection_rows(
                bundle,
                pairs,
                layer=layer,
                batch_pairs=args.batch_pairs,
                direction="bidirectional_swap",
                basis=null_basis,
            )
            summary = _summarize(
                rows,
                inference,
                seed ^ (0x9000 + null_index),
            )
            summary["null_index"] = null_index
            null_summaries.append(summary)
            raw_rows.extend(
                _annotate(
                    rows,
                    condition="random_subspace",
                    rank=selected_rank,
                    null_index=null_index,
                )
            )
        null_p = random_subspace_add_one_pvalue(
            selected_summary,
            null_summaries,
        )

    if selected_rank is None:
        final_checks = {"development_rank_selected": False}
    else:
        effect_checks = (
            rank_checks[selected_rank]["checks"]
            if args.split == "development"
            else _confirmation_effect_checks(
                selected_summary,
                full_summary,
                inference,
            )
        )
        final_checks = {
            **effect_checks,
            "beats_dimension_matched_random_subspaces": float(null_p)
            <= float(inference["random_subspace_add_one_alpha"]),
            "full_L11MLP_effect_positive": float(
                full_summary["class_margin_loss_CI95"][0]
            )
            > 0,
        }
    qualified = selected_rank is not None and all(final_checks.values())
    confirmation_gate = (
        "confirmation_may_open"
        if args.split == "development" and qualified
        else (
            "opened_from_development"
            if args.split == "confirmation"
            else "confirmation_remains_sealed"
        )
    )
    status = "DEVELOPMENT" if args.split == "development" else "SEALED"
    input_paths = tuple(
        input_path
        for input_path in (
            args.cohort,
            args.supply_audit,
            args.checkpoint_audit,
            args.config,
            args.preflight_report,
            args.development_report,
        )
        if input_path is not None
    )
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id=f"dnabert2_promoter_low_rank_{args.split}_raw_v1",
        status=status,
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "split": args.split,
            "config_sha256": config_hash,
            "selected_rank": selected_rank,
            "basis_sha256": basis_record["sha256"],
        },
        seeds={"basis_and_inference": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_L11MLP_low_rank_v1",
        "split": args.split,
        "status": status,
        "evidence_status": "POST_HOC_COHORT_REUSE",
        "config_sha256": config_hash,
        "preflight_sha256": sha256_file(args.preflight_report),
        "n_pairs": len(pairs),
        "scope": config["scope"],
        "basis": basis_record,
        "basis_diagnostics": basis_diagnostics,
        "full_L11MLP": full_summary,
        "rank_summaries": {
            str(rank): rank_summaries[rank] for rank in sorted(rank_summaries)
        },
        "rank_checks": {str(rank): rank_checks[rank] for rank in sorted(rank_checks)},
        "selection": {
            "selected_rank": selected_rank,
            "source": (
                "frozen_development_rank_rule"
                if args.split == "development"
                else "unchanged_development_rank_and_basis"
            ),
            "qualified": qualified,
            "checks": final_checks,
            "random_subspace_p_add_one": null_p,
        },
        "selected_subspace": selected_summary,
        "orthogonal_complement": complement_summary,
        "directional": directional,
        "random_subspaces": null_summaries,
        "confirmation_gate": confirmation_gate,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "rows": len(raw_rows),
        },
        "boundary": config["scope"]["boundary"],
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=f"dnabert2_promoter_low_rank_{args.split}_v1",
        status=status,
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={
            "split": args.split,
            "config_sha256": config_hash,
            "basis": basis_config,
            "inference": inference,
            "batch_pairs": args.batch_pairs,
        },
        seeds={"basis_and_inference": seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "selected_rank": selected_rank,
                "qualified": qualified,
                "checks": final_checks,
                "full_L11MLP": full_summary,
                "selected_subspace": selected_summary,
                "random_subspace_p_add_one": null_p,
                "orthogonal_complement": complement_summary,
                "directional": directional,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _validate_instrument(
    config: dict[str, Any],
    config_hash: str,
    supply: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    if config.get("status") != "FROZEN":
        raise RuntimeError("low-rank target run requires a frozen config")
    if supply.get("status") != "QUALIFIED" or not all(
        supply.get("checks", {}).values()
    ):
        raise RuntimeError("qualified D10 supply is required")
    if (
        preflight.get("status") != "QUALIFIED"
        or not preflight.get("qualified")
        or not all(preflight.get("checks", {}).values())
        or preflight.get("target_cohort_outcomes_computed") is not False
        or preflight.get("config_sha256") != config_hash
        or preflight.get("checkpoint_revision") != config["checkpoint"]["revision"]
    ):
        raise RuntimeError("qualified low-rank preflight is required")


def _summarize(
    rows: list[dict[str, Any]],
    inference: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return summarize_projection_rows(
        rows,
        bootstrap_draws=int(inference["bootstrap_draws"]),
        sign_permutations=int(inference["sign_permutations"]),
        seed=seed,
    )


def _confirmation_effect_checks(
    summary: dict[str, Any],
    full_summary: dict[str, Any],
    inference: dict[str, Any],
) -> dict[str, bool]:
    full_effect = float(full_summary["class_margin_loss"])
    capture_fraction = (
        float(summary["class_margin_loss"]) / full_effect
        if full_effect > 0
        else float("-inf")
    )
    return {
        "class_margin_loss_at_least_SESOI": float(summary["class_margin_loss"])
        >= float(inference["class_margin_loss_SESOI"]),
        "capture_fraction_at_least_threshold": capture_fraction
        >= float(inference["minimum_full_effect_capture_fraction"]),
        "accuracy_loss_at_least_SESOI": float(summary["accuracy_loss"])
        >= float(inference["accuracy_loss_SESOI"]),
        "positive_class_margin_CI": float(summary["class_margin_loss_CI95"][0]) > 0,
    }


def _annotate(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    rank: int | None = None,
    null_index: int | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "condition": condition,
            "rank": rank,
            "null_index": null_index,
            **row,
        }
        for row in rows
    ]


def _id_hash(example_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(example_ids) + "\n").encode()).hexdigest()


if __name__ == "__main__":
    main()
