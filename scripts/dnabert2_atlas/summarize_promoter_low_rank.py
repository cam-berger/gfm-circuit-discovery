#!/usr/bin/env python
"""Combine the post-hoc development and unchanged-basis low-rank results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    development = json.loads(Path(args.development).read_text())
    confirmation = json.loads(Path(args.confirmation).read_text())
    config = json.loads(Path(args.config).read_text())
    preflight = json.loads(Path(args.preflight_report).read_text())
    _validate_inputs(
        development,
        confirmation,
        config,
        preflight,
        config_hash=frozen_config_sha256(args.config),
    )

    rank = int(development["selection"]["selected_rank"])
    hidden_size = int(config["scope"]["hidden_size"])
    split_summaries = {
        split: _split_summary(result)
        for split, result in (
            ("development", development),
            ("confirmation", confirmation),
        )
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_L11MLP_low_rank_replication_v1",
        "status": "SEALED",
        "evidence_status": "SUPPORTED_POST_HOC_COHORT_REUSE",
        "config_sha256": development["config_sha256"],
        "preflight_sha256": development["preflight_sha256"],
        "node": config["scope"]["node"],
        "selected_rank": rank,
        "hidden_size": hidden_size,
        "dimension_fraction": rank / hidden_size,
        "development_basis_sha256": development["basis"]["sha256"],
        "development_basis_transferred_unchanged": True,
        "development_selection_qualified": development["selection"]["qualified"],
        "confirmation_transfer_qualified": confirmation["selection"]["qualified"],
        "splits": split_summaries,
        "licensed_claim": (
            "A four-dimensional development-fitted subspace is the dominant causal "
            "carrier of the paired L11MLP CLS-write exchange effect on promoter "
            "class margin and transfers unchanged to the disjoint confirmation "
            "split against dimension-matched random subspaces."
        ),
        "boundary": [
            config["scope"]["boundary"],
            "The orthogonal complement retains a positive class-margin effect in both splits, so the rank-4 subspace is not a complete account of L11MLP.",
            "The selected-subspace hard-label accuracy interval includes zero in both splits; the class-margin endpoint provides the clearest evidence.",
            "SVD energy and mean-difference capture are descriptive; causal support comes from projected activation exchange and matched random-subspace tests.",
            "The subspace does not identify neurons, upstream inputs, nucleotide features, or a biological promoter mechanism.",
        ],
        "next_gate": (
            "Preregister unchanged-basis transfer to genuinely new examples, then "
            "test whether L11H2/L11H5 corruption changes the rank-4 coordinates and "
            "whether restoring those coordinates rescues the classifier effect."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_low_rank_replication_v1",
        status="SEALED",
        model_identity={
            "checkpoint": config["checkpoint"]["candidate"],
            "revision": config["checkpoint"]["revision"],
        },
        input_paths=(
            args.development,
            args.confirmation,
            args.config,
            args.preflight_report,
        ),
        parameters={
            "combination": "post_hoc_development_fit_then_unchanged_basis_transfer",
            "selected_rank": rank,
            "no_new_model_scores": True,
        },
        seeds={"none": 0},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _validate_inputs(
    development: dict[str, Any],
    confirmation: dict[str, Any],
    config: dict[str, Any],
    preflight: dict[str, Any],
    *,
    config_hash: str,
) -> None:
    if (
        development.get("split") != "development"
        or confirmation.get("split") != "confirmation"
        or development.get("assay") != "promoter_L11MLP_low_rank_v1"
        or confirmation.get("assay") != development.get("assay")
        or development.get("config_sha256") != config_hash
        or confirmation.get("config_sha256") != config_hash
        or development.get("preflight_sha256") != confirmation.get("preflight_sha256")
        or preflight.get("config_sha256") != config_hash
        or preflight.get("status") != "QUALIFIED"
        or not preflight.get("qualified")
    ):
        raise RuntimeError("low-rank inputs do not share a qualified frozen instrument")
    if (
        development.get("evidence_status") != "POST_HOC_COHORT_REUSE"
        or confirmation.get("evidence_status") != "POST_HOC_COHORT_REUSE"
        or development["selection"]["source"] != "frozen_development_rank_rule"
        or confirmation["selection"]["source"] != "unchanged_development_rank_and_basis"
        or not development["selection"]["qualified"]
        or not confirmation["selection"]["qualified"]
        or development["selection"]["selected_rank"]
        != confirmation["selection"]["selected_rank"]
        or development["basis"] != confirmation["basis"]
    ):
        raise RuntimeError(
            "confirmation did not preserve the qualified low-rank selection"
        )


def _split_summary(result: dict[str, Any]) -> dict[str, Any]:
    full = result["full_L11MLP"]
    selected = result["selected_subspace"]
    complement = result["orthogonal_complement"]
    return {
        "n_pairs": result["n_pairs"],
        "full_L11MLP": _endpoint_summary(full),
        "selected_subspace": {
            **_endpoint_summary(selected),
            "full_class_margin_effect_capture_fraction": (
                selected["class_margin_loss"] / full["class_margin_loss"]
            ),
            "full_accuracy_effect_capture_fraction": (
                selected["accuracy_loss"] / full["accuracy_loss"]
            ),
            "full_AUROC_effect_capture_fraction": (
                selected["auroc_loss"] / full["auroc_loss"]
            ),
            "random_subspace_p_add_one": result["selection"][
                "random_subspace_p_add_one"
            ],
        },
        "orthogonal_complement": {
            **_endpoint_summary(complement),
            "full_class_margin_effect_fraction": (
                complement["class_margin_loss"] / full["class_margin_loss"]
            ),
        },
        "directional": {
            direction: _endpoint_summary(summary)
            for direction, summary in result["directional"].items()
        },
        "qualified": result["selection"]["qualified"],
    }


def _endpoint_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "class_margin_loss": summary["class_margin_loss"],
        "class_margin_loss_CI95": summary["class_margin_loss_CI95"],
        "class_margin_loss_p_add_one": summary["class_margin_loss_p_add_one"],
        "accuracy_loss": summary["accuracy_loss"],
        "accuracy_loss_CI95": summary["accuracy_loss_CI95"],
        "AUROC_loss": summary["auroc_loss"],
    }


if __name__ == "__main__":
    main()
