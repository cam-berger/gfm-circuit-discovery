#!/usr/bin/env python
"""Combine development and confirmation promoter address/payload results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gfmcd.reconstruction_atlas.artifacts import write_result_bundle


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
    _validate(development, confirmation)

    dev_roles = set(development["primary_inference"]["qualified_roles"])
    confirmation_roles = set(confirmation["replicated_computational_roles"])
    replicated_roles = sorted(dev_roles & confirmation_roles)
    value_replicated = "value_permute" in replicated_roles
    key_replicated = "key_permute" in replicated_roles
    residual_cosine = confirmation["residual_payload"][
        "split_mean_direction_cosine_with_development"
    ]

    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_L11_address_payload_decomposition_v1",
        "status": "SEALED",
        "checkpoint": config["checkpoint"],
        "config_sha256": development["config_sha256"],
        "candidate_pair": config["candidate_and_nulls"]["candidate_pair"],
        "source_token_dose": config["surgery"][
            "source_token_dose_per_sequence_per_head"
        ],
        "development": _split_summary(development),
        "confirmation": _split_summary(confirmation),
        "replicated_computational_roles": replicated_roles,
        "replicated_findings": {
            "content_key_address_dependence": key_replicated,
            "value_payload_alignment_dependence": value_replicated,
            "residual_write_direction_cosine": residual_cosine,
            "residual_write_direction_status": "descriptive",
        },
        "licensed_claim": (
            "The replicated L11H2/L11H5 promoter-classifier core depends on the "
            "native association between its attention weights and source value "
            "vectors. Permuting V across 40 fixed genomic token positions while "
            "leaving Q, K, and post-softmax attention probabilities native causes "
            "a target-specific promoter-margin loss that replicates against every "
            "equal-size layer-11 null pair."
            if value_replicated
            else "The frozen value-payload dependence result did not replicate."
        ),
        "falsified_or_bounded_models": [
            (
                "Content-key permutation does not meet the frozen 0.05-logit "
                "practical-effect threshold in either split, so this assay does "
                "not support a target-specific key-address role for the late pair."
            ),
            (
                "Joint K/V permutation is a secondary ALiBi-relative positional "
                "diagnostic and does not independently qualify a mechanistic role."
            ),
            (
                "The stable residual-write direction remains descriptive and has "
                "no promoter-semantic or direction-only causal label."
            ),
        ],
        "mechanistic_model": (
            "Together with D6's null query transfer, D7 narrows the late pair to a "
            "value/source-state payload-routing role: native attention probabilities "
            "must remain aligned with their value vectors. The represented feature "
            "and causal source positions are unresolved."
        ),
        "boundary": config["mechanistic_boundary"],
        "next_gate": (
            "Freeze a new source-localization assay before using the D7 outcome to "
            "select TSS-relative bands or motifs. It must preserve the replicated V "
            "estimand, compare equal-dose source interventions with matched bands, "
            "and keep semantic labeling separate from causal localization."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_address_payload_replication_v1",
        status="SEALED",
        model_identity=config["checkpoint"],
        input_paths=(
            args.development,
            args.confirmation,
            args.config,
            args.preflight_report,
        ),
        parameters={
            "combination": "frozen_development_then_untouched_confirmation",
            "no_new_model_scores": True,
        },
        seeds={"none": 0},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _validate(development: dict[str, Any], confirmation: dict[str, Any]) -> None:
    if development["split"] != "development":
        raise RuntimeError("development artifact carries the wrong split")
    if confirmation["split"] != "confirmation":
        raise RuntimeError("confirmation artifact carries the wrong split")
    for key in (
        "config_sha256",
        "assay",
        "candidate_pair",
        "source_token_dose",
    ):
        if development[key] != confirmation[key]:
            raise RuntimeError(f"split artifacts disagree on {key}")
    if not development["qualified_for_confirmation"]:
        raise RuntimeError("development did not open confirmation")


def _split_summary(result: dict[str, Any]) -> dict[str, Any]:
    residual = result["residual_payload"]
    return {
        "n_pairs": result["n_pairs"],
        "n_unique_locus_blocks": result["n_unique_locus_blocks"],
        "selected_id_sha256": result["selected_id_sha256"],
        "clean_behavior": result["clean_behavior"],
        "identity_control": result["identity_control"],
        "primary_inference": result["primary_inference"],
        "joint_KV_position_diagnostic": result["joint_KV_position_diagnostic"],
        "residual_payload": {
            "candidate_pair_write_geometry": residual["candidate_pair_write_geometry"],
            "directional_coherence_ratio": residual["directional_coherence_ratio"],
            "mean_per_locus_delta_norm": residual["mean_per_locus_delta_norm"],
            "split_mean_delta_norm": residual["split_mean_delta_norm"],
            "split_mean_direction_cosine_with_development": residual[
                "split_mean_direction_cosine_with_development"
            ],
            "per_locus_projection_on_development_direction": residual[
                "per_locus_projection_on_development_direction"
            ],
            "boundary": residual["boundary"],
        },
    }


if __name__ == "__main__":
    main()
