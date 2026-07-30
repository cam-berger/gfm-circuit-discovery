#!/usr/bin/env python
"""Combine development and sealed confirmation promoter head-atlas results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gfmcd.reconstruction_atlas.artifacts import write_result_bundle

ENDPOINTS = ("necessity", "sufficiency")


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
    _validate_inputs(development, confirmation)

    candidates = development["candidate_selection"]["selected_heads"]
    replicated = bool(
        development["qualified_on_split"] and confirmation["qualified_on_split"]
    )
    individual_summaries = _individual_summaries(development, confirmation, candidates)
    highest_effect_candidates = sorted(
        individual_summaries,
        key=lambda row: (
            -sum(row["confirmation"][endpoint]["mean"] for endpoint in ENDPOINTS)
        ),
    )[:2]
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_clean_behavior_CLS_head_attribution_and_patching_v1",
        "status": "SEALED",
        "checkpoint": config["checkpoint"],
        "preflight_report_id": development["preflight_report_id"],
        "config_sha256": development["config_sha256"],
        "behavior": config["scope"]["behavior"],
        "development": _split_summary(development),
        "confirmation": _split_summary(confirmation),
        "development_frozen_candidates": candidates,
        "candidate_set_CLS_head_output_effect_replicated": replicated,
        "individual_candidate_priorities": individual_summaries,
        "attention_source_clues": _attention_summaries(confirmation, candidates),
        "licensed_claim": (
            "Across two disjoint 192-locus GENCODE splits, the development-frozen "
            "eight-head set at the unpadded CLS destination causally supports the "
            "fine-tuned classifier's clean true-TSS versus paired-pseudo-TSS margin "
            "in both transfer directions and exceeds exact layer-profile-matched "
            "random head sets. This supports candidate-set CLS head-output mediation."
            if replicated
            else (
                "The development-frozen head set did not pass the complete sealed "
                "confirmation gate; no replicated candidate-set mediation claim is licensed."
            )
        ),
        "mechanistic_priorities": {
            "highest_individual_effect_candidates": [
                row["head"] for row in highest_effect_candidates
            ],
            "bidirectional_sign_replicated_candidates": [
                summary["head"]
                for summary in individual_summaries
                if summary["positive_sign_supported_both_directions_both_splits"]
            ],
            "source_localization_candidate": (
                "L9H7: unlike the dominant far-upstream or special-token patterns of "
                "the other candidates, its target CLS attention shifts toward the "
                "proximal-upstream and TSS-anchor bands. This is attention-only evidence."
            ),
        },
        "boundary": [
            "The replicated layer-matched null result belongs to the eight-head set, not to each member.",
            "Individual exact patches and corrected sign tests prioritize heads but do not establish individual-head specificity without a matched single-head null family.",
            "Attention source bands are candidate-generating descriptions; they do not show which source positions or motifs causally drive a head.",
            "The intervention is at each head's CLS context before the shared output projection; QK selection, OV content, downstream paths, and a sparse circuit remain unresolved.",
            "The result concerns the clean supervised TSS-versus-pseudo-TSS contrast and does not revive the failed target-preserving local-context hypothesis.",
            "Exact GUE exclusion cannot rule out undocumented near-duplicate fine-tuning sequences.",
            "No claim about transcriptional function in cells is licensed.",
        ],
        "next_gate": (
            "Freeze a second-stage decomposition before new outcomes: test L11H2/L11H5 "
            "as an exact layer-matched pair; then separately localize QK source selection, "
            "OV-carried content, and downstream path mediation with matched-size nulls."
            if replicated
            else "none"
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_head_atlas_replication_v1",
        status="SEALED",
        model_identity=config["checkpoint"],
        input_paths=(
            args.development,
            args.confirmation,
            args.config,
            args.preflight_report,
        ),
        parameters={
            "combination": "development_frozen_candidates_then_sealed_confirmation",
            "no_new_model_scores": True,
        },
        seeds={"none": 0},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _validate_inputs(development: dict[str, Any], confirmation: dict[str, Any]) -> None:
    if development["split"] != "development":
        raise RuntimeError("development artifact carries the wrong split")
    if confirmation["split"] != "confirmation":
        raise RuntimeError("confirmation artifact carries the wrong split")
    for key in ("preflight_report_id", "config_sha256", "assay"):
        if development[key] != confirmation[key]:
            raise RuntimeError(f"split artifacts disagree on {key}")
    development_heads = development["candidate_selection"]["selected_heads"]
    confirmation_heads = confirmation["candidate_selection"]["selected_heads"]
    if development_heads != confirmation_heads:
        raise RuntimeError("confirmation did not preserve development candidates")
    if (
        confirmation["candidate_selection"]["source"]
        != "unchanged_development_candidates"
    ):
        raise RuntimeError("confirmation candidates were not explicitly frozen")


def _split_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_pairs": result["n_pairs"],
        "n_unique_locus_blocks": result["n_unique_locus_blocks"],
        "clean_behavior": result["clean_behavior"],
        "attribution_exact_patch_spearman": result["attribution_exact_patch_spearman"],
        "candidate_group_tests": result["candidate_group_inference"]["tests"],
        "qualified": result["qualified_on_split"],
    }


def _individual_summaries(
    development: dict[str, Any],
    confirmation: dict[str, Any],
    candidates: list[str],
) -> list[dict[str, Any]]:
    output = []
    for head in candidates:
        split_tests = {
            split_name: {
                endpoint: result["individual_candidate_inference"]["tests"][
                    f"{head}|{endpoint}"
                ]
                for endpoint in ENDPOINTS
            }
            for split_name, result in (
                ("development", development),
                ("confirmation", confirmation),
            )
        }
        output.append(
            {
                "head": head,
                **split_tests,
                "positive_sign_supported_both_directions_both_splits": all(
                    split_tests[split][endpoint]["positive_sign_supported"]
                    for split in ("development", "confirmation")
                    for endpoint in ENDPOINTS
                ),
                "boundary": "candidate evidence; no individual matched-head null claim",
            }
        )
    return output


def _attention_summaries(
    confirmation: dict[str, Any], candidates: list[str]
) -> list[dict[str, Any]]:
    output = []
    for head in candidates:
        bands = confirmation["candidate_attention_source_bands"][head]
        genomic_bands = {name: row for name, row in bands.items() if name != "special"}
        dominant = max(
            genomic_bands,
            key=lambda name: genomic_bands[name]["target_mean"],
        )
        largest_shift = max(
            genomic_bands,
            key=lambda name: abs(genomic_bands[name]["target_minus_control"]),
        )
        output.append(
            {
                "head": head,
                "dominant_target_genomic_band": dominant,
                "dominant_target_genomic_attention_mass": genomic_bands[dominant][
                    "target_mean"
                ],
                "largest_target_control_genomic_shift_band": largest_shift,
                "largest_target_control_genomic_shift": genomic_bands[largest_shift][
                    "target_minus_control"
                ],
                "special_token_target_attention_mass": bands["special"]["target_mean"],
                "boundary": "descriptive attention only",
            }
        )
    return output


if __name__ == "__main__":
    main()
