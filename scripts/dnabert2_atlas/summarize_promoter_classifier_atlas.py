#!/usr/bin/env python
"""Combine the promoter classifier development and sealed confirmation results."""

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
    if development["split"] != "development":
        raise RuntimeError("development artifact carries the wrong split")
    if confirmation["split"] != "confirmation":
        raise RuntimeError("confirmation artifact carries the wrong split")
    for key in ("preflight_report_id", "config_sha256", "checkpoint_candidate"):
        if development[key] != confirmation[key]:
            raise RuntimeError(f"split artifacts disagree on {key}")
    clean_replicated = bool(
        development["decisions"]["clean_enrichment"]["qualified"]
        and confirmation["decisions"]["clean_enrichment"]["qualified"]
    )
    context_replicated = bool(
        development["decisions"]["context_difference"]["qualified"]
        and confirmation["decisions"]["context_difference"]["qualified"]
    )
    fully_replicated = clean_replicated and context_replicated
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "fine_tuned_promoter_margin_target_preserving_context_v1",
        "status": "SEALED",
        "checkpoint": config["checkpoint"],
        "preflight_report_id": development["preflight_report_id"],
        "config_sha256": development["config_sha256"],
        "EPDnew_cohort_status": (
            "excluded: all 768 selected 300-nt EPDnew windows occur verbatim in GUE"
        ),
        "replacement_cohort": {
            "source": "GENCODE v50 basic protein-coding MANE Select TSSs",
            "training_corpus_filter": (
                "no exact forward or reverse-complement membership in any GUE split"
            ),
            "development_n": development["n_examples"],
            "confirmation_n": confirmation["n_examples"],
        },
        "development": _split_summary(development),
        "confirmation": _split_summary(confirmation),
        "clean_promoter_margin_replicated": clean_replicated,
        "target_preserving_context_effect_replicated": context_replicated,
        "fully_replicated": fully_replicated,
        "licensed_claim": (
            "At exact-GUE-excluded GENCODE MANE TSS windows, the fine-tuned "
            "DNABERT-2 promoter logit margin distinguishes true TSSs from "
            "composition-matched same-locus pseudo-TSSs. The target-preserving local "
            "context difference-in-differences did not replicate and licenses no "
            "context-specific mechanism claim."
            if clean_replicated and not context_replicated
            else (
                "Both the promoter margin and target-preserving context effect replicated."
                if fully_replicated
                else "Neither preregistered promoter classifier effect replicated."
            )
        ),
        "boundary": [
            "This is classifier behavior on annotated human sequence, not transcriptional function in cells.",
            "Exact GUE membership is excluded, but undocumented near-duplicate or homologous GUE training sequences cannot be ruled out.",
            "The intervention preserves the five-nucleotide anchor, length, local mononucleotide composition, and target/control Hamming dose; it does not preserve dinucleotide order.",
            "No attention head, MLP, residual direction, necessity, or mediation path was tested.",
        ],
        "next_gate": (
            config["mechanistic_boundary"]["next_gate"] if fully_replicated else "none"
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_classifier_replication_v2",
        status="SEALED",
        model_identity=config["checkpoint"],
        input_paths=(
            args.development,
            args.confirmation,
            args.config,
            args.preflight_report,
        ),
        parameters={
            "combination": "development_gate_then_sealed_confirmation",
            "no_new_model_scores": True,
        },
        seeds={"none": 0},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _split_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_examples": result["n_examples"],
        "n_clusters": result["n_clusters"],
        "qualified": result["qualified_on_split"],
        "mean_target_clean_margin": result["summary"]["mean_target_clean_margin"],
        "mean_control_clean_margin": result["summary"]["mean_control_clean_margin"],
        "tests": {
            endpoint: {
                "mean": test["observed_cluster_mean"],
                "CI": test["cluster_bootstrap_CI"],
                "p_holm": test["p_holm"],
                "qualified": result["decisions"][endpoint]["qualified"],
            }
            for endpoint, test in result["inference"]["tests"].items()
        },
    }


if __name__ == "__main__":
    main()
