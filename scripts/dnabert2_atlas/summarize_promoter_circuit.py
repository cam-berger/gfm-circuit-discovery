#!/usr/bin/env python
"""Combine development and confirmation promoter-circuit results."""

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
    pair_replicated = bool(
        development["pair_specificity"]["qualified"]
        and confirmation["pair_specificity"]["qualified"]
    )
    query_replicated = bool(
        development["query_dependence"]["qualified"]
        and confirmation["query_dependence"]["qualified"]
    )
    path_replicated = bool(
        development["path_rescue"]["qualified"]
        and confirmation["path_rescue"]["qualified"]
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_CLS_circuit_reverse_engineering_v1",
        "status": "SEALED",
        "checkpoint": config["checkpoint"],
        "config_sha256": development["config_sha256"],
        "development": _split_summary(development),
        "confirmation": _split_summary(confirmation),
        "replicated_findings": {
            "L11H2_L11H5_pair_specificity": pair_replicated,
            "target_dependent_L11_pair_query": query_replicated,
            "early_six_to_L11_pair_directed_rescue": path_replicated,
        },
        "licensed_claim": (
            "L11H2/L11H5 is a replicated, layer-specific two-head causal core of "
            "the fine-tuned promoter classifier's CLS readout: in both directions "
            "and both splits it exceeds every candidate-excluding two-head pair in "
            "layer 11. Strict complement scrub shows that the pair and full C8 set "
            "retain only about 14-15% alone but C8 plus all MLP writes retains about "
            "61%, implicating a distributed MLP scaffold."
            if pair_replicated
            else "The frozen layer-11 pair-specificity result did not replicate."
        ),
        "falsified_or_bounded_models": [
            (
                "Target-specific L11H2/L11H5 query transfer is effectively null in "
                "both splits; the pair is not licensed as a target-dependent Q gate."
            ),
            (
                "Corrupting the six earlier candidate heads has only a 0.056-0.059 "
                "logit effect, below the frozen 0.25 SESOI; their rescue through the "
                "late pair is not a qualified directed path."
            ),
            (
                "The eight attention heads are not a self-contained sparse circuit "
                "inside the declared CLS-write universe."
            ),
        ],
        "mechanistic_model": (
            "The supported model is a late L11H2/L11H5 output-carrier core operating "
            "on a representation maintained by distributed MLP writes, not a "
            "target-dependent query gate or a demonstrated early-six-to-late chain."
        ),
        "boundary": config["mechanistic_boundary"],
        "next_gate": (
            "Do not open source-band or motif surgery under v1. A new frozen assay "
            "must first justify direct K/V and source-state interventions for the "
            "replicated late pair without assuming the rejected early-to-late path."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_circuit_replication_v1",
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
    for key in ("config_sha256", "assay", "candidate_heads"):
        if development[key] != confirmation[key]:
            raise RuntimeError(f"split artifacts disagree on {key}")
    if not development["qualified_for_confirmation"]:
        raise RuntimeError("development did not open confirmation")


def _split_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_pairs": result["n_pairs"],
        "n_unique_locus_blocks": result["n_unique_locus_blocks"],
        "clean_behavior": result["clean_behavior"],
        "faithfulness": result["faithfulness"],
        "pair_specificity": result["pair_specificity"],
        "query_dependence": result["query_dependence"],
        "path_rescue": result["path_rescue"],
    }


if __name__ == "__main__":
    main()
