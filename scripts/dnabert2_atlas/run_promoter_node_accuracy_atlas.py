#!/usr/bin/env python
"""Run the frozen all-head/all-MLP promoter correctness atlas."""

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
from gfmcd.reconstruction_atlas.promoter_node_accuracy import (
    paired_node_accuracy_rows,
    rank_node_summaries,
    summarize_node_accuracy_rows,
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
    _validate_inputs(config, config_hash, supply, preflight)

    development = None
    if args.split == "confirmation":
        if not args.development_report:
            raise SystemExit(
                "--development-report is required before confirmation can open"
            )
        development = json.loads(Path(args.development_report).read_text())
        if (
            development.get("split") != "development"
            or development.get("config_sha256") != config_hash
            or development.get("preflight_sha256") != sha256_file(args.preflight_report)
            or development.get("confirmation_gate") != "confirmation_may_open"
        ):
            raise RuntimeError(
                "unchanged qualified development atlas is required for confirmation"
            )

    cohort = read_jsonl(args.cohort)
    selected = [row for row in cohort if row.get("split") == args.split]
    expected_ids = list(supply["selection"]["selected_example_ids"][args.split])
    if (
        len(selected) != int(config["cohort"]["pairs_per_split"])
        or [str(row["example_id"]) for row in selected] != expected_ids
        or _id_hash(expected_ids)
        != str(config["cohort"]["selected_id_sha256"][args.split])
    ):
        raise RuntimeError("cohort rows do not match the frozen outcome-blind supply")
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
    rows = paired_node_accuracy_rows(
        bundle,
        pairs,
        batch_pairs=args.batch_pairs,
        progress=lambda message: print(f"{args.split}: {message}", flush=True),
    )
    inference = config["inference"]
    summaries = summarize_node_accuracy_rows(
        rows,
        bootstrap_draws=int(inference["bootstrap_draws"]),
        sign_permutations=int(inference["shared_sign_permutations"]),
        seed=int(inference["seed"])
        ^ (0xACE1 if args.split == "development" else 0xACE2),
    )
    ranking = rank_node_summaries(summaries)
    sesoi = float(inference["accuracy_SESOI"])
    alpha = float(inference["familywise_alpha"])
    for rank, node in enumerate(ranking, start=1):
        summary = summaries[node]
        checks = {
            "accuracy_loss_at_least_SESOI": float(summary["accuracy_loss"]) >= sesoi,
            "positive_bootstrap_lower_bound": float(summary["accuracy_loss_CI95"][0])
            > 0,
            "maxT_add_one_p_at_most_alpha": float(summary["p_maxT_add_one"]) <= alpha,
        }
        summary["rank"] = rank
        summary["gate_checks"] = checks
        summary["qualified_on_split"] = all(checks.values())

    clean = summaries[ranking[0]]["clean"]
    clean_checks = {
        "accuracy": float(clean["accuracy"])
        >= float(config["clean_behavior_gate"]["minimum_accuracy"]),
        "AUROC": float(clean["auroc"])
        >= float(config["clean_behavior_gate"]["minimum_AUROC"]),
    }
    clean_qualified = all(clean_checks.values())
    candidate_config = inference["development_candidates"]
    if args.split == "development":
        selected_heads = _top_type(
            ranking, summaries, "head", int(candidate_config["heads"])
        )
        selected_mlps = _top_type(
            ranking, summaries, "mlp", int(candidate_config["mlps"])
        )
        selection_source = "frozen_development_accuracy_first_rank"
    else:
        selected_heads = list(
            development["candidate_selection"]["selected_heads"]  # type: ignore[index]
        )
        selected_mlps = list(
            development["candidate_selection"]["selected_MLPs"]  # type: ignore[index]
        )
        selection_source = "unchanged_development_candidates"

    qualified_nodes = [
        node for node in ranking if summaries[node]["qualified_on_split"]
    ]
    input_paths = tuple(
        path
        for path in (
            args.cohort,
            args.supply_audit,
            args.checkpoint_audit,
            args.config,
            args.preflight_report,
            args.development_report,
        )
        if path is not None
    )
    status = "DEVELOPMENT" if args.split == "development" else "SEALED"
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        rows,
        experiment_id=f"dnabert2_promoter_node_accuracy_{args.split}_raw_v1",
        status=status,
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "split": args.split,
            "config_sha256": config_hash,
            "node_universe": config["node_universe"],
            "intervention": config["scope"]["intervention"],
        },
        seeds={
            "inference": int(inference["seed"])
            ^ (0xACE1 if args.split == "development" else 0xACE2)
        },
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_all_head_and_MLP_accuracy_atlas_v1",
        "split": args.split,
        "status": status,
        "config_sha256": config_hash,
        "preflight_sha256": sha256_file(args.preflight_report),
        "n_pairs": len(pairs),
        "nodes_scored": len(summaries),
        "scope": config["scope"],
        "clean_behavior": {
            "metrics": clean,
            "checks": clean_checks,
            "qualified": clean_qualified,
        },
        "candidate_selection": {
            "source": selection_source,
            "selected_heads": selected_heads,
            "selected_MLPs": selected_mlps,
        },
        "ranking": ranking,
        "node_summaries": [summaries[node] for node in ranking],
        "qualified_nodes": qualified_nodes,
        "confirmation_gate": (
            "opened_from_instrument_valid_development"
            if args.split == "confirmation"
            else (
                "confirmation_may_open"
                if clean_qualified and len(summaries) == 156
                else "confirmation_remains_sealed"
            )
        ),
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "rows": len(rows),
        },
        "boundary": config["scope"]["boundary"],
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=f"dnabert2_promoter_node_accuracy_{args.split}_v1",
        status=status,
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={
            "split": args.split,
            "config_sha256": config_hash,
            "inference": inference,
            "node_gate": config["node_gate"],
            "batch_pairs": args.batch_pairs,
        },
        seeds={
            "inference": int(inference["seed"])
            ^ (0xACE1 if args.split == "development" else 0xACE2)
        },
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "clean": clean,
                "qualified_nodes": qualified_nodes,
                "top_12": [
                    {
                        "node": node,
                        "accuracy_loss": summaries[node]["accuracy_loss"],
                        "AUROC_loss": summaries[node]["auroc_loss"],
                        "class_margin_loss": summaries[node]["class_margin_loss"],
                        "p_maxT_add_one": summaries[node]["p_maxT_add_one"],
                    }
                    for node in ranking[:12]
                ],
                "development_candidates": {
                    "heads": selected_heads,
                    "MLPs": selected_mlps,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def _validate_inputs(
    config: dict[str, Any],
    config_hash: str,
    supply: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    if supply.get("status") != "QUALIFIED" or not all(
        supply.get("checks", {}).values()
    ):
        raise RuntimeError("qualified outcome-blind supply is required")
    if (
        preflight.get("status") != "QUALIFIED"
        or not preflight.get("qualified")
        or not all(preflight.get("checks", {}).values())
        or preflight.get("target_cohort_outcomes_computed") is not False
        or preflight.get("config_sha256") != config_hash
        or preflight.get("checkpoint_revision") != config["checkpoint"]["revision"]
    ):
        raise RuntimeError("qualified outcome-unseen preflight is required")


def _id_hash(example_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(example_ids) + "\n").encode()).hexdigest()


def _top_type(
    ranking: list[str],
    summaries: dict[str, dict[str, Any]],
    node_type: str,
    count: int,
) -> list[str]:
    selected = [node for node in ranking if summaries[node]["node_type"] == node_type][
        :count
    ]
    if len(selected) != count:
        raise RuntimeError(f"node atlas could not select {count} {node_type} nodes")
    return selected


if __name__ == "__main__":
    main()
