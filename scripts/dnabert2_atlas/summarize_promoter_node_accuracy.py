#!/usr/bin/env python
"""Combine the sealed promoter node-accuracy atlas splits."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from gfmcd.reconstruction_atlas.artifacts import write_result_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    development = json.loads(Path(args.development).read_text())
    confirmation = json.loads(Path(args.confirmation).read_text())
    config = json.loads(Path(args.config).read_text())
    if (
        development.get("split") != "development"
        or confirmation.get("split") != "confirmation"
        or development.get("config_sha256") != confirmation.get("config_sha256")
        or development.get("preflight_sha256") != confirmation.get("preflight_sha256")
        or development["candidate_selection"]["selected_heads"]
        != confirmation["candidate_selection"]["selected_heads"]
        or development["candidate_selection"]["selected_MLPs"]
        != confirmation["candidate_selection"]["selected_MLPs"]
        or development.get("nodes_scored") != 156
        or confirmation.get("nodes_scored") != 156
    ):
        raise RuntimeError("atlas splits are not an unchanged complete replication")

    dev = {row["node"]: row for row in development["node_summaries"]}
    conf = {row["node"]: row for row in confirmation["node_summaries"]}
    if set(dev) != set(conf):
        raise RuntimeError("development and confirmation node families differ")
    joint_rows = []
    for node in dev:
        joint_rows.append(
            {
                "node": node,
                "node_type": dev[node]["node_type"],
                "development_rank": dev[node]["rank"],
                "confirmation_rank": conf[node]["rank"],
                "mean_rank": statistics.mean([dev[node]["rank"], conf[node]["rank"]]),
                "development_accuracy_loss": dev[node]["accuracy_loss"],
                "confirmation_accuracy_loss": conf[node]["accuracy_loss"],
                "development_AUROC_loss": dev[node]["auroc_loss"],
                "confirmation_AUROC_loss": conf[node]["auroc_loss"],
                "development_class_margin_loss": dev[node]["class_margin_loss"],
                "confirmation_class_margin_loss": conf[node]["class_margin_loss"],
                "qualified_development": dev[node]["qualified_on_split"],
                "qualified_confirmation": conf[node]["qualified_on_split"],
                "replicated_accuracy_gate": (
                    dev[node]["qualified_on_split"] and conf[node]["qualified_on_split"]
                ),
            }
        )
    joint_rows.sort(
        key=lambda row: (
            row["mean_rank"],
            -statistics.mean(
                [
                    row["development_accuracy_loss"],
                    row["confirmation_accuracy_loss"],
                ]
            ),
            row["node"],
        )
    )
    replicated = [row["node"] for row in joint_rows if row["replicated_accuracy_gate"]]
    candidates = [
        *development["candidate_selection"]["selected_heads"],
        *development["candidate_selection"]["selected_MLPs"],
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_all_head_and_MLP_accuracy_atlas_replication_v1",
        "status": "SEALED",
        "config_sha256": development["config_sha256"],
        "preflight_sha256": development["preflight_sha256"],
        "nodes_scored_each_split": 156,
        "clean_behavior": {
            "development": development["clean_behavior"],
            "confirmation": confirmation["clean_behavior"],
        },
        "development_candidates": development["candidate_selection"],
        "candidate_replication": {
            node: next(row for row in joint_rows if row["node"] == node)
            for node in candidates
        },
        "replicated_accuracy_gate_nodes": replicated,
        "joint_ranking": joint_rows,
        "strongest_claim": (
            "Nodes passing the frozen accuracy, bootstrap, and full-family maxT "
            "gates on both splits are replicated correctness-support candidates "
            "under reciprocal paired CLS-write exchange."
            if replicated
            else "No individual node passed the frozen accuracy gate on both splits."
        ),
        "boundary": config["scope"]["boundary"],
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_node_accuracy_replication_v1",
        status="SEALED",
        model_identity={
            "checkpoint": config["checkpoint"]["candidate"],
            "revision": config["checkpoint"]["revision"],
        },
        input_paths=(args.development, args.confirmation, args.config),
        parameters={
            "inference": config["inference"],
            "node_gate": config["node_gate"],
            "joint_ranking": "mean development and confirmation accuracy-first rank",
        },
        seeds={"inference": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "replicated_accuracy_gate_nodes": replicated,
                "top_15_joint": joint_rows[:15],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
