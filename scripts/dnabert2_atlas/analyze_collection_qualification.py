#!/usr/bin/env python
"""Create the compact, outcome-blind Phase-A atlas qualification record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gfmcd.adapters.dnabert2 import DNABERT2_HF_ID, DNABERT2_REVISION
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.schema import load_collection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--baseline-training", required=True)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--matching-audit", required=True)
    parser.add_argument("--comparison-audit")
    parser.add_argument("--preflight-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    collection = load_collection(args.collection)
    background = load_collection(args.baseline_training)
    audit = json.loads(Path(args.matching_audit).read_text())
    comparison = (
        json.loads(Path(args.comparison_audit).read_text())
        if args.comparison_audit
        else None
    )
    config = json.loads(Path(args.preflight_config).read_text())
    minimum = int(config["behavior"]["preflight_minimum_clusters_per_site_class"])
    behavior_minimum = int(config["behavior"]["site_class_minimum_clusters"])
    matched_counts = {
        tuple(key.split("|", 1)): int(value)
        for key, value in audit["supply"]["matched_counts"].items()
    }
    site_classes = sorted(
        str(item) for item in config["behavior"]["admitted_site_classes"]
    )
    decisions = {}
    for site_class in site_classes:
        development = matched_counts.get(("development", site_class), 0)
        confirmation = matched_counts.get(("confirmation", site_class), 0)
        eligible = min(development, confirmation) >= minimum
        decisions[site_class] = {
            "development_matched_clusters": development,
            "confirmation_matched_clusters": confirmation,
            "preflight_minimum_clusters": minimum,
            "behavior_minimum_clusters": behavior_minimum,
            "status": "PREFLIGHT_ELIGIBLE" if eligible else "QUARANTINED",
            "behavior_census_eligible": (
                min(development, confirmation) >= behavior_minimum
            ),
            "reason": (
                "all empirical control families supplied on both target splits"
                if eligible
                else "insufficient outcome-blind empirical-control supply"
            ),
        }
    target_clusters = {record.homology_cluster for record in collection}
    background_clusters = {record.homology_cluster for record in background}
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "phase": "A_COLLECTION_QUALIFICATION",
        "target_examples": len(collection),
        "target_clusters": len(target_clusters),
        "baseline_examples": len(background),
        "baseline_clusters": len(background_clusters),
        "target_baseline_cluster_overlap": len(target_clusters & background_clusters),
        "controls_per_family": config["collection"]["controls_per_family"],
        "matching_balance": audit["balance"],
        "site_class_decisions": decisions,
        "preflight_eligible_site_classes": sorted(
            site_class
            for site_class, value in decisions.items()
            if value["status"] == "PREFLIGHT_ELIGIBLE"
        ),
        "quarantined_site_classes": sorted(
            site_class
            for site_class, value in decisions.items()
            if value["status"] == "QUARANTINED"
        ),
        "behavior_census_eligible_site_classes": sorted(
            site_class
            for site_class, value in decisions.items()
            if value["behavior_census_eligible"]
        ),
        "behavior_census_underpowered_site_classes": sorted(
            site_class
            for site_class, value in decisions.items()
            if not value["behavior_census_eligible"]
        ),
        "comparison_supply": (
            comparison["supply"]["matched_counts"] if comparison is not None else None
        ),
        "target_outcomes_computed": False,
        "claim_boundary": (
            "Collection and control supply are instrument facts. They do not show "
            "that DNABERT-2 reconstructs any biological element."
        ),
    }
    inputs = [
        args.collection,
        args.baseline_training,
        args.matches,
        args.matching_audit,
        args.preflight_config,
    ]
    if args.comparison_audit:
        inputs.append(args.comparison_audit)
    write_result_bundle(
        args.out,
        payload,
        experiment_id=(
            f"dnabert2_atlas_collection_qualification_v{config['schema_version']}"
        ),
        status="PREFLIGHT",
        model_identity={
            "hf_id": DNABERT2_HF_ID,
            "revision": DNABERT2_REVISION,
            "used_for_collection_qualification": False,
        },
        input_paths=inputs,
        parameters={
            "control_families_evaluated_separately": True,
            "outcome_blind": True,
        },
        seeds={"collection": int(config["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
