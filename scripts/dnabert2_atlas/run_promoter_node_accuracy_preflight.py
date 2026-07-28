#!/usr/bin/env python
"""Qualify the all-head/all-MLP accuracy atlas on a fresh GUE slice."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from gfmcd.adapters.dnabert2 import load_dnabert2_promoter_classifier
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_assay import (
    binary_classification_metrics,
)
from gfmcd.reconstruction_atlas.promoter_head_scan import (
    native_pair_rows,
    select_gue_pairs,
)
from gfmcd.reconstruction_atlas.promoter_node_accuracy import (
    all_cls_nodes,
    paired_final_cls_residual_rows,
    paired_node_accuracy_rows,
    paired_node_set_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=32)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    supply = json.loads(Path(args.supply_audit).read_text())
    if supply.get("status") != "QUALIFIED" or not all(
        supply.get("checks", {}).values()
    ):
        raise RuntimeError("qualified outcome-blind node supply is required")
    preflight = config["preflight"]
    pairs = select_gue_pairs(
        args.gue_test,
        count=int(preflight["GUE_pairs"]),
        offset=int(preflight["GUE_pair_offset"]),
        seed=int(config["inference"]["seed"]),
    )
    bundle = load_dnabert2_promoter_classifier(
        str(config["checkpoint"]["candidate"]), device=args.device
    )
    nodes = all_cls_nodes(
        layers=int(config["checkpoint"]["layers"]),
        heads_per_layer=int(config["checkpoint"]["heads_per_layer"]),
    )
    heads = [
        (int(node["layer"]), int(node["head"]))
        for node in nodes
        if node["node_type"] == "head"
    ]
    mlps = [int(node["layer"]) for node in nodes if node["node_type"] == "mlp"]
    individual = paired_node_accuracy_rows(
        bundle,
        pairs,
        batch_pairs=args.batch_pairs,
        progress=lambda message: print(f"preflight individual: {message}", flush=True),
    )
    terminal_control = int(config["schema_version"]) >= 2
    if terminal_control:
        positive_control = paired_final_cls_residual_rows(
            bundle,
            pairs,
            direction="bidirectional_swap",
            batch_pairs=args.batch_pairs,
        )
        self_patch = paired_final_cls_residual_rows(
            bundle,
            pairs,
            direction="target_from_self",
            batch_pairs=args.batch_pairs,
        )
        control_name = "terminal_CLS_swap"
    else:
        positive_control = paired_node_set_rows(
            bundle,
            pairs,
            heads=heads,
            mlp_layers=mlps,
            direction="bidirectional_swap",
            batch_pairs=args.batch_pairs,
        )
        self_patch = paired_node_set_rows(
            bundle,
            pairs,
            heads=heads,
            mlp_layers=mlps,
            direction="target_from_self",
            batch_pairs=args.batch_pairs,
        )
        control_name = "full_node_swap"
    native_full = native_pair_rows(bundle, pairs, batch_pairs=len(pairs))
    native_half = native_pair_rows(bundle, pairs, batch_pairs=max(1, len(pairs) // 2))
    half_by_id = {str(row["pair_id"]): row for row in native_half}
    batch_drift = max(
        abs(float(row[endpoint]) - float(half_by_id[str(row["pair_id"])][endpoint]))
        for row in native_full
        for endpoint in ("target_margin", "control_margin")
    )
    labels = [label for _row in positive_control for label in (1, 0)]
    clean_margins = [
        margin
        for row in positive_control
        for margin in (
            float(row["native_target_margin"]),
            float(row["native_control_margin"]),
        )
    ]
    swapped_margins = [
        margin
        for row in positive_control
        for margin in (
            float(row["patched_target_margin"]),
            float(row["patched_control_margin"]),
        )
    ]
    clean_metrics = binary_classification_metrics(labels, clean_margins)
    swap_metrics = binary_classification_metrics(labels, swapped_margins)
    control_accuracy_loss = float(clean_metrics["accuracy"]) - float(
        swap_metrics["accuracy"]
    )
    control_class_margin_loss = statistics.mean(
        (
            float(row["native_target_margin"])
            - float(row["patched_target_margin"])
            + float(row["patched_control_margin"])
            - float(row["native_control_margin"])
        )
        / 2
        for row in positive_control
    )
    self_drift = max(
        abs(float(row[native]) - float(row[patched]))
        for row in self_patch
        for native, patched in (
            ("native_target_margin", "patched_target_margin"),
            ("native_control_margin", "patched_control_margin"),
        )
    )
    by_node: dict[str, list[float]] = defaultdict(list)
    node_type = {}
    for row in individual:
        by_node[str(row["node"])].append(abs(float(row["class_margin_loss"])))
        node_type[str(row["node"])] = str(row["node_type"])
    maximum_sensitivity = {
        kind: max(
            statistics.mean(values)
            for node, values in by_node.items()
            if node_type[node] == kind
        )
        for kind in ("head", "mlp")
    }
    finite = all(
        math.isfinite(float(row[key]))
        for row in individual
        for key in (
            "native_target_margin",
            "native_control_margin",
            "patched_target_margin",
            "patched_control_margin",
            "class_margin_loss",
            "accuracy_loss",
        )
    )
    checks = {
        "fresh_disjoint_GUE_slice": int(preflight["GUE_pair_offset"]) >= 928,
        "clean_accuracy": float(clean_metrics["accuracy"])
        >= float(preflight["minimum_clean_accuracy"]),
        "clean_AUROC": float(clean_metrics["auroc"])
        >= float(preflight["minimum_clean_AUROC"]),
        "individual_head_sensitivity": maximum_sensitivity["head"]
        >= float(
            preflight["minimum_maximum_individual_head_mean_absolute_class_margin_loss"]
        ),
        "individual_MLP_sensitivity": maximum_sensitivity["mlp"]
        >= float(
            preflight["minimum_maximum_individual_MLP_mean_absolute_class_margin_loss"]
        ),
        "self_patch_noop": self_drift
        <= float(
            preflight[
                (
                    "maximum_terminal_CLS_self_patch_absolute_margin_difference"
                    if terminal_control
                    else "maximum_self_patch_absolute_margin_difference"
                )
            ]
        ),
        "batch_invariance": batch_drift
        <= float(preflight["maximum_batch_invariance_absolute_margin_difference"]),
        "complete_node_family": len(by_node) == int(config["node_universe"]["total"]),
        "finite_outputs": finite,
    }
    if terminal_control:
        checks["terminal_CLS_swap_accuracy_sensitivity"] = (
            control_accuracy_loss
            >= float(preflight["minimum_terminal_CLS_swap_accuracy_loss"])
        )
        checks["terminal_CLS_swap_margin_sensitivity"] = (
            control_class_margin_loss
            >= float(preflight["minimum_terminal_CLS_swap_class_margin_loss"])
        )
    else:
        checks["full_node_swap_accuracy_sensitivity"] = control_accuracy_loss >= float(
            preflight["minimum_full_node_swap_accuracy_loss"]
        )
        checks["full_node_swap_margin_sensitivity"] = (
            control_class_margin_loss
            >= float(preflight["minimum_full_node_swap_class_margin_loss"])
        )
    qualified = all(checks.values())
    input_paths = (
        args.gue_test,
        args.config,
        args.supply_audit,
        args.checkpoint_audit,
    )
    raw_rows = [
        *({"record_type": "individual_node", **row} for row in individual),
        *({"record_type": control_name, **row} for row in positive_control),
        *({"record_type": "self_patch", **row} for row in self_patch),
    ]
    artifact_status = "PREFLIGHT" if qualified else "INSTRUMENT_UNRESOLVED"
    protocol_version = int(config["schema_version"])
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id=(
            f"dnabert2_promoter_node_accuracy_preflight_raw_v{protocol_version}"
        ),
        status=artifact_status,
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "config_sha256": config_hash,
            "node_universe": config["node_universe"],
            "preflight": preflight,
        },
        seeds={"GUE_selection": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": f"promoter_node_accuracy_preflight_v{protocol_version}",
        "status": "QUALIFIED" if qualified else "INSTRUMENT_UNRESOLVED",
        "qualified": qualified,
        "target_cohort_outcomes_computed": False,
        "config_sha256": config_hash,
        "checkpoint_revision": bundle.identity()["revision"],
        "input_sha256": {
            name: sha256_file(path)
            for name, path in (
                ("gue_test", args.gue_test),
                ("config", args.config),
                ("supply_audit", args.supply_audit),
                ("checkpoint_audit", args.checkpoint_audit),
            )
        },
        "GUE_pair_ids": [pair_id for pair_id, _, _ in pairs],
        "measurements": {
            "clean": clean_metrics,
            control_name: {
                "metrics": swap_metrics,
                "accuracy_loss": control_accuracy_loss,
                "class_margin_loss": control_class_margin_loss,
            },
            "maximum_individual_mean_absolute_class_margin_loss": (maximum_sensitivity),
            "self_patch_max_absolute_margin_difference": self_drift,
            "batch_invariance_max_absolute_margin_difference": batch_drift,
            "nodes_scored": len(by_node),
        },
        "checks": checks,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "rows": len(raw_rows),
        },
        "boundary": (
            "Fresh GUE preflight qualifies correctness, sensitivity, numerical "
            "invariance, and the 156-node hook family only. It reads no target "
            "cohort outcome."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=f"dnabert2_promoter_node_accuracy_preflight_v{protocol_version}",
        status=artifact_status,
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={"config_sha256": config_hash, "preflight": preflight},
        seeds={"GUE_selection": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
