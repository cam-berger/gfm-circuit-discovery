#!/usr/bin/env python
"""Qualify CLS head/MLP/query/rescue hooks on a reserved GUE slice."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from gfmcd.adapters.dnabert2 import load_dnabert2_promoter_classifier
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_circuit import (
    no_op_diagnostics,
    strict_faithfulness_rows,
)
from gfmcd.reconstruction_atlas.promoter_head_scan import (
    native_pair_rows,
    select_gue_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    frozen_hash = frozen_config_sha256(args.config)
    preflight = config["preflight"]
    pairs = select_gue_pairs(
        args.gue_test,
        count=int(preflight["GUE_pairs"]),
        offset=int(preflight["GUE_pair_offset"]),
        seed=int(config["inference"]["seed"]),
    )
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    all_heads = [
        (layer, head)
        for layer in range(int(config["checkpoint"]["layers"]))
        for head in range(int(config["checkpoint"]["heads_per_layer"]))
    ]
    all_layers = list(range(int(config["checkpoint"]["layers"])))
    rows = strict_faithfulness_rows(
        bundle,
        pairs,
        {
            "empty": ([], []),
            "all_nodes": (all_heads, all_layers),
        },
        batch_pairs=len(pairs),
    )
    diagnostics = no_op_diagnostics(bundle, pairs)
    native_full = native_pair_rows(bundle, pairs, batch_pairs=len(pairs))
    native_half = native_pair_rows(bundle, pairs, batch_pairs=len(pairs) // 2)
    full_by_id = {str(row["pair_id"]): row for row in native_full}
    half_by_id = {str(row["pair_id"]): row for row in native_half}
    batch_drift = max(
        abs(float(full_by_id[pair_id][endpoint]) - float(half_by_id[pair_id][endpoint]))
        for pair_id in full_by_id
        for endpoint in ("target_margin", "control_margin")
    )
    by_condition = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in ("empty", "all_nodes")
    }
    empty_anchor_error = max(
        abs(float(row["patched_target_margin"]) - float(row["native_control_margin"]))
        for row in by_condition["empty"]
    )
    all_anchor_error = max(
        abs(float(row["patched_target_margin"]) - float(row["native_target_margin"]))
        for row in by_condition["all_nodes"]
    )
    full_scrub_loss = statistics.mean(
        float(row["native_target_margin"]) - float(row["patched_target_margin"])
        for row in by_condition["empty"]
    )
    finite = all(
        all(
            isinstance(row[key], (int, float))
            and float("-inf") < float(row[key]) < float("inf")
            for key in (
                "native_target_margin",
                "native_control_margin",
                "patched_target_margin",
            )
        )
        for row in rows
    )
    checks = {
        "fresh_GUE_slice": int(preflight["GUE_pair_offset"]) >= 48,
        "full_scrub_reproduces_control": empty_anchor_error
        <= float(preflight["maximum_absolute_anchor_error"]),
        "all_nodes_is_native": all_anchor_error
        <= float(preflight["maximum_absolute_noop_difference"]),
        "full_scrub_has_sensitivity": full_scrub_loss
        >= float(preflight["minimum_full_scrub_target_margin_loss"]),
        "complete_attention_capture": int(diagnostics["captured_attention_layers"])
        == int(preflight["required_captured_attention_layers"]),
        "head_self_patch_noop": float(
            diagnostics["head_self_patch_max_abs_margin_difference"]
        )
        <= float(preflight["maximum_absolute_noop_difference"]),
        "query_self_patch_noop": float(
            diagnostics["query_self_patch_max_abs_margin_difference"]
        )
        <= float(preflight["maximum_absolute_noop_difference"]),
        "native_restore_noop": float(
            diagnostics["native_restore_max_abs_margin_difference"]
        )
        <= float(preflight["maximum_absolute_noop_difference"]),
        "empty_complement_noop": float(
            diagnostics["empty_complement_max_abs_margin_difference"]
        )
        <= float(preflight["maximum_absolute_noop_difference"]),
        "batch_invariance": batch_drift
        <= float(preflight["maximum_absolute_anchor_error"]),
        "finite_scores": finite,
        "node_count": len(all_heads) + len(all_layers)
        == int(config["node_universe"]["node_count"]),
    }
    qualified = all(checks.values())
    input_paths = (
        args.gue_test,
        args.config,
        args.checkpoint_audit,
    )
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        rows,
        experiment_id="dnabert2_promoter_circuit_preflight_raw_v1",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={"preflight": preflight, "config_sha256": frozen_hash},
        seeds={"GUE_selection": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_circuit_hook_preflight_v1",
        "status": "QUALIFIED" if qualified else "INSTRUMENT_UNRESOLVED",
        "qualified": qualified,
        "config_sha256": frozen_hash,
        "checkpoint_revision": bundle.identity()["revision"],
        "input_sha256": {
            "gue_test": sha256_file(args.gue_test),
            "checkpoint_audit": sha256_file(args.checkpoint_audit),
        },
        "GUE_pair_ids": [pair_id for pair_id, _, _ in pairs],
        "measurements": {
            "full_scrub_target_to_control_max_abs_error": empty_anchor_error,
            "all_nodes_to_native_max_abs_error": all_anchor_error,
            "full_scrub_target_margin_loss_mean": full_scrub_loss,
            "batch_invariance_max_abs_margin_difference": batch_drift,
            **diagnostics,
        },
        "checks": checks,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(rows),
        },
        "mechanistic_boundary": (
            "GUE preflight qualifies hook correctness and sensitivity only; "
            "it is not evidence about the GENCODE target cohort."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_circuit_preflight_v1",
        status="PREFLIGHT" if qualified else "INSTRUMENT_UNRESOLVED",
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={"preflight": preflight, "config_sha256": frozen_hash},
        seeds={"GUE_selection": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not qualified:
        raise SystemExit("promoter circuit preflight did not qualify")


if __name__ == "__main__":
    main()
