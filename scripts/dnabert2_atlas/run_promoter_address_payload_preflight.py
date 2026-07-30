#!/usr/bin/env python
"""Qualify fixed-dose promoter K/V and residual-write instrumentation."""

from __future__ import annotations

import argparse
import json
import math
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
from gfmcd.reconstruction_atlas.promoter_address_payload import (
    capture_residual_payload_rows,
    scan_qkv_operations,
)
from gfmcd.reconstruction_atlas.promoter_circuit import layer_pair_nulls
from gfmcd.reconstruction_atlas.promoter_head_scan import (
    native_pair_rows,
    select_gue_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    supply = json.loads(Path(args.supply_audit).read_text())
    calibration = json.loads(Path(args.calibration).read_text())
    preflight = config["calibration_and_preflight"]
    dose = int(config["surgery"]["source_token_dose_per_sequence_per_head"])
    if (
        supply.get("status") != "QUALIFIED"
        or int(supply["token_geometry"]["fixed_source_token_dose"]) != dose
        or calibration.get("status") != "CALIBRATION"
        or int(calibration["source_token_dose"]) != dose
        or int(calibration["GUE_pair_offset"])
        != int(preflight["calibration_GUE_pair_offset"])
    ):
        raise RuntimeError("matching supply and GUE calibration are required")
    pairs = select_gue_pairs(
        args.gue_test,
        count=int(preflight["preflight_GUE_pairs"]),
        offset=int(preflight["preflight_GUE_pair_offset"]),
        seed=int(config["inference"]["seed"]),
    )
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    heads = {
        "candidate_L11_pair": ((11, 2), (11, 5)),
        "all_layer11_heads": tuple((11, head) for head in range(12)),
    }
    rows = scan_qkv_operations(
        bundle,
        pairs,
        heads,
        (
            "identity",
            "key_permute",
            "value_permute",
            "key_value_joint_permute",
        ),
        source_token_count=dose,
        seed=int(config["surgery"]["seed"]),
        batch_pairs=len(pairs),
    )
    residual = capture_residual_payload_rows(
        bundle,
        pairs,
        layer=11,
        candidate_heads=((11, 2), (11, 5)),
        batch_pairs=len(pairs),
    )
    full_native = native_pair_rows(bundle, pairs, batch_pairs=len(pairs))
    half_native = native_pair_rows(bundle, pairs, batch_pairs=len(pairs) // 2)
    half_by_id = {str(row["pair_id"]): row for row in half_native}
    batch_drift = max(
        abs(float(row[endpoint]) - float(half_by_id[str(row["pair_id"])][endpoint]))
        for row in full_native
        for endpoint in ("target_margin", "control_margin")
    )
    identity_max = max(
        abs(float(row[endpoint]))
        for row in rows
        if row["operation"] == "identity"
        for endpoint in ("target_loss", "control_loss", "target_specific_loss")
    )
    sensitivity = {}
    projected_mde = {}
    for operation in ("key_permute", "value_permute"):
        full_values = [
            float(row["target_specific_loss"])
            for row in rows
            if row["head_pair"] == "all_layer11_heads" and row["operation"] == operation
        ]
        candidate_values = [
            float(row["target_specific_loss"])
            for row in rows
            if row["head_pair"] == "candidate_L11_pair"
            and row["operation"] == operation
        ]
        sensitivity[operation] = statistics.mean(abs(value) for value in full_values)
        projected_mde[operation] = (
            2.8
            * statistics.stdev(candidate_values)
            / math.sqrt(int(config["cohort"]["pairs_per_split"]))
        )
    null_count = len(
        layer_pair_nulls(
            ((11, 2), (11, 5)),
            num_heads=int(config["checkpoint"]["heads_per_layer"]),
        )
    )
    residual_finite = all(
        math.isfinite(float(row[key]))
        for row in (*residual["joint_rows"], *residual["head_rows"])
        for key in (
            ("target_write_norm", "control_write_norm", "target_control_cosine")
            if "delta_vector" not in row
            else (
                "target_write_norm",
                "control_write_norm",
                "target_control_cosine",
                "delta_norm",
            )
        )
    )
    checks = {
        "fresh_disjoint_GUE_slice": int(preflight["preflight_GUE_pair_offset"]) >= 80,
        "identity_reconstruction": identity_max
        <= float(preflight["maximum_identity_absolute_margin_difference"]),
        "key_sensitivity": sensitivity["key_permute"]
        >= float(
            preflight["minimum_all_layer11_key_mean_absolute_target_specific_effect"]
        ),
        "value_sensitivity": sensitivity["value_permute"]
        >= float(
            preflight["minimum_all_layer11_value_mean_absolute_target_specific_effect"]
        ),
        "projected_MDE": max(projected_mde.values())
        <= float(preflight["maximum_projected_192_pair_MDE_each_primary"]),
        "batch_invariance": batch_drift
        <= float(preflight["maximum_batch_invariance_absolute_margin_difference"]),
        "null_resolution": null_count == int(preflight["required_null_pairs"])
        and 2 / (null_count + 1) <= float(preflight["maximum_two_endpoint_Holm_floor"]),
        "residual_write_finite": residual_finite,
        "fixed_token_dose": dose
        == int(supply["token_geometry"]["fixed_source_token_dose"]),
    }
    qualified = all(checks.values())
    raw_rows = [
        *({"row_type": "qkv_operation", **row} for row in rows),
        *(
            {
                "row_type": "residual_joint",
                **{key: value for key, value in row.items() if key != "delta_vector"},
            }
            for row in residual["joint_rows"]
        ),
        *({"row_type": "residual_head", **row} for row in residual["head_rows"]),
    ]
    input_paths = (
        args.gue_test,
        args.config,
        args.supply_audit,
        args.calibration,
        args.checkpoint_audit,
    )
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id="dnabert2_promoter_address_payload_preflight_raw_v1",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "preflight": preflight,
            "source_token_dose": dose,
            "config_sha256": config_hash,
        },
        seeds={"selection_and_surgery": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_address_payload_preflight_v1",
        "status": "QUALIFIED" if qualified else "INSTRUMENT_UNRESOLVED",
        "qualified": qualified,
        "config_sha256": config_hash,
        "checkpoint_revision": bundle.identity()["revision"],
        "input_sha256": {
            name: sha256_file(path)
            for name, path in (
                ("gue_test", args.gue_test),
                ("supply_audit", args.supply_audit),
                ("calibration", args.calibration),
                ("checkpoint_audit", args.checkpoint_audit),
            )
        },
        "GUE_pair_ids": [pair_id for pair_id, _, _ in pairs],
        "measurements": {
            "identity_max_absolute_margin_difference": identity_max,
            "all_layer11_mean_absolute_target_specific_effect": sensitivity,
            "projected_192_pair_MDE": projected_mde,
            "batch_invariance_max_absolute_margin_difference": batch_drift,
            "null_pair_count": null_count,
        },
        "checks": checks,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(raw_rows),
        },
        "boundary": (
            "GUE preflight qualifies numerical reconstruction, surgery sensitivity, "
            "precision, and residual-write capture only."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_address_payload_preflight_v1",
        status="PREFLIGHT" if qualified else "INSTRUMENT_UNRESOLVED",
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={
            "preflight": preflight,
            "source_token_dose": dose,
            "config_sha256": config_hash,
        },
        seeds={"selection_and_surgery": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not qualified:
        raise SystemExit("promoter address/payload preflight did not qualify")


if __name__ == "__main__":
    main()
