#!/usr/bin/env python
"""Qualify equal-dose promoter localization instrumentation on reserved GUE."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import load_dnabert2_promoter_classifier
from gfmcd.provenance import sha256_file
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_circuit import layer_pair_nulls
from gfmcd.reconstruction_atlas.promoter_head_scan import (
    native_pair_rows,
    select_gue_pairs,
)
from gfmcd.reconstruction_atlas.promoter_source_execution import (
    scan_equal_dose_tiles,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=32)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    supply = json.loads(Path(args.supply_audit).read_text())
    calibration = json.loads(Path(args.calibration).read_text())
    preflight = config["calibration_and_preflight"]
    if (
        config.get("status") != "FROZEN"
        or supply.get("status") != "QUALIFIED"
        or calibration.get("status") != "CALIBRATION"
        or int(calibration["GUE_pair_offset"])
        != int(preflight["calibration_GUE_pair_offset"])
        or int(calibration["n_pairs"]) != int(preflight["calibration_GUE_pairs"])
    ):
        raise RuntimeError("matching frozen config, supply, and calibration required")
    pairs = select_gue_pairs(
        args.gue_test,
        count=int(preflight["preflight_GUE_pairs"]),
        offset=int(preflight["preflight_GUE_pair_offset"]),
        seed=int(config["inference"]["seed"]),
    )
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    geometry = config["tile_geometry"]
    rows = scan_equal_dose_tiles(
        bundle,
        pairs,
        tile_width_nt=int(geometry["nominal_tile_width_nt"]),
        token_dose=int(geometry["fixed_BPE_token_dose_per_sequence"]),
        maximum_cover_nt=int(geometry["maximum_actual_cover_nt"]),
        maximum_cover_difference_nt=int(
            geometry["maximum_target_control_cover_length_difference_nt"]
        ),
        candidate_heads=((11, 2), (11, 5)),
        include_all_layer_heads=True,
        include_identity=True,
        batch_pairs=args.batch_pairs,
        progress=lambda message: print(f"preflight: {message}", flush=True),
    )
    full_native = native_pair_rows(bundle, pairs, batch_pairs=len(pairs))
    half_native = native_pair_rows(bundle, pairs, batch_pairs=len(pairs) // 2)
    half_by_id = {str(row["pair_id"]): row for row in half_native}
    batch_drift = max(
        abs(float(row[endpoint]) - float(half_by_id[str(row["pair_id"])][endpoint]))
        for row in full_native
        for endpoint in ("target_margin", "control_margin")
    )
    measurements = _measurements(
        rows,
        projected_n=int(geometry["observed_outcome_blind_minimum"]["development"]),
    )
    null_count = len(
        layer_pair_nulls(
            ((11, 2), (11, 5)),
            num_heads=int(config["checkpoint"]["heads_per_layer"]),
        )
    )
    checks = {
        "fresh_disjoint_GUE_slice": int(preflight["preflight_GUE_pair_offset"]) >= 128,
        "identity_reconstruction": measurements[
            "identity_maximum_absolute_margin_difference"
        ]
        <= float(preflight["maximum_identity_absolute_margin_difference"]),
        "mask_sensitivity": measurements["operations"]["mask"]["mean_absolute"]
        >= float(preflight["minimum_mask_mean_absolute_target_specific_effect"]),
        "regional_value_sensitivity": measurements["operations"][
            "all_layer_value_swap"
        ]["mean_absolute"]
        >= float(
            preflight["minimum_all_layer11_value_mean_absolute_target_specific_effect"]
        ),
        "candidate_value_precision": measurements["operations"]["candidate_value_swap"][
            "projected_80pct_MDE"
        ]
        <= float(preflight["maximum_candidate_value_projected_MDE"]),
        "mask_precision": measurements["operations"]["mask"]["projected_80pct_MDE"]
        <= float(preflight["maximum_mask_projected_MDE"]),
        "tile_resolution": measurements["minimum_tile_pairs"]
        >= int(preflight["minimum_eligible_GUE_pairs_per_tile"]),
        "batch_invariance": batch_drift
        <= float(preflight["maximum_batch_invariance_absolute_margin_difference"]),
        "regional_null_resolution": null_count
        == int(preflight["required_regional_null_pairs"]),
        "finite_outputs": all(
            math.isfinite(float(row[endpoint]))
            for row in rows
            for endpoint in (
                "target_loss",
                "control_loss",
                "target_specific_loss",
            )
        ),
    }
    qualified = all(checks.values())
    input_paths = (
        args.gue_test,
        args.config,
        args.supply_audit,
        args.calibration,
        args.checkpoint_audit,
        args.source_manifest,
    )
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        rows,
        experiment_id="dnabert2_promoter_source_localization_preflight_raw_v1",
        status="PREFLIGHT" if qualified else "INSTRUMENT_UNRESOLVED",
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "config_sha256": config_hash,
            "preflight": preflight,
            "geometry": geometry,
        },
        seeds={"selection": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_source_localization_preflight_v1",
        "status": "QUALIFIED" if qualified else "INSTRUMENT_UNRESOLVED",
        "qualified": qualified,
        "config_sha256": config_hash,
        "checkpoint_revision": bundle.identity()["revision"],
        "input_sha256": {
            name: sha256_file(path)
            for name, path in (
                ("gue_test", args.gue_test),
                ("config", args.config),
                ("supply_audit", args.supply_audit),
                ("calibration", args.calibration),
                ("checkpoint_audit", args.checkpoint_audit),
                ("source_manifest", args.source_manifest),
            )
        },
        "GUE_pair_ids": [pair_id for pair_id, _, _ in pairs],
        "measurements": {
            **measurements,
            "batch_invariance_max_absolute_margin_difference": batch_drift,
            "regional_null_pair_count": null_count,
        },
        "checks": checks,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(rows),
        },
        "boundary": (
            "Reserved GUE preflight qualifies numerical reconstruction, tile "
            "sensitivity, BPE resolution, precision, and null resolution only."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_source_localization_preflight_v1",
        status="PREFLIGHT" if qualified else "INSTRUMENT_UNRESOLVED",
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={"config_sha256": config_hash, "preflight": preflight},
        seeds={"selection": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _measurements(rows: list[dict[str, Any]], *, projected_n: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["operation"])].append(row)
    identity = grouped["candidate_identity"]
    output: dict[str, Any] = {
        "identity_maximum_absolute_margin_difference": max(
            abs(float(row[endpoint]))
            for row in identity
            for endpoint in ("target_loss", "control_loss", "target_specific_loss")
        ),
        "minimum_tile_pairs": min(
            sum(
                row["operation"] == "mask" and row["tile_index"] == tile for row in rows
            )
            for tile in range(30)
        ),
        "operations": {},
    }
    for operation in ("mask", "candidate_value_swap", "all_layer_value_swap"):
        operation_rows = grouped[operation]
        values = [float(row["target_specific_loss"]) for row in operation_rows]
        by_tile: dict[int, list[float]] = defaultdict(list)
        for row in operation_rows:
            by_tile[int(row["tile_index"])].append(float(row["target_specific_loss"]))
        maximum_stdev = max(
            statistics.stdev(tile_values)
            for tile_values in by_tile.values()
            if len(tile_values) >= 2
        )
        output["operations"][operation] = {
            "rows": len(values),
            "mean": statistics.mean(values),
            "mean_absolute": statistics.mean(abs(value) for value in values),
            "maximum_absolute": max(abs(value) for value in values),
            "maximum_tile_stdev": maximum_stdev,
            "projected_80pct_MDE": (2.8 * maximum_stdev / math.sqrt(projected_n)),
        }
    return output


if __name__ == "__main__":
    main()
