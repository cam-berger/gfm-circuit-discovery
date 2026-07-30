#!/usr/bin/env python
"""Qualify native promoter-source edits on a fresh reserved GUE slice."""

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
from gfmcd.reconstruction_atlas.promoter_head_scan import select_gue_pairs
from gfmcd.reconstruction_atlas.promoter_native_execution import (
    build_geometry_qualified_gue_rows,
    score_identity_replay,
    score_native_edit_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--negative-calibration", required=True)
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=32)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    version = int(config["schema_version"])
    config_hash = frozen_config_sha256(args.config)
    supply = json.loads(Path(args.supply_audit).read_text())
    calibration = json.loads(Path(args.calibration).read_text())
    negative = json.loads(Path(args.negative_calibration).read_text())
    expected_revision = str(config["checkpoint"]["revision"])
    expected_pairs = int(config["cohort"]["pairs_per_split"])
    if (
        supply.get("status") != "QUALIFIED"
        or not all(supply.get("checks", {}).values())
        or calibration.get("status") != "CALIBRATION"
        or calibration.get("checkpoint_revision") != expected_revision
        or int(calibration["n_pairs"]) != 16
        or negative.get("status") != "CALIBRATION"
        or negative.get("checkpoint_revision") != expected_revision
    ):
        raise RuntimeError("matching qualified supply and calibrations are required")

    preflight = config["calibration_and_preflight"]
    candidate_pairs = select_gue_pairs(
        args.gue_test,
        count=int(preflight["preflight_GUE_candidate_pairs"]),
        offset=int(preflight["preflight_GUE_candidate_offset"]),
        seed=int(config["inference"]["seed"]),
    )
    bundle = load_dnabert2_promoter_classifier(
        str(config["checkpoint"]["candidate"]), device=args.device
    )
    rows = build_geometry_qualified_gue_rows(
        bundle,
        candidate_pairs,
        config=config,
        maximum=int(preflight["preflight_GUE_qualified_pairs"]),
    )
    operations = tuple(config["measurement"]["primary_operations"])
    primary = score_native_edit_rows(
        bundle,
        rows,
        operations=operations,
        batch_pairs=args.batch_pairs,
        progress=lambda message: print(f"preflight: {message}", flush=True),
    )
    comparison_batch = max(1, args.batch_pairs // 2)
    comparison = score_native_edit_rows(
        bundle,
        rows,
        operations=operations,
        batch_pairs=comparison_batch,
    )
    identity = score_identity_replay(bundle, rows, batch_pairs=args.batch_pairs)
    batch_drift = _maximum_batch_drift(primary, comparison)
    measurements = _measurements(primary, projected_n=expected_pairs)
    finite = all(
        math.isfinite(float(row[key]))
        for row in primary
        for key in (
            "target_clean_margin",
            "target_edited_margin",
            "control_clean_margin",
            "control_edited_margin",
            "target_specific_loss",
        )
    )
    checks = {
        "fresh_disjoint_GUE_slice": int(preflight["preflight_GUE_candidate_offset"])
        >= 544,
        "exact_geometry_qualified_pairs": len(rows)
        == int(preflight["preflight_GUE_qualified_pairs"]),
        "identity_replay": identity
        <= float(preflight["maximum_identity_absolute_margin_difference"]),
        "native_sensitivity": all(
            measurements[operation]["mean_absolute"]
            >= float(preflight["minimum_native_mean_absolute_target_specific_effect"])
            for operation in operations
        ),
        "native_precision": all(
            measurements[operation]["projected_80pct_MDE_at_target_n"]
            <= float(preflight["maximum_projected_native_MDE_at_target_n"])
            for operation in operations
        ),
        "batch_invariance": batch_drift
        <= float(preflight["maximum_batch_invariance_absolute_margin_difference"]),
        "finite_outputs": finite,
        "single_edit_design": all(
            measurement["edit_replicates_per_locus"] == 1
            for measurement in measurements.values()
        ),
    }
    qualified = all(checks.values())
    input_paths = (
        args.gue_test,
        args.config,
        args.supply_audit,
        args.calibration,
        args.negative_calibration,
        args.checkpoint_audit,
    )
    experiment_id = f"dnabert2_promoter_native_source_preflight_v{version}"
    artifact_status = "PREFLIGHT" if qualified else "INSTRUMENT_UNRESOLVED"
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        primary,
        experiment_id=f"{experiment_id}_raw",
        status=artifact_status,
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "config_sha256": config_hash,
            "preflight": preflight,
            "comparison_batch_pairs": comparison_batch,
        },
        seeds={"selection_and_edits": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": f"promoter_native_source_preflight_v{version}",
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
                ("calibration", args.calibration),
                ("negative_calibration", args.negative_calibration),
                ("checkpoint_audit", args.checkpoint_audit),
            )
        },
        "GUE_candidate_offset": int(preflight["preflight_GUE_candidate_offset"]),
        "GUE_candidate_pairs": int(preflight["preflight_GUE_candidate_pairs"]),
        "n_geometry_qualified_pairs": len(rows),
        "pair_ids": [str(row["example_id"]) for row in rows],
        "measurements": {
            "operations": measurements,
            "identity_replay_max_absolute_margin_difference": identity,
            "batch_invariance_max_absolute_margin_difference": batch_drift,
        },
        "checks": checks,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "rows": len(primary),
        },
        "boundary": (
            "This fresh GUE preflight qualifies edit sensitivity, projected "
            "precision, numerical invariance, and geometry only. It reads no "
            "GENCODE target or sealed confirmation outcome."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=experiment_id,
        status=artifact_status,
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={"config_sha256": config_hash, "preflight": preflight},
        seeds={"selection_and_edits": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _measurements(
    rows: list[dict[str, Any]], *, projected_n: int
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["operation"])][str(row["pair_id"])].append(
            float(row["target_specific_loss"])
        )
    output = {}
    for operation, by_pair in sorted(grouped.items()):
        values = [
            statistics.mean(replicates)
            for _pair_id, replicates in sorted(by_pair.items())
        ]
        stdev = statistics.stdev(values)
        output[operation] = {
            "n": len(values),
            "edit_replicates_per_locus": len(next(iter(by_pair.values()))),
            "mean": statistics.mean(values),
            "mean_absolute": statistics.mean(abs(value) for value in values),
            "stdev": stdev,
            "minimum": min(values),
            "maximum": max(values),
            "projected_80pct_MDE_at_target_n": (2.8 * stdev / math.sqrt(projected_n)),
        }
    return output


def _maximum_batch_drift(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> float:
    endpoints = (
        "target_clean_margin",
        "target_edited_margin",
        "control_clean_margin",
        "control_edited_margin",
        "target_specific_loss",
    )
    key = lambda row: (
        str(row["pair_id"]),
        str(row["operation"]),
        int(row["edit_replicate"]),
    )
    right_by_key = {key(row): row for row in right}
    if len(left) != len(right_by_key):
        raise RuntimeError("batch comparison rows do not align")
    return max(
        abs(float(row[endpoint]) - float(right_by_key[key(row)][endpoint]))
        for row in left
        for endpoint in endpoints
    )


if __name__ == "__main__":
    main()
