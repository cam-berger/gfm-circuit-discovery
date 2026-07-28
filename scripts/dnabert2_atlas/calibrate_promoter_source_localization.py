#!/usr/bin/env python
"""Calibrate equal-dose promoter tile interventions on reserved GUE pairs."""

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
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.promoter_head_scan import select_gue_pairs
from gfmcd.reconstruction_atlas.promoter_source_execution import (
    scan_equal_dose_tiles,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pair-offset", type=int, default=112)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--batch-pairs", type=int, default=16)
    args = parser.parse_args()

    supply = json.loads(Path(args.supply_audit).read_text())
    if (
        supply.get("status") != "QUALIFIED"
        or supply.get("model_outcomes_computed") is not False
    ):
        raise RuntimeError("qualified outcome-blind localization supply is required")
    pairs = select_gue_pairs(
        args.gue_test,
        count=args.pairs,
        offset=args.pair_offset,
        seed=args.seed,
    )
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    geometry = supply["geometry"]
    rows = scan_equal_dose_tiles(
        bundle,
        pairs,
        tile_width_nt=int(geometry["nominal_tile_width_nt"]),
        token_dose=int(geometry["fixed_BPE_token_dose"]),
        maximum_cover_nt=int(geometry["maximum_actual_cover_nt"]),
        maximum_cover_difference_nt=int(
            geometry["maximum_target_control_cover_difference_nt"]
        ),
        candidate_heads=((11, 2), (11, 5)),
        include_all_layer_heads=True,
        include_identity=True,
        batch_pairs=args.batch_pairs,
        progress=lambda message: print(f"calibration: {message}", flush=True),
    )
    minimum_target_n = min(
        value["eligible_pairs"]
        for value in geometry["by_split_and_tile"]["development"].values()
    )
    summaries = _summaries(rows, projected_n=minimum_target_n)
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        rows,
        experiment_id="dnabert2_promoter_source_localization_calibration_raw_v1",
        status="EXPLORATORY",
        model_identity=bundle.identity(),
        input_paths=(args.gue_test, args.supply_audit),
        parameters={
            "GUE_pair_offset": args.pair_offset,
            "pairs": args.pairs,
            "geometry": {
                key: value
                for key, value in geometry.items()
                if key != "by_split_and_tile"
            },
        },
        seeds={"GUE_selection": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_source_localization_GUE_calibration_v1",
        "status": "CALIBRATION",
        "GUE_pair_offset": args.pair_offset,
        "n_pairs": args.pairs,
        "projected_target_pairs": minimum_target_n,
        "summaries": summaries,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(rows),
        },
        "boundary": (
            "Reserved GUE calibration sets sensitivity and precision thresholds. "
            "It is not evidence about fresh GENCODE loci or promoter features."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_source_localization_calibration_v1",
        status="EXPLORATORY",
        model_identity=bundle.identity(),
        input_paths=(args.gue_test, args.supply_audit, raw_path, raw_manifest),
        parameters={
            "GUE_pair_offset": args.pair_offset,
            "pairs": args.pairs,
            "projected_target_pairs": minimum_target_n,
        },
        seeds={"GUE_selection": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _summaries(rows: list[dict[str, Any]], *, projected_n: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["operation"])].append(row)
    identity = grouped["candidate_identity"]
    summaries: dict[str, Any] = {
        "identity_maximum_absolute_margin_difference": max(
            abs(float(row[endpoint]))
            for row in identity
            for endpoint in ("target_loss", "control_loss", "target_specific_loss")
        ),
        "operations": {},
    }
    for operation in ("mask", "candidate_value_swap", "all_layer_value_swap"):
        operation_rows = grouped[operation]
        values = [float(row["target_specific_loss"]) for row in operation_rows]
        by_tile: dict[int, list[float]] = defaultdict(list)
        for row in operation_rows:
            by_tile[int(row["tile_index"])].append(float(row["target_specific_loss"]))
        tile_stdevs = {
            tile: statistics.stdev(tile_values)
            for tile, tile_values in by_tile.items()
            if len(tile_values) >= 2
        }
        maximum_stdev = max(tile_stdevs.values())
        summaries["operations"][operation] = {
            "rows": len(values),
            "mean": statistics.mean(values),
            "mean_absolute": statistics.mean(abs(value) for value in values),
            "maximum_absolute": max(abs(value) for value in values),
            "minimum_tile_pairs": min(len(value) for value in by_tile.values()),
            "maximum_tile_pairs": max(len(value) for value in by_tile.values()),
            "maximum_tile_stdev": maximum_stdev,
            "projected_80pct_MDE_at_minimum_target_n": (
                2.8 * maximum_stdev / math.sqrt(projected_n)
            ),
        }
    return summaries


if __name__ == "__main__":
    main()
