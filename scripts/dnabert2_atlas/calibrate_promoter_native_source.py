#!/usr/bin/env python
"""Calibrate native promoter-source edits on a reserved GUE slice."""

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
    parser.add_argument("--checkpoint-audit", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=16)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    version = int(config["schema_version"])
    if config.get("status") != "DRAFT":
        raise RuntimeError("native-source calibration requires a DRAFT config")
    preflight = config["calibration_and_preflight"]
    candidate_pairs = select_gue_pairs(
        args.gue_test,
        count=int(preflight["calibration_GUE_candidate_pairs"]),
        offset=int(preflight["calibration_GUE_candidate_offset"]),
        seed=int(config["inference"]["seed"]),
    )
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    rows = build_geometry_qualified_gue_rows(
        bundle,
        candidate_pairs,
        config=config,
        maximum=int(preflight["calibration_GUE_qualified_pairs"]),
    )
    if len(rows) != int(preflight["calibration_GUE_qualified_pairs"]):
        raise RuntimeError(
            f"only {len(rows)} GUE pairs support both native edit families"
        )
    operations = tuple(config["measurement"]["primary_operations"])
    raw_rows = score_native_edit_rows(
        bundle,
        rows,
        operations=operations,
        batch_pairs=args.batch_pairs,
        progress=lambda message: print(f"calibration: {message}", flush=True),
    )
    identity = score_identity_replay(
        bundle,
        rows,
        batch_pairs=args.batch_pairs,
    )
    projected_n = int(config["cohort"]["pairs_per_split"])
    measurements = _measurements(raw_rows, projected_n=projected_n)
    input_paths = (
        args.gue_test,
        args.config,
        args.supply_audit,
        args.checkpoint_audit,
    )
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id=(f"dnabert2_promoter_native_source_calibration_raw_v{version}"),
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "candidate_pool": {
                "offset": preflight["calibration_GUE_candidate_offset"],
                "pairs": preflight["calibration_GUE_candidate_pairs"],
            },
            "qualified_pairs": len(rows),
            "projected_target_pairs": projected_n,
            "native_interventions": config["native_interventions"],
        },
        seeds={"selection_and_edits": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": f"promoter_native_source_calibration_v{version}",
        "status": "CALIBRATION",
        "model_outcomes_computed": True,
        "target_cohort_outcomes_computed": False,
        "checkpoint_revision": bundle.identity()["revision"],
        "GUE_candidate_offset": int(preflight["calibration_GUE_candidate_offset"]),
        "GUE_candidate_pairs": int(preflight["calibration_GUE_candidate_pairs"]),
        "n_pairs": len(rows),
        "pair_ids": [str(row["example_id"]) for row in rows],
        "identity_replay_max_absolute_margin_difference": identity,
        "measurements": measurements,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "rows": len(raw_rows),
        },
        "boundary": (
            "Reserved GUE calibration estimates native-edit sensitivity and "
            "projected precision. It reads no GENCODE target outcome and freezes "
            "no biological conclusion."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id=f"dnabert2_promoter_native_source_calibration_v{version}",
        status="PREFLIGHT",
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={
            "native_interventions": config["native_interventions"],
            "projected_target_pairs": projected_n,
        },
        seeds={"selection_and_edits": int(config["inference"]["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _measurements(
    rows: list[dict[str, Any]],
    *,
    projected_n: int,
) -> dict[str, Any]:
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
            "projected_95pct_CI_half_width_at_target_n": (
                1.96 * stdev / math.sqrt(projected_n)
            ),
        }
    return output


if __name__ == "__main__":
    main()
