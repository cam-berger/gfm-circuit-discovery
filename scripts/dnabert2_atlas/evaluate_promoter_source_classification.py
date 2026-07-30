#!/usr/bin/env python
"""Evaluate hard-label performance for the nominated promoter source tile."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from gfmcd.adapters.dnabert2 import load_dnabert2_promoter_classifier
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.gates import frozen_config_sha256
from gfmcd.reconstruction_atlas.promoter_head_scan import read_jsonl
from gfmcd.reconstruction_atlas.promoter_performance import (
    summarize_masked_classification,
)
from gfmcd.reconstruction_atlas.promoter_source_execution import (
    score_equal_dose_masked_tile,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated-cohort", required=True)
    parser.add_argument("--cohort-manifest", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--replication-report", required=True)
    parser.add_argument("--analysis-spec", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-pairs", type=int, default=32)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = frozen_config_sha256(args.config)
    supply = json.loads(Path(args.supply_audit).read_text())
    preflight = json.loads(Path(args.preflight_report).read_text())
    replication = json.loads(Path(args.replication_report).read_text())
    spec = json.loads(Path(args.analysis_spec).read_text())
    if (
        supply.get("status") != "QUALIFIED"
        or not preflight.get("qualified")
        or preflight.get("config_sha256") != config_hash
        or replication.get("status") != "SEALED"
        or replication.get("config_sha256") != config_hash
        or not replication["replicated_behavioral_localization"]["qualified"]
        or spec.get("status") != "FROZEN_POSTHOC_DESCRIPTIVE"
    ):
        raise RuntimeError(
            "qualified supply/preflight, replicated localization, and frozen "
            "post-hoc specification are required"
        )
    interval = replication["replicated_behavioral_localization"][
        "nominal_relative_interval"
    ]
    tile_width = int(config["tile_geometry"]["nominal_tile_width_nt"])
    nominal_start = int(interval[0]) + int(config["tile_geometry"]["TSS_index"])
    if (
        interval != spec["nominated_relative_interval"]
        or nominal_start % tile_width
        or spec["metrics"]
        != [
            "accuracy",
            "AUROC",
            "promoter_to_non_promoter_label_flips",
        ]
    ):
        raise RuntimeError("analysis specification does not match the nominated tile")
    tile_index = nominal_start // tile_width
    all_rows = read_jsonl(args.annotated_cohort)
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    raw_rows = []
    summaries: dict[str, dict[str, Any]] = {}
    selected_hashes = {}
    for split_index, split in enumerate(("development", "confirmation")):
        rows = _select_rows(all_rows, supply, split=split)
        pairs = [
            (
                str(row["example_id"]),
                str(row["target"]["clean_sequence"]),
                str(row["control"]["clean_sequence"]),
            )
            for row in rows
        ]
        clusters = {
            str(row["example_id"]): str(row["homology_cluster"]) for row in rows
        }
        scored = score_equal_dose_masked_tile(
            bundle,
            pairs,
            tile_index=tile_index,
            tile_width_nt=tile_width,
            token_dose=int(
                config["tile_geometry"]["fixed_BPE_token_dose_per_sequence"]
            ),
            maximum_cover_nt=int(config["tile_geometry"]["maximum_actual_cover_nt"]),
            maximum_cover_difference_nt=int(
                config["tile_geometry"][
                    "maximum_target_control_cover_length_difference_nt"
                ]
            ),
            batch_pairs=args.batch_pairs,
            progress=lambda message, split=split: print(
                f"{split}: {message}", flush=True
            ),
        )
        for row in scored:
            row["split"] = split
            row["homology_cluster"] = clusters[str(row["pair_id"])]
            row["target_clean_prediction"] = int(float(row["target_clean_margin"]) > 0)
            row["target_masked_prediction"] = int(
                float(row["target_masked_margin"]) > 0
            )
            row["control_clean_prediction"] = int(
                float(row["control_clean_margin"]) > 0
            )
            row["control_masked_prediction"] = int(
                float(row["control_masked_margin"]) > 0
            )
        raw_rows.extend(scored)
        summaries[split] = summarize_masked_classification(
            scored,
            bootstrap_draws=int(spec["bootstrap_draws"]),
            seed=int(spec["seed"]) ^ (0x1010 * (split_index + 1)),
        )
        selected_hashes[split] = hashlib.sha256(
            ("\n".join(str(row["pair_id"]) for row in scored) + "\n").encode()
        ).hexdigest()
    pooled = summarize_masked_classification(
        raw_rows,
        bootstrap_draws=int(spec["bootstrap_draws"]),
        seed=int(spec["seed"]) ^ 0x3030,
    )
    input_paths = (
        args.annotated_cohort,
        args.cohort_manifest,
        args.supply_audit,
        args.config,
        args.preflight_report,
        args.replication_report,
        args.analysis_spec,
    )
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        raw_rows,
        experiment_id="dnabert2_promoter_source_classification_performance_raw_v1",
        status="DEVELOPMENT",
        model_identity=bundle.identity(),
        input_paths=input_paths,
        parameters={
            "config_sha256": config_hash,
            "tile_index": tile_index,
            "posthoc": True,
        },
        seeds={"bootstrap": int(spec["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_source_classification_performance_v1",
        "status": "POSTHOC_DESCRIPTIVE",
        "config_sha256": config_hash,
        "tile": {
            "tile_index": tile_index,
            "nominal_relative_interval": interval,
            "operation": "two-BPE-token post-tokenization [MASK]",
        },
        "metric_scope": (
            "Balanced real-GENCODE-TSS versus same-locus pseudo-TSS classification "
            "among pairs eligible for the previously nominated tile; margin > 0 is "
            "the promoter call."
        ),
        "selected_id_sha256": selected_hashes,
        "per_split": summaries,
        "pooled_descriptive": pooled,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(raw_rows),
        },
        "interpretation_boundary": (
            "These are post-hoc secondary performance endpoints for an already "
            "nominated intervention. They do not independently validate the region, "
            "identify a motif, or estimate performance on the original GUE test set."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_source_classification_performance_v1",
        status="DEVELOPMENT",
        model_identity=bundle.identity(),
        input_paths=(*input_paths, raw_path, raw_manifest),
        parameters={
            "config_sha256": config_hash,
            "tile_index": tile_index,
            "batch_pairs": args.batch_pairs,
            "posthoc": True,
        },
        seeds={"bootstrap": int(spec["seed"])},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _select_rows(
    all_rows: list[dict[str, Any]],
    supply: dict[str, Any],
    *,
    split: str,
) -> list[dict[str, Any]]:
    by_id = {str(row["example_id"]): row for row in all_rows if row["split"] == split}
    ids = [str(value) for value in supply["selection"]["selected_example_ids"][split]]
    if any(value not in by_id for value in ids):
        raise RuntimeError(f"annotated cohort is missing selected {split} rows")
    return [by_id[value] for value in ids]


if __name__ == "__main__":
    main()
