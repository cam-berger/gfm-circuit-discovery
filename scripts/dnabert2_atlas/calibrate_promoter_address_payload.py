#!/usr/bin/env python
"""Calibrate fixed-dose promoter K/V surgeries on reserved GUE pairs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from gfmcd.adapters.dnabert2 import load_dnabert2_promoter_classifier
from gfmcd.reconstruction_atlas.artifacts import (
    write_jsonl_bundle,
    write_result_bundle,
)
from gfmcd.reconstruction_atlas.promoter_address_payload import (
    scan_qkv_operations,
)
from gfmcd.reconstruction_atlas.promoter_head_scan import select_gue_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gue-test", required=True)
    parser.add_argument("--supply-audit", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pair-offset", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--source-token-dose", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    supply = json.loads(Path(args.supply_audit).read_text())
    if (
        supply.get("status") != "QUALIFIED"
        or int(supply["token_geometry"]["fixed_source_token_dose"])
        != args.source_token_dose
    ):
        raise RuntimeError("matching qualified supply audit is required")
    pairs = select_gue_pairs(
        args.gue_test,
        count=args.pairs,
        offset=args.pair_offset,
        seed=args.seed,
    )
    bundle = load_dnabert2_promoter_classifier("prom_300_all", device=args.device)
    rows = scan_qkv_operations(
        bundle,
        pairs,
        {
            "candidate_L11_pair": ((11, 2), (11, 5)),
            "all_layer11_heads": tuple((11, head) for head in range(12)),
        },
        (
            "identity",
            "key_permute",
            "value_permute",
            "key_value_joint_permute",
        ),
        source_token_count=args.source_token_dose,
        seed=args.seed,
        batch_pairs=args.pairs,
    )
    summaries = _summaries(rows)
    raw_path, raw_manifest = write_jsonl_bundle(
        args.raw_out,
        rows,
        experiment_id="dnabert2_promoter_address_payload_calibration_raw_v1",
        status="EXPLORATORY",
        model_identity=bundle.identity(),
        input_paths=(args.gue_test, args.supply_audit),
        parameters={
            "GUE_pair_offset": args.pair_offset,
            "pairs": args.pairs,
            "source_token_dose": args.source_token_dose,
        },
        seeds={"selection_and_surgery": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    payload = {
        "schema_version": 1,
        "study": "dnabert2_biological_reconstruction_atlas",
        "assay": "promoter_address_payload_GUE_calibration_v1",
        "status": "CALIBRATION",
        "GUE_pair_offset": args.pair_offset,
        "n_pairs": args.pairs,
        "source_token_dose": args.source_token_dose,
        "summaries": summaries,
        "raw": {
            "path": str(raw_path),
            "manifest": str(raw_manifest),
            "n_rows": len(rows),
        },
        "boundary": (
            "Reserved GUE calibration sets numerical and sensitivity thresholds "
            "only; it is not evidence about fresh GENCODE loci."
        ),
    }
    write_result_bundle(
        args.out,
        payload,
        experiment_id="dnabert2_promoter_address_payload_calibration_v1",
        status="EXPLORATORY",
        model_identity=bundle.identity(),
        input_paths=(args.gue_test, args.supply_audit, raw_path, raw_manifest),
        parameters={
            "GUE_pair_offset": args.pair_offset,
            "pairs": args.pairs,
            "source_token_dose": args.source_token_dose,
        },
        seeds={"selection_and_surgery": args.seed},
        command=sys.argv,
        repo=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _summaries(
    rows: list[dict[str, float | str]],
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["head_pair"]), str(row["operation"]))].append(row)
    output: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (head_pair, operation), group in sorted(grouped.items()):
        output[head_pair][operation] = {
            f"{endpoint}_{summary}": value
            for endpoint in (
                "target_loss",
                "control_loss",
                "target_specific_loss",
            )
            for summary, value in _endpoint_summary(
                [float(row[endpoint]) for row in group]
            ).items()
        }
    return dict(output)


def _endpoint_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "mean_absolute": statistics.mean(abs(value) for value in values),
        "maximum_absolute": max(abs(value) for value in values),
    }


if __name__ == "__main__":
    main()
