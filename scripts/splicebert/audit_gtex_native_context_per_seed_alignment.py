#!/usr/bin/env python
"""Audit the frozen per-seed alignment gates without recomputing outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402


PRIMARY_RANK = "4"
MAX_Q = 0.05
MIN_RATIO = 0.50


def causal_direction_pass(summary: dict[str, Any]) -> bool:
    return bool(
        summary["estimate"] > 0
        and summary["ci_low"] > 0
        and summary["q_bh_16_cross_basis_effects"] <= MAX_Q
        and summary["ratio_to_full_mlp"]["ci_low"] >= MIN_RATIO
        and summary["vs_random"]["ci_low"] > 0
        and summary["vs_random"][
            "q_bh_16_cross_basis_random_contrasts"
        ]
        <= MAX_Q
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result_path = Path(args.result).expanduser().resolve()
    result = json.loads(result_path.read_text())
    if (
        result.get("status") != "COMPLETE"
        or result.get("multiplicity_families", {}).get(
            "primary_cross_basis_effects"
        )
        != 16
        or result.get("instrument_gate", {}).get(
            "numpy_composition_at_most_1e_5"
        )
        is not True
    ):
        raise RuntimeError("alignment result is not auditable")
    corrected: dict[str, Any] = {}
    for endpoint in ("donor", "acceptor"):
        own = result["own_checkpoint_subspaces"]["summaries"][endpoint][
            PRIMARY_RANK
        ]
        aligned = result["cross_seed_basis_transport"]["aligned"][endpoint][
            PRIMARY_RANK
        ]
        aligned_minus_raw = {
            row["direction"]: row["summary"]
            for row in result["aligned_minus_raw_basis"]
            if row["endpoint"] == endpoint
            and str(row["rank"]) == PRIMARY_RANK
        }
        own_pass = all(
            own[direction]["ratio_to_full_mlp"]["ci_low"] >= MIN_RATIO
            for direction in ("damage", "recovery")
        )
        alignment_pass = all(
            causal_direction_pass(aligned[direction])
            for direction in ("damage", "recovery")
        )
        raw_contrast_pass = all(
            aligned_minus_raw[direction]["ci_low"] > 0
            and aligned_minus_raw[direction][
                "q_bh_8_aligned_minus_raw_basis"
            ]
            <= MAX_Q
            for direction in ("damage", "recovery")
        )
        corrected[endpoint] = {
            "cross_seed_decision_rank": int(PRIMARY_RANK),
            "own_basis_gate": own_pass,
            "aligned_effect_and_random_gate": alignment_pass,
            "aligned_minus_raw_gate": raw_contrast_pass,
            "rotated_shared_lowrank_code": (
                own_pass and alignment_pass and raw_contrast_pass
            ),
            "aligned_instance_vector_transport": result["decision"][endpoint][
                "aligned_instance_vector_transport"
            ],
        }
    output = guard_output_path(Path(args.out).resolve())
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CONTEXT_PER_SEED_ALIGNMENT_V1.md"
    )
    audit = {
        "study": "native GTEx per-seed final-MLP alignment",
        "stage": "post-outcome summary-verdict correction audit",
        "status": "PASS",
        "protocol": str(protocol.relative_to(REPO)),
        "source_result": str(result_path.relative_to(REPO)),
        "scope": (
            "Applies frozen rank-4 decision gates to existing recorded summaries; "
            "does not recompute or alter any outcome."
        ),
        "original_decision": result["decision"],
        "corrected_decision": corrected,
        "all_corrected_gates_pass": all(
            row["rotated_shared_lowrank_code"]
            and row["aligned_instance_vector_transport"]
            for row in corrected.values()
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_per_seed_alignment_audit_v1",
        sys.argv,
        repo=REPO,
        status="AUDIT",
        config={
            "decision_rank": int(PRIMARY_RANK),
            "max_q": MAX_Q,
            "minimum_ratio": MIN_RATIO,
        },
        notes=[
            "This audit changes only the summary verdict produced by an implementation bug.",
            "Every estimate, interval, p-value, q-value, and intervention array remains unchanged.",
        ],
    )
    manifest.inputs.extend([file_record(protocol), file_record(result_path)])
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(json.dumps(audit["corrected_decision"], indent=2, sort_keys=True))
    print(f"PASS: saved -> {output}")


if __name__ == "__main__":
    main()
