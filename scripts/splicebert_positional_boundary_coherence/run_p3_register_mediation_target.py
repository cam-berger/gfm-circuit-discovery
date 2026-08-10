#!/usr/bin/env python
"""Execute one deterministic shard of the pre-sealed P3 mediation reserve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import load_adapter, pick_device  # noqa: E402
from gfmcd.splicing.pbc_register_mediation import (  # noqa: E402
    MEDIATION_TARGET_SCHEMA,
    candidate_freeze_sha256,
    compact_mediation_rows,
    validate_candidate_freeze,
    validate_compact_mediation,
)
from gfmcd.splicing.pbc_register_mediation_runner import (  # noqa: E402
    MEDIATION_SHARD_SCHEMA,
    evaluate_mediation_lattice,
)
from gfmcd.splicing.pbc_register_stage import (  # noqa: E402
    RESERVE_BLOCKS,
    RESERVE_CELLS,
    verify_register_target_seal,
)
from run_p3_register_target import deterministic_shard, reserve_cells  # noqa: E402


RELOCATION_TARGET_SCHEMA = "gfmcd.pbc.p3_register_target.v1"
RELOCATION_ANALYSIS_SCHEMA = "gfmcd.pbc.p3_register_analysis.v1"
RELOCATION_AUDIT_SCHEMA = "gfmcd.pbc.p3_register_audit.v1"


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    return output, manifest


def validate_relocation_sources(
    seal: Mapping[str, Any],
    *,
    target_path: Path,
    analysis_path: Path,
    audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail before model loading unless relocation passed on this sealed reserve."""

    expected = seal["relocation_outputs"]
    if (
        target_path != Path(str(expected["target"])).resolve()
        or analysis_path != Path(str(expected["analysis"])).resolve()
        or audit_path != Path(str(expected["audit"])).resolve()
    ):
        raise RuntimeError("relocation inputs differ from the original pre-reserve seal")
    target = json.loads(target_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    audit = json.loads(audit_path.read_text())
    target_hash = sha256_file(target_path)
    status = str(analysis.get("status", ""))
    if (
        target.get("schema") != RELOCATION_TARGET_SCHEMA
        or target.get("status") != "COMPLETE"
        or target.get("stage") != "phase3_reserve"
        or int(target.get("n_blocks", -1)) != RESERVE_BLOCKS
        or int(target.get("n_cells", -1)) != RESERVE_CELLS
    ):
        raise RuntimeError("relocation target is not the complete sealed reserve")
    if (
        analysis.get("schema") != RELOCATION_ANALYSIS_SCHEMA
        or not status.startswith("P3_REGISTER_RELOCATION_SUPPORTED")
        or analysis.get("target_sha256") != target_hash
    ):
        raise RuntimeError("relocation analysis does not authorize mediation")
    if (
        audit.get("schema") != RELOCATION_AUDIT_SCHEMA
        or audit.get("status") != "PASS"
        or audit.get("target_sha256") != target_hash
    ):
        raise RuntimeError("relocation independent audit did not pass")
    return target, {
        "relocation_target": file_record(target_path),
        "relocation_analysis": file_record(analysis_path),
        "relocation_audit": file_record(audit_path),
        "relocation_status": status,
    }


def execute_target(
    *,
    seal: Mapping[str, Any],
    cohort: Mapping[str, Any],
    candidate_freeze: Mapping[str, Any],
    relocation_target: Mapping[str, Any],
    relocation_records: Mapping[str, Any],
    adapter: Any,
    batch_size: int,
    shard_index: int,
) -> dict[str, Any]:
    verify_register_target_seal(seal)
    validate_candidate_freeze(candidate_freeze)
    all_cells = reserve_cells(
        cohort,
        split_role="phase3_reserve",
        expected_blocks=RESERVE_BLOCKS,
    )
    shard_count = int(seal["mediation"]["shard_count"])
    cells = deterministic_shard(
        all_cells, shard_index=shard_index, shard_count=shard_count
    )
    rows, mechanics = evaluate_mediation_lattice(
        adapter,
        cells,
        candidate_freeze=candidate_freeze,
        relocation_rows=relocation_target["rows"],
        p3_bins=tuple(int(value) for value in seal["design"]["p3_bins"]),
        p3_attenuation_dose=float(seal["design"]["p3_attenuation_dose"]),
        batch_size=int(batch_size),
        direction=str(seal["design"]["direction"]),
    )
    block_ids = sorted({cell.block_id for cell in cells})
    model_metadata = adapter.metadata()
    raw = {
        "schema": MEDIATION_SHARD_SCHEMA,
        "status": "COMPLETE",
        "stage": "mediation",
        "split_role": "phase3_reserve",
        "mode": seal["mode"],
        "seal_sha256": seal["seal_sha256"],
        "authorization": {
            "relocation_target_sha256": relocation_records["relocation_target"]["sha256"],
            "relocation_analysis_sha256": relocation_records["relocation_analysis"]["sha256"],
            "relocation_audit_sha256": relocation_records["relocation_audit"]["sha256"],
            "relocation_status": relocation_records["relocation_status"],
            "relocation_audit_status": "PASS",
            "path_instrument_capable": True,
            "candidate_freeze_audit_status": "PASS",
        },
        "design": {
            "candidate_freeze": dict(candidate_freeze),
            "candidate_freeze_sha256": candidate_freeze_sha256(candidate_freeze),
            "candidate_freeze_file_sha256": seal["candidate_bundle"][
                "candidate_freeze_sha256"
            ],
            "token_support": seal["mediation"]["token_support"],
            "same_state_operations": seal["mediation"]["same_state_operations"],
            "hook_sites": seal["mediation"]["hook_sites"],
            "p3_attenuation_order": seal["mediation"]["p3_attenuation_order"],
            "inference_seed": seal["mediation"]["inference_seed"],
            "bootstrap_draws": seal["mediation"]["bootstrap_draws"],
        },
        "sources": {
            "relocation_target": relocation_records["relocation_target"],
            "relocation_analysis": relocation_records["relocation_analysis"],
            "relocation_audit": relocation_records["relocation_audit"],
            **{label: dict(record) for label, record in seal["bindings"].items()},
        },
        "model": dict(getattr(model_metadata, "__dict__", {})),
        "reserve_blocks": RESERVE_BLOCKS,
        "reserve_cells": RESERVE_CELLS,
        "n_blocks": len(block_ids),
        "n_cells": len(cells),
        "shard": {
            "index": int(shard_index),
            "count": shard_count,
            "block_ids": block_ids,
        },
        "mechanics": mechanics,
        "rows": rows,
    }
    if shard_count != 1:
        return raw
    compact = compact_mediation_rows(rows, freeze=candidate_freeze)
    result = {
        key: value
        for key, value in raw.items()
        if key not in {"rows", "shard"}
    }
    result.update(compact)
    result["schema"] = MEDIATION_TARGET_SCHEMA
    result["mechanics"] = {
        **mechanics,
        "normalized_provenance_ledger_sha256": compact["normalization"][
            "provenance_ledger_sha256"
        ],
    }
    validate_compact_mediation(
        result,
        freeze=candidate_freeze,
        enforce_reserve_contract=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--relocation-target", required=True)
    parser.add_argument("--relocation-analysis", required=True)
    parser.add_argument("--relocation-audit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=36)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output, manifest_path = output_paths(args.out)
    seal_path = Path(args.seal).expanduser().resolve()
    cohort_path = Path(args.cohort).expanduser().resolve()
    candidate_path = Path(args.candidate_freeze).expanduser().resolve()
    relocation_target_path = Path(args.relocation_target).expanduser().resolve()
    relocation_analysis_path = Path(args.relocation_analysis).expanduser().resolve()
    relocation_audit_path = Path(args.relocation_audit).expanduser().resolve()
    seal = json.loads(seal_path.read_text())
    # All static bindings and paths fail before the reserve or checkpoint is read.
    verify_register_target_seal(seal)
    mediation = seal["mediation"]
    shard_count = int(mediation["shard_count"])
    if not 0 <= args.shard_index < shard_count:
        raise RuntimeError("mediation shard index is outside the frozen range")
    if output != Path(str(mediation["shard_paths"][args.shard_index])).resolve():
        raise RuntimeError("mediation output path differs from the pre-reserve seal")
    bindings = seal["bindings"]
    if cohort_path != Path(str(bindings["cohort"]["path"])).resolve():
        raise RuntimeError("mediation cohort differs from the pre-reserve seal")
    if candidate_path != Path(str(bindings["candidate_freeze"]["path"])).resolve():
        raise RuntimeError("candidate freeze differs from the pre-reserve seal")
    relocation_target, relocation_records = validate_relocation_sources(
        seal,
        target_path=relocation_target_path,
        analysis_path=relocation_analysis_path,
        audit_path=relocation_audit_path,
    )
    cohort = json.loads(cohort_path.read_text())
    candidate_freeze = json.loads(candidate_path.read_text())
    validate_candidate_freeze(candidate_freeze)
    all_cells = reserve_cells(
        cohort, split_role="phase3_reserve", expected_blocks=RESERVE_BLOCKS
    )
    deterministic_shard(
        all_cells, shard_index=args.shard_index, shard_count=shard_count
    )
    model_path = Path(args.model).expanduser().resolve()
    checkpoint = model_path / "pytorch_model.bin"
    if checkpoint != Path(str(bindings["checkpoint"]["path"])).resolve():
        raise RuntimeError("mediation checkpoint differs from the pre-reserve seal")
    adapter = load_adapter(str(model_path), pick_device(args.device), eager=True)
    result = execute_target(
        seal=seal,
        cohort=cohort,
        candidate_freeze=candidate_freeze,
        relocation_target=relocation_target,
        relocation_records=relocation_records,
        adapter=adapter,
        batch_size=args.batch,
        shard_index=args.shard_index,
    )
    result["sources"]["reserve_seal"] = file_record(seal_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-p3-register-mediation-target-v1",
        sys.argv,
        repo=REPO,
        status="TARGET_SCORED_UNANALYZED",
        config={
            "stage": "mediation",
            "n_blocks": result["n_blocks"],
            "n_cells": result["n_cells"],
            "shard": result["shard"],
            "candidate_count": result["design"]["candidate_freeze"]["selection"][
                "candidate_count"
            ],
            "token_support": result["design"]["token_support"],
        },
        seeds={"confirmatory_inference": int(mediation["inference_seed"])},
    )
    for path in (
        seal_path,
        cohort_path,
        candidate_path,
        relocation_target_path,
        relocation_analysis_path,
        relocation_audit_path,
        checkpoint,
    ):
        manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "shard": result["shard"],
                "n_rows": len(result.get("rows", [])),
                "n_endpoint_rows": len(result.get("endpoint_rows", [])),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
