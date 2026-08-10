#!/usr/bin/env python
"""Analyze the sealed Phase-III P3 register-mediation target model-free."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_register_mediation import (  # noqa: E402
    CONFIRMATORY_BOOTSTRAP_DRAWS,
    INFERENCE_SEED,
    MEDIATION_TARGET_SCHEMA,
    RESERVE_BLOCKS,
    RESERVE_CELLS,
    analyze_compact_mediation,
    candidate_freeze_sha256,
    validate_candidate_freeze,
    validate_compact_mediation,
    validate_compact_relocation_binding,
)
from gfmcd.splicing.pbc_register_stage import verify_register_target_seal  # noqa: E402


ANALYSIS_SCHEMA = "gfmcd.pbc.p3_register_mediation_analysis.v1"


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(path)
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    if output == manifest:
        raise ValueError("analysis and manifest paths must be distinct")
    return output, manifest


def analyze_payload(
    payload: Mapping[str, object],
    *,
    target_sha256: str,
    seed: int = INFERENCE_SEED,
    n_bootstrap: int = CONFIRMATORY_BOOTSTRAP_DRAWS,
    enforce_confirmatory_contract: bool = True,
) -> dict[str, object]:
    """Validate all bindings and run the frozen absolute-effect analysis."""

    if payload.get("schema") != MEDIATION_TARGET_SCHEMA or payload.get("status") != "COMPLETE":
        raise ValueError("mediation target is incomplete or has the wrong schema")
    design = payload.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("mediation target lacks its frozen design")
    freeze = design.get("candidate_freeze")
    if not isinstance(freeze, Mapping):
        raise ValueError("mediation target lacks candidate_freeze")
    declared_freeze_hash = str(design.get("candidate_freeze_sha256") or "")
    observed_freeze_hash = candidate_freeze_sha256(freeze)
    if declared_freeze_hash != observed_freeze_hash:
        raise ValueError("candidate freeze digest does not match its payload")

    freeze_validation = validate_candidate_freeze(freeze)
    lattice_validation = validate_compact_mediation(
        payload,
        freeze=freeze,
        enforce_reserve_contract=enforce_confirmatory_contract,
    )
    relocation_binding = validate_compact_relocation_binding(payload)
    cell_ledger = payload.get("cell_ledger")
    assert isinstance(cell_ledger, list)
    unique_cells = {
        (
            str(row["block_id"]),
            int(row["donor_phase"]),
            int(row["acceptor_phase"]),
        )
        for row in cell_ledger
    }
    mechanics = payload.get("mechanics")
    target_design_seed = int(design.get("inference_seed", -1))
    target_design_draws = int(design.get("bootstrap_draws", -1))
    reserve_contract = bool(
        payload.get("stage") == "mediation"
        and payload.get("split_role") == "phase3_reserve"
        and int(payload.get("reserve_blocks", -1)) == RESERVE_BLOCKS
        and int(payload.get("reserve_cells", -1)) == RESERVE_CELLS
        and int(payload.get("n_blocks", -1)) == RESERVE_BLOCKS
        and int(payload.get("n_cells", -1)) == RESERVE_CELLS
        and int(lattice_validation.get("n_blocks", -1)) == RESERVE_BLOCKS
        and int(lattice_validation.get("n_cells", -1)) == RESERVE_CELLS
        and len(unique_cells) == RESERVE_CELLS
    )
    executed_hook_contract = bool(
        isinstance(mechanics, Mapping)
        and mechanics.get("passed") is True
        and mechanics.get("same_sequence_patching") is True
        and mechanics.get("same_corruption_state_patching") is True
        and list(mechanics.get("full_token_support", [])) == [393, 513]
        and all(
            float(mechanics.get(name, float("inf"))) <= 1e-8
            for name in (
                "raw_relocation_replay_intact_margin_max_abs",
                "raw_relocation_replay_corrupt_margin_max_abs",
                "raw_relocation_replay_coupling_max_abs",
            )
        )
        and len(str(mechanics.get("actual_hook_tensor_stream_sha256", ""))) == 64
        and int(mechanics.get("actual_patch_forward_events", 0)) > 0
    )
    confirmatory_contract = {
        "seed": int(seed),
        "required_seed": INFERENCE_SEED,
        "bootstrap_draws": int(n_bootstrap),
        "required_bootstrap_draws": CONFIRMATORY_BOOTSTRAP_DRAWS,
        "target_design_seed": target_design_seed,
        "target_design_bootstrap_draws": target_design_draws,
        "stage": payload.get("stage"),
        "split_role": payload.get("split_role"),
        "n_blocks": lattice_validation.get("n_blocks"),
        "required_blocks": RESERVE_BLOCKS,
        "unique_lattice_cells": len(unique_cells),
        "required_lattice_cells": RESERVE_CELLS,
        "reserve_contract": reserve_contract,
        "executed_hook_contract": executed_hook_contract,
        "passed": bool(
            int(seed) == INFERENCE_SEED
            and int(n_bootstrap) == CONFIRMATORY_BOOTSTRAP_DRAWS
            and target_design_seed == INFERENCE_SEED
            and target_design_draws == CONFIRMATORY_BOOTSTRAP_DRAWS
            and reserve_contract
            and executed_hook_contract
        ),
    }
    capable = bool(relocation_binding["path_instrument_capable"])
    if enforce_confirmatory_contract:
        capable = capable and bool(confirmatory_contract["passed"])
    result = analyze_compact_mediation(
        payload,
        freeze=freeze,
        seed=seed,
        n_bootstrap=n_bootstrap,
        relocation_authorized=bool(relocation_binding["passed"]),
        path_instrument_capable=capable,
        enforce_reserve_contract=enforce_confirmatory_contract,
    )
    decision = result["decision"]
    return {
        "schema": ANALYSIS_SCHEMA,
        "status": decision["code"],
        "stage": "mediation",
        "target_sha256": target_sha256,
        "candidate_freeze_sha256": observed_freeze_hash,
        "freeze_validation": freeze_validation,
        "lattice_validation": lattice_validation,
        "relocation_binding": relocation_binding,
        "confirmatory_contract": confirmatory_contract,
        **result,
        "reporting_firewall": {
            "absolute_effects_primary": True,
            "fractional_mediation_reported": False,
        },
        "evidence_firewall": {
            "output_mediation_adjudicated": True,
            "selector_claim_adjudicated": False,
            "attention_router_claim_adjudicated": False,
            "directed_edge_claim_adjudicated": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    target_path = Path(args.target).resolve()
    output_path, manifest_path = output_paths(Path(args.out).resolve())
    payload = json.loads(target_path.read_text())
    sources = payload.get("sources")
    seal_record = sources.get("reserve_seal") if isinstance(sources, Mapping) else None
    if not isinstance(seal_record, Mapping):
        raise RuntimeError("confirmatory mediation target lacks its original reserve seal")
    seal_path = Path(str(seal_record.get("path", ""))).resolve()
    if (
        not seal_path.is_file()
        or seal_record.get("sha256") != sha256_file(seal_path)
        or seal_record.get("size_bytes") != seal_path.stat().st_size
    ):
        raise RuntimeError("mediation reserve-seal source record changed")
    seal = json.loads(seal_path.read_text())
    verify_register_target_seal(seal)
    if target_path != Path(str(seal["mediation"]["target_output_path"])).resolve():
        raise RuntimeError("mediation target path differs from the pre-reserve seal")
    if output_path != Path(str(seal["mediation"]["analysis_output_path"])).resolve():
        raise RuntimeError("mediation analysis path differs from the pre-reserve seal")
    analysis = analyze_payload(payload, target_sha256=sha256_file(target_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        experiment_id="pbc-p3-register-mediation-analysis",
        command=sys.argv,
        repo=REPO,
        status="ANALYSIS",
        config={
            "schema": ANALYSIS_SCHEMA,
            "stage": "mediation",
            "seed": INFERENCE_SEED,
            "bootstrap_draws": CONFIRMATORY_BOOTSTRAP_DRAWS,
            "effect_units": "raw MLM logit-difference units",
        },
        seeds={"whole_block_bootstrap": INFERENCE_SEED},
        inputs=[file_record(target_path)],
        outputs=[file_record(output_path)],
        notes=[
            "Model-free analysis; D_path, R_path, and Q_path use separate 31-set null families.",
            "No fractional or percentage mediation estimate is emitted.",
        ],
    )
    manifest.write(manifest_path)


if __name__ == "__main__":
    main()
