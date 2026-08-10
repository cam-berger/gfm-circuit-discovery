"""Executed-looking synthetic candidate bundle for strict reserve-seal tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from gfmcd.provenance import file_record
from gfmcd.splicing.pbc_register_candidates import (
    CANDIDATE_SCREEN_SCHEMA,
    build_candidate_freeze,
    component_universe,
)
from gfmcd.splicing.pbc_register_stage import (
    REGISTER_MANIFEST_CHAIN_SCHEMA,
    REGISTER_PREREQUISITE_ARTIFACTS,
)


REPO = Path(__file__).resolve().parents[1]


def prerequisite_manifest_chain():
    return {
        "schema": REGISTER_MANIFEST_CHAIN_SCHEMA,
        "stage": "target",
        "status": "PASS",
        "checks": {"synthetic_fixture_chain": True},
        "artifacts": {
            label: {
                "artifact_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "created_at_utc": "2026-08-10T00:00:00+00:00",
                "git_commit": "c" * 40,
            }
            for label in REGISTER_PREREQUISITE_ARTIFACTS
        },
    }


def _audit_module():
    path = (
        REPO
        / "scripts/splicebert_positional_boundary_coherence"
        / "audit_p3_register_candidate_freeze.py"
    )
    spec = importlib.util.spec_from_file_location("pbc_candidate_fixture_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_screen(blocks: int = 120):
    universe = list(component_universe())
    priority = {"L0H0": 2.0, "L1H0": 1.9}
    singletons = []
    for index, component in enumerate(universe):
        component_id = component["component_id"]
        effect = priority.get(component_id, -1.0 - index * 0.001)
        per_block = [
            {
                "block_id": f"fixture-b{block:03d}",
                "chrom": f"chr{1 + block % 2}",
                "strand": "+" if (block // 2) % 2 == 0 else "-",
                "D_path": effect + (block % 2) * 0.0001,
                "R_path": 0.2 + (block % 2) * 0.0001,
                "Q_path": 0.25 + (block % 2) * 0.0001,
            }
            for block in range(blocks)
        ]
        singletons.append(
            {
                "component_id": component_id,
                "same_state_scrub_loss_in_omega3": sum(row["D_path"] for row in per_block) / blocks,
                "same_state_restore_gain_in_omega3": sum(row["R_path"] for row in per_block) / blocks,
                "same_state_sufficiency_gain_in_omega3": sum(row["Q_path"] for row in per_block) / blocks,
                "path_output_delta_norms": {
                    "d_path": 1.0,
                    "r_path": 1.0,
                    "q_path": 1.0,
                },
                "qk_pre_softmax_register_change": 0.1 if component["kind"] == "head" else None,
                "attention_post_softmax_register_change": 0.05 if component["kind"] == "head" else None,
                "per_block": per_block,
            }
        )
    ranking = [
        row["component_id"]
        for row in sorted(
            singletons,
            key=lambda row: (-row["same_state_scrub_loss_in_omega3"], row["component_id"]),
        )
    ]
    prefixes = []
    for size in range(1, 9):
        effect = 0.6 if size == 1 else 0.85 + size * 0.001
        per_block = [
            {
                "block_id": f"fixture-b{block:03d}",
                "chrom": f"chr{1 + block % 2}",
                "strand": "+" if (block // 2) % 2 == 0 else "-",
                "D_path": effect + (block % 2) * 0.0001,
                "R_path": 0.3 + (block % 2) * 0.0001,
                "Q_path": 0.35 + (block % 2) * 0.0001,
            }
            for block in range(blocks)
        ]
        prefixes.append(
            {
                "prefix_size": size,
                "component_ids": ranking[:size],
                "D_path": sum(row["D_path"] for row in per_block) / blocks,
                "R_path": sum(row["R_path"] for row in per_block) / blocks,
                "Q_path": sum(row["Q_path"] for row in per_block) / blocks,
                "per_block": per_block,
            }
        )
    return {
        "schema": CANDIDATE_SCREEN_SCHEMA,
        "status": "COMPLETE",
        "stage": "development",
        "n_blocks": blocks,
        "n_cells": blocks * 9,
        "token_support": [393, 513],
        "candidate_universe": universe,
        "baseline": {"mean_raw_omega3": 1.0},
        "singletons": singletons,
        "joint_prefixes": prefixes,
        "tensor_stream": {
            "sha256": "a" * 64,
            "patch_events": 100,
            "same_sequence_only": True,
            "same_corruption_state_only": True,
            "full_token_support_only": True,
        },
        "evidence_levels": {
            "qk_pre_softmax": "diagnostic_only",
            "attention_post_softmax": "diagnostic_only",
            "component_output": "measured_output_delta_norm",
            "behavior_same_state_patch": "candidate_rank_and_joint_prefix",
        },
    }


def candidate_bindings(tmp_path: Path):
    screen = candidate_screen()
    screen_path = tmp_path / "candidate_screen.json"
    screen_path.write_text(json.dumps(screen) + "\n")
    freeze, path_mde = build_candidate_freeze(
        screen, screen_record=file_record(screen_path)
    )
    freeze_path = tmp_path / "candidate_freeze.json"
    mde_path = tmp_path / "path_mde.json"
    freeze_path.write_text(json.dumps(freeze) + "\n")
    mde_path.write_text(json.dumps(path_mde) + "\n")
    audit = _audit_module().audit_payloads(
        screen,
        freeze,
        path_mde,
        screen_path=screen_path,
        freeze_path=freeze_path,
        path_mde_path=mde_path,
    )
    assert audit["status"] == "PASS"
    audit_path = tmp_path / "candidate_freeze_audit.json"
    audit_path.write_text(json.dumps(audit) + "\n")
    return {
        "candidate_screen": file_record(screen_path),
        "candidate_freeze": file_record(freeze_path),
        "candidate_freeze_audit": file_record(audit_path),
        "path_mde": file_record(mde_path),
    }
