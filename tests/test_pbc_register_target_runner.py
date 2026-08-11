from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from gfmcd.provenance import file_record
from gfmcd.splicing.pbc_register_stage import (
    PHASE2_ANALYSIS_SCHEMA,
    PHASE2_AUDIT_SCHEMA,
    REGISTER_PREFLIGHT_AUDIT_SCHEMA,
    make_register_target_seal,
)
from pbc_candidate_fixture import candidate_bindings, prerequisite_manifest_chain


REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = REPO / "scripts/splicebert_positional_boundary_coherence" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = _load("run_p3_register_target")
FREEZE = _load("freeze_p3_register_target")
ASSEMBLE = _load("assemble_p3_register_target")


def _cohort(blocks: int = 2):
    sequence = list("A" * 1022)
    sequence[392:394] = "GT"
    sequence[510:512] = "AG"
    intact = "".join(sequence)
    corrupt = intact[:511] + "C" + intact[512:]
    rows = []
    for block in range(blocks):
        for donor in range(3):
            for acceptor in range(3):
                rows.append(
                    {
                        "block_id": f"b{block}",
                        "split_role": "phase3_reserve",
                        "chrom": f"chr{1 + block}",
                        "strand": "+" if block % 2 == 0 else "-",
                        "donor_phase": donor,
                        "acceptor_phase": acceptor,
                        "sequence_intact": intact,
                        "sequence_acceptor_ac": corrupt,
                        "donor_pos": 392,
                        "acceptor_pos": 512,
                    }
                )
    return {"schema": RUN.COHORT_SCHEMA, "cells": rows}


def _write_manifest(path, artifact, inputs, *, timestamp, commit="c" * 40):
    payload = {
        "experiment_id": f"fixture-{artifact.stem}",
        "command": ["fixture"],
        "status": "COMPLETE",
        "config": {"stage": "target"},
        "seeds": {},
        "inputs": [file_record(value) for value in inputs],
        "outputs": [file_record(artifact)],
        "notes": [],
        "created_at_utc": timestamp,
        "git_commit": commit,
        "git_dirty": False,
        "environment": {},
    }
    path.write_text(json.dumps(payload) + "\n")
    return payload


def _prerequisite_chain_files(tmp_path):
    labels = (
        "phase2_target",
        "phase2_analysis",
        "phase2_audit",
        "candidate_screen",
        "candidate_freeze",
        "path_mde",
        "candidate_freeze_audit",
        "register_preflight",
        "register_preflight_analysis",
        "register_preflight_audit",
    )
    paths = {}
    dummy = tmp_path / "scientific-input.json"
    dummy.write_text("{}\n")
    for label in labels:
        artifact = tmp_path / f"{label}.json"
        payload = {}
        if label in {
            "phase2_audit",
            "candidate_freeze_audit",
            "register_preflight_audit",
        }:
            payload = {
                "status": "PASS",
                "checks": {"producer_manifest_output_bound": True},
            }
        artifact.write_text(json.dumps(payload) + "\n")
        paths[label] = artifact
        paths[f"{label}_manifest"] = artifact.with_name(
            artifact.stem + ".manifest.json"
        )
    inputs = {
        "phase2_target": [dummy],
        "phase2_analysis": [paths["phase2_target"]],
        "phase2_audit": [
            paths["phase2_target"],
            paths["phase2_target_manifest"],
            paths["phase2_analysis"],
            paths["phase2_analysis_manifest"],
        ],
        "candidate_screen": [dummy],
        "candidate_freeze": [paths["candidate_screen"]],
        "path_mde": [paths["candidate_screen"], paths["candidate_freeze"]],
        "candidate_freeze_audit": [
            paths["candidate_screen"],
            paths["candidate_screen_manifest"],
            paths["candidate_freeze"],
            paths["candidate_freeze_manifest"],
            paths["path_mde"],
            paths["path_mde_manifest"],
        ],
        "register_preflight": [
            paths["phase2_analysis"],
            paths["phase2_audit"],
        ],
        "register_preflight_analysis": [
            paths["register_preflight"],
            paths["path_mde"],
            paths["candidate_freeze_audit"],
        ],
        "register_preflight_audit": [
            paths["register_preflight"],
            paths["register_preflight_manifest"],
            paths["register_preflight_analysis"],
            paths["register_preflight_analysis_manifest"],
        ],
    }
    order = {
        "phase2_target": 0,
        "phase2_analysis": 1,
        "phase2_audit": 2,
        "candidate_screen": 0,
        "candidate_freeze": 1,
        "path_mde": 2,
        "candidate_freeze_audit": 3,
        "register_preflight": 3,
        "register_preflight_analysis": 4,
        "register_preflight_audit": 5,
    }
    for label in labels:
        _write_manifest(
            paths[f"{label}_manifest"],
            paths[label],
            inputs[label],
            timestamp=f"2026-08-10T00:00:0{order[label]}+00:00",
        )
    return paths


def test_reserve_parser_requires_exact_complete_geometry():
    cells = RUN.reserve_cells(_cohort(), expected_blocks=2)
    assert len(cells) == 18
    assert {
        cell.block_id
        for cell in RUN.deterministic_shard(cells, shard_index=0, shard_count=2)
    } == {"b0"}
    broken = _cohort()
    broken["cells"].pop()
    with pytest.raises(RuntimeError, match="nine-cell|complete"):
        RUN.reserve_cells(broken, expected_blocks=2)


def test_freeze_builds_p3_only_seal_without_importing_p10(tmp_path):
    binding_file = tmp_path / "binding.json"
    binding_file.write_text("{}\n")
    bindings = {"cohort": file_record(binding_file), **candidate_bindings(tmp_path)}
    phase2_target = {
        "schema": FREEZE.PHASE2_TARGET_SCHEMA,
        "status": "TARGET_SCORED_UNANALYZED",
        "selected_doses": {"p3": 0.5, "p10": None},
    }
    phase2_analysis = {
        "schema": PHASE2_ANALYSIS_SCHEMA,
        "status": "COMPLETE",
        "target_sha256": "target-hash",
        "authorization": {
            "authorize_p3_phase3": True,
            "authorize_joint_phase3": False,
        },
        "endpoints": {"J": {"estimate": None}},
    }
    phase2_audit = {
        "schema": PHASE2_AUDIT_SCHEMA,
        "status": "PASS",
        "analysis_sha256": "analysis-hash",
    }
    preflight = {
        "schema": "gfmcd.pbc.p3_register_preflight_analysis.v1",
        "status": "P3_REGISTER_PREFLIGHT_PASS",
        "authorization": {
            "authorize_p3_register_reserve": True,
            "authorize_joint_register_reserve": False,
        },
    }
    preflight_audit = {
        "schema": REGISTER_PREFLIGHT_AUDIT_SCHEMA,
        "status": "PASS",
        "analysis_sha256": "preflight-analysis-hash",
    }
    seal = FREEZE.build_seal(
        phase2_target=phase2_target,
        phase2_analysis=phase2_analysis,
        phase2_audit=phase2_audit,
        phase2_analysis_sha256="analysis-hash",
        phase2_target_sha256="target-hash",
        register_preflight_analysis=preflight,
        register_preflight_audit=preflight_audit,
        register_preflight_analysis_sha256="preflight-analysis-hash",
        bindings=bindings,
        include_p10=False,
        target_path=str(tmp_path / "target.json"),
        prerequisite_manifest_chain=prerequisite_manifest_chain(),
    )
    assert seal["mode"] == "P3_ONLY"
    assert seal["design"]["p3_attenuation_dose"] == 0.5
    assert seal["design"]["p10_attenuation_dose"] is None


def test_freezer_requires_complete_chronological_sibling_manifest_chain(tmp_path):
    paths = _prerequisite_chain_files(tmp_path)
    chain = FREEZE.validate_prerequisite_manifest_chain(paths)
    assert chain["status"] == "PASS"
    assert all(chain["checks"].values())

    manifest_path = paths["candidate_freeze_manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"] = []
    manifest_path.write_text(json.dumps(manifest) + "\n")
    failed = FREEZE.validate_prerequisite_manifest_chain(paths)
    assert failed["status"] == "FAIL"
    assert not failed["checks"]["candidate_freeze_sibling_manifest"]


def test_reserve_parser_rejects_geometry_tampering():
    cohort = _cohort()
    tampered = copy.deepcopy(cohort)
    tampered["cells"][0]["donor_pos"] = 391
    with pytest.raises((RuntimeError, ValueError)):
        RUN.reserve_cells(tampered, expected_blocks=2)


def test_assembler_requires_disjoint_complete_frozen_shards(tmp_path):
    binding_file = tmp_path / "binding.json"
    binding_file.write_text("{}\n")
    preflight = {
        "schema": "gfmcd.pbc.p3_register_preflight_analysis.v1",
        "status": "P3_REGISTER_PREFLIGHT_PASS",
        "authorization": {
            "authorize_p3_register_reserve": True,
            "authorize_joint_register_reserve": False,
        },
    }
    seal = make_register_target_seal(
        phase2_authorization={
            "phase2_p3_authorized": True,
            "phase2_joint_authorized": False,
            "jreg_direction": None,
        },
        preflight_analysis=preflight,
        preflight_audit={
            "schema": REGISTER_PREFLIGHT_AUDIT_SCHEMA,
            "status": "PASS",
        },
        bindings={"cohort": file_record(binding_file), **candidate_bindings(tmp_path)},
        include_p10=False,
        p3_attenuation_dose=0.5,
        p10_attenuation_dose=None,
        output_path=str(tmp_path / "target.json"),
        shard_count=2,
        prerequisite_manifest_chain=prerequisite_manifest_chain(),
    )

    def shard(index: int, blocks: list[str]):
        rows = [
            {"block_id": block, "register": register, "cell": cell}
            for block in blocks
            for register in (-1, 0, 1)
            for cell in range(9)
        ]
        no_hook = [
            {"block_id": block, "cell": cell}
            for block in blocks
            for cell in range(9)
        ]
        return {
            "schema": RUN.SHARD_SCHEMA,
            "status": "COMPLETE",
            "seal_sha256": seal["seal_sha256"],
            "mode": seal["mode"],
            "design": seal["design"],
            "authorization": seal["authorization"],
            "sources": seal["bindings"],
            "model": {"checkpoint": "fake"},
            "n_blocks": len(blocks),
            "shard": {"index": index, "count": 2, "block_ids": blocks},
            "mechanics": {
                "position_table_shape": [1026, 512],
                "no_hook_rows": no_hook,
                "register_zero_no_hook_max_abs": 0.0,
            },
            "rows": rows,
        }

    shards = [shard(0, ["b0", "b2"]), shard(1, ["b1", "b3"])]
    result = ASSEMBLE.assemble_payload(
        seal,
        shards,
        shard_records=[{"path": "s0"}, {"path": "s1"}],
        expected_blocks=4,
        expected_component_count=1,
    )
    assert result["n_blocks"] == 4
    assert len(result["rows"]) == 4 * 3 * 9
    duplicate = copy.deepcopy(shards)
    duplicate[1]["shard"]["block_ids"][0] = "b0"
    duplicate[1]["rows"] = [
        {**row, "block_id": "b0" if row["block_id"] == "b1" else row["block_id"]}
        for row in duplicate[1]["rows"]
    ]
    with pytest.raises(RuntimeError, match="overlap"):
        ASSEMBLE.assemble_payload(
            seal,
            duplicate,
            shard_records=[{"path": "s0"}, {"path": "s1"}],
            expected_blocks=4,
            expected_component_count=1,
        )
