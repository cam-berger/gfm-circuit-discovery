from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from gfmcd.splicing.pbc_register_mediation import validate_compact_mediation

import test_pbc_register_mediation as fixtures


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts/splicebert_positional_boundary_coherence" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ASSEMBLE = _load("assemble_p3_register_mediation_target")
RUN = _load("run_p3_register_mediation_target")


def _shards():
    freeze = fixtures._freeze()
    rows = fixtures._rows(blocks=4)
    common = {
        "schema": "gfmcd.pbc.p3_register_mediation_target_shard.v1",
        "status": "COMPLETE",
        "stage": "mediation",
        "split_role": "phase3_reserve",
        "mode": "P3_ONLY",
        "seal_sha256": "a" * 64,
        "authorization": {"path_instrument_capable": True},
        "design": {"candidate_freeze": freeze},
        "sources": {},
        "model": {"checkpoint": "synthetic"},
        "reserve_blocks": 384,
        "reserve_cells": 3456,
    }
    output = []
    for index, blocks in enumerate(({"b000", "b001"}, {"b002", "b003"})):
        local = [row for row in rows if row["block_id"] in blocks]
        output.append(
            {
                **common,
                "n_blocks": 2,
                "n_cells": 18,
                "shard": {"index": index, "count": 2, "block_ids": sorted(blocks)},
                "mechanics": {
                    "passed": True,
                    "same_sequence_patching": True,
                    "same_corruption_state_patching": True,
                    "full_token_support": [393, 513],
                    "rows_per_block": 2673,
                    "actual_hook_tensor_stream_sha256": f"{index + 1}" * 64,
                    "actual_patch_forward_events": 10,
                    "raw_relocation_replay_intact_margin_max_abs": 0.0,
                    "raw_relocation_replay_corrupt_margin_max_abs": 0.0,
                    "raw_relocation_replay_coupling_max_abs": 0.0,
                },
                "rows": local,
            }
        )
    return freeze, output


def _seal():
    return {
        "mode": "P3_ONLY",
        "seal_sha256": "a" * 64,
        "mediation": {
            "shard_count": 2,
            "p3_attenuation_order": [
                "construct_intact_position_table",
                "attenuate_complete_frozen_p3_band",
                "apply_requested_full_span_analytic_register_ramp",
                "apply_same_state_activation_intervention",
            ],
        },
    }


def test_assembly_normalizes_raw_shards_without_retaining_primitive_rows(monkeypatch):
    freeze, shards = _shards()
    monkeypatch.setattr(ASSEMBLE, "verify_register_target_seal", lambda _seal: None)
    result = ASSEMBLE.assemble_payload(
        _seal(),
        iter(shards),
        shard_records=[{"sha256": "a" * 64}, {"sha256": "b" * 64}],
        expected_blocks=4,
    )
    assert "rows" not in result
    assert len(result["endpoint_rows"]) == 4 * 99 * 9
    assert len(result["cell_ledger"]) == 4 * 9
    assert len(result["provenance_ledger"]) == 99 * 3
    assert result["normalization"]["primitive_rows"] == 4 * 2673
    assert result["mechanics"]["passed"] is True
    assert validate_compact_mediation(result, freeze=freeze)["passed"]


def test_assembly_rejects_overlapping_or_incomplete_shards(monkeypatch):
    _freeze, shards = _shards()
    monkeypatch.setattr(ASSEMBLE, "verify_register_target_seal", lambda _seal: None)
    shards[1]["shard"]["block_ids"] = shards[0]["shard"]["block_ids"]
    with pytest.raises(RuntimeError, match="overlap"):
        ASSEMBLE.assemble_payload(
            _seal(),
            shards,
            shard_records=[{"sha256": "a" * 64}, {"sha256": "b" * 64}],
            expected_blocks=4,
        )


def test_static_authorization_failure_occurs_before_checkpoint_loading(
    tmp_path: Path, monkeypatch
):
    output = tmp_path / "mediation.shard-000-of-002.json"
    cohort = tmp_path / "cohort.json"
    candidate = tmp_path / "candidate.json"
    seal_path = tmp_path / "seal.json"
    relocation_target = tmp_path / "relocation.json"
    relocation_analysis = tmp_path / "relocation.analysis.json"
    relocation_audit = tmp_path / "relocation.audit.json"
    for path in (cohort, candidate, relocation_target, relocation_analysis, relocation_audit):
        path.write_text("{}\n")
    seal = {
        "mediation": {"shard_count": 2, "shard_paths": [str(output), "unused"]},
        "bindings": {
            "cohort": {"path": str(cohort)},
            "candidate_freeze": {"path": str(candidate)},
            "checkpoint": {"path": str(tmp_path / "model" / "pytorch_model.bin")},
        },
    }
    seal_path.write_text(json.dumps(seal) + "\n")
    monkeypatch.setattr(RUN, "verify_register_target_seal", lambda _seal: None)
    monkeypatch.setattr(
        RUN,
        "validate_relocation_sources",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("not authorized")),
    )
    loaded = False

    def load(*_args, **_kwargs):
        nonlocal loaded
        loaded = True

    monkeypatch.setattr(RUN, "load_adapter", load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_p3_register_mediation_target.py",
            "--seal",
            str(seal_path),
            "--cohort",
            str(cohort),
            "--candidate-freeze",
            str(candidate),
            "--relocation-target",
            str(relocation_target),
            "--relocation-analysis",
            str(relocation_analysis),
            "--relocation-audit",
            str(relocation_audit),
            "--model",
            str(tmp_path / "model"),
            "--out",
            str(output),
        ],
    )
    with pytest.raises(RuntimeError, match="not authorized"):
        RUN.main()
    assert loaded is False
