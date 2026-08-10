from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gfmcd.provenance import file_record
from gfmcd.splicing.pbc_register_candidate_runner import (
    PATCH_OPERATION_REPLACE,
    activation_patch_context,
)
from gfmcd.splicing.pbc_register_candidates import (
    CANDIDATE_SCREEN_SCHEMA,
    build_candidate_freeze,
    component_universe,
    validate_candidate_screen,
)


REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = REPO / "scripts/splicebert_positional_boundary_coherence" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load("audit_p3_register_candidate_freeze")


def _screen(*, reaches: bool = True, blocks: int = 8):
    universe = list(component_universe())
    preferred = ["L0H0", "L1H0", "L2H0", "L3H0", "L0H1", "L1H1", "L2H1", "L3H1"]
    rank = {value: 2.0 - index * 0.1 for index, value in enumerate(preferred)}
    singletons = []
    for index, component in enumerate(universe):
        component_id = component["component_id"]
        d_path = rank.get(component_id, -1.0 - index * 0.001)
        per_block = [
            {
                "block_id": f"b{block:02d}",
                "chrom": f"chr{1 + block % 2}",
                "strand": "+" if (block // 2) % 2 == 0 else "-",
                "D_path": d_path + (block % 2) * 0.0001,
                "R_path": 0.20 + index * 0.00001 + (block % 2) * 0.0001,
                "Q_path": 0.25 + index * 0.00001 + (block % 2) * 0.0001,
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
        if reaches:
            d_path = 0.60 if size == 1 else 0.85 + size * 0.001
        else:
            d_path = 0.50 + size * 0.02
        per_block = [
            {
                "block_id": f"b{block:02d}",
                "chrom": f"chr{1 + block % 2}",
                "strand": "+" if (block // 2) % 2 == 0 else "-",
                "D_path": d_path + (block % 2) * 0.0001,
                "R_path": 0.30 + size * 0.01 + (block % 2) * 0.0001,
                "Q_path": 0.35 + size * 0.01 + (block % 2) * 0.0001,
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


def test_exact_102_component_census_and_executed_stream_required():
    screen = _screen()
    result = validate_candidate_screen(screen, enforce_development_contract=False)
    assert len(component_universe()) == 102
    assert result["passed"]
    declarative = copy.deepcopy(screen)
    declarative["tensor_stream"]["patch_events"] = 0
    with pytest.raises(ValueError, match="declarative|incomplete"):
        validate_candidate_screen(declarative, enforce_development_contract=False)


def test_full_span_same_state_head_and_mlp_hooks_are_real():
    attention = torch.nn.Identity()
    mlp = torch.nn.Identity()
    adapter = SimpleNamespace(
        num_heads=2,
        attention_self_modules=(attention,),
        encoder_layers=(SimpleNamespace(output=SimpleNamespace(dense=mlp)),),
    )
    head_source = torch.full((1, 520, 2), 7.0)
    mlp_source = torch.full((1, 520, 4), 9.0)
    components = [
        {
            "component_id": "L0H1",
            "layer": 0,
            "kind": "head",
            "head": 1,
        },
        {"component_id": "MLP0", "layer": 0, "kind": "mlp", "head": None},
    ]
    host = torch.zeros((1, 520, 4))
    with activation_patch_context(
        adapter,
        components=components,
        source_tensors={"L0H1": head_source, "MLP0": mlp_source},
        host_sequences=["A" * 10],
        source_sequences=["A" * 10],
        host_state="intact",
        source_state="intact",
        operation=PATCH_OPERATION_REPLACE,
    ) as provenance:
        head_out = attention(host)
        mlp_out = mlp(host)
    assert torch.equal(head_out[:, :393], host[:, :393])
    assert torch.equal(head_out[:, 513:], host[:, 513:])
    assert torch.all(head_out[:, 393:513, 2:] == 7)
    assert torch.all(mlp_out[:, 393:513] == 9)
    assert provenance["token_support"] == [393, 513]
    with pytest.raises(ValueError, match="corruption state"):
        with activation_patch_context(
            adapter,
            components=components[:1],
            source_tensors={"L0H1": head_source},
            host_sequences=["A"],
            source_sequences=["A"],
            host_state="intact",
            source_state="acceptor_ac",
            operation=PATCH_OPERATION_REPLACE,
        ):
            pass


def test_prefix_freeze_null_independence_and_independent_audit(tmp_path):
    screen = _screen(reaches=True)
    screen_path = tmp_path / "screen.json"
    screen_path.write_text(json.dumps(screen) + "\n")
    freeze, mde = build_candidate_freeze(
        screen,
        screen_record=file_record(screen_path),
        enforce_development_contract=False,
    )
    assert freeze["selection"]["classification"] == "PREFIX_80"
    assert freeze["selection"]["candidate_count"] == 2
    sets = [
        tuple(sorted(row["component_id"] for row in spec["components"]))
        for family in freeze["null_families"].values()
        for spec in family
    ]
    assert len(sets) == len(set(sets)) == 93
    assert all(
        mde["mde80"][name] <= 0.05
        for name in ("D_path", "R_path", "Q_path")
    )
    freeze_path = tmp_path / "freeze.json"
    mde_path = tmp_path / "mde.json"
    freeze_path.write_text(json.dumps(freeze) + "\n")
    mde_path.write_text(json.dumps(mde) + "\n")
    audit = AUDIT.audit_payloads(
        screen,
        freeze,
        mde,
        screen_path=screen_path,
        freeze_path=freeze_path,
        path_mde_path=mde_path,
        enforce_development_contract=False,
    )
    assert audit["status"] == "PASS"
    assert audit["recomputed"]["null_sets_total"] == 93


def test_distributed_fallback_freezes_top_eight(tmp_path):
    screen = _screen(reaches=False)
    path = tmp_path / "screen.json"
    path.write_text(json.dumps(screen) + "\n")
    freeze, _mde = build_candidate_freeze(
        screen,
        screen_record=file_record(path),
        enforce_development_contract=False,
    )
    assert freeze["selection"]["classification"] == "DISTRIBUTED_CANDIDATE_SET"
    assert freeze["selection"]["candidate_count"] == 8
    assert not freeze["selection"]["smallest_prefix_reaches_80"]
