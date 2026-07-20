#!/usr/bin/env python
"""Resumable conditional QK/OV/MLP-path execution for validated atlas relations.

Only candidates present in an independently audited stage-two package are eligible.
The full runner reuses the already-opened internal-validation split.  ``--smoke``
uses fixed discovery candidates to exercise B QK/V/QKV, A/B downstream MLP
recovery, and an MLP upstream-attention path without reading validation data.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gfmcd.splicing.circuit import clamp_write, load_adapter, patch_proj, pick_device  # noqa: E402
from run_all_head_screen import ScreenModel  # noqa: E402
from run_atlas_discovery_screen import (  # noqa: E402
    file_sha256, membership_by_packet, packet_meta, select_health_panel,
    shard_name, write_atomic,
)
from run_atlas_mechanism_validation import (  # noqa: E402
    apply_operation, capture_components, load_and_gate_cohort, resolve_repo_reference,
)
from run_atlas_model_preflight import (  # noqa: E402
    HEALTH_ACCURACY_FLOOR, HEALTH_LOGPROB_FLOOR, encode_masks, health_probes,
    health_scores, partner_destroyed, select_cover_pairs, stable_rank, target_scores,
)


IDENTITY_ATOL = 1e-4
REFERENCE_ATOL = 1e-6
# Stage 2 re-evaluates the frozen whole-effect references.  Reusing the
# producer's rank namespace keeps packet batching identical and avoids
# cross-batch float32 reduction drift in future executions.
WHOLE_REFERENCE_BATCH_ORDER_TAG = "mechanism-validation-order"


def sha_id(prefix: str, *parts: object) -> str:
    value = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:20]
    return f"{prefix}_{value}"


def parse_head(component: str) -> tuple[int, int]:
    if not component.startswith("L") or "H" not in component or component.endswith("MLP"):
        raise ValueError(f"not an attention head: {component}")
    layer, head = component[1:].split("H", 1)
    result = int(layer), int(head)
    if not 0 <= result[0] < 6 or not 0 <= result[1] < 16:
        raise ValueError(f"head outside SpliceBERT grid: {component}")
    return result


def endpoint_key(candidate_id: str, role: str, contrast: str | None) -> str:
    return sha_id("endpoint", candidate_id, role, contrast or "general")


def build_smoke_candidates() -> list[dict]:
    common_reps = {"donor": {}, "acceptor": {}}
    return [
        {
            "candidate_id": "smoke_stage2|general|B|L3H1", "kind": "general",
            "component_id": "L3H1", "component_type": "attention_head", "component_layer": 3,
            "estimand": "B", "contrast": None, "target_role": None,
            "importance_control": {"component_id": "L3H2"},
            "exact_group_id": "smoke|B|mixed",
            "exact_group_components": ["L3H1", "L3MLP"],
            "representative": common_reps,
            "validated_whole_endpoints": [
                {"target_role": role, "contrast": None} for role in ("donor", "acceptor")
            ],
            "mechanism": {
                "population_qk_v": {"enabled": True},
                "downstream_mlp_recovery": {"layers": [3, 4, 5]},
            },
        },
        {
            "candidate_id": "smoke_stage2|general|A|L0H0", "kind": "general",
            "component_id": "L0H0", "component_type": "attention_head", "component_layer": 0,
            "estimand": "A", "contrast": None, "target_role": None,
            "importance_control": {"component_id": "L0H1"},
            "exact_group_id": "smoke|A|mixed",
            "exact_group_components": ["L0H0", "L5MLP"],
            "representative": common_reps,
            "validated_whole_endpoints": [
                {"target_role": role, "contrast": None} for role in ("donor", "acceptor")
            ],
            "mechanism": {
                "population_qk_v": {"enabled": False},
                "downstream_mlp_recovery": {"layers": [0, 1, 2, 3, 4, 5]},
            },
        },
        {
            "candidate_id": "smoke_stage2|general|B|L3MLP", "kind": "general",
            "component_id": "L3MLP", "component_type": "mlp_write", "component_layer": 3,
            "estimand": "B", "contrast": None, "target_role": None,
            "importance_control": {"component_id": "L2MLP"},
            "exact_group_id": "smoke|B|mixed",
            "exact_group_components": ["L3H1", "L3MLP"],
            "representative": common_reps,
            "validated_whole_endpoints": [
                {"target_role": role, "contrast": None} for role in ("donor", "acceptor")
            ],
            "mechanism": {
                "population_qk_v": {"enabled": False},
                "upstream_attention_path": {
                    "enabled": True, "attention_group": ["L3H1"],
                    "candidate_mlp": "L3MLP", "importance_control_mlp": "L2MLP",
                },
            },
        },
    ]


def build_endpoints(candidates: list[dict]) -> list[dict]:
    endpoints = []
    for candidate in candidates:
        for whole in candidate["validated_whole_endpoints"]:
            role = whole["target_role"]
            contrast = whole.get("contrast")
            endpoints.append({
                "endpoint_key": endpoint_key(candidate["candidate_id"], role, contrast),
                "candidate_id": candidate["candidate_id"],
                "target_role": role,
                "contrast": contrast,
                "scope_kind": "contrast" if contrast else "target_role",
                "scope_value": contrast or role,
            })
    keys = [row["endpoint_key"] for row in endpoints]
    if len(keys) != len(set(keys)):
        raise RuntimeError("stage-two endpoint keys are not unique")
    return endpoints


def endpoint_packet_sets(endpoints: list[dict], packets: list[dict],
                         pairs: list[dict]) -> dict[str, set[str]]:
    role_ids: dict[str, set[str]] = defaultdict(set)
    for packet in packets:
        role_ids[packet["target_role"]].add(packet["packet_id"])
    contrast_ids: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        contrast = f"{pair['category']}|{pair['target_role']}"
        contrast_ids[contrast].update((pair["case_packet_id"], pair["control_packet_id"]))
    result = {}
    for endpoint in endpoints:
        selected = (
            role_ids[endpoint["scope_value"]]
            if endpoint["scope_kind"] == "target_role"
            else contrast_ids[endpoint["scope_value"]]
        )
        if not selected:
            raise RuntimeError(f"empty endpoint scope {endpoint['scope_value']}")
        result[endpoint["endpoint_key"]] = set(selected)
    return result


def capture_qkv_layers(model: ScreenModel, enc, layers: set[int]) -> dict[tuple[int, str], torch.Tensor]:
    store: dict[tuple[int, str], torch.Tensor] = {}
    handles = []
    for layer in sorted(layers):
        module = model.a.attention_self_modules[layer]
        for name in ("query", "key", "value"):
            handles.append(getattr(module, name).register_forward_hook(
                lambda _m, _i, output, layer=layer, name=name:
                store.__setitem__((layer, name), output.detach().clone())
            ))
    try:
        model.logits(enc)
    finally:
        for handle in handles:
            handle.remove()
    expected = {(layer, name) for layer in layers for name in ("query", "key", "value")}
    if set(store) != expected:
        raise RuntimeError("QKV capture grid incomplete")
    return store


@contextlib.contextmanager
def patch_head_qkv(adapter, component: str, source: dict[tuple[int, str], torch.Tensor],
                   positions: torch.Tensor, names: tuple[str, ...]):
    layer, head = parse_head(component)
    with contextlib.ExitStack() as stack:
        for name in names:
            stack.enter_context(patch_proj(
                adapter, layer, head, name, source[(layer, name)], positions
            ))
        yield


def score(model: ScreenModel, phase: str, logits: torch.Tensor, packets: list[dict],
          sequences: list[str], probes: list[list[int]] | None) -> list[dict]:
    return (
        target_scores(model, logits, packets) if phase == "target"
        else health_scores(model, logits, sequences, probes or [])
    )


def margin_value(phase: str, row: dict) -> float:
    return float(row["margin"] if phase == "target" else row["mean_margin"])


def operation_logits(cache: dict[str, torch.Tensor], key: str, run) -> torch.Tensor:
    if key not in cache:
        cache[key] = run()
    return cache[key]


def component_operation(component: str, estimand: str) -> dict:
    return {"component_ids": [component], "estimand": estimand}


def group_operation(components: list[str], estimand: str) -> dict:
    return {"component_ids": sorted(components), "estimand": estimand}


def family_rows_for_candidate(candidate: dict, *, phase: str, model: ScreenModel,
                              native_enc, native_logits: torch.Tensor,
                              native_heads: dict, native_mlps: dict,
                              corrupt_heads: dict, corrupt_mlps: dict,
                              native_qkv: dict, corrupt_qkv: dict,
                              positions: torch.Tensor, batch_packets: list[dict],
                              batch_sequences: list[str], probes: list[list[int]] | None,
                              logits_cache: dict[str, torch.Tensor], gates: list[dict],
                              bridge_once: bool) -> dict[str, list[dict]]:
    """Return per-family-member score arrays plus raw diagnostic score arrays."""
    adapter = model.a
    component = candidate["component_id"]
    control = candidate["importance_control"]["component_id"]
    estimand = candidate["estimand"]
    zero = estimand == "A"
    sources_h = native_heads if zero else corrupt_heads
    sources_m = native_mlps if zero else corrupt_mlps

    def whole_logits(which: str) -> torch.Tensor:
        target = component if which == "candidate" else control
        key = f"whole|{estimand}|{target}"
        return operation_logits(logits_cache, key, lambda: _run_whole(
            model, native_enc, component_operation(target, estimand), sources_h, sources_m, zero
        ))

    candidate_whole_logits = whole_logits("candidate")
    control_whole_logits = whole_logits("control")
    output = {
        "diagnostic:candidate_whole": score(
            model, phase, candidate_whole_logits, batch_packets, batch_sequences, probes
        ),
        "diagnostic:control_whole": score(
            model, phase, control_whole_logits, batch_packets, batch_sequences, probes
        ),
    }

    if candidate["component_type"] == "attention_head":
        if estimand == "B" and candidate["mechanism"]["population_qk_v"]["enabled"]:
            for label, target in (("candidate", component), ("control", control)):
                for condition, names in {
                    "QK": ("query", "key"),
                    "V_OV": ("value",),
                    "QKV_BRIDGE": ("query", "key", "value"),
                }.items():
                    key = f"qkv|{target}|{condition}"
                    logits = operation_logits(logits_cache, key, lambda target=target, names=names: _run_qkv(
                        model, native_enc, target, corrupt_qkv, positions, names
                    ))
                    scores = score(model, phase, logits, batch_packets, batch_sequences, probes)
                    output[f"diagnostic:{label}_{condition}"] = scores
                    if condition != "QKV_BRIDGE":
                        output[f"family:{label}_{condition}"] = scores
                    if bridge_once and condition == "QKV_BRIDGE":
                        expected_logits = candidate_whole_logits if label == "candidate" else control_whole_logits
                        error = float((logits - expected_logits).abs().max())
                        gates.append({
                            "gate": f"{phase}_qkv_whole_bridge:{target}", "value": error,
                            "threshold": IDENTITY_ATOL, "passed": error <= IDENTITY_ATOL,
                        })

            if bridge_once:
                for target in (component, control):
                    key = f"identity_qkv|{target}"
                    logits = operation_logits(logits_cache, key, lambda target=target: _run_qkv(
                        model, native_enc, target, native_qkv, positions,
                        ("query", "key", "value")
                    ))
                    error = float((logits - native_logits).abs().max())
                    gates.append({
                        "gate": f"{phase}_native_qkv_identity:{target}", "value": error,
                        "threshold": IDENTITY_ATOL, "passed": error <= IDENTITY_ATOL,
                    })

        for mlp_layer in candidate["mechanism"]["downstream_mlp_recovery"]["layers"]:
            candidate_restore = operation_logits(
                logits_cache, f"restore|{estimand}|{component}|L{mlp_layer}MLP",
                lambda mlp_layer=mlp_layer: _run_whole_restore(
                    model, native_enc, component_operation(component, estimand),
                    sources_h, sources_m, zero, (mlp_layer, "mlp"), native_mlps,
                ),
            )
            control_restore = operation_logits(
                logits_cache, f"restore|{estimand}|{control}|L{mlp_layer}MLP",
                lambda mlp_layer=mlp_layer: _run_whole_restore(
                    model, native_enc, component_operation(control, estimand),
                    sources_h, sources_m, zero, (mlp_layer, "mlp"), native_mlps,
                ),
            )
            output[f"diagnostic:candidate_restore_L{mlp_layer}"] = score(
                model, phase, candidate_restore, batch_packets, batch_sequences, probes
            )
            output[f"diagnostic:control_restore_L{mlp_layer}"] = score(
                model, phase, control_restore, batch_packets, batch_sequences, probes
            )
    else:
        path = candidate["mechanism"]["upstream_attention_path"]
        if path["enabled"]:
            heads = path["attention_group"]
            upstream_logits = operation_logits(
                logits_cache, f"upstream|{estimand}|{'+'.join(heads)}",
                lambda: _run_whole(
                    model, native_enc, group_operation(heads, estimand),
                    sources_h, sources_m, zero,
                ),
            )
            output["diagnostic:upstream_attention_group"] = score(
                model, phase, upstream_logits, batch_packets, batch_sequences, probes
            )
            for label, mlp_component in (("candidate", component), ("control", control)):
                block = (int(mlp_component[1:].split("MLP", 1)[0]), "mlp")
                restored = operation_logits(
                    logits_cache, f"upstream_restore|{estimand}|{'+'.join(heads)}|{mlp_component}",
                    lambda block=block: _run_whole_restore(
                        model, native_enc, group_operation(heads, estimand),
                        sources_h, sources_m, zero, block, native_mlps,
                    ),
                )
                output[f"diagnostic:upstream_{label}_mlp_restore"] = score(
                    model, phase, restored, batch_packets, batch_sequences, probes
                )
    return output


def _run_whole(model: ScreenModel, enc, operation: dict, source_heads: dict,
               source_mlps: dict, zero: bool) -> torch.Tensor:
    with apply_operation(
        model.a, operation, source_heads=source_heads, source_mlps=source_mlps, zero=zero
    ):
        return model.logits(enc)


def _run_qkv(model: ScreenModel, enc, component: str, source: dict,
             positions: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
    with patch_head_qkv(model.a, component, source, positions, names):
        return model.logits(enc)


def _run_whole_restore(model: ScreenModel, enc, operation: dict, source_heads: dict,
                       source_mlps: dict, zero: bool, block: tuple[int, str],
                       native_mlps: dict) -> torch.Tensor:
    with contextlib.ExitStack() as stack:
        stack.enter_context(apply_operation(
            model.a, operation, source_heads=source_heads,
            source_mlps=source_mlps, zero=zero,
        ))
        stack.enter_context(clamp_write(model.a, [block], native_mlps))
        return model.logits(enc)


def derived_rows(candidate: dict, score_map: dict[str, list[dict]], phase: str,
                 baseline: list[dict]) -> list[list[dict]]:
    """Transform raw score conditions to per-packet family and diagnostic measures."""
    n = len(baseline)
    rows: list[list[dict]] = [[] for _ in range(n)]
    base = [margin_value(phase, item) for item in baseline]
    values = {
        key: [margin_value(phase, item) for item in block]
        for key, block in score_map.items()
    }
    component = candidate["component_id"]
    control = candidate["importance_control"]["component_id"]
    estimand = candidate["estimand"]
    for i in range(n):
        whole_candidate = values["diagnostic:candidate_whole"][i]
        whole_control = values["diagnostic:control_whole"][i]
        common = {
            "candidate_id": candidate["candidate_id"], "component_id": component,
            "control_component_id": control, "estimand": estimand,
        }
        rows[i].extend([
            {**common, "member": "candidate_whole", "member_role": "diagnostic",
             "value": base[i] - whole_candidate},
            {**common, "member": "control_whole", "member_role": "diagnostic",
             "value": base[i] - whole_control},
        ])
        if candidate["component_type"] == "attention_head":
            if estimand == "B" and candidate["mechanism"]["population_qk_v"]["enabled"]:
                for label in ("candidate", "control"):
                    for condition in ("QK", "V_OV"):
                        rows[i].append({
                            **common, "member": f"{label}_{condition}",
                            "member_role": "family",
                            "value": base[i] - values[f"family:{label}_{condition}"][i],
                        })
                    rows[i].append({
                        **common, "member": f"{label}_QKV_BRIDGE",
                        "member_role": "diagnostic",
                        "value": base[i] - values[f"diagnostic:{label}_QKV_BRIDGE"][i],
                    })
            for layer in candidate["mechanism"]["downstream_mlp_recovery"]["layers"]:
                c_recovery = values[f"diagnostic:candidate_restore_L{layer}"][i] - whole_candidate
                k_recovery = values[f"diagnostic:control_restore_L{layer}"][i] - whole_control
                rows[i].extend([
                    {**common, "member": f"candidate_MLP_recovery_L{layer}",
                     "member_role": "diagnostic", "value": c_recovery},
                    {**common, "member": f"control_MLP_recovery_L{layer}",
                     "member_role": "diagnostic", "value": k_recovery},
                    {**common, "member": f"MLP_recovery_difference_L{layer}",
                     "member_role": "family", "value": c_recovery - k_recovery},
                ])
        else:
            path = candidate["mechanism"]["upstream_attention_path"]
            if path["enabled"]:
                upstream = values["diagnostic:upstream_attention_group"][i]
                c_recovery = values["diagnostic:upstream_candidate_mlp_restore"][i] - upstream
                k_recovery = values["diagnostic:upstream_control_mlp_restore"][i] - upstream
                rows[i].extend([
                    {**common, "member": "upstream_candidate_mlp_recovery",
                     "member_role": "diagnostic", "value": c_recovery},
                    {**common, "member": "upstream_control_mlp_recovery",
                     "member_role": "diagnostic", "value": k_recovery},
                    {**common, "member": "upstream_attention_to_candidate_vs_control_mlp_recovery",
                     "member_role": "family", "value": c_recovery - k_recovery},
                ])
        if not all(math.isfinite(float(row["value"])) for row in rows[i]):
            raise RuntimeError("non-finite stage-two derived result")
    return rows


def phase_groups(packets: list[dict], endpoints: list[dict], scopes: dict[str, set[str]],
                 phase: str) -> dict[tuple[str, ...], list[dict]]:
    if phase == "health":
        signature = tuple(sorted(row["endpoint_key"] for row in endpoints))
        return {signature: list(packets)}
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for packet in packets:
        signature = tuple(sorted(
            row["endpoint_key"] for row in endpoints
            if packet["packet_id"] in scopes[row["endpoint_key"]]
        ))
        if signature:
            groups[signature].append(packet)
    return groups


def contract_hash(package_path: Path, audit_path: Path, model_path: Path, phase: str,
                  cohort_paths: dict[str, Path], candidates: list[dict],
                  endpoints: list[dict], smoke: bool) -> str:
    executor_path = Path(__file__).resolve()
    payload = {
        "schema": "head_context_atlas.stage2_execution_contract/1",
        "package_sha256": file_sha256(package_path),
        "audit_sha256": file_sha256(audit_path),
        "executor_sha256": file_sha256(executor_path),
        "checkpoint_sha256": file_sha256(model_path / "pytorch_model.bin"),
        "cohort_hashes": {key: file_sha256(path) for key, path in cohort_paths.items()},
        "phase": phase, "candidates": candidates, "endpoints": endpoints, "smoke": smoke,
        "precision": "float32; TF32/AMP off", "seeds": [0, 0, 0],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_existing(path: Path, contract: str, phase: str, packet_id: str,
                      endpoint_keys: set[str]) -> None:
    row = json.loads(path.read_text())
    if row.get("schema") != "head_context_atlas.stage2_mechanism_shard/1":
        raise RuntimeError(f"bad existing stage-two shard {path}")
    if row.get("contract_sha256") != contract or row.get("phase") != phase:
        raise RuntimeError(f"stage-two shard contract mismatch {path}")
    if row["packet"]["packet_id"] != packet_id:
        raise RuntimeError(f"stage-two shard packet mismatch {path}")
    if {item["endpoint_key"] for item in row["endpoints"]} != endpoint_keys:
        raise RuntimeError(f"stage-two endpoint grid mismatch {path}")


def run_phase(*, phase: str, model: ScreenModel, packets: list[dict], sequences: dict[str, str],
              pairs: list[dict], candidates: list[dict], endpoints: list[dict],
              scopes: dict[str, set[str]], output_dir: Path, contract: str,
              batch_size: int, reference: dict[tuple[str, str, str], float]) -> dict:
    endpoint_by_key = {row["endpoint_key"]: row for row in endpoints}
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    membership = membership_by_packet(pairs)
    shard_dir = output_dir / f"{phase}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    groups = phase_groups(packets, endpoints, scopes, phase)
    expected_by_packet = {
        packet["packet_id"]: set(signature)
        for signature, block in groups.items() for packet in block
    }
    expected_names = {shard_name(packet_id) for packet_id in expected_by_packet}
    prior_path = output_dir / f"run_manifest.{phase}.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        if (prior.get("contract_sha256") == contract and prior.get("status") == "PASS"
                and prior.get("n_failures") == 0
                and {path.name for path in shard_dir.glob("packet_*.json")} == expected_names):
            for packet_id, keys in expected_by_packet.items():
                validate_existing(shard_dir / shard_name(packet_id), contract, phase, packet_id, keys)
            # Ephemeral markers are consumed by ``main`` and never persisted.  A true
            # no-op resume must leave even the manifests byte-for-byte unchanged.
            prior["_resume_noop_verified"] = True
            prior["_resume_n_new_shards"] = 0
            return prior

    gates = []
    first_batch = True
    bridge_tested_candidates: set[str] = set()
    qkv_candidates = {
        row["candidate_id"] for row in candidates
        if row["component_type"] == "attention_head"
        and row["estimand"] == "B"
        and row["mechanism"]["population_qk_v"]["enabled"]
    }
    candidate_packet_ids = {
        candidate_id: set().union(*(
            scopes[row["endpoint_key"]] for row in endpoints
            if row["candidate_id"] == candidate_id
        ))
        for candidate_id in qkv_candidates
    }
    qkv_anchor_packets = {
        min(packet_ids, key=lambda packet_id: (stable_rank(
            "stage2-qkv-gate", phase, candidate_id, packet_id
        ), packet_id))
        for candidate_id, packet_ids in candidate_packet_ids.items()
    }
    execution_anchor = min(expected_by_packet, key=lambda packet_id: (
        stable_rank("stage2-execution-gate", phase, packet_id), packet_id
    ))
    forced_gate_packets = qkv_anchor_packets | {execution_anchor}
    n_new = 0
    start_time = time.time()
    for signature in sorted(groups):
        group_packets = sorted(groups[signature], key=lambda row: (
            stable_rank(WHOLE_REFERENCE_BATCH_ORDER_TAG, phase, row["packet_id"]),
            row["packet_id"],
        ))
        pending = []
        for packet in group_packets:
            path = shard_dir / shard_name(packet["packet_id"])
            if path.exists():
                validate_existing(path, contract, phase, packet["packet_id"], set(signature))
            if not path.exists() or packet["packet_id"] in forced_gate_packets:
                pending.append(packet)
        active_candidates = sorted({
            endpoint_by_key[key]["candidate_id"] for key in signature
        })
        for start in range(0, len(pending), batch_size):
            batch_packets = pending[start:start + batch_size]
            batch_sequences = [sequences[row["sequence_id"]] for row in batch_packets]
            probes = (
                [health_probes(packet, sequence) for packet, sequence in zip(batch_packets, batch_sequences)]
                if phase == "health" else None
            )
            masks = probes if probes is not None else [[row["masked_base_win"]] for row in batch_packets]
            native_enc = encode_masks(model, batch_sequences, masks)
            corrupt_sequences = [
                partner_destroyed(packet, sequence)
                for packet, sequence in zip(batch_packets, batch_sequences)
            ]
            corrupt_enc = encode_masks(model, corrupt_sequences, masks)
            native_logits = model.logits(native_enc)
            corrupt_logits = model.logits(corrupt_enc)
            baseline = score(model, phase, native_logits, batch_packets, batch_sequences, probes)
            native_heads, native_mlps = capture_components(model, native_enc)
            corrupt_heads, corrupt_mlps = capture_components(model, corrupt_enc)
            qkv_layers = {
                parse_head(component)[0]
                for candidate_id in active_candidates
                for component in (
                    candidate_by_id[candidate_id]["component_id"],
                    candidate_by_id[candidate_id]["importance_control"]["component_id"],
                )
                if candidate_by_id[candidate_id]["component_type"] == "attention_head"
                and candidate_by_id[candidate_id]["estimand"] == "B"
            }
            native_qkv = capture_qkv_layers(model, native_enc, qkv_layers) if qkv_layers else {}
            corrupt_qkv = capture_qkv_layers(model, corrupt_enc, qkv_layers) if qkv_layers else {}
            positions = torch.arange(
                native_enc["input_ids"].shape[1], device=model.a.device
            )[None].expand(len(batch_packets), -1)
            if not torch.all(native_enc["attention_mask"].bool()):
                raise RuntimeError("stage-two v1 expects fixed-length packets without padding")
            if first_batch:
                repeat = model.logits(native_enc)
                error = float((repeat - native_logits).abs().max())
                gates.append({"gate": f"{phase}_deterministic_repeat", "value": error,
                              "threshold": 1e-6, "passed": error <= 1e-6})
                if phase == "health":
                    accuracy = float(np.mean([row["accuracy"] for row in baseline]))
                    logprob = float(np.mean([row["mean_logprob"] for row in baseline]))
                    gates.extend([
                        {"gate": "masked_health_accuracy", "value": accuracy,
                         "threshold": HEALTH_ACCURACY_FLOOR,
                         "passed": accuracy >= HEALTH_ACCURACY_FLOOR},
                        {"gate": "masked_health_logprob", "value": logprob,
                         "threshold": HEALTH_LOGPROB_FLOOR,
                         "passed": logprob >= HEALTH_LOGPROB_FLOOR},
                    ])
            logits_cache: dict[str, torch.Tensor] = {}
            candidate_results: dict[str, list[list[dict]]] = {}
            for candidate_id in active_candidates:
                candidate = candidate_by_id[candidate_id]
                bridge_once = candidate_id in qkv_candidates and candidate_id not in bridge_tested_candidates
                score_map = family_rows_for_candidate(
                    candidate, phase=phase, model=model, native_enc=native_enc,
                    native_logits=native_logits, native_heads=native_heads,
                    native_mlps=native_mlps, corrupt_heads=corrupt_heads,
                    corrupt_mlps=corrupt_mlps, native_qkv=native_qkv,
                    corrupt_qkv=corrupt_qkv, positions=positions,
                    batch_packets=batch_packets, batch_sequences=batch_sequences,
                    probes=probes, logits_cache=logits_cache, gates=gates,
                    bridge_once=bridge_once,
                )
                if bridge_once:
                    bridge_tested_candidates.add(candidate_id)
                candidate_results[candidate_id] = derived_rows(
                    candidate, score_map, phase, baseline
                )

            for index, packet in enumerate(batch_packets):
                endpoint_records = []
                for key in signature:
                    endpoint = endpoint_by_key[key]
                    items = candidate_results[endpoint["candidate_id"]][index]
                    if phase == "target":
                        for item in items:
                            if item["member"] in {"candidate_whole", "control_whole"}:
                                expected = reference.get((
                                    packet["packet_id"], endpoint["candidate_id"], item["member"]
                                ))
                                if expected is not None:
                                    item["whole_reference"] = expected
                                    item["whole_reference_abs_error"] = abs(item["value"] - expected)
                    endpoint_records.append({**endpoint, "members": items})
                record = {
                    "schema": "head_context_atlas.stage2_mechanism_shard/1",
                    "claim_boundary": "conditional QK/OV/MLP-path measurements; inference pending",
                    "phase": phase, "contract_sha256": contract,
                    "packet": packet_meta(packet, membership),
                    "probe_positions": probes[index] if probes is not None else None,
                    "endpoints": endpoint_records,
                }
                write_atomic(shard_dir / shard_name(packet["packet_id"]), record)
                n_new += 1
            first_batch = False

    shard_paths = sorted(shard_dir.glob("packet_*.json"))
    seen = set()
    n_members = 0
    reference_errors = []
    for path in shard_paths:
        record = json.loads(path.read_text())
        if record.get("contract_sha256") != contract:
            continue
        seen.add(record["packet"]["packet_id"])
        for endpoint in record["endpoints"]:
            n_members += len(endpoint["members"])
            reference_errors.extend(
                item["whole_reference_abs_error"] for item in endpoint["members"]
                if item.get("whole_reference_abs_error") is not None
            )
    gates.append({"gate": f"{phase}_packet_grid", "value": len(seen),
                  "threshold": len(expected_by_packet), "passed": seen == set(expected_by_packet)})
    gates.append({
        "gate": f"{phase}_qkv_candidate_gate_grid",
        "value": len(bridge_tested_candidates), "threshold": len(qkv_candidates),
        "passed": bridge_tested_candidates == qkv_candidates,
    })
    if phase == "target":
        expected_reference_count = 2 * sum(
            len(scopes[row["endpoint_key"]]) for row in endpoints
        )
        gates.append({
            "gate": "target_whole_stage_reference_grid",
            "value": len(reference_errors), "threshold": expected_reference_count,
            "passed": len(reference_errors) == expected_reference_count,
        })
    if reference_errors:
        value = max(reference_errors)
        gates.append({"gate": f"{phase}_whole_stage_reference", "value": value,
                      "threshold": REFERENCE_ATOL, "passed": value <= REFERENCE_ATOL})
    failures = [row for row in gates if not row["passed"]]
    return {
        "schema": "head_context_atlas.stage2_mechanism_phase/1",
        "status": "PASS" if not failures else "FAIL",
        "phase": phase, "contract_sha256": contract,
        "n_packets": len(expected_by_packet), "n_member_rows": n_members,
        "n_new_shards": n_new, "elapsed_seconds": time.time() - start_time,
        "n_gates": len(gates), "n_failures": len(failures),
        "gates": gates, "failures": failures,
        "shards": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in shard_paths
            if json.loads(path.read_text()).get("contract_sha256") == contract
        },
    }


def load_reference(stage2: dict, *, smoke: bool, reference_rows: Path | None) -> dict[tuple[str, str], float]:
    reference: dict[tuple[str, str, str], float] = {}
    if smoke:
        if reference_rows is None:
            return reference
        with reference_rows.open() as handle:
            for line in handle:
                row = json.loads(line)
                estimand = "A" if row["estimand"].startswith("A_") else "B"
                reference[(row["packet_id"], f"{estimand}|{row['component_id']}")] = float(row["effect_margin"])
        return reference
    target_ref = stage2["whole_validation_row_references"]["target"]
    path = resolve_repo_reference(target_ref["path"])
    if file_sha256(path) != target_ref["sha256"]:
        raise RuntimeError("whole-validation target-row reference hash mismatch")
    import gzip
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row["endpoint_role"] in {"candidate", "importance_control"}:
                member = "candidate_whole" if row["endpoint_role"] == "candidate" else "control_whole"
                reference[(row["packet_id"], f"{row['candidate_id']}|{member}")] = float(row["effect_margin"])
    return reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-audit", required=True)
    parser.add_argument("--stage1-package", required=True)
    parser.add_argument("--cohort-prefix", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_matched_v7"
    ))
    parser.add_argument("--model", default=str(
        REPO / "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"
    ))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("health", "target", "both"), default="health")
    parser.add_argument("--health-n", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reference-rows", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    package_path = Path(args.package)
    audit_path = Path(args.package_audit)
    stage1_path = Path(args.stage1_package)
    stage2 = json.loads(package_path.read_text())
    audit = json.loads(audit_path.read_text())
    stage1 = json.loads(stage1_path.read_text())
    if audit["status"] != "PASS" or audit["n_failures"] or audit["package_sha256"] != file_sha256(package_path):
        raise RuntimeError("stage-two package audit is not a current zero-failure pass")
    expected_status = "SMOKE_NO_STAGE2_CANDIDATES" if args.smoke else "FROZEN_AFTER_WHOLE_VALIDATION_BEFORE_STAGE2_FORWARD"
    if stage2["status"] != expected_status:
        raise RuntimeError(f"stage-two package status must be {expected_status}")
    if file_sha256(stage1_path) != stage2["input_hashes"]["stage1_package"]:
        raise RuntimeError("stage-one package changed after the audited stage-two freeze")
    candidates = build_smoke_candidates() if args.smoke else stage2["candidates"]
    if not candidates:
        raise RuntimeError("no validated candidates; stage-two forward is correctly unnecessary")
    if stage2["stage2_outcomes_loaded"] is not False:
        raise RuntimeError("stage-two package was not outcome-blind to stage-two forwards")

    packets, sequences, pairs, cohort = load_and_gate_cohort(
        stage1, smoke=args.smoke,
        cohort_prefix=Path(args.cohort_prefix) if args.smoke else None,
    )
    if args.smoke:
        lookup = {row["packet_id"]: row for row in packets}
        selected_pairs = select_cover_pairs(pairs, lookup)
        ids = {pair[field] for pair in selected_pairs for field in ("case_packet_id", "control_packet_id")}
        packets = [lookup[packet_id] for packet_id in sorted(ids)]
        sequences = {row["sequence_id"]: sequences[row["sequence_id"]] for row in packets}
        pairs = selected_pairs
    endpoints = build_endpoints(candidates)
    scopes = endpoint_packet_sets(endpoints, packets, pairs)
    target_ids = set().union(*scopes.values())
    target_packets = [row for row in packets if row["packet_id"] in target_ids]
    if args.smoke:
        health_packets = list(target_packets)
    else:
        health_ids = set(select_health_panel(packets, pairs, args.health_n))
        health_packets = [row for row in packets if row["packet_id"] in health_ids]
    reference_raw = load_reference(
        stage2, smoke=args.smoke,
        reference_rows=Path(args.reference_rows) if args.reference_rows else None,
    )
    reference = {}
    for candidate in candidates:
        for packet in target_packets:
            if args.smoke:
                for member, component in (
                    ("candidate_whole", candidate["component_id"]),
                    ("control_whole", candidate["importance_control"]["component_id"]),
                ):
                    key = (packet["packet_id"], f"{candidate['estimand']}|{component}")
                    if key in reference_raw:
                        reference[(
                            packet["packet_id"], candidate["candidate_id"], member
                        )] = reference_raw[key]
            else:
                for member in ("candidate_whole", "control_whole"):
                    key = (packet["packet_id"], f"{candidate['candidate_id']}|{member}")
                    if key in reference_raw:
                        reference[(
                            packet["packet_id"], candidate["candidate_id"], member
                        )] = reference_raw[key]

    model_path = Path(args.model)
    device = pick_device(args.device)
    adapter = load_adapter(str(model_path), device, eager=False)
    model = ScreenModel(adapter, batch_size=args.batch)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phases = ("health", "target") if args.phase == "both" else (args.phase,)
    reports = {}
    resume_noops = {}
    for phase in phases:
        if phase == "target" and "health" not in reports:
            prior = output_dir / "run_manifest.health.json"
            if not prior.exists() or json.loads(prior.read_text())["status"] != "PASS":
                raise RuntimeError("stage-two target phase requires a passing health phase")
        phase_packets = health_packets if phase == "health" else target_packets
        phase_scopes = (
            {row["endpoint_key"]: {packet["packet_id"] for packet in phase_packets} for row in endpoints}
            if phase == "health" else scopes
        )
        contract = contract_hash(
            package_path, audit_path, model_path, phase, cohort["paths"],
            candidates, endpoints, args.smoke
        )
        report = run_phase(
            phase=phase, model=model, packets=phase_packets, sequences=sequences,
            pairs=pairs, candidates=candidates, endpoints=endpoints, scopes=phase_scopes,
            output_dir=output_dir, contract=contract, batch_size=args.batch,
            reference=reference,
        )
        resume_noop = bool(report.pop("_resume_noop_verified", False))
        resume_n_new = report.pop("_resume_n_new_shards", None)
        resume_noops[phase] = {
            "verified": resume_noop,
            "n_new_shards": resume_n_new if resume_noop else report["n_new_shards"],
        }
        metadata = {
            "claim_boundary": "conditional QK/OV/MLP-path measurements; inference pending",
            "smoke": args.smoke, "stage2_outcomes_loaded": True,
            "whole_validation_loaded": not args.smoke,
            "package": {"path": str(package_path), "sha256": file_sha256(package_path)},
            "package_audit": {"path": str(audit_path), "sha256": file_sha256(audit_path)},
            "stage1_package": {"path": str(stage1_path), "sha256": file_sha256(stage1_path)},
            "executor": {"path": str(Path(__file__).resolve()),
                         "sha256": file_sha256(Path(__file__).resolve())},
            "cohort_inputs": {
                key: {"path": str(path), "sha256": file_sha256(path)}
                for key, path in cohort["paths"].items()
            },
            "reference_rows": (
                None if not args.reference_rows else {
                    "path": str(Path(args.reference_rows)),
                    "sha256": file_sha256(Path(args.reference_rows)),
                }
            ),
            "model": {"path": str(model_path), "device": str(device),
                      "checkpoint_sha256": file_sha256(model_path / "pytorch_model.bin")},
            "cohort": cohort["audit"], "endpoints": endpoints,
            "candidates": candidates,
        }
        if not resume_noop:
            report.update(metadata)
            write_atomic(output_dir / f"run_manifest.{phase}.json", report)
        reports[phase] = report
        if report["n_failures"]:
            raise SystemExit(1)
    if not all(row["verified"] for row in resume_noops.values()):
        write_atomic(output_dir / "run_manifest.json", {
            "schema": "head_context_atlas.stage2_mechanism_run/1", "status": "PASS",
            "phase": args.phase, "smoke": args.smoke,
            "reports": {key: {"status": value["status"], "n_packets": value["n_packets"],
                               "n_failures": value["n_failures"]} for key, value in reports.items()},
        })
    print(json.dumps({"status": "PASS", "phase": args.phase,
                      "reports": {key: value["status"] for key, value in reports.items()},
                      "resume_noop": resume_noops}, indent=2))


if __name__ == "__main__":
    main()
