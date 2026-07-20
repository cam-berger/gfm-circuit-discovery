#!/usr/bin/env python
"""Discovery-only preflight for shortlist mechanism interventions.

Exercises QK, V, QKV, and whole-head partner-source patches on corrected donor
and acceptor masked-site packets.  The run is a plumbing test over three fixed
heads spanning early/middle/late layers; it does not rank components or name a
mechanism.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gfmcd.circuits.hf_bert import capture_head_outputs, patch_heads  # noqa: E402
from gfmcd.splicing.circuit import capture_proj, load_adapter, patch_proj, pick_device  # noqa: E402
from run_all_head_screen import ScreenModel  # noqa: E402
from run_atlas_model_preflight import (  # noqa: E402
    EXCLUDED_CHROMS,
    encode_masks,
    health_probes,
    health_scores,
    partner_destroyed,
    read_jsonl,
    select_cover_pairs,
    stable_rank,
    target_scores,
)


HEADS = ((0, 0), (3, 1), (5, 15))
IDENTITY_ATOL = 1e-4
BRIDGE_ATOL = 1e-4


def head_id(head: tuple[int, int]) -> str:
    return f"L{head[0]}H{head[1]}"


def patch_qkv(adapter, head: tuple[int, int], source: dict[str, torch.Tensor],
              positions: torch.Tensor, which: tuple[str, ...]):
    stack = contextlib.ExitStack()
    layer, index = head
    for name in which:
        stack.enter_context(patch_proj(adapter, layer, index, name, source[name], positions))
    return stack


def finite_scores(rows: list[dict]) -> bool:
    return all(
        math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key.endswith("margin") or key.endswith("logprob")
    )


def selected_packets(cohort_prefix: Path) -> tuple[list[dict], list[str], dict[str, list[dict]]]:
    packets = {
        row["packet_id"]: row
        for row in read_jsonl(Path(f"{cohort_prefix}.discovery.packets.jsonl"))
    }
    sequences = {
        row["sequence_id"]: row["sequence"]
        for row in read_jsonl(Path(f"{cohort_prefix}.discovery.sequences.jsonl"))
    }
    pairs = select_cover_pairs(
        read_jsonl(Path(f"{cohort_prefix}.discovery.pairs.jsonl")), packets
    )
    membership: dict[str, list[dict]] = defaultdict(list)
    packet_ids = set()
    for pair in pairs:
        for pair_role, field in (("case", "case_packet_id"), ("control", "control_packet_id")):
            packet_id = pair[field]
            packet_ids.add(packet_id)
            membership[packet_id].append({
                "pair_id": pair["pair_id"],
                "category": pair["category"],
                "pair_role": pair_role,
            })
    ordered = sorted(packet_ids, key=lambda packet_id: (
        packets[packet_id]["target_role"], stable_rank(packet_id), packet_id
    ))
    packet_rows = [packets[packet_id] for packet_id in ordered]
    return packet_rows, [sequences[packet["sequence_id"]] for packet in packet_rows], membership


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-prefix", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_matched_v7"
    ))
    parser.add_argument("--model", default=str(
        REPO / "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"
    ))
    parser.add_argument("--device", default=None)
    parser.add_argument("--prefix", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_mechanism_preflight_v1"
    ))
    parser.add_argument("--reference-rows", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_model_preflight_v1.rows.jsonl"
    ))
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    prefix = Path(args.prefix)
    rows_path = Path(f"{prefix}.rows.jsonl")
    summary_path = Path(f"{prefix}.summary.json")
    manifest_path = Path(f"{prefix}.manifest.json")
    for path in (rows_path, summary_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite immutable preflight {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    cohort_prefix = Path(args.cohort_prefix)
    packets, sequences, membership = selected_packets(cohort_prefix)
    if len(packets) != 4 or {packet["target_role"] for packet in packets} != {"donor", "acceptor"}:
        raise RuntimeError("mechanism preflight requires the frozen four-packet/two-role cover")
    if any(packet["analysis_split"] != "atlas_discovery" or packet["chrom"] in EXCLUDED_CHROMS
           for packet in packets):
        raise RuntimeError("non-discovery/excluded packet reached mechanism preflight")
    if any(sequence[packet["masked_base_win"]] != "G" for packet, sequence in zip(packets, sequences)):
        raise RuntimeError("target coordinate does not point to native G")

    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=False)
    model = ScreenModel(adapter, batch_size=len(packets))
    target_masks = [[packet["masked_base_win"]] for packet in packets]
    probes = [health_probes(packet, sequence) for packet, sequence in zip(packets, sequences)]
    corrupt_sequences = [
        partner_destroyed(packet, sequence) for packet, sequence in zip(packets, sequences)
    ]
    enc_native_target = encode_masks(model, sequences, target_masks)
    enc_corrupt_target = encode_masks(model, corrupt_sequences, target_masks)
    enc_native_health = encode_masks(model, sequences, probes)
    enc_corrupt_health = encode_masks(model, corrupt_sequences, probes)
    batch, seq_len = enc_native_target["input_ids"].shape
    all_positions = torch.arange(seq_len, device=adapter.device)[None].expand(batch, seq_len)

    with torch.no_grad():
        native_target_logits = model.logits(enc_native_target)
        native_health_logits = model.logits(enc_native_health)
        native_target = target_scores(model, native_target_logits, packets)
        native_health = health_scores(model, native_health_logits, sequences, probes)
        native_heads_target = capture_head_outputs(adapter, lambda: model.logits(enc_native_target))
        native_heads_health = capture_head_outputs(adapter, lambda: model.logits(enc_native_health))
        corrupt_heads_target = capture_head_outputs(adapter, lambda: model.logits(enc_corrupt_target))
        corrupt_heads_health = capture_head_outputs(adapter, lambda: model.logits(enc_corrupt_health))

        rows = []
        gates = []
        reference = {}
        reference_path = Path(args.reference_rows)
        if reference_path.exists():
            for row in read_jsonl(reference_path):
                if row["component_id"] in {head_id(head) for head in HEADS} and row["estimand"] == "B_partner_dependence":
                    reference[(row["packet_id"], row["component_id"])] = float(row["effect_margin"])

        for head in HEADS:
            layer, index = head
            native_proj_target = capture_proj(
                adapter, layer, ("query", "key", "value"), lambda: model.logits(enc_native_target)
            )
            native_proj_health = capture_proj(
                adapter, layer, ("query", "key", "value"), lambda: model.logits(enc_native_health)
            )
            corrupt_proj_target = capture_proj(
                adapter, layer, ("query", "key", "value"), lambda: model.logits(enc_corrupt_target)
            )
            corrupt_proj_health = capture_proj(
                adapter, layer, ("query", "key", "value"), lambda: model.logits(enc_corrupt_health)
            )

            with patch_qkv(adapter, head, native_proj_target, all_positions, ("query", "key", "value")):
                identity_target_logits = model.logits(enc_native_target)
            with patch_qkv(adapter, head, native_proj_health, all_positions, ("query", "key", "value")):
                identity_health_logits = model.logits(enc_native_health)
            target_identity = float((identity_target_logits - native_target_logits).abs().max())
            health_identity = float((identity_health_logits - native_health_logits).abs().max())

            condition_logits = {}
            condition_health_logits = {}
            for condition, names in {
                "QK": ("query", "key"),
                "V": ("value",),
                "QKV": ("query", "key", "value"),
            }.items():
                with patch_qkv(adapter, head, corrupt_proj_target, all_positions, names):
                    condition_logits[condition] = model.logits(enc_native_target)
                with patch_qkv(adapter, head, corrupt_proj_health, all_positions, names):
                    condition_health_logits[condition] = model.logits(enc_native_health)

            with patch_heads(adapter, corrupt_heads_target, [head]):
                whole_target_logits = model.logits(enc_native_target)
            with patch_heads(adapter, corrupt_heads_health, [head]):
                whole_health_logits = model.logits(enc_native_health)
            qkv_target_bridge = float((condition_logits["QKV"] - whole_target_logits).abs().max())
            qkv_health_bridge = float((condition_health_logits["QKV"] - whole_health_logits).abs().max())

            gates.extend([
                {"gate": f"{head_id(head)}_native_qkv_target_identity", "value": target_identity,
                 "threshold": IDENTITY_ATOL, "passed": target_identity <= IDENTITY_ATOL},
                {"gate": f"{head_id(head)}_native_qkv_health_identity", "value": health_identity,
                 "threshold": IDENTITY_ATOL, "passed": health_identity <= IDENTITY_ATOL},
                {"gate": f"{head_id(head)}_qkv_to_whole_target", "value": qkv_target_bridge,
                 "threshold": BRIDGE_ATOL, "passed": qkv_target_bridge <= BRIDGE_ATOL},
                {"gate": f"{head_id(head)}_qkv_to_whole_health", "value": qkv_health_bridge,
                 "threshold": BRIDGE_ATOL, "passed": qkv_health_bridge <= BRIDGE_ATOL},
            ])

            for condition in ("QK", "V", "QKV", "WHOLE"):
                target_logits = whole_target_logits if condition == "WHOLE" else condition_logits[condition]
                health_logits = whole_health_logits if condition == "WHOLE" else condition_health_logits[condition]
                target = target_scores(model, target_logits, packets)
                health = health_scores(model, health_logits, sequences, probes)
                for packet_index, packet in enumerate(packets):
                    effect = native_target[packet_index]["margin"] - target[packet_index]["margin"]
                    health_effect = native_health[packet_index]["mean_margin"] - health[packet_index]["mean_margin"]
                    ref = reference.get((packet["packet_id"], head_id(head)))
                    rows.append({
                        "packet_id": packet["packet_id"],
                        "sequence_id": packet["sequence_id"],
                        "gene_id": packet["gene_id"],
                        "chrom": packet["chrom"],
                        "analysis_split": packet["analysis_split"],
                        "target_role": packet["target_role"],
                        "tags": packet["tags"],
                        "selected_pair_membership": membership[packet["packet_id"]],
                        "component_id": head_id(head),
                        "layer": layer,
                        "head": index,
                        "estimand": "B",
                        "condition": condition,
                        "source": "partner_destroyed_same_packet",
                        "baseline_margin": native_target[packet_index]["margin"],
                        "intervened_margin": target[packet_index]["margin"],
                        "effect_margin": effect,
                        "baseline_logprob": native_target[packet_index]["logprob"],
                        "intervened_logprob": target[packet_index]["logprob"],
                        "baseline_health_margin": native_health[packet_index]["mean_margin"],
                        "intervened_health_margin": health[packet_index]["mean_margin"],
                        "health_margin_damage": health_effect,
                        "reference_whole_effect": ref if condition == "WHOLE" else None,
                        "reference_abs_error": abs(effect - ref) if condition == "WHOLE" and ref is not None else None,
                        "claim_boundary": "mechanism plumbing only; no ranking or router/writer claim",
                    })

        reference_errors = [
            row["reference_abs_error"] for row in rows
            if row["reference_abs_error"] is not None
        ]
        if reference_errors:
            value = max(reference_errors)
            gates.append({
                "gate": "whole_head_matches_independent_preflight",
                "value": value,
                "threshold": 1e-6,
                "passed": value <= 1e-6,
            })
        gates.append({
            "gate": "finite_mechanism_rows",
            "value": int(finite_scores(rows)),
            "threshold": 1,
            "passed": finite_scores(rows),
        })

    failures = [gate for gate in gates if not gate["passed"]]
    with rows_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema": "head_context_atlas.mechanism_preflight/1",
        "status": "PREFLIGHT_PASS" if not failures else "PREFLIGHT_FAIL",
        "claim_boundary": "QK/V/QKV/whole-head plumbing on discovery smoke; no mechanism claim",
        "n_packets": len(packets),
        "n_heads": len(HEADS),
        "heads": [head_id(head) for head in HEADS],
        "n_rows": len(rows),
        "n_gates": len(gates),
        "n_failures": len(failures),
        "gates": gates,
        "failures": failures,
        "validation_loaded": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    from gfmcd.provenance import ExperimentManifest
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_mechanism_preflight_v1",
        command=["python"] + sys.argv,
        status=summary["status"],
    )
    manifest.config.update({
        "heads": [head_id(head) for head in HEADS],
        "identity_atol": IDENTITY_ATOL,
        "bridge_atol": BRIDGE_ATOL,
        "validation_loaded": False,
    })
    for path in (
        Path(f"{cohort_prefix}.discovery.pairs.jsonl"),
        Path(f"{cohort_prefix}.discovery.packets.jsonl"),
        Path(f"{cohort_prefix}.discovery.sequences.jsonl"),
        Path(args.reference_rows),
    ):
        if path.exists():
            manifest.add_input(str(path))
    manifest.add_output(str(rows_path))
    manifest.add_output(str(summary_path))
    manifest.write(str(manifest_path))
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
