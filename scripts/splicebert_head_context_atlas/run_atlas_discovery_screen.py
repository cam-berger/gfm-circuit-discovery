#!/usr/bin/env python
"""Resumable all-96-head/all-six-MLP discovery screen for the wider atlas.

The runner evaluates every unique packet once; matched context contrasts are recovered
later by joining the immutable pair table.  It has two physically separate phases:

``health``
    A deterministic, role-balanced panel with genuinely masked non-site probes.
``target``
    Native splice-site reconstruction (A) and partner-dependence scrubbing (B).

Run health first.  No validation or sealed cohort path exists in this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_atlas_matched_cohorts import CATEGORY_ROLES, CORE_CATEGORIES  # noqa: E402
from gfmcd.circuits.hf_bert import capture_head_outputs, edit_heads, patch_heads  # noqa: E402
from gfmcd.splicing.circuit import capture_writes, clamp_write, load_adapter, pick_device  # noqa: E402
from run_all_head_screen import ScreenModel  # noqa: E402
from run_atlas_model_preflight import (  # noqa: E402
    BASES,
    HEALTH_ACCURACY_FLOOR,
    HEALTH_LOGPROB_FLOOR,
    IDENTITY_ATOL,
    component_meta,
    encode_masks,
    health_probes,
    health_scores,
    partner_destroyed,
    read_jsonl,
    stable_rank,
    target_scores,
)

EXCLUDED_CHROMS = {"chr8", "chr9", "chr11", "chr16", "chr19"}
REFERENCE_ATOL = 2e-3


def shard_name(packet_id: str) -> str:
    digest = hashlib.sha256(packet_id.encode()).hexdigest()[:20]
    return f"packet_{digest}.json"


def write_atomic(path: Path, payload: dict) -> None:
    body = json.dumps(payload, sort_keys=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(body)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def membership_by_packet(pairs: list[dict]) -> dict[str, list[dict]]:
    membership: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        for pair_role, field in (("case", "case_packet_id"), ("control", "control_packet_id")):
            membership[pair[field]].append({
                "pair_id": pair["pair_id"],
                "focal_category": pair["category"],
                "pair_role": pair_role,
                "target_role": pair["target_role"],
            })
    for packet_id in membership:
        membership[packet_id].sort(key=lambda row: (
            row["focal_category"], row["target_role"], row["pair_role"], row["pair_id"]
        ))
    return membership


def select_health_panel(packets: list[dict], pairs: list[dict], n: int) -> list[str]:
    """Role-balanced panel seeded with case/control examples from every contrast."""
    if n < 40 or n % 2:
        raise ValueError("health panel must be an even n >= 40")
    by_id = {packet["packet_id"]: packet for packet in packets}
    chosen: dict[str, set[str]] = {"donor": set(), "acceptor": set()}
    pair_blocks: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for pair in pairs:
        pair_blocks[(pair["category"], pair["target_role"])].append(pair)
    for category in CORE_CATEGORIES:
        for role in CATEGORY_ROLES[category]:
            block = pair_blocks[(category, role)]
            pair = min(block, key=lambda item: (stable_rank("health", item["pair_id"]), item["pair_id"]))
            chosen[role].update((pair["case_packet_id"], pair["control_packet_id"]))
    quota = n // 2
    for role in ("donor", "acceptor"):
        candidates = sorted(
            (packet for packet in packets if packet["target_role"] == role),
            key=lambda packet: (stable_rank("health-fill", packet["packet_id"]), packet["packet_id"]),
        )
        for packet in candidates:
            if len(chosen[role]) >= quota:
                break
            chosen[role].add(packet["packet_id"])
        if len(chosen[role]) != quota:
            raise RuntimeError(f"could not fill {role} health quota {quota}")
    selected = sorted(chosen["donor"] | chosen["acceptor"], key=lambda packet_id: (
        by_id[packet_id]["target_role"], stable_rank("health-order", packet_id), packet_id
    ))
    if len(selected) != n:
        raise RuntimeError("health panel role sets unexpectedly overlap")
    return selected


def packet_meta(packet: dict, membership: dict[str, list[dict]]) -> dict:
    return {
        "packet_id": packet["packet_id"],
        "sequence_id": packet["sequence_id"],
        "gene_id": packet["gene_id"],
        "chrom": packet["chrom"],
        "strand": packet["strand"],
        "analysis_split": packet["analysis_split"],
        "target_role": packet["target_role"],
        "masked_base_win": packet["masked_base_win"],
        "masked_native_base": packet["masked_native_base"],
        "target_anchor_win": packet["target_anchor_win"],
        "partner_anchor_win": packet["partner_anchor_win"],
        "tags": packet["tags"],
        "pair_membership": membership[packet["packet_id"]],
    }


def target_batch(model: ScreenModel, packets: list[dict], sequences: list[str],
                 membership: dict[str, list[dict]], *, self_test: bool) -> tuple[list[dict], list[dict]]:
    adapter = model.a
    heads = [(layer, head) for layer in range(adapter.num_layers) for head in range(adapter.num_heads)]
    mlps = [(layer, "mlp") for layer in range(adapter.num_layers)]
    masks = [[packet["masked_base_win"]] for packet in packets]
    native_enc = encode_masks(model, sequences, masks)
    corrupt_sequences = [partner_destroyed(packet, sequence) for packet, sequence in zip(packets, sequences)]
    corrupt_enc = encode_masks(model, corrupt_sequences, masks)
    native_logits = model.logits(native_enc)
    corrupt_logits = model.logits(corrupt_enc)
    native_scores = target_scores(model, native_logits, packets)
    corrupt_scores = target_scores(model, corrupt_logits, packets)
    gates = []

    if self_test:
        repeat = model.logits(native_enc)
        gates.append({
            "gate": "target_deterministic_repeat",
            "passed": float((repeat - native_logits).abs().max()) <= 1e-6,
            "detail": f"max_abs={float((repeat-native_logits).abs().max()):.3e}",
        })
        native_heads = capture_head_outputs(adapter, lambda: model.logits(native_enc))
        with patch_heads(adapter, native_heads, heads):
            identity = model.logits(native_enc)
        error = float((identity - native_logits).abs().max())
        gates.append({"gate": "target_all_head_source_identity", "passed": error <= IDENTITY_ATOL,
                      "detail": f"max_abs={error:.3e}"})

    native_mlp = capture_writes(adapter, mlps, lambda: model.logits(native_enc))
    if self_test:
        with clamp_write(adapter, mlps, native_mlp):
            identity = model.logits(native_enc)
        error = float((identity - native_logits).abs().max())
        gates.append({"gate": "target_all_mlp_source_identity", "passed": error <= IDENTITY_ATOL,
                      "detail": f"max_abs={error:.3e}"})
    zero_mlp = {key: torch.zeros_like(value) for key, value in native_mlp.items()}
    corrupt_heads = capture_head_outputs(adapter, lambda: model.logits(corrupt_enc))
    corrupt_mlp = capture_writes(adapter, mlps, lambda: model.logits(corrupt_enc))
    components: list[list[dict]] = [[] for _ in packets]

    for layer, head in heads:
        with edit_heads(adapter, [(layer, head)], mode="zero"):
            zero_logits = model.logits(native_enc)
        with patch_heads(adapter, corrupt_heads, [(layer, head)]):
            scrub_logits = model.logits(native_enc)
        zero_scores = target_scores(model, zero_logits, packets)
        scrub_scores = target_scores(model, scrub_logits, packets)
        meta = component_meta("head", layer, head)
        for index in range(len(packets)):
            components[index].append({
                **meta,
                "A_effect_margin": native_scores[index]["margin"] - zero_scores[index]["margin"],
                "A_effect_logprob": native_scores[index]["logprob"] - zero_scores[index]["logprob"],
                "B_effect_margin": native_scores[index]["margin"] - scrub_scores[index]["margin"],
                "B_effect_logprob": native_scores[index]["logprob"] - scrub_scores[index]["logprob"],
            })
    for layer, kind in mlps:
        with clamp_write(adapter, [(layer, kind)], zero_mlp):
            zero_logits = model.logits(native_enc)
        with clamp_write(adapter, [(layer, kind)], corrupt_mlp):
            scrub_logits = model.logits(native_enc)
        zero_scores = target_scores(model, zero_logits, packets)
        scrub_scores = target_scores(model, scrub_logits, packets)
        meta = component_meta("mlp", layer)
        for index in range(len(packets)):
            components[index].append({
                **meta,
                "A_effect_margin": native_scores[index]["margin"] - zero_scores[index]["margin"],
                "A_effect_logprob": native_scores[index]["logprob"] - zero_scores[index]["logprob"],
                "B_effect_margin": native_scores[index]["margin"] - scrub_scores[index]["margin"],
                "B_effect_logprob": native_scores[index]["logprob"] - scrub_scores[index]["logprob"],
            })

    records = []
    for index, packet in enumerate(packets):
        finite = all(
            math.isfinite(component[key])
            for component in components[index]
            for key in ("A_effect_margin", "A_effect_logprob", "B_effect_margin", "B_effect_logprob")
        )
        gates.append({"gate": f"target_component_grid:{packet['packet_id']}",
                      "passed": len(components[index]) == 102 and finite,
                      "detail": f"n={len(components[index])}, finite={finite}"})
        records.append({
            "schema": "head_context_atlas.discovery_target_shard/1",
            "claim_boundary": "discovery component matrix; screening does not name head function",
            "packet": packet_meta(packet, membership),
            "native_target": native_scores[index],
            "partner_destroyed_target": corrupt_scores[index],
            "behavioral_partner_effect_margin": (
                native_scores[index]["margin"] - corrupt_scores[index]["margin"]
            ),
            "components": components[index],
        })
    return records, gates


def health_batch(model: ScreenModel, packets: list[dict], sequences: list[str],
                 membership: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    adapter = model.a
    heads = [(layer, head) for layer in range(adapter.num_layers) for head in range(adapter.num_heads)]
    mlps = [(layer, "mlp") for layer in range(adapter.num_layers)]
    probes = [health_probes(packet, sequence) for packet, sequence in zip(packets, sequences)]
    native_enc = encode_masks(model, sequences, probes)
    corrupt_sequences = [partner_destroyed(packet, sequence) for packet, sequence in zip(packets, sequences)]
    corrupt_enc = encode_masks(model, corrupt_sequences, probes)
    native_logits = model.logits(native_enc)
    native_scores = health_scores(model, native_logits, sequences, probes)
    native_mlp = capture_writes(adapter, mlps, lambda: model.logits(native_enc))
    zero_mlp = {key: torch.zeros_like(value) for key, value in native_mlp.items()}
    corrupt_heads = capture_head_outputs(adapter, lambda: model.logits(corrupt_enc))
    corrupt_mlp = capture_writes(adapter, mlps, lambda: model.logits(corrupt_enc))
    components: list[list[dict]] = [[] for _ in packets]

    for layer, head in heads:
        with edit_heads(adapter, [(layer, head)], mode="zero"):
            zero_logits = model.logits(native_enc)
        with patch_heads(adapter, corrupt_heads, [(layer, head)]):
            scrub_logits = model.logits(native_enc)
        zero_scores = health_scores(model, zero_logits, sequences, probes)
        scrub_scores = health_scores(model, scrub_logits, sequences, probes)
        meta = component_meta("head", layer, head)
        for index in range(len(packets)):
            components[index].append({
                **meta,
                "A_health_logprob_damage": native_scores[index]["mean_logprob"] - zero_scores[index]["mean_logprob"],
                "A_health_margin_damage": native_scores[index]["mean_margin"] - zero_scores[index]["mean_margin"],
                "A_health_accuracy_damage": native_scores[index]["accuracy"] - zero_scores[index]["accuracy"],
                "B_health_logprob_damage": native_scores[index]["mean_logprob"] - scrub_scores[index]["mean_logprob"],
                "B_health_margin_damage": native_scores[index]["mean_margin"] - scrub_scores[index]["mean_margin"],
                "B_health_accuracy_damage": native_scores[index]["accuracy"] - scrub_scores[index]["accuracy"],
            })
    for layer, kind in mlps:
        with clamp_write(adapter, [(layer, kind)], zero_mlp):
            zero_logits = model.logits(native_enc)
        with clamp_write(adapter, [(layer, kind)], corrupt_mlp):
            scrub_logits = model.logits(native_enc)
        zero_scores = health_scores(model, zero_logits, sequences, probes)
        scrub_scores = health_scores(model, scrub_logits, sequences, probes)
        meta = component_meta("mlp", layer)
        for index in range(len(packets)):
            components[index].append({
                **meta,
                "A_health_logprob_damage": native_scores[index]["mean_logprob"] - zero_scores[index]["mean_logprob"],
                "A_health_margin_damage": native_scores[index]["mean_margin"] - zero_scores[index]["mean_margin"],
                "A_health_accuracy_damage": native_scores[index]["accuracy"] - zero_scores[index]["accuracy"],
                "B_health_logprob_damage": native_scores[index]["mean_logprob"] - scrub_scores[index]["mean_logprob"],
                "B_health_margin_damage": native_scores[index]["mean_margin"] - scrub_scores[index]["mean_margin"],
                "B_health_accuracy_damage": native_scores[index]["accuracy"] - scrub_scores[index]["accuracy"],
            })

    gates = []
    records = []
    for index, packet in enumerate(packets):
        finite = all(
            math.isfinite(component[key])
            for component in components[index]
            for key in (
                "A_health_logprob_damage", "A_health_margin_damage",
                "B_health_logprob_damage", "B_health_margin_damage",
            )
        )
        gates.append({"gate": f"health_component_grid:{packet['packet_id']}",
                      "passed": len(components[index]) == 102 and finite,
                      "detail": f"n={len(components[index])}, finite={finite}"})
        records.append({
            "schema": "head_context_atlas.discovery_health_shard/1",
            "claim_boundary": "masked collateral-health panel; not a biological-context outcome",
            "packet": packet_meta(packet, membership),
            "probe_positions": probes[index],
            "native_health": native_scores[index],
            "components": components[index],
        })
    return records, gates


def load_reference_rows(path: Path) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], dict] = defaultdict(dict)
    for row in read_jsonl(path):
        grouped[(row["packet_id"], row["component_id"])][row["estimand"]] = row
    return grouped


def verify_reference(out_dir: Path, packet_ids: list[str], reference_path: Path) -> dict:
    reference = load_reference_rows(reference_path)
    differences = []
    for packet_id in packet_ids:
        target = json.loads((out_dir / "target_shards" / shard_name(packet_id)).read_text())
        health = json.loads((out_dir / "health_shards" / shard_name(packet_id)).read_text())
        health_by_component = {row["component_id"]: row for row in health["components"]}
        for component in target["components"]:
            key = packet_id, component["component_id"]
            if key not in reference:
                differences.append((packet_id, component["component_id"], "missing_reference", math.inf))
                continue
            a = reference[key]["A_native_reconstruction"]
            b = reference[key]["B_partner_dependence"]
            h = health_by_component[component["component_id"]]
            comparisons = {
                "A_effect_margin": (component["A_effect_margin"], a["effect_margin"]),
                "B_effect_margin": (component["B_effect_margin"], b["effect_margin"]),
                "A_health_logprob": (h["A_health_logprob_damage"], a["health_logprob_damage"]),
                "B_health_logprob": (h["B_health_logprob_damage"], b["health_logprob_damage"]),
                "A_health_margin": (h["A_health_margin_damage"], a["health_margin_damage"]),
                "B_health_margin": (h["B_health_margin_damage"], b["health_margin_damage"]),
            }
            for metric, (observed, expected) in comparisons.items():
                differences.append((packet_id, component["component_id"], metric, abs(observed - expected)))
    max_item = max(differences, key=lambda item: item[-1], default=(None, None, None, 0.0))
    return {
        "reference": str(reference_path),
        "n_comparisons": len(differences),
        "max_abs_difference": max_item[-1],
        "max_difference_location": list(max_item[:-1]),
        "tolerance": REFERENCE_ATOL,
        "gate": max_item[-1] <= REFERENCE_ATOL,
    }


def run_phase(phase: str, model: ScreenModel, packets: list[dict], sequences: dict[str, str],
              membership: dict[str, list[dict]], out_dir: Path, batch_size: int,
              gates: list[dict]) -> dict:
    shard_dir = out_dir / f"{phase}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    expected = {packet["packet_id"]: shard_name(packet["packet_id"]) for packet in packets}
    pending = [packet for packet in packets if not (shard_dir / expected[packet["packet_id"]]).exists()]
    if model.a.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.a.device)
    started = time.time()
    written = 0
    for offset in range(0, len(pending), batch_size):
        batch_packets = pending[offset:offset + batch_size]
        batch_sequences = [sequences[packet["sequence_id"]] for packet in batch_packets]
        if phase == "target":
            records, batch_gates = target_batch(
                model, batch_packets, batch_sequences, membership,
                self_test=(offset == 0 and not any((shard_dir).glob("packet_*.json"))),
            )
        else:
            records, batch_gates = health_batch(model, batch_packets, batch_sequences, membership)
        gates.extend(batch_gates)
        failures = [gate for gate in batch_gates if not gate["passed"]]
        if failures:
            raise RuntimeError(f"{phase} mechanical gate failure: {failures[:2]}")
        for packet, record in zip(batch_packets, records):
            write_atomic(shard_dir / expected[packet["packet_id"]], record)
            written += 1
        print(
            f"[atlas-screen] {phase} {min(offset+len(batch_packets),len(pending))}/{len(pending)} "
            f"new ({time.time()-started:.1f}s)", file=sys.stderr, flush=True,
        )
    missing = [name for name in expected.values() if not (shard_dir / name).exists()]
    if missing:
        raise RuntimeError(f"{phase} phase missing {len(missing)} expected shards")
    seconds = time.time() - started
    shard_bytes = sum((shard_dir / name).stat().st_size for name in expected.values())
    return {
        "n_expected": len(expected),
        "n_written_this_invocation": written,
        "n_existing_or_written": len(expected) - len(missing),
        "seconds_this_invocation": seconds,
        "seconds_per_new_packet": seconds / written if written else None,
        "peak_gpu_memory_gb": (
            float(torch.cuda.max_memory_allocated(model.a.device) / 1e9)
            if model.a.device.type == "cuda" else None
        ),
        "shard_bytes": shard_bytes,
        "mean_shard_bytes": shard_bytes / len(expected) if expected else None,
        "shard_dir": str(shard_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-prefix", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_matched_v7"
    ))
    parser.add_argument("--model", default=str(
        REPO / "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"
    ))
    parser.add_argument("--out", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_discovery_screen_v1"
    ))
    parser.add_argument("--device", default=None)
    parser.add_argument("--phase", choices=("health", "target", "both"), default="health")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--health-n", type=int, default=512)
    parser.add_argument("--n", type=int, default=None, help="stable packet prefix; smoke only")
    parser.add_argument("--packet-ids-from-summary", default=None,
                        help="use selection.packet_ids from a model-preflight summary")
    parser.add_argument("--reference-rows", default=None,
                        help="after both phases, compare against preflight rows")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")
    if args.reference_rows and args.phase != "both":
        raise ValueError("--reference-rows requires --phase both")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cohort_prefix = Path(args.cohort_prefix)
    pair_path = Path(f"{cohort_prefix}.discovery.pairs.jsonl")
    packet_path = Path(f"{cohort_prefix}.discovery.packets.jsonl")
    sequence_path = Path(f"{cohort_prefix}.discovery.sequences.jsonl")
    pairs = read_jsonl(pair_path)
    packets = read_jsonl(packet_path)
    full_target_packet_count = len(packets)
    sequence_rows = read_jsonl(sequence_path)
    sequences = {row["sequence_id"]: row["sequence"] for row in sequence_rows}
    membership = membership_by_packet(pairs)
    if any(packet["analysis_split"] != "atlas_discovery" for packet in packets):
        raise RuntimeError("non-discovery packet reached discovery-only runner")
    if any(packet["chrom"] in EXCLUDED_CHROMS for packet in packets):
        raise RuntimeError("sealed/touched chromosome reached discovery-only runner")
    if {packet["packet_id"] for packet in packets} != set(membership):
        raise RuntimeError("packet file and matched-pair membership disagree")
    if any(packet["sequence_id"] not in sequences for packet in packets):
        raise RuntimeError("packet lacks sequence")

    packets.sort(key=lambda packet: (stable_rank("screen", packet["packet_id"]), packet["packet_id"]))
    if args.packet_ids_from_summary:
        selection = json.loads(Path(args.packet_ids_from_summary).read_text())["selection"]["packet_ids"]
        by_id = {packet["packet_id"]: packet for packet in packets}
        packets = [by_id[packet_id] for packet_id in selection]
    elif args.n is not None:
        packets = packets[:args.n]
    selected_ids = {packet["packet_id"] for packet in packets}
    selected_pairs = [pair for pair in pairs if (
        pair["case_packet_id"] in selected_ids or pair["control_packet_id"] in selected_ids
    )]

    if args.packet_ids_from_summary or args.n is not None:
        health_packets = packets
    else:
        panel_ids = set(select_health_panel(packets, selected_pairs, args.health_n))
        health_packets = [packet for packet in packets if packet["packet_id"] in panel_ids]

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=False)
    model = ScreenModel(adapter, batch_size=args.batch)
    if adapter.num_layers != 6 or adapter.num_heads != 16:
        raise RuntimeError("expected SpliceBERT 6x16 architecture")

    gates: list[dict] = []
    phase_reports = {}
    order = ("health", "target") if args.phase == "both" else (args.phase,)
    for phase in order:
        phase_packets = health_packets if phase == "health" else packets
        phase_reports[phase] = run_phase(
            phase, model, phase_packets, sequences, membership, output_dir, args.batch, gates
        )

    health_summary = None
    if "health" in order:
        health_records = [
            json.loads((output_dir / "health_shards" / shard_name(packet["packet_id"])).read_text())
            for packet in health_packets
        ]
        accuracy = float(np.mean([record["native_health"]["accuracy"] for record in health_records]))
        logprob = float(np.mean([record["native_health"]["mean_logprob"] for record in health_records]))
        health_summary = {
            "n_packets": len(health_records),
            "mean_baseline_accuracy": accuracy,
            "mean_baseline_logprob": logprob,
            "accuracy_floor": HEALTH_ACCURACY_FLOOR,
            "logprob_floor": HEALTH_LOGPROB_FLOOR,
            "gate": accuracy >= HEALTH_ACCURACY_FLOOR and logprob >= HEALTH_LOGPROB_FLOOR,
        }
        gates.append({"gate": "aggregate_masked_health", "passed": health_summary["gate"],
                      "detail": f"accuracy={accuracy:.3f}; logprob={logprob:.3f}"})

    reference = None
    if args.reference_rows:
        reference = verify_reference(output_dir, [packet["packet_id"] for packet in packets],
                                     Path(args.reference_rows))
        gates.append({"gate": "cross_device_preflight_reference", "passed": reference["gate"],
                      "detail": f"max_abs={reference['max_abs_difference']:.3e}"})

    failures = [gate for gate in gates if not gate["passed"]]
    status = "COMPLETE" if not failures else "FAILED_GATE"
    manifest = {
        "schema": "head_context_atlas.discovery_screen_run/1",
        "status": status,
        "claim_boundary": "discovery component screen; requires matched analysis and causal validation",
        "cohort": {
            "prefix": str(cohort_prefix),
            "n_target_packets": len(packets),
            "n_health_packets": len(health_packets),
            "full_discovery_target_packets": full_target_packet_count,
            "target_roles": dict(Counter(packet["target_role"] for packet in packets)),
            "health_target_roles": dict(Counter(packet["target_role"] for packet in health_packets)),
            "excluded_chromosomes": sorted(EXCLUDED_CHROMS),
            "validation_loaded": False,
        },
        "model": {
            "path": str(Path(args.model)), "device": str(device),
            "layers": adapter.num_layers, "heads_per_layer": adapter.num_heads,
            "precision": "float32; TF32/AMP off",
        },
        "phase": args.phase,
        "phase_reports": phase_reports,
        "health": health_summary,
        "reference_verification": reference,
        "gates": gates,
        "n_failures": len(failures),
        "failures": failures,
        "inputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in (pair_path, packet_path, sequence_path, Path(args.model) / "pytorch_model.bin")
        ],
        "shards": {},
        "seeds": {"PYTHONHASHSEED": 0, "torch": 0, "numpy": 0},
    }
    for phase, phase_report in phase_reports.items():
        seconds_per_packet = phase_report["seconds_per_new_packet"]
        projection_n = full_target_packet_count if phase == "target" else args.health_n
        phase_report["projected_complete_n"] = projection_n
        phase_report["projected_gpu_hours"] = (
            seconds_per_packet * projection_n / 3600 if seconds_per_packet is not None else None
        )
        phase_report["projected_storage_mb"] = (
            phase_report["mean_shard_bytes"] * projection_n / 1e6
            if phase_report["mean_shard_bytes"] is not None else None
        )
    for phase in order:
        shard_dir = output_dir / f"{phase}_shards"
        manifest["shards"][phase] = {
            path.name: {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in sorted(shard_dir.glob("packet_*.json"))
        }
    write_atomic(output_dir / f"run_manifest.{args.phase}.json", manifest)
    # Convenience pointer to the most recent invocation; phase-specific manifests keep
    # the health record available during the recommended health-then-target sequence.
    write_atomic(output_dir / "run_manifest.json", manifest)
    print(json.dumps({
        "status": status,
        "phase": args.phase,
        "n_target_packets": len(packets),
        "n_health_packets": len(health_packets),
        "n_failures": len(failures),
        "reference_gate": reference["gate"] if reference else None,
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
