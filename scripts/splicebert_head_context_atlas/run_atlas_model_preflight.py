#!/usr/bin/env python
"""Small, discovery-only mechanical preflight for the wider Head--Context Atlas.

The run deliberately measures only four matched packets: one donor case/control pair
and one acceptor case/control pair chosen to cover all six overlapping core labels.
It exercises every one of SpliceBERT's 96 attention heads and six MLP writes under:

* A: zero-ablation effect on native masked-base reconstruction; and
* B: scrub-to-partner-destroyed effect on the same reconstruction.

Every intervention is also evaluated on genuinely masked, non-site MLM probes.  This is
an instrument/plumbing test, not a context-selectivity analysis and not a shortlist.
Only the outcome-independent discovery cohort is read.
"""
from __future__ import annotations

import argparse
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

from build_atlas_matched_cohorts import CORE_CATEGORIES  # noqa: E402
from gfmcd.circuits.hf_bert import capture_head_outputs, edit_heads, patch_heads  # noqa: E402
from gfmcd.splicing.circuit import capture_writes, clamp_write, load_adapter, pick_device  # noqa: E402
from gfmcd.splicing.coupling import destroy_acceptor, destroy_donor  # noqa: E402
from run_all_head_screen import BASES, ScreenModel, gate_destroy_real_bases, metrics_from_logits  # noqa: E402

EXCLUDED_CHROMS = {"chr8", "chr9", "chr11", "chr16", "chr19"}
N_HEALTH_PROBES = 4
IDENTITY_ATOL = 1e-4
HEALTH_ACCURACY_FLOOR = 0.30
HEALTH_LOGPROB_FLOOR = -math.log(4.0)


def stable_rank(*parts: object) -> int:
    text = "|".join(map(str, parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_cover_pairs(pairs: list[dict], packets: dict[str, dict]) -> list[dict]:
    """Choose one donor and one acceptor pair maximizing six-tag coverage."""
    core = set(CORE_CATEGORIES)
    best_by_signature: dict[tuple[str, tuple[str, ...]], dict] = {}
    for pair in pairs:
        case = packets[pair["case_packet_id"]]
        signature = tuple(sorted(set(case["tags"]) & core))
        key = pair["target_role"], signature
        previous = best_by_signature.get(key)
        if previous is None or stable_rank(pair["pair_id"]) < stable_rank(previous["pair_id"]):
            best_by_signature[key] = pair
    donors = [(set(sig), pair) for (role, sig), pair in best_by_signature.items() if role == "donor"]
    acceptors = [(set(sig), pair) for (role, sig), pair in best_by_signature.items() if role == "acceptor"]
    if not donors or not acceptors:
        raise RuntimeError("preflight requires at least one donor and one acceptor contrast")
    candidates = []
    for donor_tags, donor_pair in donors:
        for acceptor_tags, acceptor_pair in acceptors:
            union = donor_tags | acceptor_tags
            candidates.append((
                -len(union),
                -(len(donor_tags) + len(acceptor_tags)),
                stable_rank(donor_pair["pair_id"], acceptor_pair["pair_id"]),
                donor_pair,
                acceptor_pair,
                union,
            ))
    chosen = min(candidates)
    if chosen[-1] != core:
        raise RuntimeError(f"two-role preflight cannot cover all core tags: {sorted(chosen[-1])}")
    return [chosen[3], chosen[4]]


def health_probes(packet: dict, sequence: str, n: int = N_HEALTH_PROBES) -> list[int]:
    """Deterministic non-site positions, well separated from the focal junction motifs."""
    protected = {
        packet["donor_win"], packet["donor_win"] + 1,
        packet["acceptor_win"] - 2, packet["acceptor_win"] - 1,
    }
    candidates = [
        position for position in range(32, len(sequence) - 32)
        if all(abs(position - site) > 12 for site in protected)
    ]
    candidates.sort(key=lambda position: (stable_rank(packet["packet_id"], "health", position), position))
    selected = sorted(candidates[:n])
    if len(selected) != n or any(sequence[position] not in BASES for position in selected):
        raise RuntimeError(f"could not construct {n} health probes for {packet['packet_id']}")
    return selected


def encode_masks(model: ScreenModel, sequences: list[str], masks: list[list[int]]):
    adapter = model.a
    enc = adapter.tokenizer(
        [" ".join(sequence) for sequence in sequences],
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(adapter.device)
    for row, positions in enumerate(masks):
        for position in positions:
            enc["input_ids"][row, position + model.tok_offset] = model.mask_id
    return enc


def target_scores(model: ScreenModel, logits: torch.Tensor, packets: list[dict]) -> list[dict]:
    out = []
    for row, packet in enumerate(packets):
        position = packet["masked_base_win"]
        native_id = model.base_ids[BASES.index(packet["masked_native_base"])]
        out.append(metrics_from_logits(
            logits[row, position + model.tok_offset], native_id, model.base_ids
        ))
    return out


def health_scores(model: ScreenModel, logits: torch.Tensor, sequences: list[str],
                  probes: list[list[int]]) -> list[dict]:
    out = []
    for row, positions in enumerate(probes):
        logprobs, margins, correct = [], [], []
        for position in positions:
            native_id = model.base_ids[BASES.index(sequences[row][position])]
            metric = metrics_from_logits(
                logits[row, position + model.tok_offset], native_id, model.base_ids
            )
            logprobs.append(metric["logprob"])
            margins.append(metric["margin"])
            four = logits[row, position + model.tok_offset, model.base_ids]
            correct.append(int(model.base_ids[int(four.argmax())] == native_id))
        out.append({
            "mean_logprob": float(np.mean(logprobs)),
            "mean_margin": float(np.mean(margins)),
            "accuracy": float(np.mean(correct)),
            "n_probes": len(positions),
        })
    return out


def partner_destroyed(packet: dict, sequence: str) -> str:
    partner_is_donor = packet["target_role"] == "acceptor"
    position = packet["partner_anchor_win"]
    variants = (destroy_donor if partner_is_donor else destroy_acceptor)(sequence, position)
    damaged = variants[0]
    ok, message = gate_destroy_real_bases(sequence, damaged, position, partner_is_donor)
    if not ok:
        raise RuntimeError(f"invalid partner corruption for {packet['packet_id']}: {message}")
    if damaged[packet["masked_base_win"]] != "G":
        raise RuntimeError(f"partner corruption changed target base for {packet['packet_id']}")
    return damaged


def component_meta(component_type: str, layer: int, head: int | None = None) -> dict:
    if component_type == "head":
        return {
            "component_type": "attention_head",
            "component_id": f"L{layer}H{head}",
            "layer": layer,
            "head": head,
        }
    return {
        "component_type": "mlp_write",
        "component_id": f"L{layer}MLP",
        "layer": layer,
        "head": None,
    }


def add_rows(rows: list[dict], *, estimand: str, intervention: str,
             component: dict, packets: list[dict], membership: dict[str, list[dict]],
             baseline_target: list[dict], intervened_target: list[dict],
             baseline_health: list[dict], intervened_health: list[dict],
             counterfactual_target: list[dict] | None = None) -> None:
    for index, packet in enumerate(packets):
        base_t, int_t = baseline_target[index], intervened_target[index]
        base_h, int_h = baseline_health[index], intervened_health[index]
        rows.append({
            "packet_id": packet["packet_id"],
            "sequence_id": packet["sequence_id"],
            "gene_id": packet["gene_id"],
            "chrom": packet["chrom"],
            "analysis_split": packet["analysis_split"],
            "target_role": packet["target_role"],
            "masked_base_win": packet["masked_base_win"],
            "masked_native_base": packet["masked_native_base"],
            "tags": packet["tags"],
            "selected_pair_membership": membership[packet["packet_id"]],
            "estimand": estimand,
            "intervention": intervention,
            **component,
            "baseline_margin": base_t["margin"],
            "intervened_margin": int_t["margin"],
            "effect_margin": base_t["margin"] - int_t["margin"],
            "baseline_logprob": base_t["logprob"],
            "intervened_logprob": int_t["logprob"],
            "effect_logprob": base_t["logprob"] - int_t["logprob"],
            "counterfactual_margin": (
                counterfactual_target[index]["margin"] if counterfactual_target else None
            ),
            "behavioral_partner_effect": (
                base_t["margin"] - counterfactual_target[index]["margin"]
                if counterfactual_target else None
            ),
            "health_probe_count": base_h["n_probes"],
            "baseline_health_logprob": base_h["mean_logprob"],
            "intervened_health_logprob": int_h["mean_logprob"],
            "health_logprob_damage": base_h["mean_logprob"] - int_h["mean_logprob"],
            "baseline_health_margin": base_h["mean_margin"],
            "intervened_health_margin": int_h["mean_margin"],
            "health_margin_damage": base_h["mean_margin"] - int_h["mean_margin"],
            "baseline_health_accuracy": base_h["accuracy"],
            "intervened_health_accuracy": int_h["accuracy"],
            "health_accuracy_damage": base_h["accuracy"] - int_h["accuracy"],
            "claim_boundary": "mechanical discovery preflight; not a context/head-function claim",
        })


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
        REPO / "results/splicebert_head_context_atlas/atlas_model_preflight_v1"
    ))
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cohort_prefix = Path(args.cohort_prefix)
    output_prefix = Path(args.prefix)
    rows_path = Path(f"{output_prefix}.rows.jsonl")
    summary_path = Path(f"{output_prefix}.summary.json")
    manifest_path = Path(f"{output_prefix}.manifest.json")
    for path in (rows_path, summary_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite immutable artifact {path}")

    pair_path = Path(f"{cohort_prefix}.discovery.pairs.jsonl")
    packet_path = Path(f"{cohort_prefix}.discovery.packets.jsonl")
    sequence_path = Path(f"{cohort_prefix}.discovery.sequences.jsonl")
    all_packets = {row["packet_id"]: row for row in read_jsonl(packet_path)}
    all_sequences = {row["sequence_id"]: row for row in read_jsonl(sequence_path)}
    selected_pairs = select_cover_pairs(read_jsonl(pair_path), all_packets)
    membership: dict[str, list[dict]] = defaultdict(list)
    packet_ids = []
    for pair in selected_pairs:
        for pair_role, key in (("case", "case_packet_id"), ("control", "control_packet_id")):
            packet_id = pair[key]
            packet_ids.append(packet_id)
            membership[packet_id].append({
                "pair_id": pair["pair_id"],
                "focal_category": pair["category"],
                "pair_role": pair_role,
            })
    packet_ids = sorted(set(packet_ids), key=lambda packet_id: (
        all_packets[packet_id]["target_role"], stable_rank(packet_id), packet_id
    ))
    packets = [all_packets[packet_id] for packet_id in packet_ids]
    sequences = [all_sequences[packet["sequence_id"]]["sequence"] for packet in packets]
    probes = [health_probes(packet, sequence) for packet, sequence in zip(packets, sequences)]

    gates = []
    def gate(name: str, passed: bool, detail: str) -> None:
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    gate("four_unique_packets", len(packets) == 4, f"n={len(packets)}")
    gate("both_target_roles", {packet["target_role"] for packet in packets} == {"donor", "acceptor"},
         str(sorted({packet["target_role"] for packet in packets})))
    covered = set().union(*(set(all_packets[pair["case_packet_id"]]["tags"]) for pair in selected_pairs))
    gate("six_core_tags_covered", set(CORE_CATEGORIES) <= covered,
         str(sorted(set(CORE_CATEGORIES) & covered)))
    gate("discovery_only", all(packet["analysis_split"] == "atlas_discovery" for packet in packets), "")
    gate("excluded_chromosomes_absent", all(packet["chrom"] not in EXCLUDED_CHROMS for packet in packets), "")
    gate("coordinate_contract", all(
        packet["masked_base_win"] == (
            packet["donor_win"] if packet["target_role"] == "donor" else packet["acceptor_win"] - 1
        ) and sequence[packet["masked_base_win"]] == "G"
        for packet, sequence in zip(packets, sequences)
    ), "donor=donor_win; acceptor=acceptor_win-1; native=G")
    gate("health_probes_masked_non_site", all(len(item) == N_HEALTH_PROBES for item in probes),
         f"{N_HEALTH_PROBES} per packet")

    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=False)
    model = ScreenModel(adapter, batch_size=len(packets))
    gate("architecture", adapter.num_layers == 6 and adapter.num_heads == 16,
         f"layers={adapter.num_layers}, heads={adapter.num_heads}")
    heads = [(layer, head) for layer in range(adapter.num_layers) for head in range(adapter.num_heads)]
    mlps = [(layer, "mlp") for layer in range(adapter.num_layers)]
    for sequence in sequences:
        adapter.tokenize(sequence)
    gate("single_nt_tokenization", True, f"prefix_tokens={adapter.prefix_tokens}")

    target_masks = [[packet["masked_base_win"]] for packet in packets]
    enc_target = encode_masks(model, sequences, target_masks)
    enc_health = encode_masks(model, sequences, probes)
    start = time.time()
    baseline_target_logits = model.logits(enc_target)
    repeat_target_logits = model.logits(enc_target)
    baseline_health_logits = model.logits(enc_health)
    gate("deterministic_repeat", float((baseline_target_logits - repeat_target_logits).abs().max()) <= 1e-6,
         f"max_abs={float((baseline_target_logits - repeat_target_logits).abs().max()):.3e}")
    baseline_target = target_scores(model, baseline_target_logits, packets)
    baseline_health = health_scores(model, baseline_health_logits, sequences, probes)
    mean_health_accuracy = float(np.mean([item["accuracy"] for item in baseline_health]))
    mean_health_logprob = float(np.mean([item["mean_logprob"] for item in baseline_health]))
    gate("masked_health_accuracy", mean_health_accuracy >= HEALTH_ACCURACY_FLOOR,
         f"mean={mean_health_accuracy:.3f}, floor={HEALTH_ACCURACY_FLOOR:.2f}")
    gate("masked_health_logprob", mean_health_logprob >= HEALTH_LOGPROB_FLOOR,
         f"mean={mean_health_logprob:.3f}, floor={HEALTH_LOGPROB_FLOOR:.3f}")
    gate("finite_baselines", all(
        math.isfinite(value)
        for item in baseline_target for value in (item["margin"], item["logprob"], item["entropy"])
    ), "")

    native_heads = capture_head_outputs(adapter, lambda: model.logits(enc_target))
    with patch_heads(adapter, native_heads, heads):
        identity_head_logits = model.logits(enc_target)
    head_identity_error = float((identity_head_logits - baseline_target_logits).abs().max())
    gate("all_head_source_identity", head_identity_error <= IDENTITY_ATOL,
         f"max_abs={head_identity_error:.3e}")
    native_mlp_target = capture_writes(adapter, mlps, lambda: model.logits(enc_target))
    with clamp_write(adapter, mlps, native_mlp_target):
        identity_mlp_logits = model.logits(enc_target)
    mlp_identity_error = float((identity_mlp_logits - baseline_target_logits).abs().max())
    gate("all_mlp_source_identity", mlp_identity_error <= IDENTITY_ATOL,
         f"max_abs={mlp_identity_error:.3e}")
    native_mlp_health = capture_writes(adapter, mlps, lambda: model.logits(enc_health))

    rows: list[dict] = []
    for layer, head in heads:
        with edit_heads(adapter, [(layer, head)], mode="zero"):
            target_logits = model.logits(enc_target)
            health_logits = model.logits(enc_health)
        add_rows(
            rows, estimand="A_native_reconstruction", intervention="zero_attention_head_output",
            component=component_meta("head", layer, head), packets=packets, membership=membership,
            baseline_target=baseline_target, intervened_target=target_scores(model, target_logits, packets),
            baseline_health=baseline_health,
            intervened_health=health_scores(model, health_logits, sequences, probes),
        )
    for layer, kind in mlps:
        zero_target = {key: torch.zeros_like(value) for key, value in native_mlp_target.items()}
        zero_health = {key: torch.zeros_like(value) for key, value in native_mlp_health.items()}
        with clamp_write(adapter, [(layer, kind)], zero_target):
            target_logits = model.logits(enc_target)
        with clamp_write(adapter, [(layer, kind)], zero_health):
            health_logits = model.logits(enc_health)
        add_rows(
            rows, estimand="A_native_reconstruction", intervention="zero_mlp_residual_write",
            component=component_meta("mlp", layer), packets=packets, membership=membership,
            baseline_target=baseline_target, intervened_target=target_scores(model, target_logits, packets),
            baseline_health=baseline_health,
            intervened_health=health_scores(model, health_logits, sequences, probes),
        )

    corrupt_sequences = [
        partner_destroyed(packet, sequence) for packet, sequence in zip(packets, sequences)
    ]
    enc_corrupt_target = encode_masks(model, corrupt_sequences, target_masks)
    enc_corrupt_health = encode_masks(model, corrupt_sequences, probes)
    corrupt_target_logits = model.logits(enc_corrupt_target)
    corrupt_target = target_scores(model, corrupt_target_logits, packets)
    corrupt_head_target = capture_head_outputs(adapter, lambda: model.logits(enc_corrupt_target))
    corrupt_head_health = capture_head_outputs(adapter, lambda: model.logits(enc_corrupt_health))
    corrupt_mlp_target = capture_writes(adapter, mlps, lambda: model.logits(enc_corrupt_target))
    corrupt_mlp_health = capture_writes(adapter, mlps, lambda: model.logits(enc_corrupt_health))
    for layer, head in heads:
        with patch_heads(adapter, corrupt_head_target, [(layer, head)]):
            target_logits = model.logits(enc_target)
        with patch_heads(adapter, corrupt_head_health, [(layer, head)]):
            health_logits = model.logits(enc_health)
        add_rows(
            rows, estimand="B_partner_dependence", intervention="scrub_head_to_partner_destroyed_source",
            component=component_meta("head", layer, head), packets=packets, membership=membership,
            baseline_target=baseline_target, intervened_target=target_scores(model, target_logits, packets),
            baseline_health=baseline_health,
            intervened_health=health_scores(model, health_logits, sequences, probes),
            counterfactual_target=corrupt_target,
        )
    for layer, kind in mlps:
        with clamp_write(adapter, [(layer, kind)], corrupt_mlp_target):
            target_logits = model.logits(enc_target)
        with clamp_write(adapter, [(layer, kind)], corrupt_mlp_health):
            health_logits = model.logits(enc_health)
        add_rows(
            rows, estimand="B_partner_dependence", intervention="scrub_mlp_write_to_partner_destroyed_source",
            component=component_meta("mlp", layer), packets=packets, membership=membership,
            baseline_target=baseline_target, intervened_target=target_scores(model, target_logits, packets),
            baseline_health=baseline_health,
            intervened_health=health_scores(model, health_logits, sequences, probes),
            counterfactual_target=corrupt_target,
        )

    expected_rows = len(packets) * (len(heads) + len(mlps)) * 2
    gate("complete_component_grid", len(rows) == expected_rows,
         f"rows={len(rows)}, expected={expected_rows}")
    gate("finite_component_grid", all(
        math.isfinite(row[key])
        for row in rows
        for key in ("effect_margin", "effect_logprob", "health_logprob_damage", "health_margin_damage")
    ), "")
    per_packet_estimand = defaultdict(set)
    for row in rows:
        per_packet_estimand[(row["packet_id"], row["estimand"])].add(row["component_id"])
    gate("all_102_components_per_packet_estimand", all(
        len(components) == 102 for components in per_packet_estimand.values()
    ) and len(per_packet_estimand) == len(packets) * 2, "")

    failures = [item for item in gates if not item["passed"]]
    status = "PREFLIGHT_PASS" if not failures else "PREFLIGHT_FAIL"
    rows.sort(key=lambda row: (
        row["estimand"], row["target_role"], row["packet_id"], row["layer"],
        row["component_type"], -1 if row["head"] is None else row["head"],
    ))
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "schema": "head_context_atlas.model_preflight/1",
        "status": status,
        "claim_boundary": (
            "mechanical discovery-only preflight; component effects are plumbing diagnostics, "
            "not head rankings or biological claims"
        ),
        "selection": {
            "method": "one donor and one acceptor matched pair maximizing coverage of all six core tags",
            "pairs": selected_pairs,
            "packet_ids": packet_ids,
            "covered_core_tags": sorted(set(CORE_CATEGORIES) & covered),
        },
        "model": {
            "path": str(Path(args.model)),
            "device": str(device),
            "layers": adapter.num_layers,
            "heads_per_layer": adapter.num_heads,
            "mlps": adapter.num_layers,
            "precision": "float32; TF32/AMP off",
        },
        "masked_health": {
            "n_probes_per_packet": N_HEALTH_PROBES,
            "mean_baseline_accuracy": mean_health_accuracy,
            "mean_baseline_logprob": mean_health_logprob,
            "accuracy_floor": HEALTH_ACCURACY_FLOOR,
            "logprob_floor": HEALTH_LOGPROB_FLOOR,
        },
        "baseline_packets": [
            {
                "packet_id": packet["packet_id"],
                "target_role": packet["target_role"],
                "tags": packet["tags"],
                "health_probes": probe,
                "target": target,
                "health": health,
                "partner_destroyed_target": corrupt,
                "behavioral_partner_effect": target["margin"] - corrupt["margin"],
            }
            for packet, probe, target, health, corrupt in zip(
                packets, probes, baseline_target, baseline_health, corrupt_target
            )
        ],
        "n_rows": len(rows),
        "expected_rows": expected_rows,
        "gates": gates,
        "n_failures": len(failures),
        "failures": failures,
        "seconds": time.time() - start,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    from gfmcd.provenance import ExperimentManifest
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_model_preflight_v1",
        command=["python"] + sys.argv,
        status=status,
    )
    for path in (pair_path, packet_path, sequence_path):
        manifest.add_input(str(path))
    for name in ("config.json", "pytorch_model.bin", "tokenizer.json", "vocab.txt"):
        manifest.add_input(str(Path(args.model) / name))
    manifest.seeds.update({"pythonhashseed": 0, "torch": 0, "numpy": 0})
    manifest.add_output(str(rows_path))
    manifest.add_output(str(summary_path))
    manifest.write(str(manifest_path))
    print(json.dumps({
        "status": status,
        "n_packets": len(packets),
        "n_rows": len(rows),
        "n_failures": len(failures),
        "seconds": summary["seconds"],
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
