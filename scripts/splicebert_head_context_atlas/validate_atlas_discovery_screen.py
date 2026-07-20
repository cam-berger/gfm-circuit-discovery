#!/usr/bin/env python
"""Validate all discovery-screen shards before merge or analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_atlas_discovery_screen import EXCLUDED_CHROMS, file_sha256, shard_name  # noqa: E402
from run_atlas_model_preflight import read_jsonl  # noqa: E402

EXPECTED_COMPONENTS = {
    *(f"L{layer}H{head}" for layer in range(6) for head in range(16)),
    *(f"L{layer}MLP" for layer in range(6)),
}


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def finite_fields(row: dict, fields: tuple[str, ...]) -> bool:
    return all(field in row and math.isfinite(float(row[field])) for field in fields)


def validate_manifest_hashes(manifest: dict, out_dir: Path, phase: str,
                             failures: list[str]) -> None:
    records = manifest.get("shards", {}).get(phase, {})
    shard_dir = out_dir / f"{phase}_shards"
    actual = {path.name for path in shard_dir.glob("packet_*.json")}
    require(actual == set(records), f"{phase}: manifest/actual shard set mismatch", failures)
    for name, record in records.items():
        path = shard_dir / name
        require(path.exists(), f"{phase}: missing shard {name}", failures)
        if not path.exists():
            continue
        require(path.stat().st_size == record["size_bytes"],
                f"{phase}: size mismatch {name}", failures)
        require(file_sha256(path) == record["sha256"],
                f"{phase}: hash mismatch {name}", failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="screen output directory")
    parser.add_argument("--cohort-prefix", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_matched_v7"
    ))
    parser.add_argument("--manifest-mode", choices=("both", "separate"), default="separate")
    parser.add_argument("--packet-ids-from-summary", default=None, help="smoke subset")
    parser.add_argument("--audit", default=None)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    out_dir = Path(args.out)
    cohort_prefix = Path(args.cohort_prefix)
    pair_path = Path(f"{cohort_prefix}.discovery.pairs.jsonl")
    packet_path = Path(f"{cohort_prefix}.discovery.packets.jsonl")
    sequence_path = Path(f"{cohort_prefix}.discovery.sequences.jsonl")
    packets = {row["packet_id"]: row for row in read_jsonl(packet_path)}
    sequences = {row["sequence_id"]: row for row in read_jsonl(sequence_path)}
    pairs = read_jsonl(pair_path)
    if args.packet_ids_from_summary:
        selected = set(json.loads(Path(args.packet_ids_from_summary).read_text())["selection"]["packet_ids"])
        packets = {packet_id: packets[packet_id] for packet_id in selected}

    if args.manifest_mode == "both":
        combined = json.loads((out_dir / "run_manifest.both.json").read_text())
        manifests = {"health": combined, "target": combined}
    else:
        manifests = {
            phase: json.loads((out_dir / f"run_manifest.{phase}.json").read_text())
            for phase in ("health", "target")
        }
    failures: list[str] = []
    for phase, manifest in manifests.items():
        require(manifest["status"] == "COMPLETE", f"{phase}: manifest not COMPLETE", failures)
        require(manifest["n_failures"] == 0, f"{phase}: manifest reports failures", failures)
        require(manifest["cohort"]["validation_loaded"] is False,
                f"{phase}: validation-loaded flag not false", failures)
        validate_manifest_hashes(manifest, out_dir, phase, failures)

    target_dir = out_dir / "target_shards"
    target_files = sorted(target_dir.glob("packet_*.json"))
    require(len(target_files) == len(packets),
            f"target: expected {len(packets)} shards, found {len(target_files)}", failures)
    target_packet_ids = set()
    for path in target_files:
        record = json.loads(path.read_text())
        require(record.get("schema") == "head_context_atlas.discovery_target_shard/1",
                f"target: bad schema {path.name}", failures)
        meta = record["packet"]
        packet_id = meta["packet_id"]
        target_packet_ids.add(packet_id)
        require(path.name == shard_name(packet_id), f"target: filename mismatch {path.name}", failures)
        require(packet_id in packets, f"target: unknown packet {packet_id}", failures)
        if packet_id not in packets:
            continue
        packet = packets[packet_id]
        require(meta["analysis_split"] == "atlas_discovery",
                f"target: split mismatch {packet_id}", failures)
        require(meta["chrom"] not in EXCLUDED_CHROMS,
                f"target: excluded chromosome {packet_id}", failures)
        require(meta["masked_base_win"] == packet["masked_base_win"] and
                meta["masked_native_base"] == "G",
                f"target: coordinate mismatch {packet_id}", failures)
        ids = [component["component_id"] for component in record["components"]]
        require(len(ids) == 102 and set(ids) == EXPECTED_COMPONENTS and len(ids) == len(set(ids)),
                f"target: component grid mismatch {packet_id}", failures)
        require(all(finite_fields(component, (
            "A_effect_margin", "A_effect_logprob", "B_effect_margin", "B_effect_logprob"
        )) for component in record["components"]),
                f"target: non-finite component effect {packet_id}", failures)
        require(finite_fields(record["native_target"], ("margin", "logprob", "entropy")),
                f"target: non-finite native target {packet_id}", failures)
        require(finite_fields(record["partner_destroyed_target"], ("margin", "logprob", "entropy")),
                f"target: non-finite counterfactual target {packet_id}", failures)
    require(target_packet_ids == set(packets), "target: packet set mismatch", failures)

    health_dir = out_dir / "health_shards"
    health_files = sorted(health_dir.glob("packet_*.json"))
    expected_health = manifests["health"]["cohort"]["n_health_packets"]
    require(len(health_files) == expected_health,
            f"health: expected {expected_health} shards, found {len(health_files)}", failures)
    for path in health_files:
        record = json.loads(path.read_text())
        require(record.get("schema") == "head_context_atlas.discovery_health_shard/1",
                f"health: bad schema {path.name}", failures)
        meta = record["packet"]
        packet_id = meta["packet_id"]
        require(path.name == shard_name(packet_id), f"health: filename mismatch {path.name}", failures)
        require(packet_id in packets, f"health: packet outside target cohort {packet_id}", failures)
        if packet_id not in packets:
            continue
        packet = packets[packet_id]
        sequence = sequences[packet["sequence_id"]]["sequence"]
        probes = record["probe_positions"]
        require(len(probes) == 4 and len(set(probes)) == 4,
                f"health: probe count mismatch {packet_id}", failures)
        protected = (packet["donor_win"], packet["donor_win"] + 1,
                     packet["acceptor_win"] - 2, packet["acceptor_win"] - 1)
        require(all(32 <= position < len(sequence) - 32 and
                    min(abs(position - site) for site in protected) > 12 for position in probes),
                f"health: invalid probe coordinate {packet_id}", failures)
        ids = [component["component_id"] for component in record["components"]]
        require(len(ids) == 102 and set(ids) == EXPECTED_COMPONENTS and len(ids) == len(set(ids)),
                f"health: component grid mismatch {packet_id}", failures)
        require(all(finite_fields(component, (
            "A_health_logprob_damage", "A_health_margin_damage",
            "B_health_logprob_damage", "B_health_margin_damage",
        )) for component in record["components"]),
                f"health: non-finite component effect {packet_id}", failures)
        require(finite_fields(record["native_health"], ("mean_logprob", "mean_margin", "accuracy")),
                f"health: non-finite native health {packet_id}", failures)

    health_manifest = manifests["health"]
    require(health_manifest.get("health") and health_manifest["health"]["gate"] is True,
            "health: aggregate masked-health gate not true", failures)

    # Pair files are not consumed by inference, but every target packet must remain linked
    # to at least one matched contrast for the downstream estimand.
    linked = {
        packet_id for pair in pairs
        for packet_id in (pair["case_packet_id"], pair["control_packet_id"])
    }
    require(set(packets) <= linked, "target packet lacks matched-pair link", failures)

    audit_path = Path(args.audit) if args.audit else out_dir / "validation_audit.json"
    report = {
        "schema": "head_context_atlas.discovery_screen_audit/1",
        "status": "PASS" if not failures else "FAIL",
        "claim_boundary": "mechanical shard audit only; no component ranking or context claim",
        "n_target_packets": len(target_files),
        "n_health_packets": len(health_files),
        "n_target_component_rows": len(target_files) * 102,
        "n_health_component_rows": len(health_files) * 102,
        "manifest_mode": args.manifest_mode,
        "n_failures": len(failures),
        "failures": failures,
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
