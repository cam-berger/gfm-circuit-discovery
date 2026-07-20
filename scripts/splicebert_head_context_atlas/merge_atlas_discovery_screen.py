#!/usr/bin/env python
"""Merge validated per-packet shards into compact analysis matrices."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", required=True)
    parser.add_argument("--cohort-prefix", default=str(
        REPO / "results/splicebert_head_context_atlas/atlas_context_matched_v7"
    ))
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    screen = Path(args.screen)
    cohort_prefix = Path(args.cohort_prefix)
    output_prefix = Path(args.prefix)
    audit_path = screen / "validation_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "PASS" or audit["n_failures"]:
        raise RuntimeError("refuse to merge a screen that did not pass validation")

    target_out = Path(f"{output_prefix}.target_matrix.jsonl.gz")
    health_out = Path(f"{output_prefix}.health_matrix.jsonl.gz")
    pairs_out = Path(f"{output_prefix}.pairs.jsonl.gz")
    summary_out = Path(f"{output_prefix}.summary.json")
    manifest_out = Path(f"{output_prefix}.manifest.json")
    for path in (target_out, health_out, pairs_out, summary_out, manifest_out):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite immutable artifact {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    n_target = 0
    component_ids = set()
    target_packet_ids = set()
    with gzip.open(target_out, "wt", compresslevel=6) as handle:
        for shard in sorted((screen / "target_shards").glob("packet_*.json")):
            record = json.loads(shard.read_text())
            packet = record["packet"]
            target_packet_ids.add(packet["packet_id"])
            for component in record["components"]:
                component_ids.add(component["component_id"])
                row = {
                    **packet,
                    **component,
                    "native_margin": record["native_target"]["margin"],
                    "native_logprob": record["native_target"]["logprob"],
                    "native_entropy": record["native_target"]["entropy"],
                    "partner_destroyed_margin": record["partner_destroyed_target"]["margin"],
                    "behavioral_partner_effect_margin": record["behavioral_partner_effect_margin"],
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                n_target += 1

    n_health = 0
    with gzip.open(health_out, "wt", compresslevel=6) as handle:
        for shard in sorted((screen / "health_shards").glob("packet_*.json")):
            record = json.loads(shard.read_text())
            packet = record["packet"]
            for component in record["components"]:
                row = {
                    **packet,
                    **component,
                    "probe_positions": record["probe_positions"],
                    "native_health_logprob": record["native_health"]["mean_logprob"],
                    "native_health_margin": record["native_health"]["mean_margin"],
                    "native_health_accuracy": record["native_health"]["accuracy"],
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                n_health += 1

    pair_path = Path(f"{cohort_prefix}.discovery.pairs.jsonl")
    n_pairs = 0
    with pair_path.open() as source, gzip.open(pairs_out, "wt", compresslevel=6) as target:
        for line in source:
            if line.strip():
                pair = json.loads(line)
                if (pair["case_packet_id"] in target_packet_ids and
                        pair["control_packet_id"] in target_packet_ids):
                    target.write(json.dumps(pair, sort_keys=True) + "\n")
                    n_pairs += 1

    expected_target = audit["n_target_component_rows"]
    expected_health = audit["n_health_component_rows"]
    if n_target != expected_target or n_health != expected_health or len(component_ids) != 102:
        raise RuntimeError(
            f"merge count mismatch: target {n_target}/{expected_target}, "
            f"health {n_health}/{expected_health}, components={len(component_ids)}"
        )
    summary = {
        "schema": "head_context_atlas.discovery_screen_merge/1",
        "status": "COMPLETE",
        "claim_boundary": "mechanically merged discovery matrix; not a component-function claim",
        "n_target_rows": n_target,
        "n_health_rows": n_health,
        "n_pairs": n_pairs,
        "n_components": len(component_ids),
        "component_ids": sorted(component_ids),
        "columns": {
            "A": "native target margin/logprob loss under zero component output",
            "B": "native target margin/logprob loss when component state is scrubbed to partner-destroyed source",
            "health": "same component interventions on genuinely masked non-site probes",
        },
        "analysis_join": "pairs.case_packet_id/control_packet_id -> target_matrix.packet_id",
    }
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    from gfmcd.provenance import ExperimentManifest
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_discovery_screen_merge_v1",
        command=["python"] + sys.argv,
        status="COMPLETE",
    )
    manifest.add_input(str(audit_path))
    manifest.add_input(str(pair_path))
    for phase in ("health", "target"):
        phase_manifest = screen / f"run_manifest.{phase}.json"
        if phase_manifest.exists():
            manifest.add_input(str(phase_manifest))
    if (screen / "run_manifest.both.json").exists():
        manifest.add_input(str(screen / "run_manifest.both.json"))
    for path in (target_out, health_out, pairs_out, summary_out):
        manifest.add_output(str(path))
    manifest.write(str(manifest_out))
    print(json.dumps({
        "status": "COMPLETE", "target_rows": n_target,
        "health_rows": n_health, "pairs": n_pairs,
    }, indent=2))


if __name__ == "__main__":
    main()
