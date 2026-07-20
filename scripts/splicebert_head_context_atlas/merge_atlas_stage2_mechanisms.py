#!/usr/bin/env python
"""Merge independently audited stage-two shards and their frozen whole-group rows."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--package", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    run_dir = Path(args.run_dir)
    audit_path = Path(args.audit) if args.audit else run_dir / "validation_audit.json"
    package_path = Path(args.package)
    prefix = Path(args.prefix)
    audit = json.loads(audit_path.read_text())
    package = json.loads(package_path.read_text())
    if audit["status"] != "PASS" or audit["n_failures"]:
        raise RuntimeError("refuse to merge without a zero-failure independent stage-two audit")
    if audit["package_sha256"] != sha256_file(package_path):
        raise RuntimeError("stage-two package changed after execution audit")
    outputs = {
        "health": Path(f"{prefix}.health_rows.jsonl.gz"),
        "target": Path(f"{prefix}.target_rows.jsonl.gz"),
        "whole_group_health": Path(f"{prefix}.whole_group_health_rows.jsonl.gz"),
        "whole_group_target": Path(f"{prefix}.whole_group_target_rows.jsonl.gz"),
        "pairs": Path(f"{prefix}.pairs.jsonl.gz"),
        "summary": Path(f"{prefix}.summary.json"),
        "manifest": Path(f"{prefix}.manifest.json"),
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"refuse to overwrite immutable merge output {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    target_manifest = json.loads((run_dir / "run_manifest.target.json").read_text())
    for phase in ("health", "target"):
        manifest_path = run_dir / f"run_manifest.{phase}.json"
        if sha256_file(manifest_path) != audit["phase_reports"][phase]["run_manifest_sha256"]:
            raise RuntimeError(f"{phase} run manifest changed after the independent audit")
    candidates = target_manifest["candidates"]
    exact_group_ids = {row["exact_group_id"] for row in candidates}
    counts = {}
    packet_counts = {}
    endpoint_keys = set()
    for phase in ("health", "target"):
        manifest = json.loads((run_dir / f"run_manifest.{phase}.json").read_text())
        packet_ids = set()
        n_rows = 0
        with gzip.open(outputs[phase], "wt", compresslevel=6) as handle:
            for name in sorted(manifest["shards"]):
                record = json.loads((run_dir / f"{phase}_shards" / name).read_text())
                packet = record["packet"]
                packet_ids.add(packet["packet_id"])
                for endpoint in record["endpoints"]:
                    endpoint_keys.add(endpoint["endpoint_key"])
                    for member in endpoint["members"]:
                        handle.write(json.dumps({
                            **packet,
                            "probe_positions": record["probe_positions"],
                            **{key: endpoint[key] for key in (
                                "endpoint_key", "candidate_id", "target_role", "contrast",
                                "scope_kind", "scope_value",
                            )},
                            **member,
                        }, sort_keys=True) + "\n")
                        n_rows += 1
        if n_rows != manifest["n_member_rows"] or len(packet_ids) != manifest["n_packets"]:
            raise RuntimeError(f"{phase} stage-two merge count mismatch")
        counts[phase] = n_rows
        packet_counts[phase] = len(packet_ids)

    group_counts = {}
    for phase in ("health", "target"):
        record = package["whole_validation_row_references"][phase]
        source_path = resolve(record["path"])
        if sha256_file(source_path) != record["sha256"]:
            raise RuntimeError(f"whole-group {phase} source hash mismatch")
        output_key = f"whole_group_{phase}"
        n_rows = 0
        with gzip.open(source_path, "rt") as source, gzip.open(
            outputs[output_key], "wt", compresslevel=6
        ) as target:
            for line in source:
                row = json.loads(line)
                if row["endpoint_role"] == "exact_group" and row["group_id"] in exact_group_ids:
                    target.write(json.dumps(row, sort_keys=True) + "\n")
                    n_rows += 1
        if exact_group_ids and n_rows == 0:
            raise RuntimeError(f"no {phase} exact-group rows joined the candidate groups")
        group_counts[phase] = n_rows

    pair_record = package["whole_validation_row_references"]["pairs"]
    pair_source = resolve(pair_record["path"])
    if sha256_file(pair_source) != pair_record["sha256"]:
        raise RuntimeError("pair source hash mismatch")
    n_pairs = 0
    with gzip.open(pair_source, "rt") as source, gzip.open(
        outputs["pairs"], "wt", compresslevel=6
    ) as target:
        for line in source:
            if line.strip():
                target.write(json.dumps(json.loads(line), sort_keys=True) + "\n")
                n_pairs += 1

    summary = {
        "schema": "head_context_atlas.stage2_mechanism_merge/1",
        "status": "SMOKE" if audit["smoke"] else "STAGE2_MERGED",
        "claim_boundary": "audited conditional rows and frozen exact-group rows; inference pending",
        "smoke": audit["smoke"], "whole_validation_loaded": audit["whole_validation_loaded"],
        "stage2_outcomes_loaded": True, "n_candidates": len(candidates),
        "n_endpoints": len(endpoint_keys), "n_health_rows": counts["health"],
        "n_target_rows": counts["target"], "n_health_packets": packet_counts["health"],
        "n_target_packets": packet_counts["target"],
        "n_whole_group_health_rows": group_counts["health"],
        "n_whole_group_target_rows": group_counts["target"],
        "n_pairs": n_pairs, "candidate_ids": [row["candidate_id"] for row in candidates],
        "exact_group_ids": sorted(exact_group_ids),
        "execution_audit_sha256": sha256_file(audit_path),
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    from gfmcd.provenance import ExperimentManifest
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_stage2_mechanism_merge_v1",
        command=["python"] + sys.argv, status=summary["status"],
    )
    manifest.config.update({
        "smoke": audit["smoke"], "whole_validation_loaded": audit["whole_validation_loaded"],
        "stage2_outcomes_loaded": True,
    })
    manifest.add_input(str(audit_path))
    manifest.add_input(str(package_path))
    for phase in ("health", "target"):
        manifest.add_input(str(run_dir / f"run_manifest.{phase}.json"))
        manifest.add_input(str(resolve(package["whole_validation_row_references"][phase]["path"])))
    manifest.add_input(str(pair_source))
    for key in ("health", "target", "whole_group_health", "whole_group_target", "pairs", "summary"):
        manifest.add_output(str(outputs[key]))
    manifest.write(str(outputs["manifest"]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
