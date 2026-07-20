#!/usr/bin/env python
"""Merge independently audited mechanism-validation packet shards."""
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


def resolve_repo_reference(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required")

    run_dir = Path(args.run_dir)
    audit_path = Path(args.audit) if args.audit else run_dir / "validation_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "PASS" or audit["n_failures"]:
        raise RuntimeError("refuse to merge a mechanism run without a zero-failure independent audit")
    prefix = Path(args.prefix)
    outputs = {
        "health": Path(f"{prefix}.health_rows.jsonl.gz"),
        "target": Path(f"{prefix}.target_rows.jsonl.gz"),
        "pairs": Path(f"{prefix}.pairs.jsonl.gz"),
        "summary": Path(f"{prefix}.summary.json"),
        "manifest": Path(f"{prefix}.manifest.json"),
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"refuse to overwrite immutable merge output {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    counts = {}
    packet_counts = {}
    endpoint_ids = set()
    for phase in ("health", "target"):
        manifest = json.loads((run_dir / f"run_manifest.{phase}.json").read_text())
        n_rows = 0
        packet_ids = set()
        with gzip.open(outputs[phase], "wt", compresslevel=6) as handle:
            for name in sorted(manifest["shards"]):
                path = run_dir / f"{phase}_shards" / name
                record = json.loads(path.read_text())
                packet = record["packet"]
                packet_ids.add(packet["packet_id"])
                for result in record["results"]:
                    endpoint_ids.add(result["endpoint_id"])
                    handle.write(json.dumps({
                        **packet,
                        "probe_positions": record["probe_positions"],
                        **result,
                    }, sort_keys=True) + "\n")
                    n_rows += 1
        if n_rows != manifest["n_endpoint_rows"] or len(packet_ids) != manifest["n_packets"]:
            raise RuntimeError(f"{phase} merge count mismatch")
        counts[phase] = n_rows
        packet_counts[phase] = len(packet_ids)

    target_manifest = json.loads((run_dir / "run_manifest.target.json").read_text())
    pair_input = resolve_repo_reference(target_manifest["inputs"]["pairs"]["path"])
    pair_count = 0
    with pair_input.open() as source, gzip.open(outputs["pairs"], "wt", compresslevel=6) as target:
        for line in source:
            if line.strip():
                target.write(json.dumps(json.loads(line), sort_keys=True) + "\n")
                pair_count += 1

    summary = {
        "schema": "head_context_atlas.mechanism_validation_merge/1",
        "status": "SMOKE" if audit["smoke"] else "VALIDATION_MERGED",
        "claim_boundary": "audited whole-component/group rows; inference not yet applied",
        "smoke": audit["smoke"],
        "validation_loaded": audit["validation_loaded"],
        "n_health_rows": counts["health"],
        "n_target_rows": counts["target"],
        "n_health_packets": packet_counts["health"],
        "n_target_packets": packet_counts["target"],
        "n_pairs": pair_count,
        "n_endpoints": len(endpoint_ids),
        "endpoint_ids": sorted(endpoint_ids),
        "validation_audit_sha256": sha256_file(audit_path),
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    from gfmcd.provenance import ExperimentManifest
    manifest = ExperimentManifest.create(
        experiment_id="head_context_atlas_mechanism_validation_merge_v1",
        command=["python"] + sys.argv,
        status=summary["status"],
    )
    manifest.config.update({
        "smoke": audit["smoke"], "validation_loaded": audit["validation_loaded"]
    })
    manifest.add_input(str(audit_path))
    for phase in ("health", "target"):
        manifest.add_input(str(run_dir / f"run_manifest.{phase}.json"))
    manifest.add_input(str(pair_input))
    for kind in ("health", "target", "pairs", "summary"):
        manifest.add_output(str(outputs[kind]))
    manifest.write(str(outputs["manifest"]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
