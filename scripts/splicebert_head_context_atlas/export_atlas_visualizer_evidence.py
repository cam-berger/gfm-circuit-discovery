#!/usr/bin/env python3
"""Export an independently audited Atlas DAG bundle for gfm-visualizer.

The exporter is deliberately lossless at the representative-record boundary.  It
does not reclassify edges, recompute intervals, or discard remainder records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_SCHEMA = "head_context_atlas.representative_dag_bundle/1"
SOURCE_AUDIT_SCHEMA = "head_context_atlas.representative_dag_audit/1"
SOURCE_MANIFEST_SCHEMA = "head_context_atlas.representative_dag_manifest/1"
OUTPUT_SCHEMA = "gfm_visualizer.atlas_evidence/1"
MANIFEST_SCHEMA = "gfm_visualizer.atlas_evidence_manifest/1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def require_source_boundary(
    bundle: dict[str, Any],
    audit: dict[str, Any],
    manifest: dict[str, Any],
    bundle_sha: str,
) -> None:
    if (
        bundle.get("schema") != SOURCE_SCHEMA
        or bundle.get("status") != "VALIDATED_EXPORT_CANDIDATE"
        or bundle.get("export_eligible") is not True
        or bundle.get("n_representatives") != len(bundle.get("representatives", []))
    ):
        raise RuntimeError("source representative bundle is not export-eligible")
    if (
        audit.get("schema") != SOURCE_AUDIT_SCHEMA
        or audit.get("status") != "PASS"
        or audit.get("n_failures") != 0
        or audit.get("failures") != []
        or audit.get("smoke") is not False
        or audit.get("export_eligible") is not True
        or audit.get("bundle_sha256") != bundle_sha
        or audit.get("n_representatives") != len(bundle.get("representatives", []))
    ):
        raise RuntimeError("representative DAG audit is failed, stale, smoke, or non-exportable")
    if (
        manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("status") != "VALIDATED_EXPORT_CANDIDATE"
        or manifest.get("export_eligible") is not True
        or manifest.get("output_sha256") != bundle_sha
    ):
        raise RuntimeError("source representative manifest is failed or stale")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-bundle-id")
    parser.add_argument("--source-audit-id")
    parser.add_argument("--source-manifest-id")
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    audit_path = Path(args.audit)
    source_manifest_path = Path(args.source_manifest)
    output = Path(args.output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refuse to overwrite immutable visualizer evidence")

    bundle = load_json(bundle_path)
    audit = load_json(audit_path)
    source_manifest = load_json(source_manifest_path)
    bundle_sha = sha256_file(bundle_path)
    audit_sha = sha256_file(audit_path)
    source_manifest_sha = sha256_file(source_manifest_path)
    require_source_boundary(bundle, audit, source_manifest, bundle_sha)

    source_ids = {
        key: value
        for key, value in {
            "source_bundle_drive_id": args.source_bundle_id,
            "source_audit_drive_id": args.source_audit_id,
            "source_manifest_drive_id": args.source_manifest_id,
        }.items()
        if value
    }
    evidence = {
        "schema": OUTPUT_SCHEMA,
        "status": "AUDITED_VALIDATED",
        "export_eligible": True,
        "title": "Validated SpliceBERT Head–Context Atlas",
        "description": (
            "Measured representative screening DAGs for the validated general-use components, "
            "with separately audited population intervention overlays and matched negative contexts."
        ),
        "model": bundle["model"],
        "n_representatives": bundle["n_representatives"],
        "evidence_labels": bundle["evidence_labels"],
        "claim_boundary": (
            "Whole-component and conditional population evidence is causal at the named component/path "
            "level; nucleotide DAG edges remain activation×Jacobian screening evidence. No biological "
            "function label or feature-complete circuit is claimed."
        ),
        # Preserve the source records byte-for-byte after JSON decoding. In particular,
        # do not upgrade screening edges or renormalize away hidden remainder mass.
        "representatives": bundle["representatives"],
        "audit": {
            "status": "PASS",
            "n_failures": 0,
            "export_eligible": True,
            "bundle_sha256": bundle_sha,
            "source_audit_sha256": audit_sha,
            "source_manifest_sha256": source_manifest_sha,
        },
        "provenance": {
            **bundle["provenance"],
            **source_ids,
            "source_bundle_name": bundle_path.name,
            "source_bundle_sha256": bundle_sha,
            "source_audit_name": audit_path.name,
            "source_audit_sha256": audit_sha,
            "source_manifest_name": source_manifest_path.name,
            "source_manifest_sha256": source_manifest_sha,
            "visualizer_exporter_sha256": sha256_file(Path(__file__)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "AUDITED_VALIDATED",
        "export_eligible": True,
        "output": output.name,
        "output_sha256": sha256_file(output),
        "output_bytes": output.stat().st_size,
        "source_bundle_sha256": bundle_sha,
        "source_audit_sha256": audit_sha,
        "source_manifest_sha256": source_manifest_sha,
        "source_ids": source_ids,
        "evidence_labels": bundle["evidence_labels"],
        "n_representatives": len(evidence["representatives"]),
        "exporter_sha256": sha256_file(Path(__file__)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": evidence["status"],
        "n_representatives": len(evidence["representatives"]),
        "output_sha256": manifest["output_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
