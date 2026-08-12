#!/usr/bin/env python
"""Independently reconstruct and audit a Phase-I cohort package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.positional_boundary_coherence import (  # noqa: E402
    CAPACITY_STOP_SCHEMA,
    CANDIDATE_CENSUS_RECEIPT_SCHEMA,
    CANONICAL_CANDIDATE_PRODUCER_REVISION,
    audit_composition_balanced_cohort,
    load_blacklist,
)


def _read_json(path: str | Path) -> Any:
    with Path(path).expanduser().resolve().open() as handle:
        return json.load(handle)


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    if output.suffix != ".json":
        raise ValueError("audit output must be a .json file")
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    if output == manifest:
        raise ValueError("audit and manifest paths must be distinct")
    return output, manifest


def _candidate_census_receipt(
    candidate_file: Path,
    producer_manifest_file: Path,
    candidate_audit: dict[str, Any],
) -> dict[str, Any]:
    """Auditor-local verification of local census bytes and regeneration receipt."""

    candidate_record = file_record(candidate_file)
    producer_manifest = _read_json(producer_manifest_file)
    expected_config = {
        "acceptor_pos": 512,
        "algorithm_version": "pbc-real-census-v1",
        "allowed_chromosomes": sorted(
            [
                *(f"chr{i}" for i in range(1, 8)),
                *(f"chr{i}" for i in range(10, 23)),
                "chrX",
            ]
        ),
        "construct_length": 1022,
        "donor_pos": 392,
        "stage": "construction",
    }
    semantic_hashes = {
        "candidate_census_sha256": candidate_audit.get("candidate_census_sha256"),
        "candidate_content_sha256": candidate_audit.get("candidate_content_sha256"),
        "reconstructed_candidates_sha256": candidate_audit.get(
            "reconstructed_candidates_sha256"
        ),
    }

    def lower_hex(value: Any, length: int) -> bool:
        return (
            isinstance(value, str)
            and len(value) == length
            and all(character in "0123456789abcdef" for character in value)
        )

    if not all(lower_hex(value, 64) for value in semantic_hashes.values()):
        raise ValueError("candidate audit is missing valid semantic census hashes")
    commit = producer_manifest.get("git_commit")
    manifest_identity_ok = (
        producer_manifest.get("experiment_id")
        == "splicebert-positional-boundary-coherence:candidate-census-v1"
        and producer_manifest.get("status") == "CENSUS_BUILT_AUDIT_PENDING"
        and producer_manifest.get("config") == expected_config
        and commit == CANONICAL_CANDIDATE_PRODUCER_REVISION
        and lower_hex(commit, 40)
        and producer_manifest.get("git_dirty") is True
    )
    commit_exists = bool(
        lower_hex(commit, 40)
        and subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    if not manifest_identity_ok or not commit_exists:
        raise ValueError("candidate census producer manifest identity is not canonical")
    matching_outputs = [
        row
        for row in producer_manifest.get("outputs", [])
        if isinstance(row, dict)
        and Path(str(row.get("path", ""))).name == candidate_file.name
        and row.get("size_bytes") == candidate_record.get("size_bytes")
        and row.get("sha256") == candidate_record.get("sha256")
    ]
    if len(producer_manifest.get("outputs", [])) != 1 or len(matching_outputs) != 1:
        raise ValueError("candidate census bytes do not match the producer manifest")
    manifest_record = file_record(producer_manifest_file)
    return {
        "schema": CANDIDATE_CENSUS_RECEIPT_SCHEMA,
        "candidate_file": {
            "name": candidate_file.name,
            "size_bytes": candidate_record["size_bytes"],
            "sha256": candidate_record["sha256"],
        },
        "producer_manifest": {
            "name": producer_manifest_file.name,
            "size_bytes": manifest_record["size_bytes"],
            "sha256": manifest_record["sha256"],
            "experiment_id": producer_manifest["experiment_id"],
            "status": producer_manifest["status"],
            "config": producer_manifest["config"],
            "git_commit": commit,
            "git_dirty": producer_manifest["git_dirty"],
            "output_record_verified": True,
        },
        "semantic_hashes": semantic_hashes,
        "clean_checkout_reexecutable": False,
        "regeneration_required": True,
        "regeneration_receipt": "candidate_census_v1.manifest.json",
    }


def audit_artifact(
    *,
    cohort_path: str | Path,
    candidates_path: str | Path,
    candidate_audit_path: str | Path | None = None,
    candidate_manifest_path: str | Path | None = None,
    blacklist_path: str | Path,
    frozen_variance_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    output, manifest_path = output_paths(output_path)
    cohort_file = Path(cohort_path).expanduser().resolve()
    candidates_file = Path(candidates_path).expanduser().resolve()
    candidate_audit_file = (
        None
        if candidate_audit_path is None
        else Path(candidate_audit_path).expanduser().resolve(strict=True)
    )
    blacklist_file = Path(blacklist_path).expanduser().resolve()
    variance_file = Path(frozen_variance_path).expanduser().resolve()
    cohort = _read_json(cohort_file)
    candidates = _read_json(candidates_file)
    candidate_audit = (
        None if candidate_audit_file is None else _read_json(candidate_audit_file)
    )
    candidate_manifest_file: Path | None = None
    candidate_receipt: dict[str, Any] | None = None
    if candidates.get("candidate_mode") == "production":
        if candidate_audit is None:
            raise ValueError("production audit requires the candidate census audit")
        candidate_manifest_file = Path(
            candidate_manifest_path
            if candidate_manifest_path is not None
            else candidates_file.with_name(candidates_file.stem + ".manifest.json")
        ).expanduser().resolve(strict=True)
        candidate_receipt = _candidate_census_receipt(
            candidates_file, candidate_manifest_file, candidate_audit
        )
    blacklist = load_blacklist(_read_json(blacklist_file))
    variance = _read_json(variance_file)
    audit = audit_composition_balanced_cohort(
        cohort,
        candidates,
        candidate_audit=candidate_audit,
        candidate_census_receipt=candidate_receipt,
        blacklisted_gene_ids=blacklist,
        frozen_variance=variance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    capacity_stopped = cohort.get("schema") == CAPACITY_STOP_SCHEMA
    manifest = ExperimentManifest.create(
        "splicebert-positional-boundary-coherence:composition-balanced-cohort-audit-v1",
        sys.argv,
        repo=REPO,
        status=str(audit["status"]),
        config={"stage": "construction"},
        notes=(
            [
                "Independent reconstruction of the necessary unique-gene capacity stop.",
                "No SpliceBERT checkpoint or model outcome was loaded.",
                "A passing audit validates only the stop and never authorizes a target.",
            ]
            if capacity_stopped
            else [
                "Independent reconstruction from candidates, blacklist, and frozen variance.",
                "No SpliceBERT checkpoint or model outcome was loaded.",
            ]
        ),
    )
    for path in (
        cohort_file,
        candidates_file,
        blacklist_file,
        variance_file,
        REPO / "studies/splicebert_positional_boundary_coherence/PREREGISTRATION.md",
        REPO / "src/gfmcd/splicing/positional_boundary_coherence.py",
        Path(__file__).resolve(),
    ):
        manifest.add_input(path)
    if candidate_audit_file is not None:
        manifest.add_input(candidate_audit_file)
    if candidate_manifest_file is not None:
        manifest.add_input(candidate_manifest_file)
    manifest.add_output(output)
    manifest.write(manifest_path)
    return output, manifest_path, audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-audit", required=True)
    parser.add_argument(
        "--candidate-manifest",
        help="candidate producer manifest; defaults to <candidates>.manifest.json",
    )
    parser.add_argument("--blacklist", required=True)
    parser.add_argument("--frozen-variance", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output, manifest, audit = audit_artifact(
        cohort_path=args.cohort,
        candidates_path=args.candidates,
        candidate_audit_path=args.candidate_audit,
        candidate_manifest_path=args.candidate_manifest,
        blacklist_path=args.blacklist,
        frozen_variance_path=args.frozen_variance,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "target_execution_authorized": audit["target_execution_authorized"],
                "output": str(output),
                "manifest": str(manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not audit["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
