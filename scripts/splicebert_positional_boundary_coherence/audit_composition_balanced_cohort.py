#!/usr/bin/env python
"""Independently reconstruct and audit a Phase-I cohort package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.positional_boundary_coherence import (  # noqa: E402
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


def audit_artifact(
    *,
    cohort_path: str | Path,
    candidates_path: str | Path,
    candidate_audit_path: str | Path | None = None,
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
    blacklist = load_blacklist(_read_json(blacklist_file))
    variance = _read_json(variance_file)
    audit = audit_composition_balanced_cohort(
        cohort,
        candidates,
        candidate_audit=candidate_audit,
        blacklisted_gene_ids=blacklist,
        frozen_variance=variance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert-positional-boundary-coherence:composition-balanced-cohort-audit-v1",
        sys.argv,
        repo=REPO,
        status=str(audit["status"]),
        config={"stage": "construction"},
        notes=[
            "Independent reconstruction from candidates, blacklist, and frozen variance.",
            "No SpliceBERT checkpoint or model outcome was loaded.",
        ],
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
    manifest.add_output(output)
    manifest.write(manifest_path)
    return output, manifest_path, audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-audit", required=True)
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
