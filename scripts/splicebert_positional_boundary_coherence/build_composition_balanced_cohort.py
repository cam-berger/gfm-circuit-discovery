#!/usr/bin/env python
"""Build the sealed, model-free Phase-I positional-boundary cohort.

Input candidates are already annotation-audited source modules.  This runner
loads no checkpoint and accepts no model outcomes.  It jointly builds the
development, confirmation, and Phase-III-reserve roles so no normalized gene
can leak across role boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.positional_boundary_coherence import (  # noqa: E402
    CAPACITY_STOP_SCHEMA,
    CANDIDATE_CENSUS_RECEIPT_SCHEMA,
    CANONICAL_CANDIDATE_PRODUCER_REVISION,
    DEFAULT_BLOCK_COUNTS,
    build_composition_balanced_cohort,
    load_blacklist,
)


def _read_json(path: str | Path) -> Any:
    with Path(path).expanduser().resolve().open() as handle:
        return json.load(handle)


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    if output.suffix != ".json":
        raise ValueError("cohort output must be a .json file")
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    if output == manifest:
        raise ValueError("result and manifest paths must be distinct")
    return output, manifest


def _candidate_census_receipt(
    candidate_file: Path,
    producer_manifest_file: Path,
    candidate_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the local census bytes against its regeneration manifest."""

    candidate_record = file_record(candidate_file)
    producer_manifest = _read_json(producer_manifest_file)
    expected_config = {
        "acceptor_pos": 512,
        "algorithm_version": "pbc-real-census-v1",
        "allowed_chromosomes": [
            *(f"chr{i}" for i in range(1, 8)),
            *(f"chr{i}" for i in range(10, 23)),
            "chrX",
        ],
        "construct_length": 1022,
        "donor_pos": 392,
        "stage": "construction",
    }
    expected_config["allowed_chromosomes"] = sorted(expected_config["allowed_chromosomes"])
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
        if isinstance(row, Mapping)
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


def build_artifact(
    *,
    candidates_path: str | Path,
    candidate_audit_path: str | Path | None = None,
    candidate_manifest_path: str | Path | None = None,
    blacklist_path: str | Path,
    frozen_variance_path: str | Path,
    output_path: str | Path,
    block_counts: Mapping[str, int] = DEFAULT_BLOCK_COUNTS,
    minimum_confirmation_blocks: int = 600,
    seed: int = 73031,
    auc_n_boot: int = 2000,
    inner_folds: int = 4,
    pseudo_family_size: int = 99,
    minimum_stratum_blocks: int = 5,
) -> tuple[Path, Path, dict[str, Any]]:
    output, manifest_path = output_paths(output_path)
    candidates_file = Path(candidates_path).expanduser().resolve()
    blacklist_file = Path(blacklist_path).expanduser().resolve()
    variance_file = Path(frozen_variance_path).expanduser().resolve()
    candidate_package = _read_json(candidates_file)
    candidate_audit_file = (
        None
        if candidate_audit_path is None
        else Path(candidate_audit_path).expanduser().resolve(strict=True)
    )
    candidate_audit = (
        None if candidate_audit_file is None else _read_json(candidate_audit_file)
    )
    candidate_manifest_file: Path | None = None
    candidate_receipt: dict[str, Any] | None = None
    if candidate_package.get("candidate_mode") == "production":
        if candidate_audit is None:
            raise ValueError("production construction requires the candidate census audit")
        candidate_manifest_file = Path(
            candidate_manifest_path
            if candidate_manifest_path is not None
            else candidates_file.with_name(candidates_file.stem + ".manifest.json")
        ).expanduser().resolve(strict=True)
        candidate_receipt = _candidate_census_receipt(
            candidates_file, candidate_manifest_file, candidate_audit
        )
    blacklist_payload = _read_json(blacklist_file)
    frozen_variance = _read_json(variance_file)
    package = build_composition_balanced_cohort(
        candidate_package,
        candidate_audit=candidate_audit,
        candidate_census_receipt=candidate_receipt,
        blacklisted_gene_ids=load_blacklist(blacklist_payload),
        frozen_variance=frozen_variance,
        requested_blocks_by_split=dict(block_counts),
        minimum_confirmation_blocks=minimum_confirmation_blocks,
        seed=seed,
        auc_n_boot=auc_n_boot,
        inner_folds=inner_folds,
        pseudo_family_size=pseudo_family_size,
        minimum_stratum_blocks=minimum_stratum_blocks,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    capacity_stopped = package.get("schema") == CAPACITY_STOP_SCHEMA
    manifest = ExperimentManifest.create(
        "splicebert-positional-boundary-coherence:composition-balanced-cohort-v1",
        sys.argv,
        repo=REPO,
        status=str(package["status"]),
        config={"stage": "construction", **dict(package["config"])},
        seeds={"selection_and_audit": int(seed)},
        notes=[
            "Model-free construction; No SpliceBERT checkpoint or model output was loaded.",
            (
                "Frozen unique-gene capacity failed before selection; an independent audit "
                "may validate the stop but cannot authorize target execution."
                if capacity_stopped
                else "Target execution remains unauthorized until the independent audit passes."
            ),
        ],
    )
    for path in (
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
    return output, manifest_path, package


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-audit")
    parser.add_argument(
        "--candidate-manifest",
        help="candidate producer manifest; defaults to <candidates>.manifest.json",
    )
    parser.add_argument("--blacklist", required=True)
    parser.add_argument("--frozen-variance", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--development-blocks", type=int, default=120)
    parser.add_argument("--confirmation-blocks", type=int, default=600)
    parser.add_argument("--reserve-blocks", type=int, default=384)
    parser.add_argument("--minimum-confirmation-blocks", type=int, default=600)
    parser.add_argument("--seed", type=int, default=73031)
    parser.add_argument("--auc-bootstrap", type=int, default=2000)
    parser.add_argument("--pseudo-family-size", type=int, default=99)
    parser.add_argument("--minimum-stratum-blocks", type=int, default=5)
    return parser.parse_args(argv)


def validate_production_args(args: argparse.Namespace) -> None:
    """Refuse a CLI cohort that weakens the frozen confirmatory role sizes."""

    if args.development_blocks != 120:
        raise ValueError("production development role is frozen at exactly 120 blocks")
    if args.confirmation_blocks != 600:
        raise ValueError("production confirmation role is frozen at exactly 600 blocks")
    if args.reserve_blocks != 384:
        raise ValueError("production Phase-III reserve is frozen at exactly 384 blocks")
    if args.minimum_confirmation_blocks != 600:
        raise ValueError("production minimum confirmation count is frozen at 600")
    if args.pseudo_family_size != 99 or args.auc_bootstrap != 2_000:
        raise ValueError("production balance calibration requires 99 pseudos and 2,000 draws")
    if not args.candidate_audit:
        raise ValueError("production construction requires --candidate-audit")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_production_args(args)
    output, manifest, package = build_artifact(
        candidates_path=args.candidates,
        candidate_audit_path=args.candidate_audit,
        candidate_manifest_path=args.candidate_manifest,
        blacklist_path=args.blacklist,
        frozen_variance_path=args.frozen_variance,
        output_path=args.out,
        block_counts={
            "development": args.development_blocks,
            "confirmation": args.confirmation_blocks,
            "phase3_reserve": args.reserve_blocks,
        },
        minimum_confirmation_blocks=args.minimum_confirmation_blocks,
        seed=args.seed,
        auc_n_boot=args.auc_bootstrap,
        pseudo_family_size=args.pseudo_family_size,
        minimum_stratum_blocks=args.minimum_stratum_blocks,
    )
    print(
        json.dumps(
            {
                "status": package["status"],
                "target_execution_authorized": package["target_execution_authorized"],
                "counts": package["counts"]["blocks_by_split"],
                "output": str(output),
                "manifest": str(manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if package.get("schema") == CAPACITY_STOP_SCHEMA:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
