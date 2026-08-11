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
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.positional_boundary_coherence import (  # noqa: E402
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


def build_artifact(
    *,
    candidates_path: str | Path,
    candidate_audit_path: str | Path | None = None,
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
    blacklist_payload = _read_json(blacklist_file)
    frozen_variance = _read_json(variance_file)
    package = build_composition_balanced_cohort(
        candidate_package,
        candidate_audit=candidate_audit,
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
    manifest = ExperimentManifest.create(
        "splicebert-positional-boundary-coherence:composition-balanced-cohort-v1",
        sys.argv,
        repo=REPO,
        status=str(package["status"]),
        config={"stage": "construction", **dict(package["config"])},
        seeds={"selection_and_audit": int(seed)},
        notes=[
            "Model-free construction; No SpliceBERT checkpoint or model output was loaded.",
            "Target execution remains unauthorized until the independent audit passes.",
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
    manifest.add_output(output)
    manifest.write(manifest_path)
    return output, manifest_path, package


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-audit")
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

    if args.development_blocks < 120:
        raise ValueError("production development role requires at least 120 blocks")
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


if __name__ == "__main__":
    main()
