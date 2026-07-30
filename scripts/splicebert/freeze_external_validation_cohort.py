#!/usr/bin/env python
"""Download design metadata or freeze Stage-4 external-validation artifacts.

``download`` fetches only small, outcome-free sources. ``design-pool`` validates and freezes
eligible sequences/coordinates but performs no candidate selection. Prefer a custodian-produced
COMPASS design export; use ``combined-container`` only with an explicit acknowledgement that
``csv.reader`` parses complete row lists before the design indices are projected. ``candidate-
cohort`` later freezes carrier/matched-noncarrier pairs from complete model-feature scores without
loading a model or reading reporter outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.external_validation import (  # noqa: E402
    download_design_sources,
    freeze_candidate_cohort,
    freeze_design_pool,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="action", required=True)

    download = commands.add_parser(
        "download",
        help="download only the small, outcome-free design sources",
    )
    download.add_argument("--lock-a", required=True)
    download.add_argument("--out-dir", required=True)
    download.add_argument("--max-bytes-per-file", type=int, default=10_000_000)

    design_pool = commands.add_parser(
        "design-pool",
        help="create an immutable design pool; no carrier/control selection",
    )
    design_pool.add_argument("--lock-a", required=True)
    design_pool.add_argument("--vex-positions", required=True)
    design_pool.add_argument("--vex-sequences", required=True)
    design_pool.add_argument(
        "--compass-input",
        required=True,
        help=(
            "local custodian design CSV[.gz], or the combined GSE307247 container under the "
            "explicit combined-container mode"
        ),
    )
    design_pool.add_argument(
        "--compass-input-kind",
        required=True,
        choices=("design_only", "combined_container"),
    )
    design_pool.add_argument("--compass-custodian-manifest")
    design_pool.add_argument("--compass-metadata", required=True)
    design_pool.add_argument("--compass-fasta", required=True)
    design_pool.add_argument("--out-dir", required=True)

    candidate = commands.add_parser(
        "candidate-cohort",
        help="freeze carrier/matched-noncarrier pairs from complete model-feature scores",
    )
    candidate.add_argument("--lock-a", required=True)
    candidate.add_argument("--design-pool", required=True)
    candidate.add_argument("--feature-scores", required=True)
    candidate.add_argument("--feature-score-manifest", required=True)
    candidate.add_argument("--out-dir", required=True)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    if args.action == "download":
        output = download_design_sources(
            lock_a_path=args.lock_a,
            output_dir=args.out_dir,
            max_bytes_per_file=args.max_bytes_per_file,
            command=command,
            repo=REPO,
        )
    elif args.action == "design-pool":
        output = freeze_design_pool(
            lock_a_path=args.lock_a,
            vex_positions_path=args.vex_positions,
            vex_sequences_path=args.vex_sequences,
            compass_input_path=args.compass_input,
            compass_input_kind=args.compass_input_kind,
            compass_metadata_path=args.compass_metadata,
            compass_fasta_path=args.compass_fasta,
            compass_custodian_manifest_path=args.compass_custodian_manifest,
            output_dir=args.out_dir,
            command=command,
            repo=REPO,
        )
    else:
        output = freeze_candidate_cohort(
            lock_a_path=args.lock_a,
            design_pool_path=args.design_pool,
            feature_scores_path=args.feature_scores,
            feature_score_manifest_path=args.feature_score_manifest,
            output_dir=args.out_dir,
            command=command,
            repo=REPO,
        )
    print(output)


if __name__ == "__main__":
    main()
