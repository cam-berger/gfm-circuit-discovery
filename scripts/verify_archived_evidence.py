#!/usr/bin/env python3
"""Verify compact evidence attestations against their archived and active bytes.

Run from any directory inside or outside the checkout.  By default the repository is
discovered from each attestation's location; ``--repo`` is useful for explicit CI jobs.

    uv run python scripts/verify_archived_evidence.py \
        results/<study>/attestations/<name>.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Keep the standalone review command usable even when the project is not installed and the
# caller's current directory is a scratch checkout used to exercise an attestation.
_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from gfmcd.evidence_attestation import (
    EvidenceAttestationError,
    verify_evidence_attestation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "attestation",
        type=Path,
        nargs="+",
        help="compact attestation JSON file(s) to verify",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        help="repository path; otherwise discovered from each attestation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    failed = False
    for path in args.attestation:
        try:
            document = verify_evidence_attestation(path, repo=args.repo)
        except EvidenceAttestationError as exc:
            failed = True
            print(f"FAIL {path}: {exc}", file=sys.stderr)
            continue
        count = len(document["records"])
        noun = "record" if count == 1 else "records"
        print(f"VERIFIED {document['attestation_id']} ({count} {noun}): {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
