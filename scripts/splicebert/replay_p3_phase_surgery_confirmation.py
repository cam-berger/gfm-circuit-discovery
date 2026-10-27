#!/usr/bin/env python3
"""Re-execute C36's frozen confirmation analysis from the committed shards, model-free.

The frozen analyzer reads its package inputs and shard paths relative to the working directory, at
the paths the package pinned. Ten of those inputs were later moved under
``results/phase-lattice/development/`` (bytes unchanged; see ``tests/data/hash_pin_ledger.json``) and
the shards are committed only as a tarball, so running it in the tree raises FileNotFoundError.

This replay stages every pinned input by sha256 into a temporary root, extracts the shards there,
runs the unmodified analyzer with that root as its working directory, and requires its output to be
byte-identical to the committed ``confirmation_analysis.json``. Nothing in the tree is moved or
written.

    uv run --extra splicebert --extra dev python scripts/splicebert/replay_p3_phase_surgery_confirmation.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[2]
ANALYZER = REPO / "scripts/splicebert/analyze_p3_phase_surgery_confirmation.py"
PACKAGE = "results/phase-lattice/p3_phase_surgery_confirmation_package_v1.json"
AUDIT = "results/phase-lattice/p3_phase_surgery_confirmation_package_v1.audit.json"
RUN_DIR = "results/phase-lattice/p3_phase_surgery_confirmation_v1"
TARGET = f"{RUN_DIR}/confirmation_analysis.json"
SEARCH_ROOT = "results/phase-lattice"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(record: dict) -> Path:
    """The committed file holding a pinned record's bytes: its pinned path, else any same-hash file."""
    pinned = REPO / record["path"]
    if pinned.is_file() and sha256(pinned) == record["sha256"]:
        return pinned
    for candidate in sorted((REPO / SEARCH_ROOT).rglob(PurePosixPath(record["path"]).name)):
        if candidate.is_file() and sha256(candidate) == record["sha256"]:
            return candidate
    raise FileNotFoundError(f"no committed file matches {record['path']} ({record['sha256'][:12]})")


def stage(root: Path) -> None:
    package = json.loads((REPO / PACKAGE).read_text())
    for record in package["source_artifacts"].values():
        destination = root / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(locate(record), destination)
    run_dir = root / RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("run_complete.json", "primary_manifest.json", "direction_control_manifest.json",
                 "matched_context_manifest.json"):
        shutil.copyfile(REPO / RUN_DIR / name, run_dir / name)
    with tarfile.open(REPO / RUN_DIR / "confirmation_shards.tar.gz") as archive:
        archive.extractall(run_dir, filter="data")


def replay() -> tuple[str, str]:
    """Return (replayed sha256, committed sha256)."""
    with tempfile.TemporaryDirectory(prefix="c36-replay-") as scratch:
        root = Path(scratch)
        stage(root)
        out = root / "replay/confirmation_analysis.json"
        subprocess.run(
            [sys.executable, str(ANALYZER), "--package", str(REPO / PACKAGE),
             "--package-audit", str(REPO / AUDIT), "--run-dir", RUN_DIR,
             "--bootstrap", "20000", "--seed", "20260721", "--out", str(out)],
            cwd=root, check=True, capture_output=True, text=True,
        )
        return sha256(out), sha256(REPO / TARGET)


def main() -> None:
    replayed, committed = replay()
    if replayed != committed:
        raise SystemExit(f"NOT_REPRODUCED: replay {replayed} != committed {committed}")
    print(f"REPRODUCED_BITWISE {TARGET} sha256={committed}")


if __name__ == "__main__":
    main()
