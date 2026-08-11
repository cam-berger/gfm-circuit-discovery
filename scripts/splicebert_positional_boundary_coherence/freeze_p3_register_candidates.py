#!/usr/bin/env python
"""Deterministically freeze Phase-III candidates, nulls, and D/R/Q resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.pbc_register_candidates import (  # noqa: E402
    build_candidate_freeze,
)


def output_paths(
    freeze_path: str | Path, mde_path: str | Path
) -> tuple[Path, Path, Path, Path]:
    freeze = guard_output_path(Path(freeze_path).expanduser().resolve())
    mde = guard_output_path(Path(mde_path).expanduser().resolve())
    freeze_manifest = guard_output_path(
        freeze.with_name(freeze.stem + ".manifest.json")
    )
    mde_manifest = guard_output_path(mde.with_name(mde.stem + ".manifest.json"))
    if len({freeze, mde, freeze_manifest, mde_manifest}) != 4:
        raise ValueError("candidate-freeze output paths must be distinct")
    return freeze, mde, freeze_manifest, mde_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", required=True)
    parser.add_argument("--freeze-out", required=True)
    parser.add_argument("--path-mde-out", required=True)
    args = parser.parse_args()

    freeze_path, mde_path, freeze_manifest_path, mde_manifest_path = output_paths(
        args.freeze_out, args.path_mde_out
    )
    screen_path = Path(args.screen).expanduser().resolve()
    screen = json.loads(screen_path.read_text())
    screen_record = file_record(screen_path)
    freeze, path_mde = build_candidate_freeze(
        screen,
        screen_record=screen_record,
    )
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    mde_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    mde_path.write_text(json.dumps(path_mde, indent=2, sort_keys=True) + "\n")
    common = {
        "stage": "development",
        "candidate_screen_sha256": screen_record["sha256"],
        "candidate_count": freeze["selection"]["candidate_count"],
        "selection_class": freeze["selection"]["classification"],
    }
    freeze_manifest = ExperimentManifest.create(
        "pbc-p3-register-candidate-freeze-v1",
        sys.argv,
        repo=REPO,
        status="DEVELOPMENT_FREEZE",
        config=common,
    )
    freeze_manifest.add_input(screen_path)
    freeze_manifest.add_output(freeze_path)
    freeze_manifest.write(freeze_manifest_path)
    mde_manifest = ExperimentManifest.create(
        "pbc-p3-register-path-mde-v1",
        sys.argv,
        repo=REPO,
        status="DEVELOPMENT_RESOLUTION",
        config={**common, "mde80": path_mde["mde80"]},
    )
    mde_manifest.add_input(screen_path)
    mde_manifest.add_input(freeze_path)
    mde_manifest.add_output(mde_path)
    mde_manifest.write(mde_manifest_path)
    print(
        json.dumps(
            {
                "status": freeze["status"],
                "selection": freeze["selection"],
                "path_mde": path_mde["mde80"],
                "path_mde_passed": path_mde["passed"],
                "freeze_output": str(freeze_path),
                "path_mde_output": str(mde_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
