#!/usr/bin/env python
"""Build the model-free Phase-I freshness exclusion registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.pbc_freshness import derive_registry  # noqa: E402


def _read(path: str | Path) -> dict:
    with Path(path).expanduser().resolve(strict=True).open("rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("freshness source spec must be a JSON object")
    return payload


def build_artifact(
    *, source_spec_path: str | Path, output_path: str | Path, repo: str | Path = REPO
) -> tuple[Path, Path, dict]:
    spec_path = Path(source_spec_path).expanduser().resolve(strict=True)
    output = guard_output_path(Path(output_path).expanduser().resolve())
    if output.suffix != ".json":
        raise ValueError("freshness registry output must be JSON")
    manifest_path = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    spec = _read(spec_path)
    registry = derive_registry(spec, repo=repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert-positional-boundary-coherence:freshness-registry-v1",
        sys.argv,
        repo=REPO,
        status="BUILT_AUDIT_PENDING",
        config={"stage": "construction"},
        notes=[
            "All required historical categories are explicit and hash-pinned.",
            "No missing source is treated as an empty cohort; no model outcome is read.",
        ],
    )
    manifest.add_input(spec_path)
    for source in registry["sources"]:
        if source["requested_revision"] == "WORKTREE":
            source_path = Path(source["path"])
            if not source_path.is_absolute():
                source_path = Path(repo).resolve() / source_path
            manifest.add_input(source_path)
    manifest.add_input(Path(__file__).resolve())
    manifest.add_input(REPO / "src/gfmcd/splicing/pbc_freshness.py")
    manifest.add_output(output)
    manifest.write(manifest_path)
    return output, manifest_path, registry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-spec", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output, manifest, registry = build_artifact(
        source_spec_path=args.source_spec, output_path=args.out
    )
    print(
        json.dumps(
            {
                "status": "BUILT_AUDIT_PENDING",
                "normalized_count": registry["normalized_count"],
                "output": str(output),
                "manifest": str(manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
