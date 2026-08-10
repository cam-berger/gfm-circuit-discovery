#!/usr/bin/env python
"""Independently re-read and audit every Phase-I freshness source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.pbc_freshness import audit_registry  # noqa: E402


def _read(path: str | Path) -> dict:
    with Path(path).expanduser().resolve(strict=True).open("rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("audit inputs must be JSON objects")
    return payload


def audit_artifact(
    *,
    registry_path: str | Path,
    source_spec_path: str | Path,
    output_path: str | Path,
    repo: str | Path = REPO,
) -> tuple[Path, Path, dict]:
    registry_file = Path(registry_path).expanduser().resolve(strict=True)
    spec_file = Path(source_spec_path).expanduser().resolve(strict=True)
    output = guard_output_path(Path(output_path).expanduser().resolve())
    if output.suffix != ".json":
        raise ValueError("freshness audit output must be JSON")
    manifest_path = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    registry = _read(registry_file)
    spec = _read(spec_file)
    audit = audit_registry(registry, spec, repo=repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert-positional-boundary-coherence:freshness-registry-audit-v1",
        sys.argv,
        repo=REPO,
        status=str(audit["status"]),
        config={"stage": "construction"},
        notes=["Independent source re-read; no producer function or model output is used."],
    )
    for path in (
        registry_file,
        spec_file,
        Path(__file__).resolve(),
        REPO / "src/gfmcd/splicing/pbc_freshness.py",
    ):
        manifest.add_input(path)
    for source in spec.get("sources", []):
        if isinstance(source, dict) and source.get("revision") == "WORKTREE":
            source_path = Path(str(source.get("path", "")))
            if not source_path.is_absolute():
                source_path = Path(repo).resolve() / source_path
            if source_path.is_file():
                manifest.add_input(source_path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    return output, manifest_path, audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--source-spec", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output, manifest, audit = audit_artifact(
        registry_path=args.registry,
        source_spec_path=args.source_spec,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
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
