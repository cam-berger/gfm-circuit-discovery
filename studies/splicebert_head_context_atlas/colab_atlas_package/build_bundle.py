#!/usr/bin/env python
"""Build the self-contained Colab zip and an external hash sidecar."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent

EXPLICIT = [
    "scripts/splicebert_head_context_atlas/build_atlas_matched_cohorts.py",
    "scripts/splicebert_head_context_atlas/run_all_head_screen.py",
    "scripts/splicebert_head_context_atlas/run_group_mlp_closure.py",
    "scripts/splicebert_head_context_atlas/run_atlas_model_preflight.py",
    "scripts/splicebert_head_context_atlas/run_atlas_discovery_screen.py",
    "scripts/splicebert_head_context_atlas/validate_atlas_discovery_screen.py",
    "scripts/splicebert_head_context_atlas/merge_atlas_discovery_screen.py",
    "tests/test_atlas_context.py",
    "tests/test_atlas_matching.py",
    "tests/test_competition.py",
    "tests/test_head_context_screen.py",
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.discovery.pairs.jsonl",
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.discovery.packets.jsonl",
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.discovery.sequences.jsonl",
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.summary.json",
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.manifest.json",
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.audit.json",
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.audit.manifest.json",
    "results/splicebert_head_context_atlas/atlas_model_preflight_v1.rows.jsonl",
    "results/splicebert_head_context_atlas/atlas_model_preflight_v1.summary.json",
    "results/splicebert_head_context_atlas/atlas_model_preflight_v1.manifest.json",
    "studies/splicebert_head_context_atlas/ATLAS_COHORT_V1.md",
    "studies/splicebert_head_context_atlas/RESEARCH_SPINE.md",
    "studies/splicebert_head_context_atlas/CLAIMS.md",
]
PACKAGE_FILES = [
    "README.md", "CHECKLIST.md", "requirements-colab.txt", "run_atlas_screen_colab.ipynb",
]
MODEL_DIR = REPO / "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PACKAGE / "dist/gfm_atlas_screen_colab_v1.zip"))
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(output.suffix + ".sha256.json").exists():
        raise FileExistsError(f"refuse to overwrite existing bundle {output}")

    files: dict[str, Path] = {}
    for path in (REPO / "src/gfmcd").rglob("*.py"):
        files[str(path.relative_to(REPO))] = path
    for relative in EXPLICIT:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = path
    for path in MODEL_DIR.iterdir():
        if path.is_file():
            files[str(path.relative_to(REPO))] = path
    for name in PACKAGE_FILES:
        path = PACKAGE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[f"studies/splicebert_head_context_atlas/colab_atlas_package/{name}"] = path

    manifest = {
        "schema": "head_context_atlas.colab_bundle/1",
        "claim_boundary": "discovery-only all-component screen; no validation/sealed data",
        "files": {
            relative: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for relative, path in sorted(files.items())
        },
    }
    root = "gfm_atlas_screen/repo"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, path in sorted(files.items()):
            archive.write(path, f"{root}/{relative}")
        archive.writestr(
            f"{root}/BUNDLE_MANIFEST.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    sidecar = {
        "bundle": str(output),
        "size_bytes": output.stat().st_size,
        "sha256": sha256(output),
        "n_files": len(files) + 1,
    }
    output.with_suffix(output.suffix + ".sha256.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
