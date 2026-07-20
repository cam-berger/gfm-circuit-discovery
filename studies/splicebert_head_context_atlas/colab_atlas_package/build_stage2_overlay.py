#!/usr/bin/env python
"""Build a self-contained Stage 2 overlay for the frozen Atlas Colab bundle.

The overlay is extracted into ``gfm_atlas_screen/repo`` from the discovery bundle.
Keep every non-stdlib sibling import used by the Stage 2 executor in this explicit
list: a missing sibling otherwise appears only after a costly Colab setup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent

STAGE2_SCRIPTS = [
    # Stage 2 entry points.
    "scripts/splicebert_head_context_atlas/run_atlas_stage2_mechanisms.py",
    "scripts/splicebert_head_context_atlas/validate_atlas_stage2_mechanisms.py",
    "scripts/splicebert_head_context_atlas/merge_atlas_stage2_mechanisms.py",
    "scripts/splicebert_head_context_atlas/analyze_atlas_stage2_mechanisms.py",
    "scripts/splicebert_head_context_atlas/validate_atlas_stage2_analysis.py",
    # Direct sibling dependency imported by the Stage 2 executor. The discovery
    # bundle predates this file, so the overlay must carry it explicitly.
    "scripts/splicebert_head_context_atlas/run_atlas_mechanism_validation.py",
    # Complete transitive sibling-import closure. Carry these even though an older
    # version also exists in the discovery bundle; Stage 2 must execute against the
    # exact audited source set used to build its package.
    "scripts/splicebert_head_context_atlas/run_all_head_screen.py",
    "scripts/splicebert_head_context_atlas/run_atlas_discovery_screen.py",
    "scripts/splicebert_head_context_atlas/run_atlas_model_preflight.py",
    "scripts/splicebert_head_context_atlas/build_atlas_matched_cohorts.py",
    # Post-audit representative attribution export and its independent verifier.
    "scripts/splicebert_head_context_atlas/build_atlas_representative_dags.py",
    "scripts/splicebert_head_context_atlas/validate_atlas_representative_dags.py",
    "scripts/splicebert/export_segment_attribution_bundle.py",
]

DOCUMENTS = [
    "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_PLAN_V1.md",
    "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_AMENDMENT_A1.md",
    "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_AMENDMENT_A2.md",
    "studies/splicebert_head_context_atlas/MECHANISM_VALIDATION_AMENDMENT_A3.md",
]

COHORT_AND_PACKAGES = [
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.validation.pairs.jsonl",
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.validation.packets.jsonl",
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.validation.sequences.jsonl",
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.summary.json",
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.manifest.json",
    "results/splicebert_head_context_atlas/atlas_context_matched_v7.audit.json",
    "results/splicebert_head_context_atlas/atlas_mechanism_package_v1.json",
    "results/splicebert_head_context_atlas/atlas_mechanism_package_v1.audit.json",
    "results/splicebert_head_context_atlas/atlas_stage2_package_v1.json",
    "results/splicebert_head_context_atlas/atlas_stage2_package_v1.audit.json",
]

HEALTH_DIR = REPO / "results/splicebert_head_context_atlas/atlas_stage2_mechanism_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(PACKAGE / "dist/gfm_atlas_stage2_colab_overlay_v6.zip"),
    )
    args = parser.parse_args()
    output = Path(args.out)
    sidecar_path = output.with_suffix(output.suffix + ".sha256.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or sidecar_path.exists():
        raise FileExistsError(f"refuse to overwrite existing Stage 2 overlay {output}")

    files: dict[str, Path] = {}
    for relative in STAGE2_SCRIPTS + DOCUMENTS + COHORT_AND_PACKAGES:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = path

    # Runtime-reference closure: Stage 2 checks its conditional interventions
    # against the frozen whole-validation rows.  These paths are data dependencies
    # declared by the audited package, so import-only closure is insufficient.
    stage2_package = json.loads(
        (REPO / "results/splicebert_head_context_atlas/atlas_stage2_package_v1.json").read_text()
    )
    for name, record in sorted(stage2_package["whole_validation_row_references"].items()):
        relative = record["path"]
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing Stage 2 runtime reference {name}: {path}")
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Stage 2 runtime reference changed after freeze: {name}")
        files[relative] = path

    health_manifest = HEALTH_DIR / "run_manifest.health.json"
    if not health_manifest.is_file():
        raise FileNotFoundError(health_manifest)
    files[str(health_manifest.relative_to(REPO))] = health_manifest
    health_shards = sorted((HEALTH_DIR / "health_shards").glob("packet_*.json"))
    if len(health_shards) != 256:
        raise RuntimeError(f"expected 256 Stage 2 health shards, found {len(health_shards)}")
    for path in health_shards:
        files[str(path.relative_to(REPO))] = path

    # Import-surface regression guard: the original v1 overlay failed because this
    # direct import existed in the entry point but its file was absent.
    required = {
        "scripts/splicebert_head_context_atlas/run_atlas_mechanism_validation.py",
        "scripts/splicebert_head_context_atlas/run_all_head_screen.py",
        "scripts/splicebert_head_context_atlas/run_atlas_discovery_screen.py",
        "scripts/splicebert_head_context_atlas/run_atlas_model_preflight.py",
        "scripts/splicebert_head_context_atlas/build_atlas_matched_cohorts.py",
        "scripts/splicebert_head_context_atlas/build_atlas_representative_dags.py",
        "scripts/splicebert_head_context_atlas/validate_atlas_representative_dags.py",
        "scripts/splicebert/export_segment_attribution_bundle.py",
    }
    missing = sorted(required - files.keys())
    if missing:
        raise RuntimeError(f"missing transitive Stage 2 dependencies: {missing}")

    manifest = {
        "schema": "head_context_atlas.stage2_colab_overlay/6",
        "claim_boundary": (
            "A3 audit/recovery and post-audit DAG tooling; contains no target outcome or mechanism label"
        ),
        "base_bundle": "gfm_atlas_screen_colab_v1.zip",
        "files": {
            relative: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for relative, path in sorted(files.items())
        },
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, path in sorted(files.items()):
            archive.write(path, relative)
        archive.writestr(
            "STAGE2_OVERLAY_MANIFEST.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )

    sidecar = {
        "schema": "head_context_atlas.stage2_colab_overlay_sidecar/6",
        "bundle": output.name,
        "size_bytes": output.stat().st_size,
        "sha256": sha256(output),
        "n_files": len(files) + 1,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
