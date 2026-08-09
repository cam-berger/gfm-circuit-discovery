#!/usr/bin/env python
"""Build the self-contained R4 Colab zip and an external hash sidecar.

Follows `studies/splicebert_head_context_atlas/colab_atlas_package/build_bundle.py` rather than
inventing a second pattern: per-file sha256 in an in-zip `BUNDLE_MANIFEST.json`, an external
`.sha256.json` sidecar, and a refusal to overwrite an existing bundle.

THE SEAL IS ENFORCED BY ABSENCE.  The cohort ships as sequences, and this builder REFUSES to
write a bundle containing any chr8 or chr9 record.  A remote runtime cannot spend the seal by
accident because the data is not there -- the same mechanism the Atlas package uses.

NO GTF OR FASTA.  The cohort is bundled as outcome-independent sequence windows, so Colab needs
neither the 3.2 GB genome nor the 80 MB annotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent

EXPLICIT = [
    "scripts/splicebert_acceptor_coupling/run_r4_paired_edge_acdc.py",
    "scripts/splicebert_acceptor_coupling/run_r3_paired_component_scan.py",
    "tests/test_acdc_graph.py",
    "tests/test_acdc_kernel.py",
    "tests/test_acdc_algorithm.py",
    "tests/test_acdc_device.py",
    "tests/test_reverse_coupling_memorials.py",
    "tests/test_motif_cleanliness.py",
    "results/splicebert_acceptor_coupling/r4_reference_smoke.json",
    "studies/splicebert_acceptor_coupling/CLAIMS.md",
    "studies/splicebert_acceptor_coupling/README.md",
    "studies/splicebert_acceptor_coupling/DECISIONS_R4_EDGE.md",
    "studies/splicebert_acceptor_coupling/preregistrations/"
    "PREREGISTRATION_R4_EDGE_LEVEL_CAUSAL_CORE.md",
]
PACKAGE_FILES = ["README.md", "CHECKLIST.md", "requirements-colab.txt", "run_r4_colab.ipynb"]
#: ``data/`` is gitignored, so a git worktree does not carry the checkpoint.  The default is the
#: in-repo path; ``--model-dir`` (or ``GFMCD_MODEL_DIR``) points at the main checkout when
#: building from a worktree.
MODEL_RELATIVE = "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"
SEALED = ("chr8", "chr9")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_sealed_records(paths: dict[str, Path]) -> dict:
    """Refuse to bundle anything naming chr8/chr9 in a cohort or result payload."""
    checked = []
    for relative, path in sorted(paths.items()):
        if path.suffix not in (".json", ".jsonl"):
            continue
        text = path.read_text(errors="ignore")
        hits = [c for c in SEALED if f'"{c}"' in text]
        if hits:
            raise SystemExit(
                f"REFUSING to bundle {relative}: it names {hits}. The seal is enforced by "
                "absence -- a remote runtime must not be able to spend chr8/9 by accident."
            )
        checked.append(relative)
    return {"n_payloads_checked": len(checked), "sealed_chroms": list(SEALED)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PACKAGE / "dist/gfm_r4_edge_acdc_colab_v1.zip"))
    parser.add_argument("--model-dir", default=os.environ.get("GFMCD_MODEL_DIR"),
                        help="checkpoint directory; needed when building from a git worktree, "
                             "where data/ is gitignored and absent")
    args = parser.parse_args()
    model_dir = Path(args.model_dir) if args.model_dir else REPO / MODEL_RELATIVE
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = output.with_suffix(output.suffix + ".sha256.json")
    if output.exists() or sidecar_path.exists():
        raise FileExistsError(f"refuse to overwrite existing bundle {output}")

    files: dict[str, Path] = {}
    for path in (REPO / "src/gfmcd").rglob("*.py"):
        files[str(path.relative_to(REPO))] = path
    for relative in EXPLICIT:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = path
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory missing: {model_dir}. Pass --model-dir or set "
            "GFMCD_MODEL_DIR; data/ is gitignored so a worktree does not carry it."
        )
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            files[f"{MODEL_RELATIVE}/{path.name}"] = path
    for name in PACKAGE_FILES:
        path = PACKAGE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[f"studies/splicebert_acceptor_coupling/colab_r4_package/{name}"] = path

    seal = assert_no_sealed_records(files)

    manifest = {
        "schema": "acceptor_coupling.r4_colab_bundle/1",
        "claim_boundary": (
            "R4 edge-level paired ACDC, verification tier, validation split. Contains no chr8/9 "
            "record and no R4 outcome. Attribution RANKS, intervention DECIDES (AC4)."
        ),
        "seal_check": seal,
        "files": {
            relative: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for relative, path in sorted(files.items())
        },
    }
    root = "gfm_r4_edge_acdc/repo"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, path in sorted(files.items()):
            archive.write(path, f"{root}/{relative}")
        archive.writestr(
            f"{root}/BUNDLE_MANIFEST.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    sidecar = {
        "bundle": str(output), "size_bytes": output.stat().st_size,
        "sha256": sha256(output), "n_files": len(files) + 1,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
