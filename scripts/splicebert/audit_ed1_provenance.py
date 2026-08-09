#!/usr/bin/env python
"""ED1 fix-forward PROVENANCE AUDIT.

The manifest written beside the confirmatory artifact is inadequate and is preserved
unedited as the historical record. It states `git_commit = 8490d05` with
`git_dirty = true` -- but that parent revision does **not contain the ED1 runner at all**,
so the recorded commit cannot reconstruct the code that produced the result. It also
lists no outputs, and only the model checkpoint as an input, omitting the GTF, the genome
FASTA and the PWM baseline that the cohort and every covariate depend on.

This script does not rewrite that manifest. It emits a SEPARATE, immutable audit that
pins what the manifest should have pinned:

  * the commit that actually contains the ED1 sources, plus each source file's git blob
    hash AND its on-disk sha256 (so a dirty tree is detectable rather than merely flagged);
  * sha256 of the raw score artifacts and of every result artifact (declared as outputs);
  * sha256 + size of the biological inputs (GTF, FASTA, FASTA index, PWM baseline) and of
    the model checkpoint;
  * the interpreter, key library versions, and the sealed constants in force.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402

SOURCES = [
    "src/gfmcd/splicing/exon_intron.py",
    "src/gfmcd/splicing/coupling.py",
    "scripts/splicebert/run_ed1_architecture_boundary_pairing.py",
    "scripts/splicebert/run_ed1_cohort_pilot.py",
    "scripts/splicebert/run_ed1_matching_pilot.py",
    "scripts/splicebert/run_ed1_sensitivity_bound.py",
    "scripts/splicebert/verify_ed1.py",
    "scripts/splicebert/seal_ed1_constants.py",
    "tests/test_ed1_determinism.py",
    "tests/test_ed1_inference_calibration.py",
]
RESULTS = [
    "results/splicebert_coupling/ed1_architecture_boundary_pairing.json",
    "results/splicebert_coupling/ed1_sensitivity_bound.json",
    "results/splicebert_coupling/development/ed1_scores_val.json",
    "results/splicebert_coupling/development/ed1_scores_trainpilot.json",
    "results/splicebert_coupling/development/ed1_val_secondary_complete.json",
    "results/splicebert_coupling/development/ed1_trainpilot_sealed.json",
]
BIO_INPUTS = [
    os.environ.get("GFMCD_GTF", "/nonexistent/gencode.gtf.gz"),
    os.environ.get("GFMCD_FASTA", "/nonexistent/hg38.fa"),
    os.environ.get("GFMCD_FASTA_FAI", "/nonexistent/hg38.fa.fai"),
    str(REPO.parents[2] / "colab_results/splicebert_run/baselines/local_pair_baseline.pkl"),
    str(REPO / "data/models/splicebert_official_archive/extracted/models/"
               "SpliceBERT.1024nt/pytorch_model.bin"),
]


def sha256(path: Path, chunk: int = 1 << 22) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite-record", action="store_true")
    args = ap.parse_args()

    head = git("rev-parse", "HEAD")
    src = {}
    for rel in SOURCES:
        p = REPO / rel
        src[rel] = dict(
            git_blob=git("rev-parse", f"HEAD:{rel}") or None,
            disk_sha256=sha256(p),
            tracked_clean=not bool(git("status", "--porcelain", "--", rel)))
    res = {rel: dict(sha256=sha256(REPO / rel),
                     bytes=(REPO / rel).stat().st_size if (REPO / rel).exists() else None)
           for rel in RESULTS}
    bio = {}
    for path in BIO_INPUTS:
        p = Path(path)
        bio[p.name] = dict(path=str(p), sha256=sha256(p),
                           bytes=p.stat().st_size if p.exists() else None)

    import numpy
    import torch
    import transformers
    audit = dict(
        note="Fix-forward provenance audit for ED1. The manifest written beside the "
             "confirmatory artifact recorded git_commit=8490d05 (dirty), a revision that "
             "does NOT contain the ED1 runner; it is preserved unedited and superseded by "
             "this file for provenance purposes.",
        ed1_commit=head,
        ed1_commit_subject=git("log", "-1", "--format=%s"),
        worktree_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        tree_clean_for_ed1_sources=all(v["tracked_clean"] for v in src.values()),
        sources=src, outputs=res, biological_inputs=bio,
        environment=dict(python=sys.version.split()[0], numpy=numpy.__version__,
                         torch=torch.__version__, transformers=transformers.__version__,
                         pythonhashseed="0 (pinned at launch)"),
        sealed_constants=dict(SESOI=0.106, native_coupling_floor=0.265,
                              balance_max_smd=0.10, n_boot=2000, alpha=0.05,
                              arch_short_max=200, arch_long_min=300,
                              sham="ShamPolicy(min_sep=25, max_log2_ratio=1.0, k=1)"),
        frozen_verdict="NOT LICENSED AS A SWITCH (prereg S6 row 2); artifact records "
                       "GATES_FAILED because planted_confound_null failed. The sensitivity "
                       "bound is a POST-RUN robustness interpretation, not an outcome.")

    out = guard_output_path(
        REPO / (args.out or "results/splicebert_coupling/ed1_provenance_audit.json"),
        overwrite=args.overwrite_record)
    out.write_text(json.dumps(audit, indent=1))
    from gfmcd.provenance import ExperimentManifest
    man = ExperimentManifest.create(
        "splicebert_coupling:ed1_provenance_audit",
        [sys.argv[0], *sys.argv[1:]], repo=REPO, status="INFRASTRUCTURE",
        config=dict(n_sources=len(SOURCES), n_outputs=len(RESULTS), n_inputs=len(BIO_INPUTS)),
        notes=["Fix-forward provenance for ED1. Supersedes, but does not edit, the manifest "
               "beside the confirmatory artifact, which recorded git_commit=8490d05 (dirty) -- "
               "a revision that does not contain the ED1 runner."])
    for path in BIO_INPUTS:
        if Path(path).exists():
            man.add_input(path)
    man.write(out.parent / f"{out.stem}.manifest.json")
    miss = [k for k, v in bio.items() if v["sha256"] is None]
    print(f"ED1 provenance audit -> {out}")
    print(f"  ed1 commit {head[:12]}  sources clean: {audit['tree_clean_for_ed1_sources']}")
    print(f"  sources pinned: {len(src)}   outputs pinned: {sum(v['sha256'] is not None for v in res.values())}/{len(res)}")
    print(f"  biological inputs hashed: {len(bio)-len(miss)}/{len(bio)}" + (f"  MISSING {miss}" if miss else ""))


if __name__ == "__main__":
    main()
