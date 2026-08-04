"""Package the DNABERT-2 pipeline outputs into a shareable archive.

Two archives are possible:

  share  notebooks + every small result file (csv/json/md/png). This is what a
         colleague needs to read the findings, and it stays small enough to email.
  full   the above plus the large arrays (.npy/.npz/.pkl/.joblib), which are the
         intermediate representations. Hundreds of MB; only needed to re-run analysis
         without recomputing embeddings.

A README carrying provenance and caveats is generated into the archive, because the
pipeline as delivered did not run unmodified and a reader needs to know what changed.
"""

import json
import zipfile
from pathlib import Path

PROJECT_DIR = Path("/content/drive/MyDrive/algoverse/dnabert")
OUTPUT_DIR = PROJECT_DIR / "outputs"

LIGHT_SUFFIXES = {".csv", ".json", ".md", ".png", ".txt", ".svg", ".pdf"}
HEAVY_SUFFIXES = {".npy", ".npz", ".pkl", ".joblib", ".pt", ".bin"}
LIGHT_MAX_BYTES = 8 * 1024 * 1024


README = """# DNABERT-2 promoter mechanistic-interpretability pipeline — results

Frozen, non-fine-tuned DNABERT-2 (`quietflamingo/dnabert2-no-flashattention`,
12 layers x 768 hidden), GUE `prom_core_all` promoter classification.
Nine notebooks, run end to end on a Colab T4.

## Layout

    <notebook>/                 one directory per stage, under outputs/
    notebooks/                  the nine notebooks as executed
    MANIFEST.json               every file, size, and modification time

Each stage writes its own conclusions file — `NOTEBOOK_*_CONCLUSIONS.md`,
`NOTEBOOK*_RESULTS.md`, or `notebook*_claims_and_limitations.json`. Read those first.

## Reading the claims

Every result downstream of notebook 01 is measured through a **fixed Layer-5 linear
diagnostic probe** trained in notebook 01. That probe is not DNABERT-2's native
promoter prediction. "Moves the probe logit" and "the model classifies promoters this
way" are different claims, and the notebooks are generally careful to say the former.
Keep that distinction when quoting these numbers.

## Provenance — what was changed to make this run

The delivered notebooks did not run unmodified. The changes below are mechanical; none
alters a computation:

1. **Notebook 01** saved to a relative path (`outputs/promoter_baseline_pilot`), which
   on Colab resolves to the runtime-local disk and is discarded with the VM. Notebook
   02 loads the Layer-5 probe from Drive, so the 01 -> 02 handoff never completed.
   01 now mounts Drive and writes to the shared path. Same files, same contents.

2. **Notebook 02** was rebuilt from its Colab `.py` export. The delivered `.ipynb` was
   a 6-cell fragment of an abandoned re-run that died at model load; the `.py` was the
   complete 17-cell version and already contained the fix for that error.

3. **Notebooks 06-09** each contained a cell that raised, followed by a cell repairing
   it. That works when driving a notebook by hand but halts a Run-all. Five patches
   let each notebook past its own superseded check; every patch is taken from the
   authors' own repair cell, and each patched cell still runs to completion so later
   cells keep the names they bind. Patched sites are commented `# [staged fix]`.

4. **Notebook 09** was authored inside notebook 08's kernel and read ten names that no
   cell in any of the nine notebooks creates (`dnabert2_local`, `model_directory`,
   `tokenizer`, `MAX_LENGTH`, `clean_sources`, `class_target_archive`,
   `promoter_basis_rank1`, `gc_basis_rank4`, and two promoter-direction targets). A
   prelude rebuilds them from notebook 08's saved artifacts. The archive keys come
   from notebook 08's own `np.savez_compressed` call and match the convention notebook
   09 already uses for its GC targets.

   **Caveat worth carrying:** the cells that originally produced those values were
   deleted before export, so notebook 09 could not reproduce its own results from the
   delivered artifact set. The values are the authors'; the wiring is a
   reconstruction. Confirm against the original session before relying on notebook
   09's mediation numbers.
"""


def classify(path):
    suffix = path.suffix.lower()
    if suffix in HEAVY_SUFFIXES:
        return "heavy"
    if suffix in LIGHT_SUFFIXES and path.stat().st_size <= LIGHT_MAX_BYTES:
        return "light"
    return "oversized" if suffix in LIGHT_SUFFIXES else "heavy"


def collect():
    results = {"light": [], "heavy": [], "oversized": []}
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            results[classify(path)].append(path)
    return results


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1024


def build(include_heavy=False, log=print):
    assert OUTPUT_DIR.exists(), f"no outputs directory at {OUTPUT_DIR}"

    buckets = collect()
    notebooks = sorted(PROJECT_DIR.glob("0[1-9]_*.ipynb"))
    assert notebooks, "no notebooks found to include"

    included = list(buckets["light"])
    if include_heavy:
        included += buckets["heavy"] + buckets["oversized"]

    name = "dnabert_pipeline_full.zip" if include_heavy else "dnabert_pipeline_results.zip"
    archive_path = PROJECT_DIR / name

    manifest = {
        "notebooks": [n.name for n in notebooks],
        "included_files": len(included),
        "excluded_large_files": 0 if include_heavy else len(buckets["heavy"]) + len(buckets["oversized"]),
        "stages": {},
    }
    for path in included:
        stage = path.relative_to(OUTPUT_DIR).parts[0]
        entry = manifest["stages"].setdefault(stage, [])
        entry.append({"file": path.name, "bytes": path.stat().st_size})

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("README.md", README)
        zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        for notebook in notebooks:
            zf.write(notebook, f"notebooks/{notebook.name}")
        for path in included:
            zf.write(path, str(path.relative_to(OUTPUT_DIR)))

    # report
    log(f"\n{'stage':46s} {'files':>6s} {'size':>12s}")
    log("-" * 68)
    total = 0
    for stage in sorted(manifest["stages"]):
        entries = manifest["stages"][stage]
        size = sum(e["bytes"] for e in entries)
        total += size
        log(f"{stage:46s} {len(entries):6d} {human(size):>12s}")
    log("-" * 68)
    log(f"{'included (uncompressed)':46s} {len(included):6d} {human(total):>12s}")
    log(f"{'notebooks':46s} {len(notebooks):6d}")

    if not include_heavy:
        skipped = buckets["heavy"] + buckets["oversized"]
        skipped_bytes = sum(p.stat().st_size for p in skipped)
        log(f"\nExcluded {len(skipped)} large files ({human(skipped_bytes)}) — arrays and pickles.")
        log("Re-run with include_heavy=True if a colleague needs to recompute from them.")

    log(f"\nArchive: {archive_path}")
    log(f"Size:    {human(archive_path.stat().st_size)}")
    return archive_path
