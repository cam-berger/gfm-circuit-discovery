# Data provenance

How every load-bearing data artifact in this repo was produced, and how to regenerate it from a
clean checkout. Companion to `data/references/manifest.json` (reference downloads) and the `.meta.json`
cache sidecars (generated windows).

## Reference genome + annotation

Not in the repo (gitignored, large). Fetch and verify:

```bash
uv run python scripts/fetch_reference.py               # download + checksum + install
uv run python scripts/fetch_reference.py --verify-only # verify what's installed
```

- Installs into `data/references/` by default; override with `GFMCD_REFERENCE_DIR` (e.g. an existing
  `~/Research/data/genomes`). All genome scripts resolve this via `gfmcd.paths`.
- Files, URLs, and licenses are pinned in `data/references/manifest.json`.
- **Checksums:** `sha256` is `null` in the manifest until the first verified download. The fetcher
  computes the digest, prints it, and asks you to pin it into the manifest. We do not fabricate
  checksums (TO_FIX.md §12). Once pinned, `--verify-only` enforces them.
- Record `retrieved` (date) in the manifest when you pin a checksum.

## Cached windows (`data/cache/*.npz`, `*.npy`)

Load-bearing: they stand in for the genome for cells C/D and the codon-usage table. Each generated
cache should carry a `<file>.meta.json` sidecar (via `gfmcd.provenance.write_cache_sidecar`) recording
the generator + version, source reference checksums, config, seed, split definition, array
shapes/dtypes, timestamp, and git commit. A loader should compare requested parameters against the
sidecar and refuse a stale cache. Filenames already encode the `(n, window, seed)` recipe; the sidecar
makes the rest explicit. Backfilling sidecars for existing caches is tracked in `docs/TODO.md` §1.3.

## Checkpoints (`checkpoints/*.pt`)

Gitignored. Inspect metadata without loading onto an accelerator:

```bash
uv run python scripts/inspect_checkpoint.py checkpoints/<name>.pt
```

Every checkpoint should record model + optimizer config, objective and weights (e.g. the repeat-copy
`copy_weight`), task config, data seed and model seed separately, training step / token count, git
commit, and package versions (TO_FIX.md §13.2). `inspect_checkpoint.py` prints what is present and
flags what is missing.

## Result dumps (`results/**/fig_data/*.json`, run registry)

New result files carry a `_provenance` block (`gfmcd.provenance.stamp`) with status, seed, script,
command, git SHA + dirty flag, timestamp, and versions. Runs additionally write a `manifest.json` +
`command.txt` under `results/registry/<run_id>/`. `docs/RESULTS.md` remains the human-readable source
of truth; the registry is the machine-readable trail.
