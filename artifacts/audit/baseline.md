# Cleanup baseline

Recorded before any edit, per `TO_FIX.md` §2. This is the state the cleanup starts from.

| field | value |
|---|---|
| commit hash | `dfc45c5d73c0caa5afa77cb18cb91b2949d4dd69` (branch `master`) |
| working branch | `cleanup/scientific-claims-and-splicebert` |
| Python | 3.13.12 |
| uv | 0.11.8 (Homebrew) |
| OS | macOS 26.5.1 (build 25F80), Darwin 25.5.0 |
| accelerator | Apple Silicon MPS (available) |

## Test suite

```
uv run pytest --collect-only -q   -> 131 tests collected
uv run pytest -q                  -> 131 passed in 68.28s
```

- **131 collected · 131 passed · 0 failed · 0 skipped.**
- The full command used: `uv run pytest -q` (environment synced with
  `uv sync --extra lab --extra dev`).
- Environment redirected on this host (restricted `~/.local`): `UV_PROJECT_ENVIRONMENT`,
  `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` set to `$TMPDIR` subdirs; `UV_PYTHON_PREFERENCE=only-system`
  (matches `docs/TODO.md` §3.3).

## Missing-data / checkpoint notes

- The `models` and `genome` extras were **not** installed for this baseline (only `lab` + `dev`); the
  synthetic + lab suite is fully green without them. SpliceBERT (`models`) and real-hg38 regeneration
  (`genome`) work is deferred to the parts of the spec that need it (Parts II §12, IV).
- All 131 tests pass with 0 skips, so no test in the current suite is gated on the absent raw genome
  (`data/genomes/hg38.fa`) — the genome-dependent results live in scripts, not the test suite. Adding
  genome/checkpoint skip markers is spec §15.

## Stale test-count claims to correct (spec §2, §A3-of-my-notes)

Active docs currently assert 102 / 127 / ~130 / 130 / 53 / 92 / 2 / 3; the collected count is **131**.
Replace hard-coded counts in active prose with the command a reader should run.
