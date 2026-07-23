# Colab package — all-96-head A/B/C/D exact-effect screen

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Screens **every one of the 96 SpliceBERT heads** for its exact scrub-to-corrupt effect on splice-site
reconstruction under four lenses (A general reconstruction, B partner coupling, C **competitor-site
perturbation response**, D matched-decoy contrast), across the tagged intron-mate **discovery** cohort.
Output = an **exploratory head-effect matrix**, not a confirmed atlas.

## What this is NOT
- Not a competition-specificity result. The C specificity pilot was **discovery-stage instrument-limited**
  (|M_comp| ≈ |M_decoy|); C here is a **competitor-site perturbation response**, never "competition-specific
  recruitment."
- Not a sealed-chromosome run. chr8/9 are **hard-excluded**; `--unlock-sealed` needs a separate confirm config.
- Not a biological-function assignment. A head's topology association is descriptive only.
- Not a single-head-sparsity assumption. Per-head nulls do NOT imply a distributed mechanism is absent
  (C33); the merge step emits group/faithfulness-size summaries.

## Files
| File | Role |
|---|---|
| `../../../scripts/splicebert_head_context_atlas/run_all_head_screen.py` | core CLI (scrub-to-corrupt, 96 heads, immutable resumable shards, 12 mechanical gates) |
| `validate_head_screen.py` / `merge_head_screen.py` / `benchmark_colab.py` | validate shards · merge → matrix + C33 summary · benchmark |
| `config.discovery.json` | discovery config (override paths on Colab; verify sha256s) |
| `run_head_screen_colab.ipynb` | orchestration only (mount / clone@commit / install / verify / smoke / benchmark / launch / validate+merge) |
| `CHECKLIST.md` | green/red readiness ladder |
| `tests/test_competition.py`, `tests/test_head_context_screen.py` | mechanical-gate + determinism + vectorized/scalar-agreement tests |

## Pins (in the run manifest)
git commit + dirty, checkpoint sha `2ad91428…`, comprehensive GTF sha `83fba3e9…13a453`, FASTA, python /
torch / transformers / CUDA / GPU, precision (**float32; TF32/AMP off**), seeds + `PYTHONHASHSEED=0`,
expected/completed shard counts + every shard sha.

## Run order (Colab)
1. Local first: `pytest tests/test_competition.py tests/test_head_context_screen.py` + a local actual-model smoke.
2. Colab **smoke**: 1 example, all 96 heads, every mechanical gate; reproduce local baseline margins within tol.
3. Colab **benchmark**: 8–16 examples → GPU type, peak mem, examples/hr, sec/head, projected GPU-hours + storage.
4. **STOP.** Do not launch the full run without Cameron's explicit GO after reviewing the projection
   (`--launch-full` / a manual cell).
5. Full discovery run → `validate_head_screen.py` → `merge_head_screen.py`.

Projected full discovery run (4,864 usable genes): ~15.7 **MPS**-hours (local proxy) / ~315 MB; the Colab
CUDA benchmark supplies the real GPU-hour projection.
