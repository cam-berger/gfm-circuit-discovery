# Readiness checklist — all-head screen

## Local (must be GREEN before Colab)
- [x] `pytest tests/test_competition.py tests/test_head_context_screen.py` — 9 passed (incl. cross-hashseed determinism + vectorized/scalar agreement)
- [x] Stage -1 pre-flight: adapter loads 96 heads; source-to-source patch is a no-op (|diff|=0)
- [x] Actual-model smoke: 4 examples, 0 gate failures, exactly 96 head records/condition
- [x] Resume: re-run writes 0 new shards; manifest rebuilt from disk; `validate` ok=true
- [x] Merge: head-effect matrix + C33 group/faithfulness summary produced
- [x] Benchmark (MPS proxy): 11.6 s/ex, ~15.7 MPS-hr for 4,864 genes, 315 MB

## Colab (before full run)
- [ ] `requirements-colab.txt` installed; torch is the preinstalled CUDA build (not reinstalled)
- [ ] checkpoint sha `2ad91428…`, GTF sha `83fba3e9…13a453`, FASTA sha verified against config
- [ ] Colab smoke: 1 example × 96 heads, all 12 mechanical gates pass
- [ ] Colab smoke margins reproduce the local baseline within frozen tolerance (float32; TF32/AMP off)
- [ ] Benchmark reported: GPU type, peak mem, examples/hr, sec/head, **projected GPU-hours + storage**
- [ ] **Cameron GO after benchmark review** → only then `--launch-full`

## 12 mechanical gates (enforced in the runner, block writing on failure)
1. tokenizer one-token-per-nt · 2. cached-MLM reconstruction / source-to-source no-op · 3. target/partner/
non-intervened coords unchanged · 4. destroys use real bases + preserve length · 5. source-to-source patch
reproduces unpatched output · 6. vectorized == scalar (test) · 7. L0H0..L5H15 indexing · 8. exactly 96
finite records/example/condition · 9. no duplicate example/head/condition · 10. resume reproduces shard
hashes · 11. cohort identical across ≥2 PYTHONHASHSEED (test) · 12. chr8/9 cannot enter a discovery run.

## Hard stops
- Any mechanical gate fails → runner refuses to write; diagnose before proceeding.
- B / C behavioral effects may be null — that does NOT block the exploratory screen.
- No sealed run, no biological-function naming, no single-head-sparsity claim, no credit spend beyond the benchmark without GO.
