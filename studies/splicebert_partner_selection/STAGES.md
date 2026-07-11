# Stages log

One row per executed stage with the call and the load-bearing numbers.

## Stage -1 — pre-flight (2026-07-10) — GO
- Env resolves; SpliceBERT.1024nt loads (6 layers, hidden 512); dataset build returns rows both strands.
- Cost: full local MPS matrix ~19 h → moved to Colab GPU with scale caps.

## Stage 0 — data + local baseline (2026-07-11) — done
- GENCODE v50 / GRCh38 dataset v1 built; val.jsonl byte-identical local vs Colab.
- Local PWM+length baseline: val top1 0.412. Conflict subset = 10,034 examples where baseline picks a decoy.

## Stage 1 — behavioral matrix (2026-07-11) — GO
- 25 runs (5 regimes × 5 seeds) on Colab GPU.
- additive-frozen conflict PMR 0.807 [0.802–0.812]; randomized-backbone control 0.056 [0.047–0.062]; non-overlapping.
- **Bug fixed mid-stage:** randomized control was a no-op (transformers ≥5 init_weights); rebuilt via from_config, all 5 control seeds re-run.
- Formal gene-grouped bootstrap (503 genes): margin-diff mean 3.36, worst CI low 2.84, all 5 seeds exclude 0.
- Gate: conflict PMR ≥0.60 ✅, grouped CI>0 ✅, replication 5/5 ✅, beats baseline ✅, beats control ✅.
- **CALL: GO** (recommended). Human elected to freeze + stop for review before tracing.
- Preregistration frozen (see FREEZE.md). Test split sealed.

## Stage 2 — circuit tracing — NOT STARTED
- Blocked on explicit human GO (opens new compute).
- Plan: freeze component set → single-head zero/mean/resample screen → cross-null consensus → grouped necessity vs matched-random null → clean/corrupt patching → token-localized patching → QK/OV path decomposition → replication across additive-frozen seeds. Test opened once at the end.
