# Results — behavioral gate (2026-07-11)

## Run identity

- commit: n/a (not a git repo)
- config hash: `c17c555c7cc5a01c63766cc4bb8ee7778dea43898aa2037f53779eeacd6a703f` (base.json)
- model/checkpoint: official SpliceBERT.1024nt (6 layers, hidden 512)
- dataset manifest: dataset v1; `val.jsonl` sha256 `a10f0148…bb92d071` (local == Colab)
- seeds: 0,1,2,3,4
- test opened: **no** (chr19–chr22 sealed)

## Behavioral gates

Conflict = natural cue-conflict subset (baseline picks a decoy); values are conflict positive-margin rate (mean over 5 seeds).

| Condition | Full validation (top1) | Natural conflicts (PMR) | Claim status |
|---|---:|---:|---|
| local baseline | 0.412 | ~0 (by construction) | floor |
| additive-random-frozen | 0.07 | 0.056 | control — collapses |
| **additive-frozen** | 0.87 | **0.807** | **primary evidence** |
| interaction-frozen | 0.86 | 0.787 | diagnostic (≈ additive) |
| additive-finetuned | 0.999 | 0.997 | upper bound |
| interaction-finetuned | 0.998 | 0.997 | upper bound |

**Registered behavioral-advantage test** (gene-grouped bootstrap, 503 genes, n=4000,
additive-frozen − additive-random-frozen):
- margin difference: mean 3.36, worst-seed 95% CI [2.84, …] → excludes 0, all 5 seeds.
- PMR difference: mean 0.751, worst-seed 95% CI low 0.720 → excludes 0, all 5 seeds.

## Causal localization

- TBD — tracing stage (not started; awaiting human GO).

## Mediation

- TBD — tracing stage.

## Replication

- successful seeds: 5/5 (behavioral).
- consistency: additive-frozen conflict PMR σ ≈ 0.004; control σ ≈ 0.006; fully non-overlapping.

## External validation

- TBD — after a validated circuit.

## Final claim label

**"Frozen behavioral advantage established; pretraining-derived, linearly-accessible
partner information."** Unmet higher-level criteria: component necessity, mediation,
specificity, external increment, and confirmatory test — all deferred to the tracing
stage and one-time test unblinding.
