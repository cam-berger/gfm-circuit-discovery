> # ⛔ VOID — THE RESULT IN THIS FILE HAS BEEN RETRACTED
> **Retracted 2026-07-12 (R2, R3).** This document was written before the retraction and presents the
> partner-selection result (PMR 0.807 vs a 0.056 control, "Recommended call: GO") as established.
> **It is not.** The task carried a positional leak — the true acceptor sat at a constant index in
> 100.0% of examples, so a geometry oracle reading *zero nucleotides* scores a perfect 1.000 — and,
> independently, an oracle site detector also scores 1.000, so the task never required partner
> information at all. The v2 fix was biologically unfixable.
>
> **Do not quote any number in this file.** It is kept because a retraction is only credible if the
> thing it retracts is still on the shelf.
>
> **Current record:** `CLAIMS.md` (claims registry, with permanent retraction rows R1/R2/R3) ·
> `DECISIONS.md` · `../../docs/STATUS.md` §3.
> **The replacement study — cross-site coupling — is confirmed on held-out data.** See `CLAIMS.md`
> rows D1–D5 and `results/splicebert_coupling/confirmation.json`.

---

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
