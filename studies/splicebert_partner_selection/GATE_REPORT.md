# Behavioral gate report — SpliceBERT partner selection

**Date:** 2026-07-11 · **Stage:** behavioral gate · **Recommended call: GO** (tracing awaits explicit human authorization)

## Question

Does pretrained SpliceBERT already encode which partner each splice site belongs
to, in a form readable from its frozen representations, and is that attributable
to pretraining rather than to the trained readout?

## Design (one line)

On natural cue conflicts — cases where the local PWM+length baseline picks the
wrong pair — score a constrained **additive** readout on the **frozen** backbone
against (a) the local baseline and (b) a matched **randomized-backbone** control,
across 5 seeds. The additive readout cannot multiply the two sites, so any
partner information it uses must already be in the contextual endpoint states.

## Results

| Regime | conflict PMR (mean, 5 seeds) | val top1 | Role |
|---|---:|---:|---|
| local baseline (PWM+length) | ~0 on conflict set (0.412 overall) | 0.412 | floor |
| **additive-frozen** (PRIMARY) | **0.807** [0.802–0.812] | 0.87 | evidence |
| additive-random-frozen (CONTROL) | **0.056** [0.047–0.062] | 0.07 | collapses below floor |
| interaction-frozen | 0.787 | 0.86 | additive term buys nothing |
| additive-finetuned | 0.997 | 0.999 | upper bound |
| interaction-finetuned | 0.997 | 0.998 | upper bound |

**Registered gate — gene-grouped bootstrap** (503 genes as the resampling unit,
n=4000, additive-frozen minus additive-random-frozen):

- true-pair **margin difference**: mean **3.36**, worst-seed 95% CI low **2.84** → CI excludes 0 on **all 5 seeds**.
- positive-margin-rate difference: mean **0.751**, worst-seed CI low **0.720** → excludes 0 on all 5 seeds.

## Gate checklist (all PASS)

- conflict positive-margin rate ≥ 0.60 → 0.807 ✅
- behavioral advantage grouped CI > 0 → margin-diff CI low ≥ 2.84 ✅
- replication ≥ 3/5 seeds → 5/5 ✅
- beats local baseline ✅ and beats randomized-backbone control (0.807 vs 0.056, non-overlapping) ✅

## Interpretation and its ceiling

Established: **pretraining built machinery inside SpliceBERT that computes
partner-aware representations**, and that information is linearly/additively
accessible at the site endpoints (the interaction readout adds nothing over
additive). Because the partner cannot be inferred locally, attention must
transport cross-site information during the forward pass — i.e. the computation
is in the backbone, not the probe.

Not yet established: that this machinery is a **localized, sparse circuit the
model itself uses** for its native objective, versus diffuse information merely
*decodable* by a trained probe. That distinction is exactly what the tracing
stage tests.

## Deviations from the shipped scaffold

1. Fixed the randomized-backbone control (`init_weights()` is a no-op on
   transformers ≥5; now built via `from_config`). This is what turned a spurious
   NO-GO into the true GO.
2. Matrix trained on Colab GPU with scale caps (train 12k / val 4k / eval 4k);
   gates, controls, and metric unchanged.
3. Gate bootstrap re-run locally; reproduces Colab conflict PMRs exactly.

Full schema in `return_to_planner.json`; hashes in `FREEZE.md`.

## Recommended next step

Declare GO on the mechanistic stage. Then freeze the component set and run the
registered intervention sequence on the **additive-frozen** checkpoints, keeping
the test split sealed until components are frozen.
