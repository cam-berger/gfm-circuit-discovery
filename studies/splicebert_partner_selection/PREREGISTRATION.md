# Preregistration Draft

> ⛔ **SUPERSEDED / VOID.** This document belongs to the **retracted** v1/v2 partner-selection
> design (R2/R3) or to the pre-sealed-test protocol. It is kept for provenance only.
> **The document of record is `studies/splicebert_partner_selection/FINDINGS.md`;**
> the frozen protocol actually executed is `PREREGISTRATION_SEALED.md`.
> Retractions: `CLAIMS.md` §R1–R12.

**Status:** FROZEN 2026-07-11 at the behavioral gate (GO). This file is hashed in `FREEZE.md`; do not edit before test unblinding. Exploratory/tracing choices are logged in `DECISIONS.md`, not here.

## Primary hypothesis

A constrained additive readout on frozen pretrained SpliceBERT representations will outperform the registered local PWM+length baseline and a matched randomized-backbone readout on chromosome-held-out natural cue conflicts.

## Primary mechanistic hypothesis

A compact set of pretrained-backbone components will be specifically necessary for conflict behavior and clean activations from that set will mediate at least part of an aligned clean–corrupt behavior gap.

## Registered thresholds

These defaults must be reviewed against validation variance and then frozen:

- behavioral advantage: grouped 95% CI for margin difference above zero;
- conflict positive-margin rate: at least 0.60;
- replication: at least 3 of 5 seeds pass in the same direction;
- component necessity: selected-set effect exceeds the 95th percentile of matched random sets;
- mediation: selected-set recovery fraction at least 0.30 on a registered partner-relevant corruption;
- specificity: conflict effect exceeds matched ordinary-example effect;
- test set: opened once after analysis and components are frozen.

## Confirmatory versus exploratory

Confirmatory:

- primary chromosome split;
- local baseline;
- additive-frozen and additive-random-frozen comparison;
- natural conflict endpoint;
- registered component set and patch assay;
- external frozen circuit metric.

Exploratory:

- bilinear readouts;
- alternative sequence flanks;
- alternative candidate limits;
- chimeric factorial subgroups;
- unregistered heads/features discovered after test unblinding.

## Failure interpretations

- no frozen behavioral advantage: no evidence for pretrained partner-aware computation under this assay;
- fine-tuned only: adaptation-installed circuit;
- necessity without mediation: involved components, incomplete causal path;
- mediation without external increment: valid model mechanism without biological discovery evidence.
