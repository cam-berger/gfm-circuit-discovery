# SpliceBERT discovery methodology

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

This document records the study-specific implementation of the repository's transferable mechanistic-discovery method. It consolidates the former method note, methods audit, experiment designs, tracing plan, gate report, and migration notes.

## 1. Start from a model-native behavior

The donor–acceptor study does not begin with a head or visualization. It begins with a computation the model can express natively:

1. mask the G of a canonical donor `GT`;
2. score `logit(G) − max(logit(A), logit(C), logit(T))`;
3. corrupt the paired acceptor `AG → CC` using real bases rather than `[MASK]`;
4. define coupling as `margin(clean) − margin(corrupt)`.

The behavioral population is unconditional. Conditioning on a minimum effect is allowed only when a recovery fraction needs a stable denominator, and every such result is labeled conditional.

Controls must target plausible shortcuts:

- motif-matched non-partner sites test site detection;
- distance-matched other donors test distance decay;
- both-strand eligibility tests orientation logic;
- masked-probe MLM health tests whether an intervention preserves general competence.

## 2. Qualify the instrument before the target result

Every estimator that can create a scientific result must run a preflight in the same process before target data are interpreted. The preflight writes its values to the artifact and exits on failure.

### Required gates

1. **Positive control.** Recover a known signal with an expected size, wavelength, phase, or effect.
2. **Negative control.** Destroy or omit the signal and report the false-positive behavior.
3. **Control validity.** Measure the property that makes each control valid. “Energy matched” and “low leak” are measurements; “inert” is a conclusion that must be demonstrated.
4. **Bounds and invariants.** State mathematical/physical bounds before reading the result. Attention rows sum to one; a bounded coherence statistic stays in [0,1]; protected parameters and bands remain unchanged.
5. **Resolution and power.** Compute whether the design can separate the competing hypotheses and detect the expected effect before spending compute.
6. **Deletion audit.** Verify the targeted component is removed, protected components are retained, and the original checkpoint was not mutated.

### Bias-control rules

An anomalous control triggers an **interpretation freeze**. Diagnose the control and instrument before offering a biological story. An anomaly does not prove the instrument is broken; it does prohibit promotion of the target claim while unresolved.

Scrutiny is symmetric. A result that agrees with the hypothesis receives the same checks as a null. For each positive, name at least one artifact that could have produced it and test that possibility.

## 3. Use observational tools to generate candidates

Candidate localization may use:

- exhaustive head scans;
- attention and QK-offset profiles;
- position-table spectra and low-rank geometry;
- probes, attribution, cosine changes, or representation differences;
- biological sequence spectra and landmark-anchored matched filters.

These tools rank hypotheses. They do not establish causal relevance. The study repeatedly failed when an observational quantity was promoted directly: cosine change predicted the wrong L2H15 channel, phase purity measured sharpness, and QK oscillations did not guarantee post-softmax attention.

## 4. Progress through distinct evidence levels

For a positional candidate, separate:

`position/representation → QK logits → post-softmax attention → OV/path mediation → model behavior → biological interpretation`.

Each arrow requires a new measurement:

- **P → QK:** show a candidate head's logit structure depends specifically on the feature.
- **QK → A:** measure attention with a qualified statistic, per-head nulls, eligibility mass, deletion, reciprocal-band, and content-shuffle controls.
- **A → OV:** show the head output carries relevant information or mediates a targeted path.
- **OV → behavior:** change a native outcome while preserving model health and compare with matched interventions.
- **behavior → biology:** independently measure the proposed source/function and test a specific bridge, not a numerical coincidence.

Evidence never auto-promotes between levels.

## 5. Test necessity with matched nulls

Mean ablation is interpreted only alongside:

- protected MLM health measured on masked probes;
- distinct random head sets of the same size;
- exact enumeration when the null universe is small;
- exact add-one p-values `(hits + 1)/(n + 1)`;
- family correction across the complete scanned set.

Do not pool nulls across intervention sizes. In the period-3 mediation result, the one-head null licenses the L2H15/L3H1 comparisons; six-head nulls can be much larger and must remain separate. A 16-head intervention with no size-matched null is descriptive only.

Completeness requires a test with preserved model competence. If deleting the complement destroys the language model, a loss of target behavior cannot distinguish circuit specificity from collateral damage.

## 6. Establish mediation and direction

The core causal tools are:

- clean-to-corrupt activation patching;
- targeted mean ablation;
- key/value decomposition;
- whole-head and path rescue;
- position-localized output patching;
- carrier deletion and restoration;
- matched random restoration.

Interventions are checked algebraically before execution. A key/value swap that merely permutes an attention sum cannot answer a scientific question. A relocation whose result is forced by absolute position already encoded in vectors cannot demonstrate content addressing.

For conjunctive mediation, compare key, value, and joint patches and report the interaction term. For direction, restore the downstream component while keeping the upstream component ablated. Patch the full hypothesized mediator before declaring an edge open; a narrow token window can miss a distributed route.

## 7. Analyze position-table carriers

Carrier analysis follows a fixed sequence:

1. inventory position-table spectra without naming a biological source;
2. estimate sub-bin wavelength and rank/plane geometry;
3. identify QK readers with deletion and reciprocal-band controls;
4. validate post-softmax attention with positive and negative controls;
5. require adequate band mass before testing a head;
6. test OV/behavior only after Level A passes;
7. investigate biological origin last.

Content shuffle is not a negative control for a positional QK/attention signal: a positional comb should survive shuffling. Phase or carrier deletion is the relevant destructive control.

For non-integer wavelengths, do not assume an integer-width boxcar annihilates the tone. Measure leakage at the target wavelength. Use statistics with known bounds and per-head permutation nulls when residual structure differs across heads.

## 8. Analyze biological periodicity

An unanchored power spectrum is a quadratic detector and can be blind to weak signals. When a landmark exists, use a complex matched filter anchored to that landmark so coherent signal accumulates linearly.

The dyad assay therefore:

- estimates phase on one chromosome set;
- freezes phase before looking at held-out chromosomes;
- reports the signed held-out projection rather than a positively biased magnitude;
- keeps WW, SS, RR, and YY channels separate;
- uses offset controls to test confinement to the nucleosome core;
- calibrates jitter, false-positive rate, and effect-depth sensitivity.

Window length sets wavelength resolution. A ~51-nt junction signal cannot distinguish 10.298 from 10.630 even with more examples because the two filters are nearly collinear over that support. More samples improve precision, not resolution.

## 9. Seal confirmation and preserve provenance

Discovery, validation, and sealed test chromosomes are separate. Before unblinding:

- freeze the circuit, behavior, interventions, controls, thresholds, and exclusions;
- record hashes of controlling code;
- prove every gate can fail;
- run the complete protocol on validation data;
- open the sealed split once with an explicit command.

The frozen [`preregistrations/PREREGISTRATION_SEALED.md`](preregistrations/PREREGISTRATION_SEALED.md) is not revised after unblinding. Corrections live in the claim and decision ledgers.

Every result-producing script writes immutable machine-readable output and a manifest containing checkpoint, data, parameters, seed, command, and code revision. A superseded artifact is removed from the active index, not silently rewritten.

## 10. Stopping rules

Every branch has an explicit stopping point.

- A failed preflight stops target measurement.
- A failed evidence level stops promotion to the next level.
- A powered null closes the named hypothesis; it does not invite untargeted rescue experiments.
- An underpowered or unresolved test is recorded as inconclusive, not converted into evidence.
- A mathematical impossibility kills an experiment before compute.

Applied here, the human-only ≈10.2 proposal stopped when no corrected Level-A head survived. The multispecies ≈10.63 program may proceed to OV and behavior because eight heads pass Level A. Biological-origin work remains downstream of that computational-use test.

## 11. Transfer to another GFM

The transferable unit is the workflow, not SpliceBERT's components:

1. define a native biological computation;
2. qualify the assay at the expected effect scale;
3. localize candidates without promoting them;
4. establish necessity against matched nulls;
5. establish mediation and direction;
6. replicate on held-out data or checkpoints;
7. stop at the strongest supported level;
8. record corrections without keeping obsolete experiments active.

A second GFM need not contain L3H1, a period-3 clock, or an ≈10.63 carrier. It should be subjected to the same discipline.
