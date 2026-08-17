# Evidence ladder

> **Tier: T1 — canonical.** Current, authoritative. Read this.

The ladder prevents a familiar failure mode: a striking observation gradually acquires causal or biological language without the experiment needed for that promotion.

## General ladder

| Level | Question | Typical evidence | Licensed statement |
|---|---|---|---|
| Behavior | Does the model perform the computation? | Native-output counterfactuals and baselines | The model exhibits a scoped behavior. |
| Representation | Is relevant information encoded? | Probes, geometry, spectra, position or residual features | Information is decodable or structurally present. |
| Selection / QK | Does a component read or select it before normalization? | Logit decomposition, QK bilinear analysis | A component scores the candidate relation. |
| Routing / attention | Does the signal survive competition? | Post-softmax attention and controlled deletion | The component routes attention using the signal. |
| Output / OV | What is written downstream? | OV analysis, output projection, downstream feature change | The component writes relevant information. |
| Mediation | Does the component carry the causal effect? | Patching, rescue, transplant, dose response | The component mediates a measured fraction of the effect. |
| Directed circuit | Do connected components implement the computation? | Edge/path interventions plus necessity and mediation | A scoped directed causal pathway implements the behavior. |
| Biological interpretation | Why does this mechanism exist and what does it mean biologically? | Training-data source tests, external assays, cross-model/species evidence | A biological origin or function is supported. |

A study may enter the ladder at different observational levels, but it may not skip the causal bridge when making a causal claim.

## SpliceBERT implementation of the ladder

For positional mechanisms, the project records five compact columns:

- **P** — present in the learned position table.
- **QK** — read in pre-softmax attention logits.
- **Attention** — survives softmax and controlled competition.
- **OV / mediation** — written downstream or causally mediates the effect.
- **Behavior** — changes a model-native behavior under intervention.

| Component | P | QK | Attention | OV / mediation | Behavior | Status |
|---|:--:|:--:|:--:|:--:|:--:|---|
| Period-3 clock | ✓ | ✓ | ✓ | ✓ | ✓ | Causal support for donor–acceptor routing |
| Donor–acceptor computation | n/a | ✓ | ✓ | ✓ | ✓ | Directed information-routing circuit |
| Reverse masked-acceptor dependence (`GT-AG`) | n/a | ? | ? | ? | bounded gate | The reciprocal behavior is real at the validation-gate level. Component interventions produced bounded, inconclusive, and health-failed candidates, but the prospectively frozen reader→write→delivery chain stopped before target (AC1–AC9). No causal rung above behavior is established. |
| Natural donor-class modifier (`GC-AG` vs matched `GT-AG`) | n/a | ? | bounded | ? | bounded | Held-out behavior and L3H1 attention-geometry verification only (AC10–AC11). No `W_O` write, mediation, delivery, or biological mechanism. |
| Multi-species ≈10.63 carrier | ✓ | ✓ | ✓ | null (donor coupling) | bounded (−0.033) | Real positional attention circuit, **not** part of the donor circuit. Both interventions ran (C37→C12). General function, general OV content, and biological meaning remain open; the `U_info` write test was never computed (R31). |
| Human-only ≈10.2 candidate | ✓ | provisional | failed | — | null | Proposed chromatin circuit closed at attention |

The authoritative values and statistical conventions remain in [`figures/splicebert/EVIDENCE.md`](../figures/splicebert/EVIDENCE.md).

## The five columns in causal-abstraction terms

This mapping is additive: it names the five columns in the vocabulary of causal abstraction
(Geiger et al., 2025; see [`docs/REFERENCES.md`](REFERENCES.md) and the formal-grounding section of
[`docs/METHODOLOGY.md`](METHODOLOGY.md)). It restates what each column already reports and changes no entry,
status, or value in the tables above.

Causal abstraction treats a mechanistic hypothesis as a proposed alignment between variables of a high-level
causal model and variables of the network, tested by intervening on the network and checking the result
against the corresponding high-level intervention. Read that way:

| Column | Causal-abstraction reading | Intervened on | Outcome measured on |
|---|---|---|---|
| **P** | A candidate low-level variable exists and carries the feature. No alignment is asserted and nothing is intervened on. | — | — |
| **QK** | A candidate alignment: a component's pre-softmax score is a function of the feature. Still a decomposition of the mechanism, not an intervention. | — | — |
| **Attention** | First interventional level, but internal: the network is intervened on and the result is read out on an internal variable (the routing pattern), not on the high-level behavior variable. | low-level | low-level |
| **OV / mediation** | Interchange / injection intervention: aligned variables are set to source or counterfactual values and the result is read out on the behavior variable. This is the level at which a graded faithfulness number can be computed at all. | low-level | high-level |
| **Behavior** | The high-level variable the abstraction is required to reproduce, measured on native model output. | high-level counterfactual | high-level |

The marker semantics, unchanged: a **✓** records that the intervention defining that level was run and the
component survived it; a **?** records that the defining intervention has **not** been run, so there is no
evidence in either direction; a **—** records that the level is not applicable or was not entered; **failed**
or **null** records that the intervention *was* run and the hypothesis was rejected at that level; a
**bounded** records that the intervention *was* run and returned a measured **boundary** rather than a
verdict — a sub-SESOI point estimate, or an equivalence test whose half-width exceeds its margin. A null or
a bound is a **result, not an absence**; neither is a ✓.

The distinction between **?**, a bound, and a rejection is load-bearing. The human-only ≈10.2 candidate is a
proposed chromatin circuit **closed at attention** — its defining intervention ran and rejected it. The
multi-species ≈10.63 carrier no longer illustrates **?**: both its OV and Behaviour interventions ran on
2026-07-23 (C37, folded into C12), returning a donor-scoped value-write **non-detection** and a small
*suppressive* sub-SESOI coupling effect. What remains unknown about that carrier is its *general* function
and biological meaning — and, specifically, whether it writes onto the donor readout `U_info`, which was
never computed (**R31**) — not whether the experiments exist. These are three different epistemic states and
the table must never be read as collapsing them.

The reverse masked-acceptor rows are deliberately split. AC10's model-output ratio is behavior evidence;
AC11's donor-attention ratio is routing/attention evidence. Observing both on the same held-out cohort does
not fill the missing output/OV or mediation rungs. M1–M5 remain adaptive candidate generation. M4d's
pre-softmax score-margin transform refines three layer-0 candidates but, because its intervals are
post-selection, unadjusted, and clustering-limited, does not promote the modifier's QK rung or add a
check to either AC10/AC11 row (`ACR23`).

**Relation to graded faithfulness.** The columns are an ordered record of *which* interventions have been run
and survived. They are not a faithfulness measurement, and a row of ✓ across all five columns does **not**
assert that the abstraction is faithful. Faithfulness in the causal-abstraction sense is graded — the degree
to which the high-level model's predictions match the network's across an intervention distribution — and it
is reported separately, as a number with a stated node granularity and intervention set, in the claim ledger
rather than in this table. For the donor–acceptor computation the ladder is complete across all five columns
*and* the recorded faithfulness is low for the sparse circuit (C27) with no sparse knee on the
faithfulness–size curve (C33). Both statements are true simultaneously, and treating a full row of ✓ as
implying a sparse faithful circuit is precisely the inference the ladder exists to block.

## Promotion requirements

- **Representation → use:** show that a model component reads or routes the feature under an appropriate control.
- **Use → necessity:** remove the feature or component and measure the declared behavior.
- **Necessity → mediation:** restore or patch the candidate and recover the effect relative to matched nulls.
- **Mediation → directed circuit:** isolate the path and rule out simpler or redundant alternatives.
- **Model circuit → biological interpretation:** connect the mechanism to training-data provenance or biological function with an independently capable assay.

## Required controls

1. Positive and negative assay controls.
2. Injection/recovery or another sensitivity calibration.
3. Resolution and detection-floor statement.
4. Matched-size random or structured null interventions.
5. Specificity controls that protect reciprocal features, content, position, or unrelated behavior.
6. Checkpoint, coordinate, tokenization, metric, and sample-count invariants.
7. Multiplicity correction across the actual scanned family.
8. A frozen stopping rule for anomalous controls.

If a control fails, the status is **instrument unresolved**, not positive or negative.
