# Frozen DNABERT Mechanistic Experiment Registry

> The arc's own working registry of assay definitions, metric formulas, and
> per-block status. **Not a claim record.** The claims are in [`CLAIMS.md`](CLAIMS.md) and what was
> withdrawn from this file is in [`CORRECTIONS.md`](CORRECTIONS.md).
>
> Interpretive sentences in this file were corrected on 2026-08-11 against the committed artifacts
> (`DBPR2`–`DBPR5`, `DBPR8`). The **numbers were verified and left exactly as they were** — every metric
> and mean difference here reconciles to its artifact. What changed is what they are said to mean.

## Fine-tuned extension pointer

This file remains the **frozen** experiment registry. The later fine-tuned checkpoint analysis is not
folded into these assay definitions because it studies a distinct 300-nt `prom_300_all` behaviour and
readout. Its claim-grade record is in `CLAIMS.md` (DBP11–DBP17), its design decisions are D14–D22 in
`DECISIONS.md`, and the concise descriptive comparison is
[`FINETUNED_VS_FROZEN.md`](FINETUNED_VS_FROZEN.md).

The old fine-tuning smoke metrics withdrawn under DBPR6 remain withdrawn. They are not replaced by editing
this frozen registry; the later fine-tuned package is a separate evidentiary record.

## Canonical assay

- Model: `quietflamingo/dnabert2-no-flashattention`
- Revision: `813031b2bf86d9e960a027e2734c908009f31601`
- Condition: frozen
- Readout endpoint: encoder Layer 2 output — i.e. `hidden_states[2]`, since this adapter returns the
  12 encoder-layer outputs with no embedding output (see `DECISIONS.md` D5)
- Pooling: masked mean over biological tokens
- Readout: fixed linear probe
- Primary outcome: promoter margin
- Test balanced accuracy: 0.8102122021467204
- Test ROC-AUC: 0.8919269597801556
- Test PR-AUC: 0.8918734234801855

Two notes travel with those three metrics everywhere:

- **They are measured on leakage-controlled splits.** The raw cohort has 104 cross-split duplicate groups
  and 64 label-conflicting groups; `clean_promoter_cohort.py` removes them (326 rows) and asserts the
  retained splits are disjoint. Exact and reverse-complement identity are screened; near-duplicates are
  not (`DBP2`).
- **Layer 2 is the strongest *tested* endpoint by `0.0019` balanced accuracy on one seed** — layers 1–4
  score `0.8050 / 0.8031 / 0.8014 / 0.8014` on validation and are not separated by the data. It is a
  working choice, not a localization result (`DBP4`, `DBPR2`).

## Core metrics

For a clean/corrupt pair:

    damage = clean_margin - corrupt_margin

For clean-to-corrupt restoration:

    raw_recovery = patched_margin - corrupt_margin

    recovery_fraction =
        raw_recovery / (clean_margin - corrupt_margin)

Always retain clean margin, corrupt margin, patched margin, raw recovery,
and recovery fraction.

## Data policy

- Discovery data: rank components and derive hypotheses.
- Validation data: choose groups, ranks, and thresholds.
- Confirmation data: report final effects.
- Do not discover and confirm on the same examples.

## Experiment blocks

### 1. Global promoter behavior

Compare:

- clean sequence;
- mononucleotide shuffle;
- GC-preserving corruption;
- edit-count-matched local corruption;
- optional dinucleotide-preserving shuffle.

Status: partial.

### 2. Global branch localization

Test necessity and restoration for:

- Layer 0 attention;
- Layer 0 MLP;
- Layer 1 attention;
- Layer 1 MLP;
- Layer 2 attention;
- Layer 2 MLP.

Status: branch ablation complete; restoration requires specificity repair.

### 3. Token-span localization

Patch only nucleotide-span-aligned clean/corrupt regions.

Controls:

- changed span;
- upstream span;
- downstream span;
- random matched span;
- all biological tokens.

Status: blocked by tokenizer alignment.

### 4. Component census

Primary head candidate:

- L0H8

Secondary candidates:

- L0H3
- L0H10
- L1H8
- L2H9
- L2H3

Required evidence:

- necessity;
- matched restoration;
- mismatched-donor control;
- random-head control;
- held-out confirmation.

Status: ablation complete; donor-specific restoration not established.

### 5. Biological-feature specificity

Initial target: TATA-like promoter motifs.

Perturbations:

- motif destruction;
- GC-matched motif replacement;
- edit-count-matched non-motif edit;
- motif relocation;
- motif-negative controls.

Status: **partly run.** Motif destruction with an edit-count-matched non-motif control *was* run as a
controlled assay on a validation cohort of 71 and returned a null — see the "Biological feature test"
section below, and `DBP10`. GC-matched replacement, motif relocation, and motif-negative controls were
not run. (This line previously read "not run as a controlled assay", which contradicted the results in
this same file; corrected 2026-08-11.)

### 6. Low-rank mediation

Candidate sites:

- Layer-0 MLP;
- L0H8;
- Layer-1 residual stream;
- Layer-2 representation.

Ranks:

- 1, 2, 4, 8, 16, 32.

Status: not run against the corrected endpoint.

### 7. Pathway tracing

Initial hypothesis:

    sequence feature
    -> Layer-0 MLP and/or L0H8
    -> early attention pathway
    -> Layer-2 representation
    -> linear probe

Status: hypothesis only.

### 8. Circuit structure

Measure:

- necessity;
- delivery;
- standalone faithfulness;
- faithfulness versus circuit size;
- random-circuit controls.

Status: **mostly run against the corrected endpoint, and it produced the arc's most important negative
result.** Necessity, delivery, standalone faithfulness, and faithfulness-versus-size were all run at
`selected_layer: 2` on validation and an untouched test cohort — see `frozen_claim_tests/circuit_accounting/`
and `frozen_claim_tests/faithfulness_curve/`, and `DBP6`/`DBP7`. **Random-circuit controls were not run.**
The curve has no knee: the five-node candidate set reaches standalone faithfulness `0.194` and only the
addition of `MLP1` (size 16 → `0.487`) and `MLP2` (size 39 → `1.000`) moves it, so compactness is refuted
at the sizes tested. (This line previously read "not run against the corrected endpoint", which
contradicted the artifacts shipped alongside it; corrected 2026-08-11.)

## Existing restoration verdict

- L0H8: strong necessity, weak standalone recovery.
- Layer-0 MLP: large recovery, but not source-specific.
- Combined MLP and heads: large recovery, but not source-specific.
- Cross-layer head set: modest recovery, not source-specific.

No sequence-specific delivery claim is currently licensed.

> **This verdict was right, and it survived review.** The test-confirmation section below originally
> contradicted it by calling the representation "transferable" and the set "compact"; those sentences were
> withdrawn (`DBPR4`, `DBPR5`), not this one. When two sections of a registry disagree, the one that
> reports the *control* is the one to keep.
>
> Two wording notes. "Not donor-specific" has been rewritten as "not **source**-specific" throughout this
> section: in this study "donor" means the source example whose activations are transplanted, but
> everywhere else in this repository "donor" means a splice donor site, and the collision is a real
> reading hazard (see `DECISIONS.md` D11 and `docs/NOMENCLATURE.md`). "Strong necessity" for L0H8 is also
> bounded: the ranking is correctly signed, but necessity needs the calibration pair and a multi-seed
> replicate that have not been run (`DBP5`, `DBPR3`).

## Execution order

1. Validate existing negative-donor controls.
2. Quantify matched-minus-mismatched restoration statistically.
3. Repair token/nucleotide-span alignment.
4. Build matched corruption controls.
5. Confirm global components on held-out data.
6. Build motif-specific cohort.
7. Test motif-specific component dependence.
8. Run low-rank mediation.
9. Trace the strongest pathway.
10. Measure necessity, delivery, and faithfulness.
11. Generate final figures and tables.

## Confirmed validation result: component donor matrix

Experiment:

- 128 corrupt promoter recipients
- four same-class donor offsets
- four opposite-class donor offsets
- four component conditions
- fixed frozen Layer-2 probe
- identity intervention passed exactly

### Recovery fractions

| Component | Matched | Same class | Opposite class | Zero |
|---|---:|---:|---:|---:|
| L0H8 | 0.005147 | -0.059574 | -0.134595 | 0.330688 |
| L0 MLP | 0.718197 | 0.682961 | -0.172610 | -0.198878 |
| L0 MLP + L0H8 | 0.848330 | 0.819128 | -0.150707 | 0.125146 |
| Full early group | 0.839248 | 0.823483 | -0.131332 | 0.072801 |

### Statistical conclusions

- L0H8 matched restoration exceeds same-class restoration, but absolute
  L0H8 recovery is negligible.
- For the Layer-0 MLP and both combined groups, matched restoration is not
  significantly greater than same-class restoration.
- Same-class restoration is substantially greater than opposite-class
  restoration for every component condition.
- The compact Layer-0 MLP + L0H8 set performs at least as well as the full
  early group.

### Interpretation

The Layer-0 MLP carries a shared promoter-**class** representation. Combining it with L0H8 raises
matched recovery, and L0H8 alone recovers essentially nothing (`0.005147`).

The current leading component set is:

    Layer-0 MLP + L0H8

This is a validation result and must be confirmed on an untouched cohort.

**Corrected 2026-08-11 (`DBPR4`, `DBPR5`).** This section previously called L0H8's contribution
"sequence-sensitive" and the set "compact". Neither is licensed:

- *Sequence-sensitive* would require matched restoration to beat same-class restoration. It does not —
  the gap is `0.848330 − 0.819128 = 0.029` here and stays null on test (`DBP8`). What the artifact
  supports is **class-level** transfer.
- *Compact* is refuted by this arc's own faithfulness curve (`DBP7`): the set's standalone faithfulness
  is `0.194`, and 1.0 is reached only at size 39. Say "a structured causal core within a distributed,
  MLP-dominated computation".

## Canonical probe artifact

- Path:
  `results/dnabert_promoter/frozen/layer_probes/selected_layer_2_probe.joblib`
- SHA-256:
  `bd0d9e14e1935e3ebd8e2ce638ccd5a721feb9dc9a0895447d9b62080ce98b3b`
- Git commit preserving probe:
  `f38a345`

The same probe artifact must be used for validation and test confirmation.

> **Verified 2026-08-11.** The committed file hashes to exactly that SHA-256; commit `f38a345`
> ("Preserve canonical frozen DNABERT Layer 2 probe") exists; and the artifact is byte-identical to
> `all_layer_probes/layer_2_probe.joblib`, so the same-artifact rule was genuinely enforced across
> validation and test. Note that the superseded Layer-3 family uses a **different** probe —
> `all_layer_probes/layer_3_probe.joblib`, sha256 `2337441effb785769dec…` — which is how to tell the two
> suites apart when reading a summary (`DBPR7`).

## Test confirmation: compact donor matrix

An untouched high-signal cohort was selected from the test split:

- 128 promoters
- 128 non-promoters
- mononucleotide corruption
- fixed Layer-2 linear probe
- component conditions fixed before the confirmation run (**not preregistered** — no protocol document
  was frozen for this arc; see `DBPR8`):
  - Layer-0 MLP
  - Layer-0 MLP + L0H8

### Test recovery fractions

| Component | Matched | Same class | Opposite class | Zero |
|---|---:|---:|---:|---:|
| L0 MLP | 0.738897 | 0.685727 | -0.175620 | -0.225149 |
| L0 MLP + L0H8 | 0.879318 | 0.842935 | -0.151602 | 0.121078 |

### Test statistical conclusions

Layer-0 MLP:

- matched versus same-class:
  mean difference = 0.053170
  raw p = 0.090690, Holm-adjusted p = 0.181379 — **not significant**
- same-class versus opposite-class:
  mean difference = 0.861347
  raw p underflowed to 0.0 in the artifact

Layer-0 MLP + L0H8:

- matched versus same-class:
  mean difference = 0.036383
  raw p = 0.251984, Holm-adjusted p = 0.251984 — **not significant**
- same-class versus opposite-class:
  mean difference = 0.994537
  raw p underflowed to 0.0 in the artifact

Identity interventions produced exactly zero change.

**Corrected 2026-08-11 (`DBPR5`).** Two reporting faults were fixed above. The class-contrast rows
previously carried "Holm-adjusted p = 2.8539e-22"; that value **appears in no artifact** — the committed
statistics record `p_value: 0.0` for both — so it has been replaced with what the record actually says.
And Holm was applied across only the two matched-versus-same-class tests, not the six comparisons the
artifact contains (`matched_vs_opposite_class` is also present for both components), so the adjustment
family is stated rather than implied.

### Confirmed interpretation

Matched promoter activations are **not** significantly better than unrelated same-class promoter
activations, on either component, whereas non-promoter activations push the readout in the opposite
direction. What transplants is therefore a **class label's worth of promoter evidence, not an
example-specific payload**.

The confirmed component set is:

    Layer-0 MLP + L0H8

**Corrected 2026-08-11 (`DBPR4`, `DBPR5`).** This section previously opened by calling the Layer-0 MLP
representation "transferable" and saying L0H8 "substantially improves delivery", then called the set
"compact" — while its own third sentence recorded the null that refutes the first two. The null is the
result. `transferable`, `delivery`, and `compact` are retired for this study; the licensed readings are
`DBP7` (no compactness) and `DBP8` (class-level transfer only).

## Biological feature test: early-window TATA-like motifs

### Enrichment

TATA-like motifs defined by `TATA[AT][AT]` were not promoter-enriched
across the full 70-nt sequence.

They were modestly enriched when restricted to starts at positions 0-7:

- validation odds ratio: 1.3646, p = 0.0272
- test odds ratio: 1.6233, p = 0.00104

This window is described as the dataset-aligned early window because the
exact genomic coordinate relative to the transcription start site has not
yet been verified.

### Controlled perturbation

Validation cohort:

- 71 promoter sequences
- exactly one early-window TATA-like motif
- two-base motif disruption
- matched non-motif control with identical substitutions
- both interventions changed T to G and A to C
- identity control passed exactly

Results:

- mean motif damage: -0.302244
- mean matched-control damage: -0.376305
- mean excess motif damage: 0.074060
- fraction motif disruption more damaging: 0.521
- one-sided paired p-value: 0.233402

Exploratory stratification by motif variant and motif position did not
identify a coherent significant subgroup. In particular, starts at
positions 2-4 had mean excess damage 0.004934 and p = 0.469945.

### Conclusion

Early-window TATA-like motifs are positionally enriched, but this
controlled intervention does not establish that the frozen Layer-2
promoter readout causally depends on them.

No TATA-specific mechanistic claim is licensed.
