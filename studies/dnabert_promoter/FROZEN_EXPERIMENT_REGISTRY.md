# Frozen DNABERT Mechanistic Experiment Registry

## Canonical assay

- Model: `quietflamingo/dnabert2-no-flashattention`
- Revision: `813031b2bf86d9e960a027e2734c908009f31601`
- Condition: frozen
- Readout endpoint: encoder Layer 2 output
- Pooling: masked mean over biological tokens
- Readout: fixed linear probe
- Primary outcome: promoter margin
- Test balanced accuracy: 0.8102122021467204
- Test ROC-AUC: 0.8919269597801556
- Test PR-AUC: 0.8918734234801855

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

Status: not run as a controlled assay.

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

Status: not run against the corrected endpoint.

## Existing restoration verdict

- L0H8: strong necessity, weak standalone recovery.
- Layer-0 MLP: large recovery, but not donor-specific.
- Combined MLP and heads: large recovery, but not donor-specific.
- Cross-layer head set: modest recovery, not donor-specific.

No sequence-specific delivery claim is currently licensed.

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

The Layer-0 MLP carries a shared promoter-class representation.
L0H8 contributes sequence-sensitive information and improves restoration
when combined with the Layer-0 MLP, but is not sufficient alone.

The current leading compact component set is:

    Layer-0 MLP + L0H8

This is a validation result and must be confirmed on an untouched cohort.
