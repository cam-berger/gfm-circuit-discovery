# Preregistration: native GTEx classifier and representation bridge

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Status: **EXECUTED; PREFLIGHT FAILED; DONOR COORDINATE RETRACTED.**

Frozen on 2026-07-29 after the fixed-MLM-readout result in GFS4, but before loading any published
`classifier.*` tensor into a model forward pass. Checkpoint structure, parameter displacement, and the
authors' training/evaluation code were already known. No native-head probability, margin, corruption
effect, hidden-state projection, or cross-encoder result had been inspected.

## 1. Question and evidence boundary

The fixed MLM readout no longer exposes the base model's donor–acceptor coupling after GTEx fine-tuning,
but its general masked-nucleotide health also degrades. This arm distinguishes two possibilities:

1. coupling was re-encoded into a representation readable by the trained splice-site classifier; or
2. the native classifier recognizes splice sites without a detectable dependence on the paired acceptor
   under the established assay.

The primary readout is model-native for the fine-tuning task:

`donor margin = logit(class 1: donor) - logit(class 0: no splice site)`

at the annotated first intronic base of the donor, on an unmasked 1,024-nt sequence. The authors'
documented labels fix class `0=no-site`, `1=donor`, and `2=acceptor`. The corresponding acceptor
manipulation check uses `logit(class 2)-logit(class 0)` at the annotated first exonic base.

There is no native base-model classifier, so this arm does **not** estimate a base-versus-fine-tuned
change in native task performance. It asks what the released fine-tuned model itself uses.

## 2. Checkpoints, cohort, and sequence interventions

- All five published seeds, 2020–2024, are mandatory.
- Raw state dictionaries must load strictly into standard three-class `BertForTokenClassification`.
- The tokenizer and encoder configuration come from the hash-frozen `SpliceBERT.1024nt` checkpoint.
- The annotation-only `sealed-unc-val` cohort is reused at `n=400`; no example is selected on a model
  outcome, and chr8/9 remain sealed.
- The primary intervention replaces the paired acceptor `AG` with `CC`.
- `AG→TT` is a frozen substitution-sensitivity replication.
- Specificity controls are the unchanged distance-matched other-donor readout and the nearest eligible
  non-partner `AG→CC` intervention used in GFS4.
- The paired acceptor's own acceptor-margin drop is the manipulation check.

All interventions replace two real bases. No `[MASK]` intervention enters the classifier assay.

## 3. Native-classifier preflight

The preflight uses the first 64 annotation-only validation examples and may inspect only local
splice-site competence and a neutral distant mutation. It may not destroy the paired acceptor.
Every published seed must pass:

1. exact checkpoint keys, shapes, hashes, finiteness, and a finite forward pass;
2. one token per nucleotide, one prefix token, and exact donor/acceptor coordinate recovery;
3. final hidden-state projection reconstructs the model's native logits with maximum absolute error
   `≤1e-5`;
4. at least 70% of clean donors have donor-vs-no-site margin `>0`;
5. at least 70% of clean paired acceptors have acceptor-vs-no-site margin `>0`;
6. destroying the local donor with `GT→CC` lowers the donor margin by mean `≥1.0` nat and in at least
   75% of examples;
7. the independent `GT→AA` local destruction also lowers the margin by mean `≥1.0` nat and in at
   least 75% of examples;
8. the absolute mean effect of a deterministic distance-matched non-motif two-base mutation is at most
   25% of the smaller local-donor destruction effect.

Failure freezes the target run and returns the assay to instrument diagnosis.

## 4. Target estimands and inference

Per seed, the four scanned estimands are:

- paired-acceptor `AG→CC` donor-margin drop;
- its difference from the effect at the distance-matched other donor;
- its difference from the non-partner-AG effect at the true donor;
- paired-acceptor `AG→TT` donor-margin drop.

Uncertainty is gene-grouped percentile bootstrap. Each zero-effect test uses 9,999 gene-level sign
flips with the exact add-one correction. BH correction spans the 20 seed-by-estimand tests; a separate
BH family spans the four five-seed-mean estimands. Per-example values are retained.

The native coupling gate passes only if:

- the five-seed-mean `AG→CC` effect has a 95% interval wholly above zero and at least four of five
  seed estimates are positive;
- the `AG→TT` family replication also has a 95% interval wholly above zero;
- both family specificity contrasts have 95% intervals wholly above zero; and
- the paired acceptor's own acceptor-margin manipulation check has a 95% interval wholly above zero.

A gate failure supports only the bounded statement that the released classifier does not expose
partner dependence under this assay. It does not prove the encoder contains no partner information.

## 5. Gated representation bridge

Representation analysis may run only after the native coupling gate passes. For every seed and every
depth (embedding output plus six encoder layers), it will:

- apply the seed's frozen classifier vector to donor states from the fine-tuned encoder;
- apply the same classifier vector to donor states from the base encoder;
- report clean donor margin and paired-acceptor coupling under that common linear readout; and
- report linear CKA between base and fine-tuned clean donor-state geometries.

The final fine-tuned layer must exactly reproduce the native behavior artifact, and the base-encoder
endpoint is a cross-model compatibility diagnostic, not a native base behavior. Intermediate-layer
projections are candidate generators because the classifier was trained only on the final layer.
CKA is descriptive representation geometry. Neither establishes necessity, mediation, or a circuit.

Any later layer/head intervention requires a new frozen protocol with matched-size controls.

## Execution note

The immutable v1 preflight failed before the paired-acceptor target was opened. Across all five seeds,
acceptor recognition passed, exact projection passed, but donor recognition was zero at the nominated
coordinate. Diagnosis against the already published data-loader check showed that the authors place the
donor label at the last exonic base and verify `GT` at `label+1:label+3`; this v1 document instead
nominated the first intronic `G`. No threshold or result was changed. The corrected assay is a separately
frozen protocol in
[`PREREGISTRATION_NATIVE_CLASSIFIER_V2.md`](PREREGISTRATION_NATIVE_CLASSIFIER_V2.md).
