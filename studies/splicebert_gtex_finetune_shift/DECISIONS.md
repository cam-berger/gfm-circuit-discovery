# Decisions: SpliceBERT GTEx fine-tuning shift

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

## 2026-07-28 — preserve the original MLM readout as a derived instrument

The published checkpoint has a three-class token-classification head and no MLM head. The primary question
names the original native MLM coupling metric, so the encoder is transplanted into the exact base masked LM.
The base MLM transform/bias are retained and the normal tied decoder follows the fine-tuned nucleotide
embeddings. This measures what remains readable through the original MLM family after encoder fine-tuning.
It is not described as an author-published fine-tuned MLM.

## 2026-07-28 — use all five seeds

The upstream analysis trained seeds 2020–2024 and reports aggregate benchmark performance. Seed 2020 may be
used for an outcome-limited mechanics smoke, but no scientific result may select or headline one seed.

## 2026-07-28 — validation only

The existing sealed chr8/9 test has already served its confirmatory purpose. This matched-checkpoint study
uses the validation cohort and treats the five checkpoint seeds as its replication family.

## 2026-07-28 — seed-2020 smoke stopped at the behavior gate

The preregistered mechanics smoke ran the unchanged validation protocol with `n=32`, `n_mech=12`, and
`n_null=20`. Its behavior arm completed (`coupling=+0.011`, bootstrap CI `[-0.074,+0.090]`), after which
the frozen mechanism cohort builder found only 5 of 72 screened examples above the required `0.3`
denominator threshold and refused the requested 12. No result JSON was written by the legacy runner, so
the console outcome is preserved in
`results/splicebert_gtex_finetune_shift/seed-2020/smoke_sealed_val_v1.failed.json`.

This is an outcome-limited smoke and cannot establish collapse. It does trigger the preregistered stopping
rule: do not trace the old mechanism until behavior is established. The next run is therefore restricted
to the unchanged behavior/specificity arms on the full validation cohort across all five seeds, with paired
per-example output. Because that comparison runner is finalized after seeing the seed-2020 smoke, its result
is verification tier, not confirmatory.

## 2026-07-29 — freeze a native-classifier bridge before opening the released head

GFS4 leaves encoder erasure and readout incompatibility unresolved because the fixed MLM head loses both
coupling and broad masked-nucleotide competence. The next assay therefore uses the authors' released
three-class head on unmasked sequence. Its donor-vs-no-site logit margin follows the documented
`0=no-site, 1=donor, 2=acceptor` training labels.

[`PREREGISTRATION_NATIVE_CLASSIFIER.md`](PREREGISTRATION_NATIVE_CLASSIFIER.md) was frozen before any
native classifier forward pass. Local donor destruction, clean donor/acceptor recognition, a distant
neutral mutation, and exact final-projection reconstruction form the preflight. Paired-acceptor outcomes
remain unopened until every seed passes. Layerwise or geometric representation results remain unopened
until the native coupling behavior gate passes.

## 2026-07-29 — v1 preflight catches an upstream label-coordinate mismatch

The immutable native preflight v1 failed 20 of 51 checks. All five seeds passed strict loading, finite
forwards, exact final-state projection, and clean acceptor recognition, but none recognized a donor at
the first intronic `G`. The target paired-acceptor intervention had not been run.

The upstream `SpliceaiDataset.check_dataset` code resolves the anomaly: class-1 positions are last-exonic
boundary labels and their `GT` is checked at `label+1:label+3`, whereas the original MLM assay names the
motif's `G` itself as `donor_pos`. The v1 coordinate definition is retracted as an instrument error.
[`PREREGISTRATION_NATIVE_CLASSIFIER_V2.md`](PREREGISTRATION_NATIVE_CLASSIFIER_V2.md) freezes the
one-nucleotide correction before inspecting that corrected output. Thresholds and all other design
choices are unchanged.
