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
