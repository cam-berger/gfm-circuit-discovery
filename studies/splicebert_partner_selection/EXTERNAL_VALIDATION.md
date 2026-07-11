# External Biological Validation Plan

## Goal

Determine whether a frozen circuit-derived score predicts an independent splicing outcome beyond local motif and distance features.

## Acceptable datasets

An external dataset must provide sequence/genomic coordinates and an outcome such as exon inclusion change, cryptic splice-site use, aberrant junction counts, or experimentally measured splice disruption. It must not be used to choose circuit components.

## Normalized event schema

Each event should contain:

- `event_id`, `gene_id`, chromosome, strand;
- reference and alternative alleles or perturbation;
- affected acceptor/donor candidates;
- transcript-oriented sequence window;
- measured outcome and uncertainty;
- cohort/experiment identifier for grouped splitting;
- provenance and source-version fields.

## Frozen predictors

- local PWM/length score change;
- whole-model pair-margin change;
- selected circuit activation or mediation score change;
- optional fine-tuned model score, labeled separately.

## Analysis

Use held-out genes or experiments. Compare nested predictive models and report bootstrap intervals grouped by gene/experiment. Freeze transformations and thresholds before opening the final external holdout.

## Claim rule

A circuit is a `BIOLOGICAL_PREDICTOR` only when its frozen score adds predictive value beyond registered local controls on independent data. A novel biological rule requires further independent or experimental validation.
