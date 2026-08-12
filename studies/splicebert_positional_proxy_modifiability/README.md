# SpliceBERT positional-proxy modifiability

> **Tier: T2 — prospective successor.** One audited construction-stage row; no model outcome yet.

## What this study is

A prospective successor to `studies/splicebert_positional_boundary_coherence` (PBC v1, closed) and its
in-flight PBC v2 successor line. It asks a narrower, purely logistical question before any model runs:
**can a fresh, globally gene-disjoint, real-gene-module panel actually be built at the scale a
composition-balanced `Ψ` (or paired `ΔΨ`) measurement needs?** This is supply/demand and statistical
power, not biology — no SpliceBERT checkpoint is loaded anywhere in this study.

`CLAIMS.md` currently carries exactly one row, **PPM1**: on the audited eligible census, no fresh
gene-disjoint real-module panel can satisfy the single-checkpoint `Ψ` endpoint, the composition-balance
requirement, and the `0.05` resolution requirement together, for a structural reason (donor/acceptor
gene-pool overlap) rather than a fixable filter-tuning one. Read `CLAIMS.md` for the full scope and every
boundary before citing it — in particular, PPM1 is silent on the paired `ΔΨ` endpoint, which remains
open.

## Read-only boundary — the closed PBC v1 study

`studies/splicebert_positional_boundary_coherence/` and
`results/splicebert_positional_boundary_coherence/` are the **closed PBC v1 study**. This study, like
the PBC v2 successor line before it, **never writes into either tree**. The one file this study reads
from there —
[`results/splicebert_positional_boundary_coherence/development/composition_balanced_cohort_v1.json`](../../results/splicebert_positional_boundary_coherence/development/composition_balanced_cohort_v1.json)
(+ its manifest) — is carried into this repository read-only, as an input the PPM1 audit verifies
against; it is not regenerated, edited, or reinterpreted here. Any future work in this study that wants
a different certificate, cohort, or gene pool must construct its own, under
`results/splicebert_positional_proxy_modifiability/`, never by editing the PBC v1 artifact in place.

## Relationship to the PBC v2 successor line

The controlling artifacts for PPM1 — `results/splicebert_positional_clock/real_panel_capacity_v1.json`
and its independent audit — were produced by, and carried in read-only from, the still-in-flight PBC v2
capacity/power ladder family (`results/splicebert_positional_clock/`, `scripts/splicebert_positional_clock/`).
This study does not own that family's development rungs, superseded versions, or any other
in-flight work; it owns exactly the audit that backs PPM1, plus this ledger. See
[`results/splicebert_positional_clock/RESULTS_INDEX.md`](../../results/splicebert_positional_clock/RESULTS_INDEX.md)
for what that tree currently contains.

## Ledgers

- [`CLAIMS.md`](CLAIMS.md) — one row, PPM1, as above.
- [`RESULTS.md`](RESULTS.md) and [`DECISIONS.md`](DECISIONS.md) are **intentionally empty prospective
  ledgers**, matching the PBC v1 convention: they acquire rows only from immutable, audited outputs
  produced under this study's own name, or from a decision this study itself is forced to make. Nothing
  here promotes an entry from another study's ledger.

## What is NOT claimed here

No SpliceBERT checkpoint has been loaded by anything in this study. PPM1 is a gene-pool
supply/demand and statistical-power finding about a **candidate construction design** — it says nothing
about SpliceBERT's behaviour and nothing about splicing biology. See `CLAIMS.md`'s own "What is NOT
claimed" section for the complete list of boundaries, including the open paired-`ΔΨ` question.
