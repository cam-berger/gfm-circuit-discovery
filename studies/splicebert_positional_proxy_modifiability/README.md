# SpliceBERT positional-proxy modifiability

> **Tier: T2 — prospective successor.** One audited construction-stage row; no model outcome yet.

## What this study is

A prospective successor to `studies/splicebert_positional_boundary_coherence` (PBC v1, closed) and its
in-flight PBC v2 successor line. It opened on a narrower, purely logistical question, asked before any
model runs: **can a fresh, globally gene-disjoint, real-gene-module panel actually be built at the scale
a composition-balanced `Ψ` (or paired `ΔΨ`) measurement needs?** That question is supply/demand and
statistical power, not biology, and `PPM1` — the study's only claim — answers it with no checkpoint
loaded anywhere in its controlling artifacts.

**Since 2026-08-12 the study also contains two model runs.** A development-stage **pilot**
(`results/splicebert_positional_proxy_modifiability/lineage_pilot_v1.json`) continue-pretrains
SpliceBERT on two corpora differing only in a reading-frame residue mod 3, and a three-arm
**engineering pilot** (`lineage_carrier_dependence_v1.json`) adds the native no-edit control arm the
first lacked. Both back **no claim row** and have **no independent audit**; the first has one
training seed per arm and the second has two, so neither is a training-process statement. See
[`RESULTS.md`](RESULTS.md)'s audit note and the 2026-08-12 entries in [`DECISIONS.md`](DECISIONS.md)
for the full scope. Earlier revisions of this README stated that no checkpoint is loaded anywhere in
this study; that was true until that date.

**The second run bounds the first.** With a native, no-edit arm in place, the rise in `Ψ` under
continued pretraining is **generic continuation** rather than phase-preservation: the unedited arm
gives the *largest* `Ψ` in both seeds. What both runs do agree on, and what the second measures
directly rather than inferring from a phase-blind number, is that the preference — trained or not —
**depends on the learned period-3 carrier**, while training leaves that carrier's amplitude, phase
and per-dimension distribution essentially untouched.

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
- [`RESULTS.md`](RESULTS.md) carries one development-stage pilot row (no claim, no audit); it acquires
  **claim-backing** rows only from immutable, independently audited outputs produced under this study's
  own name. Nothing here promotes an entry from another study's ledger.
- [`DECISIONS.md`](DECISIONS.md) records why this successor exists, the base-checkpoint decomposition,
  the necessity and relocation tests that falsified the original premise, and the 2026-08-12 pilot.

## What is NOT claimed here

PPM1 is a gene-pool supply/demand and statistical-power finding about a **candidate construction
design**, with no checkpoint loaded in any of its controlling artifacts — it says nothing about
SpliceBERT's behaviour and nothing about splicing biology. See `CLAIMS.md`'s own "What is NOT claimed"
section for the complete list of boundaries, including the open paired-`ΔΨ` question, which the 2026
pilot does **not** resolve (it has one seed per arm and yields no seed-level L1/L2 covariance).

The study's **original premise is falsified** and is not claimed anywhere: the learned period-3 carrier
is necessary for `Ψ` but does not encode which register is preferred, so no corpus manipulation can
re-target the preference through the carrier. The pilot's `ΔΨ` is not a re-targeting result, and it
licenses no mechanism, route, or biological statement. It was described as a *reinforcement* result;
that description is withdrawn — the three-arm run shows the reinforcement is generic continuation,
and what residue mod 3 does is suppress it, between two edited arms that both sit below native.
