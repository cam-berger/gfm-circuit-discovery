# D12 preregistration — bounded downstream domain and TSS polarity

> **Tier: T1 — frozen protocol.** Frozen 2026-07-28; amended 2026-07-28 (Amendment 1, pre-outcome).
> Controls
> [`../../configs/dnabert2_atlas/promoter_domain_v2.frozen.json`](../../configs/dnabert2_atlas/promoter_domain_v2.frozen.json)
> (`schema_version` 2, `status` `FROZEN`, study `dnabert2_biological_reconstruction_atlas`, assay
> `promoter_domain_v2`, `supersedes` `promoter_domain_v1.frozen.json`). The original
> [`../../configs/dnabert2_atlas/promoter_domain_v1.frozen.json`](../../configs/dnabert2_atlas/promoter_domain_v1.frozen.json)
> is preserved unedited and superseded, not deleted — see **Amendment 1** below for why. Read before
> treating any tile-level confirmation number from this assay as evidence.

D12 is the confirmatory follow-up to D8 (`studies/dnabert2_reconstruction_atlas/CLAIMS.md`, D8 entry).
It reuses D8's own sealed confirmation split, under a protocol frozen before any of its primary-endpoint
tiles are read. The exact boundary of what has and has not already been opened on that split is stated
in **Provenance and honesty** below.

## Question

Is the D8 mask profile a bounded downstream domain with a distal edge and a TSS polarity reversal,
rather than a single 10-nt tile?

## Background — why this hypothesis (exploratory, not confirmatory)

D8 froze thirty non-overlapping 10-nt nominal tiles across the 300-nt promoter window (`TSS_index`
249) and reported exactly one development-and-confirmation-qualified test: tile 26, nominal +11:+21
relative to the TSS, joint 60-test family `p=0.0260` in development, replicated at +0.2056 in
confirmation (`studies/dnabert2_reconstruction_atlas/CLAIMS.md`, D8 entry).

Re-aggregating the **frozen D8 development** per-locus records after that result was read — an
exploratory, hypothesis-generating pass, not a new confirmatory analysis — shows tile 26 sitting inside
a three-tile plateau with a sharp distal edge and an upstream sign reversal:

| Tile | TSS-relative | Raw target-specific loss | 95% CI | Localisation contrast | D8 60-test family p |
|---|---|---|---|---|---|
| 22 | −29:−19 | −0.279 | [−0.518, −0.020] | — | — |
| 25 | +1:+11 | +0.268 | [+0.090, +0.451] | +0.225 | 0.073 |
| 26 | +11:+21 | +0.245 | [+0.076, +0.429] | +0.266 | **0.026** |
| 27 | +21:+31 | +0.252 | [+0.086, +0.431] | +0.221 | 0.081 |
| 28 | +31:+41 | +0.002 | [−0.093, +0.099] | — | — |

Tiles 25, 26, and 27 all clear both frozen D8 SESOIs (0.10 raw, 0.05 localisation contrast) with
positive lower bounds; they differ only in D8's own 60-test (30 tiles × 2 operations) family
correction, which tile 26 alone clears. Tile 28, one tile further downstream, drops to a raw effect
indistinguishable from zero — a sharp distal edge, not a gradual decay. Tile 22, upstream of the TSS,
is negative with a bootstrap interval excluding zero — the opposite sign from the downstream plateau.

Further exploratory re-aggregation favors a domain reading over a single-tile reading:

- A pooled per-locus domain contrast of +0.319, 95% CI [+0.073, +0.580], n=39 loci, computed under
  **D8's own localisation-contrast definition**: per-locus mean(`target_specific_loss` over inner
  tiles 25, 26, 27) minus that locus's mean over *all* its other eligible tiles (at least 10
  required), with all three inner tiles required present. **This is not the E1/E2 estimand.** E1 and
  E2 compare the inner tiles against small, specifically named outer/upstream tile sets ({28, 29} and
  {21, 22, 23}), not against a locus's full remaining tile panel. This number motivates the domain
  hypothesis; it is not an E1, E1b, or E2 estimate, and it should not be read as one.
- **Composition dissociation:** tiles 26 and 28 have near-identical masked-cover content (GC 0.617 vs.
  0.621, CpG 0.054 vs. 0.055) yet a paired effect difference of +0.295, 95% CI [+0.085, +0.518], on 70
  shared loci — the distal edge is not explained by a composition difference between the two covers.
- The effect is uniform across strata D8 already carried: CpG-island +0.286 vs. non-island +0.313;
  CAGE-supported +0.312 vs. not +0.290.

None of these numbers are confirmatory: they come from the spent D8 development split and license only
the hypothesis and the endpoint design below. See **Provenance and honesty**.

## Scope

**Licensed if qualified:** a replicated bounded classifier-relevant downstream source domain and a
replicated TSS-relative polarity in mask sensitivity.

**Explicitly excluded**, regardless of outcome:

- a biological promoter element,
- a nucleotide motif,
- native-sequence necessity,
- mediation by L11H2/L11H5 or L11MLP,
- transcriptional function in cells.

## Checkpoint

`magicslabnu/DNABERT-2-finetuned-prom_300_all`, revision `7f2a5f5b8e7e6fffb2dcb396a77d1bc78b7f5f12`,
device `mps` — the same checkpoint and revision D8 used.

## Cohort and split reuse

The cohort is D8's own annotated source,
`data/processed/dnabert2_atlas/promoter_source_localization_v1/cohort.jsonl`, 192 pairs per split.
D12 scores only the **confirmation** split; it opens no new locus blocks.

The confirmation split is reused, and only `mask|tile26` has ever been opened on it. **Design
decision:** the primary endpoints — E1, E1b, and E2 — are defined over tiles whose confirmation values
have never been read, so that tile 26's already-public value cannot partially inform a primary result.
E4 includes tile 26 for exactly this reason and is descriptive only — it can never qualify.

## Tile geometry

300-nt input; `TSS_index` 249 (TSS-relative position = window index − 249); nominal tile width 10 nt;
a fixed dose of 2 genomic BPE tokens per sequence; actual cover at most 18 nt; target/control
cover-length difference at most 2 nt.

## Tiles scored on confirmation

Only 8 of the 30 D8 tiles are scored: `{21, 22, 23, 25, 26, 27, 28, 29}`. Tiles `{0–20, 24}` (22
tiles) stay sealed for future work. Tile 26 is scored **only** to satisfy memorial M2 — it enters no
primary or secondary endpoint.

| Tile | TSS-relative window | Used by |
|---|---|---|
| 21 | −39:−29 | E2 (upstream) |
| 22 | −29:−19 | E2 (upstream), E3 |
| 23 | −19:−9 | E2 (upstream) |
| 25 | +1:+11 | E1 (inner), E2 (downstream) |
| 26 | +11:+21 | M2 record anchor only |
| 27 | +21:+31 | E1 (inner), E1b, E2 (downstream) |
| 28 | +31:+41 | E1 (outer), E1b |
| 29 | +41:+51 | E1 (outer) |

`tiles_to_score` and `tiles_remaining_sealed` partition `{0, …, 29}` exactly: every tile is either
scored now or explicitly sealed, with no gap and no overlap.

## Endpoints

The definitions below are **v2**
(`configs/dnabert2_atlas/promoter_domain_v2.frozen.json`), in force since Amendment 1. **v1**
(`configs/dnabert2_atlas/promoter_domain_v1.frozen.json`) is preserved, unedited, and superseded — it
required every named tile to be present at a locus before scoring it, which collapsed the scorable
cohort before any confirmation outcome was read. See **Amendment 1** immediately below for the
numbers and the reasoning.

**E1 — downstream edge (primary).** Per locus: mean(`target_specific_loss` over tiles 25, 27) minus
mean over tiles 28, 29. **Completeness rule (v2):** at least one inner tile (25 or 27) and at least
one outer tile (28 or 29) must be present at that locus; the mean is taken over whichever are
available. Excludes tile 26. One-sided positive alternative. SESOI: 0.10 logit margin.

**E1b — sharp edge (primary, added in v2).** Per locus: `target_specific_loss` at tile 27 minus
`target_specific_loss` at tile 28. Both tiles required — a locus missing either is not scored. This is
the sharpest available statement of the +31 nt boundary: it needs only two tiles per locus, so it is
not subject to E1's inner/outer averaging. One-sided positive alternative. SESOI: 0.10 logit margin.

**E2 — TSS polarity (primary).** Per locus: mean(`target_specific_loss` over tiles 25, 27) minus mean
over tiles 21, 22, 23. **Completeness rule (v2):** at least one downstream tile (25 or 27) and at
least one upstream tile (21, 22, or 23) must be present at that locus; the mean is taken over
whichever are available. Excludes tile 26. One-sided positive alternative. SESOI: 0.15 logit margin.

**E3 — upstream anti-evidence (secondary, unchanged by Amendment 1).** Per locus:
`target_specific_loss` at tile 22 alone. **Design decision:** D8's protocol used a one-sided positive
alternative throughout, which made an upstream negative effect structurally invisible regardless of
its size. E3 is explicitly **two-sided**. SESOI: 0.10 absolute logit margin.

**E4 — full domain (descriptive only, unchanged by Amendment 1).** Per locus:
mean(`target_specific_loss` over tiles 25, 26, 27) minus mean over tiles 28, 29. Includes tile 26,
whose confirmation value D8 already opened, so E4 **can never qualify** — it is reported for context
only.

## Amendment 1 — endpoint completeness rule (pre-outcome)

**Trigger.** v1's E1 and E2 required *every* named tile to be present at a locus before scoring it —
tiles 25, 27, 28, 29 for E1; tiles 25, 27, 21, 22, 23 for E2. Tile eligibility varies by locus (not
every locus has an eligible two-token BPE cover at every nominal tile), so this all-or-nothing rule
discarded most of the cohort. Computed under v1's own endpoint definitions, from
`data/processed/dnabert2_atlas/promoter_source_localization_v1/development.raw.jsonl`:

| Endpoint (v1 definition) | Estimate | 95% CI | n |
|---|---|---|---|
| E1 — all of 25, 27, 28, 29 required per locus | +0.1419 | [−0.1009, +0.4074] | 34 |
| E2 — all of 25, 27, 21, 22, 23 required per locus | +0.7433 | [+0.2346, +1.2806] | 22 |

E1's development interval **spans zero** under this rule. Opening the sealed confirmation split on an
endpoint whose own development estimate cannot be distinguished from zero would spend that seal on a
test that was never adequately powered by its own design.

**Decision.** Freeze v2 (`configs/dnabert2_atlas/promoter_domain_v2.frozen.json`, which declares
`"supersedes": "configs/dnabert2_atlas/promoter_domain_v1.frozen.json"`). v1 is not edited — it stays
immutable as the record of what was originally specified and why it needed to change. v2 relaxes E1
and E2 to a **completeness rule** (at least one inner/downstream tile and at least one outer/upstream
tile present, averaging over whichever are available) and adds a new primary endpoint, **E1b**, that
needs only tiles 27 and 28 (both required) and states the downstream edge in its sharpest, most
tile-parsimonious form. `inference.multiplicity` becomes `Holm across E1, E1b, E2, E3`. E3 and E4 are
unchanged. `tiles_to_score` and `tiles_remaining_sealed` are unchanged and were re-verified to still
partition `{0, …, 29}` exactly, with no gap and no overlap.

Under the v2 definitions, the same development data give:

| Endpoint (v2 definition) | Estimate | 95% CI | n |
|---|---|---|---|
| E1 — mean(avail {25,27}) minus mean(avail {28,29}), ≥1 tile each side | +0.2291 | [+0.0917, +0.3752] | 149 |
| E1b — tile 27 minus tile 28, both required | +0.2631 | [+0.0695, +0.4890] | 90 |
| E2 — mean(avail {25,27}) minus mean(avail {21,22,23}), ≥1 tile each side | +0.4045 | [+0.1661, +0.6451] | 160 |
| E3 — tile 22 alone, two-sided (unchanged) | −0.2793 | [−0.5307, −0.0420] | 126 |

**No confirmation-split outcome had been read when this amendment was made.** This is a pre-outcome
design correction to a definitional flaw, not a post-outcome adjustment.

**Disclosure.** *"The endpoint definitions in v2 were selected after inspecting several variants on
the development split. That selection inflates the development estimates reported here and they must
not be read as effect-size estimates. It does not affect the confirmation test, which remains sealed
for every endpoint except `mask|tile26`."*

## Memorials

All four can fail; M1 and M2 sit on the measurement path.

**M1 — identity.** A no-op intervention (zero tokens replaced) must yield `target_specific_loss` with
maximum absolute value below tolerance `1e-5`.

**M2 — record anchor.** Re-scoring the unchanged `mask|tile26` confirmation test must reproduce
**every** recorded per-example `target_specific_loss` in
`data/processed/dnabert2_atlas/promoter_source_localization_v1/confirmation.raw.jsonl` (rows with
`operation == 'mask'`, 124 rows), per example, to within `1e-4` maximum absolute difference. Recorded
mean for reference: 0.2056. This bar is verified-achievable, not aspirational: a 6-row spot check under
the pinned environment already reaches a worst absolute difference of `4.3e-06`.

**M3 — cover balance.** For every scored tile-locus, abs(target cover length minus control cover
length) is at most 2 nt.

**M4 — equal dose.** Every scored tile masks exactly 2 genomic tokens on each branch.

## Inference

Cluster unit: unique 1-Mb locus block. 5,000 bootstrap draws, 9,999 sign-flip draws, seed `20260729`,
add-one p-values, Holm correction across E1, E1b, E2, E3 (updated by Amendment 1; v1 read `Holm across
E1, E2, E3`), `α=0.05`.

## Split gate

Development predictions are frozen from D8's own
`data/processed/dnabert2_atlas/promoter_source_localization_v1/development.raw.jsonl` — producing them
**requires no model run**.

Confirmation opens only if all four memorials pass **and** the frozen development prediction file
exists with a recorded SHA-256.

An endpoint qualifies only if, jointly:

1. the point estimate meets the endpoint's SESOI;
2. the 95% locus-block bootstrap interval excludes zero on the alternative side;
3. the Holm-adjusted add-one sign-flip p is at most 0.05;
4. the confirmation sign matches the frozen development prediction.

## Stop rules

- If M1 or M2 fails, record `INSTRUMENT_UNRESOLVED` and open no confirmation endpoint.
- If E1 and E2 both fail, the domain claim is not supported; do not run Phase H.
- Failure of E3 licenses no null: it is powered only for effects at or above 0.10 absolute.

## Provenance and honesty

*"The hypothesis was generated by re-aggregating the frozen D8 development records after those records
had been read. The development split is therefore spent and contributes predictions only. The
confirmation split is reused from D8, on which exactly one endpoint (`mask|tile26`) was previously
opened; E1–E3 are defined to exclude tile 26 so that no primary endpoint is informed by that opening.
E4 includes tile 26 and can never qualify."*

## Prior art

**Hadad E, Kogman N, Golan L, et al. (2026).** Optimizing genomic language models for promoter
prediction. *NAR Genom Bioinform* 8(1):lqag025. doi:10.1093/nargab/lqag025. Peer-reviewed. Uses
positional SHAP to show promoter language models can learn genuine positional structure under
shuffled-promoter negatives. D12 differs by being a **causal-intervention test with matched equal-dose
controls**, not an attribution result.

**Cheng B, Zhang J (2026).** The Mechanistic Invariance Test. arXiv:2604.06549. **Preprint, not
peer-reviewed — cited as such.** Reports that composition dominates positional effects 46-fold across
five genomic language models. D12's hypothesis — a bounded, TSS-polarized downstream domain — would
contradict this claim if confirmed. That contradiction, not agreement with prior work, is the reason
confirmation is required before D12 licenses anything.

## What would refute this

- **E1 ≤ 0, or its 95% interval spans zero:** refutes a bounded downstream domain. The plateau read
  from D8 development would not replicate.
- **A positive E1 and a positive E2, but no tile-22 sign reversal on E3:** refutes only the
  *polarity-reversal* reading, not the bounded-domain reading. E2 tests the upstream tiles as a group,
  so a group-level polarity difference can pass without the specific tile-22 negative value
  replicating.
- **E3 failing to clear its SESOI is not a null result.** E3 is two-sided and powered only for effects
  at or above 0.10 absolute; a small or non-significant tile-22 effect under that power does not
  license the conclusion that no reversal exists.
