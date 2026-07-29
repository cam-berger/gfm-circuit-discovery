# D13 preregistration — composition vs. position in the confirmed TSS polarity

> **Tier: T1 — frozen protocol.** Frozen 2026-07-29.
> Controls
> [`../../configs/dnabert2_atlas/promoter_composition_v1.frozen.json`](../../configs/dnabert2_atlas/promoter_composition_v1.frozen.json)
> (`schema_version` 1, `status` `FROZEN`, study `dnabert2_biological_reconstruction_atlas`, assay
> `promoter_composition_v1`). Read before treating any result from this assay as evidence.

D13 is the causal follow-up to D12's confirmed TSS-polarity result. D12 qualified one endpoint,
`E2_tss_polarity`
(`results/dnabert2_reconstruction_atlas/promoter_domain_confirmation_v1.json`), and refuted the
bounded-domain/edge reading it had also tested. D13 asks whether that surviving polarity is a
positional effect or a composition artifact — the field's default explanation for
promoter-classifier positional signal (see **Prior art**). It is an EXCLUSION study: its licensed
conclusion, if qualified, is that composition does *not* account for the polarity, not a positive
claim about what does.

## Question

Does the target-minus-control GC/CpG composition differential explain the confirmed TSS polarity
in mask sensitivity, or does the polarity survive at matched composition?

## What D12 established and refuted

D12 scored four primary/secondary endpoints plus one descriptive endpoint on its own confirmation
split. One qualified:

| Endpoint | Mean | 95% CI | n | Holm p | Qualified |
|---|---|---|---|---|---|
| `E1_downstream_edge` | +0.0375 | [−0.0964, +0.1666] | 138 | 0.47 | **No** |
| `E1b_sharp_edge` | +0.0496 | [−0.0746, +0.1815] | 77 | 0.47 | **No** |
| `E2_tss_polarity` | **+0.3262** | **[+0.1110, +0.5564]** | 149 | **0.0096** | **Yes** |
| `E3_upstream_anti_evidence` | −0.2164 | [−0.4189, +0.0005] | 128 | 0.1476 | No |
| `E4_full_domain` (descriptive only) | +0.0846 | [−0.0442, +0.2295] | 160 | — | n/a |

The sealed, qualified result is `E2 = +0.3262 [+0.1110, +0.5564], Holm p = 0.0096, n = 149`
(`E2_tss_polarity` in the result record): downstream tiles {25, 27} lose more target-specific
classifier margin under masking than upstream tiles {21, 22, 23}. **E1 and E1b both failed** —
neither the relaxed multi-tile downstream-vs-outer contrast (E1) nor the sharper paired
tile-27-minus-tile-28 difference (E1b) cleared its SESOI, excluded zero, or cleared Holm
correction. **The bounded-domain and distal-edge readings are therefore refuted.** (E3, the
upstream anti-evidence endpoint, also failed to qualify, but — per D12's own protocol — an
underpowered two-sided test licenses no null; its failure is not a refutation and is not treated
as one here.) D13 inherits only the polarity; it carries forward no claim, and licenses no claim,
about a bounded domain or an edge at or near +31.

## Background — the exploratory motivation (post-hoc, not confirmatory)

Re-aggregating the D12 confirmation rows — already opened, spending nothing new — against the
cohort's target/control cover sequences, computed by
`scripts/dnabert2_atlas/analyze_polarity_composition_posthoc.py` and recorded in
`results/dnabert2_reconstruction_atlas/polarity_composition_posthoc_v1.json`:

| Quantity | Value | n |
|---|---|---|
| Polarity (reproducing E2) | +0.3262, 95% CI [+0.1049, +0.5547] | 149 |
| r(polarity, δGC) | +0.384 | 149 |
| r(polarity, δCpG) | +0.214 | 149 |
| Polarity, best-matched half (mean\|δGC\|=0.0722) | +0.2967, 95% CI [+0.0465, +0.5480] | 74 |
| Polarity, worst-matched half (mean\|δGC\|=0.3069) | +0.3553, 95% CI [−0.0033, +0.7246] | 75 |

where, per tile cover, `δGC = GC(target cover) − GC(control cover)` (and analogously for `δCpG`,
a CpG-dinucleotide rate), aggregated per locus over whichever of the downstream/upstream tiles are
present, exactly mirroring how the polarity itself aggregates `target_specific_loss`. Loci are
split at the median of `|δGC_downstream − δGC_upstream|` into a best- and worst-matched half.

The correlations are positive and not negligible — composition differential does track the
polarity to some degree, which is exactly why this needs a causal test rather than a re-read of
D12's own numbers. But the polarity does not vanish in the best-matched half, where the mean
absolute GC differential (0.0722) is roughly a third of the frozen tight caliper's tolerance
band; if composition were the whole story, tight matching should have driven the effect toward
zero, and it does not. **None of these numbers are confirmatory.** They are observational,
computed on data the composition hypothesis did not generate a prediction from in advance, and
they license only the hypothesis and the design below — not a claim about causation either way.
See **Provenance and honesty**.

## Provenance and honesty

*"The composition hypothesis is motivated by an exploratory re-analysis of D12 confirmation rows
that had already been opened, recorded as `polarity_composition_posthoc_v1.json`. That analysis is
observational and adjusts for composition rather than intervening on it. D13 uses a fresh cohort
and an outcome-blind matched selection so that the exclusion, if it survives, rests on a
confirmatory design."*

## Scope

**Licensed if qualified:** that the target-minus-control GC/CpG composition differential does not
account for the replicated TSS polarity in mask sensitivity.

**Explicitly excluded**, regardless of outcome:

- a bounded downstream domain or any distal edge — **refuted by D12**,
- a biological promoter element,
- a nucleotide motif,
- native-sequence necessity,
- mediation by any internal component,
- transcriptional function in cells,
- exclusion of compositional features other than GC and CpG.

## Checkpoint

`magicslabnu/DNABERT-2-finetuned-prom_300_all`, revision
`7f2a5f5b8e7e6fffb2dcb396a77d1bc78b7f5f12`, device `mps` — the same checkpoint and revision D8 and
D12 used.

## Cohort and supply

Unlike D12, which reused D8's own confirmation split, D13 draws a **fresh** cohort: GENCODE v50
protein-coding TSSs including non-MANE-Select transcripts, restricted to 1-Mb blocks unused by D4
and D7–D12. This is a supply-definition change, not a reuse, so that matched/unmatched selection
cannot be steered by any locus whose D8/D12 tile-level behavior is already known.

A model-free controller pre-flight verified 814 available fresh 1-Mb blocks, 802 of which carry a
non-MANE-Select TSS. The same pre-flight, run against the existing 384-pair
`promoter_source_localization_v1` cohort as a feasibility proxy (no model, composition-only): 371
of 384 loci admit covers on both the downstream and upstream side; of those, 230 (62%) admit a
matched pair at the frozen caliper (GC 0.05, CpG 0.03), 224 admit **both** arms (matched and
unmatched), and 363 (98%) admit an unmatched arm at the 0.20 GC-separation floor. Matched-pair
yield is therefore the binding constraint, at roughly 62% of loci that clear the first cut. The
config sets `target_raw_pairs_per_split = 220` specifically to land at least
`minimum_matched_pairs_per_split = 120` matched pairs per split under this measured yield, with
headroom.

Supply requirements, all enforced before scoring: exact forward- and reverse-strand GUE exclusion
(no leakage from the benchmark's own train/test partition), composition-matched same-locus
pseudo-TSS controls, and every block disjoint from every prior cohort (D4, D7–D12).

## Regions and cover selection

**Regions.** Downstream: TSS-relative `[+1, +31]`. Upstream: TSS-relative `[−39, −9]`. These are
the two regions whose contrast D12 confirmed. They are **not** a claim that the downstream region
is bounded; D12 refuted the edge.

**Matching.** For one BPE cover, `delta_X = X(target cover) − X(control cover)`, for `X` in {GC
fraction, CpG count per nucleotide}. The pair statistic is
`abs(delta_X(downstream cover) − delta_X(upstream cover))`. The **tight caliper** admitting a
matched pair is GC ≤ 0.05 and CpG ≤ 0.03; the **unmatched arm** requires a GC separation of at
least 0.20. Every cover carries a fixed dose of 2 genomic BPE tokens, at most 18 nt of actual
cover, with target/control cover-length difference at most 2 nt — the same equal-dose geometry
D8/D12 used.

**Selection is outcome-blind.** Among all eligible downstream/upstream cover pairs for a locus,
the selection rule chooses the pair minimizing `abs(delta_GC difference)`, then
`abs(delta_CpG difference)`, then cover-center distance, with ties broken by the lowest start
coordinate. No classifier margin or loss is consulted when choosing which covers to score.

**These calipers are not to be widened.** Widening the tight caliper or the unmatched-arm
separation is the exact thing this study controls for; no result in this study licenses widening
either bound.

## Endpoints

**P1 — matched polarity (primary).** Per locus: `target_specific_loss` at the matched downstream
cover minus `target_specific_loss` at the matched upstream cover, over pairs inside the tight
caliper. One-sided positive alternative. SESOI: 0.15 logit margin. Meeting SESOI with a positive
lower bound means composition does not account for the polarity.

**P2 — composition-attributable component (primary).** Per locus: unmatched-arm polarity minus
matched-arm polarity, on loci supplying both arms. Two-sided alternative. SESOI: 0.15 logit
margin. This is the component of the polarity attributable to the composition differential.

**P3 — dose-response (secondary, never gates confirmation).** The OLS slope of per-pair polarity
on `abs(delta_GC difference)`, across all eligible pairs. Two-sided alternative. SESOI: slope of
0.50. A flat slope supports position; a steep positive slope supports composition.

## Calibration

**Planted positive.** The deliberately unmatched arm (GC separation at least 0.20) must produce a
larger polarity than the matched arm on the same loci, by at least the P2 SESOI, measured on
reserved GUE data. Rationale, as configured: this study's primary claim is an exclusion, and
without demonstrated sensitivity to a composition effect, a null P2 is uninterpretable. This can
fail.

**Permuted-label FPR.** With target/control labels permuted 256 times, the matched-arm polarity
center must fall within 0.05 of zero. This can fail.

**Projected MDE ceiling:** 0.15.

## Why the planted positive is load-bearing

D13's primary claim is an **exclusion**: a qualified P1 says composition does not account for the
polarity. An exclusion is only as good as the instrument's demonstrated ability to detect the
thing being excluded. A null P1 from an instrument that has never been shown sensitive to a real
composition effect is not evidence of anything — it is indistinguishable from an instrument too
weak to detect a true composition effect of any size.

This study already has a recorded instance of exactly this failure mode: **IF30**
(`studies/dnabert2_reconstruction_atlas/CLAIMS.md`) — the native-source preflights v7 and v8
(`promoter_native_source_preflight_v7.json`, `promoter_native_source_preflight_v8.json`) were
`INSTRUMENT_UNRESOLVED` for a GC-depletion instrument: three-base GC depletion projected an MDE of
0.403 at 40 loci, and a disjoint dinucleotide-only slice projected 0.364, in both cases too coarse
to interpret, and no target outcome was read in either run. That instrument was retired rather
than trusted. D13's planted-positive calibration exists so that a null P1 is never reported
without first showing the same measurement pipeline can detect a real, deliberately planted
composition effect of the relevant size. If the planted positive fails, D13 records
`INSTRUMENT_UNRESOLVED` and opens nothing — see **Stop rules**.

## Memorials

All four can fail.

**M1 — identity (on measurement path).** Masking zero tokens must yield `target_specific_loss`
with maximum absolute value below tolerance `1e-5`. This is **not** architecture-forced: it is a
genuine no-op of the same token-substitution path the endpoints use, unlike D12's M1, which
patched Q/K/V identity and was largely architecture-forced.

**M2 — caliper respected.** Every scored matched pair satisfies both tight calipers; every scored
unmatched pair meets the minimum GC separation.

**M3 — equal dose.** Every scored cover masks exactly 2 genomic tokens on each branch, and
target/control cover lengths differ by at most 2 nt.

**M4 — region containment.** Every downstream cover lies within `+1:+31` and every upstream cover
within `−39:−9`, inclusive of partial overlap, with the cover center inside the region.

## Inference

Cluster unit: unique 1-Mb locus block. 5,000 bootstrap draws, 9,999 sign-flip draws, seed
`20260730`, add-one p-values, Holm correction across P1 and P2 (P3 is secondary and is not
Holm-adjusted with them), `α = 0.05`.

## Split gate

**Development requires**, jointly: a qualified outcome-blind supply audit; a QUALIFIED preflight
including both calibration arms (planted positive and permuted-label FPR); and a clean
target-minus-control promoter margin of at least 0.50 logits with a positive 95% lower bound.

**Confirmation opens if** P1 qualifies in development **and** the planted-positive calibration
passed.

**Confirmation uses** the unchanged development definitions, calipers, regions, thresholds, and
seeds — nothing is re-tuned between splits.

## Stop rules

- If the planted-positive calibration fails, record `INSTRUMENT_UNRESOLVED` and open nothing. A
  null P1 without demonstrated sensitivity licenses no exclusion.
- If P1 fails while the planted positive passed, composition is supported as an explanation of the
  polarity, and that must be reported as the finding.
- P3 is secondary and never gates confirmation.
- No result licenses any statement about a bounded domain or a distal edge; D12 refuted those.

## Prior art

**Saxonov S, Berg P, Brutlag DL (2006).** *PNAS* 103(5):1412–1417. doi:10.1073/pnas.0510310103.
72% of human promoters are high-CpG.

**Abeel T, Van de Peer Y, Saeys Y (2009).** *Bioinformatics* 25(12):i313–i320.
doi:10.1093/bioinformatics/btp191. Promoter prediction is biased toward CpG-island promoters.

**Krützfeldt L-M, Schubach M, Kircher M (2020).** *PLOS ONE* 15(12):e0237412.
doi:10.1371/journal.pone.0237412. Negative-set choice determines whether models learn GC content.

**Hadad E, Kogman N, Golan L, et al. (2026).** *NAR Genom Bioinform* 8(1):lqag025.
doi:10.1093/nargab/lqag025. Peer-reviewed. Promoter language models rely on localized CpG-rich
signatures under random-genomic negatives.

**Cheng B, Zhang J (2026).** arXiv:2604.06549. **Preprint, not peer-reviewed.** Claims composition
dominates positional effects 46-fold — a claim this study would contradict if P1 qualifies. That
contradiction, not agreement with prior work, is the reason a causal confirmatory design is
required before D13 licenses anything.

## What would refute this

Refutation is evaluated per endpoint. "Failing" means: the point estimate falls short of the
endpoint's SESOI, or its 95% cluster-bootstrap interval fails to exclude zero on the alternative
side, or the Holm-adjusted add-one p exceeds `α = 0.05` (see **Endpoints** and **Inference**).

- **P1 failing while the planted positive passed** supports composition as the explanation of the
  polarity, and must be reported as the finding — not as an inconclusive result.
- **P1 failing while the planted positive also failed** licenses nothing: the instrument was never
  shown able to detect a real composition effect, so a null P1 is uninterpretable (see **Why the
  planted positive is load-bearing**).
- **A large, qualified P2 alongside a surviving (qualified) P1** means composition contributes to
  the polarity without accounting for it — report the decomposition, not a clean exclusion.
- **No outcome of this study**, however it resolves, licenses a claim about a bounded downstream
  domain or a distal edge. D12 already refuted that reading; D13 cannot revive it.
