# STATUS — where the program stands

> **⚠️ SUPERSEDED 2026-07-12 — sealed test complete; several numbers below are RETRACTED.**
> Read `studies/splicebert_partner_selection/CLAIMS.md` §Retractions (R7–R12) and
> `SEALED_TEST_RESULT.md` before quoting anything on this page.
> In particular: `p = 0.000999` is a function of the `--n-null` flag, not the data (R9; honest
> value **0.0104**); `MLM health 0.9958 → 0.9953` scored the model copying unmasked input and
> could not fail (R8; real baseline **0.53**); coupling **1.359** is a *conditional* mean (R10;
> unconditional **0.973**); the competing-acceptor "binding" result is an algebraic identity
> (R7, withdrawn); and every pre-fix number is **plus-strand only** (R11).

**[INDEX]** — the current state of all three tracks, in one place, including what has been retracted.
**Read this if** you are picking the work up, reviewing it, or deciding what to run next.
**This file is the single source of truth** for test counts, track status, and claim status. Where any
other document disagrees with this one, this one is right and the other is stale — tell someone.

**Verified:** 2026-07-12, against a live `uv run pytest -q` and the artifacts under `results/`.

---

## The one-paragraph state

The **instruments are sound and the toy-model science is settled**: a circuit tracer with a measured
error rate against hand-built ground truth, and a cue-conflict protocol that separates a real mechanism
from a pooled shortcut. Two circuits are traced end-to-end on GFM-nano — an anchored reading-frame
counter and a shortcut-proof repeat-copy circuit — and they sit on opposite sides of the organising
hypothesis. On a **real** GFM the picture is younger and more painful: the first registered SpliceBERT
study was **retracted in full** after its task was found to be degenerate, and the replacement design —
cross-site coupling — has produced the program's **first held-out-confirmed circuit in a pretrained
genomic model** (head L3H1). That result is real but is documented only in `studies/` and `results/`;
it has not yet been written into the argument.

**The honest summary for a reviewer:** the method is the asset. The reading-frame and repeat-copy
tracks are finished and defensible. The SpliceBERT track has one confirmed circuit, one fully retracted
study, and a claims registry that is ~1 day behind its own artifacts.

---

## Repository health

| Check | State |
|---|---|
| Test suite | **167 collected · 150 pass · 16 skip · 1 FAIL** (`uv run pytest -q`, ~90 s) |
| The failure | `tests/test_partner_data.py::test_fixed_flanks_would_fail_the_memorial_test` — `PartnerConfig.__init__() got an unexpected keyword argument 'flanks'`. A *memorial* test (it exists to prevent the R2 positional leak from returning) is calling a constructor signature that has since changed. **This must be fixed, not deleted** — see `TODO.md`. |
| The 16 skips | Genome-dependent (`data/genomes/hg38.fa` is not in the repo) and checkpoint-dependent tests. Expected. |
| Environment | `uv` (`uv sync --extra lab --extra dev`). The old conda env `gfm-thrust1` is **retired and gone**; any doc still naming it is stale. |
| Reproducibility blocker | hg38 + GENCODE are not in the repo and there is no fetcher. This gates the real-CDS cells (C, D) and the pooled-oracle check. `TODO.md` §1.1. |

Earlier documents cite test counts of 92, 102, 109, 127 and "~130". They were all true once. The number
above is the one that is true now.

---

## Track 1 — Reading frame (GFM-nano) · **SETTLED**

*The negative, and the forcing experiment that licenses it.*

Real hg38 GFM-nano **does not** build an anchored frame counter, and the reason is measurable: a pooled
codon statistic recovers the frame at 0.9509 with no start codon at all, so the circuit is worth ~0.4%
of the loss. Make the frame *necessary* (uniform codons, amino-acid readout) and the circuit appears
abruptly and verifies on every causal assay. A necessity 2×2 then shows the counter is displaced only
when the anchor is unreliable **and** a pooled shortcut is available.

| Result | Where |
|---|---|
| The pooled ceiling, the forcing task, the λ sweep, the 2×2 | `RESULTS.md` §4–§9 |
| The trace, and the naturally-occurring causally-inert head (L1H2) | `RESULTS.md` §6, `DECISIONS.md` D27 |
| The relay, and the retraction of four of the five "relays" | `RELAY_TRACE.md`, `DECISIONS.md` D34–D35 |

**Open:** component localisation on cell D's relay (T4); three seeds per cell (everything is one seed);
and the LayerNorm hole below.

## Track 2 — Repeat copy / induction (GFM-nano) · **SETTLED**

*The positive: where a shortcut is provably unavailable, the circuit forms with no coercion.*

| Measurement | Value |
|---|---|
| Copy accuracy (seed 0 / seed 99) | 0.9608 / 0.9662 |
| Base-rate ceiling | 0.2516 |
| Ceiling audit — local naive Bayes / RF-limited CNN, source denied | 0.2484 / 0.2473 (**stop-condition did not fire**) |
| Source oracle (upper reference) | 1.0000 |
| Cue conflict — follows distal edited source / local continuation | 0.9635 / 0.0130 |
| Head structure | five L1 heads, partially redundant (Σ single Δloss 2.91 vs joint 1.32) |
| Natural hg38 repeats (chr17–19) | distal-following 0.7820 / 0.7830; prefix-scramble collapses it to 0.2330 |

Full record: `branches/codex-repeat-induction/` — `FINDING.md`, `COPY_CEILING.md`, `COPY_TRACE.md`,
`NATURAL_REPEATS.md`. This is the one track whose ceiling was measured rather than argued.

## Track 3 — SpliceBERT (a real, pretrained GFM) · **ONE CONFIRMED CIRCUIT, ONE RETRACTED STUDY**

This is the track the next GFM springs from, so its history matters more than its headline.

### 3a. Partner selection (`splicebert_partner_selection_v1`) — **RETRACTED IN FULL**

The registered study asked SpliceBERT to pick the true acceptor–donor pair against decoys. Claims
C1–C4 are **all retracted**. Three failures, recorded permanently in
`studies/splicebert_partner_selection/CLAIMS.md`:

- **R1** — the randomised-backbone control was a **no-op**. `init_weights()` does nothing after
  `from_pretrained` on transformers ≥5, so the "random" backbone was the pretrained one, byte-identical.
- **R2** — a **positional leak**. `dataset.py` pinned the acceptor to a constant flank, so the true
  acceptor sat at position 200 in **100.0%** of examples. A geometry oracle reading **zero nucleotides**
  scores a perfect 1.000. The headline "0.807 vs 0.056" measured window geometry, not biology. Separately,
  the decoys were spurious motifs, so an oracle *site detector* also scores 1.000 — the task never
  required partner information at all.
- **R3** — the v2 fix (real annotated decoys) was **biologically unfixable**. The decoys were *minor*
  donors (median 1 supporting isoform vs 18 for true donors), so "pick the partner" collapsed into "pick
  the better donor": a plain PWM scores 0.687 against 0.392 chance, and the acceptor contributed nothing
  (+0.008). In a real genome, a strong donor at a plausible exon distance from an acceptor generally *is*
  its partner — clean negatives barely exist, and the forced-choice task is intrinsically degenerate.

**The lesson, and it is the reusable one:** a forced-choice task invites the model to win by any route,
and the registered controls did not cover the routes that existed. The replacement design removes the
task entirely.

> ⚠️ Four files in `studies/splicebert_partner_selection/` — `GATE_REPORT.md`, `RESULTS.md`,
> `return_to_planner.json`, `STAGES.md` — were written before the retraction and still present the
> retracted "GO" as an established result. They now carry VOID banners. **Do not quote them.**
> `CLAIMS.md` and `DECISIONS.md` are the current record.

### 3b. Cross-site coupling — **CONFIRMED ON HELD-OUT DATA**

The replacement design: no task, no label, no decoys. Read SpliceBERT's **native** masked-LM
recognition of a splice site, destroy the *partner* site by substituting real bases (never `[MASK]`),
and measure the damage against matched controls.

Discovery ran on train chromosomes; the ladder was then **re-run on gene-disjoint held-out chromosomes
(chr11/16/19, n = 400, 1000 null sets)** by `scripts/splicebert/run_confirm.py`. It replicates:

| Held-out finding | Value | Control |
|---|---|---|
| L3H1's rank among all 96 heads | **1st** | next head (L4H2) recovers 0.186 |
| Denoise recovery, L3H1 alone | ~~0.808~~ | ⚠️ **plus-strand only (R11)**. Sealed test: patching C restores **79%**; key+value at the acceptor **0.663**. |
| ~~Matched-random null (1000 single-head sets)~~ | ~~observed 1.193~~ | ~~p = 0.000999~~ ⚠️ **RETRACTED (R9)** — 1000 draws from a 95-head universe are 95 distinct values; p was `1/(n_null+1)`, a function of the flag. Honest value **0.0104**. Sealed test: **p = 0.0025** over 400 distinct 2-head sets. |
| Partner-dependent effect (2×2 factorial DiD) | **1.110** [1.018, 1.205] | — |
| Rank-2 subspace of L3H1's output | removes **80% of the effect** | random rank-2 removes ~2%. ⚠️ Absolute values (1.359 → 0.271) were **conditional** (R10) and **plus-strand** (R11) — fractions stand, absolutes do not. |
| MLM health under ablation | **0.5146 → 0.5107 (99.2% retained)** | ⚠️ The old **0.9958 → 0.9953** is **RETRACTED (R8)**: it scored the model COPYING UNMASKED INPUT and *rose* when all 96 heads were ablated. It could not fail. Masked probes give a baseline of **0.53**, not 0.99. |

Behavioural support (train, n = 250): partner acceptor **+0.380** vs motif-matched non-partner AG
**−0.000** and distance-matched random position **+0.010**; own donor +0.380 vs other donors +0.003
(paired Wilcoxon p = 9.3e-18); bidirectional.

**Two caveats that must travel with this result:**
1. **It is coupling across the *intron*, not the exon.** Measured at matched separation, exon +0.288 vs
   intron +0.490 — a ratio of 0.24×. This **retracts the team's own earlier "exon definition" reading**,
   which was distance decay. And only *short* introns fit SpliceBERT's 1024-nt context, so it cannot be
   generalised to typical human introns.
2. **The test split (chr8, chr9) is still sealed.** This is a held-out *validation* result, not a test
   result.

**Claims registry: reconciled 2026-07-12.** `CLAIMS.md` now matches its artifacts.
- **D1, D4, D5 → `artifact-supported (held-out)`.** Promoted on `confirmation.json`.
- **D4′ → retracted (R4).** The claim that L3H1 is *dedicated* to the coupling — "partner-independent
  role ≈ 0" — is **falsified by the same run that confirmed the rest of D4**. Held-out gives
  **−0.200 [−0.272, −0.128]**, a CI excluding zero. The head carries the coupling; it does not carry
  *only* the coupling. The flattering "≈ 0" came from the discovery split (n_eval = 50) — which is
  precisely the failure mode this study already paid for once.
- **Claim label: `CIRCUIT_CANDIDATE`**, not `CAUSALLY_VALIDATED_CIRCUIT`. We have necessity and
  mediation, but no path account (QK/OV, edge patching), only one ablation null, and an unexplained 19%
  residual. Those three are what the label is waiting on.

---

## The hypothesis, and its honest status

The organising claim, and the *only* wording that should be used going forward:

> **The shortcut-gap hypothesis.** Relational mechanisms become more likely when they provide enough
> objective improvement over a registered class of local or pooled shortcuts to overcome their
> optimisation and complexity cost. Shortcut insufficiency creates *pressure* for a relation; it does
> not guarantee that gradient descent finds one, and shortcut availability does not prove that no
> relational mechanism exists.

Earlier documents state a stronger **"if and only if" sufficiency law**. That wording is **not
established** and is retained only where its historical status is marked. It rests on n = 8 arms, one
task, one architecture. Treat any doc asserting the "iff" as a record of what we believed at the time.

---

## The holes, ranked

| # | Hole | Why it matters |
|---|---|---|
| 1 | **LayerNorm faithfulness is UNMEASURED** | Every tracer error rate came from LayerNorm-free planted models. GFM-nano and SpliceBERT both have LayerNorm. This is the last open hole in Gate 0, and it gates every causal claim downstream. One day of work. |
| 2 | **`tests/test_partner_data.py` fails** | It is a memorial test guarding against the R2 leak. A guard that does not run is not a guard. |
| 3 | **hg38/GENCODE not fetchable** | No clean-checkout reproduction of cells C/D or the pooled-oracle ceiling. `TODO.md` §1.1. |
| 4 | **One seed nearly everywhere** | Every 2×2 and Phase-C cell number is a single seed. Nothing below Phase D may be called SETTLED at the cell level. |
| 5 | **Ceilings are lower bounds** | Both ceilings we predicted from were beaten by the model. Ceilings need a model-class-matched estimator, not naive Bayes. This cuts *against* us — it makes circuits rarer. |
| 6 | **The coupling result is not in the argument** | `REC.md` and the proposal predate it. The first real-GFM circuit is currently invisible to anyone reading the outward-facing docs. |

---

## What to run next

1. **Promote or correct D4/D5** in `studies/splicebert_partner_selection/CLAIMS.md` against
   `confirmation.json` (above). Cheap, and the registry is the thing a reviewer will read.
2. **Fix the memorial test.** It is the guard on the bug that cost the study.
3. **Close the LayerNorm hole.** Plant a circuit in an LN-equipped tiny transformer; re-run Tests A and C.
4. **Write the coupling circuit into `RESULTS.md` and `REC.md`** — it is the program's first real-GFM
   result and it is currently orphaned in `results/`.
5. **Then** pick the next GFM, via `REAL_GFM_CIRCUIT_DISCOVERY_PROTOCOL.md`.
