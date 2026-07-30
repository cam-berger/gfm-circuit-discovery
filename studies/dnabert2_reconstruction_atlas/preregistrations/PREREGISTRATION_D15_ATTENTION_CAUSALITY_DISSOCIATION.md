# D15 preregistration — regional information density vs the ALiBi-forced attention prior

> **Tier: T1 — frozen protocol.** Frozen 2026-07-29; Amendment 1 pre-outcome.

> **Status: FROZEN 2026-07-29**, after the §10 design red-team and the §7 development pilot, and
> **before any confirmation-split scoring**. Sections §2–§3 analyse already-open artifacts and fix
> the hypothesis; §4 onward is the confirmatory protocol.

**Track B of the DNABERT-2 reconstruction atlas.** D14 closed the *what* of D12's TSS polarity
(not arrangement). This asks the *how*: does the promoter classifier's attention go where its
evidence is?

---

## 1. Intent — the one sentence this must answer

> **Does the sequence region the promoter classifier's dominant carrier heads attend to coincide
> with the region that behaviourally carries the promoter decision?**

The record has a live contradiction this bears on: D8's regional `L11H2/L11H5` value swap was null at
**every one of 30 tiles** (largest effect `+0.00124` against a `+0.005` SESOI) while D6/D7 show the
same pair is a replicated causal core with value-payload dependence. If attention and evidence do
not coincide, that null is explained instead of left open.

**Failing to resolve the route is an acceptable outcome. Reporting a route the evidence does not
license is not.**

---

## 2. Architectural deductions — ARCHITECTURE, NOT EVIDENCE

Zero forward passes. These constrain the design and the interpretation; none is counted as a result.

DNABERT-2 is MosaicBERT-derived with **symmetric ALiBi and no position embeddings**
(`bert_layers.py:394`, `relative_position = torch.abs(memory_position - context_position)`; the
embedding block is commented `# no position embeddings! ALiBi`). The classifier reads
`hidden_states[:, 0]` (`bert_layers.py:501`) — `[CLS]` at window index 0, i.e. **TSS-relative −249**.
Bias is `−m_h·|i−j|` with the twelve slopes fixed by `_get_alibi_head_slopes` (`bert_layers.py:372`).

Over the 384-locus cohort the mean window is **63.79 BPE tokens**; the TSS sits near token 50.
Effective reach `1/m_h` spans **1.4 tokens (H8) to 256 tokens (H7)**.

**The consequence that governs every interpretation below.** `far_upstream` is defined as
`(−249, −50)` — 200 of 300 nt, and the band *nearest* `[CLS]`. The ALiBi-only prior therefore places
**49%–89% of every head's `[CLS]` mass in `far_upstream`, for all twelve slopes**:

| head | slope | prior `special` | prior `far_upstream` | prior `prox_up` | prior `tss` | prior `prox_dn` | prior `dist_dn` |
|---|---:|---:|---:|---:|---:|---:|---:|
| H8 | 0.7071 | 0.507 | 0.493 | 0.000 | 0.000 | 0.000 | 0.000 |
| H0 | 0.5000 | 0.394 | 0.607 | 0.000 | 0.000 | 0.000 | 0.000 |
| H2 | 0.1250 | 0.118 | 0.877 | 0.004 | 0.000 | 0.000 | 0.001 |
| H5 | 0.0156 | 0.034 | 0.731 | 0.119 | 0.011 | 0.022 | 0.083 |
| H7 | 0.0039 | 0.032 | 0.661 | 0.146 | 0.015 | 0.029 | 0.117 |

**A head's far-upstream attention dominance is architecture-forced and is NOT evidence of a content
preference.** Every attention statement below is reported as *measured minus prior*, never as raw
measured mass.

---

## 3. What the already-open artifacts show — hypothesis-generating, not the test

Committed results and the **development** split only.

### 3a. The dominant carriers do no content-driven source selection

Measured `[CLS]` band mass **minus** the ALiBi-only prior (head atlas, confirmation split):

| head | slope | excess `far_up` | excess `prox_up` | excess `tss` | excess `prox_dn` | excess `dist_dn` | necessity |
|---|---:|---:|---:|---:|---:|---:|---:|
| **L11H2** | 0.1250 | **+0.046** | +0.000 | +0.000 | +0.000 | +0.000 | **0.3154** |
| **L11H5** | 0.0156 | **+0.032** | +0.002 | +0.000 | +0.001 | −0.003 | **0.2084** |
| **L9H7** | 0.0039 | **−0.525** | **+0.401** | **+0.051** | **+0.082** | +0.021 | 0.0169 |
| L10H5 | 0.0156 | −0.143 | +0.026 | +0.003 | +0.005 | +0.009 | 0.0080 |
| L4H6 | 0.0078 | +0.024 | −0.051 | −0.003 | −0.004 | +0.059 | 0.0294 |

L11H2 and L11H5 — the replicated causal core (D6/D7), top two by exact patch — sit **essentially at
their positional prior**: no measurable content-driven source selection. The one head that
redirects strongly onto the TSS neighbourhood, **L9H7**, contributes about **19× less**.

### 3b. Head reach does NOT predict causal importance — a null, and a failed prediction of mine

Within-layer Spearman of `log10(1/m_h)` against exact-patch effect, all 144 heads:

| split \| endpoint | mean within-layer ρ | layers positive | sign-test p |
|---|---:|---:|---:|
| development \| necessity | 0.142 | 6/12 | 1.000 |
| development \| sufficiency | 0.213 | 8/12 | 0.388 |
| confirmation \| necessity | 0.284 | 9/12 | 0.146 |
| confirmation \| sufficiency | 0.062 | 7/12 | 0.774 |

**Nothing qualifies.** The D14 prereg §2 predicted that only long-reach heads could carry downstream
evidence to `[CLS]`. That prediction **fails** and is recorded rather than dropped.

A post-hoc observation — that D5's frozen eight-head set is enriched for long-reach head indices
(mean `log10(1/m)` 1.618 vs null 1.104, exact p=0.0232) — is **downgraded**: the statistic was chosen
after seeing the head list, and the better-powered within-layer test above is null.

### 3c. Behaviourally the per-nucleotide effect is concentrated near the TSS

D8's development 30-tile `mask` scan, aggregated per band, cluster-bootstrapped over homology clusters:

| band | nt range | tiles | mean per-tile target-specific loss | 95% CI | sign-flip p |
|---|---|---:|---:|---|---:|
| `far_upstream` | −249:−50 | 20 | +0.0307 | [−0.0128, +0.0745] | 0.167 |
| `proximal_upstream` | −50:−2 | 5 | −0.1004 | [−0.2156, +0.0141] | 0.091 |
| `proximal_downstream` | +3:+13 | 1 | +0.2678 | [+0.0926, +0.4494] | 0.0028 |
| `distal_downstream` | +13:+51 | 4 | +0.1254 | [+0.0486, +0.2035] | 0.0015 |

### 3d. **The pilot refuted my first hypothesis. Recorded, not quietly replaced.**

I began this study expecting to show the carriers attend to a region that is **inert**. The
development pilot (§7) **refutes that**: masking the entire far-upstream band (30 tokens, ~163 nt
covered) costs `+0.4635`, against `+0.5663` for the 7-token signal window. Nineteen far-upstream
tiles at `+0.031` each are not individually detectable, but they are **not collectively nothing** —
`5 ×` a far-upstream window is `+0.606` against a measured band effect of `+0.4635`, i.e. real and
somewhat subadditive (77% of additive), not saturated and not absent.

**The surviving, weaker, correct claim is about density, not inertness:** per masked token the signal
window is `0.0809` and a far-upstream window `0.0173` — a **4.67× density ratio** — while the
far-upstream region, being five times larger, contributes a comparable *total*. The question this
study can answer is therefore whether **attention tracks information density**, and the answer
§3a points to is that it tracks the architectural prior instead.

### 3e. Cohort provenance — the two sides come from different cohorts

Attention and per-head causal effects come from the **head atlas** cohort (D5/D6). Behavioural mask
effects come from the **source-localization** cohort (D8/D12/D14). These are disjoint sets of 1-Mb
GENCODE blocks built to the same target/pseudo-TSS specification, so the comparison is
**population-level across two identically-constructed cohorts**, never within-locus. Scope limit, §8.

---

## 4. Estimands

All quantities are **target-specific losses** in classifier-logit-margin units:

```
TSL(arm) = [margin(target_clean) − margin(target_masked)]
         − [margin(control_clean) − margin(control_masked)]
```

Each arm masks a contiguous nucleotide window at a **fixed, exactly equal BPE token dose** in target
and control, with total nucleotide cover matched to ≤ 8 nt:

| arm | nominal | TSS-relative | token dose | role |
|---|---|---|---:|---|
| `S` | 240:280 | **−9:+31** | 7 | the behaviourally implicated region |
| `F1..F5` | 0:40 … 160:200 | −249:−209 … −89:−49 | 7 each | far-upstream comparators |
| `U` | 200:240 | −49:−9 | 7 | specificity check |
| `B` | 0:200 | −249:−49 | 30 | whole far-upstream band, over-dosed |

**E_reg — primary, dose-matched.** Per locus, `TSL(S) − mean_w TSL(F_w)` over available far-upstream
windows (≥1 required). Equal dose, equal width, equal contiguity; **only the region differs**.

> **E_reg is a REGIONAL contrast, not a positional one.** It cannot separate "this position matters"
> from "the composition that happens to live at this position matters" — precisely the question
> D14's P2 left open at declared-insufficient power. Naming it positional would repeat the estimand
> drift the red-team caught in D14. See §8.1.

**Why the `[MASK]` objection does not apply to E_reg.** Both sides mask exactly 7 tokens with matched
cover, so any artifact that is a function only of "seven tokens were replaced by `[MASK]`" cancels
**exactly** in the difference. This is a structural argument, not an empirical bound.

**E_band — secondary, declared underpowered (§7).** Per locus, `TSL(S) − TSL(B)`; dose 7 against
dose 30. Retained because it is the direct test of §3d, and its development estimate already refutes
the inert reading.

Inference: cluster bootstrap over `homology_cluster` (8000 draws, seed 20260729); add-one two-sided
cluster sign-flip permutation (9999 draws); Holm over `{E_reg, E_band}`.

---

## 5. Frozen decision rule — written before any confirmation scoring

**SESOI = 0.25 logits**, set on scientific grounds and not from the pilot: the replicated clean
promoter margin is `+1.89` (D4 confirmation), and a regional contrast worth under one-eighth of the
entire decision is not a meaningful asymmetry.

An endpoint **qualifies** iff all three hold: mean ≥ +0.25; cluster-bootstrap lower bound > 0;
Holm-adjusted sign-flip p < 0.05.

| E_reg | verdict |
|---|---|
| qualifies | **REGIONAL DENSITY CONFIRMED.** At equal dose, width and contiguity, the −9:+31 region carries materially more of the promoter decision than far-upstream windows do. Combined with §3a this licenses *"attention allocation does not track information density"* — and nothing stronger. |
| mean ≥ 0.25 but interval includes zero, or Holm fails | **INCONCLUSIVE.** Report at achieved precision; no claim. |
| fails, interval excludes the SESOI | **REFUTED.** No regional density difference at 40-nt scale; D12's 10-nt polarity does not generalise. |
| mean ≤ −0.25 with a zero-excluding interval | **REVERSED.** Far-upstream is denser; §3a's reading is wrong and direct reading is supported. |

`E_band` is reported with its interval and its pre-declared power. **It cannot qualify a claim on its
own** and is not permitted to convert an E_reg failure into a positive finding. If `M3` (§6) fails,
`E_band` is *uninterpretable*, not *failed*.

---

## 6. Memorials and controls — each must be able to fail, on the measurement path

| id | check | scale | can it fail? |
|---|---|---|---|
| **M1** clean anchor | mean clean target−control margin ≥ +1.5 (D4 confirmation: +1.89; dev pilot: +1.84) | population | **yes** — wrong checkpoint, wrong cohort, or a flipped margin sign fails it |
| **M2a** equal dose | target token count == control token count exactly, every arm | mechanics | **instrument-enforced** — `explicit_token_mask_effects_for_batch` raises on unequal dose. **FLAGGED architecture-forced; not counted as evidence** |
| **M2b** cover **balance** | per-arm **mean** `cover_target − cover_control` within ±2.0 nt | mechanics | **yes** — see Amendment 1 |
| **M3** non-degeneracy | `\|mean TSL(B)\|` (dose 30) > `mean \|TSL(F_w)\|` (dose 7) | population | **yes** — the floor/ceiling guard on the over-dosed arm |
| **M4** batch invariance | a 16-locus probe rescored at batch size 4 reproduces every arm value to < 1e-4 | mechanics | **yes** — this class of check caught real bugs in D12 |

### Amendment 1 — pre-outcome, made after the smoke gate and before any confirmation scoring

The smoke run (development, n=24, `promoter_region_density_smoke_v1.json`) exposed two defects in
the memorial specification above. Both are fixed here; neither was informed by any confirmation-split
value.

1. **M2b as originally written could not fail.** It read "every qualified pair
   `|cover_target − cover_control| ≤ 8 nt`" — but the qualification filter *drops* every pair above
   8 nt, so the check is satisfied by construction. This is exactly the vacuous-memorial failure the
   D14 red-team caught, recurring. **M2b is replaced** by a quantity the filter does not constrain:
   the **per-arm mean** cover difference, tolerance ±2.0 nt. A systematic BPE asymmetry between
   target and pseudo-TSS sequence would move it. The per-pair bound is still reported, explicitly
   flagged as forced.
2. **M1 and M3 are population-mean memorials and were being gated at smoke scale.** Both failed at
   n=24 (M1 `+1.176` vs the `+1.5` bar; M3 `0.216` vs `0.221`) while both pass comfortably on the
   full 192-locus development split (M1 `+1.8412`; M3 `0.4635` vs `0.120`). Gating mechanics on a
   population mean is a category error. They are now **evaluated only when the split supplies ≥ 100
   loci**, and are reported as `evaluated: false` otherwise. The confirmatory run supplies 192, so
   both are live gates there.

Mechanics validated by the smoke and unaffected: **M4 batch invariance passed at 3.58e-06** against
a 1e-4 tolerance, and all eight arms scored with supply 0.63–1.00.

**N1 — null control, on the measurement path.** `TSL(F2) − TSL(F4)`: two mid-band far-upstream
windows with near-identical ALiBi prior. Preregistered expectation `|N1| < SESOI`. Development:
**−0.0539**. A large `N1` would mean the instrument reports regional differences between windows that
should not differ, and would invalidate E_reg.

**Specificity check (directional, not an endpoint).** `TSL(S) − TSL(U)`, where `U` = −49:−9 was
*negative* per tile in development. Prediction: positive. A failure is reported but does not change
the §5 verdict.

**Deliberately not used as a memorial:** a zero-dose no-op returning exactly 0.0 — forced by
construction, validates nothing (the R21 lesson from the sibling study).

---

## 7. Power — from the development pilot (open split, no seal spent)

n=192 development loci; `homology_cluster` is 1:1 with locus here, so clustering is equivalent to
independent loci and is retained for consistency.

| arm | supply | mean cover (target/control) nt | mean TSL |
|---|---:|---|---:|
| `S` | 188/192 (0.979) | 35.21 / 35.86 | +0.5663 |
| `F1`…`F5` | 185–191/192 (0.96–1.00) | ≈35.2 / ≈35.6 | +0.264, +0.055, +0.092, +0.114, +0.077 |
| `B` | 124/192 (0.646) | 162.67 / 162.67 | +0.4635 |

| endpoint | n | sd | SE | MDE(80%, two-sided) | dev estimate | power at dev estimate | power at SESOI 0.25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **E_reg** | 188 | 1.815 | 0.132 | **0.371** | **+0.4495** | **≈92%** | **≈47%** |
| E_band | 124 | 3.459 | 0.311 | 0.870 | +0.0254 | — | **≈12%** |

**Disclosed before the run:** `E_reg`'s MDE (0.371) **exceeds its own SESOI (0.25)**. It is powered
for a development-scale effect and **underpowered at the SESOI boundary**; an inconclusive result
there must not be read as a null. **`E_band` is declared underpowered** (12% at the SESOI) and can
only be reported descriptively — the same discipline D14 applied to its P2.

Development `E_reg` is **not skew-inflated**: mean +0.4495, median +0.4238, 10%-trimmed +0.4095,
113/188 positive, cluster-bootstrap CI [+0.1966, +0.7141]. A 5/95 winsorised variant gives +0.4221
and is reported as robustness only; the **untrimmed mean is the preregistered statistic**.

**Rejected variance lever, recorded so it is not retried:** dropping `F1` (the `[CLS]`-adjacent
window, mean +0.264 against +0.055…+0.114 for `F2`–`F5`) *raises* E_reg to +0.4839. Including F1 is
therefore conservative and it stays in.

---

## 8. Scope — what this study cannot license whatever it finds

1. **E_reg is regional, not positional.** It does not separate position from the composition
   resident at that position. D14's P2 left that open at insufficient power and this design does not
   close it.
2. **No mediation claim.** Attention mass is descriptive; the §3a/§4 pairing crosses evidence rungs
   deliberately. This shows no component causally mediates the `S` effect.
3. **"Relay" remains an inference, not a measurement.** Nothing here patches a path or identifies a
   relay position or layer.
4. **The `[MASK]` cancellation argument covers E_reg only** — it does not cover `E_band`, whose two
   sides carry doses 7 and 30.
5. **Cross-cohort comparison** (§3e).
6. **No biological claim** — no motif, no CpG-island mechanism, no transcriptional function in cells.
7. **§3 is hypothesis-generating** on reused, already-open data. Only §4–§5 are confirmatory.

---

## 9. Seal disclosure

The confirmation split of `promoter_source_localization_v1` was opened by D8 (per-tile `mask`, tiles
21–23, 25–29), D12 (E1–E4) and D14 (three arms at the E2 tiles). This is a **partial seal**: the
40-nt/200-nt span instrument has never been run on these loci and neither `E_reg` nor `E_band` has
ever been estimated on them. **Per-tile confirmation values for tiles 0–20 and 24 remain sealed and
this study does not read them** — the span instrument is a distinct intervention and does not consume
that seal.

---

## 10. Design red-team — completed before freezing

Five lenses, run against the draft. Three changed the design.

1. **Identifiability — CHANGED THE DESIGN.** The draft called the primary endpoint *positional*.
   The `S` window straddles a GC-rich CpG-island core; far-upstream windows do not. The contrast is
   therefore confounded with regional composition, which D14 explicitly could not resolve. Renamed
   `E_pos → E_reg` and scoped in §4/§8.1. *This is the same estimand-drift failure the D14 red-team
   caught; it recurred and was caught again.*
2. **Hypothesis survival — CHANGED THE DESIGN.** The draft's §3d asserted far-upstream was plausibly
   inert and built `E_band` to confirm it. The pilot refuted this. Rather than delete the hypothesis,
   §3d now records the refutation, the claim was weakened from *inertness* to *density*, and
   `E_band`'s role changed from confirmatory to descriptive-with-declared-underpowering.
3. **Architectural confound — CHANGED THE DESIGN.** `F1` (−249:−209) abuts `[CLS]` and carries by
   far the largest ALiBi prior weight of any window; its effect (+0.264) is 2–5× the other far
   windows. Excluding it would have inflated E_reg (+0.4839). It is retained, and §7 records the
   direction so the choice cannot be relitigated post hoc.
4. **Statistics — no change, one disclosure.** `homology_cluster` is 1:1 with locus on this split, so
   the cluster bootstrap is not buying independence it does not have; disclosed in §7. Holm covers
   only the two gated endpoints. The winsorised variant is robustness-only to avoid two bites.
5. **Memorial quality — no change.** M2a is forced by the instrument and is flagged rather than
   counted (R21 discipline). M1, M2b, M3, M4 and N1 can each fail, and N1 sits on the measurement
   path: if two equivalent far-upstream windows differ by more than the SESOI, E_reg is not
   measuring what it claims.
