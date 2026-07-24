# PREREGISTRATION — G1: cross-example / cross-context transduction of the donor-side delivered state

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**STATUS: FROZEN 2026-07-23 (owner-authorized) — confirmatory pending on the sealed H_conf split.**
Design red-team (4-lens) DONE and its 9 must-fixes folded into the runner; discovery/H_dev MDE gate DONE;
tunables sealed from the H_dev pilot. See the FREEZE ADDENDUM immediately below. The original draft text
that follows is preserved as the design of record; where the addendum resolves a *(to be sealed)* /
*(owner)* item, the addendum governs.

---

## FREEZE ADDENDUM (2026-07-23) — resolutions, red-team fixes, sealed values

**Red-team of record:** `redteam/REDTEAM_BRIEF_G1.md` + `redteam/REDTEAM_SYNTHESIS_G1.json` — house 4-lens
adversarial design red-team on the concrete design + H_dev pilot. Disposition **FIXABLE_THEN_RUNNABLE,
survivor=YES, 0 FATAL, 9 must-fix**. All 9 folded into `scripts/splicebert/run_g1_cross_example_transduction.py`
(its `RED-TEAM FIXES` docstring block, F1–F9) and re-validated (smoke PASS; H_dev pilot re-run clean).

**The load-bearing identification fix (F1, the crux).** The pilot Δ_spec = τ(REAL) − τ(MEAN) < 0 is
substantially a *norm-wastage denoising* signature (a single high-variance source vs its low-variance
population mean), NOT by itself proof the example-specific content is uninformative. The **parallel/perp
decomposition** is added as the adjudicator and **gates the GENERIC verdict**: decompose s_j into s_∥ (along
the generic axis ŝ̄) + s_⊥ (orthogonal, example-specific); inject s_⊥ **alone at low α = 0.25** (linear
regime — `margin_hard` saturates at α=1, P3.3) vs a norm-matched random-perp control, with its own M8 OOD
gate. **s_⊥ sub-SESOI ⇒ off-axis example content is causally inert for the donor margin ⇒ GENERIC;
s_⊥ restoring ⇒ example-specific transduction survives.** H_dev result: `PERP_MATCHED = −0.013
[−0.042,+0.017]` ⊂ ±SESOI (inert), `PAR_NAT = +0.187` (carries the transfer).

**Scope of any null (F4 / R24).** A GENERIC / NO-TRANSDUCTION verdict bounds ONLY the **C16/C26 delivered
FINAL-RESIDUAL state**, via **additive** injection through the **short-source rank-8 U_D**. It does NOT bound
C17 (routing), C18 (transporters), or C23 (address+payload joint) — the estimand intervenes on none of them —
nor earlier-layer or non-additive example-specificity. Verdict labels carry the `_C16C26_finalresidual`
suffix to enforce this. This does not contradict C26; it is C26's interventional corollary.

**§11 owner-decision resolutions.** (1) **Δ_spec is primary**, understood post-red-team as denoising-confounded
and adjudicated by the PERP arm (the ROADMAP-literal "above matched-random" reads positive but is the
C16/C26-anticipated result). (2) **Additive** injection (algebra+empirics: additive ≡ component-replacement
at α=1 SELF to 3.7e-9). (3) **`SESOI_spec = 0.05` normalized SEALED**; scale relation to the 0.05-nat
absolute floor documented in the runner (with pooled E[coup]≈0.41, 0.05 normalized ≈ 0.02 nat — stricter, so
the dual `|Δm|≥0.05 nat AND |τ|>0.05` threshold holds); H_dev Δ_spec −0.32 is ≈6× SESOI. (4) **Atlas
internal-validation targets ONLY**; partner-selection val NOT reused; **sealed chr8/9 NOT opened** (verified
absent from both Atlas splits). (5) Contexts (G1-E) **DEFERRED ENTIRELY** — the 6-context max-T secondary
family is not reported until per-context SELF/U_D-fit gating is implemented (red-team Lens3-F2); pooled
primary unaffected. (6) **Single-pair short-intron sources** (dist∈[60,150]). (7) **No A9 acceptor-side
re-run**. (8) Red-team **DONE** (this). (9) Tolerances SEALED: **M3 |SELF−0.875|<0.10**, **M8 band
‖H_inj‖/‖H_cl‖∈[0.5,2.0]**, `--max-src 8`, dose grid endpoints as drawn, low-α = 0.25. (10) CLAIMS wording:
the C16/C26-scoped GENERIC statement above.

**Confirmatory protocol (frozen).** H_conf (untouched validation half, ~551 genes/1012 packets); **≥8 seeded
source draws that resample sources WITH REPLACEMENT** (F3 — the old n_src==|src_all| made draws identical, the
A9/C30 R22 blind spot) with `n_src=200 < |src_all|`; across-draw SD(Δ_spec) reported (instability trigger
SD > 0.5·SESOI_spec). Global PASS = M1,M2,M3(sealed tol),M4(computed gene-disjoint),M5,M6,M7(per-cohort),
M8(OOD band),M9(cohort hash) + planted-positive + cross-gene permuted-label FPR. POS_SHAM=0 labeled FORCED
(F2), excluded from the AND. SRC_SHAM role narrowed to the sham-frac ratio-exclusion (its absolute-τ
equivalence bar is unreachable at this precision, F8); no-decoy fallback pairs EXCLUDED (not zero-filled).

**Pilot (H_dev) provisional verdict: `GENERIC_READOUT_AXIS_C16C26_finalresidual`** (all gates pass;
`results/splicebert_coupling/g1_cross_example_transduction_pilot.json`). H_conf confirms or refutes.

---

Motivation of record: `docs/REVIEW_2026-07-22.md` §5; requirement of record: `docs/METHODOLOGY.md:85-88`;
design of record: `MECHANISM.md` §2 ("Cross-context transduction") and `docs/ROADMAP.md` Track 1.

Provenance to be pinned at freeze: SpliceBERT.1024nt sha256 `2ad91428…0149d`, post-LN BERT-MLM 6L×16H,
hidden 512, absolute learned position embeddings (P=1026). Env `gfm-thrust1`; compute **MPS** (no CUDA).
Deterministic cohorts (`PYTHONHASHSEED=0`, `sorted(set())`).

---

## R-block — what this study does NOT get to claim, decided before it runs

Two near-forced results are foreseeable from existing artifacts and must be labelled at design time, in
the R21/R23 tradition, so they cannot later be read as evidence.

- **Injecting a `U_info`-aligned vector will raise the donor margin, near-certainly.** C26 records rank-1
  subspace overlap **0.970** between independently derived `U_info` bases on disjoint gene halves, and C16
  records rank-1 in-sample recovery ≈0.74. A subspace that reproduces across disjoint genes at 0.97 is, to
  that extent, a **gene-generic readout axis**. A positive "cross-example injection moves the target
  margin" result is therefore **substantially anticipated by C16/C26** and is *not* by itself evidence
  that an example-specific state was transferred. It is reported as a plumbing/generality demonstration
  and **flagged**, never interpreted as information transduction.
- **The "non-target-position" control is ARCHITECTURE-FORCED at the primary injection site.** The MLM head
  is position-wise and `logits = cls(hidden_states[-1])` is bit-exact
  (`src/gfmcd/splicing/unembed.py`), so injecting at any non-donor final-residual position changes the
  donor margin by **exactly 0** by construction. The same holds for masked-MLM health at other positions.
  Both are **reported as LABELED FORCED** (R21 class) and neither may gate or support a conclusion. They
  become informative only in the deferred earlier-layer variant (§10).

**Consequence for the design:** the discriminating estimand is not *"does the transplanted state move the
target"* but *"does the **example-specific** part of the transplanted state add anything over a norm-matched
gene-generic axis"*. That contrast (§1, G1-B) is the primary.

---

## Relation to prior evidence (read before assuming novelty)

- **A5/C26** derived `U_info` on one gene half and tested it on the disjoint half, reaching held-out rank-8
  sufficiency **+0.875** (`a5_uinfo_replication.json`, n=400, 320 genes). That transfers the **basis** across
  genes; the injected **content** is still the target example's own clean component. It is a within-example
  patch read through a cross-gene basis.
- **A9/A9b/C30** already ran a genuinely **cross-gene interchange** — delivering example *j*'s clean acceptor
  payload through example *i*'s held-clean address, with frozen calipers, reciprocal swaps, capped source
  reuse, and disjoint gene splits. So the repository is **not** free of cross-example intervention, and the
  review's "nearly all … intra-example" framing should be read as *nearly*, not *all*.
- **What is still missing, and what G1 adds:** (i) transduction of the **delivered donor-side state**, not
  the acceptor-side value payload; (ii) a **dose grid** rather than a single unit swap; (iii) a **source pool
  and subspace derived on a gene pool that never contains a target**, with no target-side clean run entering
  the primary estimator; (iv) targets drawn from **non-canonical sequence contexts** (Atlas clustered,
  multi-pair, cassette, complex) rather than only canonical short introns; (v) explicit separation of
  example-specific content from a gene-generic readout axis (the R-block point, which A9's
  target-normalized transfer `T(i←j)` does not address).

---

## 0. INTENT (north star — re-read at every gate)

> Does the donor-side delivered state identified by C16–C18/C23/C26 behave as a **transferable
> information-bearing state** — derived on genes the target never contributed to, injected into a
> gene-disjoint and context-mismatched target, moving that target's native donor margin in a
> dose-ordered, source-specific way — or does intra-example patching succeed only because source and
> target are the same example, with any cross-example effect reducing to a push along a gene-generic
> readout axis?

Model-native observable: the target's masked-donor margin under injection at the final donor residual,
normalized by that target's own partner-corruption coupling. This is a **generality** study of an existing
mechanism. It names no payload semantics (R22 stands) and adjudicates no minimality question (C33 stands).

---

## 1. ESTIMANDS

**Populations.** `D_src` = source/derivation genes; `H_dev` = target development genes; `H_conf` =
untouched confirmatory target genes. Pairwise gene-disjoint; sealed chr8/9 **not opened** (§7).

**Objects.**
- `U_D` = `top_k_subspace(rows, 8)` on `rows = H_cl − H_co` at the donor, **rows drawn from `D_src` only**
  — the exact C16 definition, projected onto and never refit on any target (A5/A9b discipline).
- **Transduced state** `s_j = project(H_cl,j − H_co,j, U_D)` for source example `j ∈ D_src`.
- **Gene-generic reference state** `s̄ = project(mean_j (H_cl,j − H_co,j), U_D)` over `D_src`, rescaled to
  `‖s_j‖` per pairing — the same axis, stripped of example-specific content.
- **Injection** at dose `α`: `H_i(α) = H_co,i + α · s`, evaluated as `margin_hard(cls(H_i(α)), bid)`.
  At `α=1` with `s = s_i` this is algebraically identical to `patch_component(H_co,i, H_cl,i, U_D)`, i.e.
  exactly the A5/C26 estimator (memorial M3).
- **Coupling denominator** `coup_i = m_cl,i − m_co,i` (partner AG→CC at `[a−2,a−1]`, real bases, never
  `[MASK]`), per target.
- **Transduction** `τ(arm, α) = E_i[ m_i(α) − m_i(0) ] / E_i[ coup_i ]` — a **ratio of means** with a
  two-way (source-gene × target-gene) dyadic bootstrap; **never mean-of-ratios**. Absolute-nat numerator
  `E_i[m_i(α) − m_i(0)]` reported alongside, always.

**Arms** (all norm-matched to the REAL arm; matching is memorial M5):

| Arm | Injected `s` | Role |
|---|---|---|
| REAL | `s_j`, `j` matched to `i` on the §2 calipers | treatment |
| MEAN | `s̄` rescaled to `‖s_j‖` | **the discriminating null** — same axis, no example content |
| SELF | `s_i` (target's own) | **reference ceiling**, reproduces C26 (M3) |
| RANDOM | norm-matched random rank-8 subspace component | matched-random floor (C16 anchor ≤0.05) |
| SCRAMBLE | `U_D`-coordinates of `s_j` permuted across the 8 basis dims, re-embedded | norm- and subspace-preserving content destruction |
| SRC-SHAM | `s_j^{np}` from `j`'s **non-partner** corruption contrast (C2 behaviour ≈ +0.007) | source-side content null |
| POS-SHAM | `s_j` injected at a distance-matched non-donor position | **LABELED FORCED = 0** (R-block) |

**Primary estimands.**
- **G1-A (existence; anticipated, flagged):** `τ(REAL, α=1)` and its dose ordering over the frozen grid.
- **G1-B (source-specificity; THE PRIMARY):** the paired within-target contrast
  `Δ_spec = τ(REAL) − τ(MEAN)` at `α=1`. Falsified by `Δ_spec` CI ⊂ ±SESOI_spec.
- **G1-C (floor separation):** `τ(REAL) − τ(RANDOM)` and `τ(REAL) − τ(SCRAMBLE)`, paired.
- **G1-D (transduction efficiency; descriptive):** `τ(REAL)/τ(SELF)` with dyadic CI. Reported as a
  conditional ratio, **not** as a fraction of the mechanism.
- **G1-E (cross-context):** `τ(REAL)` and `Δ_spec` per Atlas context, and the transfer contrast
  `τ(context c) − τ(single-pair)` — sources are canonical single-pair short introns throughout, so this is
  the context-generality test.

**What falsifies the study's motivating claim.** If `Δ_spec` is an equivalence-null and `τ(REAL) −
τ(RANDOM)` is also an equivalence-null, the donor-side state does **not** transduce across examples and
C16–C18/C23's interventional reach is bounded to intra-example patching — a material scope correction that
this design must be able to deliver.

---

## 2. METHOD, cohort, matching

- **Cohort.** Targets from the audited Head–Context Atlas v7 matched cohort (`ATLAS_COHORT_V1.md`; contexts
  single-pair / donor-cluster / acceptor-cluster / multi-pair / cassette / complex; internal-validation
  split = 4,019 packets, 1,486 genes, gene- and sequence-window-disjoint from discovery; mechanical audit
  `atlas_context_matched_v7.audit.json` PREFLIGHT_PASS, 0 failures). Sources from the Atlas discovery split,
  restricted to canonical single-pair short introns so the source distribution is the distribution on which
  A0→A12 characterized the circuit.
- **Source↔target matching (frozen calipers, to be sealed from a `D_src`/`H_dev` pilot):** site type, native
  `coup`, donor–acceptor distance, coding status, local composition (GC / pyrimidine fraction), absolute
  packet position, and intervention norm `‖s‖`. Target: max absolute SMD ≤ **0.10** (the Atlas standard).
  Reciprocal pairing where available; **cap source reuse** (`--max-src`, to be sealed) — the A9 discipline
  that the R22 dyadic requirement is built on.
- **Reused machinery, no new kernel:** `splicing.circuit.CouplingTask` (coupling, margins, health),
  `splicing.subspace` (`top_k_subspace`, `project`, `patch_component`, `random_subspace`,
  `energy_spectrum`), `splicing.unembed.margin_hard` / `margin_smooth`, `circuit.gboot_ratio`, the Atlas
  context/matching modules (`splicing.atlas_context`, `splicing.context_controls`). Adapter loaded
  `eager=True`.
- **Dose grid (pre-saturation; to be sealed on `H_dev`):** `α ∈ {−1.0, −0.5, −0.25, 0, +0.25, +0.5, +1.0,
  +1.5}`. `margin_hard` saturates (P3.3 recorded this), so monotonicity is preregistered **only over the
  pre-saturation range**, with the saturation criterion fixed on `H_dev` and `margin_smooth` reported as the
  continuous secondary readout.

**EXCLUDED by design:** no refitting of `U_info` on any target; no selection of α, arm, caliper, context,
or threshold after `H_conf` is opened; no earlier-layer injection (§10, deferred).

---

## 3. MEMORIALS (on the measurement path; each CAN fail unless flagged FORCED)

| # | Memorial | Passes iff | Note |
|---|---|---|---|
| M1 | `margins_head_reproduces_cached` | `margin_hard(cls(H_co))` reproduces the cached full-forward `m_co` < **1e-6** | the load-bearing anchor: if the head-on-captured-residual is not the true readout, every τ is wrong. A5 achieved 0.0 exactly. **Failable, on-path** |
| M2 | `zero_dose_identity` | `τ(arm, α=0) == 0` exactly | catches indexing/batch/broadcast bugs; **weak** — near-trivial once M1 passes. Plumbing lint, not evidence |
| M3 | `self_transduction_reproduces_C26` | SELF arm at α=1 reproduces C26's held-out rank-8 sufficiency **+0.875** within a frozen tolerance *(to be sealed)* | exact-arithmetic reproduction of a published quantity through the identical estimator; **genuinely failable**, and the strongest memorial here |
| M4 | `no_target_leakage_into_U` | no `H_dev`/`H_conf` gene contributed a row to `U_D`; asserted by gene-set hash | leakage would manufacture G1-B |
| M5 | `norm_matching_exact` | `‖s‖` equal across REAL/MEAN/RANDOM/SCRAMBLE/SRC-SHAM to < **1e-9** per pairing | a norm-mismatched control manufactures the contrast |
| M6 | `random_direction_floor` | RANDOM `τ ≤ 0.05` (C16 anchor) | specificity floor; failable |
| M7 | `denominator_robust` | per-context `E[coup]` dyadic CI excludes 0 **and** `E[coup] ≥ 0.20` (A5's `positive_control_partner_coupling`) | else that context is reported **difference-scale only**, no `τ` |
| M8 | `ood_band` | `‖H_i(α)‖ / ‖H_cl,i‖` inside a frozen band and donor-logit entropy inside a frozen band, in every interpreted arm×dose *(bands to be sealed on `H_dev`)* | the only health check that CAN fail here |
| M9 | `determinism` | `PYTHONHASHSEED` pinned, `sorted(set())`, cross-seed cohort identity | IC1 Amendment A2 lesson |
| **F1** | `nonsite_mlm_health` | unchanged at non-donor positions | **LABELED FORCED** — position-wise head; cannot fail |
| **F2** | `pos_sham_zero` | POS-SHAM `τ == 0` | **LABELED FORCED** — position-wise head; cannot fail |

---

## 4. CALIBRATION (sensitivity + specificity; hard preflight gates)

**Planted-positive (sensitivity; travels the real channel).** Build synthetic sources
`s* = β · project(H_cl,i − H_co,i, U_D)` for known `β ∈ {0.25, 0.5, 1.0}` and pass them through the identical
injection + estimator path. PASS iff the recovered `τ` is monotone in `β` and the fitted slope CI covers 1
(the A9 `Vplant` construction, which reached `r=0.9998`). A planted positive that does **not** travel the
scoring/decision layer is not accepted — this is the gate class that caught three decision-layer bugs in A9.

**Permuted-label FPR (specificity).** Under the null of no arm difference, permute the REAL/MEAN arm labels
within target and recompute the full G1-B decision (dyadic bootstrap included), ≥200 permutations. PASS iff
empirical FPR ≤ **0.05**. This calibrates the decision layer that declares source-specificity, not merely the
estimator.

**Planted-confound (matching does not do the work).** An arm pairing each target with the source most
**extreme** on the §2 calipers. If `τ` tracks caliper mismatch rather than state content, matching is the
active ingredient and G1-B is not interpretable. Report realized SMDs with the result regardless.

**Behavioural qualification per context.** M7 is run per Atlas context before any interpretation; a context
failing it is reported **instrument-limited**, never null.

---

## 5. IDENTIFIABILITY

- **Gene-generic axis vs example-specific content (the central threat; R-block).** Handled by making
  `Δ_spec = τ(REAL) − τ(MEAN)` the primary, with MEAN norm-matched per pairing. If `s_j` and `s̄` are
  near-collinear across the cohort — pre-declared diagnostic: mean cosine `⟨s_j, s̄⟩/(‖s_j‖‖s̄‖)` and the
  `energy_spectrum` of `{s_j}` — then the example-specific component is small **by construction** and the
  honest verdict is *generic readout axis* (§6), not a null of transduction. Report the cosine distribution
  with the verdict, always.
- **Readout-push vs information transfer.** SRC-SHAM (source built from a non-partner corruption whose
  behavioural effect is ≈+0.007, C2) is the source-side content null. If SRC-SHAM at matched norm reaches a
  material fraction of REAL, the intervention is pushing the readout, not delivering partner information;
  this triggers the §9 stopping rule.
- **Denominator dependence.** All normalized quantities are conditional on `coup`; ratios are never summed,
  never presented as additive fractions (house convention 5), and are reported with their absolute-nat
  numerators.
- **Saturation.** `margin_hard` saturates; monotonicity is claimed only pre-saturation, with `margin_smooth`
  as the continuous check.
- **Held-out ≠ sampling stability (R22).** Every selected caliper, dose, tolerance, and band is set on
  `D_src`/`H_dev` and confirmed on an untouched `H_conf`; **and** the primary is repeated across ≥5
  independently seeded source-pool draws (§7). Within-sample held-out alone is explicitly **not** sufficient.
- **Dyadic dependence (R22).** Sources and targets are both reused, so every CI is a **two-way
  (source-gene × target-gene) dyadic bootstrap**. One-way gene clustering is reported only as a diagnostic
  and never licenses a verdict.
- **Not established by this design:** payload semantics (R22); minimality (C33); binding (R23); any
  earlier-layer transduction locus; anything biological.

---

## 6. DECISION RULE (frozen; result → conclusion)

**SESOI (discovery-fixed, house floor).** Absolute `SESOI_abs = 0.05` nats of donor-margin movement.
Normalized `SESOI_τ = 0.10` **and** `|Δm| ≥ SESOI_abs` — the P3-style dual threshold, so a large ratio on a
tiny denominator cannot pass. Specificity contrast `SESOI_spec = 0.05` normalized *(to be sealed; see §7)*.
Any NULL/equivalence call requires **CI half-width < its SESOI**; otherwise the verdict is *inconclusive*.

**Two families, correctly composed** (P3 discipline): (i) **positive** family — one-sided `H0: effect ≤
SESOI`, max-T dyadic-bootstrap adjusted lower bound > SESOI, over `{Δ_spec, τ(REAL)−τ(RANDOM),
τ(REAL)−τ(SCRAMBLE)}`; (ii) **equivalence** family — TOST at the family-adjusted α, adjusted CI ⊂ ±SESOI,
over `{SRC-SHAM, RANDOM, POS-SHAM(forced)}`. Contexts (§1 G1-E) form a **secondary** max-T family and are
reported, not used to gate the pooled primary. Every arm comparison is the **paired within-target
difference** CI, never non-overlap of two arm CIs.

| Result | Licensed conclusion |
|---|---|
| §3 memorials + §4 gates PASS; `τ(REAL)` dose-ordered above the dual threshold; `Δ_spec`, `τ(REAL)−τ(RANDOM)`, `τ(REAL)−τ(SCRAMBLE)` all lower-bound > SESOI; SRC-SHAM sub-SESOI | **TRANSDUCING STATE.** *"The donor-side delivered state transfers across gene-disjoint examples: a state derived on genes the target never contributed to moves the target's native donor margin dose-ordered and source-specifically, above matched-random, scrambled-source, and gene-generic-axis controls."* Interventional generality established at the tested contexts. |
| `τ(REAL)` above threshold but `Δ_spec` equivalence-null | **GENERIC READOUT AXIS.** *"Cross-example injection moves the target margin, but the transferred content is not distinguishable from a gene-generic readout axis; example-specific information transfer is not established."* Coheres with C26's 0.970 rank-1 overlap. A legitimate, reportable outcome. |
| `τ(REAL) − τ(RANDOM)` equivalence-null | **NO TRANSDUCTION.** The state does not transfer; C16–C18/C23's interventional reach is **bounded to intra-example patching** and `MECHANISM.md` §2 is answered negatively. Requires the achieved MDE to be reported. |
| SRC-SHAM ≥ `--sham-frac` × REAL *(to be sealed, e.g. 0.5)* | **READOUT PUSH, not transduction** — no transduction interpretation; see §9 stopping rule. |
| Any interpreted CI half-width ≥ its SESOI | **Inconclusive** for that quantity. Never reported as a null. |
| M7/M8 fail in a context | That context is **instrument-limited**; excluded from interpretation, reported with its diagnostics. |
| Across-draw SD(`Δ_spec`) > 0.5 × `SESOI_spec` | **Unstable across seeded draws (R22 trigger)** — characterize the instability; no mean claim. |

Mixed or partial patterns → the weakest verdict consistent with the evidence. No verdict may be assembled
from arms that failed their memorials.

---

## 7. POWER / MDE, SAMPLING STABILITY, AND SPLITS

**Variance anchor (measured, not assumed).** `a5_uinfo_replication.json`: held-out rank-8 sufficiency
**0.875, CI [0.853, 0.901]** at n=400 / **320 genes** ⇒ gene-clustered half-width ≈ **0.024** on the
normalized scale ⇒ implied between-gene SD ≈ **0.22**. Projected half-widths, `HW ≈ 1.96 · 0.22/√G`:

| Target genes `G` | HW (one-way, normalized) | HW × 1.4 dyadic allowance |
|---:|---:|---:|
| 320 | 0.024 | 0.034 |
| 700 | 0.016 | 0.023 |
| 1,400 | 0.011 | 0.016 |

Paired arm differences share the target's `H_co` and `coup`, so their SD is bounded above by the marginal SD
under any non-negative arm correlation; the table is therefore conservative for `Δ_spec`. Against
`SESOI_spec = 0.05` normalized, the Atlas internal-validation pool (**1,486 genes**, split into `H_dev` and
`H_conf`) clears the bar with margin at ≈700 confirmatory genes.

**Pre-declared MDE gate (the P3.4 fix — the reason that study died).** Before `H_conf` is opened, compute the
projected dyadic half-width for **every** contrast from `H_dev`. Any contrast whose projected HW ≥ its SESOI
is **declared inconclusive in advance and not run**. No compute is spent on an estimand that cannot reach a
verdict.

**Sampling stability (R22, mandatory).** ≥5 independently seeded draws of the `D_src` derivation set and the
source pool; report per-draw `τ(REAL)`, `Δ_spec`, and the across-draw SD; apply the §6 instability trigger.
A result that reproduces on `H_conf` but not across draws is **not** a result — the R22 lesson, stated in the
prereg so it cannot be relearned after the fact.

**Splits.** `D_src` = Atlas discovery genes (autosomes ∉ {chr8, chr9, chr11, chr16, chr19}). `H_dev` /
`H_conf` = a deterministic disjoint halving of the Atlas internal-validation genes; `H_conf` is untouched
until every gate passes. **Sealed chr8/9 is NOT opened by this study.** Whether the partner-selection val
split (chr11/16/19) may also serve as targets is an open decision (§11) — as drawn here it does not, because
A5/C26 already read it.

---

## 8. COMPUTE FEASIBILITY ON THE MPS BUDGET (honest assessment)

**This design is reachable at MPS budget, and the reason is structural, not optimistic.**

`logits = cls(hidden_states[-1])` is bit-exact on this checkpoint (`src/gfmcd/splicing/unembed.py`
docstring, verified for the donor-anchor work). The primary injection is at the **final** donor residual, so:

- **Forward passes are needed only to capture residuals**, twice per example (clean, corrupt) plus once per
  source for the SRC-SHAM contrast. At ~1,500 targets + ~1,500 sources that is **≈7,500 forward passes** —
  the same order as A5 (n=400) and well below IC1's confirmatory run.
- **The entire arm × dose × draw grid costs MLM-head evaluations only** (512 → 512 → vocab on cached
  residuals), i.e. ~7 arms × 8 doses × 5 draws ≈ 280 head evaluations per target. On cached tensors this is
  seconds-to-minutes of CPU, not model inference.

**Contrast with P3.4, which died on a power wall.** P3.4's estimands were second differences of a
high-variance MLM `ΔL` whose gene-clustered half-width was already 0.0198 nats at n=384, against a 0.02-nat
equivalence bar, at a ≤140-unit budget. G1 differs on all three axes: the outcome is the well-measured
coupling contrast (§7 anchor), the target pool is ~1,486 audited genes rather than ~140 units, and the
expensive dimension (arms × doses) is head-only. G1 is not the same kind of study and must not inherit that
verdict — but it must still pass its own §7 MDE gate before `H_conf` is opened.

**Where the feasibility claim would break, stated plainly.** If the injection locus moves to an **earlier
layer** (e.g. the L3H1 output or the three Atlas-licensed loci), every arm × dose requires a partial forward
pass and the cost multiplies by roughly (arms × doses × loci) ≈ 100–200×. At that point the design is
**not** demonstrated feasible on MPS and would need its own budget analysis. That variant is deferred (§10),
deliberately, and is not smuggled into this prereg.

---

## 9. GATES, SMOKE, AND STOPPING RULE

- **Global PASS = AND(** M1–M9, planted-positive, permuted-label FPR ≤ 0.05, planted-confound, per-context
  M7 **).** Run STOPS on failure. F1/F2 are recorded as FORCED and excluded from the AND.
- **`--smoke` (memorials/arithmetic only; validation sealed):** M1, M2, M3, M4, M5, M9 plus the
  planted-positive and permuted-label calibrations on a tiny **`D_src`-only** cohort. `H_dev` and `H_conf`
  residuals are **not captured** during smoke; no `τ`, no `Δ_spec`, no verdict is computed or written. Smoke
  must pass before freeze, per house convention 1.
- **Stopping rule for anomalous controls (frozen).** The run stops, writes its diagnostics, and produces
  **no primary interpretation** if any of: RANDOM `τ > 0.05` (M6 breach — the floor is not a floor);
  SRC-SHAM ≥ `--sham-frac` × REAL; M3 misses C26's 0.875 beyond tolerance (the pipeline is not the A5
  pipeline); M8 OOD band breached in an interpreted arm; permuted-label FPR > 0.05; POS-SHAM ≠ 0 (which would
  indicate a plumbing defect, since it is architecturally forced to 0).
- **Amendments:** pre-run only, logged inline as `**Amendment (YYYY-MM-DD, pre-run, no data seen).**` Once
  any primary quantity is inspected, a fix makes the analysis **outcome-aware verification**, not
  confirmatory (the IC1 A2 precedent, which cost that study its confirmatory status).

---

## 10. SCOPE

- **Reaches (if positive):** interventional **generality** of the donor-side delivered state across
  gene-disjoint examples and across the tested Atlas sequence contexts — the requirement in
  `docs/METHODOLOGY.md:85-88`. Evidence level: behaviour-level mediation with a transfer estimand.
- **Does NOT establish:** payload **semantic identity** (R22 stands); **minimality** or faithfulness (C33
  stands); **binding** (R23); the transduction **locus** — a positive result localizes nothing beyond the
  final donor residual; **site-choice steering** (Atlas WH7/S1–S5 is a different, more stringent estimand on
  relative partner coupling `R_ij(p)`); anything **biological**.
- **Deferred, documented:** (a) **earlier-layer injection** (L3H1 output, Atlas-licensed loci) — feasibility
  not demonstrated (§8); (b) **acceptor-side / bidirectional** transduction; (c) **cross-checkpoint** and
  cross-model transduction (the Caduceus milestone); (d) **sequence-realizable** induction of the state
  (Atlas WH8).
- **Outputs:** `scripts/splicebert/run_g1_cross_example_transduction.py`;
  `results/splicebert_coupling/g1_cross_example_transduction.json` + `.manifest.json` +
  `preflight_g1_cross_example_transduction.json`; per-arm/per-dose/per-context tables with dyadic CIs;
  realized matching SMDs; the `⟨s_j, s̄⟩` cosine distribution; all five seeded draws; the achieved MDE for
  every contrast; a required REFLECTION that re-anchors the result against §0.
- **Provenance:** checkpoint sha256; Atlas cohort + audit hashes; seeds and `PYTHONHASHSEED` to the manifest;
  a manifest written beside every output; no artifact overwritten.

---

## 11. OPEN DESIGN QUESTIONS — owner must decide before freeze

1. **Is G1-B the primary, or the ROADMAP-literal version?** `MECHANISM.md`/`ROADMAP.md` specify controls
   "above matched-random and within-example". This draft promotes `Δ_spec = τ(REAL) − τ(MEAN)` to primary
   because C26's 0.970 rank-1 overlap makes the ROADMAP-literal contrast near-anticipated (R-block). The
   owner must confirm this promotion or restore the literal reading — it changes the headline claim.
2. **Additive injection vs component replacement.** They are algebraically identical at `α=1` for the SELF
   arm but diverge off unit dose. Additive is chosen here for an exact `α=0` anchor and a clean dose grid.
   Confirm.
3. **SESOI_spec = 0.05 normalized** is asserted, not derived. It should be sealed from a `D_src`/`H_dev`
   pilot, and reconciled with the house 0.05-nat absolute floor (they are not the same scale).
4. **Which split supplies targets.** Atlas internal-validation only (as drafted), or also the
   partner-selection val split chr11/16/19? The latter was already read by A5/C26, so reusing it weakens the
   "never contributed" property. Sealed chr8/9 is assumed untouched — confirm.
5. **Context set and its multiplicity.** Six Atlas contexts are proposed as a secondary max-T family. Should
   any context be primary (cassette and complex are the most reviewer-relevant), and does that change the
   power target?
6. **Source-pool composition.** Sources restricted to canonical single-pair short introns (the
   characterization distribution). An alternative — context-matched sources — tests a different question
   (does transduction need a matching context?) and would double the arm count.
7. **Relationship to A9/A9b.** Should G1 also re-run the A9 acceptor-side interchange on the same targets, so
   the two transduction loci are compared under one design? This roughly doubles compute and adds a
   multiplicity family.
8. **Whether a 4-lens design red-team is required before freeze.** Every comparably expensive study in this
   repository (P3, P3.4, IC1, IC2, Atlas C) received one, and two of them were retired or re-scoped by it.
   This draft assumes one is mandatory; the owner should confirm and schedule it.
9. **Tolerances not yet sealed:** M3's C26-reproduction tolerance, M8's norm/entropy bands, `--max-src`,
   `--sham-frac`, the saturation criterion, and the dose grid endpoints.
10. **What is written into `CLAIMS.md` on each verdict.** In particular, the **NO TRANSDUCTION** branch
    materially bounds C16–C18/C23 and its exact wording should be agreed before the run, not after.
