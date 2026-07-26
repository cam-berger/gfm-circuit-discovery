# Preregistration — DNABERT-2 TATA→TSS (Phases 1–3)

> **Tier: T2 — provenance.** Version **v0.3**, 2026-07-26. **NOT FROZEN.**
> Frozen only when the gate owner signs G0 and resolves blockers B1/B2 (`../G0_GATE_PACKET.md` §6).
> Amendments before the run are logged inline as `Amendment (date, pre-run, no data seen)`.
> No confirmatory model output exists. The confirmation split has not been created.

Design of record after two adversarial design red-team rounds. §11 records what was rejected and why —
that record is load-bearing, because several estimands here are *not* the obvious ones.

---

## 0. Intent

> Does frozen DNABERT-2's native MLM use **TATA identity** and **native TATA-to-TSS geometry** when
> reconstructing a masked TSS-centred token span in human TATA-associated promoters — beyond
> target-token frequency, local composition, AT-richness, BPE segmentation, generic promoter identity,
> and arbitrary editing?

Re-read at every gate. The estimand must measure *this*; the result must answer *this*.

---

## 1. Readout

For example *i* in condition *c*:

```
ell_i(c) = ( SUM_{t in T_i} log p_theta( true_id(t) | all of T_i masked ) ) / n_nt(T_i)
```

* `T_i` = every BPE token overlapping nucleotide positions **−3 … +3**, masked **simultaneously**.
* This is a **pseudo-log-likelihood** (a sum of independent masked marginals), **not** a joint
  log-probability. Named as such in every result, figure, and claim.
* `T_i` — both token ids and covered nt interval — is **invariant across all arms of example *i***,
  enforced by construction and verified per example (**M2**). `n_nt(T_i)` is therefore a within-example
  constant: it re-weights examples in the cohort mean and **cannot bias any within-example contrast**.

Also recorded per arm: unnormalised sum, per-token log-probs, target entropy, top-1/top-5 recovery,
count and lengths of masked tokens.

---

## 2. The matched-slot construction

Arms are built by overwriting one or two length-`L` slots (`L = 7`, the PWM width) in the native window.
Words are drawn under a **hard four-way matching constraint**, verified per example:

1. **length** identical;
2. **mononucleotide composition** identical — `W` and `F2` are **permutations of one another**;
3. **PWM status** as the arm requires (pass / fail against the frozen threshold);
4. **token-geometry invariance (M2b)** — versus its partner arm: identical **total token count**,
   identical **token boundaries from slot-end to target**, identical **target tokens**.

Word roles: `W` = PWM-passing (a TATA); `F1`, `F2`, `F3` = PWM-failing fillers.

**Constraint 4 is not cosmetic.** Measured on 400 real promoters, naive TATA-vs-filler pairs differ in
total token count in **31–39%** of cases and in inter-slot token boundaries in **45–49%**. Because
ALiBi biases attention by *distance in token index*, that is a mechanical edit→target channel with
nothing to do with TATA (rival **R10**). It is **designed out** by constrained filler selection, not
modelled away — which matters because the imbalance is the majority regime, not an edge case.

**Feasibility, measured, not assumed:**

| Constraint | Availability |
|---|---|
| composition-matched **passing** alternative `W` (PWM MA0108.3, rel ≥ 0.85) | **97.9%** of passing words, median 7 |
| composition-matched **failing** permutation `F2` | **100%**, median 97 (min 5) |
| token-geometry-matched filler at each ladder slot | **98.3–100%** of promoters, median 32–43 candidates |

The surplus of matched fillers is what makes **≥3 independently seeded filler draws** affordable (§8).

---

## 3. Estimands

### 3.1 Content — at the **native** slot, one edit versus one edit

| Arm | native slot |
|---|---|
| `C0` | native TATA (unedited) |
| `C1b` | `W` — PWM-passing, **not** the native word |
| `C1a` | `F2` — a PWM-failing permutation of `W` |

```
Psi_content   = mean_i [ ell_i(C1b) - ell_i(C1a) ]     PRIMARY   (both arms synthetic -> memorisation-free)
Psi_content_A = mean_i [ ell_i(C0)  - ell_i(C1a) ]     secondary (higher power; carries memorisation)
memorisation  = Psi_content_A - Psi_content            REPORTED, not hidden
```

Position, length, edit count, composition and token geometry are all held fixed; **only letter order —
hence TBP-consensus status — varies.** Rivals R1/R2 die by construction rather than by covariate.

### 3.2 Geometry — a **two-edit** ladder anchored on a counter-locality contrast

For destination `d`: native slot ← `F1` (always), slot `d` ← `W` (arm `A(d)`) or `F2` (arm `B(d)`).

```
V_i(d)       = ell_i(A_i(d)) - ell_i(B_i(d))              "the value of a TATA at position d"
Psi_geometry = mean_i [ V_i(d_native) - V_i(d = -23) ]    PRIMARY
```

**Why `d = −23` is the comparator.** "Effect decays with distance" is *not* evidence of promoter-specific
geometry — any distance-decaying readout produces it, and ALiBi supplies one for free. Slot −23 is
**closer to the target than native**, so the two accounts make **opposite** predictions:

| Account | prediction |
|---|---|
| generic locality / ALiBi decay | `V(−23) ≥ V(−31)` ⇒ `Psi_geometry ≤ 0` |
| TATA→TSS positional grammar | `V(−23) < V(−31)` ⇒ `Psi_geometry > 0` |

A positive `Psi_geometry` therefore runs **against** the architectural null and cannot be manufactured
by it. (ALiBi cannot *create* a nonzero `V(d)` in any case — `V` is a TATA-vs-filler difference at a
fixed slot with composition matched — but it can shape the decay profile, which is what a
further-upstream-only ladder would have confounded.)

**Secondary:** the full ladder `D = {−23, −41, −51, −61, −71}` as a **spacing response curve** — its own
multiplicity family, and the object baseline **B4** must reproduce.

### 3.3 Dependence between the primaries

`Psi_geometry` contains the `V(d_native)` term that `Psi_content` measures. The two are **not independent
evidence**, and must never be reported as such. The dependence is intended: it maps onto the verdict
ladder — content alone ⇒ **PARTIAL**; content *and* position-specificity ⇒ **PASS**.

---

## 4. Cohort

* **EPDnew human Rel. 006**, 29,598 promoters × hg38. sha256-pinned in `data/external/epdnew/`.
* **Coordinates:** frozen and *verified* — see `../G0_GATE_PACKET.md` §3 (3,000/3,000 `SE` reproduction;
  anchor histogram mode −30/−31).
* **TATA rule:** JASPAR **MA0108.3**, motif-**START** anchor, relative threshold **0.85**, match in
  **−33 … −27** on the transcription strand ⇒ **3,297 promoters (11.1%)**. Threshold re-confirmed on
  development only. **TATA score bin is a mandatory stratum**, so the strict-consensus subset
  (`TATAWAWR`, n = 518) is recoverable as the top bin.
* **Inr strata** (mandatory): perfect `BBCA(+1)BW` / Inr-like / non-Inr. The cohort's **Inr composition is
  itself a G1 diagnostic** — human TATA and strong-Inr promoters anti-correlate and the target span *is*
  the Inr, so a structurally Inr-poor cohort must be visible before any effect is interpreted.
* **"Focused"** — **BLOCKER B1**, unresolved. Either freeze a FANTOM5-CAGE shape index as an inclusion
  rule, or delete "focused" from the cohort definition **and** every licensed claim. It may not be
  asserted unmeasured.
* **Window:** −96 … +64 nt, transcription-oriented. **BLOCKER B2** — candidate only; frozen after the
  window-length preflight (§6.1).
* **Ladder-slot motif screen:** every ladder slot is screened against a frozen core-promoter motif
  library and overlaps recorded. Slot −23 overlaps **BRE^d** (≈ −23…−17); both arms of `V(−23)` destroy a
  native BRE^d equally, so it cancels inside `V`, but it is recorded rather than discovered later.
* **Clustering:** gene symbol ∪ ≥90% identity over the window, frozen before splitting.
* **Complete-case across the full ladder:** a promoter is retained only if *every* arm passes *every*
  memorial. Per-arm exclusion would be selection on a post-treatment variable (a collider). The drop rate
  is reported and tested for stratum dependence.
* **Attrition table** — a **G1 deliverable**, end to end (PWM hit → `W`/`F2` availability →
  token-geometry match → M2 at −23 → N-free → dedup → cluster → split), with a **minimum-N stop rule**.
* **Split:** homology/gene-cluster-aware development / confirmation, immutable before any confirmatory
  `ell` is computed.

---

## 5. Memorials — each can fail; each on the measurement path

| ID | Check | Status |
|---|---|---|
| **M1 COORD** | hg38 extraction, transcription-oriented, `==` EPD `SE`; TSS = first uppercase base | **RUN — 3,000/3,000 PASS** |
| **M2 TARGET-INVARIANCE** | target token ids + covered nt interval identical across every arm | 0.0–0.2% failure at ≥20 nt; 3.2% at −23 |
| **M2b TOKEN-GEOMETRY** | total token count **and** slot-end→target boundaries identical across partner arms | satisfiable for 98.3–100% |
| **M3 EDIT-EXACTNESS** | constructed sequence differs from base at exactly the intended indices, nowhere else | |
| **M4 COMPOSITION** | `W`, `F1`, `F2`, `F3` are exact permutations of one another; dinucleotide distance reported | |
| **M5 DETERMINISM** | two forward passes of the same input give bitwise-identical `ell` | |
| **M6 MASK ACCOUNTING** | masked-token count and covered nt equal across arms; identical true ids scored | |
| **M7 PWM ANCHOR** | motif-START histogram on the frozen cohort: mode ∈ {−31, −30}, ≥70% of mass in −33…−29 | **RUN — mode −30/−31, 89% PASS** |
| **M8 ENV PIN** | vendored sha256s, checkpoint sha256, `transformers==4.44.2`, `missing=[] unexpected=[]` | **RUN — PASS** |
| **M9 EDIT-COUNT SYMMETRY** | `V(d_native)` computed in one-edit **and** two-edit form (fixed `F1` at a far decoy) must agree | closes the asymmetry noted in §11 |

**Explicitly labelled as forced, not evidence:** nothing in this list is architecture-forced. Should any
later anchor turn out to be forced by construction, it is labelled a *normalisation anchor*, never sold
as a completeness proof (this repository's **R21**).

---

## 6. Instrument preflight — development only

1. **Window-length sensitivity.** `ell(C0)` and `V(d_native)` at {160, 320, 640, 1024} nt. A 160-nt input
   is ~34 tokens and may be short-OOD. Freeze the smallest window whose `ell(C0)` is stable. *(Discharges
   B2.)*
2. **Contiguous-span masking sanity.** The checkpoint was trained with ~15% **scattered** masking; a 2–3
   token contiguous mask is a mild shift. Confirm non-degeneracy against scattered masks.
3. **Retokenisation propagation** on the frozen ladder, real promoter sequence.
4. **Negative controls.** filler-vs-filler sham; far-ladder slots; `A1` pseudo-TSS anchor; `A2` TATA-like
   non-promoter decoy; `A3` strand control.
5. **Behavioural health.** clean target log-prob; generic masked-DNA log-prob on a fixed non-promoter
   panel; sequence/tokenisation plausibility; performance by target-token length. An intervention that
   damages general masked-DNA behaviour is not promoter-specific evidence.

---

## 7. Calibration — hard gates

* **Planted positive / sensitivity.** The checkpoint is **frozen**, so it cannot have learned an invented
  rule; a synthetic dependency would be undetectable *by construction* and would validate nothing. The
  plant must therefore be **model-native**: place an **exact copy of the target span's own sequence** at
  slot `d`. Masked LMs exploit local repeats, so this is a real dependency travelling the **same
  channel** as the claim. A **graded** version (0/2/4/6 mismatches) reads the detection floor directly.
  Failure to detect it ⇒ underpowered ⇒ **STOP**.
* **Sham / specificity.** `F3`-vs-`F2` at slot `d`: filler against filler, exercising every code path.
  CI must include 0.
* **Permuted null.** Sign-flip permutation on the paired `V_i`; observed `|Psi|` exceeded ≤5% of the time,
  with the threshold calibrated against the permutation null itself.
* **Live sham.** Far-ladder slots on real data: CIs must include 0. A nonzero far-slot value means the
  assay responds to *any* motif insertion.

---

## 8. Statistics

* Unit: promoter; paired within-promoter contrasts.
* **≥3 independently seeded filler draws.** The estimand is therefore an **expectation over the filler
  distribution**; the unit is the per-promoter mean over draws, and per-draw estimates are reported.
  Rationale is this repository's **R22**: a within-sample held-out split does **not** test sampling
  stability, and a single filler draw is exactly that fragile axis.
* Uncertainty: **gene/homology-cluster bootstrap CI** + **exact add-one paired permutation** p.
* **Multiplicity.** The gate is the **pooled** estimate only. `d_native` and `d = −23` are frozen a
  priori (no selection). The ladder is its own corrected family. Stratum-level results (Inr class, score
  bin, spacing) are **descriptive**, corrected within their own family, and **may never rescue a pooled
  null**. The two primaries are distinct and conjunctive, not corrected against each other.
* Exclusions enumerated before unblinding.

---

## 9. SESOI — anchored a priori, on-scale

SESOI is **not** fitted to achievable precision. It is set before the TATA effect is seen, as the larger of:

1. the **assay detection floor** read from the graded copy-plant (§7); and
2. an interpretive floor fixed on the same units.

The development pilot then answers only *whether that SESOI is reachable at the achievable cohort size*.
If it is not, the honest outcome is a **documented stop before Phase 3** with an MDE statement — not a
smaller SESOI.

---

## 10. Baselines (G2)

Every baseline **B0–B5** must emit **`V̂_i(d)` on the same arms**, so `Psi` is computed identically for
baselines and for DNABERT-2. A baseline that only scores clean sequence cannot enter G2. Roles are as in
the phase plan: B0–B3 hard gates, B4 interpretive positive reference, B5 capacity reference.

Note that **R0 cannot threaten the paired estimand at all** (§1) — B0 is retained as a descriptive floor.

---

## 11. Rejected alternatives (why the estimands are not the obvious ones)

Two design red-team rounds. Recorded because each rejection changes what the study measures.

| Rejected | Why |
|---|---|
| **Content null = an edit at a distant neutral window** (`Psi = ell(C2) − ell(C1a)`) | Confounds *motif identity* with *position*, *distance to target*, and *local memorisation density*; and cannot refute **R2** (AT-richness), since any non-TATA replacement of an AT-rich word changes AT content. Replaced by an in-place, permutation-matched contrast. `C2` is retained as a **secondary** edit-genericity control for **R3**. |
| **Content primary measured at `d = native ± 2`** | Was adopted in an earlier draft to equalise edit count. If the TATA reader is sharply tuned to −31/−30, a ±2 shift already destroys it ⇒ a **self-inflicted null**. Fixed by decoupling: content is a one-edit-vs-one-edit contrast **at** the native slot; geometry is a separate two-edit ladder. |
| **Geometry = displacement swap vs a neutral two-window swap** | Different words at different positions; and a further-upstream-only ladder confounds promoter geometry with generic distance decay. Replaced by `V(d)`, a TATA-vs-filler DiD at a *fixed* slot, anchored on the counter-locality slot −23. |
| **Composition matching against the native word** | Only **44.8%** of native words admit a composition-matched alternative under the strict 8-mer `TATAWAWR` consensus — it would have halved the cohort. Making `W` and `F2` permutations **of each other** (neither needs to match the native) gives **97.9% / 100%** availability under the PWM. |
| **"Match and model" the retokenisation** (the source plan's approach) | Measured: the imbalance is the **majority regime** (31–49%), not an edge case. A nuisance covariate cannot carry that. Designed out via **M2b** instead. |
| **Planted positive = a synthetic deterministic dependency** | The checkpoint is **frozen** and cannot learn an invented rule, so the gate could never fire — a memorial that cannot fail validates nothing. Replaced by the model-native graded copy-plant. |
| **Positive control = Inr-flank disruption immediately outside the target** | An edit at −4/+4 retokenises the target and therefore **violates M2 by construction**; the control could never run under this study's own rules. The graded copy-plant serves as the single positive control. |
| **SESOI set from the pilot** | That fits SESOI to achievable precision. Re-anchored on the detection floor + an interpretive floor, both fixed before the TATA effect is seen. |

---

## 12. Decision rule — frozen before confirmation

| Outcome | Condition | Licensed statement |
|---|---|---|
| **PASS** | both primaries CI-exclude 0 **and** point ≥ SESOI on the **pooled** estimate; signs stable across gene/homology clusters **and** across filler draws; all memorials + calibration pass; B0–B3 do not account for the effects; negative controls below threshold; tokenisation and generic MLM health within bounds | "DNABERT-2's native MLM uses TATA identity **and** its TSS-relative position when reconstructing TATA-promoter TSS spans." |
| **PARTIAL** | `Psi_content` passes, `Psi_geometry` does not | "DNABERT-2 uses TATA-associated sequence context in promoter reconstruction." **Not** assembly; no geometry-centred circuit tracing without a redesign. |
| **FAIL** | neither passes with a qualified instrument | "The frozen checkpoint does not demonstrate the preregistered promoter-assembly behaviour at the achieved resolution." Report the MDE. |
| **INSTRUMENT UNRESOLVED** | any memorial, control, BPE invariant, or health gate fails | **No** biological interpretation in either direction. |

Sub-threshold effects are classified **by the CI**, not the point estimate: *bounded-null* = whole CI
inside ±SESOI; *small-but-detectable* = CI excludes 0 but point < SESOI.

A failed or heterogeneous result is preserved. The cohort is not revised after confirmation.

---

## 13. Scope — what this can and cannot reach

* **Reaches (on PASS):** the **Behaviour** rung of `docs/EVIDENCE_LADDER.md` — a scoped model behaviour
  under native-output counterfactuals with matched nulls. Nothing higher.
* **Does not establish:** any internal mechanism, component, head, or circuit; any biological function;
  any statement about transcription. Phases 4–7 are deliberately unlicensed until G3 passes.
* **Does not establish out-of-pretraining generalisation.** hg38 was in pretraining. Gene-disjoint
  confirmation protects analysis decisions and uncertainty estimates only, and is never described as
  held-out generalisation. `Psi_content` is memorisation-free by construction; the memorisation component
  is reported separately; a cross-species (mouse EPDnew, mm39/mm10 — **not yet acquired**) arm is a
  deferred secondary check.
* **Resolution of the content claim:** `Psi_content` shows that PWM-passing *arrangements* at the native
  slot raise TSS-span reconstruction relative to PWM-failing arrangements of the same letters. That is
  the licensed resolution — it is not evidence that the model possesses a TATA *concept*.
* **Provenance:** checkpoint `7bce263b…abda0` / sha256 `7ff39ec7…7aeba`; vendored code sha256 in
  `src/gfmcd/adapters/dnabert2/vendor/VENDOR.sha256`; EPDnew Rel. 006 sha256-pinned; PWM MA0108.3;
  `transformers==4.44.2`; all seeds recorded per run.
