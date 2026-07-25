# DRAFT v2 — PA1: does the acceptor supply a positional address for the donor?

> **Tier: T2 — provenance. STATUS: DRAFT, NOT FROZEN.** No primary statistic read. Arc label `PA1`
> is a placeholder pending Cam's assignment. Freeze only after the design red-team (§10) and the
> pilot (§9) pass.
>
---

# BOTTOM LINE — the one result this branch contributes

**Everything below is the working record. This is what it amounts to.**

### The experiment

Mask the donor window `[d−7, d+7]` so the **entire 5′SS consensus `[d−3, d+6]` is hidden**. In one
forward pass, score every candidate offset `k` in that window for "does a `GT` boundary belong here"
(factorized `GT` margin). Do it twice — partner acceptor intact (`A⁺`) and destroyed (`AG→CC`, `A⁻`) —
and subtract:

```
P(k) = D(k | A⁺) − D(k | A⁻)
```

The two sequences differ in exactly two bases far from the mask, so local content cancels and `P(k)`
is the acceptor's causal contribution to donor-ness at offset `k`. **`P` is the sealed C1 coupling
evaluated at every position instead of only at the donor.**

### The result

**`P(k)` peaks at the true donor.**

| cohort | mask | peakedness `P(0) − mean_k P(k)` |
|---|---|---|
| discovery, n=48 | `[d−7, d+7]` | **+0.346 [+0.236, +0.463]** |
| **fresh, gene-disjoint, n=96 / 93 genes** | `[d−7, d+10]` | **+0.165 [+0.116, +0.217]** |

Gene-clustered BCa; both CIs exclude 0. The second cohort is hard-asserted disjoint from the first.
(The smaller value is the wider mask, not a discrepancy.)

### Why it matters

C1 establishes that corrupting the acceptor changes the donor score by **+0.872** — a **scalar**. It
says nothing about *where* that influence lives. This says: the acceptor's contribution is
**spatially concentrated at the donor position**, not spread over the neighbourhood, and it **survives
complete destruction of the local donor motif**. The acceptor's evidence is position-specific and
does not require the donor's own motif to be visible.

Method-wise the transferable move is small and general: **take a scalar causal effect and make it
position-resolved, then ask where it lives.** Nothing here is splicing-specific.

### What this does NOT establish

- **Nothing about whether that localization moves with the acceptor.** The translation arms
  (§12B/§12D) reproduce a donor-locked pattern on independent genes, but the reciprocal positive
  control **failed** (§12D), so they are **uninterpretable**. No claim about acceptor-anchor vs
  donor-query is licensed.
- **Nothing about payload semantics.** R22 and C30's closed-bounded status stand untouched.
- **Nothing biological.** Model mechanism only; the resemblance to donor-first recognition is a
  discussion-section hypothesis, not evidence.
- **Tier: EXPLORATORY.** Discovery splits only; the sealed chr8/9 cohort was never opened. No row is
  added to `CLAIMS.md` and none is licensed.

**Reproduce:** `uv run --extra splicebert --extra genome python scripts/splicebert/run_pa1_step4.py --n 96 --pool 700`

---

> **v2 (2026-07-25, pre-run, no data seen)** — four review changes, all accepted:
> (1) readout is now a **factorized `GT` margin**, not a single-`G` margin — once the donor window is
> masked, single-`G` is not a donor-site score; (2) primary is now a **two-model anchor comparison on
> common support**, not raw cross-correlation, which confounds translation with amplitude;
> (3) controls expanded to distinguish all four candidate anchors, with **≥2 edit positions**;
> (4) the long mask is treated as an **empirical assumption with its own preflight**, not a
> derivation — the earlier "mask interactions cancel" claim was too strong and is withdrawn.

---

## 0. INTENT (north star — re-read at every gate)

> **Move the acceptor relative to a fixed, obscured donor: does the acceptor-dependent
> donor-prediction profile move with it?**

Behaviour-level. The converse of C32 (which is QK/selection-level only) and distinct from the P3
shift diagnostic (which tracked **L3H1's attention peak** from a fixed donor query, not the position
at which a donor is *predicted*).

**Prior (exploratory, n=24/23 genes, `results/phase-lattice/p3_shift_diagnostic_v2.json`):** under
`intron_insert`, L3H1's attention peak followed the acceptor-relative coordinate in 22/24, 22/24,
23/24 (d=+1,+2,+3, intact PE), degrading to 14/24, 17/24, 14/24 under the C8 carrier deletion. The
fixed-position donor-margin contrast was inconclusive (`+0.021 [−0.145, +0.167]`) — expected, since a
scalar cannot shift.

---

## 1. ESTIMAND

### 1.0 Mask geometry

**Coordinate convention (audited against `coupling.py:67`):** `donor_pos = d` is the **first intron
base = the G of GT**; the T is at `d+1`. Candidate offsets `k` are array offsets from `d`, NOT
biological 5′SS coordinates.

- **Mask block:** `[d−7, d+7]` (15 nt), applied in a **single forward pass**.
- **Candidate grid:** `k ∈ [−6, +6]` (13 candidates). Candidate `k` scores the dinucleotide at
  `(d+k, d+k+1)`; the block extends to `d+7` so that **both members of every candidate are masked**.
  (A `[d−6, d+6]` block would leave the T of candidate `k=+6` visible — the v1 error.)
- The 5′SS consensus spans `[d−3, d+6]` and is therefore **entirely destroyed**, which is the point:
  the model must infer where a donor belongs from surviving context plus the acceptor.

### 1.1 Readout — factorized GT margin

Single-`G` margin is **not** a donor-site score once the window is masked (the T is masked too). For
candidate `k`, with `ℓ_p(·)` the logit vector at masked position `p`:

```
D(k) = ℓ_{d+k}(G) + ℓ_{d+k+1}(T)  −  max_{xy ≠ GT} [ ℓ_{d+k}(x) + ℓ_{d+k+1}(y) ]
```

The max factorizes exactly and cheaply:

```
max_{xy≠GT}[·] = max(  max_{x≠G} ℓ_{d+k}(x) + max_y ℓ_{d+k+1}(y),
                       ℓ_{d+k}(G)          + max_{y≠T} ℓ_{d+k+1}(y)  )
```

This is a **pseudo-likelihood** score from independent per-position marginals of one masked forward,
not a joint likelihood — labelled as such, never reported as `log P(GT)`.

### 1.2 The position-resolved coupling profile

```
P(k) = D(k | A⁺) − D(k | A⁻)
```

`A⁺` = native; `A⁻` = partner AG→CC at `[a−2, a−1]` (`CORRUPT_ALT`). The two sequences are
byte-identical except two bases far from the mask, so **local content cancels exactly** and `P(k)` is
the acceptor's causal contribution to predicting a GT boundary at offset `k`.

**Note on the C1 anchor.** `P(0)` is **not** the C1 coupling — the window mask destroys the motif that
C1's single-G mask leaves intact. The exact arithmetic anchor is preserved as a **standalone
single-mask cell** (memorial M1), run outside the profile. Never report `P(0)` as coupling.

### 1.3 PRIMARY — two-model anchor comparison on common support

Raw dot-product cross-correlation confounds translation with amplitude and uses different overlap at
different shifts. Instead, compare two prespecified models on **identical support**:

```
donor-locked     :  P_edit(k) ≈ P_native(k)
acceptor-locked  :  P_edit(k) ≈ P_native(k − δ)

S(δ) = { k : both P_native(k) and P_native(k−δ) are defined }      (common support)
ρ_D  = corr( P_edit , P_native(·)    )  on S(δ)
ρ_A  = corr( P_edit , P_native(·−δ)  )  on S(δ)

Δρ(δ) = z(ρ_A) − z(ρ_D)                  z = Fisher transform
```

**PRIMARY:** mean `Δρ` over phase-preserving `δ ∈ {−6, −3, +3, +6}`, gene-clustered BCa CI.
`Δρ > 0` favours acceptor-relative localization. Correlation is used (not raw MSE) because it is
scale-free, so amplitude changes from edit damage cannot masquerade as translation.

**SECONDARY (intuitive summaries, not the identification):** the displacement `δ̂` (parabolic
interpolation about the discrete argmax of the correlation-vs-τ curve) and the slope `β` of `δ̂` on
`δ`. Argmax of `P` itself is reported third — the exploratory artifact shows `mean_peak_shift = 1.92`
at d=+1 while 22/24 track exactly, i.e. argmax is outlier-dominated.

**The licensed claim is conjunctive**, not "slope 1 = the acceptor is the ruler":

> A slope near 1, **together with** superiority of the acceptor-locked model **and** null sham
> effects, indicates that the acceptor-dependent donor profile translates with the acceptor.

### 1.4 Secondaries from the dense ladder

The dense ladder (§2) supports two further preregistered quantities the sparse phase-preserving
ladder structurally cannot:

- **`Δβ_phase = β_pp − β_pb`** — phase-preserving minus phase-breaking slope.
- **`Aφ`** — amplitude of a period-3 component in the residual of `δ̂ ~ β·δ` over all 13 levels
  (least-squares projection onto `cos(2πδ/3)`, `sin(2πδ/3)`), gene-clustered CI.

Both are **secondary, Bonferroni-corrected across the two (α=0.025 each)**, and may **not** rescue a
null primary. **Prior expectation is null:** C19 already found the established coupling insensitive
to intron-length mod 3. Motivation is that real introns are not uniformly mod-3 clean, and the v2
artifact shows acceptor-relative attention tracking falling 0.92→0.58 under carrier deletion — so the
question is worth asking, at secondary status, with that prior stated.

---

## 2. METHOD — arms and ladder

**Ladder (dense):** `δ ∈ {−6 … +6}`, all 13 integers.
Phase-preserving `{−6,−3,0,+3,+6}` (primary); phase-breaking `{−5,−4,−2,−1,+1,+2,+4,+5}` (secondary).

| Arm | Edit location | Donor | Acceptor | Separation | Expected if acceptor-anchored |
|---|---|---|---|---|---|
| **T1, T2 — target** | two distinct sites in `[d+7, a−71]` | fixed | moves δ | **changes** | profile moves δ |
| **S1 — translation** | whole window, genuine genomic flank | moves δ | moves δ | fixed | no shift *relative to donor*; **exact +δ shift in absolute coords** (→ planted positive, §4) |
| **S2 — downstream** | 30 nt after the partner acceptor | fixed | fixed | fixed | no shift |
| **N — nonpartner AG** | target edit; `P` computed vs a matched **non-partner** AG corruption | fixed | partner intact | changes | no partner-specific shift |

**Two target sites (T1, T2)** at distinct positions within the eligible window, so that one deleted
regulatory element cannot drive the result. Agreement between T1 and T2 is a reported diagnostic.

**Edit construction.** Insertions reuse `choose_insertion` (deterministic, locally
composition-matched, forbidden from increasing local `AG`/`GT`/`GC` counts). Deletions mirror it and
restore length 1024 from genuine right-flank genome. All edits are placed **outcome-blind** from
coordinates alone.

**Protected zones (bit-identical, inherited from IC1):** donor `[d−3, d+6]`; acceptor `[a−70, a+3]`.
Edit sites confined to `[d+7, a−71]`, which requires separation ≳ 98 nt for two sites ≥10 nt apart;
eligibility is computed **before any model call** and the excluded fraction is reported.

**Secondary condition:** every cell is additionally evaluated under the exact C8 period-3 deletion
(RFFT bins 339–345, `make_deleted`).

---

## 3. MEMORIALS (must be able to fail; on the measurement path)

| # | Memorial | Threshold | Can fail? |
|---|---|---|---|
| M1 | **Standalone single-mask cell** (mask only `d`, motif otherwise intact) reproduces `CouplingTask.coupling` | max abs diff `< 1e-4` | yes — exact arithmetic anchor |
| M2 | Native `A⁺` window-mask profile is **live and localizes the true donor**: `argmax_k D(k)` within ±1 of `k=0` in ≥70% of examples | ≥0.70 | yes — the core assay-validity gate |
| M3 | `P` profile magnitude: `max_k P(k) − min_k P(k)` exceeds the `δ=0` replicate noise floor in ≥80% of examples | ≥0.80 | yes |
| M4 | Every variant is length 1024; donor GT intact; partner AG intact in the `A⁺` arm | all true | yes |
| M5 | No edit increases local `AG`/`GT`/`GC` counts in a ±18 nt window | all true | yes |
| M6 | Protected zones byte-identical between native and every variant | all true | yes |
| M7 | `δ=0` cell bit-identical to native ⇒ `Δρ(0)=0`, `δ̂=0` exactly | exact | yes (index bugs) |
| M8 | **Mask-width stability**: primary sign and rank order preserved across mask half-widths `w ∈ {6,7,8}` | sign-stable | yes |
| M9 | MLM health on non-site positions retains ≥50% of intact headroom over chance (carrier-surgery standard) | ≥0.50 | yes |
| M10 | **Estimator resolution** — re-index the readout grid by τ, require `δ̂ = τ` | exact, τ∈{−3..3} | yes, but **forced by re-indexing**: validates the estimator only. Labelled; not biological evidence |

Global `PASS = AND(M1..M10)`. Run **stops** on failure. `--smoke` runs M1, M4–M7, M10 on ≤8 examples.

---

## 4. CALIBRATION

- **Planted positive (sensitivity, model-mediated — not forced).** The **S1 translation arm read in
  absolute coordinates**. The model receives a genuinely translated window and must emit a profile
  translated by exactly `+δ`; the estimator must recover it. This exercises masking, scoring, and
  displacement end to end, and unlike M10 it is not a re-indexing identity. Require `δ̂ = δ` within
  ±0.5 in ≥80% of examples at `|δ|=1`, else **STOP: the assay cannot resolve one nucleotide.**
- **Null / specificity.** Permute the `δ̂`↔`δ` pairing within gene, refit, require fire rate ≤0.05
  over ≥200 permutations, threshold calibrated on the permutation null itself.
- **Sham specificity.** `Δρ` on S1 (donor-relative), S2, and N must each be a bounded null. A target
  positive with a same-sized sham positive is **not licensed**.

---

## 5. IDENTIFIABILITY

| Confound | Entangled? | Handling |
|---|---|---|
| Period-3 carrier phase | yes, at δ ∉ 3ℤ | primary on phase-preserving δ only; phase-breaking becomes a named secondary (§1.4), never pooled |
| Absolute position | partly | S1 read in donor-relative coordinates |
| Generic edit damage / length | no | S2; and `A⁺`/`A⁻` pairing cancels effects common to both |
| Profile amplitude vs translation | **yes, if MSE on raw profiles** | resolved by using scale-free correlation on common support (§1.3) |
| Deleted regulatory element at one site | no | two target sites T1/T2 |
| PPT / branchpoint | no | `[a−70, a+3]` protected |
| Local 5′SS consensus pinning the answer at `d` | **structurally** | not removed — this is the competing hypothesis; `Δρ ≈ 0` is its predicted signature. Substantive alternative, not nuisance |

**Honest non-identifiable outcome:** if phase-preserving and phase-breaking disagree and the
phase-preserving arm is underpowered (§9), report **"translation not separable from carrier phase
disruption at achieved precision"** and emit no addressing verdict.

---

## 6. DECISION RULE (frozen before data)

SESOI on `Δρ`: **0.15** (Fisher-z units).

| Result | Conclusion |
|---|---|
| `Δρ` CI excludes 0, point ≥ 0.15, **and** `β` near 1, **and** all sham nulls bounded | **The acceptor-dependent donor profile translates with the acceptor** — consistent with the acceptor acting as a positional anchor for donor localization |
| `Δρ` CI ⊆ `[−0.15, +0.15]` | **Bounded null: donor-anchored** in this subdomain. Report achieved MDE |
| `Δρ` CI excludes 0, point < 0.15 | small-but-detectable sub-SESOI translation (classify by CI) |
| Any sham not bounded-null | **not licensed** |
| M2 or M3 fails | **assay invalid** — no verdict; see §9 fallback |

Classification is by **CI**, never point estimate, and never by a gate algebraically tautological with
the primary.

---

## 7. HELD-OUT

`val` (chr11/16/19) → discovery; `test` (chr8/9) sealed → confirm. Mask width, estimator, candidate
grid, and δ set are selected on discovery **only** and frozen before the sealed cohort opens.
Inference unit = **gene** (gene-clustered BCa bootstrap).

---

## 8. COST

One forward per (example, arm, δ, acceptor-state) — `k` costs nothing because the whole profile comes
from one pass:

```
natives (shared)            3 states                                    =    3 fwd/example
arms × ladder × states      4 arms × 12 δ × 3 states                    =  144 fwd/example
P3-deleted condition        ×2                                          ≈  294 fwd/example
```

≈5.9×10⁴ forwards at n=200. Wall-clock timed in the pilot.

---

## 9. PILOT — run FIRST (gates everything)

On ≤24 discovery examples, before any confirmatory build:

1. **M2 — is the native `A⁺` profile live and does it localize the donor near `k=0`?** Go/no-go.
2. **M3 — is `|P|` large enough** to estimate displacement?
3. **§4 planted positive — does the assay resolve a 1-nt translation?** Go/no-go.
4. **M1 — does the standalone single-G cell reproduce C1?**
5. **M8 — mask-width stability** across `w ∈ {6,7,8}`.
6. **M9 — MLM health** under the long mask.
7. **Per-example variance of `Δρ` and `δ̂`** → MDE on the primary; if MDE > 0.15 at feasible n, **do
   not run the confirmatory**.
8. **Wall-clock** → §8 reduction decision.

**Fallback if the long mask fails.** Do **NOT** fall back to scanning a single masked position with
the rest intact — that leaves the true GT visible for every `k≠0` and answers a different question.
Any fallback must keep the **donor core masked in every candidate-scoring pass**: mask the consensus
`[d−3, d+6]` in all passes and additionally mask the candidate pair `(d+k, d+k+1)`. This costs 13×
but preserves the question.

---

## 10. RED-TEAM BEFORE FREEZE (not yet done)

Lenses: **identifiability**, **statistical**, **estimand↔intent**, **calibration**. Freeze on survival.

---

## 11. SCOPE

**Can establish:** whether the acceptor's causal contribution to donor-site prediction is anchored to
a separation-relative coordinate, at **behaviour level**, in canonical internal introns with
sufficient separation, on the frozen `SpliceBERT.1024nt` checkpoint.

**Cannot:** payload semantics (R22 stands); training origin (deferred, P3.4); transcript-level
splicing; promotion of the exploratory 22/24 attention tracking; biology.

**Second causal stage (NOT this study).** To license the "combined with relative-position
information" *mechanism*, a follow-up must show the translation **collapses when the L3H1/address
pathway is disrupted**, against matched head and positional controls. This study is the behavioural
premise for that stage, not the mechanism itself.

---

## 12D. STEP 4 — fresh gene-disjoint cohort (2026-07-25) — **INSENSITIVE / NO CONCLUSION**

Runner `scripts/splicebert/run_pa1_step4.py`; artifact `results/splicebert_coupling/pa1_step4.json`
(+manifest, +`_smoke`). **n=96, 93 genes, val, verified gene-disjoint from the pa1_pilot 48** (hard
assert in the runner). Mask asymmetric `[d−7, d+10]` so a donor consensus displaced by ≤+3 stays
hidden. `Δρ` primary; `δ̂` not computed (retired, §12C). Sealed chr8/9 **never opened**.

| arm | δ | n | **Δρ** [95% gene-clustered] | excl 0 | median ‖ΔP‖∞ |
|---|---:|---:|---|---|---:|
| T1 acceptor moves (mid) | 1 | 96 | **−2.350** [−2.564, −2.128] | yes | 0.078 |
| T1 | 2 | 96 | −2.332 [−2.546, −2.118] | yes | 0.122 |
| T1 | 3 | 96 | −1.841 [−1.970, −1.702] | yes | 0.070 |
| T2 acceptor moves (⅓) | 1 | 96 | −2.045 [−2.284, −1.801] | yes | 0.105 |
| T2 | 2 | 96 | −2.128 [−2.372, −1.878] | yes | 0.111 |
| T2 | 3 | 96 | −1.622 [−1.758, −1.487] | yes | 0.077 |
| **RECIP donor moves** | 1 | 96 | **−0.425** [−0.597, −0.235] | yes (**wrong sign**) | 0.413 |
| **RECIP** | 2 | 96 | **−0.594** [−0.789, −0.401] | yes (**wrong sign**) | 0.426 |
| **RECIP** | 3 | 96 | **+0.085** [−0.013, +0.178] | **no** | 0.353 |
| S2 sham downstream | 1 | 96 | −0.997 [−1.234, −0.757] | yes | 0.262 |
| S2 | 2 | 96 | −1.050 [−1.295, −0.796] | yes | 0.285 |
| S2 | 3 | 96 | −2.172 [−2.363, −1.979] | yes | 0.046 |

### Verdict (frozen rule, applied as written)

> **INSENSITIVE — the reciprocal positive control did not move the profile. T1/T2 are
> UNINTERPRETABLE; no conclusion about either competing algorithm.**

T1/T2 reproduce the §12B donor-locked pattern on independent genes (Δρ ≈ −1.6 to −2.4, every CI
excluding 0), but that pattern **cannot be read as evidence** while the positive control fails: an
assay that never shows a moved profile *under any manipulation* would produce exactly these numbers.

### Diagnosis — the reciprocal edit reshapes rather than translates

`RECIP` perturbs the profile by **‖ΔP‖∞ ≈ 0.35–0.43 vs a native peakedness of 0.165** — i.e. **2.1–2.6×
the entire signal**. That is not a translation; the profile is being replaced. Cause is almost
certainly the design: the compensating exon insertion sits at `d−20`, only **13 nt from the mask edge**
(`d−7`), directly inside the surviving context that localizes the donor. The edit destroys the
localizer it was meant to displace.

Secondary red flag: **S2 (nothing moves) perturbs the profile MORE than T1 does** (0.262 vs 0.078 at
δ=1). A generic edit 30 nt past the acceptor moves `P` ~3× more than relocating the acceptor. The
readout is more sensitive to incidental sequence change than to the targeted intervention, which
independently caps how much T1/T2 can carry.

### What DID replicate

**Native peakedness on the fresh gene-disjoint cohort: `+0.165 [+0.116, +0.217]`, CI excludes 0**
(lower than §12C's +0.346 because the mask is wider here, not a discrepancy). **`P(k)` is localized at
the true donor with the 5′SS consensus fully masked — replicated on 93 independent genes.** This is
the one observation from the whole PA1 arc that has survived an independent cohort.

### Not actioned (deliberately)

The obvious fix — move the compensating insertion far from the mask (`d−60`+, gated on exon length),
or compensate outside the donor–acceptor span entirely — is **not** run here. Re-running a failed
positive control with a tweaked design until it passes, after seeing the target arms, is exactly the
outcome-aware drift this protocol exists to prevent. Any revival needs the amended RECIP frozen
*before* the target arms are re-read, on a third cohort.

---

## 12C. SENSITIVITY LADDER L0+L1 (2026-07-25) — **sensitivity ESTABLISHED on Δρ**

Runner `scripts/splicebert/run_pa1_sensitivity.py`; artifact
`results/splicebert_coupling/pa1_sensitivity.json` (+manifest). n=24 val discovery, MPS.
EXPLORATORY, outcome-aware. Sealed split untouched.

**§12B's "may be non-identifiable in the masked-window framing" is WITHDRAWN.** An internal
shifted-write plant supplies the calibration a sequence-level plant could not.

### L1 — final-residual shifted-write plant (near-forced; instrument check)

| planted δ | `δ̂` | frac ≤0.5 | **Δρ** |
|---:|---:|---:|---:|
| 0 | +0.000 | 1.00 | 0.000 |
| 1 | +1.142 | 0.54 | **+0.879** |
| 2 | +1.135 | 0.54 | +0.747 |
| 3 | +2.249 | 0.58 | +0.319 |

- δ=0 exact ⇒ plumbing sound.
- **`δ̂` FAILS at every nonzero δ even on a directly-planted residual translation.** Independent
  confirmation that `δ̂`'s correlation surface is multimodal. **`δ̂` is retired as a gate and as a
  primary; descriptive/visualization only.** This, not the model, explains the §12A PP failure.
- **`Δρ` responds correctly and positively to a known shift.** Sensitivity decays with δ
  (+0.88 → +0.75 → +0.32), so the calibrated window is small |δ|.

**The discriminating comparison now exists:**

```
planted translation (known +1) :  Δρ = +0.88     <- instrument CAN register a moved write
acceptor moved +1 (endogenous) :  Δρ = −2.35     <- it registers the opposite, large
```

### L0 — query field (proposed insertion; THIS RUN IS CONFOUNDED)

L3H1 attention mass to the four-token acceptor window, from each query `d+k`:

```
k:               -6     -5     -4     -3     -2     -1      0     +1     +2     +3     +4     +5     +6
native field: 0.392  0.480  0.458  0.399  0.457  0.453  0.375  0.430  0.475  0.378  0.448  0.451  0.316
acceptor mass retained: 5.51 -> 5.46 (d=1) -> 5.46 (d=3) -> 5.28 (d=6);  field displacement ~0 at all d
```

Field is **flat** (k=0 not special) and **spatially pinned** while retrieval mass is retained.
**DISCOUNT THIS RUN:** all 13 positions in the block are `[MASK]`, so their queries differ only by
position embedding — a flat, position-determined, translation-invariant field is close to forced.
**Re-run L0 with the donor window UNMASKED before using it.**

### The surviving dissociation (mask-independent, and the most interesting result here)

> Attention to the acceptor is **flat over k**, but `P(k)` is **peaked at k=0** (peakedness
> +0.346 [+0.236, +0.463]).

The localization of the acceptor's effect is **not** inherited from where retrieval happens. Evidence
is retrieved broadly across the window; it moves the GT margin only at the position that is already a
donor candidate on other grounds (surviving context outside the mask). This is a direct mechanistic
instantiation of **partner verification over site localization** and is independently testable.

### Architecture caveat carried forward

A head writes at its **query** position. Wherever the querying position is the donor, "the write
stays at the donor" is architecture-expected, so the **L2 L3H1 transplant measures CAPACITY, not the
endogenous algorithm's choice** — it should be labelled that way in any writeup.

---

## 12B. CORRECTION TO §12A (2026-07-25, same day, outcome-aware)

**§12A's verdict wording is WITHDRAWN.** Stopping was correct; the stated reason was not licensed.
Six defects, all confirmed:

1. **PP was not a known-positive for the target.** S1 `window_translation` moves the donor–acceptor
   pair *together* across the absolute PE table. The target holds the donor and moves only the
   acceptor. SpliceBERT has learned absolute position embeddings and a causal period-3 carrier and is
   **not translation-equivariant**, so S1 exercises a different channel; its failure never bounded the
   target's detectability. §12A's "the assay cannot resolve a one-nucleotide translation" **does not
   follow** and is withdrawn.
2. **The STOP gate ran on the SECONDARY statistic.** §1.3 makes `Δρ` primary; §4 gated on `δ̂`. Not
   interchangeable. Internal inconsistency in this prereg, applied as if frozen.
3. **`mean_dhat` was mislabelled "population mean."** It is the mean of per-example argmax-based
   `δ̂`, not the displacement of the mean profile. Estimating displacement from the actual
   population-mean profile gives ≈ **−2** while `Δρ` favours the planted **+1** ⇒ `δ̂` has competing
   correlation peaks and is **unsuitable as a hard gate**.
4. **M2 tests the wrong object** (raw `D(k|A⁺)` rather than `P`). The pilot's own width sweep shows
   `P` is significantly donor-localized at every width. Remove raw-D M2; keep a `P`-liveness gate.
5. **M3 is VACUOUS.** `P_rep` re-runs deterministic inference on identical inputs ⇒ noise floor is
   exactly 0.0 and `span > noise` cannot fail. Textbook forced memorial. Replace with
   independent-cohort / independent-seed stability.
6. **M8 was never implemented** (replaced by the width sweep, PASS dropped) and **M9 reports raw
   health without computing the headroom ratio** it is gated on.

**Direct-target diagnostic (read-only, discovery cohort, n=48; EXPLORATORY, outcome-aware).** Using
the intervention the study is actually about — `intron_insert`, donor fixed, acceptor moves:

| acceptor δ | ρ_D | ρ_A | Δρ | ‖ΔP‖∞ | ‖ΔP‖ / peakedness |
|---:|---:|---:|---:|---:|---:|
| +1 | 0.901 | −0.235 | −2.35 | 0.173 | 0.50 |
| +2 | 0.855 | −0.198 | −2.20 | 0.208 | 0.60 |
| +3 | 0.971 | +0.453 | −2.07 | 0.103 | 0.30 |
| +6 | 0.961 | +0.443 | −1.96 | 0.123 | 0.36 |

**Null-instrument check (this is what licenses reading the above):** the edit perturbs `P` by
**30–60% of the acceptor's own peakedness** (0.346), so the intervention is live — the donor-locked
win is *not* forced by an inert edit. **`Δρ ≈ −2.2` is Fisher-z saturation off `ρ_D ≈ 0.9`, NOT an
effect size**; the licensed statement is "the profile stays put." Note `ρ_A` rises −0.2 → +0.45
between δ∈{1,2} and δ∈{3,6}: a period-3 signature in the profile itself.

**Strongest current EXPLORATORY reading (not confirmatory; discovery split, outcome-aware):**

> The acceptor contributes a donor-localized boost, but moving the acceptor does not move that boost.
> This is more consistent with a **fixed donor query flexibly retrieving its acceptor** than with the
> acceptor acting as a reverse positional ruler for locating the donor. It coheres with C5/C32 (donor
> is the query, acceptor the key/value) and with the exploratory finding that L3H1's attention peak
> follows a moved acceptor from a fixed donor query — attention retargets; the delivered contribution
> stays at the donor.

**OPEN AND LOAD-BEARING — sensitivity is still unestablished.** No channel has been shown capable of
moving this profile. The natural plant (translate the donor's own 5′SS consensus) is **impossible by
construction**: the consensus `[d−3, d+6]` lies entirely inside the `[d−7, d+7]` mask, so a moved
consensus stays masked and invisible; pushing it outside the mask re-enters M2's failure (the
window-masked model does not localize donors from context, 0.146 within ±1). **M2's failure and the
missing plant are the same problem**, so "non-identifiable in the masked-window framing" is a live
possible outcome of any redesign and must be pre-declared, not discovered mid-run.

**Redesign (agreed, NOT yet implemented):** make T1/T2 the primary diagnostic; `Δρ` primary, `δ̂`
descriptive only; keep `P`-liveness, drop raw-D M2; replace M3 with cross-cohort/seed stability;
replace S1 with a sensitivity intervention that shifts the acceptor-dependent profile while leaving
the donor's absolute measurement geometry fixed; re-run on a fresh gene-disjoint discovery cohort
before freezing; sealed split stays closed.

---

## 12A. PILOT OUTCOME (2026-07-25) — superseded in part by §12B above

Runner `scripts/splicebert/run_pa1_pilot.py`; artifact `results/splicebert_coupling/pa1_pilot.json`
(+ manifest). n=48 val (discovery), 48 genes, MPS. **No sealed split opened. No addressing verdict.**

**Gate results**

| Gate | Result | Status |
|---|---|---|
| M1 single-mask reproduces C1 | `max_abs_diff = 0.0` | **PASS** (exact) |
| M2 native `D(k|A⁺)` localizes donor | argmax within ±1 of 0 in **0.146** (need 0.70) | **FAIL** |
| M3 profile magnitude vs noise floor | span > noise in 100% | PASS |
| M7 δ=0 identity | `Δρ=0`, `δ̂=0` exactly | PASS |
| M9 MLM health | 0.535 | reported |
| M10 estimator resolution (forced) | mean abs err **0.038–0.063 nt** at half=7 | **PASS** |
| **PP planted positive** | best cell **0.69** within ±0.5 (need 0.80) | **FAIL → STOP** |

**Amplitude/resolution trade-off (width sweep, gene-clustered CIs).** Peakedness
`P(0) − mean_k P(k)` falls 5.4× as the mask widens to the width the ladder requires:

| half | mask | candidates | `D` argmax==0 | `P(0)` | peakedness |
|---|---|---|---|---|---|
| 2 | 5 nt | 3 | 0.77 | +0.899 [0.616, 1.188] | +1.882 [1.263, 2.531] |
| 3 | 7 nt | 5 | 0.67 | +0.904 [0.603, 1.260] | +1.602 [1.094, 2.180] |
| 4 | 9 nt | 7 | 0.29 | +0.506 [0.305, 0.734] | +0.741 [0.474, 1.049] |
| 5 | 11 nt | 9 | 0.31 | +0.494 [0.324, 0.681] | +0.645 [0.419, 0.880] |
| 7 | 15 nt | 13 | 0.10 | +0.278 [0.188, 0.373] | +0.346 [0.236, 0.463] |

**What the pilot did and did not show.**

- **Positive:** the coupling profile `P(k)` is genuinely **localized at the true donor at every mask
  width**, including the full 15-nt block (peakedness CI excludes 0 throughout). The acceptor supplies
  a position-specific GT boost through a hole that destroys the entire 5′SS consensus. This is a real,
  reproducible observation — but it is a *feasibility* statistic on discovery, **not a claim**.
- **Blocking:** the planted positive fails at every cell. At half=7 the *population mean* recovers
  δ=1 well (`δ̂ = +0.95`) but only 69% of examples land within ±0.5; at δ=2 and δ=3 even the mean
  collapses (`−0.02`, `−0.13`). Because **M10 is exact (0.04 nt)**, the failure is **signal, not
  arithmetic**.
- **Instability:** an earlier n=48 draw under a different cohort tag gave `δ̂ = +0.13` at half=7, δ=1
  (vs `+0.95` here). Per-cohort instability of the same statistic — the R22 lesson applies: any
  revival needs **multiple independently-seeded draws**, not one.
- **M2 was specified on the wrong object.** It gates the raw `D(k|A⁺)` profile — whether the model as
  a whole can find its donor from flanks alone — which is strictly stronger than the assay needs. The
  assay's object is `P(k)`, and that one is localized. Any redefinition of M2 onto `P` is now an
  **outcome-aware amendment** (the statistic has been seen) and cannot be presented as a clean
  preregistered gate.

**Diagnosis and options (for Cam — not actioned).**

1. **Population-mean displacement.** Estimate `δ̂` on the gene-averaged profile with a gene-clustered
   bootstrap instead of per-example-then-average. Directly motivated by δ=1 recovering on the mean
   while per-example dispersion fails. Does **not** explain the δ=2/3 collapse.
2. **Replace the planted positive.** `window_translation` does not purely translate — it pulls fresh
   genomic flank and shifts every token against the PE table, so at larger δ the profile changes for
   content reasons. A cleaner plant would move the acceptor window with total length and all other
   absolute positions fixed (§4's original alternative).
3. **Accept the resolution floor** and run only `|δ| ∈ {3, 6}` at half=7, dropping the ±1/±2
   secondaries — but the pilot gives no evidence the estimator works at δ=3 either.
4. **Abandon the localization framing.** The durable pilot observation is that `P(k)` is
   position-specific; a study of that profile's *shape* may be better posed than a study of its
   *translation*.

**Status: the frozen §4 STOP rule was applied as written. This prereg is not frozen and the
confirmatory is not licensed.**

---

## 12. PROVENANCE

Checkpoint `SpliceBERT.1024nt`, sha256 `2ad91428…0149d`. Reuses `choose_insertion`,
`insertion_variant`, `translation_variant`, `validate_variant`, `_shift_tuple`, `make_deleted` from
`scripts/splicebert/run_p3_shift_diagnostic.py`; `CouplingTask`, `CORRUPT_ALT`, `gboot` from
`src/gfmcd/splicing/circuit.py`. New: deletion variant, window-mask profile scan, factorized GT
margin, anchor-model comparison. Results write a manifest; nothing overwrites the record.
