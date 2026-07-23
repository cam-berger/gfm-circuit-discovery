# B1 v2 — redesigned value-main OV estimand — focused 3-lens re-audit synthesis (2026-07-22)

> **Resolution (2026-07-22, applied):** the must-fixes below were applied in-place to
> [`../preregistrations/PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR_V2.md`](../preregistrations/PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR_V2.md),
> now a **SPLIT-FREEZE** — behavioural arm FROZEN_RUNNABLE, OV arm design-frozen pending its `--smoke` gate on
> the main checkout. Chiefly: the value-main was pinned to the **simple single-row `DI−DD`** (the symmetric form
> had re-imported ½·Δ_bind → +0.073 not −0.005 on the C35 router, M-1a); the router-negative moved to the
> readout scale with `SESOI_v` pre-declared ≥0.02 (M-1b); both OV nulls made covariance-matched (M-2); and the
> ill-posed readout-agnostic L1 demoted to a reported diagnostic. This synthesis is retained as the re-audit
> record of how those corrections were reached.

**Reviews:** [`../preregistrations/PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR_V2.md`](../preregistrations/PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR_V2.md)
(DRAFT, not frozen). This is a **focused** re-audit of one thing: the v2 OV estimand that replaces v1's
non-identifiable `ΔO_h = A_I·V_I − A_D·V_D` (the R23-class routing/OV conflation the 5-lens B1 red-team killed)
with the **C35 (P3.3) value main effect** — on the carrier-deleted base, restore a reader head's carrier-intact
values with attention held at the deleted pattern, and attribute an OV write only to that value-restore
contribution; the interaction `Δ_bind` is architecture-forced (R23), reported not interpreted. Three
independent adversarial lenses (forced-identity, calibration-validity, power-feasibility) tried to KILL the v2
estimand against the real C35 code + data.

**Outcome: the v1 fatal is genuinely DISSOLVED, but v2 is NOT freezable as written.** The core value-main move
is architecturally sound — it vanishes for a pure router by construction, not by luck — so this is not a DEFER.
But two load-bearing specification defects (one convergent across two lenses) sit exactly on the pass/fail line
of the anti-R23 anchor and the L2 specificity null, and both are fixable pre-run on the frozen checkpoint.
Disposition: **FIXABLE_THEN_RUNNABLE.**

The two arms earn opposite verdicts, as in v1:

- **Behavioural arm** (`ΔC(D10)` coupling under BAND_P10 deletion): **FROZEN_RUNNABLE.** Power is not the wall
  v1 feared — the sizing-pilot gate (must-fix #5) is already in v2, and the paired within-example contrast on a
  near-inert deletion needs only modest pairing (ρ>0.42), not the ρ≥0.98 v1 assumed.
- **OV/mediation arm** (value-main → `S_ov` onto `U_info`): **FIXABLE_THEN_RUNNABLE.** The estimand is
  identifiable in principle; the blockers are (M-1) an internally inconsistent value-main definition that
  straddles M-vmain's pass/fail line, and (M-2) an isotropic L2 sufficiency null that contradicts the prereg's
  own must-fix #4 / M-covnull. Both are pre-run design edits.

| Lens | Verdict | Headline |
|---|---|---|
| Forced-identity | fixable | Core move dissolves v1's forced identity (value-main = 0 iff `V_I=V_D`, a router-null by construction). BUT `Mv_h` is defined two incompatible ways — the line-101 **symmetric** formula reintroduces ½·`Δ_bind` (the forced R23 term) and gives **+0.073, not a null**, on the C35 router; only the **simple single-row** form reaches the −0.005 anchor the design relies on. |
| Calibration-validity | fixable | Sensitivity+specificity skeleton is correct (live-channel `PP-ovplant`, router-negative `N-router`, plumbing anchor `M-av-plumb`). BUT the L2 `S_ov` null is **isotropic** (`null_recoveries`→`random_subspace`), not covariance-matched — contradicts must-fix #4 / M-covnull and trivializes the L2 confirmatory-positive bar. N-router is cross-feature/cross-layer (period-3 L3H1, not an in-family BAND_P10 router). |
| Power-feasibility | fixable | **Behavioural arm = GO (marginal)** — ρ>0.42 suffices, sizing-pilot-gated, both branches pre-committed. **OV arm = MARGINAL/conditional-GO** — tractable at MPS, honestly downgraded to bounded one-sided "not detected"; modal outcome is Endpoint B (R_1024 may be ∅), which is a legitimate pre-committed result, not a block. Router-negative imports a **coupling-scale** −0.005 to bound a **head-output-scale** L1 estimand — a level mismatch. |

---

## The convergent load-bearing issue (verified against code + data)

### M-1 (CONVERGENT: forced-identity major + power-feasibility major) — the value-main estimand is under-/mis-specified at exactly the anchor that excises R23

Two lenses hit the same seam from different sides. Both are verified.

**(a) The line-101 formula is the SYMMETRIC two-row main; the prose, the causal counterpart, and the C35 anchor
are the SIMPLE single-row main.** Verified:

- Prereg line 101: `Mv_h = ½[(O_h(A=D10,V=I) − O_h(A=D10,V=D10)) + (O_h(A=I,V=I) − O_h(A=I,V=D10))]` — the
  **symmetric** average of both attention rows.
- Prereg line 106–107: `s_v^h` is "restore carrier-intact values, **holding attention fixed** … attention held
  at the deleted pattern" = the **simple** single row `(A=D10,V=I) − (A=D10,V=D10)`.
- C35 code `run_p3_av_decomposition.py:134`: `value = DI - DD` — the **simple** single-row main. The
  N-router anchor (prereg line 202) cites this as "value main −0.005."
- Verified arithmetic on `results/p3_av_decomposition.json` (confirm split, the −0.005 the design relies on):
  - simple main `DI − DD` = 0.024072 − 0.029052 = **−0.00498** (:91–97 vs :77–81) → matches the JSON
    `value_main` −0.00498 (:112–118). This is a **genuine null**, CI [−0.0132, +0.0034] straddles 0.
  - symmetric main `½[(DI−DD)+(II−ID)]` = ½[−0.00498 + (0.46446 − 0.31262)] = ½[−0.00498 + 0.15184] =
    **+0.07343** — **NOT a null.**
  - identity check: symmetric = simple + ½·`Δ_bind` = −0.00498 + ½(0.15682) = +0.07343, with
    `Δ_bind` = `II−ID−DI+DD` = **+0.15682** (:119–125, matches JSON). `Δ_bind` is the **architecture-forced**
    R23 term (code note :146–147; CLAUDE.md R23).

So **if the implementation follows the line-101 symmetric formula, the "value main" carries ½ of the forced
`Δ_bind` back into the estimand** — the exact R23 contamination v2 claims to remove — and **M-vmain (line 170)
FAILS on the known router** (+0.073 ⊄ ±SESOI_v for any reasonable SESOI_v). The −0.005 the design leans on is
reachable **only** with the simple single-row form. (Note: at the pure head-output level where `O = A·V` is
exact, both forms vanish for a router `V_I=V_D`; the contamination bites because M-vmain/N-router are
benchmarked against the C35 **coupling-margin** cells, where downstream nonlinearity makes `Δ_bind` large — see
(b) — so the symmetric form is +0.073 there. The design must not straddle the two.)

**(b) Even the simple −0.005 is a COUPLING-margin quantity; B1's L1 `s_v`/`f_v` is a HEAD-OUTPUT-residual
quantity.** Verified: `run_p3_av_decomposition.py:61–67,114–116` — every cell is `task.margin(...)` on the full
forward pass with Q/K/V patched, i.e. a **coupling margin**, not the head's `W_O`-projected output norm. B1's L1
estimand `f_v^h = ‖Mv_h‖ / ‖O_h(I) − O_h(D10)‖` (prereg line 105) is a **head-output** norm ratio. L3H1 is a
router **at the coupling projection** (C35/R24); its period-3-dependent value component can be nonzero on its
own `W_O` output while projecting to ≈0 coupling. Importing −0.005 as the L1 null bound is therefore an
unjustified level transfer — and if L3H1's head-output `s_v` is not null, N-router fires, M-vmain fails, and the
OV arm exits to Endpoint B **at the smoke gate** for reasons unrelated to R_1024.

**Why this is fixable, not fatal:** the simple single-row value main `DI − DD = A_D·(V_I − V_D)` is zero **iff**
`V_I = V_D` — a router-null by construction, architecturally non-forced in the correct direction (the forced
piece is `Δ_bind`, explicitly quarantined). The estimand is identifiable; it just must be pinned to the simple
form end-to-end and its router-null bound must be **measured** at the L1 head-output scale in `--smoke`, not
imported from the coupling scale.

---

## The other must-fix (calibration lens major, verified)

### M-2 — the L2 sufficiency null is ISOTROPIC, contradicting must-fix #4 / M-covnull

Verified:

- Prereg line 118: L2 sufficiency `S_ov^h` routed "vs `null_recoveries` (add-one p)."
- `src/gfmcd/splicing/anchor_gates.py:29–35`: `null_recoveries` calls `random_subspace(D, k, rng)`.
- `src/gfmcd/splicing/subspace.py:38–41`: `random_subspace` = "Uniform random orthonormal U" via QR of a
  standard-normal draw — **isotropic over the full 512-dim residual**, NOT covariance-matched.
- Internal contradiction: prereg line 117 sends `Ω_h` "vs the **covariance-matched** null," but the very next
  line (118) reuses the isotropic `null_recoveries` for `S_ov`. M-covnull (line 172) reads on "the null"
  globally and is thereby **violated by the L2 arm**.

The value-restore component `Mv_h` lives in the head's low-rank OV output range (rank ≤ d_head=32) inside a
512-dim residual, so an isotropic 512-dim random subspace almost never touches that manifold or `U_info` ⇒ its
`S_ov` recovery ≈ 0 ⇒ the real value-restore beats it **trivially**. This is exactly the specificity hole
must-fix #4 closed for the L1 null but left standing on the **L2 confirmatory-positive path** (A-OV promotion
requires `S_ov` beats `null_recoveries`, line 253). Backstop: A-OV also requires the coupling arm to fire
positive, which the behavioural prior says is ≈inert (deletion *raises* coupling), so a spurious `S_ov` alone
cannot manufacture A-OV — but the L2 specificity must still be fixed to match the design's own contract.

---

## Minors (fixable, none decision-bearing)

1. **N-router is cross-feature/cross-layer (calibration major, verification-conservative).** Period-3 L3H1
   (deep L3, BAND 339–345) is not an in-family BAND_P10/L0 routing-only head, so a null there does not certify
   value-main is non-inflated for R_1024. Worse, L3H1's **discovery**-split `value_main` CI **excludes 0**
   (−0.01055, CI [−0.02033, −0.00020], JSON :60–66); only the confirm split (−0.005, CI incl 0) is the clean
   null. This can only make the anchor *fail* (→ conservative Endpoint B), not manufacture a false positive.
   Fix: add an in-family BAND_P10 routing-only negative, or bound M-vmain's inference to "the pipeline can null
   a known router" and pre-declare SESOI_v against the confirm-split scale.
2. **`f_v` is a norm ratio ∈ [0, ∞), not a partition fraction** (forced-identity minor). `‖Mv‖`, `‖Ma‖`,
   `‖Δ_bind‖` do not sum, so `f_v` can exceed 1 when routing/value mains partially cancel; the "fraction of the
   head's total deletion effect" wording (line 108) reads misleadingly. Not decision-bearing (A-OV rests on
   `s_v` beating the null, not `f_v`). Fix: label it a diagnostic ratio or use a signed projection.
3. **A-null prose drops the L0 qualifier** (forced-identity minor). Section 6 licenses A-null when "no
   carrier-specific value write on **ANY L0 reader**" (line 260, per must-fix #7 restricting per-head OV to L0),
   but the conclusion (lines 261–264) says "writes nothing carrier-specific onto the donor readout" unqualified.
   Fix: add "via any L0 reader"; the model-global coupling-equivalence arm backstops the "not used for coupling"
   clause.
4. **PP-ovplant does not certify threshold sensitivity** (calibration minor). Its L1 half is near-trivial
   plumbing (a forward-pass value diff recovers any planted value by construction) with plant norm merely
   "controlled," not pinned to MDE/SESOI_v; its `S_ov` half risks the P3.4 off-manifold trap (a random-direction
   write is off-manifold ⇒ `S_ov`≈0 even for a "true" plant). Fix: pin plant norm to SESOI_v, scope PP-ovplant
   to the L1 channel, and note L2 `S_ov` sensitivity to a carrier-derived (non-`U_info`) write is uncalibrated
   (PP-U is on-manifold-but-circular).
5. **PP-read is an easier-feature floor** (power minor). Period-3 post-softmax survival does not license
   inferring period-10 (≈10.63) survival; R_1024 = ∅ → Endpoint B remains a live modal outcome. Fix: foreground
   this; PP-read is a sensitivity floor, not evidence for period-10 survival.
6. **OV L1 compute un-budgeted** (power minor). The per-head C35 2×2 (n≈120 for ONE head in C35) is repeated
   across ~7 L0 readers + M-read establishment + L2 patching + Preflight R + equivalence cohort, with no
   wall-clock estimate (§7). Fix: add a per-head × |R_1024| × n sizing note; the L0-restriction (must-fix #7)
   already bounds the head count.

---

## Behavioural arm (FROZEN_RUNNABLE — verified)

Not a power wall. `ΔC(D10)` is a paired within-example, gene-clustered contrast on a near-inert deletion:
`results/phase-lattice/carrier_dose.json` — control_band = [94…100] = BAND_P10, coupling 0.9460 (α=1.0) →
0.9700 (α=0.0), `|point|` ≈ 0.024, monotone, deletion *increases* coupling. The freeze gate
`HW < SESOI_C − |point|` = 0.05 − 0.024 = 0.026 at n=256 needs only ρ > 0.42 (C8 single-arm HW 0.0198@n384 →
0.0242@n256 → paired 0.0242·√(2(1−ρ)) < 0.026 ⇒ ρ > 0.42), not v1's assumed ρ≥0.98. The genuine risk is
`|point|`≈0.024 near SESOI_C=0.05 (whole-CI-inside can fail INTO the "positive → re-select control band"
branch), and **both branches are pre-committed** (risk-2, §6). The sizing pilot (must-fix #5) is already a hard
pre-freeze gate. This arm is freezable as written.

---

## MUST-FIXES (pre-run, all MPS-reachable on the frozen checkpoint)

1. **Pin the value-main OV estimand to the SIMPLE single-row form end-to-end** — replace the symmetric `½[…]`
   in the line-101 `Mv_h` formula with `(A=D10,V=I) − (A=D10,V=D10)`, matching the `s_v` prose (line 106–107)
   and the C35 anchor (`run_p3_av_decomposition.py:134`, `value = DI − DD`). Propagate to `f_v`, `s_v`,
   PP-ovplant recovery, and the L2 value-restore patch. (M-1a)
2. **Measure the router-negative at the L1 HEAD-OUTPUT scale in `--smoke`; do not import the −0.005 coupling
   figure.** Add an explicit memorial that L3H1's head-output `s_v` reproduces the SIMPLE main null (not
   +0.073) before it is trusted as N-router; if L3H1's output-scale `s_v` is not null, re-scope M-vmain or pick
   a head-output-null router. (M-1b)
3. **Specify a COVARIANCE-MATCHED `null_recoveries` variant for `S_ov`** (draw the null rank-k subspaces from
   the head's OV-output empirical covariance, not `subspace.random_subspace`), or state that A-OV's L2
   promotion is licensed only against that covariance-matched null. Resolve the line-117/118 inconsistency so
   M-covnull holds on the L2 arm. (M-2)
4. **Bound N-router / SESOI_v** — either add an in-family BAND_P10 routing-only negative, or explicitly restrict
   M-vmain's inference to "the pipeline can null a known router" and pre-declare SESOI_v against the confirm-
   split scale so L3H1's discovery-split −0.0106/CI-upper−0.0002 cannot spuriously void the arm.
5. **Wording/scoping tightenings** (minors 2–6): `f_v` as a diagnostic ratio; L0 qualifier on the A-null
   conclusion; PP-ovplant pinned to SESOI_v and scoped to L1 with L2 `S_ov` sensitivity flagged uncalibrated;
   PP-read foregrounded as an easier-feature floor; a per-head compute budget in §7.

---

## Recommendation

**FIXABLE_THEN_RUNNABLE.** The v2 redesign genuinely dissolves v1's convergent fatal: the simple single-row
value main is a router-null by construction (zero iff `V_I=V_D`), non-forced in the correct direction, with the
forced `Δ_bind` explicitly quarantined — the C35 data corroborate it empirically (L3H1 simple value main −0.005,
a genuine null, beside routing +0.284 and forced `Δ_bind` +0.157). No convergent fatal survives; nothing here is
a frozen-checkpoint dead end. **The behavioural arm is FROZEN_RUNNABLE now.** The OV arm needs the five
must-fixes — chiefly the estimand-definition pin (M-1) and the covariance-matched L2 null (M-2), both pre-run
edits — before freeze. The modal OV outcome may still be a bounded A-null or Endpoint B (R_1024 may be ∅); that
is a legitimate, pre-committed honest result, not a reason to block freeze. Apply must-fixes 1–5, re-run the
`--smoke` gate (M-del, M-av-plumb, the head-output-scale M-vmain, PP-ovplant, M-ctrl, M-p3contrast, M-det,
Preflight R non-empty on period-3), then freeze.
