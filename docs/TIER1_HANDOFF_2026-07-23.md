# Tier-1 handoff — the last substantive SpliceBERT loops

> **Tier: T2 — provenance/handoff.** Author: origin-evidence session (frame df5fc482), 2026-07-23.
> Branch: `tier1-handoff`. For the executing agent: read this top-to-bottom, then read the two
> controlling documents it points at (the G1 prereg and `docs/METHODOLOGY.md`) before writing any code.
> **Nothing here authorizes opening the sealed chr8/9 cohort. It stays sealed.**

## 0. Orientation — where the project actually is

The discovery program on the frozen SpliceBERT checkpoint is essentially complete. The live claim ledger
([`CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md)) is down to **two** open/bounded rows
(C12, C30), plus **one** bounded-but-arguable row (C38). Carrier origin was closed this session as
corpus-attributed (C39/C40, R25). Everything else is `closed` or `established`.

So "what's left on SpliceBERT" is three items, and only **one contains new science worth compute**. This
handoff scopes all three and tells you which to run, which to close, and — for the one worth running —
exactly what to do and what NOT to do. The three items are independent; they can be worked in any order.

| Item | Claim | Verdict of this handoff | Compute? |
|---|---|---|---|
| **A. Cross-example transduction** | C38 → G1 redesign | **RUN, gated** — the one substantive loop | Yes, MPS-feasible |
| **B. Payload semantic identity** | C30 | **CLOSE bounded** — non-identifiable on frozen ckpt | No (recommend) |
| **C. ≈10.63 general OV/MLM use** | C12 | **OWNER DECISION** — run frozen simple estimand, or close | Small if run |

---

## A. Cross-example transduction — the one loop worth compute (C38 → G1)

### A.1 State of the evidence (the numbers, verbatim from the artifact)

Controlling artifact: `results/splicebert_coupling/b3_cross_example_transduction_n512.json` (C38, n=512
pairs). What B3 established and what it could not:

- **Transfer exists and is well-behaved.** Dose-monotone: `R_xfer` = 0 → 0.537 → **0.931** → 1.224 across
  α = 0/0.5/1.0/1.5; dose slope β = 0.281; primary `R_xfer@1.0` dyadic CI **[0.828, 1.050]**. Calibration
  passed (planted-positive self-gain 0.346; permuted-label FPR CI includes 0).
- **Content-specificity is UNRESOLVED — this is the whole problem.** The mismatched-stratum contrast
  `R_mm_diff = 0.158` has dyadic CI **[−0.110, 0.331]** — straddles zero. The norm/energy-matched
  scramble still recovers a mismatched `R_xfer = 0.773` (CI [0.610, 1.040]) — it does **not** collapse.
  `mf2_confirmed = False`, `mf2_refuted = False`.
- **It is not a sample-size problem.** Powering n 260 → 512 did **not** shrink the dyadic half-width
  (0.201 → 0.221). The variance is **gene-heterogeneity-dominated**. More validation data cannot fix it —
  this is the load-bearing fact for the decision below.

**Interpretation on the record:** the transfer effect = a substantial **content-independent geometry
floor** + an **unresolved example-specific content component**, not separable on the partner-selection
validation split. C38 is therefore `bounded / INCONCLUSIVE at achieved precision`.

### A.2 Why B3 could not resolve it, and why G1 can (design, not effort)

The reason B3 stalled is anticipated and already written up in the redesign's R-block. Two effects are
*forced* by prior results and must never be read as evidence:

1. **A `U_info`-aligned injection raising the donor margin is near-certain** — C26 records rank-1 subspace
   overlap **0.970** between `U_info` bases derived on disjoint gene halves. A subspace that reproduces
   across genes at 0.97 is, to that extent, a **gene-generic readout axis**. So "injection moves the
   target" (B3's positive result) is substantially anticipated and is NOT evidence of example-specific
   transfer.
2. **The non-target-position control is architecture-forced to exactly 0** — `logits =
   cls(hidden_states[-1])` is bit-exact, so injecting at any non-donor final-residual position moves the
   donor margin by 0 by construction. It cannot gate anything.

**The discriminating estimand is therefore not "does the state move the target" but "does the
example-specific part add anything over a norm-matched gene-generic axis."** That is what G1 is built to
measure and B3 was not.

### A.3 What to execute — the G1 prereg

The redesign already exists, fully drafted:
[`PREREGISTRATION_G1_CROSS_EXAMPLE_TRANSDUCTION.md`](../studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_G1_CROSS_EXAMPLE_TRANSDUCTION.md)
(392 lines, **STATUS: DRAFT — NOT FROZEN, NOT RUN**). Read it in full. The essentials:

- **Primary estimand G1-B** (`Δ_spec = τ(REAL) − τ(MEAN)` at α=1): does the example-specific state beat a
  norm-matched gene-generic axis? Seven arms (REAL, MEAN, SELF, RANDOM, SCRAMBLE, SRC-SHAM, POS-SHAM),
  dose grid, two-way (source-gene × target-gene) dyadic bootstrap — **ratio of means, never
  mean-of-ratios**.
- **Three frozen outcomes** (§6 decision table): TRANSDUCING STATE / GENERIC READOUT AXIS / NO
  TRANSDUCTION. All three are legitimate, reportable results — a NULL here is a *material scope
  correction* to C16–C18/C23 (bounds their interventional reach to intra-example patching), not a failure.
- **Cohort:** targets from the audited Head–Context Atlas v7 internal-validation split (4,019 packets,
  1,486 genes, gene- and window-disjoint from discovery); sources from the Atlas discovery split
  restricted to canonical single-pair short introns. **Splits:** `D_src` = Atlas discovery autosomes ∉
  {chr8,9,11,16,19}; `H_dev`/`H_conf` = deterministic disjoint halving of Atlas internal-validation genes.

### A.4 The hard preconditions — do these before ANY confirmatory compute

These are the gates that killed the predecessor study (P3.4) and must not be skipped:

1. **Owner decisions (§11) — resolve BEFORE freeze.** Three are genuinely open and change the headline:
   (1) is `Δ_spec = τ(REAL) − τ(MEAN)` the primary, or the ROADMAP-literal "above matched-random and
   within-example"? This draft promotes `Δ_spec` because of C26's 0.970 overlap; the owner must confirm or
   revert. (2) additive injection vs component replacement (identical at α=1 SELF, diverge off-dose).
   (3) `SESOI_spec = 0.05` normalized is asserted, not derived — seal it from a `D_src`/`H_dev` pilot.
2. **Pre-declared MDE gate (the P3.4 fix).** Before `H_conf` is opened, compute the projected dyadic
   half-width for **every** contrast from `H_dev`. Any contrast whose projected HW ≥ its SESOI is
   **declared inconclusive in advance and not run.** No compute on an estimand that cannot reach a verdict.
   The variance anchor is measured, not assumed (`a5_uinfo_replication.json`: between-gene SD ≈ 0.22 ⇒ at
   ~700 confirmatory genes HW ≈ 0.023 dyadic vs SESOI 0.05, clears with margin).
3. **Sampling stability (R22, mandatory).** ≥5 independently seeded `D_src`/source-pool draws; report
   per-draw `τ(REAL)`, `Δ_spec`, across-draw SD. A result that reproduces on `H_conf` but not across draws
   is not a result.
4. **Smoke first (val sealed).** `--smoke` runs memorials + planted-positive + permuted-label FPR on a
   tiny `D_src`-only cohort; captures NO `H_dev`/`H_conf` residuals; writes no τ/Δ/verdict. Must pass
   before freeze.
5. **Freeze discipline.** Amendments are pre-run only, logged inline. Once any primary quantity is
   inspected, a fix makes the analysis outcome-aware verification, not confirmatory (the IC1/A2 precedent).

### A.5 Compute plan (MPS-feasible — this is why G1 ≠ P3.4)

Feasible at MPS budget for a structural reason: the primary injection is at the **final** donor residual,
and `cls(hidden_states[-1])` is bit-exact. So forward passes are needed only to **capture residuals**
(twice per target: clean+corrupt; once per source for SRC-SHAM) ≈ **7,500 forward passes** at ~1,500
targets + ~1,500 sources — same order as A5. The entire arm × dose × draw grid (~7 arms × 8 doses × 5
draws ≈ 280 per target) is **MLM-head evaluations on cached residuals** — seconds-to-minutes of CPU, not
model inference. Env `gfm-thrust1`, MPS, `PYTHONHASHSEED=0`, deterministic `sorted(set())` cohorts.
**Feasibility breaks** if the injection locus moves to an earlier layer (×100–200) — that variant is
deferred (§10), do not smuggle it in.

### A.6 Implementation start point

Fork `scripts/splicebert/run_b3_cross_example_transduction.py` (the B3 estimator — dose grid, dyadic
bootstrap, calibration gates already implemented) into `scripts/splicebert/run_g1_cross_example_transduction.py`.
The deltas from B3 → G1: add the MEAN arm (the discriminating null) and SRC-SHAM; source pool derived on a
gene pool that never contains a target; targets from the Atlas cohort rather than partner-selection val;
the §7 MDE pre-gate; ≥5 seeded draws. Outputs (per §10): `g1_cross_example_transduction.json` +
`.manifest.json` + `preflight_g1_cross_example_transduction.json`, and **register every tracked file in
`results/MANIFEST.csv/.json`** — `scripts/verify_manifest.py` and `tests/test_manifest.py` enforce this
(it was the test-failure this session; don't repeat it).

### A.7 The alternative the owner should weigh

G1 as drawn deliberately does **not** open the sealed chr8/9 cohort — it targets the Atlas
internal-validation split. That is the right call: the sealed cohort is one-shot, and the §7 anchor shows
the Atlas pool clears the MDE bar with margin. **If G1's MDE gate passes and the run still returns
GENERIC READOUT AXIS or NO TRANSDUCTION, that is the answer** — C38/C16–C18 reach is bounded to
intra-example patching, MECHANISM §2 is answered negatively, and there is no reason to spend the sealed
cohort. Only escalate to the sealed cohort if the owner judges content-specificity load-bearing for the
paper's central claim AND G1 comes back genuinely borderline (not equivalence-null). Default: **do not
open it.**

---

## B. Payload semantic identity (C30) — recommend CLOSE, do not run

C30 (`bounded (R22)`): the payload is acceptor-local and not explained by preregistered bulk composition,
but its semantic identity is unresolved; "splice-site strength" is withdrawn. MaxEnt showed only a weak,
sample-variable, non-dyadic-significant tendency.

The close-attempt prereg
([`PREREGISTRATION_B2_PAYLOAD_STEERING.md`](../studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_B2_PAYLOAD_STEERING.md))
was subjected to the house 5-lens design red-team (2026-07-22) and adjudicated
**FIXABLE_THEN_RUNNABLE, survivor = NO** — it states plainly that closing C30 is **NON-IDENTIFIABLE on a
frozen checkpoint at MPS budget**: a one-sided positive search re-running the same estimand cannot
distinguish naming the payload from confirming it.

**Handoff recommendation:** mark C30 **closed-bounded** (permanent: payload is acceptor-local, composition
excluded, semantic identity non-identifiable on the frozen checkpoint) rather than spending compute on a
non-identifiable search. Semantic identity is a next-project question (needs a different probe or an
external biological anchor, not more of the same intervention). If the owner disagrees, the path is to fix
the B2 draft per the red-team and re-adjudicate for a survivor — but do not run B2 as drawn.

---

## C. ≈10.63 general OV / MLM-output use (C12) — owner decision, small if run

C12 (`bounded, donor-scoped by C37`): the ≈10.63 carrier's OV write and behavioral function are measured
**for the donor circuit** (C37: OV value-write null + small suppressive coupling); its **general OV
content (MLM output)** remains open. Origin is now settled (C39/C40), which lowers the value of chasing
general OV — the interesting origin question is answered.

The relevant prereg
([`PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR_V2.md`](../studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR_V2.md))
is **split-frozen**: the confirmatory branch (simple single-row value main effect + carrier-band deletion)
is frozen and runnable; the **general masked-MLM-NLL** secondary estimand carries an explicit **MPS-wall
flag** (high-variance `ΔL`, same class of power wall that killed P3.4).

**Handoff recommendation:** two clean options, owner picks —
1. **Run the frozen simple estimand only** (single-row value main effect + band deletion on the MLM
   output), skip the MPS-wall-flagged general-NLL branch. Small compute, gives a bounded general-OV read.
2. **Close C12** as donor-scoped-terminal: general OV is low-value now that origin is closed, and the
   general-NLL branch is under-powered on MPS. Cleanest if the paper doesn't need a general-OV statement.

Do **not** run the general-NLL branch expecting a verdict — it is flagged unpowered for a reason.

---

## What NOT to do (guardrails for the executing agent)

- **Do not open the sealed chr8/9 cohort.** No item here authorizes it. G1 targets the Atlas
  internal-validation split by design.
- **Do not re-run B3 to try to shrink the C38 interval.** It is variance-limited, not n-limited
  (0.201 → 0.221 when n went 260 → 512). Only the G1 *redesign* changes the estimand.
- **Do not read a G1 NULL as failure.** All three §6 outcomes are reportable; a NULL is a scope correction.
- **Do not skip the MDE gate or the ≥5 seeded draws.** These are the P3.4 and R22 lessons, written into the
  prereg so they cannot be relearned after the fact.
- **Do not use "surgery" in new code/prose for band-deletion work** — it tripped a content-safety filter
  this session. Use "band-deletion / carrier-transfer."
- **Register every new `results/` file in the MANIFEST** (`scripts/verify_manifest.py`,
  `tests/test_manifest.py`). Missing rows fail CI.

## Deliverables the executing agent should produce

1. **G1:** resolved §11 owner decisions (logged), MDE-gate output, smoke pass, then (if gate passes) the
   confirmatory run → `g1_cross_example_transduction.json` + manifest + preflight, per-arm/dose/context
   tables, the achieved MDE for every contrast, ≥5 seeded draws, and a REFLECTION re-anchored to §0. A new
   CLAIMS row (C41?) at whatever evidence level the §6 decision reaches; reconcile C38's provisional
   branch state at merge.
2. **C30:** a CLAIMS/CORRECTIONS closure entry (or, if owner overrides, a red-team-fixed B2 draft).
3. **C12:** either the frozen simple-estimand result + CLAIMS update, or a closure entry.

## Provenance pins (for reproducibility)

- Checkpoint: `SpliceBERT.1024nt`, sha256 `2ad91428…0149d`, post-LN BERT-MLM 6L×16H, hidden 512, absolute
  learned P (1026). Env `gfm-thrust1`, MPS.
- Variance anchor: `a5_uinfo_replication.json` (rank-8 sufficiency 0.875 [0.853,0.901], n=400/320 genes).
- C38 controlling artifact: `results/splicebert_coupling/b3_cross_example_transduction_n512.json`.
- Atlas cohort: `ATLAS_COHORT_V1.md`, audit `atlas_context_matched_v7.audit.json` (PREFLIGHT_PASS).
