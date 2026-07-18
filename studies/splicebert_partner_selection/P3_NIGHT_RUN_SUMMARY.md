# P3 carrier program — overnight run summary (2026-07-17)

**For the morning read.** You gave 2 hours of autonomy to run the P3 carrier program from your memo. Here
is what happened, honestly scoped. Full detail: `DECISIONS_A0.md` (P3 section), `CLAIMS.md` (C34, R23),
`PREREGISTRATION_P3_CARRIER_CAUSAL_ROLE.md` (v2), `REDTEAM_SYNTHESIS_P3_4lens.md`.

## TL;DR

1. **Formalized P3.1–P3.3 as one prereg**, then ran a **4-lens design red-team BEFORE any run**. It caught
   two estimand-level FATALs that **retire your "decisive" P3.3 binding test as non-identifiable** — the
   cycle working exactly as intended, saving a forced-result. Logged **R23**.
2. **Reframed to the identifiable core and ran it (MPS):**
   - **P3.2 (where must the carrier be?) → the carrier's causal locus is SPATIALLY DISTRIBUTED, not at the
     sites** (C34). Well-powered, clean nulls, disc/confirm split.
   - **P3.3 (A-vs-V decomposition) → routing ≫ value**: the carrier's coupling contribution routes through
     L3H1's **attention pattern (where it looks), not its values (payload)** — the routing-carrier picture,
     causally.
3. **Deferred honestly:** recognition (D_rec — needs a pseudo-site-decoy study; decoys too sparse to power
   tonight); construction-vs-binding (non-identifiable); P3.4 origin (separate, as you specified).

**Net:** the period-3 carrier is a **distributed relative-positional ROUTING coordinate** that shapes
L3H1's attention address across the donor–acceptor span and does **not** carry the payload. This bounds the
SpliceBERT carrier account at the causal-locus + A-vs-V level. **Next per the depth-first spine: transfer**
(the deferred Caduceus package).

## The design red-team retired the flagship endpoint (R23) — the most important outcome

Your memo called the P3.3 L3H1 attention×value 2×2 "the decisive experiment" for address×payload binding.
The identifiability lens showed it is **architecture-forced**: L3H1's output is bilinear `O = A·V`, so the
2×2 interaction `Δ_bind` is the exact identity `(A_I−A_D)(V_I−V_D)` — non-zero for **any** perturbation
that moves both A and V, with **no carrier-specificity** (there's an energy-matched null for the main
deletion but none for the interaction; a junk band or random PE perturbation would give the same positive
`Δ_bind`). Separately, `payload_source` (meant to license "independently-constructed payload") has **no
delivery-independent, carrier-uncontaminated observable**. Both are new instances of R7/R15/R21.

I therefore **did not run a binding claim**. P3.3 was reframed to an **A-vs-V routing decomposition**
(conditional mediation; axis "A vs V", not "address vs payload" — V carries position, R15), and
construction-vs-binding was routed to Endpoint B (non-identifiable). Note: **C23's separate reconstruct-
inject binding result stands** — R23 is only about the carrier-deletion+patching design being unable to
establish binding. (All four lenses returned `fixable`; the full must-fix list is in the synthesis and was
applied to prereg v2.)

## P3.2 — carrier LOCUS: spatially distributed, not site-local (→ C34)

Position-selective restoration: in the period-3-deleted model, add the carrier delta `P_intact−P_deleted`
back at per-example windows; measure normalized coupling recovery `R_C = (C_a−C_del)/(C_intact−C_del)`.
Disc/confirm gene split (n=160, 144 genes), gene-clustered ratio CIs, health-gated. Plumbing memorials
EXACT (construction 1.5e-8; restore-none==deleted 0.0; restore-all→R_C 1.00 forced; monotone dose).

| Arm (±6 nt) | Discovery R_C | Confirm R_C | reading |
|---|---|---|---|
| donor window | +0.016 [0.007,0.027] | +0.019 [0.009,0.029] | site recovers ~2% |
| acceptor window | +0.016 [0.008,0.025] | +0.007 [−0.003,0.020] | ~1% |
| both windows | +0.035 [0.022,0.050] | +0.041 [0.023,0.065] | ~4% |
| **phase-irrelevant** (null) | +0.002 | +0.003 | ≈0 — specificity holds |
| **decoy-acceptor** (null) | +0.002 | −0.002 | ≈0 — wrong-identity site inert |
| all positions (forced) | +1.001 | +1.002 | memorial |
| dose (both, ±32) | +0.410 | +0.500 | recovery needs a broad span |

**Reading:** restoring the carrier only at the local donor/acceptor windows recovers **~2–4%** of coupling
(CI-excludes-0 but far below the R_C≥0.2 recovery SESOI); the phase-matched-irrelevant and decoy-acceptor
controls are ≈0 (so the small local effect is site-specific, **not** a hybrid-position-table edge
artifact); and substantial recovery requires restoring the carrier **across a broad span** (~40–50% only at
±32 nt, covering most of the 60–150 nt inter-site region). This is the P3.2 **"distributed upstream
organization"** outcome — a **well-powered positive** characterization (tight CIs, clean nulls), not an
underpowered null. Coheres with **C32** (address predominantly relative offset — a relative code needs the
carrier along the span) and **C33** (coupling irreducibly distributed). Honest caveat: the deletion
memorial M1 (≤5% residual) passes on discovery (3.3%) but is **marginal on confirm (5.5%)** from the
`min_effect` conditioning; the denominator stays robust, so the locus finding is unaffected.

## P3.3 — A-vs-V decomposition: the carrier routes through attention, not values

L3H1 factorial via `capture_proj`/`patch_proj` (patch head-1's Q,K to carrier-x → attention pattern `A_x`;
value to carrier-y → `V_y`; module runs natively). **Plumbing anchors EXACT 0.0** (M4 diagonal;
off-diagonal *native* value-overwrite; head-identity — pinning head-1 = L3H1), per the calibration lens.

**Confirmatory (n=120, disc/confirm gene split), deleted base:**

| | Discovery (n=62) | Confirm (n=58) |
|---|---|---|
| cell (A_D,V_D) baseline | +0.039 | +0.029 |
| cell (A_I,V_D) attention restored | +0.295 | +0.313 |
| cell (A_D,V_I) values restored | +0.028 | +0.024 |
| cell (A_I,V_I) both | +0.421 | +0.464 |
| **routing main** (A_I,V_D − A_D,V_D) | **+0.257 [0.172,0.351]** | **+0.284 [0.192,0.386]** |
| **value main** (A_D,V_I − A_D,V_D) | −0.011 [−0.020,−0.000] | −0.005 [−0.013,+0.003] |
| `Δ_bind` (reported, R23 — forced) | +0.136 [0.085,0.195] | +0.157 [0.097,0.221] |

**Two things worth your eye:** (1) the **value channel is ≈0** on both splits while the **attention channel
carries +0.26/+0.28** — a clean, well-powered dissociation. (2) `Δ_bind` came out **CI-positive** (+0.14 /
+0.16) — **exactly what R23 predicted** and why I did not run it as a binding test: the bilinear identity
guarantees a positive interaction for any A-and-V-moving perturbation. Had the original design run, this
would have been mis-reported as "binding." (Global health 0.49 across all four cells = the deleted-base
accuracy, unchanged by the single-head edit — healthy per P3.2's kept-fraction; local validity from the
exact M4 anchors.)

**Reading:** the carrier's contribution to coupling routes almost entirely through L3H1's **attention
pattern** (the positional address — *where it looks*), not its **values** (the payload — *what it
collects*). The value channel being ≈0 is a genuine, non-forced result (restoring V_I *could* have moved
coupling if the payload were carrier-dependent, and it doesn't) — coheres with C24/C28/C30 (payload is
acceptor-local content, generated by the L0→MLP2 backbone, not carrier-built) and C22/C25/C32 (attention =
positional geometry). The main effects are identifiable; only the interaction is forced (R23).

## Honest status / caveats / what I did NOT do

- **Binding**: not claimed (R23, non-identifiable by this design). The `Δ_bind` interaction is reported but
  not interpreted; a real test needs an importance-matched interaction null (deferred).
- **Recognition (D_rec)**: **deferred**. A valid assay needs a strong-motif *pseudo-site* decoy (new
  machinery), a position-invariant contrast, and ~1000+ matched pairs; the good decoys are sparse
  (`real_acceptor_decoy` available 1/16 in the sanity cohort). Not honestly runnable tonight.
- **P3.4 origin** (whole-codon masking, CDS/UTR/noncoding, codon slots): separate study, as you specified —
  not run, so a coding-origin result can't redefine the causal result.
- **Compute**: MPS, no CUDA. Cohorts sized to the budget (n=160 / 120), with disc/confirm gene splits; the
  sealed chr8/9 test was **not** opened.
- **Statistical fixes not all exercised**: prereg v2 specifies the two-family (max-T positive + TOST
  equivalence) inference and per-Y SESOI; the runs report gene-clustered CIs against the SESOI thresholds,
  but a full FWER-corrected family pass is specified for a publication-grade rerun.

## Independent verification (passed)

An independent adversarial verifier re-derived every number (routing/value/`Δ_bind` to 1e-12; all `R_C`
from arm means), confirmed BAND 339–345 is period-3 (bin 342 = 3.0000), and hard-tested both attack
surfaces: **verdict `minor_issues`, high confidence, no conclusion-flipping bug, numbers reproduce.** It
confirmed the P3.2 hook indexes windows *per-example* (no shared-position broadcast bug — donor+acceptor
additivity + monotone dose confirm placement), and that the P3.3 factorial builds `A_x@V_y` correctly
(`cell_II≠cell_ID` proves the value path is live, so value≈0 is a real dissociation, not a dead patch);
`min_effect` conditioning is *conservative* (biases against "distributed"). Three minor items (none flip a
conclusion): (1) M1 confirm-miss 5.5% — disclosed; (2) **value-main is best stated as sub-SESOI
equivalence** (|−0.011|≪0.05, wrong-signed; discovery marginally excludes 0, confirm includes 0) — a
practical zero, not a positive value effect; (3) E-band specificity is P3.1-scope (period-3 specificity
here rests on `delta` = the period-3 component *by construction* + the phase/decoy nulls + C8's established
energy-matched inertness). Nuance: "distributed" here means *spatially extended across the span* (dose
reaches ~50% at ±32 nt; full recovery only at all-positions), not literally uniform delocalization.

## Files (committed this session)

- `PREREGISTRATION_P3_CARRIER_CAUSAL_ROLE.md` (v2, reframed) · `REDTEAM_SYNTHESIS_P3_4lens.md`
- `scripts/splicebert/run_p3_carrier_locus.py` → `results/p3_carrier_locus.json`
- `scripts/splicebert/run_p3_av_decomposition.py` → `results/p3_av_decomposition.json`
- `CLAIMS.md` (C34 + R23) · `DECISIONS_A0.md` (P3 section) · `CLAUDE.md` (Current direction + mechanism)
- `studies/caduceus_partner_transfer/` (deferred transfer package, preserved earlier this session)

## Recommended next step

Per the depth-first spine, SpliceBERT's carrier account is now bounded. **Transfer** is the next study — the
hardened Caduceus-PS behavioral-transfer package is ready in `studies/caduceus_partner_transfer/` (needs a
compute decision — Colab CUDA — before running). If instead you want to close the recognition question
first, that's a scoped pseudo-site-decoy study (P3.1 L1), designed but not yet built.
