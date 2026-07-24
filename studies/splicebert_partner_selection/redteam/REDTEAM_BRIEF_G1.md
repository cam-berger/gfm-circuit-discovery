# G1 design red-team brief — cross-example transduction of the donor-side delivered state

> **Tier: T2 — provenance.** Redteam synthesis / design record. Not a canonical claim source.

> Input to the house design red-team (formalize-audit-verify step 2 / prereg §11-#8). Written AFTER the
> pilot (H_dev) so the red-team attacks the *concrete* design with real feasibility numbers, BEFORE any
> `H_conf` (confirmatory) read or freeze. `H_conf` is untouched. Attack the design, not the prose.

## Controlling documents
- Prereg (design of record): `preregistrations/PREREGISTRATION_G1_CROSS_EXAMPLE_TRANSDUCTION.md`
- Estimator: `scripts/splicebert/run_g1_cross_example_transduction.py`
- Pilot artifact: `results/splicebert_coupling/g1_cross_example_transduction_pilot.json`
- Smoke artifact: `results/splicebert_coupling/g1_cross_example_transduction_smoke.json`

## INTENT (north star)
Does the donor-side delivered state (C16–C18/C23/C26) behave as a **transferable, example-specific
information-bearing state** — derived on genes the target never contributed to, injected into a
gene-disjoint / context-mismatched target, moving its native donor margin dose-ordered and
*source-specifically* — or does any cross-example effect reduce to a push along a **gene-generic readout
axis**? (No payload semantics; R22 stands. No minimality; C33 stands.)

## Estimand, as implemented
- **Injection core = the A5/C26 estimator** (NOT B3's L3H1 value path). `U_D = top_k_subspace(H_cl−H_co at
  the donor, 8)` on D_src donors only; `s_j = project(H_cl,j−H_co,j, U_D)`; inject additively at the
  **final donor residual** `H_i(α)=H_co,i+α·s`, read via bit-exact `margin_hard(cls(H_i(α)))`.
- `τ(arm,α) = E_i[m_i(α)−m_i(0)] / E_i[coup_i]` — ratio-of-means, **two-way (target-gene × source-gene)
  dyadic bootstrap** (verbatim from B3), never mean-of-ratios.
- **PRIMARY: `Δ_spec = τ(REAL) − τ(MEAN)` at α=1, paired within-target.** MEAN = `project(mean_j diff_j,
  U_D)` rescaled to `‖s_j‖` per pairing (the gene-generic axis, stripped of example content).
- Arms: REAL, MEAN, SELF (ceiling; reproduces C26), RANDOM (matched-random floor), SCRAMBLE (U_D-coords
  of s_j permuted; norm+subspace preserved, content destroyed), SRC_SHAM (source built from a
  **non-partner** decoy-AG corruption), POS_SHAM (LABELED FORCED = 0; position-wise head).
- Additive ≡ component-replacement at α=1 SELF verified to 3.7e-9 (§11-#2 resolved: additive).

## Cohort (audited Atlas Head–Context v7; gene-disjoint by construction)
- **Sources D_src** = discovery **single-pair short-intron** (`target_partner_distance∈[60,150]`) **donor**
  packets: 300 packets / 297 genes (the A0→A12 characterization distribution).
- **Targets** = validation **donor** packets (all 6 contexts), 2013 packets / 1112 genes, deterministically
  halved by gene hash → **H_dev 561 genes/1001 packets**, **H_conf 551 genes/1012 packets**.
- Disjointness keys on `gene_id` (discovery vs validation is an 80/20 gene-hash split). **Sealed chr8/9 and
  excluded chr11/16/19 are absent from both splits — verified.** Atlas→CouplingExample adapter maps
  `donor_win→donor_pos`, `acceptor_win→next_acceptor_pos`; frozen coupling path uses only
  `.sequence/.donor_pos/.next_acceptor_pos`; `eligible()` (GTF GAP/GT filter) is bypassed by design.

## Mechanics — SMOKE PASS (D_src-only, val sealed)
M1 head-readout err **0.0**, M2 zero-dose **0.0**, M3 SELF-suff **0.72** (held-out half), M5 norm-match
**3.6e-15**, checkpoint unmutated. Planted-positive monotone [0.225,0.443,0.819], slope +0.79.
Permuted-label FPR **0.02 ≤ 0.05**.

## Pilot result (H_dev; development split — allowed to see)
| arm | τ | CI | | contrast | est | CI |
|---|---|---|---|---|---|---|
| REAL | +0.644 | [0.551,0.750] | | **Δ_spec = REAL−MEAN** | **−0.336** | **[−0.473,−0.229]** |
| MEAN | +0.979 | [0.855,1.139] | | REAL − RANDOM | +0.596 | [0.504,0.701] |
| SELF | +0.882 | [0.852,0.911] | | REAL − SCRAMBLE | +0.678 | [0.514,0.864] |
| RANDOM | +0.047 | [0.015,0.077] | | | | |
| SCRAMBLE | −0.035 | [−0.168,0.092] | | | | |
| SRC_SHAM | −0.005 | [−0.174,0.177] | | | | |

- SELF **0.88** reproduces C26's held-out 0.875. RANDOM **0.05** matches the C16 anchor ≤0.05.
- **β_dose(REAL)=0.265, monotone (pre-saturation) = true.**
- **Identifiability:** `cos(s_j,s̄)` mean 0.43 / p50 **0.50** (p10 −0.18, p90 0.88). Energy spectrum of
  {diff_j}: top-1 **0.46**, top-2 0.57, top-8 0.73 → the example-specific component is **real (54% off the
  top axis), not collinear-by-construction**.
- **MDE gate (projected to H_conf ~551 genes):** Δ_spec reaches_verdict=TRUE (effect-resolvable,
  |−0.34|−0.12=0.21>0.05); REAL, MEAN, SELF, RANDOM, REAL−RANDOM, REAL−SCRAMBLE all verdict-capable.
  **SCRAMBLE and SRC_SHAM *absolute* τ are instrument-limited** (HW>SESOI) — their *contrasts* resolve.

**Provisional (H_dev) verdict: GENERIC READOUT AXIS.** Transfer is real and content-dependent (REAL ≫
RANDOM, SCRAMBLE, SRC_SHAM) but the example-specific part does **not** beat — is beaten by — a norm-matched
gene-generic axis (Δ_spec ≈ −0.34). Coheres with C26's 0.970 cross-gene overlap; a material scope
correction bounding C16–C18/C23 to a shared readout direction, not example-specific transduction.

## §11 owner-decision defaults taken (to confirm; all arms computed regardless of #1)
1. **Δ_spec primary** (over ROADMAP-literal) — the pilot vindicates it: the ROADMAP-literal "above
   matched-random" reads POSITIVE (REAL−RANDOM +0.60) and would mislabel this a success, while Δ_spec
   reveals it is generic. 2. **Additive** (algebra+empirics). 3. **SESOI_spec=0.05** (Δ_spec effect 0.34 ≫;
   comfortably resolvable). 4. **Atlas val only; chr8/9 sealed**. 5. contexts secondary max-T. 6. single-pair
   short sources. 7. **no** A9 re-run. 8. this red-team. 9. tolerances from pilot. 10. CLAIMS wording per §6.

## The attacks the red-team MUST press (rank FATAL / MAJOR / MINOR; give the failing scenario)
1. **The MEAN-denoising / fairness crux.** MEAN is the population-average direction (n≈300, low variance);
   REAL is one source (high variance). Is Δ_spec<0 a *denoising artifact* (comparing a denoised vector to a
   noisy one) rather than evidence the example-specific content is uninformative? Is the GENERIC_READOUT_AXIS
   scope ("example-specific transduction NOT ESTABLISHED", not "absent") the correct, non-overclaiming read?
   Would a **parallel/perpendicular decomposition** of s_j (component along s̄ vs orthogonal) settle whether
   the off-axis example-specific part is genuinely non-restoring? Should that arm be added before freeze?
2. **Statistics.** Dyadic bootstrap correctness on heavy-tailed coupling (mean 0.41, median 0.05);
   ratio-of-means robustness; the MDE-gate effect-resolvability rule; SRC_SHAM/SCRAMBLE instrument-limitation
   honesty; multiplicity across 7 arms + 6 contexts; the ≥5-draw stability (R22) sufficiency.
3. **Estimand ↔ intent.** Does Δ_spec measure *example-specific transduction*? Is short-source→varied-target
   matching sound given the primary is paired-within-target? Is bypassing `eligible()` (admitting GC-AG
   donors, no GAP filter) a threat? R22/R23/R24 compliance (no semantic naming; no A×V binding claim; no
   value-null over-generalization).
4. **Calibration + plumbing.** Does the planted-positive travel the *interpreted* channel (final-residual
   add), unlike B2's fatal? Does permuted-label FPR test the REAL-vs-MEAN decision? Is POS_SHAM=0 honestly
   FORCED? Is the SRC_SHAM decoy-AG corruption a valid non-partner content null, or could it accidentally
   destroy real signal / find no decoy (fallback behavior)? Atlas adapter correctness (donor/acceptor
   mapping, the 1 GC donor, sham fallback when no decoy AG exists).

Return per lens: FATAL/MAJOR/MINOR findings with a concrete failing scenario, then a one-line disposition
(RUNNABLE_AS_IS / FIXABLE_THEN_RUNNABLE / NOT_IDENTIFIABLE) and survivor = YES/NO.
