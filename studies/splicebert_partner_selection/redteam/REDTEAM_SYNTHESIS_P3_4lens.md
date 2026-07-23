# P3 (carrier causal role) — 4-lens design red-team synthesis

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

Adversarial DESIGN red-team of `../preregistrations/PREREGISTRATION_P3_CARRIER_CAUSAL_ROLE.md` (v1), 2026-07-17, four
independent lenses (identifiability, statistical, estimand↔intent, calibration/plumbing). **All four
verdicts: `fixable`. Zero global fatals — but TWO estimand-level FATALs that retire the flagship
endpoint.** This file records the load-bearing findings and how prereg **v2** resolves each. Full lens
JSON is in the session transcript.

## Headline: the "address × payload binding" endpoint is NOT identifiable — RETIRED

The memo named P3.3 (the L3H1 attention×value 2×2) "the decisive experiment." The identifiability lens
shows it is **architecture-forced**:

- **FATAL-1 (Δ_bind is forced).** L3H1's output is bilinear `O = A·V`, so the 2×2 DiD is the exact
  identity `Δ = (A_I−A_D)(V_I−V_D)` — non-zero whenever *any* perturbation moves both A and V. There is an
  energy-matched (E) null for the *main* deletion but **none for the interaction**, so the same positive
  `Δ_bind` would arise from the junk-band E or any random PE perturbation. A positive `Δ_bind(C)` is NOT
  evidence of binding.
- **FATAL-2 (payload_source is non-identifiable).** No delivery-independent, carrier-uncontaminated
  observable exists for the acceptor-local payload: the carrier feeds the generator, and MLP2 is dual
  address+payload (C29). So "independently-constructed payload" cannot be licensed.
- Supporting (identifiability majors): the P3.3 **value main effect is the R15 position-in-V confound**
  (V carries position ⇒ the A/V axis is *not* an address/payload axis); **`S_A` restoration under `A_I` is
  forced** (`S_A = f(A)` alone); **`U_info(D)` is an SVD of ~0 signal** (degenerate under M1 collapse).

**Resolution (v2):** the "address × payload BINDING" claim and the "independently-constructed payload"
clause are **withdrawn as non-identifiable** (logged as **R23**). P3.3 is **reframed** to an *A-vs-V
routing decomposition on coupling*, a **conditional-mediation** quantity (never "binding"), and is
runnable only *with an importance-matched interaction null*. Construction-vs-binding → **Endpoint B
(unresolved)**. This mirrors R15/R21/R22: report the non-identifiable/composite outcome, do not launder it.

## Per-lens load-bearing issues → resolution

### Identifiability (verdict fixable; 2 fatal, 5 major)
| Issue | Resolution in v2 |
|---|---|
| Δ_bind architecture-forced, no interaction null (FATAL) | Retire "binding"; reframe A-vs-V; REQUIRE importance-matched interaction null (Δ_bind(D) CI-above the control-band *distribution*); report the bilinear floor `‖(A_I−A_D)(V_I−V_D)‖`. |
| payload_source non-identifiable (FATAL) | Drop it; construction-vs-binding → Endpoint B; conditional-mediation language only. |
| Value main effect = R15 position-in-V | Relabel axis "A vs V" (not address/payload); caveat V carries position; clean payload isolation unresolved → bounds claim. |
| S_A restoration under A_I forced | Remove S_A from P3.3 readouts; routing test on **coupling C** under (A_I,V_D) vs (A_D,V_D); flag Δ_bind(S_A)≡0. |
| U_info(D) degenerate (SVD of ~0) | Define U_info **once on intact**, only PROJECT; never refit in D; gate estimability by `energy_spectrum` floor; what-vs-whether only in non-degenerate (partial-recovery) arms. |
| E is spectral-power-matched, not importance-matched | Match the generic-damage counterfactual on **downstream MLM cost**; use the control-band **distribution** spanning importance; require treatment to exceed the whole distribution (add-one p). |
| P3.2 irrelevant window not mod-3-phase-matched; hybrid-P edge | Irrelevant control window **mod-3-phase-matched** to the acceptor window; add a **decoy-acceptor window** control (splice-like local content, wrong identity). |
| Routing not falsifiable (C32); map non-exhaustive | State plainly: routing is the **maintained** hypothesis (C32); P3.1 tests only whether recognition *additionally* collapses; map non-exhaustive → mixed patterns to Endpoint B. |
| M4 diagonal-only | Add off-diagonal + head-identity arithmetic anchors (see plumbing lens). |

### Statistical (verdict fixable; 7 major, 1 minor)
| Issue | Resolution in v2 |
|---|---|
| R_Y unstable when denom `Y_I−Y_D` CI straddles 0 (robust only for C) | Per-Y **deletion-effect memorial M1_Y**: report R_Y only if the denominator's gene-CI excludes 0 by a Fieller-safe margin AND ≥ SESOI; else report on the **difference scale** (Y_a−Y_D). **R_Y reported only for Y=C** (denom ≈0.68, robust). |
| No n / no MDE | Add a POWER section: per-Y MDE from a pilot; Δ_bind interaction needs ~4× a main-effect n; ΔD_rec separation needs ~1000+ matched pairs (⇒ D_rec deferred). Every NULL call must be a *powered* equivalence (CI half-width < SESOI), not a wide CI. |
| FWER family under-counted (~35), P3.1 outside | Enumerate the full confirmatory family; **two families**: positive (max-T bootstrap, one-sided > SESOI) and equivalence (TOST, family-adjusted α, CI half-width < SESOI else "inconclusive"). |
| CI-3-way doesn't compose with Holm | As above (max-T for positives; TOST for equivalence). Replace every "arm-CI-separation D vs E" with **the paired within-example difference** CI excludes 0 and ≥ SESOI. |
| Δ_bind must be per-example paired | Compute `Δ_bind_i` per example, carry all 4 cells through **one** gene-resample; pin as a memorial. |
| Null-draw counts too small (0.04/0.032 floors) | Non-exhaustible (window/6-head) nulls ≥400; report exact floor for exhaustible single-head; anchor P3.2 specificity on the matched-window CI, size-matched null secondary. |
| SESOI single-scale | Per-Y SESOI on its own scale (C≈0.05 nat ≈5–10% of C8; S_A as fraction of intact mass; R_C dual threshold R≥0.2 AND abs ≥ absolute-SESOI). |
| AUC via DeLong (i.i.d.) | Gene-clustered bootstrap of the AUC (moot — D_rec deferred). |

### Estimand↔intent (verdict fixable; 8 major, 1 minor)
| Issue | Resolution in v2 |
|---|---|
| D_rec = residual-motif discriminability, not recognition; decoys not motif-matched or are real sites; no strong-motif pseudo-site decoy | **D_rec (recognition) DEFERRED** to a dedicated study: needs a pseudo-site decoy generator (new machinery), a position-invariant contrast, importance-matched E, ~1000+ pairs, gene-clustered AUC. Foundation sanity: `real_acceptor_decoy` available 1/16 — decoys are too sparse to power tonight. |
| M7 (D_rec>0.5) rewards leaky matching | Moot (deferred); if run, M7 ≥0.70 vs pseudo-site + a perfectly-matched-decoy≈0.5 specificity memorial. |
| "Partner unavailable" corruption artifact | Moot (deferred); one operator AG→CC (never [MASK]), symmetric to real+decoy. |
| Donor recognition arm unoperationalized | Moot (deferred); scope recognition to acceptor OR build a donor caliper first. |
| Recognition/routing not operationally separable (position-defined decoys) | Deferred; the position-invariant contrast is the fix when D_rec is built. |
| P3.1 map not MECE | Add explicit branches; mixed/partial → Endpoint B; bind the semantic map to the CI/SESOI classification. |
| Δ_bind consistent with non-binding (saturation/bilinearity/downstream-AND) | Retire binding (R23); pre-saturation readout (U_info projection magnitude) reported alongside the hard margin; health-gate each cell; null-head Δ_bind. |
| "Independently-constructed" overreach; no TOST | Retire the clause (R23). |
| "Coding-associated" is a P3.4 origin claim | Strike every origin adjective from the endpoints; cohort **not** conditioned on coding state; coding kept only as a recorded covariate. |

### Calibration / plumbing (verdict fixable; 5 major, 2 minor)
| Issue | Resolution in v2 |
|---|---|
| M4 validates only the diagonal (R21 trap); may patch the wrong head | Add **native value-overwrite off-diagonal anchors**: recomposed `A_D@V_I` == a real deleted-model forward with value overwritten by `V_I`, and `A_I@V_D` likewise on intact, to **<1e-6**; **head-identity anchor**: patched `(A_D,V_D)` head-1 context == `capture_head_outputs(deleted)[3][:,1]`. (My `capture_proj`/`patch_proj` route patches Q/K/V *inputs* and lets the module run natively ⇒ no manual softmax, one W_O, no double-√d — the SDPA/scaling pitfalls are avoided by construction; still assert eager + fp32 + training=False.) |
| PP3 restore-all→R≈1 is forced | Keep M5/PP3 as labeled-forced; add genuine memorials: **element-wise `P_hybrid` construction anchor** + **monotone dose-response** R_C(w): empty~0 → all~1, strictly increasing. |
| S_A / U_deliv links uncalibrated | Add **PP-SA** (a manipulation known to move S_A — corrupt/relocate the acceptor window — must change `attn_to_window` by the known amount) and **PP-U** (C16/C17 planted positive: patch clean-U → recover ~0.9 vs random rank-8 ≤0.05). |
| Health gate only on I/D/E globals | Extend health gating to **every** P3.2 hybrid and **every** P3.3 cell; note P3.3 edits L3H1 locally ⇒ global health necessary-not-sufficient, local validity rests on the M4 off-diagonal anchors. |
| E comb-read asserted globally; single vs distribution | Per-band comb-read gate at runtime (exclude bands <95%); treat E as a **distribution** (add-one p, R12). |
| Loose bars (M7>0.5, M4<1e-4, M8 lint) | M4 → **<1e-6** (repo anchors ~2e-6); M8 is a construction lint, not evidence. |

## Net design (v2)

- **CONFIRMATORY, runnable tonight, identifiable:** **P3.2 — the carrier's causal locus** (position-selective
  restoration; R_C on coupling; phase-matched-irrelevant + decoy-acceptor + monotone-dose memorials;
  per-hybrid health gate; gene-clustered ratio CI). This *is* Cam's "Where must the carrier be present?"
- **SECONDARY, runnable if plumbing smokes clean:** **P3.3 — A-vs-V routing decomposition on coupling**
  (off-diagonal + head-identity anchors; per-example paired Δ_bind; importance-matched interaction-null
  distribution). Reports A-vs-V main effects + whether the A×V interaction is carrier-specific — **not
  binding**.
- **CONTEXT (cheap):** **P3.1 chain** — coupling collapse (C8) + S_A collapse; routing = maintained (C32);
  P3.1 tests only whether recognition *additionally* collapses.
- **DEFERRED (documented, not run tonight):** **D_rec (recognition)** — needs a pseudo-site-decoy study;
  **construction-vs-binding** — non-identifiable (R23) → Endpoint B.
