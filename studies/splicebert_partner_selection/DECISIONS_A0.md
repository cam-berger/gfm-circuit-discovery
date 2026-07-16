# A0 decisions & running log — stage-wise provenance of L3H1's acceptor targeting

Frozen protocol: `PREREGISTRATION_A0_TARGETING_PROVENANCE.md`. This file is the post-unblinding
log and the setup-time reasoning; the prereg is frozen and only takes **pre-run** amendments.

---

## The consolidated direction (the anchor we reflect against)

**Where we are.** The donor circuit `L2H15 (ADDRESSER) → L3H1 (READER) → MLP2 → downstream
transporters → MLM` is established at the routing level: ~all of L2H15's measured effect on the
donor coupling is mediated through layer-3 attention, overwhelmingly L3H1 (`l2_accounting.json`,
calibration = 1.000). The acceptor-address experiment then found the discriminating fact (C20):

> L2H15 supplies ~0.66 of the **coupling** but only ~0.11 of the matched-decoy attention
> **preference**.

**The open question this leaves — and the trap.** That gap does *not* license "the preference is
built diffusely." It is equally consistent with the preference being **pre-resident** in the
residual reaching layer 3 (embeddings / early-layer site features), with L2H15's real job being
to refine the key/query and supply payload — a *coupling* role, not a *targeting-construction*
role. A component-attribution scan run now would manufacture a distributed constructor whether or
not one exists. The non-additivity of the serial `L2H15→MLP2→L3H1` recoveries (key 0.43, query
0.18, MLP2 0.65 all measure overlapping flow) means the existing numbers cannot decide this.

**So the direction is: settle provenance before attribution.** The ordered plan:

1. **A0 — provenance (this experiment).** Reconstruct L3H1's donor-query targeting score `S_A`
   stage-by-stage through the residual it receives, by resample-patching (post-LN ⇒ intervention,
   not linear attribution). Decide **where** the `a⁺`-over-real-acceptor preference becomes
   *sufficient*, and whether it is carried by the **key** side, the **query** side, or a **Q×K
   interaction** — via three curves `S_K/S_Q/S_J`, a reciprocal leave-one-stage-out necessity
   check, and an absolute-attraction diagnostic. A0 **routes** everything after it (decision
   table, prereg §5): pre-existing positional preference, pre-existing sequence salience,
   constructed keys, or constructed queries are four different next-experiments.
2. **A1 — targeted attribution**, *restricted to whatever stage A0 says matters* (never a blind
   full scan): fast attribution to rank, exact patching to confirm, joint restore + leave-one-out,
   MLP-specific nulls. Skipped entirely if A0 says the preference is pre-resident.
3. **Q×K×V joint accounting** to replace the non-additive serial recoveries with an interaction
   decomposition (Heimersheim–Nanda; the current key/query/MLP2 numbers are conduits, not shares).
4. **Controlled biological edits** (PPT / branch-point / geometry) — but only the arm A0's
   token-vs-position branch selects.
5. **Freeze the entry components + graph**, then **B′** — edge-restricted sufficiency + necessity
   (path-patching / causal-scrubbing, preregistered `R(C)`/`N(C)` margins, topology-matched random
   paths), the sealed confirmatory endpoint. Replaces the discredited "restore-everything" design.

**The reflection rule.** Every A0 build step is checked against this anchor and the frozen prereg.
`S_A` is a mechanistic intermediate only — no partner-selection *behavioural* claim is revived (the
natural matched-decoy behavioural task was withdrawn as degenerate). A null / "it was already in
the embeddings" result is the *more* informative outcome: it retires the diffuse-builder story.

---

## Setup-time decisions (pre-run, no confirmatory data seen)

- **D-A0-1 — reference = mean-resample over the eligible set at matched positional role
  (donor / a⁺ / aʳ / a⁻).** Norm-preserving (unlike zero-ablation) and carries no example-specific
  a⁺/aʳ distinction, so a surviving `Δk_s` is contributed by the prefix-clean stages. The corrupt
  run is **rejected** as reference: it perturbs the a⁺ token at the embedding stage and would
  conflate token restoration with downstream construction. (Prereg §3; user-confirmed default.)
- **D-A0-2 — reconstruct only {donor, a⁺, aʳ, a⁻} positions.** Every LayerNorm is positionwise, so
  L3's Q/K at those tokens are exact under clamping only those tokens. A preflight memorial checks
  3-position vs all-position clamp agree < 1e-4.
- **D-A0-3 — real-acceptor contrast is primary, generic-AG secondary.** Named "real-acceptor
  contrast," not "identity-clean": it controls splice-site strength (both real) but not residual
  differences in strength/geometry/distance/context, which are reported (signed distance gap), not
  claimed away. The withdrawn partner-selection claim stays withdrawn.
- **D-A0-4 — one new source module (`provenance.py`) + one runner + one test file.** Matches the
  acceptor-address footprint (`address.py` + runner + test); the only new primitive is the
  embedding-output clamp. No sprawl.

## Redrafts (mismatches found between the built experiment and the frozen prereg)

_(logged as they are found during build/smoke — critique → redraft → re-verify)_

- **R-A0-1 — `circuit.py` violated its own transformers pin (found during first smoke import).**
  `circuit.py:40` did `from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS` and
  registered a custom attention at import time (added in `70f53e3` for the period-3 `roll_attention`
  study). That symbol only exists in transformers ≥ 4.48, but `pyproject` pins `>=4.40` and the
  `gfm-thrust1` env has 4.44.2 — so **importing `circuit` failed entirely**, blocking A0 and every
  circuit-dependent assay. Fixed with a compat guard: the import and the `.register(...)` are
  wrapped, and `roll_attention` now raises a clear "needs transformers>=4.48" error only if
  actually invoked. Verified: `circuit` imports, 16 existing memorial tests pass, the model loads
  and forwards on 4.44.2. **Not** an env upgrade (would leave the pin wrong and risk the
  DNABERT-2/SpliceBERT loading gotchas). The period-3 roll tests still require ≥4.48 to run; that
  is now explicit rather than a cryptic import crash.
- **R-A0-2 — `n_real ≥ 60` was infeasible (found during first smoke).** The real-acceptor decoy
  yield is ~11.5% (30 real decoys at n = 260 in `acceptor_address.json`), so the draft floor of 60
  would have blocked the confirmatory run. Pre-run prereg amendment (§6): floor → **n_real ≥ 30**
  with modest power labelled, and the primary arm now draws real pairs from the **full screened
  pool** (`--real-cap`, target ~60–120) rather than the first `n`, maximizing its power without
  inflating the n ≥ 250 secondary arm. This uses a *prior-experiment* yield count, not any A0
  outcome — val stays sealed.
- **R-A0-3 — the residual-space `stage8_arithmetic` memorial was non-informative.** The
  reconstruction lives in Q/K space (`capture_proj`), so "cached L3 input == stage-7 residual" is a
  trivial identity (stage 7 *is* the clean run). Realized instead — and non-circularly — as
  `stage7_equals_clean` (no-clamp Q/K == clean Q/K) **plus** `positionwise_reconstruction`
  (3-position clamp == all-position clamp at tracked positions). Both pass **exactly (0.0)** in
  smoke. Prereg §6 amended.

### Second adversarial pass (re-check after build; "go back over to verify consistency")

- **R-A0-4 — attraction was joint-config only; the diagnostic needs per-curve.** `sweep_arm`
  computed the absolute-window attraction only under the joint config `(q_s, k_s)`, but prereg §2
  requires it **per curve**, and its purpose (catch a query effect that lifts the whole acceptor
  class vs background while `a⁺−a(other)` stays flat) is served by the **query-config** `(q_s,
  k_F)`. Fixed: attraction is now computed and reported under all three configs (`S_K/S_Q/S_J`).
  Read the query-config across stages against the flat `S_Q` difference to detect a class-lift.
- **R-A0-5 — primary-arm distance confound was reported on the wrong set.** `result["match_audit"]`
  is the secondary/generic `pairs`; the real-acceptor arm's signed distance gap (a⁺ vs aʳ, prereg
  §1/§4) was not reported for the actual `real_pairs`. Fixed: `primary_real_acceptor_contrast`
  now carries `real_distance_confound = match_audit(real_pairs)["real_acceptor_distance"]` and its
  own `n_genes`, so identity-vs-distance-decay is adjudicated on the set it belongs to.
- **R-A0-6 — the `S_A(0)` floor is not zero, and the prereg claimed it was.** With role-specific
  mean-resample references, `S_A(0)` measures the **population-generic** (position/distance)
  preference, not zero. This is the correct, more informative design (floor = generic preference;
  `S_A(7)−S_A(0)` = example-specific construction). Prereg §3 amended (pre-run); no code change —
  the curve is read as accumulation from the generic floor, and `S_A(0)` is reported not gated.
- **Strengthened a vacuous check.** `reference_is_clean_sequence` was a hardcoded `PASS`
  (a control that cannot fail). Now a falsifiable tripwire: asserts the partner AG is intact
  (`seq[a⁺−2:a⁺]=="AG"`, ≥95%) in the reference set — it reads `CC` and fails hard if the rejected
  corrupt-run reference ever leaked in.

**Re-verification after these fixes:** compile OK; 8 A0 unit tests pass; smoke — `qk_reconstruction`,
`stage7_equals_clean`, `positionwise_reconstruction` all exact (0.0); `reference_is_clean_sequence`
passes as a real check; per-curve attraction + LOO finite; endpoint identity `S_K(7)=S_Q(7)=S_J(7)`
exact. The only failing preflight checks are the two scale-dependent ones (match-audit distance
bias and n_real floor), which resolve at confirmatory n. **Clean.**

## Confirmatory outcome (2026-07-16) — CONFIRMATORY, preflight 9/9

n=260 generic, n_real=120, 230 genes. Preflight all pass (qk/stage7/positionwise EXACT=0.0).
Result → `a0_targeting_provenance.json` (REFLECTION filled), claim **C21**.

**Finding (decision-table Row 1 + query class-lift).** The targeting *difference* `S_A` is
**pre-resident in the token+positional embeddings, not constructed by layers 0–2**: ~fully present
at the all-ref floor (S_J primary +3.14 vs full +3.04; secondary +3.49/+3.54), flat prefix curve
with a mid-hump, and leave-one-stage-out isolates embeddings (drop +3.19/+3.08; all layer stages
\|drop\|<0.9). The **query pathway supplies class-level acceptor gain** (query-config attraction
raises a⁺ *and* decoy together, peak at mlp2), **not** identity discrimination — this **resolves
C20** (L2H15 66% coupling / 11% preference) and **retires the diffuse-builder hypothesis**. `S_A`
is an intermediate: the primary arm's distance confound (median +19.5nt) and the secondary arm's
site-strength conflation mean A0 does **not** adjudicate identity-vs-geometry.

**A1 routing (updated per the decision table).** Do **NOT** run a layers-0–2 builder scan (there is
no example-specific constructor to find). Route instead to:
1. **Token-vs-position 2×2** (prereg §5.1) — split the embedding-resident difference into
   positional geometry vs local sequence content. This is now the primary next experiment.
2. **mlp2/L2H15 query-gain → coupling** — a focused test that the layer-2 query's class-level
   acceptor attraction (not the a⁺−decoy difference) is what scales the donor coupling; this is the
   clean re-characterization of L2H15's role that C21 opened.

The Q×K×V joint-accounting and B′ (edge-restricted sufficiency/necessity) steps of the anchor are
unchanged, and now gated behind the token-vs-position result.

## A1a token-vs-position outcome (2026-07-16) — CONFIRMATORY, preflight 8/8

Frozen protocol `PREREGISTRATION_A1_TOKEN_VS_POSITION.md`; result `a1_token_position.json`
(REFLECTION filled); claim **C22**. n=260 generic / n_real=120, 225 genes; `shared_ref_neutralises`
=0.0 EXACT.

**Finding: the embedding-resident targeting difference is POSITION-EMBEDDING GEOMETRY.** In both
arms `position_effect ≫ content_effect`, CI-separated (primary +2.72 vs +0.40; secondary +3.59 vs
+0.68); **pos_neutral ≈ 0** (equalising a⁺/decoy positions collapses the whole preference even with
content clean); content is **modest but reproducible** (~15–19% of full). **floor = 0.000** — no
directly recoverable contribution to this targeting score lies outside the 4-token key window *at
this stage* (this does NOT prove PPT/branch-point info never reaches the mechanism; earlier heads
could compress nonlocal context into these local vectors first). Holds in the **distance-matched**
secondary arm ⇒ position-embedding geometry (distance-decay ruled out; **absolute vs relative vs
mod-3 not yet distinguished** — a joint donor+acceptor translation test, deferred).

**Ledger impact (formal).** **R20** retracts C20's *partner-identity interpretation* of the
`S_QK`/`S_A` attention preference (the empirical statistic and its ~15–19% content component
survive). Behaviour-level partner-specificity (C19) is untouched — separate evidence. The circuit
still couples partner-specifically at behaviour; the *mechanism* of L3H1's attention targeting is
positional. Do not silently rewrite C20: its result is preserved, interpretation superseded.

**Next steps (updated).**
1. **A2 — address–payload factorial (the key synthesis experiment, NEXT):** QK provides a largely
   positional **address** (where L3H1 reads); test whether the OV/value stream carries a
   partner-specific **payload** (what it retrieves), reconciling positional targeting with C19's
   partner-specific behaviour. Factorial over {attention address: clean/corrupt} × {acceptor
   payload: corrupt / U_info-projected / full-clean / energy-matched-random}, patch actual
   activations, run downstream, read the **C19 behavioural statistic**. Decisive subtest: force the
   **same attention row** on clean vs corrupt and vary only the values. Then localise whether MLP2
   constructs the payload. Prereg `PREREGISTRATION_A2_ADDRESS_PAYLOAD.md`.
2. **Absolute-vs-relative geometry** (deferred): joint donor+acceptor translation test —
   `S_A(d+Δ, a+Δ) ≈ S_A(d,a)` ⇒ relative; changes ⇒ absolute. AFTER A2.
3. Q×K×V joint accounting and B′ remain, gated behind A2.

## A2 address–payload outcome (2026-07-16) — CONFIRMATORY, preflight 7/7

Frozen protocol `PREREGISTRATION_A2_ADDRESS_PAYLOAD.md`; result `a2_address_payload.json`
(REFLECTION filled); claim **C23**. n=300, 261 genes, coupling 0.96; reconstruction & inject-identity
memorials EXACT=0.0.

**Finding: ADDRESS–PAYLOAD BINDING.** Reconstruct-inject of L3H1's donor output Σⱼ A_dj V_j:
address_only +0.01, payload_only +0.05 (each ~nothing), joint +0.37, **interaction +0.30** (near
pure). Same-row value effect: clean address +0.35 vs corrupt address +0.05. A random rank-8 value
subspace recovers +0.18 (payload distributed). **This reconciles A0/A1a/C19**: position-embedding
geometry routes L3H1 (*where*), the value at that location carries partner content (*what*),
behaviour needs both. R20 stands (attention is not identity; the *complete head* is partner-specific
via binding).

**Next (A2b — the queued refinement, frozen in the A2 prereg §4):**
1. **U_info-subspace payload localisation** — is the distributed payload the C16/C17 `U_info`
   object? `r_j^U = r_j^- + P_U(r_j^+−r_j^-)` in the acceptor window (P_U = top-8 of clean−corrupt
   final residuals), recompute values + downstream; energy-matched random rank-8 + non-site controls.
   Needs the residual-patch/recompute machinery.
2. **MLP2 constructs the payload?** {L2H15 clean, MLP2 recompute} vs {L2H15 clean, MLP2 clamped
   corrupt} vs {MLP2 clean, no L2H15} vs {both}; mediation `I_{L2H15→MLP2}`, both directions.
3. Then absolute-vs-relative geometry; Q×K×V; B′.

## Provenance (git checkpoints for the A0–A2 arc)

- **Reconstruction machinery** (`circuit.py` compat guard, `provenance.py`, memorial tests):
  commit `6a5e80e` — `fix: support intervention reconstruction for A0-A2`.
- **Study** (preregs A0/A1/A2, runners, results+manifests, CLAIMS C20-caveat/C21/C22/C23/R20,
  this log): the following `study:` commit on branch `donor-graph-address-payload`.
- **Model checkpoint** `SpliceBERT.1024nt/pytorch_model.bin` sha256
  `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d` (in every result manifest).
- A2b (residual-patching payload localisation) begins from this clean checkpoint as a separate
  changeset, so its new machinery does not blur the A0–A2 provenance.

## A2b-(1) payload localization outcome (2026-07-16) — CONFIRMATORY, preflight exact

Frozen protocol `PREREGISTRATION_A2b_PAYLOAD_LOCALIZATION.md`; result `a2b_payload_localization.json`
(REFLECTION filled); claim **C24**. n=300, 256 genes; `value_linearity` memorial 1.9e-6 (post-LN
`V=W_V·r` — the analytic reconstruction is exact). New module `payload.py`, runner, tests — a
**separate changeset** from A0–A2 (commits `6a5e80e`/`3a7d3a7`), per the provenance discipline.

**Finding (decision-rule "NEITHER" branch + one positive).**
- POSITIVE: the payload is **localized to the 4-token acceptor window** — restoring its clean values
  alone recovers +0.38 ≈ A2 joint (+0.37).
- NEGATIVE: within the window the payload is **not concentrated in the preregistered rank-8
  `U_info`**. `U_info` recovers +0.06, **not** above an energy-matched random rank-8 distribution
  (p=0.22, percentile 0.80); complement carries ~all (perp +0.37), N_U +0.011. *(Fractions are
  non-additive: raw R_U/R_full ≈ 16%, N_U/R_full ≈ 2.9%, floor-subtracted ≈ 4.5% — denominator-free
  facts carry it.)*
- **Scoped (narrowed):** the payload is distributed **relative to `U_info`**, not distributed in
  general — its **intrinsic dimensionality is unmeasured** (a privileged low-rank subspace could
  still live in the 504-d complement; needs a rank sweep / independent `U_source`).
- **`U_info` is NOT redefined**: it remains the valid DELIVERED donor-side format (C16/C17); A2b
  shows the same basis is not privileged at the acceptor SOURCE. L3H1 does **selection + representation
  conversion** (broad acceptor value → compact donor-side `U_info`, via W_V/W_O).

**Redirect for A2b-(2) (MLP2 construction).** The user's rule gated the MLP2-construction mediation
on "a specific payload whose construction MLP2 can be tested against." A2b-(1) found **no** specific
low-rank payload (it is distributed), so "does MLP2 construct `U_info`" is moot. The sharp question
becomes: what shapes the **distributed acceptor-window value** L3H1's OV reads? Candidates: (a) it is
largely the acceptor's own **local sequence representation** (input-driven, not MLP2-constructed) —
testable by whether L2H15/MLP2 ablation changes the acceptor-window value vs the donor query; (b)
reconcile with `l2_accounting` (L2H15 = 0.66 of coupling) — is L2H15's contribution the **address**
(positional query/key, per A1a) rather than the acceptor payload? That address-vs-payload attribution
of L2H15 is the natural next experiment, replacing the (now moot) `U_info`-construction test.

## A3 L2H15 address-vs-payload outcome (2026-07-16) — CONFIRMATORY, memorials exact

Frozen protocol `PREREGISTRATION_A3_L2H15_ADDRESS_PAYLOAD.md`; result
`a3_l2h15_address_payload.json` (REFLECTION filled); claim **C25**. n=300, 246 genes;
reconstruction_matches & baseline_reproduces_ablated EXACT=0.0 (faithful reconstruct-inject under
the L2H15 ablation). Reused `payload.inject_margin_donor` + `circuit.ablate` — no new kernel.

**Finding: L2H15 = the ADDRESS builder.** Restoring L3H1's intact attention rescues +0.454
[0.39,0.53]; restoring its intact values rescues +0.019 [0.01,0.03]; joint +0.479, interaction
+0.006. address/joint = 0.95. So L2H15's ~0.66-of-coupling role is to shape L3H1's **positional
attention address** (A1a/R20), NOT to supply the acceptor payload (A2b/C24). Donor-only: joint 0.48
vs l2_accounting all-position 0.959 (rest via donor±5 delivery, C17).

## The crystallized circuit (A0–A3)

```
L2H15 builds the positional ADDRESS (C25; 0.66 of coupling)
        ↓
L3H1 attends there — position-embedding geometry, not identity (A1a/R20)
        ↓
L3H1 reads the ACCEPTOR's own local value payload (acceptor-local, distributed, NOT L2H15-built;
   NOT the U_info subspace) (A2/A2b, C23/C24)
        ↓
L3H1 CONVERTS it (W_V/W_O) to the compact donor-side U_info (C24; U_info = delivery format)
        ↓
U_info transported to the donor ±5 neighbourhood (C16–C18)
        ↓
partner-specific donor prediction (C19)
```

**Next (cleanly gated):**
1. **Absolute-vs-relative geometry of the address** — joint donor+acceptor translation test
   (`S_A(d+Δ,a+Δ) ?= S_A(d,a)`), now sharply motivated (L2H15 builds this address).
2. **Independent replication of donor-side `U_info`** (rank curve, held-out gene split) WITHOUT
   redefining it (per the methodology note).
3. Only then, if L3H1's compression matters, derive a separately-named `U_source` at the acceptor.

## A4 address-geometry — INCONCLUSIVE (confounded), 2026-07-16

Frozen protocol `PREREGISTRATION_A4_ADDRESS_GEOMETRY.md`; result `a4_address_geometry.json`
(REFLECTION = INCONCLUSIVE). Readout: L2H15's contribution G15 = S_A^intact − S_A^ablated, S_A =
L3H1 QK preference (a+ vs decoy). Two pre-run redrafts, then a confound found post-run:

- **R-A4-1 (pre-run, smoke):** the raw single-site logit ℓ(d→a) carried a weak, sign-ambiguous
  L2H15 signal (G15(0) −0.18, CI incl 0). Switched S_A to the **preference** ℓ(d→a+)−ℓ(d→a−) where
  L2H15's contribution robustly lives (C20); G15(0) then −0.47, CI excludes 0.
- **CONFOUND (post-run, diagnostic):** G15 = −0.47 at Δ=0 but ~0 at every Δ≠0, common ≈ differential.
  The disambiguating diagnostic (intact S_A under common shift) is decisive: **the intact preference
  itself collapses** (+3.48 at Δ=0 → ~0 at any Δ≠0). Clamping ONLY the donor/a+/a− tokens' position
  embeddings (neighbours native) makes an **inconsistent position field** that destroys the QK — not
  a clean absolute-vs-relative signal. **No claim recorded.**
- **Redesign (next):** a CONSISTENT global `position_ids += Δ` shift on a **shorter window** (~512nt,
  leaving PE-table headroom), so the whole region translates consistently. Still position-embedding
  only (no sequence padding). **Memorial to add:** intact-`S_A`-survival under the shift — every
  future PE-geometry test must include it.
- **Lesson:** PE interventions on isolated tokens confound any QK readout via the inconsistent field.

A4 is **not committed** as a result (inconclusive). The U_info independent replication (rank curve,
held-out split, no redefinition) remains the other queued task.

## A5 U_info replication (task 3) — CONFIRMATORY, VALIDATED, 2026-07-16

Frozen protocol `PREREGISTRATION_A5_UINFO_REPLICATION.md`; result `a5_uinfo_replication.json`
(REFLECTION filled); claim **C26**. n=400, 320 genes, disjoint 199/201 gene split; `margins_fn`
memorial EXACT=0.0. Reuses `subspace` + `task.pair` + `margin_hard` — no new kernel; **U_info NOT
redefined** (exact C16 definition).

**Finding: U_info is a STABLE, generalizing representation.** Subspace overlap(U_A,U_B) ≫ random at
every rank (top-1 0.970 vs 0.002; rank-8 0.630 vs 0.016); held-out sufficiency 0.875 [0.853,0.901] ≈
in-sample 0.911 ≫ random 0.014; held-out necessity 0.916. Reproduces C16 (in-sample 0.911≈0.92,
top-1 0.743≈0.74). Rank curve: dominated by the top-1–2 directions, rises smoothly to 0.94 at K=32
(rank 8 reasonable, not magic). C16–C18 strengthened; consistent with C24 (U_info = stable donor-side
DELIVERED representation, not acceptor source).

**Remaining queued work:** the redesigned address-geometry test (consistent global position_ids
shift on short windows, with the intact-S_A-survival memorial) — A4 was inconclusive/confounded.
