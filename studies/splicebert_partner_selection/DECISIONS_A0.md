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
