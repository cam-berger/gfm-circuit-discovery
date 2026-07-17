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
L2H15 AMPLIFIES/ROUTES the positional ADDRESS into L3H1 (C25; 0.66 of coupling) — the discriminative
   geometry is PRE-RESIDENT in the embeddings (C21); L2H15 does NOT originate it (C25↔C21 reconciled)
        ↓
L3H1 attends there — position-embedding geometry, not identity (A1a/R20)
        ↓
L3H1 reads the ACCEPTOR's own local value payload (acceptor-local, distributed, NOT L2H15-built;
   NOT the U_info subspace) (A2/A2b, C23/C24)
   ⇒ this ACCEPTOR-LOCAL, serially-interdependent (L0→L1→L2) source is why the sparse circuit is
   NON-FAITHFUL (A6/C27: F=0.005 vs D=0.986; A7/C28: ΔACC=+0.337=92% of ceiling; single layers near-silent → pipeline)
   ⇒ A8/C29: that generator is a serial L0→L1→MLP2 MLP-BACKBONE pipeline (L0 attn→MLP0→MLP1→MLP2;
      MLP2 dual address+payload; L2 attention NOT involved; L2H15 address-only) — held-out confirmed
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

## Methodology review → T0 reconciliation + T1 (A6), 2026-07-16

A methodological review (IOI/Wang, path-patching/Goldowsky-Dill, ACDC/Conmy, AtP*/Kramár, EAP-IG/Hanna,
Hydra/McGrath, causal-scrubbing/Chan, circuit-tracing/Anthropic) landed the central charge: **A0–A5
validated *paths*, never tested the whole graph for faithfulness / completeness / minimality.** Verdict:
substantially correct (R20 is the very "named a variable before identifying it" failure mode it warns of).

- **T0 — C25↔C21 reconciliation (applied):** the review's "L2H15 builds the address is too strong" is
  right, and it reconciles with our own **C21** (discriminative geometry pre-resident in embeddings;
  L2H15 supplies class-level *gain*). C25 headline "builds" → "acts through / amplifies the address
  channel"; the 95%/4% causal split is UNCHANGED (no R-number — interpretive verb only). Crystallized
  diagram updated.
- **Refinements I pushed back with:** faithfulness-of-current-circuit *before* the exhaustive scan (it
  sizes the residual cheaply); causal-scrubbing is the most *lenient* tool (pair with interchange +
  held-out); "the only route" collides with Hydra (target = minimal-sufficient + redundancy quantified).

### A6 circuit faithfulness/necessity/minimality — CONFIRMATORY, 2026-07-16

Frozen `PREREGISTRATION_A6_CIRCUIT_FAITHFULNESS.md`; result `a6_circuit_faithfulness.json` +
`a6_reconcile.json` (REFLECTION filled); claim **C27**. n=200, 179 genes, coupling 0.94. **No new
kernel** (reuses `patch_heads` + `clamp_write` + `clamp_embed`). Node-level activation patching,
scrub-**to-corrupt**; two arithmetic memorials `empty→corrupt`/`full→clean` **EXACT=0.0** (node cover
complete; damage confound excluded; the healthy extremes are the health guard).

**Finding: necessary DELIVERY SPINE, non-faithful sparse generator — generation is DISTRIBUTED or
INTERACTION-DEPENDENT (node resolution).** `N(C)=1.00`, `N(C\embed)=0.989` (necessity real, not a trivial embed-corruption artifact), but
`F(C)=0.005 [0.003,0.008]` under complement-noising (residual **99.5%**); no single omitted node
recovers it (max add-one gain 0.005 / 93 nodes) and random same-size circuits are equally non-faithful
⇒ **no sparse node set of this size is faithful**. **Reconciles with C16/C17** via the post-hoc
diagnostic: injecting the circuit's **clean OUTPUTS** recovers `D(C\embed)=0.986 [0.982,0.991]` —
sufficient-when-injected, but the circuit cannot *reconstruct* those outputs from a scrubbed network.
So delivery/readout is sparse (`U_info`); the coupling *generation* (building the acceptor-local,
distributed payload, C24) is distributed. **Quantifies C7.** Minimality uninterpretable at F≈0.

**Next (Cam's decided order, 2026-07-16):**
1. **A7 — position×layer source localization (redesigned).** Separate the *state clamp* (inject the
   clean **pre-L3H1** acceptor-window residual = upper-bound control, ≈ C24, should recover) from
   *writers-open* (let native **non-circuit** L0–L2 components write clean ONLY within a region,
   scrubbed elsewhere — can the network **locally construct** the payload?). Disjoint regions
   {acceptor 4-tok, donor±5, PPT/branch-point, remainder, combos} × layer-blocks. Report
   `ΔR = F(C∪R) − F(C)` and pairwise interaction `I(R₁,R₂) = F(C∪R₁∪R₂) − F(C∪R₁) − F(C∪R₂) + F(C)`
   to split **localization** from **interaction**. Decision table: state-high+acceptor-writers-high →
   local convergence; state-high+writers-low → local-when-finished but imports info; several-weak-
   jointly-strong → structured distributed; none → broad / wrong partition (→ hierarchical group tests).
2. **Scoped upstream edge discovery** — EAP-IG/AtP* rank → exact **group AND edge** patching.
   **Block-level, not singleton-only** (A6 found negligible singleton recovery ⇒ synergistic generators
   need group tests).
3. **Backward-pruned faithfulness–size curve** (DEFERRED, redesigned — forward greedy is ill-identified
   at F≈0: it selects noise/order-dependent). Prune from the **full model**, recompute conditional
   effects each removal, test groups + nodes, select on one gene split / measure the curve on another,
   report node counts for 50/80/90/95% faithfulness.
4. **Semantic tests** on the resulting source representation.

C27 wording scoped to "distributed **or interaction-dependent** at node resolution" per Cam (A6 rules
out the sparse generator + single-node omissions, NOT a structured jointly-necessary module). **Commit
`c98055a`→`4907a19` (amended with the scoping fix) held LOCAL (not pushed).**

### A7 source localization — CONFIRMATORY, 2026-07-16

Frozen `PREREGISTRATION_A7_SOURCE_LOCALIZATION.md`; result `a7_source_localization.json` (REFLECTION
filled); claim **C28**. n=200, 179 genes. Built to the review's redesign — **no claim redefined**.
Reuses A6's scrub + token masks; adds `clamp_write_mask` (bool-mask MLP write) + `clamp_layer_input`
(residual pre-hook). Four exact plumbing memorials `=0.0` (base scrub, node cover, token-mask,
layer-input pre-hook).

**Finding: local_convergence_at_acceptor.** State-clamp ceiling (clean pre-L3H1 residual at the
acceptor window + clean spine) `F_state=0.372` (ceiling 0.365 ≈ C23/C24 cap). **Writers-open ΔR (L0–2):
ACC +0.337 [0.30,0.38] = 92% of ceiling; DON ≈0; PPT ≈0; REM +0.025 (7%).** Single-layer ΔACC (L0
+0.047, L1 +0.009, L2 +0.001) sum 0.057 ≪ 0.337 joint ⇒ **serial interdependence (L0→L1→L2** — later
layers can't run on a corrupted upstream state; a pipeline, not 3 independent factors); interactions
modest (ACC×REM I=+0.052, ACC×DON +0.027, PPT×* ≈0). So the payload generation is **positionally
LOCALIZED at the acceptor window** and built by a **serially-interdependent L0→L1→L2 module** — NOT
broadly diffuse, NOT imported from the branch-point/PPT **for this channel/stage** (ΔPPT≈0; a scoped
negative, not a biological claim about splicing; coheres with C22/C19).
**Resolves the A6/C27 open question** (structured jointly-necessary module, the review's alternative to
irreducibility). Scope: bounds the *generation* feeding L3H1's channel (0.37 ceiling); delivery-side
distribution is separate.

**Decision-table row fired:** `state-high + ΔACC-high` → **next = edge scan into the acceptor-window
residual across L0–L2** (block-level / synergistic-group aware, per the decided order: A7 done → edge
discovery → backward-pruned size curve → semantics). A6/A7 pushed to origin (`b3d3f16`) with the
serial-interdependence / scoped-PPT wording corrections.

### A8 generator path — CONFIRMATORY (held-out), 2026-07-16

Frozen `PREREGISTRATION_A8_GENERATOR_PATH.md` (audited pre-run: 5 fixes — `M'(∅)`≠corrupt, `N_h` from
full A, value/address split, exact C24 anchor, MLP2 circuit-value check); result `a8_generator_path.json`
(REFLECTION filled); claim **C29**. n=220, 205 genes, D/H 112/108. **No new kernel** (A6/A7 scrub +
A2b/A3 reconstruct-inject). Value-path metric (L3H1 **address held clean**, only the value varies):
`value_linearity=2.7e-6` exact, `full_clean`=+0.354 reproduces C24, `generator_ceiling`=0.289.

**Finding: the generator is a serial L0→L1→MLP2 MLP-BACKBONE pipeline (held-out confirmed).**
Leave-block-out necessity (replicating on H): **L0** N=+0.895 [H .87,.92], 40% sufficient alone
(R=0.403) = origin; **L1** N=+0.528 [H .43,.57], near-silent alone (0.057), weak with only L2 downstream
(P=0.095) = refiner requiring L0 upstream; **L2 attention N=−0.031 (not required for payload generation
under the block intervention; L2H15 stays essential on the separate ADDRESS path, C25).** *(All N/P/R
are conditional causal effects, NOT additive fractions of the payload — they do not sum.)* Backbone =
the MLPs (indiv N: **MLP2 +0.893, MLP1 +0.393, MLP0 +0.324**; top attn feeder **L0H7 +0.352**). **MLP2
dual-role** (C21 address assembler + dominant payload relay); **L2H15 address-only** (payload N=+0.001) —
address/payload separable within L2. **Block-vs-singleton vindicated:** 4 L2 heads flag load-bearing
though the block is unnecessary (within-block interaction artifact) → group tests trustworthy, as the
review insisted. Held-out `R_H(L0+L1+MLP2)`≈1.0 ≥ 0.80.

**Completes the review's program:** non-faithful sparse circuit (A6) → acceptor-local serial generation
(A7) → specific generalizing serial MLP-backbone generator (A8). **A8 committed LOCAL (not pushed).**
**Next (remaining decided order):** (a) exact **head-level** minimal graph + head→MLP **edges**
(interaction-aware, since L2 singletons mislead); (b) the **semantic identity** of the payload the
pipeline builds (matched-resampling / interchange — the address geometry A4-redux is the sibling
address-side question); (c) the deferred backward-pruned faithfulness–size curve.

### Consolidated mechanism (post-A8, 2026-07-16; Cam's form)

```text
ADDRESS
position-embedding geometry
    → L2H15 and MLP2                       (C21/C22/C25; MLP2 = query/address assembler)
    → L3H1 attention selects the acceptor  (A1a/R20: position geometry, not identity)

PAYLOAD
acceptor-local state
    → L0 attn: L0H7 + redundant H12/H10    (A8/C29 origin; A10/C31: L0H7 dom 0.35, H12/H10 redundant I=-0.13)
    → MLP0 (MEDIATES) → MLP1 → MLP2        (A10/C31: strict serial chain, each necessary held-out, none alone;
                                            L0 heads reach payload THROUGH MLP0, mediated)
    → L3H1 value pathway                    (A2/A2b/C23/C24: acceptor-local, distributed-rel-to-U_info)
    ⇒ ENCODES LOCAL SPLICE-SITE STRENGTH    (A9/C30: V2b local-MaxEnt r=-0.51 + V3 model-logit r=+0.62
                                             name it, held-out/calibrated; V2a full-23nt null → local;
                                             not identity; composition confound open)

BINDING AND DELIVERY
address × payload at L3H1                   (C23: near-pure interaction, neither alone)
    → compact donor-side U_info             (C24/C26: L3H1 converts to the stable delivered format)
    → sparse delivery spine                 (C16–C18: MLP3/4 + L4/L5 transporters; A6: necessary, sparse)
    → partner-specific donor prediction     (C19)
```

Payload variable **NAMED** (A9/C30: local acceptor-window splice-site strength). Open: head-level
**edges** (A10, next), **address geometry** absolute/relative (A11 = A4-redux), faithfulness–size
**curve** (A12). MLP2 sits on **both** the address and payload paths (dual-role).

### A9 payload semantics — CONFIRMATORY (held-out, calibrated), 2026-07-16

Frozen `PREREGISTRATION_A9_PAYLOAD_SEMANTICS.md` (audited pre-run + your conditional-go red-lines);
result `a9_payload_semantics.json` (REFLECTION filled); claim **C30**. n=240, 207 genes. MaxEntScan
pinned (`maxentpy 0.0.4`, tables sha256 `8b6d7b46…`). **No new kernel** (A8 reconstruct-inject
cross-example + frozen `U_info` projected-onto). Matched **interchange** (deliver j's clean payload
through i's held-clean address); `value_linearity`=2.7e-6 exact.

**The calibration gate you insisted on caught THREE decision-layer bugs across 4 runs before any
verdict was allowed out:** (1) categorical "mismatch-drops" test wrongly applied to ordinal variables
(planted failed → fixed with sign-correct monotonic ordinal correlation); (2) ad-hoc bootstrap-CI
significance too loose, FPR=0.18 → fixed with a permutation-calibrated threshold (out-of-sample
calibrate/test split); (3) planted direction too weak (equivalence NaN) → fixed with a recovering
Vplant-scaled direction. Run 4: **planted r=0.9998** (sensitivity), **permuted FPR=0.047** (specificity).

**Finding: the payload encodes LOCAL acceptor-window SPLICE-SITE STRENGTH.** Held-out: **V2b**
(local-4-tok frozen MaxEnt, model-independent) names it (r=−0.51, monotonic, eq 0.986); **V3** (model
acceptor logit) names it (r=+0.62); **V2a** (full 23nt incl. flank/PPT) does NOT (r=−0.04) →
**cross-validates C28's ΔPPT≈0** (local window, not the whole site). NOT identity (V1). **Composition
(V4) NON-IDENTIFIABLE** (0 matched pairs) → confound OPEN. **Verdict corrected** from the script's v1
coarse label (which only checked V2a) to `externally_grounded_local_splice_strength`.

**Completes the arc:** A6 not-a-sparse-circuit → A7 acceptor-local serial → A8 L0→L1→MLP2 MLP-backbone
generator → **A9 that generator computes local splice-site strength**. **Next: A10** head-level edges
(interaction-aware, reusing the A8 value-path metric), then A11 geometry, A12 size-curve.

### Verification correction pass — A6–A8, 2026-07-16 (adversarial verify-workflow, 16 agents)

An adversarial verification (3 verify + 12 refute + synthesis) confirmed the **empirical findings are
sound and robust** (F≈0 conservative; gene-grouped bootstraps; honest de-duplicated nulls; A8 held-out
genuine) but caught **rigor-framing overstatements** — now corrected in the ledger (→ **R21**):

- **Embed-clamp-dominated memorials (C27/C28/C29):** `clamp_embed` overwrites the embedding at all
  positions, so `empty→corrupt`/`full→clean` are bit-identical to the source run regardless of the
  head/MLP patches — they validate embed-plumbing, NOT node-cover completeness. Completeness rests on
  architecture ({embed, all heads, all MLPs}=all writers) + the genuine `value_linearity` memorial
  (A8/A10, on-path). C27/C28 corrected; C29's `value_linearity` is the real, on-path memorial.
- **N(C)=1.00 (C27)** is a forced identity (embed∈C), not a measured necessity → demoted; `N(C\embed)=
  0.989` (post-hoc reconcile) is the real figure. Same forced-identity pattern as A8's `R_full=1.0`
  normalization anchor (harmless). Both are new instances of **R7**.
- **"damage confound excluded" (C27)** softened to "avoided by scrub-to-corrupt; partial-scrub health
  noted-not-gated."
- **Cross-draw caveat:** A6/A7/A8 run on different example draws (tags a6/a7/a8-val, coupling
  0.94/1.02/1.01) — cross-experiment F/ceiling are consistent-but-not-exact replications.
- Confidence: **C29 high** (most rigorous), **C27/C28 medium** (post-correction). The A9-verification is
  **tabled** (Cam) for after; A10 reuses the corrected framing.

### A10 generator edges — CONFIRMATORY (held-out), 2026-07-16

Frozen `PREREGISTRATION_A10_GENERATOR_EDGES.md`; result `a10_generator_edges.json` (REFLECTION filled);
claim **C31**. n=220, 196 genes, D/H 110/110. **No new kernel** (A8 value-path reconstruct-inject; real
memorial `value_linearity`=2.9e-6; scrub-cover R21-vestigial).

**Finding: the head-level wiring is `{L0H7 + redundant L0H12/L0H10} → MLP0 (mediating) → MLP1 → MLP2`,
a strict serial chain (held-out).** L0* = {H7,H8,H10,H11,H12,H13}; L0H7 dominant (N_h=+0.348 [0.30,0.40],
replicates D 0.374), H12 +0.160, H10 +0.116; **H12×H10 interaction −0.126 (partially redundant)**, H7
independent. **Head→MLP mediation TRUE** (L0-heads-only 0.136 → +MLP0 0.356): L0 attention reaches the
payload THROUGH MLP0. **Strict serial MLP chain**: N_MLP0=0.354, N_MLP1=0.434, N_MLP2=0.88 (all held-out
CI-low>0), none sufficient alone (X_only≈0.01). Refines C29 to head→MLP wiring; conditional effects NOT
additive. Scope: value-path (address clean → L2H15 address role removed by construction); L1/L2 head
resolution deferred. **Next: A11** address geometry (absolute/relative), **A12** faithfulness-size curve.

### A11 address geometry — CONFIRMATORY, 2026-07-16 (A4 redesign, confound resolved)

Frozen `PREREGISTRATION_A11_ADDRESS_GEOMETRY.md`; result `a11_address_geometry.json` (REFLECTION
filled); claim **C32**. n=1037, 641 genes. **A4 redesign**: consistent GLOBAL `position_ids` shift on a
truncated W=640 window (PE-table headroom), with the mandatory **`intact_S_A_survives`** memorial —
which **PASSED** (`S_A(0)=2.94`, every `S_A(Δ)≥0.5·S_A0`), so the shift preserves the preference,
**resolving the A4 confound** (A4's isolated-token clamp collapsed `S_A`; the memorial now catches this
pre-run). truncation↔full corr 0.958. No new kernel (A1a/A4 `S_A` instrument).

**Finding: PREDOMINANTLY RELATIVE + modest absolute component (resolves C22).** `S_A` survives the
±192nt common translation (+2.9→+3.6) ⇒ relative offset dominates; but `E_common` significant at 5/6
shifts, U-shaped (slope +0.0012/nt CI [.00102,.00138] excludes 0, `max|E|/|S_A0|=0.237`) ⇒ a modest
~24% absolute modulation (the |Δ|=192 endpoints possibly PE-edge-confounded; interior +64/+128 the
cleaner evidence). Toeplitz residual 0.56 (descriptive). **Verdict corrected** from the script binary
"ABSOLUTE" (overstates — `S_A` survives). The address SIBLING of the payload arc: address = *where*
(mostly relative offset); payload = *what* (local splice strength, A9/C30).

**Next: A12** faithfulness-size curve (built, queued) — completes the A6→A12 program.

### A12 faithfulness–size curve — CONFIRMATORY (held-out), 2026-07-16 — PROGRAM COMPLETE

Frozen `PREREGISTRATION_A12_FAITHFULNESS_SIZE.md`; result `a12_faithfulness_size.json` (REFLECTION
filled); claim **C33**. n=200, 184 genes, D/H 102/98. **No new kernel** (A6 faithfulness scrub; R21
embed-anchors). Rank nodes by F-loss removed from the FULL model on D (well-identified backward);
keep-top-k curve on held-out H.

**Finding: IRREDUCIBLY DISTRIBUTED — gradual curve, NO knee.** Held-out node counts (of 103): 50%→34,
80%→56, 90%→72, 95%→80. F≈1 to |K|=80 then smooth decline. 90% faithful core = 65 heads (all layers
11–14 each) + all 6 MLPs + embed. **Strong independent cross-validation of A6–A11:** the top-ranked
nodes (backward removal-loss, no knowledge of the hand-built circuit) ARE the identified mechanism —
embed #1, MLP2/3/1/0/4/5 #2–8, L3H1 #4, generator L0H7/H12/H10 #9/11/13, transporters L4H2/L5H11/L4H0
#10/12/15. Reading: a structured CORE (the circuit) atop a broadly distributed ~72-node faithful core.

---

## PROGRAM COMPLETE (A6→A12) — the donor mechanism, end to end

Answering the 2026-07-16 methodological review ("validated paths ≠ complete/faithful circuit"):

- **A6/C27** circuit NON-FAITHFUL (F=0.005), delivery sparse / generation distributed.
- **A7/C28** generation ACCEPTOR-LOCAL + SERIAL.
- **A8/C29** serial **L0→L1→MLP2 MLP-backbone generator** (held-out; MLP2 dual-role; L2H15 address-only).
- **A9/C30** payload = **LOCAL SPLICE-SITE STRENGTH** (interchange, calibrated, held-out; composition open).
- **A10/C31** wiring **L0H7/H12/H10 → MLP0(mediating) → MLP1 → MLP2** (held-out).
- **A11/C32** address = **PREDOMINANTLY RELATIVE + modest absolute** (A4 redesign, survival-gated).
- **A12/C33** **IRREDUCIBLY DISTRIBUTED** faithfulness (no knee; 90%→72/103) with the identified circuit
  as the top-ranked structured core (independent cross-validation of A6–A11).
- **Verify + R21**: adversarial verification of A6–A8 → corrected the embed-clamp-dominated memorial +
  forced-N(C)=1.0 framing (empirical findings unaffected).

**Open / next:** A9 composition confound (V4 non-identifiable, re-run with looser calipers); the A9
adversarial verification (tabled by Cam); L1/L2 individual-head resolution; the PE-edge caveat at A11's
|Δ|=192; true lazy-greedy A12 refinement.

---

## A9b — payload semantics *net of bulk composition*, and R22 (2026-07-17)

Ran through the `formalize-audit-verify` cycle. Prereg frozen post-estimand-red-team (re-scoped to
**bulk/frequency** composition; core-motif entanglement declared intrinsic). Estimator: cross-fitted DML
(Robinson partialling), gene-clustered folds, two-way gene-dyadic Poisson-multiplier bootstrap, two
planted-CONFOUND gates (composition-only→β_str null; collinear-strength→β_str recovered), CI-excludes-0.

**Cycle caught three things before/at the claim:**
1. **SMOKE (dim):** `Kz=17` (di+tri needed 17 PCs for 90% var) → dim(Z)=20 broke the confound gates
   (nuisance overfit). Fixed with a hard `--kz-max 8` (dim≤11, n≥10·dim), per the prereg's low-dim rule.
2. **Offline estimator dev (cached arrays):** GBR-DML **overfits** the low-dim Z (out-of-sample
   R²=−0.11 vs ridge +0.36) and manufactured a *phantom nonlinear confound* in the gates. Switched to
   **linear (ridge) DML**; rebuilt the confound plant **fold-free** (full-fit, not OOF) to remove
   fold-coupling. Confirmatory gates then PASS: A β=−0.02 covers 0, B β=0.98 covers 1, planted β=0.77
   detected, permuted FPR=0.05.
3. **Result (broad, a9b-val, n=240):** bulk composition does NOT predict the transfer
   (`CV-R²(S~Z)=−0.07 H / +0.006 D`); collinearity low; support 53–69>40 → **identifiable, composition
   REFUTED**. But V2b β=+0.042 [−0.02,+0.12] H (incl 0), +0.009 D — the external-strength effect **does
   not survive** in this design; only V3 (circular) is marginal (H only).

**Path A (reconciliation, per Cam):** A9's own competitor-matched design + composition control, swept
across independently-seeded draws (`a9c_sweep_seeded_draws.json`):
- `tag="a9-val"` **reproduces A9 exactly** (r=−0.512 H / −0.323 D) → Path-A logic faithful; the whole
  A9↔A9b gap is the tag-seeded example draw (`load_examples` shuffles by tag).
- Across 5 seeded draws: **H r_raw ranged −0.512→+0.017** (mean −0.22; 4/5 negative), **D averaged ≈0**;
  **every composition-adjusted dyadic CI included 0, including a9-val's**. Composition-matched subsets
  were 4/4 negative (a weak directional tendency).

**R22 (material correction).** A9's held-out r=−0.51 is *reproducible on a9-val* (not an implementation
error) but **unstable across independently seeded draws** and **non-significant under two-way gene-dyadic
inference**. C30 **bounded**: payload is acceptor-local + NOT bulk composition (survives); **semantic
identity UNRESOLVED** — the "splice-strength" naming is withdrawn; only a weak negative directional
tendency remains. V3 does not rescue (circular + split-inconsistent). The frozen mechanism claim drops
"splice-strength payload" → "an acceptor-local payload".

**Two methodology lessons (fold into the skill):** (1) a within-sample gene held-out (D/H) tests
generalization *conditional on that sample* but is **not** a test of sampling stability — the whole draw
is one tag-seeded sample; effect-existence needs **multiple independently-seeded (ideally disjoint)
draws**. (2) interchange pairs reuse **sender/receiver genes** → need **two-way (dyadic) inference**, not
permutation-on-correlation.

**Next (per Cam):** absolute MaxEnt strength is probably NOT the semantic target → test **conditional
splice competitiveness** (donor context, competing acceptors, intronic context — the IC1–IC5 program),
not further rescues of the absolute-strength claim. Weak-tendency quantification, if wanted, = a NEW
preregistered **pooled disjoint-gene, two-way-dyadic** meta-analysis (not repartitioning one finite
sample).
