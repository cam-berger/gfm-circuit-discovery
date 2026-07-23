# A0 decisions & running log — stage-wise provenance of L3H1's acceptor targeting

Frozen protocol: `preregistrations/PREREGISTRATION_A0_TARGETING_PROVENANCE.md`. This file is the post-unblinding
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

Frozen protocol `preregistrations/PREREGISTRATION_A1_TOKEN_VS_POSITION.md`; result `a1_token_position.json`
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
   constructs the payload. Prereg `preregistrations/PREREGISTRATION_A2_ADDRESS_PAYLOAD.md`.
2. **Absolute-vs-relative geometry** (deferred): joint donor+acceptor translation test —
   `S_A(d+Δ, a+Δ) ≈ S_A(d,a)` ⇒ relative; changes ⇒ absolute. AFTER A2.
3. Q×K×V joint accounting and B′ remain, gated behind A2.

## A2 address–payload outcome (2026-07-16) — CONFIRMATORY, preflight 7/7

Frozen protocol `preregistrations/PREREGISTRATION_A2_ADDRESS_PAYLOAD.md`; result `a2_address_payload.json`
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

Frozen protocol `preregistrations/PREREGISTRATION_A2b_PAYLOAD_LOCALIZATION.md`; result `a2b_payload_localization.json`
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

Frozen protocol `preregistrations/PREREGISTRATION_A3_L2H15_ADDRESS_PAYLOAD.md`; result
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

Frozen protocol `preregistrations/PREREGISTRATION_A4_ADDRESS_GEOMETRY.md`; result `a4_address_geometry.json`
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

Frozen protocol `preregistrations/PREREGISTRATION_A5_UINFO_REPLICATION.md`; result `a5_uinfo_replication.json`
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

Frozen `preregistrations/PREREGISTRATION_A6_CIRCUIT_FAITHFULNESS.md`; result `a6_circuit_faithfulness.json` +
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

Frozen `preregistrations/PREREGISTRATION_A7_SOURCE_LOCALIZATION.md`; result `a7_source_localization.json` (REFLECTION
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

Frozen `preregistrations/PREREGISTRATION_A8_GENERATOR_PATH.md` (audited pre-run: 5 fixes — `M'(∅)`≠corrupt, `N_h` from
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

Frozen `preregistrations/PREREGISTRATION_A9_PAYLOAD_SEMANTICS.md` (audited pre-run + your conditional-go red-lines);
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

Frozen `preregistrations/PREREGISTRATION_A10_GENERATOR_EDGES.md`; result `a10_generator_edges.json` (REFLECTION filled);
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

Frozen `preregistrations/PREREGISTRATION_A11_ADDRESS_GEOMETRY.md`; result `a11_address_geometry.json` (REFLECTION
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

Frozen `preregistrations/PREREGISTRATION_A12_FAITHFULNESS_SIZE.md`; result `a12_faithfulness_size.json` (REFLECTION
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

---

## P3 — the period-3 carrier's causal ROLE (recognition / routing / binding) & LOCUS (2026-07-17)

Per Cam's memo (one bounded P3 carrier program): formalize **P3.1 (recognition-vs-routing chain) + P3.2
(where must the carrier be) + P3.3 (L3H1 attention×value factorial)** as one prereg; **P3.4 (carrier
origin) separately gated** so a coding-origin result cannot redefine the causal splice result. Name
defended: *a learned period-3 positional carrier — a rank-2 cyclic coordinate read as relative-offset
classes* (not "carrier wave", not codon-frame).

**Formalize → 4-lens design red-team (identifiability / statistical / estimand↔intent / calibration-
plumbing) BEFORE any run.** All four `fixable`; **two estimand-level FATALs retired the flagship
endpoint (→ R23):**
1. **`Δ_bind` (the "decisive" binding test) is architecture-forced** — L3H1's output is bilinear `O=A·V`,
   so the 2×2 DiD equals the identity `(A_I−A_D)(V_I−V_D)`, non-zero for ANY perturbation moving both A and
   V, with no carrier-specificity. The same positive `Δ_bind` arises from a junk band or random PE.
2. **`payload_source` is non-identifiable** — no delivery-independent, carrier-uncontaminated observable
   (carrier feeds the generator; MLP2 dual-role, C29). ⇒ "independently-constructed payload" not licensable.
Plus majors: value-main-effect = R15 position-in-V (A/V axis ≠ address/payload); `S_A` restoration under
`A_I` is forced (`S_A=f(A)`); `U_info(D)` degenerate (SVD of ~0); E must be **importance-matched** not
spectral-power-matched; P3.2 irrelevant window must be **mod-3-phase-matched**; R_Y unstable unless the
denominator is robust (⇒ **R_C only**); FWER family under-counted (two families: max-T positive + TOST
equivalence); `Δ_bind` must be **per-example paired**; SESOI per-Y; D_rec recognition needs a **pseudo-site
decoy** (decoy-sparse: `real_acceptor_decoy` avail 1/16) ⇒ **DEFERRED**. Synthesis:
`redteam/REDTEAM_SYNTHESIS_P3_4lens.md`. Prereg **v2** (reframed, binding retired) frozen.

**What survived → confirmatory, identifiable, run on MPS:**

**P3.2 — carrier LOCUS (Cam's "where must it be?"). RESULT: SPATIALLY DISTRIBUTED, not site-local (→ C34).**
Position-selective restoration (batch-aware `position_embeddings` hook adding the carrier delta at
per-example windows; plumbing smoked EXACT — construction 1.5e-8, none==deleted 0.0, all→R_C 1.00 forced,
monotone dose). Discovery/confirm gene split (n=160, 144 genes, 77/83), health-gated, gene-clustered ratio
CIs. Deletion collapses coupling 96.7%/94.5%. **Local windows recover ~2–4%** (donor ±6 R_C +0.016/+0.019,
acceptor +0.016/+0.007, both +0.035/+0.041 — all ≪ SESOI 0.2); **specificity nulls ≈0** (phase-matched
irrelevant +0.002/+0.003, decoy-acceptor +0.002/−0.002 ⇒ site-specific, not a hybrid-P artifact); **broad-
span restoration required** (both-window dose → R_C ≈0.41/0.50 only at ±32 nt). ⇒ distributed positional
coordinate across the span; **Endpoint-B locus, well-powered positive** (not underpowered null). Coheres
with C32 (relative offset) + C33 (irreducibly distributed). **M1 status (R24): M1(≤5% residual) is a
preregistered Global-PASS gate — PASS on discovery (3.3%) but FAILED on confirm (5.5%>5%), so the confirm
split is gate-qualified / verification-level, NOT a clean preregistered confirmation** (denominator robust,
locus pattern identical on both splits; robustness ≠ retroactive gate pass). `results/p3_carrier_locus.json`,
`run_p3_carrier_locus.py`.

**P3.3 — A-vs-V routing decomposition at L3H1 (reframed, NOT binding). RESULT (confirmatory n=120, disc/
confirm 62/58): routing main ≫ value main.** Plumbing anchors EXACT 0.0 (M4 diagonal, off-diagonal native
value-overwrite, head-identity — pins head-1=L3H1). Cells (A,V): DD +0.03/+0.03, ID +0.30/+0.31, DI
+0.03/+0.02, II +0.42/+0.46 ⇒ **routing main (restore L3H1's intact-carrier attention) = +0.257
[0.172,0.351] / +0.284 [0.192,0.386]; value main (restore its values) = −0.011/−0.005 ≈ 0** (a genuine
non-forced sub-SESOI null — within this intervention L3H1's values are not carrier-dependent; **R24**: this
does NOT globally exclude carrier effects on payload *construction* upstream — the carrier feeds the
L0→MLP2 generator, C29/C30). `Δ_bind` = +0.136/+0.157 **CI-positive but
ARCHITECTURE-FORCED (R23) — reported, NOT interpreted** (the pre-run red-team's foresight, confirmed: the
original design would have mis-read this as binding). The carrier's coupling contribution routes through L3H1's
**attention pattern (where it looks / the address), not carrier-dependent changes in its values (what that
head collects at the acceptor)** — the routing-carrier picture, causally (**R24**: within-intervention, not a
global payload-construction claim). Main effects are identifiable (only the interaction `Δ_bind`=+0.08
is forced, R23 — reported not interpreted). Coheres with C22/C25/C32 (attention = positional address) and
C24/C28/C30 (payload acceptor-local, generated by the L0→MLP2 backbone). `run_p3_av_decomposition.py`.

**Net mechanism update:** the period-3 carrier is a **distributed relative-positional ROUTING coordinate**
that shapes **L3H1's attention address** (P3.3: routing main ≫ value) and must be present **across the
donor–acceptor span**, not at the sites (P3.2). Within this intervention L3H1's values are **not**
carrier-dependent (value channel ≈0; **R24**: this does not exclude upstream carrier effects on payload
construction) and its "binding" role is **not identifiable** by carrier-deletion+patching (R23; C23's separate
reconstruct-inject binding stands). **Recognition (D_rec) deferred** (needs a pseudo-site-decoy study);
**P3.4 origin** separate. This closes the SpliceBERT carrier account at the causal-locus + A-vs-V level.

### C34/C35 adjudication (2026-07-18, post-verification, per Cam) → R24 + R23 elevated

Cam's review asked to **close C34/C35 carefully before P3.4/transfer**. Three corrections, all
honesty-improving, none conclusion-flipping (empirical findings — distributed locus C34, routing≫value C35 —
unchanged):

1. **C35 scoped to the intervention (R24a).** "The carrier does NOT build the payload" is withdrawn as a
   categorical cross-intervention claim. The value main effect is a null on **L3H1's own value vectors** under
   carrier patching — it cannot exclude carrier effects on payload *construction* upstream (the carrier feeds
   the L0→MLP2 generator, C29/C30). Supported wording: *the carrier's contribution to coupling is mediated
   predominantly through L3H1's attention pattern, not through carrier-dependent changes in L3H1's values; this
   does not exclude upstream carrier effects on payload construction.* (Same empirical-vs-interpretation
   discipline as R20.)
2. **C34 confirm split is gate-qualified, not a clean confirmation (R24b).** M1 (`C_del/C_intact ≤ 0.05`) was
   frozen into **Global PASS** (prereg §8): PASS on discovery (3.3%), FAIL on confirm (5.5%>5%). Not every
   preregistered precondition held on the confirm split ⇒ the replication is verification-level, not a clean
   preregistered confirmation. Denominator robust + pattern identical on both splits, but **robustness does not
   retroactively pass a failed preregistered gate**. A publication-grade confirmatory locus needs a disjoint
   untouched split on which M1 also passes (tighten the deletion, or the `min_effect` conditioning that inflates
   residual coupling).
3. **R23 elevated as a reusable methodological result (not buried).** The positive L3H1 A×V interaction
   `Δ_bind` (+0.14/+0.16, CI-positive on both splits) was **algebraically forced** by the bilinear `O=A·V`
   (2×2 DiD = identity `(A_I−A_D)(V_I−V_D)`) ⇒ it **cannot support a binding claim**. Lesson: *a factorial
   interaction on a bilinear head output is architecture-forced; require a carrier-specific (importance-matched)
   interaction null before interpreting it.* The pre-run red-team caught this; the confirmatory `Δ_bind` came
   out exactly as predicted, validating the catch.

Logged **R24** in `CLAIMS.md`. **Next per Cam's sequence:** P3.4 (carrier origin) → recognition feasibility
gate → freeze the bounded SpliceBERT account → Caduceus transfer.

### P3.4 (carrier origin) — design red-teamed → NOT FROZEN → DEFERRED (2026-07-18)

Formalized the modernized carrier-origin prereg (`preregistrations/PREREGISTRATION_P3_4_CARRIER_ORIGIN.md` v1): every test
reframed from the old `preregistrations/PREREGISTRATION_PERIOD3_COMPUTATION.md`'s raw ΔL to a **carrier-deletion × context
INTERACTION** (E1 completion w/ mask-burden control; E2 coding-scaffold difficulty/composition-matched; E3
segmentation; E4a reading-frame-vs-absolute-phase 3×3; E4b translation leakage). Ran a **4-lens design
red-team BEFORE any compute** (`redteam/REDTEAM_SYNTHESIS_P3_4.md`).

**Outcome: NOT freezable as written — 3 convergent FATALs (verified against code/data), an R23-class pre-run
catch:**
1. **Origin vs usage** (estimand + identifiability lenses). Frozen-checkpoint ΔL = the carrier's *current
   marginal MLM contribution* = **usage**, not the training **origin** the intent claims (a feature induced by
   pressure X can be load-bearing for Y at convergence; deletion reveals Y). §9 promoted a P→behaviour(MLM)
   characterization into an origin claim — a ladder jump. Relabel fixes the language, not the ambition.
2. **E4a = codon composition, not reading-frame** (identifiability + statistics; **CODE-VERIFIED**).
   `codon_slot = (g − cds_start_0 − frame) % 3` (phase.py L68), `g = window_start + seq_pos`; abs-phase
   `φ = (seq_pos+1) % 3` ⇒ **`s − φ` cancels `seq_pos` → per-example constant**. Slot cannot vary at fixed φ
   within an example; the 3×3 is between-example only, and at fixed φ the deleted carrier is ~constant ⇒ any
   slot effect is slot-correlated **content** (codon/wobble composition), not the carrier encoding frame.
   `M-balance` (position counts) gates the wrong unit (real n = genes-per-frame-class ~13–27).
3. **Power wall** (statistics; **VERIFIED vs C8**). `carrier_surgery.json` coding-ΔL gene-clustered CI
   half-width **0.0198 nats at n=384** — the SESOI(0.02) scale for a *raw mean*. Every P3.4 estimand is a
   higher-variance contrast at the ≤140-unit MPS budget (projected exons for CI-HW<0.02: E2/E3 ~540–1200;
   E1/E4b ~850–2350; E4a ~2000+). The **equivalence-gated** readings are the maintained-hypothesis rows
   (positional-coordinate / general-substrate), which are unreachable ⇒ the modal outcome of a correct run is
   "inconclusive."

Plus (majors): E1 doesn't separate within-codon completion from **positional routing-to-same-frame-neighbours**
(the maintained hypothesis itself), and its planted-positive is dead on a frozen model (never learned a
synthetic completion rule); E2 can't separate a coding-state computation from "a positional coordinate merely
more useful where coding has more mod-3 structure"; E2/E3 match on `L_I` (a component of ΔL = L_D − L_I) →
**collider** (circuit.py docstring L17–21 warns of exactly this pattern). **Data IS buildable** (feasibility
lens verified `codon_slot` on real frame-2 sequence, 609/609; GTF has CDS+frame+UTR+start/stop) — the blocker
is **scientific, not engineering**.

**Cam's decision (AskUserQuestion, 2026-07-18): DEFER origin → transfer.** The carrier's *origin* is a
training-dynamics question a frozen-checkpoint MLM ablation cannot identify; it needs training-time evidence
(checkpoint sweep: does the carrier appear before/after coding-MLM competence; carrier-penalized retrain;
cross-model) — a separate, larger study, out of scope on MPS. **The SpliceBERT carrier account is CLOSED** at
the causal-locus + A-vs-V level (C8/C19/C32/C34/C35). **Methodological lesson (reusable):** a frozen-model
ablation identifies *current function*, never *training origin*; "why was it learned" needs training-time
interventions. **Next: Caduceus-PS behavioural transfer** (`studies/caduceus_partner_transfer/`; 5-lens
red-team done, 20-fix MASTER list; blockers = Colab-CUDA compute [no CPU/MPS fallback] + 4 forks for Cam).

### P3 closure — nucleotide-invariant phase surgery → C36 (2026-07-22)

Cam reopened the period-3 current-use question after the exploratory ±1/±2/±3 sequence-shift pattern.
The insertion route was retained as outcome-aware candidate generation only. A fixed-length sequence
rotation redesign also failed: strict construction had zero yield, and relaxed rotations altered local
MaxEnt splice-site landscapes, so sequence composition/site creation remained confounded.

The final intervention changes no nucleotide. Inside frozen annotation-safe blocks it replaces only the
learned P3 position component `C3[x]` with `C3[x±r]`. The primary compares modulo-preserving shifts
`{3,6,9}` with adjacent off-register shifts and subtracts identical surgery in an upstream-intron block.
A 5-lens red-team retained one coupling endpoint, demoted the unstable L3H1 phasor to descriptive, added
19 same-geometry signed-hidden-axis controls, and added a matched non-CDS-exon boundary.

Freeze: 943 primary genes, 150 direction-control genes, 52 matched context pairs, 457 reserve genes made
non-runnable; package/audit and 21-gate sealed development smoke passed before target compute. The target
run was interrupted after primary 943/943 and controls 109/150; immutable resume verified and skipped all
finished shards, then completed controls 150/150 and context 52/52. Aggregate outcomes remained unread
until all manifests were complete.

**C36 result:** all 11 instrument gates pass. Primary `Ψ_C=+0.141 [0.110,0.172]` nats (n=943), above the
0.05 SESOI; in-register CDS coupling ≈0.757 versus adjacent off-register ≈0.614, while intronic surgery
is nearly flat. Learned direction +0.135 exceeds 19 same-geometry controls (−0.0054..+0.0012); paired
excess +0.136 [0.051,0.222], exact rank p=0.05. Matched CDS-minus-non-CDS-exon +0.006
[−0.080,+0.093] is inconclusive.

**Licensed:** present-day, phase-aligned causal use of the learned P3 hidden direction for coupling in
CDS blocks. **Not licensed:** coding enrichment, annotated reading-frame representation, spliceosomal
frame sensing, biological mutation effects, or training origin. The failed context boundary is retained;
no reserve extension is opened.

## B3 — cross-example transduction (the generalization gap) (2026-07-23) → C38, INCONCLUSIVE (verification)

Cam authorized the confirmatory control redesign + one verification-level re-run (D-4(ii) fixed-band
fallback, D-1 M8-floor amendment, D-7(i) run). The frozen B3 prereg was reconciled onto
`b3-confirmatory-redesign` with a pre-run amendment (§A0–A6): A1 compliance restoration (paired MF2 per §6,
Endpoint C, norm-matched scramble, MASKED health), A2.1 the M8 floor, A2.2 the fixed margin-gap band,
A6 the health downgrade. Ladder (SpliceBERT.1024nt, MPS, `PYTHONHASHSEED=0`):

**Smoke (val n=24):** memorials pass (`value_linearity` 2.6e-6, `recon` 0.0); calibration passes (planted
self-gain 0.382, permuted-FPR includes 0). The R-h2 health positive-control did NOT fire — a 50×
single-token L3H1 clamp leaves distant non-site accuracy at 0.540 vs base 0.538 → distant-probe health is
structurally unfireable by a *local* injection → **health DOWNGRADED to a reported diagnostic (A6)**, not a
POSITIVE gate (owner call; the confound it guarded — a globally-broken forward pass — cannot arise for a
local injection).

**Pilot (train/discovery n=260):** primary `R_xfer` 0.892, dyadic HW 0.172 (< SESOI), `N_req` 194 (< 260 →
powered). Argmax-paired `Δ_MF2` hw 0.321 ≥ SESOI → the A2.2 trigger fires → confirmatory uses the fixed band.

**Confirmatory (val n=260, `--mf2-band`), verification tier:**
- Primary transfer LARGE + dose-monotone: `R_xfer` 0→0.534→0.912→1.178, β=0.271, dyadic CI [0.757, 1.112].
- POSITION-specific: decoy collapses `R=−0.065` CI [−0.172, −0.006].
- **NOT content-specific (a whisker short of powered):** paired band `Δ_MF2 = +0.028` CI [−0.215, 0.187],
  hw **0.2008 ≥ SESOI_R 0.20 by 0.0008** → INCONCLUSIVE, not the powered-equivalence Endpoint C. The
  margin-mismatched non-partner source transfers ~as well (0.884 vs 0.912); a coordinate-scrambled payload
  still partially transfers (`R` 0.339, CI-low 0.171). Both corroborate the not-content-specific direction.
- Health diagnostic: all arms ≈0.529 (no global forward-pass breakage).

**Verdict: `INCONCLUSIVE_underpowered` (verification tier).** Not the decisive Endpoint C the redesign
targeted — the MF2 powered-equivalence missed by 0.0008; the pre-committed SESOI/n were NOT tweaked
post-hoc. **Licensed (C38, verification):** cross-example payload transfer generalizes across gene-disjoint
matched targets in a large, dose-monotone, position-specific way. **Not licensed:** content-specificity
(unresolved; leans Endpoint C — recovery not separable from a generic acceptor-state push); any
confirmatory C-claim (verification tier, val seen; a clean call needs the sealed chr8/9 cohort — NOT
opened). B2 remains retired (FATAL-1/R20); C30 stays open.

**Open for Cam:** (a) the MF2 equivalence is 0.0008 short — whether to power it up (a NEW pre-committed n,
not a post-hoc tweak) for a decisive Endpoint C, or accept the verification-tier INCONCLUSIVE-leaning-C;
(b) CLAIMS.md on this branch is behind master (C37 present there) — reconcile before/at merge (C38 is
provisional). Files: `b3_cross_example_transduction{,_pilot,_smoke}.json`; prereg §A0–A6;
`docs/B3_CONFIRMATORY_CONTROL_REDESIGN_2026-07-23.md`.

### B3 MF2 power extension (2026-07-23, A7) — the equivalence is not powerable on val

Cam chose (a): power up MF2 for a decisive call. Per A7 (pre-run, verification-level) the confirmatory was
re-run at a **pre-committed n=512** (from 260) with the **identical** estimand / `SESOI_R` / three-way
dyadic estimator / fixed band — a power extension, not a control tweak. **Result: still
`INCONCLUSIVE_underpowered`, and the dyadic half-width did NOT shrink** (hw 0.2008 @260 → **0.2205** @512).
The MF2 paired `Δ_MF2` moved 0.028 → **+0.158** [−0.110, 0.331] (now leaning matched > mismatched), and
with more data the **scramble control clarified to NON-collapse** (R 0.339 [0.241, 0.452], CI-low 0.241 >
SESOI). Primary stayed robust (R 0.931 [0.828, 1.05], β 0.281, decoy collapses −0.049 [−0.105, −0.015]).

**Reading:** the transfer is a **large content-independent geometry floor** (a norm/energy-matched scramble
recovers ~⅓) **plus an unresolved content component** (scramble kills the majority; `Δ_MF2` leans
content-specific but straddles 0). The two probes are not jointly decisive, and — the methodologically
important point — **the dyadic CI is gene-heterogeneity-dominated, so more val examples cannot power the
equivalence.** Per the A7 stopping rule (binding, anti-chasing) the run **STOPS** at achieved precision; no
further power-up. **A clean content-specificity call requires the sealed chr8/9 cohort (NOT opened).** C38
updated to the powered (n=512) characterization; the n=260 result stays immutable. Files:
`b3_cross_example_transduction_n512.json`; prereg §A7.
