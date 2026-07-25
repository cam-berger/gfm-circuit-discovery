# CLAUDE.md — agent entry point & project core

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Concise orientation for an agent session. Canonical detail lives elsewhere (linked below); this file
aggregates **what the project is, where it stands, and where it's going** so you don't have to
reconstruct it each session.

## Mission

Discover **how genomic foundation models (GFMs) implement biological computations** — move from a model
*behavior* to a *causally supported mechanism* with explicit controls and a clear stopping point for
interpretation. **SpliceBERT is the first worked example, not the boundary.** The point is a
transferable *method*, not a catalogue of activations.

- Method & rules: `README.md`, `docs/METHODOLOGY.md`, `docs/EVIDENCE_LADDER.md`, `docs/TRANSFER_GUIDE.md`.
- **Never promote evidence** across levels (P → QK → Attention → OV/mediation → Behaviour → Biology).
  Attribution ranks; **intervention decides**. A position-table feature ≠ used by attention ≠ a causal
  computation ≠ a biological mechanism.

## Where the science stands — the SpliceBERT donor circuit (the active thread)

**Behaviour (sealed):** corrupting a donor's paired acceptor changes the native donor score
(**+0.87**), partner-specific (~37×) and donor-specific (~74×) — the donor "knows" its intron mate.

The **donor-graph arc A0→A12** dissected this end-to-end. The mechanism, as currently supported:

```
ADDRESS   position-embedding geometry (PREDOMINANTLY RELATIVE offset + modest absolute; A11/C32)
          → L2H15 / MLP2 route it → L3H1 attention selects the acceptor (not identity; A1a/R20/C22)
PAYLOAD   acceptor-local state → L0H7/H12/H10 → MLP0(mediates) → MLP1 → MLP2   (A8,A10 / C29,C31)
          ⇒ builds an ACCEPTOR-LOCAL payload that is NOT bulk composition (A9b) but whose SEMANTIC
            IDENTITY IS UNRESOLVED — "splice-strength" naming WITHDRAWN (R22); weak neg. tendency only
BIND/DELIVER  address × payload at L3H1 → donor-side low-rank U_info → transporter heads → donor readout
FAITHFULNESS  the whole computation is IRREDUCIBLY DISTRIBUTED (A12/C33): 90% needs ~72/103 nodes, NO
              knee — a rediscovered STRUCTURED CORE (the circuit above) embedded in a distributed remainder.
```

Claim map (full text + CIs + retractions in `studies/splicebert_partner_selection/CLAIMS.md`; running
rationale in `DECISIONS_A0.md`):

- **C1–C26** foundation: the sealed behaviour, the `{L3H1, L2H15}` subcircuit, the period-3 clock (C8),
  the low-rank donor-side `U_info` readout (C16–C18, replicated C26), the behavioural signature (C19).
- **C27 (A6)** circuit necessary but **NON-FAITHFUL** (F=0.005); delivery sparse, generation distributed.
- **C28 (A7)** generation **acceptor-local + serial**.  **C29 (A8)** serial **L0→L1→MLP2 MLP-backbone**
  generator (held-out; MLP2 dual address+payload; L2H15 address-only).
- **C30 (A9/A9b — R22-BOUNDED)** payload is **acceptor-local + NOT bulk composition** (A9b, identifiable);
  **semantic identity UNRESOLVED** — the splice-strength naming is **withdrawn (R22)**: A9's `r=−0.51`
  reproduces on `a9-val` but is unstable across seeded draws and non-significant under dyadic inference;
  only a weak negative directional tendency survives.
- **C31 (A10)** wiring **L0H7/H12/H10 → MLP0(mediating) → MLP1 → MLP2**.
- **C32 (A11)** address **predominantly relative + modest absolute** (A4 redesign; survival-gated).
- **C33 (A12)** **irreducibly distributed** faithfulness; the identified circuit is the top-ranked core.
- **R1–R35** retractions/corrections. **R25–R28** are the newest and the least cross-referenced — read
  them before quoting C30, C41, or the carrier-origin status: **R25** carrier origin deferred →
  corpus-attributed; **R26** SEALED §2's illustrative val figures superseded; **R27** A9b's primary DML
  nuisance switched from the preregistered GBR to linear Ridge; **R28** C41's planted-positive gate
  applied as a point check with the planted-confound arm not run.
  **R21** (load-bearing): the `empty→corrupt`/`full→clean` "exact
  memorials" are **embed-clamp-dominated** — they validate embed plumbing, NOT node-cover completeness;
  `N(C)=1.0` is a forced identity; the genuine on-path memorial is `value_linearity` (A8/A10). **R22**
  (2026-07-17): C30's "payload = splice strength" is **withdrawn** — reproducible on `a9-val` but unstable
  across independently seeded draws and non-significant under two-way gene-dyadic inference; composition-
  exclusion (A9b) survives, semantic identity does not. **Lesson:** within-sample held-out ≠ sampling
  stability; interchange pairs need sender/receiver dyadic inference.

## Current direction (in progress)

- **A9b — payload semantics *net of bulk composition* (DONE, 2026-07-17).** Cross-fitted DML (linear
  nuisance — GBR overfit the low-dim Z), two planted-CONFOUND gates + dyadic bootstrap, all calibration
  PASS. **Result:** bulk composition does NOT explain the payload transfer (`CV-R²(S~Z)<0`; identifiable,
  support 53–69>40) → the reviewer's composition confound is **REFUTED**. **But** a cross-sample sweep
  found A9's "payload = splice strength" (`r=−0.51`) exactly reproducible on `a9-val` yet **unstable across
  independently seeded draws** (held-out r −0.512→+0.017; discovery ≈0) and **non-significant under two-way
  gene-dyadic inference** (every composition-adjusted CI incl 0). ⇒ **R22** (C30 bounded; splice-strength
  naming withdrawn; semantic identity UNRESOLVED — only a weak negative tendency survives; V3 is circular).
  Files: `a9b_payload_semantics_composition.json`, `a9c_matched_composition_reconcile.json`.
- **Frozen mechanism claim:** *"A structured, top-ranked causal core binds a predominantly relative
  positional address to an acceptor-local payload, then delivers the resulting information to the donor
  readout within an irreducibly distributed computation."* (No "splice-strength" — semantic identity is
  unresolved per R22; the weak strength tendency belongs in discussion, not the established mechanism.)
  **P3 refinement (2026-07-17; C34/C35 adjudicated 2026-07-18 → R24):** the positional address is supplied by
  the period-3 carrier acting as a **distributed relative-positional routing coordinate** — read **across the
  donor–acceptor span** (C34, not at the two sites) and routing through **L3H1's attention pattern, not
  carrier-dependent changes in its values** (P3.3). Within that intervention L3H1's values are not
  carrier-dependent (**R24**: the categorical "does not carry the payload" is withdrawn — it does *not* exclude
  upstream carrier effects on payload construction); its "binding" role is not identifiable by
  carrier-deletion+patching (R23 — architecture-forced `Δ_bind`; C23's separate reconstruct-inject binding
  stands). C34's confirm split is gate-qualified (preregistered M1 gate failed 5.5%>5% → verification-level).
- **Intron-coupling arc (IC1–IC5) — opened after R22.** Tests conditional splice competitiveness (the
  post-R22 target), not absolute MaxEnt strength.
  - **IC1 DONE (DIAGNOSTIC / outcome-aware verification, NOT confirmatory):** distal intron interstitial
    ARRANGEMENT is **NOT a meaningful partner-specific coupling driver** — `ΔC̄=+0.032 nats` CI [+0.012,+0.053],
    statistically detectable but **SUB-SESOI (<0.05)**; the paired **`ΔC_specific=ΔC_partner−ΔC_sham=+0.006`,
    CI [−0.011,+0.024] INCLUDES 0** ⇒ no meaningful partner-specific effect. Well-powered (MDE 0.029),
    deterministic reproducible cohort. Heavily hardened
    (5-lens red-team → 4 pilots → freeze-audit → re-audit → mechanical audit → 2 amendments: A1 outcome-blind
    OOD-gate fix, A2 reproducibility + verdict fixes). ΔC was seen before A2 ⇒ verification, not confirmatory;
    a publication-grade null needs an untouched disjoint cohort. Coheres with the **local-window** picture
    (C28/C30). Prereg + amendments:
    `studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_IC1_DISTAL_INTRON_NECESSITY.md`.
  - **IC2 (acceptor-competition) — RETIRED (2026-07-22; formerly parked pending the P3 SpliceBERT account,
    2026-07-18).** The Head–Context Atlas **refuted** the competition-specific premise this run would confirm
    (HCA4 `|M_comp|=0.088 < |M_decoy|=0.135`; no competition-specific head/MLP module, HCA10) ⇒ confirmatory
    **not licensed**; running it would only confirm a behavior already shown not to be competition-specific.
    Its sibling draft **IC2a** is retired with it. Revivable design asset only. Prior parked-state detail
    (retained for the record): A parallel-thread exploration hardened to **v5.3** (feasibility pilot + calibration DONE,
    survived a 4-lens red-team → 2-lens re-audit → final freeze-audit): the synthetic reciprocal-swap assay is
    viable; **SLOT arm robust** — a foreign window's native-vs-displaced slot reliance `τ_D`≈0.53 (~10× SESOI):
    donor support depends on *which scaffold slot holds a competent acceptor*, not on which reliance-matched
    window occupies it. The honest **transport DiD** `β_transport=τ_D−τ_P` is intrinsically noisy (SD 0.774),
    so its ±0.05 equivalence is unreachable at the ~1,985-gene cap (would report at achieved precision).
    **Confirmatory NOT run** — Cam's spine is P3.4, not IC2. Freeze-audit flagged fixable §7-classifier blockers
    for any revival. Prereg
    `studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_IC2_ACCEPTOR_COMPETITION.md`
    v5.3; pilots `run_ic2_feasibility.py`,
    `run_ic2_calibration.py`; results `ic2_feasibility.json`, `ic2_calibration.json`.
- **P3 — the period-3 carrier's causal ROLE & LOCUS (DONE, 2026-07-17; depth-first spine).** Formalized
  P3.1–P3.3 as one prereg
  (`studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_P3_CARRIER_CAUSAL_ROLE.md` v2);
  P3.4 origin separately gated.
  **4-lens design red-team BEFORE running retired the flagship "binding" endpoint (→ R23):** the L3H1
  attention×value interaction `Δ_bind` is **architecture-forced** (`O=A·V` ⇒ the 2×2 DiD is the identity
  `(A_I−A_D)(V_I−V_D)`, no carrier-specificity) and `payload_source` is **non-identifiable**. Reframed to
  the identifiable core (`studies/splicebert_partner_selection/redteam/REDTEAM_SYNTHESIS_P3_4lens.md`):
  - **P3.2 carrier LOCUS → C34: SPATIALLY DISTRIBUTED, not site-local.** Position-selective restoration
    (disc/confirm split n=160): local donor/acceptor windows recover only ~2–4% of coupling (≪ SESOI 0.2),
    phase-matched-irrelevant & decoy-acceptor nulls ≈0, broad-span restoration needed (dose → R_C≈0.5 only
    at ±32 nt). The carrier is a distributed positional coordinate across the span (Endpoint-B locus,
    well-powered). Coheres with C32 (relative offset) + C33 (irreducibly distributed).
    **Caveat (R24): confirm split gate-qualified — preregistered M1 gate failed (5.5%>5%) → verification-level,
    not a clean confirmation.** `results/p3_carrier_locus.json`.
  - **P3.3 A-vs-V decomposition (NOT binding) → routing main ≫ value main** (confirmatory n=120, disc/conf).
    On the deleted base, restoring L3H1's intact-carrier **attention pattern** recovers **+0.257/+0.284**
    coupling; restoring its **values** recovers **≈0** (−0.011/−0.005, a genuine non-forced sub-SESOI null ⇒
    within this intervention L3H1's values are not carrier-dependent — **R24**: NOT a global "carrier does not
    build the payload" claim). The carrier's coupling contribution routes through L3H1's **attention address
    (where it looks), not carrier-dependent changes in its values** — the routing-carrier picture, causally.
    `Δ_bind`=+0.14/+0.16 CI-positive but **architecture-forced (R23) — reported, not
    interpreted** (pre-run red-team foresight confirmed). `results/p3_av_decomposition.json`.
  - **Deferred:** D_rec **recognition** (needs a strong-motif pseudo-site decoy study; decoys sparse);
    **construction-vs-binding** (non-identifiable, R23); **P3.4 origin** (separate, whole-codon/CDS/slots).
- **P3.4 (carrier origin) — DESIGN RED-TEAMED → NOT FROZEN → DEFERRED (2026-07-18).** Modernized the old
  period-3 prereg (all tests as carrier-deletion × context INTERACTIONs, not raw ΔL) and ran a **4-lens design
  red-team BEFORE any compute** → **not identifiable as an origin study on a frozen checkpoint** (3 convergent
  FATALs, verified vs code/data): (1) **origin vs usage** — frozen-model ΔL measures what the carrier is *used
  for now*, not the training *origin* the intent claims; (2) **E4a = codon COMPOSITION, not reading-frame**
  (code-verified: `codon_slot − abs-phase` is a per-example constant, and the carrier is ~constant at fixed φ);
  (3) **power wall** — the 0.02-nat equivalence bar (the maintained-hypothesis readings) is **unreachable at
  MPS budget** (C8 raw-mean gene-clustered CI half-width 0.0198 at n=384; every estimand is a higher-variance
  contrast at ≤140 units). Data IS buildable (`codon_slot` verified on real frame-2 sequence) — the blocker is
  **scientific, not engineering**. **Cam's decision: DEFER origin → transfer.** Origin needs *training-time*
  evidence (checkpoint sweep / carrier-penalized retrain / cross-model), out of scope for a frozen-checkpoint
  MLM study on MPS.
  `studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_P3_4_CARRIER_ORIGIN.md`
  (design of record, NOT frozen),
  `studies/splicebert_partner_selection/redteam/REDTEAM_SYNTHESIS_P3_4.md`.
- **SpliceBERT carrier account CLOSED** at the causal-locus + A-vs-V level, and the **period-3
  present-day-use family (C34/C35/C36) is DONE (2026-07-22)**: C8 dependence, C19 phase-independence, C32
  relative address, C34 distributed locus (R24: confirm gate-qualified), C35 attention routing (R24:
  value-null within-intervention), **C36 clean nucleotide-invariant confirmation (`Ψ_C=+0.141 [0.110,0.172]`)
  — the clean phase-aligned-use evidence C34's gate-qualified confirm lacked**. Only **origin** remains
  (training-time, deferred); D_rec recognition deferred; construction-vs-binding non-identifiable (R23).
- **CURRENT: SpliceBERT cross-context transduction + site-choice steering.** The audited Head–Context Atlas
  is committed and supplies matched multi-context cohorts and candidate components. The next causal bridge
  is gene-disjoint cross-example/context transduction, not another attribution screen; payload semantics and
  steering must retain R22 multi-draw + dyadic inference.
- **DEFERRED: Caduceus-PS behavioural TRANSFER.** Package `studies/caduceus_partner_transfer/` has a v2
  five-lens-hardened prereg and locally validated Colab pipeline, but is not frozen or run. CUDA is required;
  remaining pre-deploy calls are documented in its README. This remains the method-portability milestone
  after the current SpliceBERT context work.
- **Open:** payload semantic identity (unresolved → IC program); L1/L2 individual-head resolution; A11
  PE-edge caveat at |Δ|=192; A12 static-rank is an upper bound (true lazy-greedy would refine); **carrier
  ORIGIN (training-time question, deferred).**

## How work is done here (conventions — follow these)

1. **Prereg → smoke → confirmatory → REFLECTION → claim.** Every experiment gets a frozen study-local
   preregistration with falsifiable memorials, a `--smoke` (memorials/arithmetic only, val
   sealed), then a confirmatory run. Draw no conclusions before the memorials pass.
2. **Memorials must be able to fail, and must be on the measurement path** (R7/R21). Prefer exact
   arithmetic anchors (e.g. `value_linearity`); flag forced-by-architecture "memorials" as such.
3. **Calibration for identification claims:** planted-positive (sensitivity) + permuted-label FPR
   (specificity) gate the run — they caught 3 decision-layer bugs in A9 before any verdict.
4. **Held-out** (discovery/held-out gene split) wherever a selection or ranking is involved.
5. **Claim ledger discipline:** keep `CLAIMS.md` compact; preserve correction/retraction detail with an
   R-number in `CORRECTIONS.md` and the decision log rather than editing history silently. Report
   conditional causal effects as such, **never additive fractions**.
6. **Adversarial verification** (Workflow-orchestrated) of finished claims — it caught real overstatements
   (R21) and the A9b estimand flaw. Correct honestly; the empirical findings usually survive.
7. **Results write a manifest** beside their output; never silently overwrite the scientific record.
8. **Repo is private until submission; commit/push only per the user's cadence** (this session commits
   each confirmed experiment and pushes to keep the remote current with the evidence).

## Environment & running

- Environment (**uv is canonical**; see `REPRODUCE.md`): `uv run python scripts/splicebert/run_<exp>.py ...`.
  Historical runs used the conda env `gfm-thrust1`, retained as execution provenance only.
  (DNABERT-2/SpliceBERT load + unpadding gotchas are handled in `src/gfmcd/`). `maxentpy 0.0.4` (from
  github kepbod/maxentpy) is installed for A9/A9b's MaxEntScan (tables sha256 `8b6d7b46…b7c00`).
- Model: SpliceBERT.1024nt, post-LayerNorm BERT-MLM (6L×16H, hidden 512), **absolute learned position
  embeddings** (P=1026). Post-LN ⇒ QKV linear in the layer-input residual (enables exact
  reconstruct-inject). Checkpoint sha256 `2ad91428…0149d`.
- Run from repo root. Long runs go to the background; smoke-gate first.

## Key files

- `studies/splicebert_partner_selection/CLAIMS.md` — compact live claims (source of truth for wording).
- `docs/PREREG_CONFORMANCE.md` — where each runner departed from its frozen protocol (188 deviations,
  25 families). Read before citing any preregistered gate as enforced.
- `studies/splicebert_partner_selection/CORRECTIONS.md` — full historical ledger and R1–R35 corrections.
  (R1–R34 are table rows; R25–R28 are `##` sections later in the same file — check both when scanning.)
- `studies/splicebert_partner_selection/DECISIONS_A0.md` — the donor-graph running log (A0→A12, rationale,
  the consolidated mechanism diagram, the verification-correction pass).
- `studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_A*.md` — frozen protocols per experiment.
- `scripts/splicebert/run_a*_*.py` — the experiment scripts (reconstruct-inject, scrub, interchange).
- `src/gfmcd/splicing/`, `src/gfmcd/circuits/` — assays and reusable primitives.
- Current record branch: `master` (record consolidated 2026-07-24; reconciliation staged on `ultrareview-cleanup-2026-07-24`).
