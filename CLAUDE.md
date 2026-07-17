# CLAUDE.md — agent entry point & project core

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
          ⇒ computes LOCAL ACCEPTOR-CONSENSUS "SPLICE STRENGTH"   (A9/C30; being hardened in A9b)
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
- **C30 (A9)** payload = **local splice-site strength** (matched interchange, calibrated, held-out;
  composition confound OPEN → A9b).
- **C31 (A10)** wiring **L0H7/H12/H10 → MLP0(mediating) → MLP1 → MLP2**.
- **C32 (A11)** address **predominantly relative + modest absolute** (A4 redesign; survival-gated).
- **C33 (A12)** **irreducibly distributed** faithfulness; the identified circuit is the top-ranked core.
- **R1–R21** retractions/corrections. **R21 is load-bearing for current work:** the
  `empty→corrupt`/`full→clean` "exact memorials" are **embed-clamp-dominated** — they validate embed
  plumbing, NOT node-cover completeness; `N(C)=1.0` is a forced identity. The genuine on-path memorial
  is `value_linearity` (A8/A10). Do not over-claim the scrub-cover memorials.

## Current direction (in progress)

- **A9b — payload semantics *net of composition*** (the remaining vulnerability: the bridge from causal
  circuit to interpretable computation). An estimand red-team established the honest framing: splice
  strength is **separable from BULK/frequency composition** (GC, PPT-fraction, k-mer frequency) but
  **definitionally inseparable from the CORE MOTIF it is computed from** (V2b is a function of the 4 core
  nt). Being rebuilt as a **cross-fitted double-ML** estimator with a low-dim frequency adjustment set,
  **two planted-CONFOUND stop-gates** (composition-only → β_str null; collinear-strength → β_str
  recovered), dyadic bootstrap, CI-excludes-zero decision. Prereg:
  `PREREGISTRATION_A9b_PAYLOAD_SEMANTICS_COMPOSITION.md`. **Then** an adversarial verification of the
  final A9b spec (the A9 verification was tabled pending A9b).
- **Frozen mechanism claim (target):** *"A structured, top-ranked causal core binds a predominantly
  relative positional address to an acceptor-local splice-strength payload, then delivers the resulting
  information to the donor readout within an irreducibly distributed computation."*
- **Open:** A9b composition confound (in progress); L1/L2 individual-head resolution; A11 PE-edge caveat
  at |Δ|=192; A12 static-rank is an upper bound (true lazy-greedy would refine).

## How work is done here (conventions — follow these)

1. **Prereg → smoke → confirmatory → REFLECTION → claim.** Every experiment gets a frozen
   `PREREGISTRATION_*.md` with falsifiable memorials, a `--smoke` (memorials/arithmetic only, val
   sealed), then a confirmatory run. Draw no conclusions before the memorials pass.
2. **Memorials must be able to fail, and must be on the measurement path** (R7/R21). Prefer exact
   arithmetic anchors (e.g. `value_linearity`); flag forced-by-architecture "memorials" as such.
3. **Calibration for identification claims:** planted-positive (sensitivity) + permuted-label FPR
   (specificity) gate the run — they caught 3 decision-layer bugs in A9 before any verdict.
4. **Held-out** (discovery/held-out gene split) wherever a selection or ranking is involved.
5. **Claim ledger discipline:** append to `CLAIMS.md`; correct/retract with an R-number rather than
   editing history silently. Report conditional causal effects as such, **never additive fractions**.
6. **Adversarial verification** (Workflow-orchestrated) of finished claims — it caught real overstatements
   (R21) and the A9b estimand flaw. Correct honestly; the empirical findings usually survive.
7. **Results write a manifest** beside their output; never silently overwrite the scientific record.
8. **Repo is private until submission; commit/push only per the user's cadence** (this session commits
   each confirmed experiment and pushes to keep the remote current with the evidence).

## Environment & running

- Conda env **`gfm-thrust1`**: `conda run -n gfm-thrust1 python scripts/splicebert/run_<exp>.py ...`
  (DNABERT-2/SpliceBERT load + unpadding gotchas are handled in `src/gfmcd/`). `maxentpy 0.0.4` (from
  github kepbod/maxentpy) is installed for A9/A9b's MaxEntScan (tables sha256 `8b6d7b46…b7c00`).
- Model: SpliceBERT.1024nt, post-LayerNorm BERT-MLM (6L×16H, hidden 512), **absolute learned position
  embeddings** (P=1026). Post-LN ⇒ QKV linear in the layer-input residual (enables exact
  reconstruct-inject). Checkpoint sha256 `2ad91428…0149d`.
- Run from repo root. Long runs go to the background; smoke-gate first.

## Key files

- `studies/splicebert_partner_selection/CLAIMS.md` — the claim/retraction ledger (source of truth for claims).
- `studies/splicebert_partner_selection/DECISIONS_A0.md` — the donor-graph running log (A0→A12, rationale,
  the consolidated mechanism diagram, the verification-correction pass).
- `studies/splicebert_partner_selection/PREREGISTRATION_A*.md` — frozen protocols per experiment.
- `scripts/splicebert/run_a*_*.py` — the experiment scripts (reconstruct-inject, scrub, interchange).
- `src/gfmcd/splicing/`, `src/gfmcd/circuits/` — assays and reusable primitives.
- Active branch: `donor-graph-address-payload`.
