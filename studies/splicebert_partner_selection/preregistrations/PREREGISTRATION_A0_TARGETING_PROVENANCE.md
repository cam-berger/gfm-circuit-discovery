# Pre-registration — A0: stage-wise provenance of L3H1's acceptor targeting (donor-graph experiment 4)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_a0_targeting_provenance.py` is executed. Every stage
definition, readout, reference construction, statistic, and pass/fail threshold below is fixed
here and imported from `gfmcd.splicing.circuit` / `gfmcd.splicing.address`, not redefined in the
runner. No number produced by the run may revise any threshold in §5. Pre-run corrections are
logged inline as `**Amendment (YYYY-MM-DD, pre-run, no data seen).**`; post-unblinding
corrections go to `DECISIONS_A0.md` / `CLAIMS.md`, never here.

---

## 0. Why this document exists, and where it sits

The acceptor-address experiment (C20) left a specific, sharp discrepancy on the table:

- L2H15 supplies **~0.66** of the donor coupling (`l2_accounting.json`, restore arithmetic
  verified, calibration = 1.000);
- L2H15 supplies only **~0.11** of the matched-decoy attention **preference** (C20, retained
  0.89 under ablation, p = 0.021).

The tempting read — "the preference is built diffusely by many upstream heads" — is **not**
licensed, and a naive component-attribution scan would manufacture a distributed constructor
whether or not one exists. The discrepancy is equally consistent with a different and more
likely mechanism: **L3H1's preference for the intron-mate acceptor is largely resident in the
residual stream before layer 3** (in the token/positional embeddings or early-layer site
features), while L2H15's causal contribution is to **refine the acceptor key, shape the donor
query, and supply payload/timing** — a coupling role, not a targeting-construction role. The
non-additivity of the serial `L2H15 → MLP2 → L3H1` recoveries (key@acceptor 0.43, query@donor
0.18, MLP2 conduit 0.65 all measure overlapping flow — they are **not** a partition) means the
existing numbers cannot decide this.

A0 asks the **provenance** question that must be answered *before* any component scan:

> At what depth in the residual stream supplied to L3H1 does the donor query's targeting of the
> real intron-mate acceptor (over a phase/distance-matched non-partner **real** acceptor) become
> **sufficient**, and is that sufficiency carried by the **key** side, the **query** side, or a
> **Q×K interaction**?

The answer routes A1. It distinguishes **four genuinely different mechanisms**: (i) pre-existing
positional preference, (ii) pre-existing sequence-content salience, (iii) constructed acceptor
**keys**, (iv) constructed donor **queries**. Only mechanisms (iii)/(iv) justify a builder scan;
(i)/(ii) send us to a positional-geometry or controlled-biological-edit experiment instead.

**Firewall placement.** A0 lives entirely at **QK / selection** (a pre-softmax logit
reconstruction) with an attention-level diagnostic. It reaches **no** OV/mediation, behaviour,
or biology level. `S_A` is a **mechanistic intermediate only**: the natural matched-decoy
*behavioural* task was withdrawn as degenerate, and A0 does **not** resurrect a
partner-selection behavioural claim (§7).

---

## 1. The target — imported unchanged

- **Head / query token.** L3H1 = `(3, 1)`; the query token is the masked **donor** GT position
  (token index = `donor_pos + prefix_tokens`), identical to the acceptor-address experiment.
- **Candidates.** Per eligible example, the model's intron-mate partner `a⁺ = next_acceptor_pos`
  and, as the **primary** contrast, a phase/distance-matched non-partner **real** acceptor
  `aʳ = real_acceptor_decoy(x, tol = --real-tol)`; the generic phase-matched AG decoy
  `a⁻ = phase_matched_decoy(x, tol=40)` is the **secondary** contrast. Key windows use
  `ACC_WINDOW_OFFSETS = (−3, −2, −1, 0)` around each candidate.
- **Contrast naming (per review).** The `a⁺` vs `aʳ` comparison is the **real-acceptor
  contrast**, *not* "identity-clean." Both are real 3′ splice sites, so splice-site *strength*
  is controlled by construction, but it does **not** isolate partner identity from residual
  differences in acceptor strength, geometry, distance, or local context — those are reported
  covariates, not eliminated. The withdrawn partner-selection claim stays withdrawn.
- **Pre-softmax score.** `qk_logit_to_window` (`address.py`): head-1 slice `[32:64]` of L3's
  Q/K projection outputs (`capture_proj`), `q·k/√32`, mean over the 4-token window. This is the
  same instrument the QK-reconstruction preflight already locks to the model's own L3H1
  attention (`softmax(q·k/√32 + mask)` reproduces `attentions[3][:,1]` to < 1e-3).

---

## 2. Three targeting reconstructions, plus an absolute-attraction diagnostic

Let `q_F`, `k_{a⁺,F}`, `k_{aʳ,F}` be the query/keys from the **true full** L3 input, and at
reconstruction stage `s` let `q_s`, `k_{a⁺,s}`, `k_{aʳ,s}` be the same projections computed from
the stage-`s` reconstructed residual (§3). Write `Δk_s = k_{a⁺,s} − k_{aʳ,s}` (window-mean keys).
The three curves, each `/√32`:

| Curve | Definition | Isolates |
|---|---|---|
| **`S_K(s)`** key provenance | `q_F · Δk_s / √32` | where the **acceptor key** distinction becomes sufficient (query fixed at final) |
| **`S_Q(s)`** query provenance | `q_s · Δk_F / √32` | where the **donor query** distinction becomes sufficient (keys fixed at final) |
| **`S_J(s)`** joint | `q_s · Δk_s / √32` | the realized targeting under both reconstructed |

`S_J(7) = S_K(7) = S_Q(7) = S_A` (the true full score) by construction — the shared endpoint is
a memorial (§5). **Disagreement among the three curves is itself the Q×K-interaction readout**:
`S_J(s) − S_K(s) = (q_s − q_F)·Δk_s/√32` is the query deficit *conditional on stage-s keys*, and
`S_Q` supplies the complementary query provenance *conditional on final keys*; neither alone is
"the query contribution," and their divergence localizes the interaction (per review §1).

**Absolute-attraction diagnostic (required).** A preference *difference* `a⁺ − aʳ` can miss a
query effect that raises **all** real acceptors relative to background while barely moving
`a⁺ − aʳ` — that contributes to coupling but vanishes from `S_A`. So alongside the differences,
report, per stage and per curve, the **acceptor-window attraction vs matched background**:
`LSE_win(a⁺) − LSE_bg` and `LSE_win(aʳ) − LSE_bg`, where `LSE_win` is the log-sum-exp of the
window's pre-softmax logits and `LSE_bg` is the same over a distance-matched non-site background
(`build_windows(...).control`). This closes the "query@donor lifts the whole acceptor class"
possibility that the generic-AG secondary curve only partially addresses.

---

## 3. Cumulative-sufficiency reconstruction (the confirmatory A0 method)

**Framing (per review §2).** A prefix-clean curve locates where clean information becomes
**sufficient** in the residual supplied to L3H1 — **not** where it was originally computed (a
layer-0 signal that only becomes usable after MLP2 will make the curve jump at MLP2). A0 is
therefore explicitly a *sufficiency* localization, disambiguated by a reciprocal *necessity*
diagnostic.

**Method.** Because SpliceBERT is **post-LayerNorm**, the residual is not an additive sum of
writes; reconstruction is by intervention, not by frozen-LayerNorm linear attribution. On the
**clean** forward pass, `clamp_write` forces the residual writes of every block **after** stage
`s` to their **reference** value at the reconstructed positions, and the model's own forward
(all intervening LayerNorms, attention mixing) runs; L3's Q/K are read with `capture_proj`. Every
LayerNorm is positionwise, so reconstructing **only** `{donor, a⁺, aʳ}` (and `a⁻` for the
secondary arm) is exact for L3's Q/K at those positions — this is why the reconstruction is cheap
and why those three positions suffice (review §4).

**Sublayer-resolution stages (per review §3).** The block-level stages cannot separate an
attention jump from an MLP jump, so A0 uses **sublayer** stages:

| `s` | Stage-`s` residual = clean through … , reference after |
|---|---|
| 0 | (all reference — denoising floor) |
| 1 | token + positional embedding output (post-embedding LN) |
| 2 | + attention 0 |
| 3 | + MLP 0 |
| 4 | + attention 1 |
| 5 | + MLP 1 |
| 6 | + attention 2 |
| 7 | + MLP 2  (= true full L3 input) |
| 8 | actual cached L3 input — **arithmetic verification only** |

Stage 7 (nothing clamped) is the clean run; stage 8 is the independently-cached clean L3 input.
Their L3 Q/K/`S_A` at the three positions must coincide within tolerance (§5 memorial). Stage 0
(everything reference) is the **generic-preference floor**.

> **Amendment (2026-07-16, pre-run, no A0 data seen).** The draft said `S_A(0) ≈ 0` is expected.
> With the **role-specific** mean-resample reference (below), the a⁺-role mean and the
> a(other)-role mean differ by the *population-average* role difference (chiefly absolute position
> / donor-distance and average local context), so `S_A(0)` measures the **population-generic**
> preference and is generally **nonzero** — this is by design and is the *more* informative floor:
> it separates population-generic geometric/site preference (`S_A(0)`) from **example-specific
> construction** (`S_A(7) − S_A(0)`). The stage curve is read as a *sufficiency accumulation from
> the generic floor to the example-specific full*, not from zero. `S_A(0)` is reported, never
> gated. (A grand-mean reference collapsing all roles would force a zero floor but wrongly credit
> the clean stages with the generic positional preference, so it is not used.)

**Reference construction (the one resolved design decision — flagged, amendable pre-run).** The
reference is a **mean-resample** over the eligible pair set (seed 0): for each block write and the
embedding output, and for each positional role `r ∈ {donor, a⁺, aʳ, a⁻}`, the reference at `r` is
the mean of that component's clean write over all examples at their role-`r` position. This keeps
the reconstructed residual on-distribution (norm-preserving, unlike zero-ablation) and carries
**no** example-specific `a⁺`/`aʳ` distinction, so any surviving `Δk_s` is contributed by the
prefix-clean stages. A run where `a⁺` was destroyed (the corrupt run) is **rejected** as the
reference: it perturbs the `a⁺` token at the embedding stage, which would trivially "restore" the
preference the instant the clean embedding is patched in and conflate token restoration with
downstream construction.

**Reciprocal necessity diagnostic (required, per review §2).** In addition to the prefix
**denoising** curve above:

- **Leave-one-stage-out noising.** Start from the full clean reconstruction (stage 7); replace
  **each single** stage group `g ∈ {embed, attn0, mlp0, attn1, mlp1, attn2, mlp2}` with its
  reference counterpart; record the drop `S_A(full) − S_A(¬g)`.

**Localization is strong** iff the stage that produces the prefix-denoising jump is the same
stage whose leave-one-out removal is necessary. **Disagreement is a real result** — it signals
serial dependence, redundancy, or interaction, and is handed to A1 rather than smoothed over.

---

## 4. Estimand and statistics (explicit, per review §4)

- **What clean and reference runs differ in:** identical input sequence and mask; the reference
  substitutes the mean-resample write (§3) at `{donor, a⁺, aʳ, a⁻}` for all blocks after stage
  `s`. Nothing else differs. The corruption (`AG→CC`) is **not** used anywhere in A0.
- **Reconstructed positions:** `{donor, a⁺, aʳ}` for the primary arm, `+ a⁻` for the secondary;
  positionwise-LN exactness is a preflight memorial.
- **Real-decoy matching:** `aʳ` matched on reading phase (exact) and donor-distance (within
  `--real-tol`, default 60, reported as a **signed** gap `dist(aʳ) − dist(a⁺)` because
  non-partner real acceptors sit farther from the donor on average); splice-site strength matched
  by construction (both real); local context **reported, not matched** (`match_audit`).
- **Primary statistic:** the **raw** QK-logit differences `S_K(s), S_Q(s), S_J(s)` (and the
  absolute-attraction terms), in logit units, with gene-grouped bootstrap 95% CIs (`gboot`,
  n_boot = 2000, seed 0). The **shape** of the curve across `s` is the result.
- **Secondary statistic:** normalized sufficiency `Ŝ(s) = S(s) / S(7)`, reported **only** when the
  denominator magnitude `|S(7)|` exceeds a preregistered floor `--min-denom` (default 0.15 logit
  units); otherwise the ratio is withheld and only the raw curve is reported (per review §4:
  no unstable normalized ratios on near-zero denominators). Normalized quantities use
  ratio-of-means with `gboot_ratio`, never a mean of per-example ratios (R10).
- **Nulls.**
  - *Stage localization:* the reciprocal leave-one-out is the null-free necessity check; add-one
    permutation `p` (`empirical_upper_p_value`) is reported for any single stage claimed as the
    sufficiency locus, against the other stage groups' drops.
  - *A1 head null (foreshadowed, not run here):* the layers-0–2 matched single-head null (47
    heads, `matched_random_head_sets`, R9-deduped).
  - *A1 MLP null (foreshadowed):* "matched by class and layer" is **impossible** for MLPs (one
    per layer, per review §4). The preregistered MLP null is a **resample-pair** null — recompute
    the stage's `Δk` with `(a⁺, aʳ)` replaced by a matched non-partner real-acceptor pair at the
    same positional roles — plus a **matched-control-position** null (the same intervention read
    at `build_windows(...).control`). Neither is used in A0; both are fixed here so A1 cannot
    choose its null after seeing A0.

---

## 5. Gates, decision table, and conditional branch

**Preflight (written in the same process, before any §2 number is interpreted;
`results/splicebert_coupling/preflight_a0.json`, global `PASS` = AND):**

- `positionwise_reconstruction` — the 3-position reconstruction reproduces the **full-tensor**
  L3 Q/K and `S_A` at `{donor, a⁺, aʳ}` within 1e-4 (validates the positionwise-LN restriction).
- `stage7_equals_clean` — stage-7 reconstruction (no clamps) reproduces the clean run's L3 Q/K to
  < 1e-5 (the reconstruction plumbing adds nothing when it clamps nothing).
- `stage8_arithmetic` — the independently-cached clean L3 input equals the stage-7 residual at the
  three positions within 1e-4.
- `qk_reconstruction` — `softmax(q·k/√32 + mask)` reproduces `attentions[3][:,1]` to < 1e-3
  (G-QK, imported unchanged from the acceptor-address preflight; SDPA-zero trap guarded with
  `eager=True`).
- `probability_invariants` — eager attention rows sum to 1 ± 1e-3, L3H1 not identically zero.
- `resolution_pair_match` — the real-acceptor arm exists (`n_real ≥ --min-real`, default 60) and
  the generic arm passes G-Match (phase mismatch 0; `dist_bias ≤ 8`).
- `reference_is_clean_sequence` — assert the reference run uses the **unperturbed** sequence
  (tripwire against the rejected corrupt-run reference).
- `both_strands_present`; `checkpoint_recorded` (sha256).

**If preflight fails, the run STOPS** and no §2 curve is reported (a `--force` flag exists for
DIAGNOSTIC-only inspection and marks its output non-confirmatory).

**Decision table (routes A1):**

| A0 outcome | Mechanism | Next experiment |
|---|---|---|
| Most `S_A` present at stage 1 (embeddings), and leave-one-out confirms `embed` necessary | pre-existing preference (positional or sequence) | **§5.1 token-vs-position branch first**, then geometry-intervention *or* controlled biological edits |
| Sharp jump at one sublayer, denoising & leave-one-out agree | genuine construction at that sublayer | A1 scan **restricted to that sublayer's components**, then joint restore + leave-one-out |
| `S_A` accumulates across ≥ 3 stages, or denoising & leave-one-out disagree | distributed / serial / interaction | A1 full layers-0–2 scan + joint reconstruction; interaction resolved with `S_K`/`S_Q`/`S_J` |
| Divergence dominated by `S_Q` (not `S_K`) | targeting is **query-constructed**, not key-constructed | A1 targets the donor-query pathway (L2H15 query@donor, C6), not acceptor-key builders |

### 5.1 Conditional token-vs-position branch (per review §3)
Triggered **only** if stage 1 dominates. A `2×2` intervention recomputes the embedding LayerNorm
under `{clean, reference} token embedding × {clean, reference} positional embedding` at the three
positions, reading `S_A`. This separates **sequence content** (→ controlled PPT/branchpoint
edits are the right next step) from **positional geometry** (→ a positional/distance intervention
is the right next step — *not* biological editing). **Scope note:** with single-nucleotide
tokens, the stage-1 score aggregates only the **4-token AG-neighbourhood window**
`(−3,−2,−1,0)`; an upstream polypyrimidine tract / branch point (~10–40 nt upstream) is **not**
inside that window and can enter `S_A` **only** via the attention stages (2/4/6) moving it into
the `a⁺`/`aʳ` residual. So a large stage-1 preference is, by construction, local-window +
position — the branch adjudicates which — and a PPT/branchpoint story would have to appear as an
**attention-stage** jump, not a stage-1 one. This scoping is itself a reportable A0 result.

---

## 6. Data, compute, outputs, and smoke discipline

- **Data:** val (chr11/16/19), eligibility per `CouplingConfig`, both strands (R11); target
  **n ≥ 250** eligible pairs with a valid phase-matched decoy (secondary/generic arm) and
  **n_real ≥ 30** with a real-acceptor decoy (primary arm). Seed 0 for all references, nulls,
  bootstraps.

  > **Amendment (2026-07-16, pre-run, no A0 data seen).** The draft required `n_real ≥ 60`. The
  > real-acceptor decoy yield is a *known design fact* from the prior acceptor-address confirmatory
  > run (`acceptor_address.json`: 30 real decoys at n = 260, rate ~11.5%) — not an A0 outcome — so
  > 60 is infeasible at n = 260 and would falsely block the run. Two pre-run corrections: (i) the
  > primary-arm floor is **n_real ≥ 30**, with its **modest power labelled** (the real arm confirms
  > the generic curve is not a salience artifact; the well-powered stage resolution comes from the
  > n ≥ 250 generic arm); (ii) the primary arm draws its real-acceptor pairs from the **full
  > screened pool** (`--real-cap`, target ~60–120), not the first `n`, so it is as powered as the
  > data allows without inflating the secondary arm's cost. No A0 curve was computed before this
  > edit.
- **Compute:** `load_adapter(DEFAULT_MODEL, device, eager=True)`; minutes on CPU/MPS.
- **Smoke discipline (per review §4 — keep val sealed).** The smoke run checks **arithmetic and
  preflight invariants ONLY** (`positionwise_reconstruction`, `stage7_equals_clean`,
  `qk_reconstruction`, a model-level three-curve **endpoint-identity** self-check `S_K(7)=S_Q(7)=
  S_J(7)` and curve **finiteness**, shapes, pair counts) and **emits no `S_K`/`S_Q`/`S_J` effect
  direction or magnitude**. No small-`n` inspection of biological outcomes precedes the frozen run,
  so the confirmation set is not turned exploratory. If any effect-direction number is ever
  inspected before the confirmatory run, those genes are excluded from confirmation and the
  adaptation is recorded in `DECISIONS_A0.md`.

  > **Amendment (2026-07-16, pre-run, no A0 data seen).** The draft listed a residual-space
  > `stage8_arithmetic` memorial ("cached L3 input == stage-7 residual"). The reconstruction works
  > in **Q/K space** (via `capture_proj`), not residual space, so that check is realized exactly as
  > `stage7_equals_clean` (no-clamp reconstruction reproduces the clean L3 Q/K) **and**
  > `positionwise_reconstruction` (3-position clamp == all-position clamp at the tracked positions);
  > together these lock that the 7 components fully span the L3 input with nothing missing. The
  > separate residual comparison would be a trivial identity (stage 7 *is* the clean run) and is
  > dropped as non-informative. Logged in `DECISIONS_A0.md` (R-A0-3).
- **Outputs:** `scripts/splicebert/run_a0_targeting_provenance.py`; the reconstruction kernel is a
  new module `src/gfmcd/splicing/provenance.py` (the embedding-output clamp is the only new
  primitive); `results/splicebert_coupling/a0_targeting_provenance.json` + `<stem>.manifest.json`
  (`status = "CONFIRMATORY"` **only when** `preflight_PASS and n ≥ 250 and n_real ≥ 30`, else
  `"DIAGNOSTIC"`); `results/splicebert_coupling/preflight_a0.json`; memorial tests in
  `tests/test_a0_memorials.py` (three-curve endpoint identity `S_K(7)=S_Q(7)=S_J(7)`; the
  mean-resample reference carries no `a⁺`/`aʳ` distinction; pure QK window / attraction readouts
  match hand-computation; positionwise exactness runs live in the preflight).

---

## 7. Post-hoc reflection (REQUIRED — do not skip)

The result JSON's `REFLECTION` block must state, verbatim and without softening:

1. **Pass/fail** against every preflight memorial and the §5 decision table — which of the four
   mechanisms the curves select, or that they are ambiguous.
2. **Sufficiency ≠ computation.** Restate that the prefix curve is a *sufficiency* localization;
   report whether the denoising jump and the leave-one-out necessity **agree**, and if not, name
   the serial/redundant/interaction pattern handed to A1.
3. **Key vs query vs interaction.** Report `S_K`, `S_Q`, `S_J` divergence and the
   absolute-attraction diagnostic; state whether targeting is key-constructed, query-constructed,
   or interaction-dominated, and whether a query effect exists that raises the acceptor class
   without moving `a⁺ − aʳ`.
4. **`S_A` is an intermediate, not behaviour.** Confirm no partner-selection behavioural language
   appears; the real-acceptor contrast is named as such and its residual strength/geometry/context
   confounds are reported (signed distance gap), not claimed away.
5. **Escaped failure modes:** any stage whose result could not have failed, any curve identity
   that holds to too many significant figures (the tell of one quantity wearing two hats), and any
   reference-construction artifact (norm inflation, mean-resample leakage of the `a⁺`/`aʳ`
   distinction).

A null, ambiguous, or "it was already in the embeddings" result is a real result about where the
targeting lives, and is reported as such — it is arguably the *more* informative outcome, because
it retires the diffuse-builder hypothesis that motivated A0.

---

**Provenance.** Model checkpoint `SpliceBERT.1024nt/pytorch_model.bin` sha256 `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`. Results and manifests carry the same hash via `ExperimentManifest`; the reconstruction machinery and study code are committed in the `fix:`/`study:` changeset for this arc.
