# Pre-registration — A1a: token-vs-position decomposition of the embedding-resident targeting difference

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_a1_token_position.py` is executed. Every cell, readout,
reference, and threshold is fixed here and imported from `gfmcd.splicing.provenance` /
`address.py`. Pre-run corrections are logged inline as
`**Amendment (YYYY-MM-DD, pre-run, no data seen).**`; post-unblinding corrections go to
`DECISIONS_A0.md` / `CLAIMS.md`.

---

## 0. Why this document exists, and where it sits

A0 (C21) established that L3H1's donor-query **targeting difference** `S_A` (a⁺ over a
phase/distance-matched decoy) is **pre-resident in the token+positional embeddings** and not
constructed by layers 0–2, while the upstream query pathway supplies a class-level acceptor
attraction *gain*. A0 deliberately did **not** adjudicate what the embedding-resident difference
*is*: its primary (real-acceptor) arm carries a large distance confound (median +19.5 nt) and its
secondary (generic-AG) arm conflates identity with site-strength salience. A0's own decision table
(§5.1) routes here:

> Is the embedding-resident targeting difference carried by **positional geometry** (the a⁺/decoy
> sit at different absolute positions ⇒ different position embeddings ⇒ different donor→key
> geometry) or by **local sequence content** (the a⁺/decoy differ in their local token identities)?

This is a `2×2` intervention on the two additive embedding components. It reaches **no** level
beyond **QK / selection**; `S_A` is a mechanistic intermediate, and **no** partner-selection
behavioural claim is made (the natural matched-decoy task was withdrawn as degenerate).

---

## 1. Target and readout — imported unchanged

- **Head / query.** L3H1 `(3,1)`; query token = masked donor. `S_A = mean_win(q_d·k_{a⁺}) −
  mean_win(q_d·k_{decoy})`, head-1 slice `[32:64]`, `/√32`, window `(−3,−2,−1,0)`
  (`provenance.sa_difference`). Same instrument A0/C20 use.
- **Pairs.** The A0 confirmatory sets: primary = real-acceptor contrast (a⁺ vs `aʳ`, from the full
  screened pool, `--real-cap`), secondary = generic-AG (a⁺ vs `a⁻`, n ≥ 250). Both strands (R11).
- **Contrast naming.** "real-acceptor contrast" (not identity-clean); confounds reported.

---

## 2. The 2×2 intervention

BERT computes `embedding = LayerNorm(word(input_ids) + position(position_ids) + type)`. The two
additive components are clamped **separately** at the **8 key-window positions** (a⁺ window ∪
decoy window; the donor query position is never clamped) to a **shared reference**, so neutralising
a dimension removes that dimension's a⁺/decoy discrimination. The model then runs normally (the
embedding LayerNorm and layers 0–5 recompute) and L3's Q/K are read (`capture_proj`).

- **Shared reference (the design choice that makes the isolation clean).** For each window offset
  `δ ∈ {−3,−2,−1,0}`, `word_ref[δ]` = mean of the clean **word** embedding over **all pairs and
  over both roles** (a⁺ and decoy) at `site+off+δ`; `pos_ref[δ]` likewise for the **position**
  embedding. Clamping the a⁺ window token at offset `δ` **and** the decoy window token at offset
  `δ` **both** to `*_ref[δ]` makes their embedding on that dimension *identical* — so any surviving
  `S_A` on the neutralised dimension is not from that dimension. (This is a **shared** grand mean
  over both roles, deliberately *not* the role-specific mean-resample of A0, whose purpose was the
  opposite — to preserve the generic role difference.)

- **Four cells** (per pair, on the clean sequence):

  | cell | word emb | position emb | reads |
  |---|---|---|---|
  | `full` | clean | clean | `S_A` (must ≈ A0 `S_J` full — cross-check) |
  | `word_neutral` | **shared ref** | clean | `S_A` with local content neutralised |
  | `pos_neutral` | clean | **shared ref** | `S_A` with positional geometry neutralised |
  | `both_neutral` | **shared ref** | **shared ref** | the **floor** (a⁺/decoy window inputs identical) |

- **Attribution (raw logits primary; gene-bootstrap CIs, ratio-of-means for any normalised form):**
  - **position_effect** = `S_A(full) − S_A(pos_neutral)` — how much removing positional geometry
    drops the targeting difference.
  - **content_effect** = `S_A(full) − S_A(word_neutral)` — how much removing local content drops it.
  - **interaction** = `S_A(full) − S_A(word_neutral) − S_A(pos_neutral) + S_A(both_neutral)`.
  - **floor** = `S_A(both_neutral)` — reported, and cross-checked against A0 (below).

---

## 3. Gates and their falsifiers

**Preflight** (`results/splicebert_coupling/preflight_a1_token_position.json`, global `PASS` = AND):

- `qk_reconstruction` — `softmax(q·k/√32 + mask)` reproduces `attentions[3][:,1]` < 1e-3 (G-QK).
- `probability_invariants` — eager attention rows sum to 1 ± 1e-3, L3H1 not identically zero.
- `shared_ref_neutralises` — **the key memorial.** With **both** components clamped to the shared
  reference, the a⁺ and decoy key-window **embedding-LayerNorm outputs are identical** at matched
  offsets to < 1e-4 (the shared reference truly removes the discrimination at the embedding). This
  can fail (a wrong reference, a broadcast bug) and is what licenses the decomposition.
- `full_matches_A0` — `S_A(full)` reproduces the A0 `S_J` full value for the same arm within a
  reported tolerance (both use the same clean forward + `sa_difference`); **reported**, and it
  fails only if the instrument drifted.
- `resolution_pair_match`, `real_acceptor_arm_available` (n_real ≥ 30), `both_strands_present`,
  `checkpoint_recorded`.

**If preflight fails the run STOPS** (a `--force` flag marks output DIAGNOSTIC).

**A0-consistency (reported, not a hard gate):** because A0 showed the difference is
embedding-resident, `both_neutral` (a⁺/decoy window embeddings identical) should leave only the
difference that attention mixes in from **surrounding** context; `|floor|` is expected **small**
relative to `S_A(full)`. A large floor would mean substantial discrimination lives **outside** the
4-token window (e.g. an upstream polypyrimidine tract reached by attention) — a real, reportable
result that revises A0's "embedding-resident" scope, not a bug.

**Scope note (carried from A0 §5.1).** The window is 4 single-nucleotide tokens, so `content_effect`
is **local-window** content only; an upstream PPT / branch point (~10–40 nt upstream) is outside
the window and can enter `S_A` only via attention — which A0 found adds little (layer stages
|LOO| < 0.9). So a PPT story would surface as a large `floor`, not as `content_effect`.

---

## 4. Decision

| Outcome | Reading | Next |
|---|---|---|
| `position_effect` ≫ `content_effect` (CI-separated) | the difference is **geometry** (distance/absolute position) | distance/geometry intervention; the identity interpretation is further weakened (consistent with the primary-arm distance confound) |
| `content_effect` ≫ `position_effect` | the difference is **local sequence** (site strength / local motif) | controlled local-content edits (matched AG, vary local pyrimidine content) |
| both comparable / large interaction | geometry and content are entangled at the embedding | report the interaction; both intervention arms needed |
| large `floor` | discrimination lives outside the 4-token window (attention-reached context) | widen the window / attention-context provenance |

The **primary (distance-confounded) vs secondary (distance-matched)** contrast is itself
diagnostic: position-domination in the primary but content-domination in the secondary would show
the "preference" is distance in the real-acceptor arm and site-strength in the generic arm — both
non-identity, keeping the withdrawn partner-selection claim withdrawn.

---

## 5. Data, compute, outputs, smoke discipline

- **Data:** A0 val sets (chr11/16/19), both strands; n ≥ 250 generic, n_real ≥ 30. Seed 0.
- **Compute:** `load_adapter(DEFAULT_MODEL, device, eager=True)`; minutes.
- **Smoke (`--smoke`):** arithmetic/preflight invariants ONLY (`qk_reconstruction`,
  `shared_ref_neutralises`, shapes, pair counts) + a finiteness self-check on the four cells;
  **no** `S_A`/effect direction emitted — val stays sealed.
- **Outputs:** `scripts/splicebert/run_a1_token_position.py`; kernel added to
  `provenance.py` (`clamp_embed_component`, `capture_embed_components`, shared-ref builder — no new
  module); `results/splicebert_coupling/a1_token_position.json` + `<stem>.manifest.json`
  (`status = CONFIRMATORY` only when `preflight_PASS and n ≥ 250 and n_real ≥ 30`); memorial tests
  in `tests/test_a0_memorials.py` (shared reference makes a⁺/decoy embedding-components identical).

---

## 6. Post-hoc reflection (REQUIRED)

The result JSON `REFLECTION` must state: (1) pass/fail of every preflight memorial and the §4
decision, per arm; (2) `S_A(full)` vs the A0 cross-check; (3) the `floor` and its A0-consistency
reading (is the difference within-window or attention-reached?); (4) that `S_A` is a mechanistic
intermediate, confounds (primary distance, secondary site-strength) reported, no identity claim;
(5) escaped failure modes — any effect that could not have failed, the `shared_ref_neutralises`
memorial value, and whether position/content separate or are entangled.

---

**Provenance.** Model checkpoint `SpliceBERT.1024nt/pytorch_model.bin` sha256 `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`. Results and manifests carry the same hash via `ExperimentManifest`; the reconstruction machinery and study code are committed in the `fix:`/`study:` changeset for this arc.
