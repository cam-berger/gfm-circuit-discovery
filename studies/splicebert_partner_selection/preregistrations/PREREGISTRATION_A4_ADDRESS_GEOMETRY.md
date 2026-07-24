# Pre-registration — A4: is L2H15's address contribution absolute or relative position? (donor-graph experiment 8)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_a4_address_geometry.py` runs. Imported from
`gfmcd.splicing.provenance` (position-embedding clamp) / `address` / `circuit`. Pre-run corrections
logged inline as `**Amendment (YYYY-MM-DD, pre-run, no data seen).**`.

---

## 0. Why this document exists

A1a (C22): L3H1's targeting is **position-embedding geometry**, absolute-vs-relative left open.
A3 (C25): **L2H15 builds that address**. So the geometry question is about **L2H15's contribution**
specifically. Define L2H15's contribution to targeting as

> `G₁₅(d, a) = S_A^intact(d, a) − S_A^{L2H15-ablated}(d, a)`,

where `S_A(d, a)` is L3H1's donor-query **QK-logit preference** for the intron-mate acceptor over a
phase/distance-matched decoy, `S_A = ℓ(d→a⁺) − ℓ(d→a⁻)` (head-1, /√32) — the A1a targeting score.
`G₁₅` is L2H15's contribution to that preference. **Readout note (pre-run, from prior results):**
L2H15's contribution lives in the *preference* (C20: ~11% of `S_A`, ≈ +0.4 on the QK logit), **not**
the raw single-site logit `ℓ(d→a⁺)`, on which L2H15's effect is small and sign-ambiguous — so the
preference (not the raw logit) is the readout. The intervention is on **position embeddings only** —
never sequence padding or moving content (which would add boundary/token-context/mask confounds).

QK/selection level; `G₁₅` is a mechanistic intermediate; no behavioural claim.

---

## 1. Intervention and readout

- **PE shift.** SpliceBERT uses learned absolute position embeddings `PE`
  (`bert.embeddings.position_embeddings.weight`, `[P,512]`). To place the donor/acceptor at shifted
  coordinates, clamp their position-embedding outputs to `PE[pos+shift]`
  (`provenance.clamp_embed_component("position", …)`), **content unchanged**, then run the forward.
- **`S_A(config) = ℓ(d→a⁺) − ℓ(d→a⁻)`** = L3H1 donor-query QK-logit **preference** for the
  intron-mate acceptor `a⁺` over a phase-matched decoy `a⁻` (`address.phase_matched_decoy`,
  `qk_logit_to_window`, head-1, /√32) under the config's PE placement.
- **`G₁₅(config) = S_A^intact(config) − S_A^{ablated}(config)`**, the ablated arm with L2H15
  mean-ablated (`ablate(ad, [ADDRESSER])`) **composed with** the same PE clamp.
- **Data.** Eligible coupling examples **with a phase-matched decoy**, val, both strands, seed 0.
  **Filter:** keep examples where `{donor, a⁺ window, a⁻ window} + shift` stays in `[0, P−1]` for
  every shift in the grid — n ≥ 150. Clamped tracked positions: donor, a⁺ window (4), a⁻ window (4).

---

## 2. The two operations (per the design)

Grid Δ ∈ `{−96, −48, −24, 0, +24, +48, +96}` (nt).

**1. Common translation** — move donor and **both acceptor sites** (`a⁺` and `a⁻`) **together** by
Δ; every offset (`d→a⁺`, `d→a⁻`, `a⁺→a⁻`) **unchanged**:
`E_common(Δ) = G₁₅(d+Δ, a⁺+Δ, a⁻+Δ) − G₁₅(d, a⁺, a⁻)`.
- Near-invariance (`E_common ≈ 0`) ⇒ **relative** address (offset only, not absolute coordinates).
- Systematic change ⇒ dependence on **absolute** input position.

**2. Differential translation** — move the **partner `a⁺` window only** by Δ (donor and `a⁻`
fixed); the `d→a⁺` offset changes by Δ (the natural complement; the mid-run spec truncated at
operation 1, so this is the principled pairing and can be revised):
`E_diff(Δ) = G₁₅(d, a⁺+Δ, a⁻) − G₁₅(d, a⁺, a⁻)`.
- If the address is relative, `G₁₅` should **track the `d→a⁺` offset** ⇒ `E_diff` is non-flat.
- Read together: **relative** ⟺ `E_common ≈ 0` **and** `E_diff` offset-sensitive.

Per Δ and operation, gene-grouped bootstrap CI on `E(Δ)`; report `G₁₅(0)`, `max_Δ|E(Δ)|` and its
ratio to `|G₁₅(0)|`, and the OLS slope of `G₁₅` vs Δ for each operation.

**Descriptive secondary (forward-free):** raw L3H1 positional-QK kernel
`K(p,q) = (PE_p W_Q^{h1})·(PE_q W_K^{h1})/√32`; report its Toeplitz residual fraction (variance along
constant-offset diagonals ÷ total). Low ⇒ the raw PE kernel is offset-only — a crude proxy
(ignores layers 0–2 and L2H15), descriptive only; the causal `E_common`/`E_diff` are the result.

---

## 3. Gates and falsifiers

**Preflight** (`preflight_a4_address_geometry.json`, global PASS = AND):

- `shift0_is_native` — **key memorial:** `S_A(Δ=0)` (clamp to native `PE[pos]`) reproduces the
  unclamped native `ℓ` < 1e-4 (the clamp is a no-op at Δ=0).
- `l2h15_contributes` — `G₁₅(0) = S_A^intact − S_A^ablated ≠ 0` with `|mean G₁₅(0)| > 0` and its
  gene-bootstrap CI excluding 0 (L2H15 measurably shapes L3H1's donor→acceptor targeting; else the
  geometry of `G₁₅` is undefined).
- `qk_reconstruction` — `softmax(q·k/√32+mask)` reproduces `attn[3][:,1]` < 1e-3.
- `probability_invariants` — eager rows sum to 1 ± 1e-3, not identically zero.
- `grid_fits` — n_kept ≥ `--min-n` (default 150); `pe_table_size` (record `P`);
  `both_strands_present`; `checkpoint_recorded`.

**If preflight fails the run STOPS** (`--force` → DIAGNOSTIC). `--smoke` = preflight/memorials only.

---

## 4. Firewall, scope, next

QK/selection. Decides the geometry of the address **L2H15 builds** (C25) — relative vs absolute —
not content (R20) or payload (C24). **Next:** if relative → a finer offset/mod-3 sweep and the
biological reading (distance-based partner search); if absolute → what anchors it.

---

## 5. Outputs, reflection

- `scripts/splicebert/run_a4_address_geometry.py` (reuses `clamp_embed_component`,
  `qk_logit_to_window`, `ablate`; no new kernel); `results/…/a4_address_geometry.json` + manifest +
  preflight.
- **REFLECTION (required):** (1) pass/fail every memorial and the §2 decision (relative / absolute /
  mixed) from `E_common` and `E_diff`; (2) the two `E(Δ)` curves with CIs, `max|E|/|G₁₅(0)|`, slopes;
  (3) the raw-PE-kernel Toeplitz descriptive with its proxy caveat; (4) that this is L2H15's
  *address-contribution* geometry, not a content/payload claim; (5) escaped failure modes — the
  `shift0_is_native` memorial, `G₁₅(0)` sign/magnitude, edge effects from the inconsistent position
  field, whether any Δ could not have moved `G₁₅`.

---

**Provenance.** Model checkpoint sha256 `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`;
built on commits `6a5e80e`/`3a7d3a7`/`de38143`/`067da83`/`37bbeb6`.
