# Pre-registration — A5: independent replication of donor-side U_info (donor-graph experiment 9)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_a5_uinfo_replication.py` runs. Imported from
`gfmcd.splicing.subspace` / `circuit` / `unembed`. Pre-run corrections logged inline as
`**Amendment (YYYY-MM-DD, pre-run, no data seen).**`.

---

## 0. Why this document exists

C16–C18 established `U_info` — the top-8 SVD subspace of the clean−corrupt **final donor residual**
`H_cl − H_co` — as the low-rank object carrying the coupling to the donor logit (0.92 recovery,
top-1 0.74). A2b (C24) showed the same 8-D basis is **not** privileged at the *acceptor source*.
Before building further on `U_info`, **validate it — do not redefine it**:

> Is `U_info` a **stable model representation** (an independently-derived version aligns and
> recovers on held-out genes, with a sensible rank curve) or a **sample-specific readout**?

Reaches **Behaviour** (native margin recovery). **This uses the exact C16 definition** (top-k SVD
of `H_cl − H_co` at the donor); it changes nothing about `U_info`. Validation only.

---

## 1. Data, definition, readout — all imported

- **Data.** Eligible coupling examples, val (chr11/16/19), both strands, seed 0. Corruption =
  destroy the partner AG (`task.pair`).
- **`U_info(rows, K) = top_k_subspace(rows, K)`** on `rows = H_cl − H_co`, `H = hidden_states[-1]`
  at the donor (the C16 definition, unchanged).
- **Readout.** `margins_fn(H) = margin_hard(ad.model.cls(H), task.bid)` — the MLM head on the final
  residual → donor margin. `coup = m_cl − m_co`. Recovery/necessity ratios are ratio-of-means with
  gene-grouped bootstrap.
- **Gene split.** Deterministic 50/50 by gene (sorted unique genes, even/odd index; seed 0) so **no
  gene appears in both** halves A and B — the independence the replication requires.

---

## 2. Measurements (rank grid `K ∈ {1,2,4,8,16,32}`)

Derive `U_A(K)` on split A's `(H_cl−H_co)`, `U_B(K)` on split B's. All causal tests use
`patch_component` (replace a residual's span-`U` component) through `margins_fn`.

1. **Subspace stability.** `overlap(K) = subspace_overlap(U_A(K), U_B(K))` (mean squared cosine of
   principal angles) vs the **random null** `E[subspace_overlap(U_A(K), random_subspace(D,K))]`
   (seed 0, `--n-null` draws). High and ≫ null ⇒ stable across independent gene splits.
2. **Held-out sufficiency.** Derive on A, test on the **held-out** split B:
   `R_suff(U_A→B, K) = Σ(margins_fn(patch_component(H_co^B, H_cl^B, U_A(K))) − m_co^B) / Σ coup^B`.
   Compare to **in-sample** `R_suff(U_B→B, K)` and to **random** `R_suff(random, K)`. Symmetric
   (also B→A); report the mean of the two held-out directions.
3. **Held-out necessity.** `N(U_A→B, K) = Σ(m_cl^B − margins_fn(patch_component(H_cl^B, H_co^B,
   U_A(K)))) / Σ coup^B` — remove the clean `U_info` component on held-out genes.
4. **Rank curve.** `R_suff(U_A→B, K)` and `overlap(K)` across the grid — does recovery saturate near
   rank 8, or keep rising (higher intrinsic rank) / saturate earlier (lower)? Rank 8 is **tested**,
   not assumed.

---

## 3. Gates and falsifiers

**Preflight** (`preflight_a5_uinfo_replication.json`, global PASS = AND):

- `margins_fn_reproduces_cached` — **key memorial:** `margins_fn(H_cl)` reproduces the cached clean
  margins `m_cl` < 1e-2 (the MLM head on the captured final residual is the true readout; else every
  recovery is wrong — the `donor_forward` `mf_check`).
- `positive_control_partner_coupling` — `coup ≥ 0.20`.
- `insample_reproduces_C16` — in-sample `R_suff(U_B→B, 8)` reproduces C16's ~0.9 (reported; a large
  miss means the pipeline differs from the anchor).
- `both_genes_split` — A and B non-empty, **disjoint** gene sets, each ≥ `--min-half` (default 40).
- `both_strands_present`, `checkpoint_recorded`.

**If preflight fails the run STOPS** (`--force` → DIAGNOSTIC). `--smoke` = preflight/memorials only.

---

## 4. Decision

- **Stable representation** — `overlap(8) ≫ random`, `R_suff(U_A→B,8) ≈ R_suff(U_B→B,8) ≫ random`,
  necessity replicates, and the rank curve saturates in the low ranks (≈8 or below): `U_info` is a
  real, generalizing model representation; C16–C18 are strengthened.
- **Sample-specific readout** — `overlap(8)` near the random null, or `R_suff(U_A→B,8) ≈ random`:
  `U_info` is a convenient in-sample readout, not a stable subspace; C16–C18's *interpretation*
  (not the empirical recovery) is weakened and flagged.
- **Wrong rank** — if recovery keeps climbing past 8, the intrinsic dimensionality is higher than 8;
  report the curve (still no redefinition — a note for future work).

This does **not** touch R20/C24/C25. `U_info`'s definition is unchanged regardless of outcome.

---

## 5. Outputs, reflection

- `scripts/splicebert/run_a5_uinfo_replication.py` (reuses `subspace`, `task.pair`, `margins_fn`;
  no new kernel); `results/…/a5_uinfo_replication.json` + manifest + preflight.
- **REFLECTION (required):** (1) pass/fail every memorial and the §4 decision; (2) `overlap(K)` vs
  random, `R_suff` held-out vs in-sample vs random, necessity, across the rank curve; (3) whether
  rank 8 is supported by the curve; (4) that `U_info` is **not** redefined — this is validation of
  the C16 object; (5) escaped failure modes — the `margins_fn` memorial, gene-split disjointness,
  whether the random null is a fair baseline for subspace overlap.

---

**Provenance.** Model checkpoint sha256 `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`;
donor-side `U_info` per C16/`donor_anchor.json`, C17/`donor_forward.json` — unchanged.
