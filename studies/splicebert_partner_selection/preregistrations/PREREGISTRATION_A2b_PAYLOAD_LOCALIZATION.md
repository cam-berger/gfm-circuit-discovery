# Pre-registration — A2b-(1): localize L3H1's value payload to U_info (donor-graph experiment 6)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_a2b_payload_localization.py` runs, from the clean
`donor-graph-address-payload` checkpoint (A0–A2 committed: `fix` 6a5e80e, `study` 3a7d3a7). Every
payload, control, statistic, and threshold is fixed here and imported from
`gfmcd.splicing.subspace` / `gfmcd.splicing.circuit` / `gfmcd.circuits.hf_bert`.

---

## 0. Why this document exists

A2 (C23): L3H1 binds a positional **address** to a partner-specific value **payload** — joint +0.37,
address-only/payload-only ~0. A2 left two things open (C23 scope): the payload's **meaning**, and
whether it is genuinely **distributed** (a single random rank-8 draw already recovered +0.18). A2b
asks:

> Is L3H1's partner **payload** the low-rank `U_info` object that C16/C17 route to the donor logit
> (0.93 of L3H1's delivery), or is any energy-matched rank-8 value direction equally good?

Reaches **Behaviour** (native margin recovery). `U_info` is the C16 causal variable, not a new
node. This does **not** revisit R20 (attention is positional); it localizes the *value* content.

## 0.1 The exact-linearity that makes this analytic

SpliceBERT is **post-LayerNorm**: the layer-3 value projection is `V = W_V · r + b_V`, **linear** in
the L3-input residual `r = hidden_states[3]` (no LayerNorm between the layer input and QKV). So for
any residual edit `δr` at the acceptor window, the value edit is exactly `δV = W_V δr`, and — with
the **clean attention row `A⁺` held fixed** — L3H1's donor head output is
`o_d(payload) = A⁺_d · V⁻ + Σ_{j∈acc} A⁺_{dj} · W_V δr_j`, computed analytically and injected
(`patch_heads`, head `(3,1)`, donor only) with **no re-run of a patched forward**. The attention row
is therefore *numerically identical* across all payloads by construction. The linearity is a
preflight memorial (`V⁺−V⁻ == W_V Δr`, §3).

---

## 1. Data, readout, U_info

- **Data:** eligible coupling examples, val (chr11/16/19), both strands, n ≥ 250, seed 0.
  Corruption = destroy the partner AG (`substitute(seq, next_acceptor_pos−2, "CC")`).
- **Readout:** donor margin; `Y⁺=margin(clean)`, `Y⁻=margin(corrupt)`, coupling `=Y⁺−Y⁻`;
  recovery `R = Σ(m_cell−Y⁻)/Σ(Y⁺−Y⁻)` (ratio-of-means, gene-grouped bootstrap, seed 0).
- **U_info:** `U = top_k_subspace(H_cl − H_co, 8)` over final residuals `H = hidden_states[-1]` at
  the donor (the C16 definition); projector `P_U(x) = (x U^⊤) U`. Acceptor window `(−3,−2,−1,0)`
  around `next_acceptor_pos`; `Δr_j = r⁺_j − r⁻_j` at that window (`r = hidden_states[3]`).

---

## 2. Payloads (clean address `A⁺` fixed; inject into the corrupt run) and controls

For an acceptor-window residual edit giving `δr_j`, `o_d = base + Σ_{j∈acc} A⁺_{dj} W_V δr_j`,
`base = A⁺_d · V⁻`.

| payload | `δr_j` at acceptor | tests |
|---|---|---|
| `address_floor` | `0` | base `A⁺·V⁻` — cross-checks A2 `address_only` (~+0.01) |
| `full` | `Δr_j` | acceptor-window payload **ceiling** (restores clean values at the window only) |
| `U` | `P_U Δr_j` | **sufficiency** of `U_info` |
| `perp` | `(I−P_U) Δr_j` | **complement sufficiency**; `= r^{\U}` (see below) |
| `random[i]`, i=1..R | `s_i · P_i Δr_j` | **energy-matched random rank-8 distribution** |
| `nonsite` | `P_U(r⁺−r⁻)` at a distance-matched **control** window, delivered via `A⁺` there | acceptor-**specificity** |
| `crossgene` | `P_U Δr_j` from a **rolled (mismatched) example** | payload must be example-specific |

- **Identity:** `r^{\setminus U} = r⁺ − P_U Δr = r⁻ + (I−P_U)Δr = r^{⊥}`, so complement-sufficiency
  and the necessity leg are the **same** measurement. **Necessity** `N_U = R(full) − R(perp)`.
- **Energy match (the crucial control, per review).** `W_V` is anisotropic, so residual-space
  energy is not enough. Each random rank-8 `P_i` is scaled **per example** so its **delivered value
  energy** matches `U_info`'s: `s_i(b) = ‖W_V P_U Δr_{b,acc}‖_F / ‖W_V P_i Δr_{b,acc}‖_F`. Then
  `R(U)` is compared to the distribution `{R(random[i])}`.
- **Both directions.** Denoising (above, into corrupt). **Noising** (into clean): `o_d = A⁺·V⁺ −
  Σ_{acc} A⁺_{dj} W_V (P_U Δr_j)` (remove `U_info` from the clean value), margin **loss** vs clean;
  `N_U^{noise} = (Y⁺ − m) / (Y⁺ − Y⁻)`.

**Statistics:** `Specificity_U = R(U) − E[R(random)]`; `U_info`'s **percentile** in `{R(random[i])}`
and add-one permutation `p = (1 + #{R(random) ≥ R(U)}) / (R+1)`. Gene-bootstrap CIs on every `R`.

---

## 3. Preflight memorials (global PASS = AND; run STOPS on fail, `--smoke` = these only)

- `value_linearity` — **the load-bearing memorial:** `V⁺_acc − V⁻_acc == W_V^{h1}(r⁺−r⁻)_acc`
  < 1e-4 (licenses the whole analytic reconstruction; fails if there is any LN before QKV).
- `inject_identity` — injecting `A⁻·V⁻` (the model's own corrupt head output) reproduces `m_co`
  < 1e-3 (the donor-only head patch is faithful).
- `attention_row_fixed` — every payload's injected head output uses the same `A⁺` (structural;
  asserted by construction — the reconstruction never re-runs a patched forward).
- `projector_idempotent` — `‖P_U² − P_U‖ < 1e-5`.
- `projector_orthogonal` — `‖P_U(I−P_U)‖ < 1e-5`.
- `decomposition` — `‖P_U Δr + (I−P_U)Δr − Δr‖ < 1e-5`.
- `probability_invariants` — eager L3H1 attention rows sum to 1 ± 1e-3, not identically zero.
- `positive_control_partner_coupling` — coupling ≥ 0.20.
- `both_strands_present`, `checkpoint_recorded`.

> **Amendment (2026-07-16, pre-run, caught in smoke, no recovery seen).** The draft listed
> `full_reconstructs_clean` (`o_d(full) == A⁺·V⁺`) and `full_matches_A2_joint`. Both are **wrong**:
> `full` restores clean values at the **acceptor window only** (corrupt elsewhere), whereas
> `A⁺·V⁺` / A2's joint restore **all** positions — and the corruption perturbs non-acceptor values
> that L3H1 also reads (empirically `o_d(full)` differs from `A⁺·V⁺` by ~0.11). `full` is the
> acceptor-window payload **ceiling**, not the clean value. Dropped both; `value_linearity`
> (exact, 1.9e-6) + `inject_identity` (exact) are the correct validators, and `address_floor`
> cross-checks A2's `address_only` (~+0.01) in the results, not the preflight.

---

## 4. Decision rule — call U_info the acceptor-side payload iff ALL:

1. **Sufficient above random:** `R(U)` above the energy-matched random rank-8 distribution
   (add-one `p` ≤ 0.05, high percentile).
2. **Necessary:** `N_U = R(full) − R(perp) > 0` with gene-bootstrap CI lower bound > 0.
3. **Complement does not explain it:** `R(perp)` materially below `R(full)`.
4. **Acceptor-specific:** `nonsite` recovers ≈ 0.
5. **Example-specific:** `crossgene` recovers ≈ 0; and it generalizes across held-out genes
   (gene-bootstrap CIs).

**Outcomes (per the design):**
- Sufficient **and** necessary **and** specific → closes address → `U_info` payload → behaviour.
- Sufficient but **not** above random → partner info is broadly distributed; `U_info` is a
  convenient readout, not a privileged mechanism (A2's "distributed" upgraded from provisional).
- Necessary but not sufficient → `U_info` participates in a distributed conjunction.
- Neither → C17 identified the *delivered donor-side* subspace, not the *acceptor-side source*
  representation.

Only after this does the **MLP2-construction** mediation (A2b-(2)) run — it is sharply
interpretable only if there is first a specific payload whose construction MLP2 can be tested
against.

---

## 5. Outputs, reflection

- `scripts/splicebert/run_a2b_payload_localization.py`; kernel `src/gfmcd/splicing/payload.py`
  (new — separate changeset, so A2b machinery does not blur A0–A2 provenance);
  `results/splicebert_coupling/a2b_payload_localization.json` + manifest + preflight;
  memorial tests in `tests/test_payload_memorials.py`.
- **REFLECTION (required):** (1) pass/fail every memorial and the §4 decision (which outcome);
  (2) `R(U)`, its percentile/`p` in the random distribution, `Specificity_U`; (3) `N_U` (both
  directions), `R(perp)`; (4) `nonsite`/`crossgene` specificity; (5) escaped failure modes —
  the linearity memorial value, energy-match adequacy, whether any control could not have failed.

---

**Provenance.** Model checkpoint sha256 `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`;
built on A0–A2 commits `6a5e80e` / `3a7d3a7`.
