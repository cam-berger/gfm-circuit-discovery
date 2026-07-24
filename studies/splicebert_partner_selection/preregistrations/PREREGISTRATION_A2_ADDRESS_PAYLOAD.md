# Pre-registration — A2: address–payload factorization of L3H1 (donor-graph experiment 5)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_a2_address_payload.py` runs. Every cell, readout, and
threshold is fixed here and imported from `gfmcd.circuits.hf_bert` / `gfmcd.splicing.circuit`.
Pre-run corrections logged inline as `**Amendment (YYYY-MM-DD, pre-run, no data seen).**`.

---

## 0. Why this document exists

A1a (C22) + R20: L3H1's donor-query attention **targeting** difference is dominated by
**position-embedding geometry**, not content/identity — yet the circuit couples **partner-
specifically** at behaviour (C19, ~37×). The reconciliation hypothesis:

> QK attention is a largely **positional address** (*where* L3H1 reads); the OV/value stream
> carries the partner-specific **payload** (*what* it retrieves). Attention need not itself encode
> partner identity for the complete head to produce partner-specific behaviour.

A2 tests this by intervening **independently** on the attention address `A` and the value payload
`V` of L3H1's donor output and reading the **behavioural** donor-margin recovery. It reaches the
**Behaviour** level (a native margin, no probe). The decisive subtest forces the **same attention
row** and varies only the values.

---

## 1. Decomposition, data, readout

L3H1's donor output is `o_d = W_O · Σ_j A_{dj} V_j`, `A_{dj}` the post-softmax donor attention
row (the **address**), `V_j = W_V r_j` the value vectors (the **payload**). We reconstruct
`context_d = Σ_j A_{dj} V_j` for any (address, payload) combination and **inject it as L3H1's
donor head output** (`patch_heads`, head `(3,1)`, `token_mask` = donor only), then run the rest of
the network and read the native margin. This isolates L3H1's donor channel exactly (C17's
meet-in-the-middle) and needs **no new intervention primitive**.

- **Data:** eligible coupling examples, val (chr11/16/19), both strands (R11), n ≥ 250, seed 0.
  Corruption = destroy the partner AG (`substitute(seq, next_acceptor_pos−2, "CC")`) — the
  standard donor coupling (C9/l2_accounting), **not** the a⁺/decoy address pairs.
- **Readout:** donor margin `margin = logit(G) − max(logit A,C,T)`; `Y⁺ = margin(clean)`,
  `Y⁻ = margin(corrupt)`; coupling `= Y⁺ − Y⁻`. Recovery of a cell
  `R = Σ(m_cell − Y⁻) / Σ(Y⁺ − Y⁻)` (ratio-of-means, gene-grouped bootstrap, unconditional).
- **Capture (eager, attentions read):** `A⁺,A⁻` = `attentions[3][:,1, donor, :]` (clean/corrupt);
  `V⁺,V⁻` = `capture_proj(value)` head-1 slice; `head_out⁺/⁻` = `capture_head_outputs` at
  `(3,1,donor)` (the model's own `Σ A V`, for the memorial); `m_cl,m_co` margins.

---

## 2. The factorial and the decisive subtest

Reconstruct `context(A^a, V^p) = A^a_{donor} · V^p` (`einsum('bs,bsd->bd')`); inject at L3H1 donor;
run the **corrupt** forward downstream; read the margin.

| cell | address | payload | expectation |
|---|---|---|---|
| `baseline` | A⁻ | V⁻ | R ≈ 0 (reproduces corrupt — memorial) |
| `address_only` | **A⁺** | V⁻ | positional routing alone |
| `payload_only` | A⁻ | **V⁺** | partner value alone (corrupt routing) |
| `joint` | **A⁺** | **V⁺** | ≈ L3H1 full restore (C17 ≈ 0.83 — memorial) |
| `payload_random` | A⁺ | V⁻ + P_rand(V⁺−V⁻) | energy-matched random 8-dim value subspace (control) |

- **Interaction** `I_{A×V} = R(A⁺,V⁺) − R(A⁺,V⁻) − R(A⁻,V⁺) + R(A⁻,V⁻)`.
- **Decisive same-row subtest:** with the **address fixed**, the value effect
  `R(A⁺,V⁺) − R(A⁺,V⁻)` and `R(A⁻,V⁺) − R(A⁻,V⁻)`. If the partner-coupling recovers when only the
  **values** differ under a **fixed** attention row, attention is the positional router and the
  values carry the discriminating payload — the reconciliation.

**Interpretation grid** (per the design):
- `address_only` alone recovers → positional targeting itself drives much of the coupling.
- `payload_only` (or same-row value effect) recovers → partner info lives in the value payload.
- only `joint` recovers → L3H1 **binds** address and payload; both jointly necessary.
- `payload_random` recovers as much as `joint` → the payload is not specific (a generic value
  direction suffices) — a control that can fail.

---

## 3. Gates and falsifiers

**Preflight** (`preflight_a2_address_payload.json`, global `PASS` = AND):

- `reconstruction_matches` — **key memorial:** the reconstruction `A⁺_donor · V⁺` reproduces the
  model's own captured head output `head_out⁺[(3,1),donor]` < 1e-4 (licenses reconstruct-inject).
- `inject_identity` — injecting the corrupt head output `(A⁻,V⁻)` reproduces the corrupt margin
  `m_co` < 1e-3 (the donor-only head patch is faithful; `baseline` R ≈ 0).
- `probability_invariants` — eager attention rows sum to 1 ± 1e-3, L3H1 not identically zero.
- `positive_control_partner_coupling` — clean−corrupt coupling ≥ 0.20 (the effect exists).
- `joint_recovers` — the `joint` cell recovers ≥ 0.30 (the reconstruct-inject reaches L3H1's
  known restore; sanity vs C17 ≈ 0.83, a partial-mediator floor).
- `both_strands_present`, `checkpoint_recorded`.

**If preflight fails the run STOPS** (`--force` → DIAGNOSTIC). `--smoke` runs preflight/memorials
only, no recovery magnitudes (val sealed).

---

## 4. Firewall and scope

Reaches **Behaviour** (native margin recovery). `A`/`V` are model activations; the reconstruction
is exact (memorial). This tests the **address–payload split of L3H1's coupling contribution**, not
partner identity in attention (that is R20-retracted) — it asks whether the *value* carries the
partner information that attention routes positionally.

**Deferred, frozen here as the follow-ups (A2b), NOT in this run:**
- **U_info-subspace payload.** Whether the partner payload is the C16/C17 `U_info` object:
  `r_j^U = r_j^- + P_U(r_j^+−r_j^-)` in the 4-token acceptor window (`P_U = top_k_subspace(H_cl−H_co,8)`),
  recompute L3H1 values + downstream; energy-matched random rank-8 residual subspace + non-site
  window controls. (C17 already routes 0.93 of L3H1's delivery through `U_info`; this localizes the
  payload to it.)
- **MLP2 construction of the payload.** If `U_info`/value is sufficient: {patch L2H15 clean, MLP2
  recompute} vs {L2H15 clean, MLP2 clamped corrupt} vs {MLP2 clean, no L2H15} vs {both}, mediation
  `I_{L2H15→MLP2}`, run in both intervention directions (MLP2 nonlinear).

---

## 5. Outputs, reflection

- `scripts/splicebert/run_a2_address_payload.py`; `results/splicebert_coupling/a2_address_payload.json`
  + `<stem>.manifest.json` (`CONFIRMATORY` iff `preflight_PASS and n ≥ 250`) + `preflight_a2_address_payload.json`.
- **REFLECTION (required):** (1) pass/fail every preflight memorial and the §2 interpretation grid;
  (2) the same-row value effect and `I_{A×V}` — is it address, payload, or binding; (3) whether
  `payload_random` recovers (specificity of the payload); (4) that this is the coupling-contribution
  split, not attention identity (R20 stands); (5) escaped failure modes — memorial values, any cell
  that could not have failed, the reconstruct-inject exactness.

---

**Provenance.** Model checkpoint `SpliceBERT.1024nt/pytorch_model.bin` sha256 `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`. Results and manifests carry the same hash via `ExperimentManifest`; the reconstruction machinery and study code are committed in the `fix:`/`study:` changeset for this arc.
