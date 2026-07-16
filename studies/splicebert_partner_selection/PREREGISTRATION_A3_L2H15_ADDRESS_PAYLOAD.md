# Pre-registration — A3: does L2H15 build L3H1's address or its payload? (donor-graph experiment 7)

**Status: FROZEN.** Written before `run_a3_l2h15_address_payload.py` runs. Imported from
`gfmcd.circuits.hf_bert` / `gfmcd.splicing.circuit` / `gfmcd.splicing.payload`. Pre-run
corrections logged inline as `**Amendment (YYYY-MM-DD, pre-run, no data seen).**`.

---

## 0. Why this document exists

`l2_accounting` (C9/D11): L2H15 supplies ~0.66 of the donor coupling, ~all of it mediated through
L3H1 (head output @ all-pos rescues 0.959). A1a/R20: L3H1's attention is a positional **address**;
A2 (C23): L3H1 binds that address to a value **payload**; A2b (C24): the payload is acceptor-local
but not the `U_info` subspace. The natural next attribution — replacing the moot MLP2→`U_info`
test:

> Does L2H15's coupling contribution flow through L3H1's **address** (the attention row `A`,
> query/key-determined) or its **payload** (the values `V = W_V r`), or their **interaction**?

Reaches **Behaviour** (native coupling). This localizes L2H15's role in the head, not a new node.

---

## 1. Design — the A2 reconstruct-inject, with L2H15-ablation as the perturbation

L3H1's donor output is `o_d = W_O Σ_j A_{dj} V_j`. We reconstruct `context_d = A_d · V` for
(address, payload) drawn from **L2H15-intact** vs **L2H15-ablated** runs, inject it as L3H1's donor
head output (`payload.inject_margin_donor`, head `(3,1)`, donor only) **into the L2H15-ablated
forward**, and read the native donor **coupling**.

- **Data:** eligible coupling examples, val (chr11/16/19), both strands, n ≥ 250, seed 0.
  Corruption = destroy the partner AG (`substitute(seq, next_acceptor_pos−2, "CC")`).
- **Ablation:** L2H15 = ADDRESSER `(2,15)`, mean-ablated (`ablate(ad, [ADDRESSER])`), active during
  every ablated capture **and** every injection forward.
- **Readout:** donor margin `m`; coupling `Y = m(clean) − m(corrupt-partner)`. `Y_intact` (L2H15
  present), `Y_ablated` (L2H15 mean-ablated). Rescue of a cell
  `R = Σ(Y_cell − Y_ablated) / Σ(Y_intact − Y_ablated)` (ratio-of-means, gene-grouped bootstrap).
- **Captures (per condition c ∈ {clean, corrupt}):** intact `A⁺_c = attentions[3][:,1,donor,:]`,
  `V⁺_c = capture_proj(value)` head-1; ablated `A⁻_c`, `V⁻_c` (same, inside the ablation);
  native margins `m_intact_c`, `m_ablated_c`. `A`/`V` are condition-specific (clean-A⁺ ≠ corrupt-A⁺).

---

## 2. The factorial

Reconstruct `context(A^a, V^p)_c = A^a_c · V^p_c` per condition; inject into the **L2H15-ablated**
run of that condition; `Y_cell = m_cell(clean) − m_cell(corrupt)`.

| cell | L3H1 address `A` | L3H1 payload `V` | tests |
|---|---|---|---|
| `baseline` | ablated `A⁻` | ablated `V⁻` | reproduces `Y_ablated` (memorial) |
| `address` | **intact `A⁺`** | ablated `V⁻` | L2H15's contribution is the **address** |
| `payload` | ablated `A⁻` | **intact `V⁺`** | L2H15's contribution is the **payload** |
| `joint` | **intact `A⁺`** | **intact `V⁺`** | L3H1's whole donor output restored (ceiling) |

- **Interaction** `I_{A×V} = R(joint) − R(address) − R(payload) + R(baseline)`.
- **Prediction (per the design):** address builder → `R(address)` rescues, `R(payload)` ≈ 0;
  payload builder → `R(payload)` rescues; only `R(joint)` → L2H15 helps **bind** address and payload.

---

## 3. Gates and falsifiers

**Preflight** (`preflight_a3_l2h15_address_payload.json`, global PASS = AND):

- `reconstruction_matches` — `A⁻·V⁻` reproduces the model's own **ablated** L3H1 head output at the
  donor < 1e-4 (the reconstruct-inject is faithful under ablation).
- `baseline_reproduces_ablated` — injecting `A⁻·V⁻` reproduces the native ablated coupling
  `Y_ablated` < 1e-3 (`baseline` R ≈ 0).
- `probability_invariants` — eager L3H1 rows sum to 1 ± 1e-3, not identically zero (both intact and
  ablated captures).
- `positive_control_L2H15_contributes` — `Y_intact − Y_ablated ≥ 0.20` (L2H15 supplies coupling;
  l2_accounting ≈ 0.66).
- `joint_rescues` — `R(joint) ≥ 0.30` (restoring L3H1's donor output recovers L2H15's contribution
  that flows through L3H1's donor channel; sanity vs l2_accounting head-output 0.959, which
  restored **all** positions — donor-only is a floor).
- `both_strands_present`, `checkpoint_recorded`.

**If preflight fails the run STOPS** (`--force` → DIAGNOSTIC). `--smoke` = preflight/memorials only.

---

## 4. Firewall, scope, decision

Reaches **Behaviour** (native coupling). This attributes L2H15's *coupling contribution* to L3H1's
address vs payload; it does not revisit R20 (attention is positional) or C24 (payload distributed).
**Decision:** `R(address) ≫ R(payload)` (CI-separated) → L2H15 builds the positional **address**
(consistent with A1a: L2H15 shapes L3H1's query/key geometry); `R(payload) ≫ R(address)` → L2H15
supplies the acceptor **payload**; comparable + large `I_{A×V}` → L2H15 helps **bind** them.
Whichever wins, `R(joint)` bounds how much of L2H15's contribution flows through L3H1's donor output
vs L3H1's output elsewhere / non-L3H1 routes.

**Next (gated on this):** if address → the absolute-vs-relative geometry test (what positional
signal L2H15 shapes); if payload → what acceptor-value directions L2H15 sets (the `U_source`
question, C24). MLP2 mediation re-enters only against whichever specific target survives.

---

## 5. Outputs, reflection

- `scripts/splicebert/run_a3_l2h15_address_payload.py` (reuses `payload.inject_margin_donor`,
  `circuit.ablate`; no new kernel); `results/splicebert_coupling/a3_l2h15_address_payload.json` +
  manifest + preflight.
- **REFLECTION (required):** (1) pass/fail every memorial and the §4 decision (address / payload /
  binding); (2) `R(address)`, `R(payload)`, `R(joint)`, `I_{A×V}` with CIs; (3) how much of L2H15's
  contribution flows through L3H1's **donor** output (`R(joint)`) vs the l2_accounting all-position
  0.959; (4) that this is L2H15-attribution, not a new claim about attention identity (R20) or
  payload dimensionality (C24); (5) escaped failure modes — memorial values, any cell that could not
  have failed, reconstruct-inject exactness under ablation.

---

**Provenance.** Model checkpoint sha256 `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`;
built on commits `6a5e80e`/`3a7d3a7` (A0–A2) and `de38143`/`067da83` (A2b).
