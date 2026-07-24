# Pre-registration — A8: backward-pruned exact path search through the acceptor-local generator (donor-graph experiment 12)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_a8_generator_path.py` runs. Design per the 2026-07-16 review
(backward pruning, scaffold-aware forward confirmation, value-path metric, discovery/confirmation gene
split). Reuses A6/A7 scrub (`patch_heads`/`clamp_write_mask`/`clamp_layer_input`) + A2b/A3
reconstruct-inject (`payload.value_weight_head`/`deliver`/`head_output`/`inject_margin_donor`). No new
kernel. Pre-run corrections logged inline as `**Amendment (YYYY-MM-DD, pre-run, no data seen).**`.

---

## 0. Why this document exists

A7/C28: the acceptor payload is built by a **serially interdependent L0→L1→L2 module** at the acceptor
window (ΔACC = 92% of ceiling; single layers near-silent). A8 finds the **minimal exact subgraph** of
that generator and its **serial structure**. Per the review, this is **not** a flat "48 writers ×
singleton" scan (singletons are silent → forward selection ranks noise). It is a **backward-pruned,
scaffold-aware exact path search**, scored **through L3H1's value path with the address held clean** so
payload-generation is not contaminated by address changes.

> Which acceptor-window L0–L2 writers, in which serial arrangement, are **necessary and jointly
> sufficient** to construct the payload L3H1 reads — and does the minimal graph generalize to held-out
> genes?

Behaviour-level (native donor margin, value-path). Localizes; does **not** claim node semantics.

---

## 1. Scaffold, generator, metric

- **Scaffold `C`** (always clean): the delivery spine + the L0–L2 circuit nodes —
  embed, L1H12, L2H15, MLP2, L3H1, MLP3, MLP4, L4H0, L4H2, L5H11. **L3H1's ATTENTION row `A⁺` is
  additionally held CLEAN** (reconstruct-inject); only its **value** path carries the tested set.
- **Generator `A`** (openable, at the **acceptor 4-token window** only; everything else corrupt):
  `g_L0 = {16 L0 heads}+MLP0`, `g_L1 = {15 L1 heads (¬H12)}+MLP1`, `g_L2 = {15 L2 heads (¬H15)}`
  (MLP2 is scaffold). `A = g_L0 ∪ g_L1 ∪ g_L2` (46 heads + MLP0 + MLP1).
- **Value-path metric.** For an opened subset `S ⊆ A`, run the S-open scrub (scaffold clean; complement
  corrupt; `S` clean at ACC), capture `r3(S)` = layer-3 input at ACC. With post-LN linearity
  `δV = W_V·δr` (`deliver`), `δr(S) = r3(S)_ACC − r3^corr_ACC`, reconstruct L3H1's donor output with the
  **clean** attention row: `o_d(S) = A⁺_d·V⁻ + Σ_{j∈ACC} A⁺_{dj} W_V δr(S)_j` (`head_output`), inject
  donor-only (`inject_margin_donor`) → margin `M'(S)`. **Normalized recovery**
  `R(S) = (M'(S) − M'(∅)) / (M'(A) − M'(∅))` (so `R(∅)=0`, `R(A)=1`). Address identical across all `S`
  by construction — this is a pure **payload-generation** scan.
  - **Amendment (2026-07-16, pre-run, audit, no data seen):** `M'(∅)` is **NOT** the corrupt margin —
    the scaffold (L1H12/L2H15/MLP2/embed) is clean at ACC even with no writers open, so `M'(∅)` is the
    **scaffold-only payload baseline**. `R(S)` therefore measures the **non-circuit writers' marginal
    payload above the scaffold** (the correct generator question). `M'(∅)` and `M'(A)` are reported
    **absolutely**; and `M'(A)` is compared to A7's **plain-forward** ΔACC to report the **value/address
    split** of the acceptor generation (what fraction is payload vs address).

---

## 2. Backward pruning + scaffold-aware confirmation (discovery split D, held-out split H)

Deterministic 50/50 gene split (even/odd unique-gene index, seed 0). **Discover on D, freeze, confirm
on H.** Coarse→fine (stop descending a branch once its group's necessity is null on D):

1. **Blocks.** `R(A)`, `R(A\g_Lℓ)` → **conditional necessity** `N_{gℓ} = R(A) − R(A\g_Lℓ)`.
2. **Serial confirmation.** `P(g_Lℓ) = R(g_Lℓ ∪ A_{>ℓ}) − R(A_{>ℓ})` (does an early writer matter when
   the **downstream relay** `A_{>ℓ}` is present?). Single blocks `R(g_Lℓ)` (value-frame single-layer).
   `N` vs `P` separates "necessary in the full pipeline" from "sufficient-with-downstream".
3. **Heads.** For each block with `N_{gℓ}` CI-low > 0 on D, split into heads (+its MLP): leave-one-out
   **`N_h = R(A) − R(A\{h})`** (**audit fix:** from the FULL generator, matching the block definition;
   the within-block form was uninformative under serial interdependence) → the load-bearing heads.
4. **Circuit-node value check (audit add).** Additionally scrub each **scaffold** L0–L2 node
   {MLP2, L1H12, L2H15} at ACC (move it from scaffold to openable): `N_circ = R(A) − R(A\{circ})` — does
   an address-builder (C25) also feed the **payload**? Reported, not part of `G*`'s minimality.
5. **Edges (continuation).** For surviving heads, resolve head→(L3H1 value residual) and head→head /
   head→MLP write paths. (v1 reports the head-level graph; the edge level is the follow-up.)

The **frozen graph `G*`** = scaffold ∪ {surviving groups/heads with D-necessity CI-low > 0}.

---

## 3. Gates and falsifiers

**Preflight** (`preflight_a8_generator_path.json`, global PASS = AND):

- `value_linearity` — **A2b memorial:** `V⁺ − V⁻ == W_V·(r3^cl − r3^co)` at ACC < 1e-4 (the
  reconstruct is analytic; else every `R(S)` is wrong).
- `generator_ceiling` (**audit fix**, replaces the wrong `empty_payload_is_corrupt`) — `M'(A) − M'(∅)
  ≥ 0.10`: the non-circuit writers' payload contribution above the **scaffold-only** baseline `M'(∅)` is
  measurable (else the value-path frame is underpowered → STOP). `M'(∅)` and `M'(A)` reported absolutely.
- `full_clean_value_restore` — `M'` with `δr` from the **fully clean** run reproduces the A2b/C24
  clean-acceptor-value restore, gated **∈ [0.30, 0.45]** (≈ +0.38; same computation as A2b by
  construction — anchors the value frame; a miss means the deliver/head_output path diverged).
- `scrub_cover` — the A6/A7 memorials `empty_circuit_is_corrupt` / `full_circuit_is_clean` = 0.0 (the
  `r3(S)` capture uses the same complete node cover).
- `positive_control_partner_coupling` (`coup ≥ 0.20`), `both_genes_split` (D,H disjoint, each ≥ 40),
  `both_strands_present`, `checkpoint_recorded`.

**If preflight fails the run STOPS** (`--force` → DIAGNOSTIC). `--smoke` = memorials/plumbing on tiny-n
(val sealed).

---

## 4. Decision (frozen)

Discover on D, **confirm on held-out H**:

- **Minimal generator found** iff the frozen `G*` recovers `R_H(G*) ≥ 0.80` of `R(A)` on held-out genes
  **and** every retained group's leave-out necessity `N_g` has H-bootstrap CI-low > 0. Report `G*`, its
  size, and the serial `N_g`/`P_g` profile.
- **Serial structure** classified from `N`/`P`: a strict pipeline shows each layer necessary
  (`N_{gℓ}>0` all ℓ) with the last layer `P(g_L2)=R(g_L2)` largest; a shortcut shows one layer carrying
  most of `R(A)` alone.
- **Not minimal / does not generalize** — if `R_H(G*) < 0.80` or retained groups' H-necessity includes
  0: the discovered graph overfits D; report and widen (finer groups, or the `ACC×REM` 7% tail).

Target is a fraction of **A7's acceptor-generation `R(A)`**, NOT model-level coupling (the delivery-side
distribution is a separate axis, C27/A6). Non-additivity respected: `N_g` and `P_g` reported separately.

---

## 5. Outputs, reflection

- `scripts/splicebert/run_a8_generator_path.py` (reuses A6/A7 scrub + A2b/A3 reconstruct-inject; no new
  kernel); `results/…/a8_generator_path.json` + manifest + preflight.
- **REFLECTION (required):** (1) pass/fail every memorial and the §4 verdict; (2) the block `N_g`/`P_g`
  profile (serial structure) with CIs; (3) the surviving head set per layer; (4) `G*`, its size, and
  held-out `R_H(G*)` vs the 0.80 target; (5) discovery-vs-confirmation agreement (did D-necessity
  replicate on H?); (6) escaped failure modes — the value-linearity + empty-payload + C24 memorials, the
  address-clean isolation (attention identical across S), gene-split disjointness, whether backward
  pruning avoided the forward-selection noise the review warned of, non-additivity.

---

**Scope.** Behaviour-level, value-path (address held clean), acceptor-window generation. Bounds the
**generation** feeding L3H1 (A7's `R(A)` ceiling), not model-level coupling. Groups are coordinate/layer
sets, not semantic variables (semantics = the final program step). No prior claim redefined.

**Provenance.** Model checkpoint sha256
`2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`; generator `A`, `R(A)` per A7/C28;
value linearity per A2b/C24.
