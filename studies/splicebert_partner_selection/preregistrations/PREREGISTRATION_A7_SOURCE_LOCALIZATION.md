# Pre-registration — A7: position×layer localization of the distributed generation (donor-graph experiment 11)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_a7_source_localization.py` runs. Design per the 2026-07-16
review. Imported from `gfmcd.circuits.hf_bert` (`patch_heads`) / `gfmcd.splicing.circuit`
(`clamp_write`, `build_windows`, `ratio_of_means`, `gboot_ratio`) / `gfmcd.splicing.provenance`
(`clamp_embed`). One small reusable primitive added: `clamp_layer_input` (a `forward_pre_hook`
residual patch at a layer boundary) for the state-clamp control. Pre-run corrections logged inline as
`**Amendment (YYYY-MM-DD, pre-run, no data seen).**`.

---

## 0. Why this document exists

A6/C27: the 10-node donor circuit `C` is **fully necessary** (`N=1.00`) but **non-faithful** under
complement-noising (`F(C)=0.005`); injecting its clean **outputs** recovers `D=0.986`. So the coupling
**generation** (building the acceptor payload L3H1 reads) is **distributed or interaction-dependent at
node resolution** — A6 excluded a sparse generator and single-node omissions, **not** a structured
jointly-necessary module. A7 asks **where** that generation lives and **how** it is organized:

> When only the delivery spine `C` is clean, can the network **locally reconstruct** the payload if we
> re-open its native writers in one **region × layer-block** at a time — and is the missing generation
> **localized**, **imported** from another region, or **interaction-dependent**?

Behaviour-level (donor margin). No new mechanistic claim about node *semantics*.

---

## 1. Two conditions (the review's separation)

Baseline = A6's `F(C)` (scrub every non-circuit node to **corrupt** at all positions; circuit clean).
All conditions run from a **clean** base; "keep clean at R" = do not patch there (base is clean).

1. **State clamp (upper-bound control).** On top of the `F(C)` scrub, force the **clean pre-L3H1
   residual** (`hidden_states[3]`, the input L3H1 reads) at the acceptor 4-token window
   (`clamp_layer_input(3, H_cl3, ACC)`). Hands the delivery spine the *finished* clean acceptor state.
   **Expected to recover strongly** (≈ C24/D) — it is the ceiling, **not** the question.
2. **Writers-open (the question).** On top of the `F(C)` scrub, leave the **non-circuit** L0–L2
   components in layer-block `B` **clean only at region `R`** (scrubbed to corrupt elsewhere); the
   delivery spine `C` stays clean, all other non-circuit nodes stay corrupt. Asks whether the network
   can **locally construct** the payload from `R`'s native writers. Implemented as a **token-masked**
   complement scrub (`patch_heads`/`clamp_write` with `token_mask`/`positions` = ¬R for block-`B`
   non-circuit nodes, all-positions for the rest).

**Writers** in a block = its heads + MLP **minus** circuit nodes: L0 = 16 heads + MLP0; L1 = 15 heads
(¬H12) + MLP1; L2 = 15 heads (¬H15) + (MLP2 is circuit). Only **L0–L2** writers are opened (they build
the pre-L3H1 payload); downstream non-circuit nodes stay corrupt (delivery is via the clean spine).

---

## 2. Regions, blocks, measurements

**Disjoint regions `R`** (per example, from `build_windows` + offsets):
- `ACC` = acceptor 4-token window (`a−3..a`; overlaps the corruption at `a−2`);
- `DON` = donor ±5;
- `PPT` = branch-point/polypyrimidine window `a−30..a−5` (upstream of `ACC`, disjoint);
- `REM` = all remaining valid (non-special, non-pad) positions.

**Layer-blocks `B`**: `L0-2` (headline) and `L0`, `L1`, `L2` (resolution).

**Measurements** (ratio-of-means vs `coup`, gene-bootstrap CIs):
- `F(C)` baseline (re-measured; must reproduce A6's ≈0.005).
- `F(C∪R,B)` for `R ∈ {ACC,DON,PPT,REM}` × `B ∈ {L0-2,L0,L1,L2}` → **marginal** `ΔR,B = F(C∪R,B) − F(C)`.
- **State clamp** `F_state(ACC)` (control).
- **Interaction** at `B=L0-2` for pairs `{ACC×PPT, ACC×DON, ACC×REM, PPT×DON}`:
  `I(R₁,R₂) = F(C∪R₁∪R₂) − F(C∪R₁) − F(C∪R₂) + F(C)` (needs the joint `F(C∪R₁∪R₂)`).
  `I>0` ⇒ super-additive (the regions **jointly** construct the payload); `I≈0` ⇒ separable.

---

## 3. Gates and falsifiers

**Preflight** (`preflight_a7_source_localization.json`, global PASS = AND):

- `empty_circuit_is_corrupt` / `full_circuit_is_clean` — the two A6 arithmetic memorials, **EXACT
  < 1e-4** (base scrub + node cover correct).
- `writers_open_full_is_clean` — **key mask memorial:** opening **all** L0–L2 non-circuit writers at
  **all** positions (`R`=full, `B`=L0-2) *and* the circuit reproduces the **clean** margin only if
  additionally the downstream is clean; the exact check is that a token-mask of **all-True** on the
  complement scrub reproduces `F(C)` (mask no-op) and a mask of **all-False** reproduces **clean**
  (< 1e-4) — validates the position-masking plumbing in both limits.
- `state_clamp_is_identity_at_full` — clamping the layer-3 input to its **own clean** value at all
  positions, with no scrub, reproduces the clean margin (< 1e-4) — the `clamp_layer_input` primitive
  is a faithful no-op.
- `positive_control_partner_coupling` (`coup ≥ 0.20`), `both_strands_present`, `checkpoint_recorded`.

**If preflight fails the run STOPS** (`--force` → DIAGNOSTIC). `--smoke` = memorials/plumbing on
tiny-n (val sealed).

---

## 4. Decision (frozen; the review's table)

Read `F_state(ACC)` (ceiling) against the writers-open `ΔR,B` and interactions:

| pattern | interpretation | → next |
|---|---|---|
| state high **and** `ΔACC` high | generation converges **locally** at the acceptor | edge scan into the acceptor residual |
| state high **and** `ΔACC` low | payload local when finished but **imports** info | trace incoming paths into acceptor writers (`ΔPPT/ΔDON/ΔREM`, interactions) |
| regions weak alone, **strong jointly** (`I≫0`) | **structured distributed** generation | region↔region path / group scan |
| no region recovers | broad across layers **or** wrong partition | hierarchical layer/component **group** tests |

"High/low" judged against `F(C)` (gene-bootstrap CI-low > 0 for a real ΔR,B) and against the state-clamp
ceiling as the fraction `ΔR,B / (F_state − F(C))`. No sparse-generation claim either way — A7 localizes;
it does **not** by itself establish irreducibility (that needs the backward-pruned size curve, deferred).

---

## 5. Outputs, reflection

- `scripts/splicebert/run_a7_source_localization.py` (reuses A6's scrub machinery + token masks;
  adds `clamp_layer_input`); `results/…/a7_source_localization.json` + manifest + preflight.
- **REFLECTION (required):** (1) pass/fail every memorial and the §4 read; (2) the `ΔR,B` grid
  (region × layer-block) with CIs, as a fraction of the state-clamp ceiling; (3) the interaction
  terms `I(R₁,R₂)` — localization vs interaction; (4) the state-clamp control value; (5) which §4 row
  fired and the resulting next-scan target; (6) escaped failure modes — the three plumbing memorials,
  whether `PPT`/`REM` are fairly defined (disjoint, non-empty), whether the ceiling `F_state` is a fair
  upper bound, non-additivity (ΔR are marginal, not summed).

---

**Scope.** Behaviour-level, NODE×POSITION granularity, complement-noising. Only L0–L2 writers opened
(payload generation is pre-L3H1). Regions are coordinate windows, not semantic variables (semantics =
step 4 of the program). No prior claim redefined; `U_info` untouched.

**Provenance.** Model checkpoint sha256
`2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`; circuit `C`, `F(C)` per A6/C27.
