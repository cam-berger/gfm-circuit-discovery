# Pre-registration — A10: head-level edges of the acceptor generator (donor-graph experiment 14)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_a10_generator_edges.py` runs. Reuses the A8 value-path metric
(address held clean, reconstruct-inject) + A6/A7 scrub. No new kernel. Pre-run corrections logged
inline as `**Amendment (YYYY-MM-DD, pre-run, no data seen).**`.

---

## 0. Why this document exists

A8/C29 established the generator's block structure (serial L0→L1→MLP2, MLP-backbone) with **block-level**
necessity and **D-only** per-head numbers. A10 resolves the **wiring at head resolution, held-out**,
interaction-aware (the L2-singleton artifact proved singletons mislead → group tests are trustworthy):

> Which L0 heads are the minimal load-bearing set (held-out); do they reach the payload **through** the
> MLPs (mediated) or **directly**; and is the MLP chain strictly serial (MLP1 needs MLP0, MLP2 needs
> MLP1)?

Behaviour-level, **value-path** (address held clean). No claim about node *semantics* (that is A9/C30).

---

## 1. Metric, scaffold, generator (identical to A8/C29)

- **Metric.** A8 value-path `M'(S)`: run the S-open scrub (scaffold clean; complement corrupt; `S`
  clean at the acceptor 4-token window ACC), capture `r3(S)`; reconstruct L3H1's donor output with the
  **CLEAN** attention row `A⁺` and only the S-driven value edit (`value_linearity` memorial); inject
  donor-only → `M'(S)`. `R(S)=(M'(S)−M'(∅))/(M'(A)−M'(∅))`; `R(∅)=0`, `R(A)=1`.
- **Scaffold `C`** (always clean): delivery spine + L0–L2 circuit nodes (embed, L1H12, L2H15, MLP2, L3H1,
  MLP3, MLP4, L4H0, L4H2, L5H11); L3H1 **address** additionally clean.
- **Generator `A`** (openable at ACC): `{46 non-circuit L0–L2 heads} + {MLP0, MLP1}` (as A8).
- **Discovery / held-out** 50/50 gene split (seed 0). Discover on D, confirm on H.

---

## 2. Conditions (edge-resolving; all node-sets `S ⊆ A` opened at ACC)

1. **L0-head resolution.** `N_h = R(A) − R(A\{h})` for each of the 16 **L0** heads, on **D and H**.
   Minimal L0 set `L0* = {h : held-out N_h CI-low > 0}`. Confirm `R(L0* ∪ A_{L1,L2,MLP})` on H.
2. **Top-head interactions.** For the top-2 L0 heads by `N_h`, pairwise
   `I = R(A) − R(A\{h1}) − R(A\{h2}) + R(A\{h1,h2})` — do they act jointly or independently?
3. **Head→MLP mediation.** `R(L0heads only)` (only the 16 L0 heads open; MLP0/MLP1 + L1/L2 heads
   corrupt) vs `R(L0heads + MLP0)`. If the L0-head contribution is small alone but large with MLP0 →
   the L0→payload path is **mediated by MLP0**; if large alone → a **direct** head→L3H1-value path.
4. **MLP serial chain.** `R(MLP0 only)`, `R(MLP0+MLP1)`, and the circuit-value-check `R(A) − R(A + scrub
   MLP2@ACC)` (`N_MLP2`, held-out). Edge `MLP0→MLP1`: `R(MLP1 only) − R(MLP1 | MLP0 also corrupt=default)`
   — does MLP1 contribute **without** MLP0 upstream? (Serial ⇒ near-zero alone.)

Necessities and interactions are **conditional causal effects**, NOT additive fractions (they do not
sum; a serial chain makes multiple links each necessary — the C29 caveat).

---

## 3. Gates and falsifiers

**Preflight** (`preflight_a10_generator_edges.json`, global PASS = AND): **`value_linearity` (< 1e-4) —
the REAL memorial** (validates the reconstruct-inject on the measurement path); `empty_circuit_is_corrupt`
/ `full_circuit_is_clean` (= 0.0) are **embed-clamp-dominated (R21) — embed-plumbing checks only, NOT
cover-completeness** (kept as sanity checks); `generator_ceiling` `M'(A)−M'(∅) ≥ 0.10`,
`positive_control_partner_coupling`, `both_genes_split`
(D,H disjoint, each ≥ 40), `both_strands_present`, `checkpoint_recorded`. `--smoke` = memorials only.

---

## 4. Decision (frozen)

- **Minimal L0 set found** iff `L0*` heads have **held-out** `N_h` CI-low > 0 and `R_H(L0* ∪ rest) ≥ 0.90
  · R_H(A)` (dropping non-`L0*` L0 heads loses little on held-out).
- **Mediated vs direct**: `R(L0heads only) < 0.5 · R(L0heads + MLP0)` ⇒ L0→payload is **MLP0-mediated**.
- **Serial chain**: `R(MLP1 only) ≈ 0` and `N_MLP0, N_MLP1, N_MLP2` all held-out CI-low > 0 ⇒ a strict
  L0→MLP0→MLP1→MLP2 serial pipeline; a large `R(MLP1 only)` ⇒ a shortcut.

Held-out disagreement with D on any edge → report as discovery-only (overfit), not confirmed.

---

## 5. Outputs, reflection

- `scripts/splicebert/run_a10_generator_edges.py` (reuses the A8 reconstruct-inject value-path metric +
  A6/A7 scrub; no new kernel); `results/…/a10_generator_edges.json` + manifest + preflight.
- **REFLECTION:** the held-out minimal L0 set + interactions; the mediation verdict; the MLP-chain
  serial structure with held-out necessity; discovery-vs-held-out agreement; escaped failure modes (the
  memorials, singleton-vs-group where relevant, gene-split disjointness, non-additivity).

**Scope.** Value-path (address held clean), acceptor-window generator wiring, held-out. Bounds the
generation feeding L3H1's channel (A8 ceiling). No prior claim redefined; semantics is C30.

**Provenance.** Checkpoint sha256 `2ad91428…0149d`; metric/generator per A8/C29.
