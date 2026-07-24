# Pre-registration — A6: circuit faithfulness, necessity, minimality (donor-graph experiment 10)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_a6_circuit_faithfulness.py` runs. Imported from
`gfmcd.circuits.hf_bert` (`patch_heads`, `matched_random_head_sets`) / `gfmcd.splicing.circuit`
(`clamp_write`, `capture_writes`, `mlm_health`, `ratio_of_means`, `gboot_ratio`) /
`gfmcd.splicing.provenance` (`clamp_embed`, `embed_module`). Pre-run corrections logged inline as
`**Amendment (YYYY-MM-DD, pre-run, no data seen).**`.

---

## 0. Why this document exists

The A0–A5 arc validated *paths* (path patching, QK/OV factorials, reconstruction) one hypothesis at
a time. It never tested the **whole graph**: that the proposed circuit is **faithful** (carries the
coupling when everything outside it is corrupted), **necessary** (the coupling dies when it is
corrupted), **minimal** (every node is load-bearing), and **complete** (nothing important is
omitted). This is the IOI-style faithfulness/completeness/minimality battery (Wang et al. 2022),
which our program has not yet run. A6 runs it at **node granularity** (edge-level discovery is the
next study). It answers, before any expensive upstream edge scan:

> Does the proposed donor circuit `C` actually hold the clean−corrupt coupling on its own, and how
> much (and which) unexplained effect is left outside it?

Behaviour-level (native donor margin) + health (masked-MLM). No new mechanistic claim about *what*
the nodes represent — only *whether the node set is the circuit*.

---

## 1. The circuit `C` (FROZEN node set)

From the live ledger, at (component) granularity. **Read/address arm:** embeddings (position
geometry, C21/C22) → **L1H12** (C6), **L2H15** (C6/C9/C25), **MLP2** (query assembly, C21) → **L3H1**
(the reader, C3–C25). **Delivery arm:** L3H1 → **MLP3, MLP4** (acceptor-dependent `U_info` assembly,
C17) → transporters **L4H0, L4H2, L5H11** (C18) → donor logit.

- `C_heads = {(1,12), (2,15), (3,1), (4,0), (4,2), (5,11)}` (6 heads)
- `C_mlps  = {(2,"mlp"), (3,"mlp"), (4,"mlp")}` (3 MLPs)
- `C_embed = {embed}` (whole embedding output; the position-vs-content split within it is C22's job)
- **10 nodes total.**

**Softest inclusions (their minimality result is informative, not assumed):** MLP2 (provenance-level
in C21, not yet behaviour-level) and L1H12 (C6 "contributes upstream"). **Deliberately excluded:**
MLP5 (writes `U_info` *acceptor-independently*, C17 → not coupling-carrying) — a completeness probe.

---

## 2. Method — node-level activation patching (post-LN ⇒ patch OUTPUTS, downstream recomputes)

- **Data.** Eligible coupling examples, val (chr11/16/19), both strands, seed 0. Corruption =
  destroy the partner AG (`task.pair`); clean/corrupt are token-aligned (substitution, same length).
- **Node cover.** Every residual writer is patchable: embedding output (`clamp_embed`), every
  attention head (`patch_heads`, per-head), every MLP write `layer.output.dense` (`clamp_write`,
  `(l,"mlp")`). A node is *set to a source run* by overwriting its **output** with that run's cached
  value; downstream layers recompute from the (mixed) residual — proper activation patching.
- **Base run = CLEAN.** All conditions noise **from clean**:
  - **Faithfulness** `F(C)`: patch the **complement** of `C` → **corrupt** (circuit stays clean).
    `F(C) = (M − M_corr)/(M_cl − M_corr)`, ratio-of-means over examples (`coup = M_cl − M_corr`).
  - **Necessity** `N(C)`: patch **`C`** → **corrupt** (complement stays clean).
    `N(C) = (M_cl − M)/(M_cl − M_corr)` — fraction of coupling destroyed.
  - **Minimality**: for each node `n∈C`, `F(C\{n})` (complement ∪ {n} → corrupt). Marginal
    `ΔF_n = F(C) − F(C\{n})`.
  - **Completeness (residual scan)**: for each node `m∉C` (all 90 non-circuit heads + 3 non-circuit
    MLPs), `F(C∪{m})` (keep `m` clean too). Gain `ΔF_m = F(C∪{m}) − F(C)`.
- **Endpoints (all three FROZEN per the review; A6 MEASURES behaviour + health).**
  1. **Behaviour** (primary): native donor margin coupling `M_cl − M_corr` (`task.margin`).
  2. **Health** (guard against the C7/R12 "killed the model, not the effect" confound).
     **Amendment (2026-07-16, pre-run, no data seen):** the guard is the **two exact arithmetic
     memorials**, not a masked-accuracy probe. Because every scrub sets nodes **to the corrupt run**
     (a healthy 2-nt-changed forward), and `empty_circuit_is_corrupt` / `full_circuit_is_clean`
     verify the two extremes reproduce the corrupt and clean runs **exactly (<1e-4)**, every partial
     scrub is a hybrid of two *verified-healthy* activation sets — it cannot be a damaged model in the
     C7/R12 sense. This is strictly stronger than `mlm_health` at the extremes and avoids running the
     probe under example-specific caches (which would misalign). `health_baseline` (clean `mlm_health`,
     no scrub) is still recorded as the competence ceiling. Residual caveat: a partial hybrid could be
     mildly off-distribution; noted, not gated.
  3. **Address / payload** (frozen, **deferred to the T2 edge scan**): L3H1 `S_A` and the
     acceptor-window value contribution — measured per-edge where "does this edge feed the address or
     the payload" is the question. A6 does not compute them (node-level behaviour + health suffice
     for the masquerade guard).
- **Size-matched null.** Random node sets matching `C`'s composition (6 heads via
  `matched_random_head_sets`, + 3 random MLPs), `--n-null` draws, seed 0 → the distribution of
  `F(random)` and, for completeness, the single-node-addition null.

---

## 3. Gates and falsifiers

**Preflight** (`preflight_a6_circuit_faithfulness.json`, global PASS = AND):

- `empty_circuit_is_corrupt` — **key memorial:** base=clean, patch **all** heads+MLPs+embed →
  corrupt reproduces `M_corr` **< 1e-4** (proves the node cover is complete — every residual writer
  is patched; a miss means an unpatched path and every ratio is wrong).
- `full_circuit_is_clean` — **key memorial:** base=corrupt, patch **all** nodes → clean reproduces
  `M_cl` **< 1e-4** (plumbing drives the output from the other direction).
- `positive_control_partner_coupling` — `coup ≥ 0.20`.
- `health_baseline` — clean `mlm_health ≥ 0.5` (records the ceiling for the guard).
- `both_strands_present`, `checkpoint_recorded`.

**If preflight fails the run STOPS** (`--force` → DIAGNOSTIC). `--smoke` = preflight/memorials only,
tiny-n, val sealed (no F/N/scan reported).

---

## 4. Decision (thresholds frozen BEFORE inspection)

- **Faithful** iff `F(C) ≥ 0.80` **and** `F(C)` exceeds the 95th percentile of the size-matched
  random-circuit null (add-one `p < 0.05`) **and** health under the `F(C)` scrub `≥ 0.90 ×`
  clean health. Report `F(C)` with gene-bootstrap CI regardless.
- **Necessary** iff `N(C) ≥ 0.80` and above the null.
- **Minimal** iff **every** `n∈C` has `ΔF_n` gene-bootstrap CI-low `> 0`. Any node whose CI includes
  0 is **not** load-bearing at this granularity → reported as prunable (not silently kept).
- **Complete** iff **no** `m∉C` has `ΔF_m` above the 95th percentile of the single-node-addition null.
  Any `m` above → `C` is **incomplete**; `m` is flagged as an omitted node and becomes a **T2**
  target. The residual `1 − F(C)` is reported and, if `>` the null band, localized by the top-`ΔF_m`.

Faithfulness is **not** additive across arms (A2b lesson): `F(C)` and `1 − N(complement)` are
reported separately, never summed.

---

## 5. Outputs, reflection

- `scripts/splicebert/run_a6_circuit_faithfulness.py` (reuses `patch_heads`, `clamp_write`,
  `clamp_embed`, `mlm_health`, `matched_random_head_sets`; **no new kernel**);
  `results/…/a6_circuit_faithfulness.json` + manifest + preflight.
- **REFLECTION (required):** (1) pass/fail every memorial and the §4 verdicts (faithful / necessary /
  minimal / complete); (2) `F(C)`, `N(C)` with CIs and vs the size-matched null; (3) the minimality
  table (per-node `ΔF_n`, which nodes are load-bearing vs prunable — especially MLP2 and L1H12); (4)
  the completeness residual `1 − F(C)` and the top-`ΔF_m` omitted nodes (the T2 shortlist); (5) health
  under the scrubs (was the faithfulness test confounded by model damage?); (6) escaped failure modes
  — the two arithmetic memorials, the health guard, non-additivity, whether the null is fairly
  size-matched (heads **and** MLPs).

---

**Scope.** NODE-level (not edge — that is T2). Whole-module embeddings (position-vs-content is C22).
Scrub-**to-corrupt** (health-preserving 2-nt contrast, *not* mean-ablation — sidesteps the C7/R12
general-damage confound). No claim about node *semantics*.

**Provenance.** Model checkpoint sha256
`2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`; circuit `C` per C6/C9/C16–C25.
