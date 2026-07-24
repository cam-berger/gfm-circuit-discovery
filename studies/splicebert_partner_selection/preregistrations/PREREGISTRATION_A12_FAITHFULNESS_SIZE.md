# Pre-registration — A12: backward-pruned faithfulness–size curve (donor-graph experiment 16)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_a12_faithfulness_size.py` runs. Cam's redesign (deferred from
the A6 discussion — forward greedy is ill-identified at F≈0; **prune from the full model** instead).
Reuses A6's behaviour-level faithfulness scrub (`patch_heads`/`clamp_write`/`clamp_embed`). No new
kernel. Pre-run corrections logged inline.

---

## 0. Why this document exists

A6/C27: the 10-node circuit is **0.5% faithful** (`F(C)=0.005`) — faithfulness requires far more of the
network, but *how much* is unquantified. A12 measures the **faithfulness–size curve**:

> Under complement-scrub-to-corrupt faithfulness, how many nodes must be kept clean to recover
> 50/80/90/95% of the coupling — and is that curve gradual (irreducibly distributed) or does it have a
> knee (a mid-size faithful circuit)?

Behaviour-level (donor margin). Node granularity (embed + 96 heads + 6 MLPs = 103 nodes).

---

## 1. Metric and pruning (identical faithfulness to A6)

- **`F(K)`** = keep node-set `K` clean, scrub the complement **to corrupt** (`patch_heads`/`clamp_write`/
  `clamp_embed`); `F(K) = (M(K) − M_corr)/(M_cl − M_corr)`, ratio-of-means. `F(ALL)=1`, `F(∅)=0` by
  construction (normalization anchors — R21: the embed clamp forces these, so they are anchors, not
  memorials; the scrub machinery is validated in A6).
- **Backward greedy from the FULL model.** Start `K = ALL` (F=1). At each step, evaluate `F(K\{n})` for
  every remaining node `n`, **recompute**, and remove the node whose removal loses the **least** F
  (least-important-first). Continue until `F < 0.30` (covers the 50–95% region), then stop (the tail is
  not needed). This yields a pruning **order** and the `F`-vs-`|K|` curve.
- **Discovery / held-out.** Deterministic 50/50 gene split (seed 0). **Discover the pruning order on D;
  measure the reported `F`-vs-`|K|` curve on held-out H** using the D-order (no re-selection on H).

---

## 2. Gates and falsifiers

**Preflight** (`preflight_a12_faithfulness_size.json`, global PASS = AND):

- `full_is_clean` — `F(ALL)` on the split reproduces the clean margin (== 1 by construction; < 1e-4).
- `empty_is_corrupt` — `F(∅)` reproduces the corrupt margin (== 0; R21: embed-clamp-forced anchor).
- `positive_control_partner_coupling` (`coup ≥ 0.20`), `both_genes_split` (D,H disjoint, each ≥ 40),
  `both_strands_present`, `checkpoint_recorded`. (No non-embed on-path memorial exists here — F(K) IS
  the A6 scrub, validated there; A12 adds no new mechanic.)

**If preflight fails the run STOPS** (`--force` → DIAGNOSTIC). `--smoke` = preflight + a few pruning
steps on tiny-n (val sealed, no curve reported).

---

## 3. Decision (frozen)

Report, on **held-out H** (using the D-discovered order):
- **`F`-vs-`|K|` curve** with gene-bootstrap CIs at each reported size.
- **Node counts** `|K|` at which held-out `F_H` first reaches **{50, 80, 90, 95}%** (interpolated).
- **Shape:** a **knee** (small `|K|` jump to high F, then flat) ⇒ a mid-size faithful circuit exists; a
  **gradual** curve (F rises ~linearly with `|K|`) ⇒ faithfulness is irreducibly distributed.
- **Composition of the faithful core:** which node types (heads/MLPs/embed, which layers) survive to the
  90% point — cross-check against the A6–A10 circuit (does the delivery spine + generator sit in the
  faithful core?).

**Caveat (frozen):** greedy is not guaranteed optimal (a better set of the same size may exist);
report the greedy curve as an **upper bound on the minimal faithful set size** at each F level.

---

## 4. Outputs, reflection

- `scripts/splicebert/run_a12_faithfulness_size.py` (reuses A6's faithfulness scrub; no new kernel);
  `results/…/a12_faithfulness_size.json` + manifest + preflight.
- **REFLECTION:** the held-out `F`-vs-`|K|` curve + the 50/80/90/95% node counts; knee vs gradual; the
  faithful-core composition vs the A6–A10 circuit; discovery-vs-held-out agreement; escaped failure
  modes (greedy suboptimality upper-bound; R21 embed-anchor caveat; the F(K) scrub is A6's, validated).

**Scope.** Behaviour-level, node granularity, complement-scrub-to-corrupt. Quantifies the "how
distributed" question A6 opened. Greedy = upper bound. No prior claim redefined.

**Provenance.** Checkpoint sha256 `2ad91428…0149d`; `F(K)` faithfulness per A6/C27.
