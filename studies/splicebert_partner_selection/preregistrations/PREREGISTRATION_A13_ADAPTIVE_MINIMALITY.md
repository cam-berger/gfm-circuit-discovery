# Pre-registration — A13: adaptive node-level minimality (donor-graph experiment 17)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: DRAFT.** Freeze before `run_a13_adaptive_minimality.py` runs.

A13 is **node-level and donor-side**: it tests whether C33's no-knee result survives the adaptive
recompute A12 preregistered but did not execute. It runs locally (~1.8 h) on the A6 scrub and needs no
new kernel. Its former edge-level sibling `A14` was **superseded before freezing** by
[`R4`](../../splicebert_acceptor_coupling/preregistrations/PREREGISTRATION_R4_EDGE_LEVEL_CAUSAL_CORE.md),
which asks the edge question as a paired estimand over both reading directions; the shared engineering
record (kernel, τ floor, GPU policy) moved with it to
[`DECISIONS_R4_EDGE.md`](../../splicebert_acceptor_coupling/DECISIONS_R4_EDGE.md).

---

## 0. Why this document exists

C33 reports that faithfulness has **no sparse knee** — about 72 of 103 nodes are needed for 90%
recovery — and its own caveat says the estimator is a *static* ranking, an upper bound on the minimal
faithful set. Two independent records say that upper bound has never been tightened:

- `docs/PREREG_CONFORMANCE.md:213` — A12's frozen protocol specified iterative backward greedy
  ("at each step ... recompute"); the delivered runner is **a single-pass leave-one-out rank from the
  full model**, and no record anywhere states the departure.
- `docs/ROADMAP.md`, open problem 2 — "adaptive minimality remains open."

A13 executes the search A12 preregistered and did not run, at the same node granularity, with the same
scrub. It asks one question:

> Is C33's no-knee result a property of the computation, or an artifact of the static ranking?

Behaviour-level (donor margin). Node granularity (embed + 96 heads + 6 MLPs = 103 nodes). **No new
kernel** — `patch_heads` / `clamp_write` / `clamp_embed` exactly as A6 and A12 use them.

A13 is deliberately the cheap arm. It runs locally (~1.8 h at n=64) and it must complete and pass its
calibration gates **before** the A14 edge kernel is built, because A13 validates the metric, the splits
and the two calibration constructions that A14 reuses.

---

## 1. Metric and pruning

- **`F(K)`** = keep node-set `K` clean, scrub the complement **to corrupt**;
  `F(K) = (M(K) − M_corr)/(M_cl − M_corr)`, ratio-of-means. Identical to A6/A12 — same function, same
  corruption operator (`CORRUPT_ALT = "CC"` at the intron-mate acceptor AG), same eligibility filter,
  `min_effect=None`.
- **Adaptive backward greedy, with recompute.** This is the difference from A12:

  ```
  K ← ALL                                  (|K| = 103, F = 1)
  while |K| > 1:
      for each n ∈ K:  evaluate F(K \ {n})       # on D, every step
      remove argmin_n [ F(K) − F(K \ {n}) ]      # least-important-first
      record (|K|, F(K))
  ```

  Σ|K| ≈ 5,356 subgraph evaluations. A12's estimator evaluated 103 and never recomputed.
- **Discovery / held-out.** Deterministic 50/50 gene split, seed 0 — **the same split A12 used**, so the
  two curves are comparable without a re-derivation. Discover the removal order on `D`; report the
  `F`-vs-`|K|` curve on held-out `H` using the D-order. No re-selection on H.
- **Split.** `val` only. The sealed chr8/9 test is **not opened** by A13.

---

## 2. Gates and falsifiers

**Forced anchors — recorded, but they are NOT memorials** (R21/R7 discipline; the embed clamp forces
both, exactly as A12's record states):

- `full_is_clean` — `F(ALL) = 1` (< 1e-4).
- `empty_is_corrupt` — `F(∅) = 0` (< 1e-4).
- `curve_monotone_in_removal_order` — true by construction of the greedy. Logged, not gating.

**Memorials — each can fail and each is on the measurement path:**

| Memorial | Construction | Gate |
|---|---|---|
| `planted_circuit_recovery` | Pre-clamp all nodes outside a planted set `P` (\|P\| = 8, drawn to span layers 0–5 and include one MLP) to corrupt. That model's ground-truth faithful set **is** `P`. Run the identical greedy on it. | TPR over `P` = 1.00 **and** `\|K\|₉₅ ≤ \|P\| + 2` |
| `nonpartner_null` | Re-run the whole search with the corruption moved to a motif-matched non-partner AG (`CouplingExample.other_ag`), where the pilot measures ≈ +0.016 coupling against −0.436 for the true partner. | Must report **NO CIRCUIT**: either the positive-control denominator fails (< 0.20), or `F_H` never reaches 0.90 at any `\|K\|` |
| `positive_control_partner_coupling` | Unconditional mean coupling on the loaded cohort. | ≥ 0.20 |
| `both_genes_split` | D and H gene-disjoint. | each ≥ 40 |
| `both_strands_present` | | both `+` and `−` |
| `checkpoint_recorded` | sha256 of `pytorch_model.bin`. | matches `2ad91428…0149d` |

`planted_circuit_recovery` is the sensitivity arm and `nonpartner_null` the specificity arm required by
the project's calibration rule. **Both are full re-runs of the search, not point checks.** If either
fails the run STOPS and reports DIAGNOSTIC; `--force` overrides only into a diagnostic artifact.

`--smoke` = anchors + memorials on tiny-n (n ≤ 24), a few pruning steps, no curve, val sealed.

---

## 3. Decision (frozen before the run)

Report on held-out `H`, using the D-discovered removal order:

- The adaptive `F`-vs-`|K|` curve with gene-bootstrap CIs at each reported size.
- `|K|` at which `F_H` first reaches **{50, 80, 90, 95}%** (interpolated).
- The static A12 curve overlaid, on the same split, for a like-for-like difference.

**Verdict rule.** `|K|₉₀^adaptive` against A12's own preregistered knee threshold of **40**
(`docs/PREREG_CONFORMANCE.md:72` records that A12 renders "knee" as `|K|₉₀ ≤ 40`):

| Outcome | Reading |
|---|---|
| `\|K\|₉₀ ≤ 40` | A knee exists under adaptive search. **C33's no-knee is an artifact of the static ranking.** C33 gets a status change and a numbered `R` entry. |
| CI of `\|K\|₉₀` covers 72 | **C33 confirmed under a strictly stronger search.** The static ranking was not the limitation; "irreducibly distributed" is strengthened, not weakened. |
| 40 < `\|K\|₉₀`, CI excludes 72 | Bounded. Report both curves and the difference; claim neither reading. |

**Composition.** Which node types and layers survive to the 90% point, cross-checked against the
A6–A10 circuit — does the delivery spine plus generator sit inside the adaptive faithful core?

**Caveat (frozen).** Greedy is not optimal. A13's curve remains an **upper bound** on the minimal
faithful set at each F level — a tighter upper bound than A12's, not a minimality claim. C33's caveat
survives A13 in weakened form regardless of outcome.

---

## 4. Outputs, reflection

- `scripts/splicebert/run_a13_adaptive_minimality.py`; reuses A6's scrub, no new kernel.
- `results/splicebert_coupling/a13_adaptive_minimality.json` + `.manifest.json` +
  `preflight_a13_adaptive_minimality.json`.
- A test that loads the artifact and pins `preflight.PASS` and the `planted_circuit_recovery` TPR.
  (`docs/PREREG_CONFORMANCE.md:207` records that no test in the suite loads any A6–A12 artifact; the
  A13/A14 work does not repeat that.)
- **REFLECTION:** the held-out adaptive curve and its 50/80/90/95% counts; adaptive-vs-static difference
  on the same split; the verdict under §3; faithful-core composition against A6–A10; whether the
  calibration arms behaved as designed; escaped failure modes.

**Scope.** Behaviour-level, node granularity, complement-scrub-to-corrupt, `val` split. A13 produces no
edges and makes no statement about edge-level sparsity — that is A14. No prior claim is redefined by
A13 except, conditionally, C33's status.

**Provenance.** Checkpoint sha256 `2ad91428…0149d`; `F(K)` faithfulness per A6/C27; split per A12.
