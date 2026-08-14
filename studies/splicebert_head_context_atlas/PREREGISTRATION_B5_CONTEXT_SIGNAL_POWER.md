# PREREGISTRATION — B5: The 20 context signals (HCA18) — power, not method (v2, FROZEN)

Status: **FROZEN 2026-07-22** after the 5-lens pre-compute design red-team (`REDTEAM_SYNTHESIS_B5.md`). v1
was a DRAFT; this v2 incorporates the six pre-run must-fixes from the red-team (all applied **before any
verdict was computed** — the branch schedules **zero model forward passes**, so freezing after design review
is outcome-blind). No lens returned `fatal`; the corrected design's terminal disposition is
**CLOSE-AS-UNDERPOWERED**, and the arithmetic below can *only* license CLOSE (see §5/§6). This is the one
runnable survivor of the reviewer roadmap B1–B5; it is executed this session as a deterministic power-calc.

**v1→v2 changes (pre-run, no verdict seen), per `REDTEAM_SYNTHESIS_B5.md`:** (1) the pooled-taxonomy
reachability branch is **removed** — it reused a fine signal's `se` on a sign-heterogeneous, cross-role,
gene-double-counting 6,487-pair cohort whose pooled `se` is unmeasured and R22-forbidden to re-draw; pooling
is marked **NON-IDENTIFIABLE-AT-DESIGN** and the 3 cassette signals route to CLOSE, never a pooled redesign.
(2) The verdict now leads with the **hard one-case-per-gene gene cap** (yield = 1.0), with explicit precedence
over the §5 marginal band; the single marginal cell (acceptor L2H4) is reported at its thin margin, not
narrated as a clean close. (3) The **census provenance is corrected**: the supply wall is the exact-other-tag
support **gene** pool, contextualized by the *atlas* census `atlas_context_census_v1.json`
(`junction_window_fit_fraction = 0.293`, ~29%), **not** `census_v1.json` (which is the intron-mate
splice-*competition* cohort's sealed split, a different estimand). (4) Two **ceiling-channel planted-positives**
PP3/PP4 are added so the reachability comparison — the branch's actual product — is exercised. (5) Demand and
supply are **unit-matched to effective pairs** via the per-cell design factor. (6) Minors: component-type count
corrected to 16 `mlp_write` + 4 `attention_head`; the `Z_POWER_80` constant is reused verbatim as
`float(norm.ppf(.975)+norm.ppf(.80))`; M4/M7 relabeled architecture-**forced** (memorial weight on M3); a
design-effect memorial M9 is added; and `N_req` is stated as a **selected, single-draw lower bound**, so
reachability is one-sided-falsifiable.

Provenance: SpliceBERT.1024nt sha256 `2ad91428…0149d`, post-LN BERT-MLM 6L×16H, hidden 512, absolute learned
position embeddings P (1026×512). Env `gfm-thrust1`; compute **MPS** (no CUDA) — but **no model run is
scheduled** (§7): B5 is deterministic arithmetic over committed JSON. Deterministic (`PYTHONHASHSEED=0`,
`sorted(set())`). All numbers below are read from committed artifacts under
`results/splicebert_head_context_atlas/`.

Pre-scoped in `EXPERIMENTS.md` as the power/census prerequisite to **WH4** (choice-topology / cluster
redistribution) and **WH5** (cassette construction): "perform an outcome-independent cohort/MDE redesign for
the acceptor-cluster, donor-cluster, and cassette-donor signals, targeting `MDE80 ≤ 0.05` without relaxing
balance, specificity, or multiplicity" (execution-order item 2). B5 supplies the power arithmetic that gates
whether WH4/WH5 can be frozen at all.

---

## What HCA18 established (the founding facts this branch must not re-litigate)

From `studies/splicebert_head_context_atlas/CLAIMS.md` (HCA18) and
`results/splicebert_head_context_atlas/atlas_discovery_analysis_v1.context.jsonl`
(the 20 rows with `context_associated == true`, verified this session):

- **20 matched context-associated discovery signals**: **17 material-suppress**, **3 material-support**, each
  surviving BH-within-estimand and cross-half stability. Concentration: **11 acceptor-cluster, 6 donor-cluster,
  3 donor-side cassette-exon**. Component types: **16 `mlp_write` + 4 `attention_head`** (three distinct heads:
  L2H4 and L2H15 once each, L3H1 under two estimands). *(v1 said "17 mlp_write"; corrected to 16/4.)*
- **NONE is validation-ready.** Every one of the 20 **fails the frozen precision gate** `MDE80 ≤ 0.05`; the
  observed `mde80` spread is **0.0901 → 0.2992** nats (verified: min L2H4 acceptor 0.0901, max L0MLP donor
  0.2992). The context mechanism shortlist is therefore correctly **empty**.
- **HCA18 boundary (binding on this branch):** *"Do not erase the 20 signals or call them validated
  specificity. A fresh, better-powered context design is required before exact mechanism validation."*

**The gate is frozen at `MDE80 ≤ 0.05` (`tau = 0.05` on every context row).** This branch does not move it.

---

## 0. INTENT (north star — re-read at every gate)

*Given the frozen precision gate `MDE80 ≤ 0.05` and the REAL observed dispersion of the 20 HCA18 context
signals, what matched-pair sample size N would each signal require to become validation-ready — and does that
N exceed the achievable supply, whose binding wall is the exact-other-tag support **gene** pool (downstream of
the atlas's ~29% junction-window-fit representation)? If demand exceeds supply for all 20, the signals close as
UNDERPOWERED LEADS (HCA18's own boundary). Because the observed `se` is a single, selection-biased discovery
draw, the honest arithmetic can only ever **rule signals OUT** (license CLOSE), never license a
"reachable"/redesign disposition — that would require a fresh, unselected measurement B5 cannot supply.*

Model-native observable: none is read here. The observables are **already-computed** per-signal `mde80`,
`se_gene_dyadic`, `n_effective_pairs`, `n_pairs`, `n_endpoint_genes` (from
`atlas_discovery_analysis_v1.context.jsonl`) and the matched-cohort supply counts
(`atlas_context_matched_v7.summary.json`, discovery side). B5 is an arithmetic adjudication over frozen
numbers, not a new measurement.

**This is the intended terminal disposition step for HCA18** — a licensed CLOSE-AS-UNDERPOWERED. It reaches no
biology and names no mechanism.

---

## 1. ESTIMANDS

All quantities are per-signal (indexed by the 20 `(component_id, contrast)` rows) and per-context (the three
cells `acceptor_cluster|acceptor`, `donor_cluster|donor`, `cassette_exon|donor`).

The repo's own precision machinery (`scripts/splicebert_head_context_atlas/analyze_atlas_discovery.py`) fixes
the identity used throughout:

```
Z_POWER_80  = norm.ppf(0.975) + norm.ppf(0.80) = 2.8015852181129683   # reused VERBATIM as float(norm.ppf(...))
mde80       = Z_POWER_80 * se_gene_dyadic
n_effective = (Σ w)^2 / Σ w^2   (Kish effective size, gene-dyadic weights)
```

Because the gene-dyadic SE scales as `se ∝ n_effective^(-1/2)` (a mean of paired differences under two-way
gene clustering), `mde80 ∝ n_effective^(-1/2)`. This yields the estimands, both **frozen before any
computation** (the observed numbers are already public, so the estimands fix the *decision function* over
them, not the inputs):

- **E1 — Required effective N (the demand), in EFFECTIVE matched pairs.** For each signal, the effective
  sample size at which its MDE80 would reach the gate, holding its observed dispersion fixed:
  `N_req(signal) = n_effective(signal) · (mde80(signal) / 0.05)^2`.
  Sign/reference: `N_req > n_effective` for every signal whose `mde80 > 0.05` (all 20). The **multiplier**
  `(mde80/0.05)^2` is the honest cost, reported alongside. **`N_req` is a LOWER BOUND on true demand** (§5,
  R22): the observed `se` is selection-biased small (the 20 passed `q_bh ≤ 0.05` + cross-half stability) and
  `N_req ∝ se²`, so the true required N is ≥ `N_req`.

- **E2 — Achievable N ceiling (the supply), unit-matched to EFFECTIVE pairs.** Per context cell, the maximum
  matched-pair supply obtainable **without relaxing balance, specificity, or multiplicity** (EXPERIMENTS.md
  item 2). The binding wall is the **exact-other-tag support gene pool**
  `G = n_case_genes_in_exact_other-tag_support` (cases with a clean control matching on all *other* context
  tags + 16 continuous confounds), because the cohort sampler is strictly **one case per gene**
  (`select_one_per_gene`, `build_atlas_matched_cohorts.py:203–213,254`) — so genes, not the larger packet pool,
  bound precision. Two supply figures, both converted from genes (raw pairs) to **effective** pairs by the
  per-cell design factor `DE = n_effective / n_pairs`:
  - **hard one-per-gene cap (yield = 1.0, the LEAD):** `N_ceil_hard_eff = G · DE` — the absolute upper bound
    (every support gene yields a matched pair). Most generous supply; if demand exceeds *this*, CLOSE is
    unassailable.
  - **realistic cap (observed yield):** `N_ceil_real_eff = G · r · DE`, `r = n_matched / n_cases_sampled`
    (≈0.80) — the honest expected supply; strictly ≤ hard cap, widening every CLOSE margin.

- **Adjudication quantity (the intent):** `closed(signal) := N_req(signal) > N_ceil_hard_eff(context(signal))`.
  A signal is **closed** iff its (lower-bound) demand exceeds even the most-generous hard-cap supply. The branch
  verdict is the logical aggregate over the 20 signals (§6). The reverse — `reachable := N_req ≤ N_ceil` — is
  **computed and reported as a per-signal diagnostic and exercised by calibration (PP3/PP4)**, but is **barred
  from licensing any disposition on the real signals** (§5: one-sided-falsifiable on a selected single-draw
  `se`).

- **POOLING is NON-IDENTIFIABLE-AT-DESIGN (removed from the decision rule).** A coarser pooled cell buys pairs
  by dropping the exact-other-tag constraint and merging sign-heterogeneous (17 suppress + 3 support),
  cross-role (acceptor + donor), gene-double-counting contrasts; the pooled effect's gene-dyadic `se` is
  **unmeasured**, and R22 forbids re-drawing it here (that would change the numbers under adjudication).
  Applying a fine signal's `mde80` to the 6,487-pair pooled cohort is a definitional/unit entanglement
  (R23/R15-class: the estimand silently changes under pooling). B5 therefore makes **no pooled reachability
  claim**; the 3 cassette signals adjudicate on the fine regime → CLOSE.

**Frozen observed inputs (read, verified, not to be recomputed):** the 20 rows and their `mde80` /
`se_gene_dyadic` / `n_effective_pairs` / `n_pairs` / `n_endpoint_genes` from
`atlas_discovery_analysis_v1.context.jsonl`; the discovery-side supply counts
(`n_case_genes_in_exact_other-tag_support`, `n_matched`, `n_cases_sampled`) from
`atlas_context_matched_v7.summary.json`; the atlas junction-window-fit context
`atlas_context_census_v1.json` (`junction_window_fit_fraction = 0.29303`).

---

## 2. METHOD (reuse the repo's own power identity; one small net-new deterministic script)

**Reused / validated machinery (exact paths):**
- Precision identity and effective-N definition — `scripts/splicebert_head_context_atlas/analyze_atlas_discovery.py`
  (`Z_POWER_80`; `mde80 = Z_POWER_80*se`; `dyadic_stats` `n_effective`). B5 **reuses this identity verbatim**
  for the `N_req` back-out, calling `float(norm.ppf(.975)+norm.ppf(.80))` (NOT a rounded `2.8016` literal — see
  M3) — it does not re-derive a power formula.
- The 20 signal rows — `results/splicebert_head_context_atlas/atlas_discovery_analysis_v1.context.jsonl`
  (`context_associated == true`), cross-checked against `atlas_discovery_analysis_v1.audit.json`.
- Matched-cohort supply — `results/splicebert_head_context_atlas/atlas_context_matched_v7.summary.json`
  (discovery side) built by `build_atlas_matched_cohorts.py`, validated by `validate_atlas_matched_cohorts.py`;
  the one-per-gene sampler is `select_one_per_gene` (`build_atlas_matched_cohorts.py:203–213,254`).
- Atlas junction-window-fit context — `results/splicebert_head_context_atlas/atlas_context_census_v1.json`
  (`junction_window_fit_fraction`).

**Net-new machinery (deterministic, no model):**
- `scripts/splicebert_head_context_atlas/analyze_b5_context_signal_power.py` — a pure-arithmetic reader that
  (1) loads the 20 rows, (2) computes `N_req` and the multiplier per E1, (3) computes the hard and realistic
  effective ceilings per E2, (4) runs memorials M1–M9 and calibration PP1–PP4/N1–N3 as hard preflight gates,
  (5) emits the closed/reachable table + verdict, and (6) writes
  `results/splicebert_head_context_atlas/b5_context_signal_power_v1.json` **with a manifest beside it**. Runs
  in seconds on CPU.

**EXCLUDED:** (a) no re-estimation of any `mde80` / `se` (the observed dispersion is the frozen input — R22
guard: re-drawing would change the numbers under adjudication); (b) no relaxation of matching
balance/specificity/multiplicity to inflate the ceiling; (c) **no pooled-cohort reachability** (E2 pooling is
NON-IDENTIFIABLE-AT-DESIGN, §1/§5); (d) **no model forward pass** — a confirmatory context run is a *downstream*
WH4/WH5 deliverable gated on B5's verdict, not part of B5.

---

## 3. MEMORIALS (on the measurement path; each CAN fail)

| # | Memorial | Passes iff | Note |
|---|---|---|---|
| M1 | signal-count reconciliation | exactly **20** rows with `context_associated==true`; **17** suppress + **3** support; **11/6/3** acceptor/donor/cassette; **16** `mlp_write` + **4** `attention_head` | on-path; fails if the artifact drifted from HCA18 (component-type split added in v2) |
| M2 | gate is frozen | every context row carries `tau == 0.05`; B5 never writes a different gate | fails if a relaxed gate leaked in |
| M3 | **precision identity round-trip (the load-bearing failable anchor)** | recomputed `float(norm.ppf(.975)+norm.ppf(.80)) * se_gene_dyadic` reproduces stored `mde80` to `<1e-9` for all 20 | on-path arithmetic anchor; fails on a wrong constant/column or a rounded `2.8016` literal (|Δ|≈1.5e-5 > tol) |
| M4 | required-N back-out consistency | at `N = N_req(signal)`, `mde80·sqrt(n_eff/N_req)` returns `0.05000 ± 1e-9` for all 20 | **FORCED** (R7/R21): an exact algebraic identity of the `N_req` definition — a code-duplication check, not physics. Reported, not leaned on. |
| M5 | supply counts are real, not capped-away | discovery `n_matched` reproduces **640/653/641**; `n_cases_sampled==800`; exact-tag support gene pools **1982/1780/2672** read from v7 discovery side | on-path; fails if the ceiling used a wrong supply column |
| M6 | ceiling monotone & bounded | per cell `N_ceil_real_eff ≤ N_ceil_hard_eff = G·DE ≤ G` (realistic ≤ hard-effective ≤ raw genes) | fails on an over-generous ceiling |
| M7 | effect-size does NOT enter power | permuting the 20 `mean_difference` values leaves every `N_req` bit-identical | **near-FORCED** (R7): the demand formula never reads `mean_difference`; a defensive specificity check, reported not leaned on |
| M8 | determinism | `PYTHONHASHSEED=0`; `sorted(set())`; re-run yields byte-identical payload (excl. timestamp) | reproducibility (IC1 A2) |
| M9 | design-effect visible | per cell report `DE = n_effective/n_pairs` (≈0.971/0.984/0.976) and `n_endpoint_genes` (1081/1117/1098); DE ∈ (0,1] and used in the ceiling | on-path (v2): makes the effective/raw unit-matching auditable; fails if DE is dropped or > 1 |

The genuine failable on-path content is **M3** (and the count/supply anchors M1/M5). M4 and M7 are labeled
architecture-**forced** per the retraction-guard discipline: they validate the code path, not the physics, and
carry no independent evidential weight.

---

## 4. CALIBRATION (sensitivity + specificity; hard preflight gates)

Because B5 is arithmetic, calibration targets the **decision function**, not a measurement channel. The
non-trivial product is the **ceiling comparison** `reachable := N_req ≤ N_ceil` — so the sensitivity control
must travel *that* channel (the v1 dead-planted-positive fix, MAJOR-C).

**Planted-positive (sensitivity):**
- **PP1 (already-powered boundary):** synthetic row `mde80 = 0.04 < 0.05` ⇒ `N_req ≤ n_effective` ⇒ flagged
  reachable via the already-powered shortcut.
- **PP2 (exact boundary):** synthetic `mde80 = 0.05000` ⇒ `N_req == n_effective` (multiplier 1.000),
  classified reachable-at-current-N.
- **PP3 (CEILING-CHANNEL reachable — the real sensitivity control):** synthetic `se` chosen so
  `N_req ∈ (n_eff, N_ceil_hard_eff)` for a reference cell (acceptor, `N_ceil_hard_eff ≈ 1925`), e.g. target
  `N_req ≈ 1200`. The pipeline **must** emit reachable **via the ceiling comparison** (not the already-powered
  shortcut). A bug in the ceiling comparison (wrong regime/column/yield) fails here.
- **PP4 (CEILING-CHANNEL not-reachable):** synthetic `se` so `N_req` sits just **above** `N_ceil_hard_eff`
  (e.g. `N_req ≈ 2000 > 1925`). The pipeline **must** emit NOT reachable via the ceiling comparison.

**Null / specificity (must not fire):**
- **N1 (effect-permutation):** permute the real `mean_difference` labels (M7). No signal's `N_req` or
  closed/reachable status may change. Target FPR 0/20 flips.
- **N2 (ceiling-shuffle):** assign each signal a *wrong* context cell's ceiling. No signal may flip to
  reachable under an incorrect (in particular, a lower) ceiling.
- **N3 (gate-relaxation tripwire):** any code path reading `tau != 0.05` for a context row aborts the run
  (HCA18 boundary + M2).

Calibration is a **hard preflight**: PP1–PP4 pass AND N1–N3 fire correctly, else the run STOPS before emitting
a verdict.

---

## 5. IDENTIFIABILITY (where this branch fails honestly)

- **E1 (demand) is identified — as a LOWER BOUND.** `N_req` is an exact function of frozen public numbers under
  the repo's own power identity (`se ∝ N^-1/2`, standard for a clustered mean). Two selection effects make it a
  **lower bound on true demand**, not a "conservative" point: (a) the 20 signals are the winners of a
  `q_bh ≤ 0.05` + cross-half-stability screen (`analyze_atlas_discovery.py`), a winner's-curse on precision ⇒
  observed `mde80` optimistically small; (b) `N_req ∝ se²`, a single-draw estimate with ~50% chance the true
  `se` is larger. **Consequence: reachability is one-sided-falsifiable — the arithmetic can only rule OUT
  (license CLOSE); it can NEVER license a "reachable"/redesign disposition on this selected discovery `se`**
  (R22). Any future "worth-powering" claim needs a fresh, unselected measurement B5 does not have.

- **E2 (supply) is an ESTIMATE with a predefined honest failure mode, led by the hard gene cap.** The binding
  wall is the exact-other-tag support **gene** pool `G` (one-per-gene sampler ⇒ genes bound precision, not the
  larger packet pool). The hard cap `G·DE` (yield 1.0) is the absolute upper bound. Assumptions, each stated so
  a red-team can attack them: (1) **retention yield** `r ≈ 0.80` — if marginal cases match *worse*, the
  realistic ceiling is lower (strengthens CLOSE); the hard cap already assumes the best case. (2) **design
  factor** `DE = n_effective/n_pairs` stable at scale — if the dyadic design effect worsens as the case pool
  grows toward the fixed gene pool (`max_control_reuse = 2`), effective supply falls faster than raw
  (strengthens CLOSE). (3) **junction-window-fit context** — the atlas cohort fits ~29% of junctions in the
  1024-nt window (`atlas_context_census_v1.json`); the support gene pool `G` is already downstream of this and
  of the matching stringency, so it *is* the real wall (no separate window-fit deflation needed). *(v1 cited
  `census_v1.json`'s ~10% / 3,014-exon / 444-gene figure — that is the intron-mate splice-**competition**
  cohort's sealed split, a different estimand; removed in v2.)*

- **Predefined common-support / marginal band, with precedence.** A cell is **closed** iff its minimum `N_req`
  exceeds the hard-cap effective supply `N_ceil_hard_eff`. Where the minimum `N_req` lands in a marginal band
  `[1.0×, 1.5×]·N_ceil_hard_eff`, the closure is reported at its **thin margin** (honest close), because
  `N_req` is a lower bound (true demand ≥ `N_req`, so exceeding even the hard cap already licenses CLOSE). This
  **precedence resolves the v1 §5/§6 contradiction**: the hard-cap CLOSE takes precedence over a
  "non-identifiable" reading of the marginal band, precisely because the demand is one-sided (a lower bound).

- **R22 guard:** the observed `mde80` are single-sample discovery estimates used only for a lower-bound demand;
  any WH4/WH5 redesign must re-establish effect existence with multiple independently-seeded draws + two-way
  gene-dyadic inference. B5 licenses no effect and no redesign.

- **R20 guard:** B5 makes no direction/preference/concept claim; it counts pairs and precision only. The
  suppress/support labels are carried through as HCA18 wrote them.

---

## 6. DECISION RULE (frozen; result → conclusion)

**SESOI:** the frozen precision gate itself — a signal is validation-*eligible* iff its achievable
`MDE80 ≤ 0.05`. Supply is led by the **hard one-per-gene cap** `N_ceil_hard_eff = G·DE` (yield 1.0).

Let `N_req*` be the minimum `N_req` within a context cell.

- **CLOSE-AS-UNDERPOWERED (the licensed HCA18 answer)** iff **for every one of the 20 signals**
  `N_req > N_ceil_hard_eff(context(signal))` — i.e. even the most-generous hard-cap supply is exceeded.
  Conclusion: *"All 20 HCA18 context signals are under-powered leads. Reaching `MDE80 ≤ 0.05` requires more
  effective matched pairs than exist even at the absolute one-case-per-gene supply cap (acceptor min ~1.05×,
  donor min ~3.9×, cassette min ~1.9× over; realistic-yield margins wider). The signals are retained as
  documented leads (not erased, not validated); no context mechanism is licensed on the frozen taxonomy at the
  representation/supply budget."*

- **No other disposition is licensable on these inputs.** A "reachable"/SCOPED-REDESIGN verdict is **barred**
  (§5: `N_req` is a selected single-draw lower bound ⇒ one-sided-falsifiable). A POOLED-ONLY redesign is
  **barred** (§1: pooling is NON-IDENTIFIABLE-AT-DESIGN; the 3 cassette signals adjudicate fine → CLOSE). The
  `reachable` diagnostic is still computed per-signal and exercised by PP3/PP4, but it does not drive the
  disposition. The ONLY way the branch does **not** CLOSE is a memorial/calibration failure (run STOPS, no
  verdict) or a data-artifact drift caught by M1/M5 — neither of which changes the science.

**Precedence (resolves v1 §5/§6):** the hard-cap CLOSE test above takes precedence over the marginal-band
"non-identifiable" reading; a signal over the hard cap is CLOSED and its margin reported. The single marginal
cell is **acceptor L2H4**: raw-gene headline `N_req 2,019 / G 1,982 = 1.019×`; unit-matched to effective pairs
`2,019 / (1,982·0.9713 ≈ 1,925) = 1.049×` — unit-matching **widens** the margin (strengthens CLOSE). Reported
honestly at ~1.02–1.05× over, not as a clean close.

**Pre-registered expectation from the read numbers (stated to be falsified by a memorial failure, not
assumed):** minimum `N_req` per context `acceptor 2,019` / `donor 6,790` / `cassette 4,897`; hard-cap
effective ceilings `≈1,925 / ≈1,751 / ≈2,609`; every `N_req*` exceeds its hard-cap ceiling ⇒
**CLOSE-AS-UNDERPOWERED**. The frozen gates of §8, not this expectation, license the verdict once the script
runs.

---

## 7. POWER / MDE & HELD-OUT

- **No model forward pass.** B5's "power" is the object of study, not a constraint on B5 itself; the branch
  runs deterministic arithmetic over committed JSON in seconds. The MPS budget binds the *downstream* WH4/WH5
  run, which is exactly what B5 quantifies. (Contrast P3.4, where the MPS power wall was the fatal; here the
  power wall is the *finding*, computed rather than hit.)
- **The supply gap IS the answer.** Reaching `MDE80 ≤ 0.05` for even the best signal needs ≥ ~2,019 effective
  matched pairs vs the current ~622; the full set spans ~2,019–22,996 required effective pairs. The
  exact-other-tag support gene pools (1,780–2,672 genes/context, downstream of the ~29% atlas window-fit) cap
  effective supply at ~1,751–2,609 — **below** even the smallest demand in every context.
- **Held-out / selection discipline:** B5 selects no genes and reads no validation outcome. The 20 signals were
  fixed by HCA18's discovery split; supply is read from the outcome-independent v7 candidate
  (`status: OUTCOME_INDEPENDENT_MATCHED_CANDIDATE`, gene-hash-disjoint discovery/validation, chr8/9 absent). No
  sealed split is opened. Any WH4/WH5 redesign B5 licenses inherits the full discovery→held-out→multi-draw
  discipline (R22) and must not read chr8/9 on a compelling direction (EXPERIMENTS.md item 5).

---

## 8. GATES

- **Global PASS = AND(** M1..M9, PP1, PP2, PP3, PP4, N1, N2, N3 **).** Any memorial or calibration failure STOPS
  the run before a verdict is written. N3 additionally aborts on any `tau != 0.05`.
- **`--smoke`:** M1 (count reconciliation, incl. 16/4 component split), M2 (frozen gate), M3 (precision
  round-trip), M7 (effect-permutation invariance), M9 (design-effect visible) on the 20 real rows plus the four
  synthetic calibration rows (PP1–PP4) — the arithmetic-only checks, **no ceiling estimate emitted, no verdict
  emitted**. Must pass before the full ceiling/verdict computation. (No "val sealed" content beyond not opening
  chr8/9, which B5 never touches.)

---

## 9. SCOPE

- **Reaches:** a **representation/power adjudication** of the HCA18 context shortlist — the evidence-ladder
  position is *below* the causal-screen column the 20 signals already occupy; B5 adds **no** ladder promotion.
  It licenses only a disposition (CLOSE) and, informationally, the supply gap.
- **Does NOT establish:** that any of the 20 signals is real, validated, specific, or biological (HCA18
  boundary — B5 must not erase them and must not call them validated); any context *mechanism*; any component
  *function name*; any QK/OV/MLP-path decomposition. A CLOSE verdict means "under-powered lead," **not** "no
  effect."
- **Guards honored:** frozen `MDE80 ≤ 0.05` gate (never relaxed); real observed dispersion (never re-drawn,
  R22) treated as a one-sided lower bound; no direction/concept claim (R20); pooling flagged
  NON-IDENTIFIABLE-AT-DESIGN, never sold as fine specificity; correct atlas census provenance; forced memorials
  (M4/M7) labeled; the marginal cell reported at its thin margin.
- **Provenance:** checkpoint sha `2ad91428…0149d`; `PYTHONHASHSEED=0`, `sorted(set())`; inputs are
  `atlas_discovery_analysis_v1.context.jsonl` (+ `.audit.json`), `atlas_context_matched_v7.summary.json`,
  `atlas_context_census_v1.json`; output `b5_context_signal_power_v1.json` writes a manifest beside it and never
  overwrites the scientific record.
- **Amendments:** pre-run only, logged `Amendment (date, pre-run, no data seen)`. Once the verdict payload is
  inspected, any fix is *outcome-aware verification*, not confirmatory (IC1 A2 lesson).
