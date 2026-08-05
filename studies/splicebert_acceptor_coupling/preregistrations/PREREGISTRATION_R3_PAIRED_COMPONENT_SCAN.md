# Preregistration R3 — paired component scan: where the two directions share components and where they do not

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN 2026-08-05, before the R3 runner's first non-smoke execution.** Freeze covers §1–§9.
Departures go to [`../CORRECTIONS.md`](../CORRECTIONS.md) under a new `ACR` number.

---

## 0. Question, and why it is one estimand rather than two screens

`AC1` establishes that the reverse direction — mask the acceptor, corrupt the partner donor — produces
a coupling of the same size as the forward direction on the same introns, partner- and site-specifically.
It **identifies no component.** The question this arm answers is:

> **Which components are shared between the two directions, and which are direction-specific?**

That is a *paired difference*, not a comparison of two independently-run screens. Running both readouts
on the **same exons in the same process** makes cohort, distance, composition, and gene structure cancel
inside each contrast. Two separate screens would leave every one of those as a rival explanation for any
difference found.

**Tier: verification.** Validation split; **chr8/9 not opened and not spent**. This arm cannot promote
`AC1` and does not attempt a mediation claim: it is a **causal screen** whose output is a shortlist and a
set of paired differences. Per this repository's standing rule, **attribution and screening RANK;
intervention DECIDES** — §8 states exactly what would still be required to say a component *mediates*
either direction.

## 1. What the record already says, and its exact status

- **Forward, established:** `L3H1` is the reader (`C3`–`C5`, `C17`); `L2H15` acts through it on the
  address channel (`C25`); `L5H11`/`L4H2`/`L4H0` are the downstream transporters (`C18`).
- **Reverse, shortlisted only:** a motif-clean 400-shard screen
  (`results/splicebert_head_context_atlas/head_screen_clean_alt/`, `destroy_alt_mode =
  MOTIF_CLEAN_ALT=AC`) ranks `L3H1` **1 of 96** at acceptor targets, `+0.1825 [+0.1175, +0.2503]`, 82% of
  its behavioural effect against 71% at donor targets. `L5H11` falls to `0.11×` with a CI covering zero;
  `L4H11`/`L4H9`/`L4H7` rise 7–22×. **This may shortlist and may not be cited:** the artifact is
  gitignored and untracked, `planted_motif_enforced = false`, and it is a train-only alt-3′ss competition
  cohort at behaviour `+0.2226` against this study's `+0.9763` — a 4.4× different regime. It is also a
  **noising** estimand (§2).
- **A withdrawn null is still live in a canonical artifact.**
  `results/splicebert_head_context_atlas/head_effect_reanalysis_train_only_v1.json` (canonical,
  `MANIFEST.csv:941`) still carries coordinate-invalid `directional_B` acceptor values — `L3H1`
  `−0.001037` — which read as a clean reciprocal null. **No number from that file may be used here**, as
  a comparator or otherwise.

## 2. Estimand — mean-ablation primary, and why

Per component `c` and per example, on **identical exons**:

    drop_D(c) = Y_D(baseline) − Y_D(c mean-ablated)          forward necessity
    drop_A(c) = Y_A(baseline) − Y_A(c mean-ablated)          reverse necessity
    frac_X(c) = ratio_of_means(drop_X(c), Y_X(baseline))     share of that direction's coupling

    **PRIMARY:  Δ(c) = frac_A(c) − frac_D(c)**, gene-clustered bootstrap, Holm-corrected over 96.

`Δ ≈ 0` → the component is **shared**. `Δ` materially non-zero → **direction-specific**. Normalising
within direction before differencing prevents a difference in overall coupling magnitude from being read
as a difference in component usage; on this cohort the two baselines are near-identical
(`+0.9763` vs `+0.9682`), so the normalisation is a safeguard, not a fix.

**Mean-ablation is the primary estimand, deliberately.** The two idioms in this repository disagree by
more than an order of magnitude on exactly the class of component this arm is about: forward `L2H15`
reads `+0.0115` under noising (`donor_heads.json` `sanity`) and `+0.5146` under mean-ablation
(`development/ablation_scan.json`) — a **45×** gap, for the reason `run_ablation_scan.py`'s docstring
gives: denoising measures sufficiency-under-restoration, ablation measures necessity-under-removal. A
noising-primary sweep would return a **false null for the addresser**, which is the single most
interesting cell here. The shortlist in §1 is noising-based and is why it says nothing about `L2H15`.

**Noising is run as a second estimand on the detected set only** (§5), not on all 96: it is a
confirmation of a different flavour of causal claim, not a substitute sweep.

## 3. Components, cohort, controls

- **Components: the 96 attention heads**, exhaustively, `(layer, head)` for `layer ∈ 0..5`,
  `head ∈ 0..15`, mean-ablated one at a time via `circuit.ablate(ad, [(l, h)], mode="mean")`.
  **No pre-selection.**
  **MLP writes are OUT OF SCOPE and this is a capability boundary, not a choice.** `edit_heads`
  (`src/gfmcd/circuits/hf_bert.py:69`) splits and replaces *head context vectors* and has no MLP path;
  `clamp_write` forces a block's write to a **cached source** and is a path-patching primitive, not a
  mean-ablation. Mean-ablating an MLP write is a different intervention with its own health profile and
  needs a primitive that does not exist in this tree. Given `HCA18` reports MLP writes dominating its
  acceptor-cluster signals, this is a **material gap in coverage** and is named here rather than
  discovered later: R3 can say nothing about the MLP stack in either direction.
- **Cohort:** R1b's own draw-0 tag `r1b-d0`, val (chr11/16/19), paired on exons having both matched
  comparators. **`load_examples(min_effect=None)` is mandatory** — `min_effect` screens on `task.pair`,
  i.e. on the **forward** effect, and would condition the reverse arm on a forward selection variable.
- **Operators:** reverse `GT → AC` (`MOTIF_CLEAN_DONOR_ALT`, R44-clean, re-verified per construct);
  forward `AG → CC` (C42-validated). Matched at two substitutions by `matched_operators`.
- **Specificity null — per component, not a head-set null.** `drop_control_X(c)`, the same ablation
  scored on a **pseudo-coupling** that destroys a distance-matched NON-site
  (`matched_decoy_donor` for the reverse arm, `matched_decoy_acceptor_null` for the forward), per
  `run_ablation_scan.py`. **A matched-random head-set null is prohibited here**: `run_ablation_scan.py:6-9`
  records that it *"was NOT a null … circular"* — its tail is mechanism, not noise.
- **Health, per component, per direction.** `CouplingTask.mlm_health` for the forward arm and
  `reverse_coupling.reverse_mlm_health` for the reverse — the latter exists because the former **masks
  the donor**, so reusing it would health-gate a different forward pass than the one being damaged. Both
  use **masked** probes (R8/R12/R13). Floor **0.5**; a component below it is reported as
  `HEALTH_FAILED` and its `Δ` is **not interpreted**.

## 4. Memorials — each can fail, each on the measurement path

1. **Forward anchor (arithmetic, on-path).** The forward arm must return `L3H1` as the **maximum**
   `frac_D` over all 96, with `frac_D(L3H1) > frac_D(L2H15) > 0` — `run_donor_heads.py`'s own `sane`
   check. If it fails, this is not the instrument that produced `C3`–`C5` and **no `Δ` is reported.**
2. **Behavioural anchors.** Baseline `Y_D` and `Y_A` on this cohort must reproduce `AC1`'s
   `+0.9682` / `+0.9763` within `±0.15`.
3. **Construction.** Zero planted motifs across all reverse constructs (R44), verified per example.
4. **Pairing identity.** Both directions are measured on the byte-identical example list, asserted by
   hash; a `Δ` computed across different cohorts is the defect this arm exists to avoid.
5. **Health floor.** Reported per component per direction; components failing it are excluded from
   interpretation but **published, not dropped**.
6. **Specificity null is live.** The across-component SD of `drop_control` must be strictly smaller than
   that of `drop_partner` in both directions — if the null is as dispersed as the signal, it is not a null.
7. **Permuted-direction FPR.** Direction labels permuted within example, 99 seeds; the gate is the
   **rate** at which any `Δ` clears Holm-corrected significance, `≤ 0.05`. A rate, not a mean absolute
   value — a mean absolute value measures sampling noise and shrinks with `n` at any specificity.
8. **Planted positive.** A synthetic `3 × SESOI` shift injected into one component's reverse arm must be
   recovered as a material `Δ` in `≥ 18/20` seeds.

## 5. Decision rule, frozen

Evaluated in order.

1. Any memorial in §4 fails → **`INCONCLUSIVE`**; no component statement is licensed.
2. Components are partitioned by their Holm-corrected `Δ`:
   - `|Δ|` CI within `±SESOI` (`0.05`) → **`SHARED`**
   - `Δ` CI excludes 0 **and** nearer edge `≥ SESOI`, `Δ > 0` → **`REVERSE_SPECIFIC`**
   - same with `Δ < 0` → **`FORWARD_SPECIFIC`**
   - otherwise → **`UNRESOLVED`** (explicitly *not* "shared")
3. **Rank 1 in each direction is reported as an identification; ranks 2+ are reported as a detected set
   and NEVER as an ordering.** From `donor_heads.json`'s per-head half-widths the familywise MDE80 at
   this `n` is ≈`0.006` in ratio units, so anything ≥0.6% of coupling is detectable — but the observed
   rank-2-to-rank-7 gaps in the §1 shortlist are `0.0016–0.008`, and ordering a single such pair would
   need on the order of `10⁴` paired exons before correction. **This arm is powered to identify and to
   detect; it is not powered to rank.**
4. The **detected set plus rank 1 in each direction** then get the noising second estimand. A component
   that is `SHARED` under ablation but not under noising, or vice versa, is reported as **estimand-
   dependent** — that disagreement is a result, not a defect to be resolved by picking one.

## 6. What a shared or specific verdict does and does not mean

- `SHARED` means *this component's necessity, as a fraction of its own direction's coupling, is
  indistinguishable across the two readouts at this SESOI.* It does not mean the component performs the
  same computation, reads the same positions, or writes the same subspace. That is a QK/OV question and
  **no QK/OV analysis has ever been run at an acceptor readout in any form.**
- `FORWARD_SPECIFIC` / `REVERSE_SPECIFIC` locate a difference; they do not explain it.
- Per `C33`, a per-component null does **not** imply distributed absence: the forward computation has no
  sparse faithfulness knee, and nothing here changes that for either direction.

## 7. Pre-stated expectations, so they can be wrong

From the §1 shortlist (noising, different cohort, untracked): `L3H1` **`SHARED`** and rank 1 in both;
`L5H11` **`FORWARD_SPECIFIC`**; `L4H11`/`L4H9`/`L4H7` **`REVERSE_SPECIFIC`**. **`L2H15` is unpredicted**
— the shortlist's estimand cannot see it, and its verdict is the arm's most informative single cell. A
shortlist built on a 4.4×-weaker cohort under a different estimand failing to reproduce would itself be
a finding about the shortlist, not about the mechanism.

## 8. What would still be required to say "mediates"

This arm cannot license mediation for either direction. Reaching it needs, at minimum: a
reconstruct–inject factorial separating address from payload at the acceptor readout (`capture_proj` /
`patch_proj`, with the `value_linearity` arithmetic memorial); position-resolved patching via
`patch_heads(..., token_mask=…)` to distinguish a genuine reverse address from the forward geometry read
backwards; and held-out or sealed evidence, of which the reverse direction currently has **none** at
component level.

## 9. Outputs

`scripts/splicebert_acceptor_coupling/run_r3_paired_component_scan.py`;
`results/splicebert_acceptor_coupling/r3_paired_component_scan.json` + `<stem>.manifest.json`;
`status = "VERIFICATION"` unconditionally. A `REFLECTION` block states pass/fail against every §4
memorial verbatim, the §5 partition, every `HEALTH_FAILED` component, and any quantity that could not
have failed. Execution order: `--smoke` (memorials only) → preflight at reduced `n` (gates only,
endpoints withheld) → the settled run.
