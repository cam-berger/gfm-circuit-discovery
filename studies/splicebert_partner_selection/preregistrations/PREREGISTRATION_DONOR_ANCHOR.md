# Pre-registration — the donor-logit anchor (donor-graph experiment 1a)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: FROZEN.** Written before `run_donor_anchor.py` is executed. Every subspace
definition, intervention, control, and pass/fail threshold below is fixed here and imported
from `gfmcd.splicing.circuit` / `gfmcd.splicing.unembed`, not redefined in the runner. No
number produced by the run may revise any threshold in §4.

---

## 0. Why this document exists, and where it sits

The larger goal (see the program note in `DECISIONS.md`) is to stop searching for heads and
instead build a **causal, information-specific graph from L3H1 to the masked-donor logit** —
a meet-in-the-middle trace through MLP3 → L4-attn → MLP4 → L5-attn → MLP5. That trace has a
forward arm (from L3H1) and a **backward arm (from the output)**. The backward arm cannot
start until the output end is anchored: we must know *exactly which scalar* the graph
explains and *which low-dimensional residual direction* produces it. That is this experiment,
and nothing more.

**This experiment does not advance the circuit.** It defines the target. It reaches no new
level on the interpretive firewall (§5); it makes the existing Behavior-level readout
precise and hands the backward arm a well-defined direction `d` and information object `Δh`.

**The failure mode it is built to survive.** The core deliverable is a subspace
decomposition (an SVD), and *an SVD cannot fail*. It will always return a "dominant
direction." So the decomposition is licensed **only** by causal tests that can fail (§4):
a size-matched random-subspace null (a random direction must *not* do as well — R9/R12), and
a destructive alive-check (killing the residual must drive the score to the base-rate floor —
R8; a score that survives destruction is measuring copying, not donor evidence). The
descriptive SVD numbers in §3 are reported, but they are not evidence on their own.

---

## 1. The target — imported unchanged from `circuit.py`

Identical to the sealed pre-registration; **no redefinition**.

- **Readout** — native masked-LM output. Mask the G of the donor GT at the position set by
  `CouplingTask.encode`; score `margin = logit(G) − max(logit A, C, T)`. No probe, no head,
  no labels.
- **Corruption** — `substitute(seq, next_acceptor_pos − 2, "CC")`: replace the partner
  acceptor's `AG` at `[a−2, a−1]` with real bases (never `[MASK]`; `CC` cannot re-create an
  AG). This is a one-way, multiset-changing edit, not a swap (R7).
- **Coupling** — `margin(clean) − margin(corrupt)`, per example; the scalar is its mean.
- **Eligibility** — internal exon; in-window intron-mate acceptor; canonical GT donor;
  donor→acceptor gap in [60, 150] nt; **both strands** (R11). Val split (chr11/16/19).

**Recovery fractions use ratio-of-means**, `Σ(m_patched − m_co) / Σ(m_cl − m_co)`, with
gene-grouped bootstrap CIs (`gboot_ratio`, n_boot = 2000) — the `readout_path.py` convention.
This avoids the R10 conditional-inflation trap; **no per-example conditioning on coupling**.

---

## 2. The unembed (must-build) and its correctness gate

The MLM head is stock `BertOnlyMLMHead`:
`h → predictions.transform.dense → GELU → LayerNorm → decoder(+bias) → logits`. It is
**nonlinear** (GELU + LayerNorm), so there is no single global linear "G direction"; the read
direction is *local* to each residual.

Build `unembed(h) → logits` applying the real `cls` head to any residual, and
`donor_margin(h)`. This is the reusable logit-lens the backward arm needs.

**G0 — correctness gate (can fail → STOP):** `unembed(last_hidden_state)` must reproduce
`model(**enc).logits` at the masked donor positions to **max-abs < 1e-3**. If it does not,
the unembed is wrong and no downstream number is computed.

---

## 3. Subspaces (descriptive; reported with baselines, not as evidence)

Per eligible example, run clean and corrupt forwards **once**, caching the final residual at
the masked donor: `h_cl`, `h_co`, margins `m_cl`, `m_co`. Because `logits = unembed(h)`
exactly, every §4 causal test on the final residual is an application of `unembed()` to a
modified residual — **no re-forward** — which is what makes 1a cheap and exact.

- **U_read** — top-k right singular vectors of the stacked local read gradients
  `g_i = ∇_h donor_margin(h)|_{h_cl,i}` (leaf-tensor autograd; smooth surrogate
  `logit(G) − logsumexp(A,C,T)` used for the gradient only, verified in D3 to track `margin`).
  *What the unembed reads.*
- **U_info** — top-k right singular vectors of the stacked `Δh_i = h_cl,i − h_co,i`.
  *What acceptor-corruption moves — the information object.*
- **D1 (rank spectrum):** dims of `Δh` and of the gradients to reach 80% / 90% cumulative
  energy. Low-rankness is an empirical claim that can fail.
- **D2 (alignment):** subspace overlap (mean squared cosine of principal angles) between
  U_read and U_info at k ∈ {1,2,4,8}, against the random-overlap baseline (two random rank-k
  subspaces in R^512, ≈ k/512). This is the "read and info-object coincide" result.
- **D3 (linearity):** Pearson corr between the local-linear prediction `g_i·Δh_i` and the
  actual coupling `m_cl − m_co`, and the fraction of coupling it explains. Reported so the
  §4 sufficiency numbers are read against how nonlinear the head actually is here.

---

## 4. Causal tests — FROZEN pass/fail, each with its falsifier

All patches are one-way substitutions of a **subspace component** (project onto U, replace),
never a swap. `P_k = U_k U_kᵀ`. Primary subspace for the gates is **U_info**; U_read is run
identically and reported alongside.

| Test | Intervention | PASS threshold | What FALSIFIES it |
|---|---|---|---|
| **C-Suff** (sufficiency / low-rank) | Corrupt→clean patch of only the rank-k component: `h' = h_co − P_k h_co + P_k h_cl`; recovery = ratio-of-means of `m' − m_co` over `m_cl − m_co`. | rank-1 ≥ **0.50** and rank-8 ≥ **0.80** | rank-8 recovery < 0.50 → the readout is **not** low-rank at the final residual. |
| **C-Nec** (necessity) | Clean, **replace** the rank-k component with the corrupt run's: `h' = h_cl − P_k h_cl + P_k h_co`; necessity = ratio-of-means of `m_cl − m'` over `m_cl − m_co`. | rank-8 ≥ **0.80** | replacing rank-k barely moves the margin → the direction does not carry the coupling. |
| **C-Null** (specificity, size-matched) | N = **200** random rank-k subspaces, patched exactly as C-Suff. | observed recovery > **max** of the 200 randoms, add-one `p = (1+#{rand≥obs})/201` ≤ **0.05**, at k = 1 and k = 8 | a random rank-k subspace matches → the direction is not special (any subspace works). |
| **C-Alive** (destructive alive-check, R8) | Replace the **whole** masked-donor final residual with the no-info baseline residual (§4.1). | resulting margin ≤ **floor + 0.10 nats** | margin stays high after total destruction → the score reflects copying/position, not donor evidence, and the whole metric is suspect. |

**Amendment (2026-07-14, pre-run, no data seen).** C-Nec was originally "project the
rank-k component out of the clean residual." That measures the drop in the *absolute* clean
margin, which is not the coupling and can exceed it, so its fraction is unbounded and the
0.80 threshold is ill-posed. Replaced by the **dual of C-Suff** — substitute the corrupt
run's rank-k component into the clean residual — which isolates the coupling-carrying content
and is bounded in [0, 1]. Same intent (does removing the read subspace's clean content
destroy the coupling), correct operationalization. No outcome was observed before this edit.

### 4.1 The no-info floor and baseline residual
Define a **masked non-site G** set: for each eligible window, a G ≥ 30 nt from any annotated
splice site, masked and scored identically to the donor. Then:
- `floor` = mean `margin` over that set — the model's base-rate G preference with no donor
  context.
- `h_baseline` = mean **final residual** over that set — the residual C-Alive injects (a
  genuine no-donor-context state; note the corrupt residual `h_co` is *not* no-info, it still
  has the donor, only the partner acceptor is gone).

The mean-ablation margin (final residual → its cross-position mean) is reported as a secondary
floor. Every reachability check for C-Alive is against `floor`, computed from data, not
assumed. §7 additionally checks that `m_cl − floor` is substantial: if the donor margin is not
above base-rate, there is no donor-specific readout to anchor, which is itself a finding.

### 4.2 Reachability audit (the rule: a control that cannot fail is not a control)
- C-Suff can fail: if `Δh` is high-rank, no rank-8 patch reaches 0.80.
- C-Nec can fail: if the margin is spread across many dims, projecting out 8 leaves it intact.
- C-Null can fail: random rank-k in R^512 recovers ≈ k/512 of the coupling in expectation; if
  the observed subspace does not exceed all 200 draws, it is not special. This is the one
  non-circular specificity test and it carries the claim.
- C-Alive can fail: R8 is the proof — an unmasked-copy score *rose* under total ablation. If
  our margin does the same, we learn the metric is broken before building anything on it.

---

## 5. Interpretive firewall placement

| Level | This experiment |
|---|---|
| P → QK → A | not touched |
| OV / mediation | characterizes the final-residual → donor-logit readout (an OV/output-side fact) |
| Behavior | makes the **existing** Behavior-level coupling readout precise; adds no new behavior |
| Biology | not touched |

The anchor **defines** `d = U_read` (backward-arm target) and `Δh / U_info` (forward-arm
object) and proves they are read, specific, and destructible. It does **not** by itself
extend the circuit one edge deeper. That is 1b onward.

---

## 6. Data, compute, outputs

- **Data:** val (chr11/16/19), eligibility per `CouplingConfig`, target n ≥ 250 eligible
  donor–acceptor pairs; the matched non-site-G set for the floor. Seed 0 for all nulls.
- **Compute:** `load_adapter(DEFAULT_MODEL, "mps", eager=False)` — attention matrices are not
  needed here (hidden states + unembed only). MPS watermark env vars set defensively. Runtime
  is minutes; no L3–L5 caching, so no memory pressure.
- **Outputs:** `scripts/splicebert/run_donor_anchor.py`; `src/gfmcd/splicing/unembed.py` (the
  reusable logit-lens + `donor_margin`); `results/splicebert_coupling/donor_anchor.json` +
  `.manifest.json` (via `ExperimentManifest`).

---

## 7. Post-hoc reflection (REQUIRED — do not skip)

After the run, a `REFLECTION` block (in the result JSON and reported to the user) must state:

1. **Pass/fail** against every §4 threshold, verbatim, no softening.
2. **Intent check** against the original program: does this correctly anchor the
   *masked-donor logit* end of the L3H1→donor graph, and does read/info **alignment** (D2)
   hold strongly enough that the backward arm has a single well-defined target? If U_read and
   U_info are near-orthogonal, the "one information object flows to one readout" premise is
   wrong and the **program reframes** — this must be surfaced, not patched.
3. **Escaped failure modes:** anything that, in hindsight, could not have failed, or any
   number that looks too clean (agreement to many significant figures is the R7 tell of a
   single intervention wearing two hats).

A failing or ambiguous anchor is a real result about the target and is reported as such.
