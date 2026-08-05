# Pre-registration — A14: edge-level ACDC on the masked-donor margin (donor-graph experiment 18)

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: DRAFT.** Freeze before `run_a14_edge_acdc.py` runs, and after A13 has completed and passed its
calibration gates. Kernel and Colab-packaging decisions are recorded in
[`../DECISIONS_ACDC.md`](../DECISIONS_ACDC.md).

Method reference: Conmy, Mavor-Parker, Lynch, Heimersheim & Garriga-Alonso, *Towards Automated Circuit
Discovery for Mechanistic Interpretability*, NeurIPS 2023 (arXiv:2304.14997v4), Algorithm 1.

---

## 0. Why this document exists

A13 answers whether adaptive search changes the minimality picture at **node** granularity. It cannot
say which *connections* carry the coupling. The donor-graph arc's central structural claims are edge
claims stated at node resolution — C31's `{L0H7, L0H12, L0H10} → MLP0 → MLP1 → MLP2` wiring, C29's
backbone, C35's routing through L3H1's attention pattern — and each was found by a hand-designed,
hypothesis-directed intervention.

A14 asks the question those cannot answer:

> Does an automated, hypothesis-free edge search, given only the task and the corruption operator,
> recover the causal core we found by hand — and what does it find that we did not?

This is ACDC's own validation design (its §4.1 Q1/Q2), applied to a genomic foundation model rather
than to GPT-2 Small. A negative result is informative: it bounds how much of the arc's structure is
recoverable without the hypotheses that generated it.

---

## 1. The graph

**Nodes (103).** `embed`, 96 attention heads (6 layers × 16), 6 MLPs — identical to A12/A13, so node
results are directly comparable across the three experiments.

**Edges (4,636), head-granular.** `u → v` for every node `u` strictly upstream of `v` in the ordering
(embed ≺ layer-`l` heads ≺ layer-`l` MLP ≺ layer-`l+1` heads ≺ …), plus `u → readout` for all 103 nodes.
Count verified by enumeration and pinned in the runner as a memorial.

**Q/K/V split (12,988 edges) is a second arm**, run only on the subgraph surviving the head-granular
sweep plus a one-hop margin. It is what separates *address* from *payload* at edge resolution: whether
L3H1 retains **K**-edges from L2H15 (C22/C32, addressing) and **V**-edges from MLP2 (C29/C31, payload).

---

## 2. The kernel — and why the standard one does not apply

ACDC assumes an additive residual stream: `r_v = Σ_{u ∈ pa(v)} write(u)`. **That assumption is false
for SpliceBERT.** HF BERT is post-LayerNorm and re-normalizes twice per encoder layer, so the residual
at any depth is not a sum of upstream writes.

A14 therefore does **not** linearize or fold LayerNorm. It replays the exact recursion, once per
destination node:

```
r_v ← sel_v(embed)
for each block b strictly upstream of v, in write order:
    r_v ← LN_b( r_v + sel_v(b) )

    sel_v(b) = live_out(b)              if edge b → v ∈ H
             = corrupt_cached_out(b)    otherwise
```

This reduces to the true clean forward when `H = ALL`, and to the true corrupt forward when `H = ∅`.
Those two identities are the kernel's correctness test and they are **falsifiable**: a wrong routing,
a wrong write order, or a wrong LN association breaks them. They are unlike A2b's `decomposition`
check, which `docs/PREREG_CONFORMANCE.md:139` records as a forced arithmetic identity with no
discriminating power.

Heads within a layer receive different `r_v`, so the kernel substitutes a per-head
`BertSelfAttention.forward` computing `Q_h = W_Q^h r_{(l,h)}` and likewise for K and V — sound because
SpliceBERT is post-LN, so QKV is linear in the layer-input residual (the same property `payload.py`
already relies on). Kernel lives at `src/gfmcd/circuits/acdc.py`; it is model-generic over HF BERT
encoders, not donor-specific.

---

## 3. Metric and algorithm

- **Pruning metric: `ΔF`**, where `F` is the A6/A12/A13 faithfulness — so A14's numbers are in the same
  units as C27, C33 and A13, and the node-vs-edge curves can be laid on one axis.
- **`KL(G(x) ‖ H(x, x′))`** over the four-base distribution at the masked donor is **logged at every
  accepted and rejected decision**, and one τ arm is pruned on KL instead. ACDC's Appendix E.1 records
  that task-specific metrics systematically miss *negative* components — units that actively harm the
  task. That blind spot is disclosed, and the KL arm is how it is measured rather than assumed away.
- **Algorithm 1, verbatim.** Reverse-topological node order; for each node `v`, for each parent `w`,
  remove `w → v` permanently if `metric(H \ {w→v}) − metric(H) < τ`. Nodes with no outgoing edge in `H`
  are skipped, as in the reference implementation.
- **Parent order is a hyperparameter.** It is recorded in the manifest, and one arm runs a permuted
  parent order — ACDC's Appendix J documents a case where the order changed the result.
- **Discovery on `D`, evaluation on held-out `H`**, same seed-0 gene split as A12/A13. `val` only; the
  sealed chr8/9 test is **not opened** by A14.

---

## 4. τ is set by measurement, not by choice

ACDC's decision rule is `ΔF < τ`. A τ below the numerical reproducibility floor of the hardware
produces a **device-dependent circuit** — the same code and the same data give different answers on
different accelerators, and nothing in the output reveals it.

This is not hypothetical for this repository. The Head–Context Atlas Colab package gates its
cross-device smoke at **absolute tolerance 0.002**, and amendment A3 accepts a cross-batch reference
discrepancy in `(1e-6, 1e-4]`. Any τ grid chosen a priori risks sitting under that floor.

**Therefore the τ grid is derived, and gated:**

1. Evaluate a fixed panel of 32 subgraphs on the local reference device and on the run device.
2. `device_floor` = max absolute `ΔF` across the panel. Recorded in the manifest.
3. **`tau_above_device_floor` (memorial, can fail):** `min(τ_grid) ≥ 3 × device_floor`.
4. The grid is five points, log-spaced from `min(τ)` to `100 × min(τ)`.

If the floor is large enough that a five-point grid cannot span the informative range above it, the run
STOPS and reports the floor as an instrument limit. That is a licensed outcome, not a failure to work
around.

---

## 5. Gates and falsifiers

**Forced anchors — recorded, not memorials:** none. Unlike A12/A13, `H = ALL` and `H = ∅` are *not*
forced here: they are the reconstruction identities the new kernel can violate, so they gate.

| Memorial | Gate |
|---|---|
| `kernel_exact_full` | `H = ALL` reproduces the clean logits, max abs err < 1e-4 |
| `kernel_exact_empty` | `H = ∅` reproduces the corrupt logits, max abs err < 1e-4 |
| `per_head_input_identity` | with all edges present, each head's reconstructed `r_v` equals the true `hidden_states[l]`, max abs err < 1e-5 |
| `edge_count` | enumerated graph has exactly 4,636 head-granular / 12,988 Q/K/V edges |
| `tau_above_device_floor` | `min(τ_grid) ≥ 3 × device_floor` (§4) |
| `decision_determinism` | the first 200 edge decisions, re-run on the same device under `torch.use_deterministic_algorithms(True)`, are bit-identical in accept/reject sequence |
| `planted_circuit_recovery` | as A13 §2, lifted to edges: a planted edge set must be recovered at TPR 1.00 |
| `nonpartner_null` | full search under non-partner-AG corruption must report **NO CIRCUIT** |
| `positive_control_partner_coupling` | ≥ 0.20 |
| `both_genes_split`, `both_strands_present`, `checkpoint_recorded` | as A13 |

`decision_determinism` exists because ACDC is a sequential greedy: one nondeterministic kernel call
early in the sweep changes every subsequent decision. Without this gate a re-run silently produces a
different circuit.

Any memorial failure STOPS the run. `--smoke` = memorials on n ≤ 24 with a truncated node list, val
sealed, no circuit reported.

---

## 6. The reference edge set — frozen here, before the run

TPR/FPR against a reference is meaningless if the reference can be adjusted after seeing the result.
The reference is fixed now.

**`R_core` (8 edges), assembled from C29, C31, C35 and the frozen `CIRCUIT` constant:**

```
L0H7  → MLP0        L0H12 → MLP0        L0H10 → MLP0        (C31)
MLP0  → MLP1        MLP1  → MLP2                            (C31, C29)
MLP2  → L3H1        L2H15 → L3H1                            (C29, C22/C32)
L3H1  → readout                                             (C35, CIRCUIT.READER)
```

**`R_ext` (12 edges)** = `R_core` + `embed → L0H{7,12,10}` + `embed → L2H15`. Secondary.

**This set is assembled, not quoted.** No single row in `CLAIMS.md` states it as one circuit; it is
composed here from rows that each establish part of it at node resolution. That composition is itself
a claim of this preregistration and is frozen with it.

**Reference status.** Following ACDC §4.1: the hand-found circuit is a *practitioner-reported*
reference, not ground truth. It was built by humans who recorded the limits of their understanding, it
may contain extraneous edges, and it may miss real ones. **Disagreement is reported in both
directions and is not by default an ACDC error.**

---

## 7. Decision (frozen before the run)

Report, on held-out `H`, across the τ sweep:

- **Pareto frontier:** edges retained vs held-out `F`, with gene-bootstrap CIs — the analogue of ACDC
  Figure 4 and the edge-level counterpart to A13's node curve. Both curves on one axis.
- **Edge counts** at which `F_H` reaches {50, 80, 90, 95}%.
- **Knee vs gradual**, by the same rule A13 uses at node level.
- **TPR / FPR / AUC** of the recovered edge set against `R_core` and `R_ext`.
- **Recovered-but-unreferenced edges**, ranked by `ΔF` — the "what did it find that we did not" list,
  reported as *candidates*, at attribution level only. An edge surviving ACDC is not thereby an
  established mechanism; promoting any of them requires its own directed intervention.
- **KL arm and permuted-parent-order arm** against the mainline, as stability checks.

**Q/K/V arm** (second stage): whether L3H1 retains K-edges from L2H15 and V-edges from MLP2, reported
as the edge-level status of the address/payload distinction.

**Caveat (frozen).** ACDC is greedy. Its edge count at each F is an **upper bound** on the minimal
faithful edge set, exactly as C33's caveat states for nodes. A14 makes no minimality claim.

---

## 8. Execution boundary

Per `../DECISIONS_ACDC.md`: the τ sweep, both calibration arms, and the Q/K/V arm exceed three hours
and run on **Colab GPU** from a hash-verified bundle. A13 and the reference smoke run locally. The
bundle carries `val` sequences only — chr8/9 are physically absent, so the seal cannot be spent by
accident on a remote runtime.

**Val is already seen** by A12, and by A13, for the adjacent question. A14 reports at **verification
tier** on `val`. Whether a contradiction with C33 justifies spending the sealed chr8/9 test is a
decision for the owner at the REFLECTION gate; this protocol does not license it.

---

## 9. Outputs, reflection

- `src/gfmcd/circuits/acdc.py`; `scripts/splicebert/run_a14_edge_acdc.py`;
  `scripts/splicebert/build_acdc_cohort.py`; `studies/splicebert_partner_selection/colab_acdc_package/`.
- `results/splicebert_coupling/a14_edge_acdc.json` (+ `_qkv`, `_kl`, `_permorder`, `_planted`,
  `_nonpartner`, `_reference_smoke`) with manifests and preflights;
  `results/splicebert_coupling/acdc_cohort_v1.sequences.jsonl` + manifest.
- `tests/test_acdc_kernel.py` — exactness on a tiny randomly-initialized BERT, genome-free, runs in CI.
- `tests/test_acdc_memorials.py` — skipif-gated, pins preflight PASS, `edge_count`, and the planted TPR.
- **REFLECTION:** the Pareto frontier and edge counts; knee vs gradual against A13's node result;
  TPR/FPR/AUC against `R_core`; the unreferenced-edge candidate list with its evidence level stated;
  the device floor and the τ grid it forced; KL-arm and parent-order stability; escaped failure modes.

**Scope.** Behaviour-level, edge granularity, complement-scrub-to-corrupt, `val` split, verification
tier. Evidence level is OV/mediation plus behaviour — **not biology**, and no payload semantics (R22).
A14 may change the status of C33 and may add candidate edges; it redefines no established claim on its
own.

**Provenance.** Checkpoint sha256 `2ad91428…0149d`; `F` per A6/C27; split per A12; cohort sha256
recorded in the bundle manifest.
