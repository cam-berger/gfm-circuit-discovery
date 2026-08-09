# Pre-registration — R4: paired edge-level causal core

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status: DRAFT.** Freeze before `run_r4_paired_edge_acdc.py` runs. Engineering record:
[`../DECISIONS_R4_EDGE.md`](../DECISIONS_R4_EDGE.md) (D1–D9). Method: Conmy, Mavor-Parker, Lynch,
Heimersheim & Garriga-Alonso, *Towards Automated Circuit Discovery for Mechanistic Interpretability*,
NeurIPS 2023 (arXiv:2304.14997), Algorithm 1.

---

## 0. What R3 left open, and why edges are the next question

`AC4` partitions the 96 heads into **73 SHARED / 17 UNRESOLVED / 4 FORWARD_SPECIFIC /
2 REVERSE_SPECIFIC**, with `L3H1` rank 1 in both directions and `L2H15` rank 2 in both. Its own scope
line is the reason this protocol exists:

> **Attribution RANKS; intervention DECIDES — this licenses no mediation claim in either direction.**

and the study README states the open question directly: *"Whether the two directions are one computation
or two is exactly what the mirrored mechanistic battery would test, and it has not run."*

**Shared components do not imply shared wiring.** Two directions can recruit the same 73 heads and route
between them differently; a component partition cannot separate those. R4 asks the question at the level
where it is separable:

> Do the two directions share their **edges**, or only their **components**?

R4 also closes two coverage gaps `AC4` names explicitly rather than leaving them implicit:

1. **MLP writes were out of scope in R3** — no mean-ablation primitive for them exists in this tree, and
   `HCA18` reports MLP writes dominating its acceptor-cluster signals. The R4 kernel decomposes MLP
   residual writes exactly and treats all six as first-class nodes, so MLP edges are in scope here.
2. **R3's noising second estimand (prereg §5.4) never ran.** R4 runs both idioms on one cohort, so the
   45× forward-`L2H15` disagreement between them is *measured* rather than inherited.

---

## 1. Design — paired, as R3 is

Both readouts on the **same exons in the same process**, so cohort, distance, composition and gene
structure cancel inside every contrast. This is R3's design lifted from components to edges.

| | mask | corrupt | quantity |
|---|---|---|---|
| forward | `donor_pos` | partner acceptor `[p−2, p)` → `CC` | `Y_D` |
| reverse | `next_acceptor_pos − 1` | partner donor `[d, d+2)` → `AC` | `Y_A` |

**Operators must match substitution counts.** `AG→CC` is two substitutions and `GT→AC` is two;
`reverse_coupling.matched_operators` gates it, per `AC2`'s scope line. A symmetry statement across
directions under mismatched edit sizes measures the operator, not the computation. The one-substitution
alt `AT` is **not** used here — matching, not maximal effect, is what R4 needs.

**Graph.** 103 nodes (`embed` + 96 heads + 6 MLPs) — identical to A12/A13/R3's component set plus the
MLPs R3 could not reach. **4,636** head-granular edges; **12,988** with Q/K/V split, run only on the
surviving subgraph plus a one-hop margin. Both counts are pinned memorials.

**Kernel.** SpliceBERT is post-LayerNorm, so ACDC's residual-additivity assumption is false here. The
kernel replays the LN recursion per destination rather than folding or freezing LN
(`DECISIONS_R4_EDGE.md` D2). Measured on the real checkpoint at S=1024: `kernel_exact_full` **3.05e-5**,
`kernel_exact_empty` **1.53e-5**, `per_head_input_identity` **2.86e-6**.

---

## 2. Two metric idioms, reported separately and never pooled

R3 chose mean-ablation as primary because the two idioms disagree 45× on exactly the class of component
at issue (forward `L2H15`: `+0.0115` noising vs `+0.5146` ablation) — a noising-primary sweep returns a
false null for the addresser. R4 inherits that hazard directly, because faithfulness `F`
(complement-scrub-to-corrupt) **is** a noising metric.

- **Primary: mean-ablation**, matching R3, so R4's partition is commensurable with `AC4`'s.
- **Secondary: `ΔF` noising**, which is R3 §5.4's unrun estimand.
- `KL(G(x) ‖ H(x, x′))` logged at every accepted and rejected decision.

**Frozen before the run (D9): an addresser edge dropped under the noising arm is NOT evidence against
`C22`/`C32`.** It is the expected behaviour of that idiom on that component, and is reported as a metric
limitation. TPR against any reference edge set is reported **per idiom**, never pooled.

---

## 3. τ is measured, not chosen

ACDC's rule is `Δ(metric) < τ`. A τ below the hardware's numerical reproducibility floor yields a
device-dependent circuit with nothing in the artifact disclosing it. The Atlas package gates cross-device
agreement at **2e-3** and amendment A3 accepts `(1e-6, 1e-4]`, so an a-priori grid is not admissible.

`device_floor` is measured on a fixed 32-subgraph panel across the reference and run devices, recorded in
the manifest, and gated by **`tau_above_device_floor`: `min(τ) ≥ 3 × device_floor`**. The grid is five
points log-spaced from `min(τ)` to `100 × min(τ)`. If the floor leaves no informative range the run
**stops and reports an instrument limit** — it does not proceed at a relaxed floor.

---

## 4. Gates and falsifiers

| Memorial | Gate |
|---|---|
| `kernel_exact_full` / `kernel_exact_empty` | `H=ALL` ≡ clean, `H=∅` ≡ corrupt, max abs err < 1e-4 |
| `per_head_input_identity` | reconstructed `r_v` == true `hidden_states[l]`, < 1e-5 |
| `edge_count` | 4,636 head-granular / 12,988 Q/K/V |
| `tau_above_device_floor` | `min(τ) ≥ 3 × device_floor` (§3) |
| `decision_determinism` | first 200 decisions bit-identical on re-run, same device |
| `equivalence_check` | candidate-parallel retains the same edges as sequential on 8 nodes spanning layers 0–5, else the sequential path stands (D7.6) |
| `planted_circuit_recovery` | a planted edge set is recovered at TPR 1.00 |
| `donor_operator_admissible` | **0 plants / N constructs**, verified per construct with `planted_donor_motifs` — never from a tuple (R44) |
| `matched_operators` | `edit_size("GT", donor_alt) == edit_size("AG", acceptor_alt)` |
| `forward_anchor` | `L3H1` rank 1 in the forward direction, reproducing `donor_heads.json` / `AC4` |
| `both_strands_present`, `checkpoint_recorded`, `chr8_9_absent` | as R1b/R3 |

Any failure STOPS the run. `--smoke` = memorials on n ≤ 24 with a truncated node list, no circuit
reported.

---

## 5. Decision (frozen)

Report, per metric idiom, on the paired cohort:

- **Paired edge partition** `Δ(e) = frac_A(e) − frac_D(e)`, Holm-corrected over the tested edge set,
  into `SHARED` / `UNRESOLVED` / `FORWARD_SPECIFIC` / `REVERSE_SPECIFIC` — the edge-level counterpart of
  `AC4`. **`UNRESOLVED` is NOT `SHARED`**, per `AC4`'s own discipline.
- **Pareto frontier**: edges retained vs held-out recovery, per direction, with gene-bootstrap CIs.
- **Edge counts** at 50/80/90/95% recovery, and whether the curve has a knee.
- **Wiring-vs-component contrast**: for the 73 heads `AC4` calls `SHARED`, are their *in-edges* also
  shared? A component that is shared while its wiring is direction-specific is the result R4 exists to
  be able to see, and it is not visible to `AC4`.
- **MLP edges**, reported separately as the coverage `AC4` could not reach.
- **Q/K/V arm** on survivors: does `L3H1` retain K-edges from `L2H15` (address) and V-edges from `MLP2`
  (payload), and does that differ by direction?

**Power.** `AC4` is powered to identify rank 1 and to detect a set, **not** to order ranks 2+ (median
`|Δ| = 0.016`; ordering an adjacent pair needs ~10⁴ paired exons). R4 inherits that limit and does not
claim edge *orderings* it cannot support. Only partition membership is reported.

---

## 6. What R4 cannot claim

- **`AC4`'s boundary stands: attribution RANKS, intervention DECIDES.** ACDC is a greedy search over
  ablations. It prunes and ranks; it establishes no mediation in either direction. An edge surviving the
  search is a **candidate**, at attribution level.
- **Greedy ⇒ upper bound.** The retained edge count at each recovery level bounds the minimal faithful
  edge set from above, exactly as `C33`'s caveat states for nodes. No minimality claim.
- **No symmetry claim beyond the operator pairing.** As `AC1`'s scope line says, comparability under one
  matched operator pairing is not a statement about the computation in general.
- **Verification tier.** Validation split; **sealed chr8/9 not opened and not spent**.
- **Nothing about splicing.** Model-internal throughout.

---

## 7. Outputs

- `src/gfmcd/circuits/acdc_graph.py`, `acdc_kernel.py`, `acdc.py`, `acdc_device.py`
- `scripts/splicebert_acceptor_coupling/run_r4_paired_edge_acdc.py`
- `results/splicebert_acceptor_coupling/r4_paired_edge_acdc.json` (+ `_meanabl`, `_noising`, `_qkv`,
  `_planted`, `_reference_smoke`) with manifests and preflights
- `tests/test_acdc_graph.py`, `test_acdc_kernel.py`, `test_acdc_algorithm.py` (genome-free, CI);
  `tests/test_r4_memorials.py` (skipif-gated, pins preflight PASS, `edge_count`, planted TPR)
- **REFLECTION:** the paired edge partition per idiom; the wiring-vs-component contrast against `AC4`;
  MLP-edge coverage; the device floor and the τ grid it forced; equivalence-gate outcome; escaped
  failure modes.

**Provenance.** Checkpoint sha256 `2ad91428…0149d`; cohort as R3; operators per `AC2` / R44.
