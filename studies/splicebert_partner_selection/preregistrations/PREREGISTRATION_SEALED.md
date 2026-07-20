# Pre-registration — the sealed chr8/chr9 test

**Status: FROZEN.** Written before `run_sealed_test.py --split test` was executed.
The test chromosomes (chr8, chr9) have never been evaluated. They may be opened **once**.

Everything below — the circuit, the readout, the corruption, the interventions, the
controls, and the pass/fail thresholds — is fixed here and imported from
`gfmcd.splicing.circuit`, not redefined in the runner. Nothing in `run_sealed_test.py`
selects a component. **No result on chr8/chr9 may revise any of it.**

---

## 0. Why this document exists in this form

An adversarial pre-flight audit of the protocol (36 agents, six lenses) found **four
blockers**, three of which were *controls that could not fail*. They are listed here rather
than quietly fixed, because the reason they survived so long is that each one produced a
number that looked like a result:

| # | Defect | What it did |
|---|--------|-------------|
| B-1 | `coupling.py:326` took `exons[i+1]` as the transcript-order next exon. `exons` is sorted by **genomic** start, so on the minus strand the next exon is `exons[i-1]`. | Every minus-strand exon got a negative intron length, was nulled, and was dropped. **The entire study was plus-strand only** — 41,153/41,153 minus-strand exons on chr8 discarded. The sealed split would have run on 735 examples instead of 1,568. |
| B-2 | The competing-acceptor "binding" arm **swapped** L3H1's (key, value) pair between the partner and a decoy. | A (k,v) transposition relabels the summation index in `out_i = Σ_j softmax(q_i·k_j)·v_j`. The head's output is **unchanged exactly**. It reproduced on a randomly-initialised model (\|both-swap − baseline\| = 1.2e-06). The pre-registered gate would have passed on untrained weights. |
| B-3 | `other_ag` — the "motif-matched **non-partner** AG" pool — excluded the exon's own acceptor but never `next_acceptor_pos`, which *is* the partner under this design. | The partner was a member of the null pool in **400/400** examples; the median-position pick landed on it 7/400 (val), 10/400 (test). On those examples the control corruption was byte-identical to the treatment. Removing them flips the null's sign: **+0.0033 → −0.0127**. |
| B-4 | Gate A2 used `other_gt[0]` — the **leftmost** GT in the window (median 389 nt from the corruption vs 98 nt for the real donor) — and divided by `max(\|mean\|, 1e-3)`. | It confounded partner-selectivity with attention distance-decay, and divided an effect by a mean whose expectation is zero. 0.000% of bootstrap resamples failed the 5× gate. |

All four are fixed. **The circuit was re-selected on the corrected both-strand data and is
unchanged** (exhaustive 96-head ablation, val n=160: L3H1 0.830 [0.654, 0.994] #1, L2H15
0.515 [0.396, 0.625] #2 — same heads, same order).

---

## 1. The frozen circuit

```
C = { L3H1 (reader), L2H15 (addresser) }
```

Selected on **train** (chr1–7, 10, 12–15, 17, 18, 20–22, X, Y), confirmed on **val**
(chr11, 16, 19). Defined in `gfmcd/splicing/circuit.py::CIRCUIT`. Not revisable by test.

**Known incompleteness, declared in advance:** L1H12 is a consistent third contributor
(0.266 [0.178, 0.367]). It is *not* in C. The claim is "C is the dominant sub-circuit,"
never "C is the circuit."

## 2. The behaviour

- **Readout** — the model's native masked-LM output. Mask the G of the donor GT; score
  `margin = logit(G) − max(logit A, C, T)`. No probe, no readout head, no labels.
- **Corruption** — substitute `CC` for the partner acceptor's `AG` at `[a−2, a−1]`.
  Real bases, never `[MASK]` (off-manifold). `CC` cannot re-create an AG.
- **Coupling** — `margin(clean) − margin(corrupt)`.
- **Eligibility** — internal exon; in-window intron-mate acceptor; canonical GT donor;
  donor→acceptor gap in [60, 150] nt. **Both strands.**

**The behavioural claim is UNCONDITIONAL.** The mechanism arms condition on coupling ≥ 0.3
because a recovery *fraction* needs a non-zero denominator; every such number is labelled
conditional. These are not interchangeable — on val the same population gives **0.973
[0.919, 1.023]** unconditional and **1.386** conditional, a 1.44× inflation with disjoint
CIs. The previously reported held-out coupling of **1.359 was the conditional mean** and is
retracted as a behavioural magnitude.

## 3. Interventions (all frozen)

| Arm | Intervention | Control |
|---|---|---|
| Necessity | mean-ablate L3H1, L2H15, C | matched-random 2-head sets (**distinct**, k=2 from C(94,2)); MLM health on **masked** probes |
| Mediation | patch L3H1's key / value / key+value at the acceptor AG window `[a−3 … a]`, clean → corrupt | query@donor; key+value at a distance-matched control position; every window validated against the corruption mask by `guards.py` |
| Binding | **one-way COPY** of the partner's clean key and/or value **into a distance-matched decoy AG** | key-only and value-only must each fail; key+value at a control position must fail |
| Faithfulness | ablate C; patch C | complement (94 heads); all 96 (ceiling) |

**The binding arm is a copy, never a swap.** A swap is an identity (B-2). A one-way copy
from a *different* forward pass changes the multiset of (key, value) pairs, so every cell
can legitimately return zero.

## 4. Pre-registered pass/fail gates

Set from val. **Every gate must be capable of failing** — two earlier ones were not, and
were deleted rather than re-tuned.

| Gate | Threshold | val |
|---|---|---|
| A1 | unconditional coupling, gene-clustered CI excludes 0 | +0.958 [0.855, 1.059] ✅ |
| A2 | own donor − **distance-matched** other donor: paired CI excludes 0 | +0.927 [0.817, 1.041] ✅ |
| A3 | partner AG − **non-partner** AG (partner excluded from the pool): paired CI excludes 0 | +0.945 [0.846, 1.045] ✅ |
| B1 | ablating C loses ≥ 60% of the coupling | 86.8% ✅ |
| B2 | matched-random null p ≤ 0.05 (**distinct** head sets) | 0.0025 ✅ |
| B3 | MLM health retained ≥ 90% under C-ablation, on **masked** probes | 97.8% ✅ |
| C1 | K×V interaction I, CI excludes 0 | +0.532 [0.478, 0.583] ✅ |
| C2 | key+value recovery ≥ 0.50 | 0.680 ✅ |
| D1 | relocated key+value @ **decoy** ≥ 0.30 | 0.669 ✅ |
| D2 | value alone ≤ 0.15 — content with no address does nothing | 0.004 ✅ |
| D3 | **CONTROL:** key routed to a **non-site** ≤ 0.15 | (re-run) |
| D4 | relocation retains ≥ 70% of the in-place recovery | 0.669/0.686 = 98% ✅ |
| E1 | circuit denoise recovery ≥ 0.50 | 0.83 ✅ |

**Arm D was recalibrated on val, before the seal, and the reason is recorded.** The first
draft gated "*neither* channel alone ≥ 0.15", assuming that routing the head to a decoy
without the partner's content would recover nothing. On val, `key@decoy` recovers **0.238** —
because the decoy *is* a real AG, so its value carries generic "an acceptor is here" content,
worth about a third of the read. That is a finding, not a failure. The gate now tests the
*content* channel (D2) and adds a control that can fail (D3: route the key to a **non-site**,
where there is no acceptor content to retrieve).

Also fixed here: `key+value@control-pos` was originally labelled a control and recovers
**0.656** — as much as at the decoy (0.669) or in place (0.686). It is not a control; it is a
third demonstration of the same thing. **The partner's (key, value) pair restores the coupling
wherever it is written.** The head locates its partner by *content*, not by position: the key
is an address in representation space, not a coordinate in the sequence. Expecting that cell
to be null was expecting the hypothesis to be false.

## 5. What is NOT confirmatory

Built after the val numbers were seen; reported on test **without** thresholds, as
characterisation:

- the position profile (where L3H1 writes),
- the downstream readout chain,
- the L2H15 accounting,
- the circuit-size curves.

## 6. What this test cannot settle

- **Completeness.** "Ablate everything outside C" has **no power**: deleting 47 heads breaks
  the language model (health 0.53 → 0.45), so the coupling dies whatever you delete. Against
  a size-matched null, p = 0.42 (32 heads) and p = 0.93 (47 heads) — and at *matched health*
  the circuit-preserving ablation is **gentler** on the coupling than random. This is
  reported as inconclusive, not as evidence either way.
- **Long introns.** Only introns short enough to fit 1024 nt are testable (gap ≤ 150 nt).
  Nothing here generalises to typical human introns.
- **Cross-model generality.** Not tested. One model.

---

## Freeze

**sha256 (first 16), recorded before `--split test` was ever run:**

| file | hash |
|---|---|
| `src/gfmcd/splicing/circuit.py` | `5677cc525c226930` |
| `scripts/splicebert/run_sealed_test.py` | `a3f33e59d42a4171` |
| `src/gfmcd/splicing/coupling.py` | `7d9569ac7514c58c` |
| `src/gfmcd/circuits/hf_bert.py` | `597374d253221d8a` |
| `src/gfmcd/circuits/guards.py` | `885544ff2935e4f9` |

**Validation before the seal: `--split val` → 13/13 pre-registered checks PASS.**
The protocol reproduces every held-out result under the corrected code, and every gate is
one that can fail — including D3, the control that routes the head's key to a **non-site**
(recovers 0.100; a real decoy AG recovers 0.238; the partner's value with no address
recovers 0.004).

At the time of the freeze, `results/splicebert_coupling/` contained **no test artifact of any
kind**. chr8 and chr9 had never been evaluated.

Signed off before unblinding. The test split is opened by
`run_sealed_test.py --split test --i-understand-this-unseals-the-test-split`.
