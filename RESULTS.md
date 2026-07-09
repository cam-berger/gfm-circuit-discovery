# RESULTS — every measured number, with status

**Updated 2026-07-09 14:05** · 23 commits · 102 tests passing
Companion docs: `REC.md` (the argument) · `docs/DECISIONS.md` (D1–D31, rationale) · `docs/EXPERIMENTS.md` (plain language) · `docs/PREREGISTRATION.md` (frozen predictions)

This file exists so that nothing gets remembered more favourably than it happened. Every claim carries
a status. Retracted claims stay in, with what killed them.

```
SETTLED     measured, controlled, reproducible
PENDING     run in flight; prediction registered before the result
WITHDRAWN   measured, but the measurement is not interpretable; not reported
RETRACTED   claimed, then falsified by our own control
```

---

## 0. The one-line result

**SETTLED.** A model builds a relational circuit **iff no statistic pooled over its context window is a
sufficient statistic for the target** — and the threshold is *imperfection*, not weakness. A 96.5%
shortcut does not displace a counter when the task demands 100%.

---

## 1. Instruments, and their measured error rates

All validated against hand-weighted circuits before touching a real model. `docs/DECISIONS.md` D2–D20.

| instrument | measured performance | status |
|---|---|---|
| Direct-path attribution (frozen attention, identity dictionary) | conservation error `8e-22`; OV subgraph recall = precision = **1.0** | SETTLED |
| **Blind spot:** QK edges under frozen attention | marker edge scored **0.000**; ablating it costs **0.617** accuracy | SETTLED (D12) |
| **Blind spot:** naive pattern-gradient on a confident head | `max abs(dL/dscore) = 3.7e-15` — gradient annihilated by softmax saturation | SETTLED (D20) |
| Integrated-gradient QK attribution (bias-only baseline) | recovers the marker edge; 100% of key attribution on the marker token; **0.0** on a paired control head that is live but unread | SETTLED (D20) |
| Combined OV + QK edge recall vs `PLANTED_EDGES` | **1.0** | SETTLED |
| Dictionary recovery under planted superposition (64 atoms in 32 dims, k=3) | EV 0.966; matched \|cos\| min **0.998**; **MCC 0.945** | SETTLED (D14–15) |
| End-to-end (dictionary ∘ attribution) on a deterministic circuit | EV **1.0000**; edge recall 1.0, precision 0.6; phase from features **0.662** vs **1.000** from neurons | SETTLED (D17) |
| Faithfulness gap (frozen ReLU gate vs real) | linear attribution **overstates the true causal effect by 2×** | SETTLED (D13) |
| JSAE Jacobian sparsity | separates composition from a *bag* (+0.0637); fails against a *lone shortcut* (+0.0063) → measures **in-degree** | SETTLED (D16) |

**LayerNorm faithfulness: UNMEASURED.** Every number above came from LN-free planted models. GFM-nano
has LayerNorm. This is the last hole in Gate 0.

---

## 2. Ground-truth models

`src/gfmcd/models/planted.py`. Hand-weighted, no training.

| twin | mechanism | accuracy | marker ablation |
|---|---|---|---|
| compositional | attends the marker, computes `(i − start − 1) mod 3` | 0.586 | 1.000 → **0.383** |
| bag of motifs | five local detectors, independent votes, no frame variable | 0.583 | 0.583 → **0.583** |

Behaviourally matched to **\|Δ\| = 0.0034**. A positive control alone cannot falsify; twins can.

Third model — `build_bag_with_unread_marker_head`: a live marker head (W_Q, W_K intact) whose output is
never read. Built as a paranoid control for the QK instrument. **It later occurred naturally in a trained
model** (§6).

---

## 3. Data, and its external validation

`src/gfmcd/data/genome.py`, hg38 + GENCODE v50. **SETTLED.**

| check | result |
|---|---|
| In-frame stop codons using our frame labels | **0 of 117,481** |
| Stops with frame shifted +1 / +2 | 4.19% / 5.54% |
| `index % 3` predicts phase (randomised windows) | 0.3446 (chance 0.3333) |
| `index % 3` predicts phase (CDS-anchored windows — the bug) | **1.000** |

Biology confirms the strand/phase arithmetic; the mutation test confirms the guard against a
position-embedding lookup table (D21). Test chromosomes are locked in code behind `unlock_test_split()`.

Structural facts, measured:

| fact | value |
|---|---|
| Train CDS exons ≥150 nt | 624,533 |
| Start-codon exons / internal exons | 72,668 / 551,865 |
| Anchored windows with ≥1 **upstream** ATG | **42.9%** (mean 0.60/window) |
| Anchored windows where a sequence rule can uniquely identify the start | 70.7% (hand rule) |

---

## 4. Real DNA: no anchored frame circuit

**SETTLED.** `scripts/frameshift_conflict.py`, `scripts/probe_frame.py`.

Ablation **cannot** detect this circuit — destroying the ATG leaves the pooled route intact and the model
falls back on it (D24). Only cue conflict works: insert one nucleotide after the ATG so anchor and local
statistics disagree by exactly one codon.

```
                       trained WITH anchors     trained WITHOUT anchors (control)
distance from ATG      anchor    content        anchor    content
4–15 nt                0.5944    0.1524         0.5999    0.1553
32–63 nt               0.2423    0.5424         0.2490    0.5313
128–255 nt             0.0626    0.8354         0.0569    0.8406
```

The **control never saw a start codon and cannot be anchored**, yet its curve is identical — so near-field
"anchor following" is pooling across the insertion boundary, not counting. Both follow content at range.
The ~32 nt crossover measures the pooling width.

Phase probe, layer by layer (untrained control in parentheses):

```
resid.embed  0.4061 (0.4057)   L0  0.7936 (0.4094)   L5  0.7914 (0.4115)
```

Flat from L0 — a single pooling operation, not a computation. Baselines: nucleotide identity 0.4105,
`index % 3` 0.3401, label-shuffled 0.3351.

**Why:** `H(nt) = 1.3714`, `H(nt | phase) = 1.3473`. Perfect phase buys **0.0241 nats**; the model already
banks 0.79 from local statistics, so an anchored counter is worth **≤ 0.005 nats — 0.40% of a 1.27-nat
loss** (D24). Longer training cannot help; there is nothing to grok toward.

---

## 5. The forcing task: make it necessary, and it appears

**SETTLED.** `src/gfmcd/data/uniform_cds.py`, `scripts/verify_uniform.py`.

Uniform codons ⇒ `I(nt;phase) = 0.000001` nats (real hg38 CDS: 0.0241). Label = the amino acid of the
containing codon; the frame is never labelled. Guessing the alignment caps accuracy at **0.3768**.

```
step 2750   aa 0.4006   frameless (on the ceiling)
step 3000   aa 0.4426
step 3250   aa 0.9790
step 3750   aa 0.9987
```

2,750 steps on the plateau, then ~500 steps to solve it. At the final checkpoint every assay flips:

| assay | real hg38 model | forcing-task model |
|---|---|---|
| phase probe L0 → L2 | 0.7936 → 0.7923 (flat) | 0.4116 → **1.0000** (built across layers) |
| amino-acid accuracy | — | 1.0000 |
| start codon ablated | — | **0.3552** (below the 0.3768 guessing ceiling) |
| downstream Met ablated (control) | — | 0.9871 |
| cue conflict, 128–255 nt | 0.06 anchor / 0.84 content | **1.0000 anchor / 0.0000 content** |

---

## 6. The trace, and a naturally-occurring inert marker head

**SETTLED.** `scripts/trace_uniform.py`, `scripts/qk_ig_nano.py`.

Circuit: **L0 heads assemble the local trigram** (ablating `L0H0` halves `L1H3`'s attention to the marker,
42.4% retained; every other L0 head leaves it at 83–111% → K-composition) → **`L1H3` attends the T of the
start codon** (0.80 of its mass; 230× more than to a decoy ATG) → **MLPs count** (phase builds L0 → L2).

The model grew a *second* marker head that nothing reads:

```
head    attn→start   →decoy    ratio      ablation Δloss     accuracy
L1H2      0.6656     0.0016    406.5×      +0.000002         1.0000
L1H3      0.8344     0.0036    230.1×      +8.363416         0.3875
```

`L1H2` is **four million times less causal** than `L1H3` while looking like the *better* marker head.

| method | verdict on L1H2 (inert) vs L1H3 (essential) |
|---|---|
| attention pattern | L1H2 looks **better** |
| naive score gradient | ranks L1H2 **above** L1H3 (1881 vs 905) — backwards |
| integrated gradient | ranks correctly (83.3% / 16.5%) but still credits the inert head 16.5% |
| ablation on **accuracy** | right by luck — the metric saturates at 1.0000 |
| ablation on **loss** | 8.36 vs 0.000002 — unambiguous |

**Attribution ranks; only intervention decides.** `L1H2` is `planted.build_bag_with_unread_marker_head`,
which we invented as a synthetic control and worried was contrived (D27).

---

## 7. The λ sweep — the core result

**SETTLED.** `scripts/verdict_lambda.py`. Every arm is μ = 1 (pure amino-acid readout).

**All seven reach `aa_acc = 1.0000`. The training accuracy does not distinguish the mechanisms.**

```
arm       aa acc   ATG-abl ΔAA   anchor   content   mechanism
uniform   1.0000      0.6320     1.0000    0.0000   ANCHORED COUNTER
lam0.25   1.0000      0.5951     0.5835    0.2579   ANCHORED COUNTER
lam0.5    1.0000      0.5455     0.6048    0.2023   ANCHORED COUNTER
lam0.75   0.9999      0.2836     0.7025    0.2640   ANCHORED COUNTER
lam1      1.0000      0.0153     0.0027    0.9924   anchor-free
lam1.5    0.9988      0.0127     0.0000    0.9997   anchor-free
lam2      1.0000      0.0124     0.0000    1.0000   anchor-free
```

**These are the planted twins, learned, on a sweep.** Identical behaviour, opposite mechanism. Scored on
accuracy — as I did for an hour — the report would have been *"the circuit forms at every λ, faster each
time,"* wrong about four of seven arms (D29).

Grok step among the *anchored* arms: 3250 → 3000 → 1500 → ~750 as λ rises. A partial shortcut **scaffolds**
discovery of the exact mechanism. The apparent speed-up at λ ≥ 1 is a *different, easier mechanism*, not a
faster counter.

---

## 8. The criterion: sufficiency, not weakness

**SETTLED** for the criterion; **RETRACTED** for my first two explanations of it.

Anchor-free ceiling = best phase accuracy with **no start codon**, from full codon-level evidence pooled
over a context window.

```
data          I(nt;phase)   ctx=60  ctx=120  ctx=240   mechanism the model built
uniform-64      0.00003     0.000    0.000    0.000    ANCHORED COUNTER
sense-61        0.00162     0.505    0.814    0.965    ANCHORED COUNTER
λ = 0.50        0.00393     0.410    0.508    0.533    ANCHORED COUNTER
λ = 0.75        0.00854     0.575    0.815    0.935    ANCHORED COUNTER
λ = 1.00        0.01471     0.821    0.976    1.0000   anchor-free
λ = 2.00        0.05778     0.995    1.000    1.0000   anchor-free
real hg38 CDS   0.02410     0.754    0.922    0.9509   (pending — cells C, D)
```

**The circuit forms iff the pooled route cannot reach the ceiling.** 0.965 is not enough to displace a
counter when the task demands 1.0. Sufficiency is a knife edge — which is why the transition is sharp and
looks like grokking.

Real hg38 CDS frame accuracy, all measured on the same windows:

```
bag, ±15 nt naive Bayes                      0.6419
anchor-free stop-codon scan, 240 nt          0.8242
the trained model actually achieves          0.7936   <- sits at the anchor-free ceiling
learned anchor detector (ATG + Kozak + ORF)  0.9215   <- the anchor is worth ~+0.10 more
```

**The model is not lazy relative to what is locally available.** It extracts nearly all the distributed
frame signal the genetic code leaves behind, and stops exactly where an anchor would be required.

---

## 9. The necessity 2×2 — IN FLIGHT

μ = 1 throughout. Verdict frozen in `docs/PREREGISTRATION.md` **before any cell ran**: far-field
cue-conflict anchor share > 0.50 **and** start-codon ablation ΔaaAcc > 0.10 beyond a decoy-Met control.
Neither criterion is the training accuracy.

```
                    detector EASY                    detector HARD
shortcut ABSENT     [A] uniform codons               [B] uniform + decoys + planted Kozak
shortcut PRESENT    [C] real CDS, no upstream ATG    [D] real CDS, natural
```

Cell B is a **matched-difficulty** control: decoy rate 45.9% / mean 0.60 (real: 42.9% / 0.60); Kozak
strength calibrated so the Bayes-optimal detector hits real data's 0.9215; shortcut still absent
(`I(nt;phase) = 0.000087`). It carries no ORF-integrity cue — with uniform codons stops occur at 3/64 in
*every* frame.

All four cells, same footing (μ = 1, 900 val windows). **SETTLED.**

| cell | shortcut | detector | aa acc | ATG-abl effect (vs decoy ctl) | anchor | content | mechanism |
|---|---|---|---|---|---|---|---|
| **A** | absent | easy | 1.0000 | **+0.6094** (0.0129) | **1.0000** | 0.0000 | anchored counter |
| **B** | absent | hard | 0.9605 | **+0.5794** (0.0135) | 0.3891 | 0.2905 | mixed / partial relay |
| **C** | present | easy | 0.9997 | **+0.3529** (0.0148) | 0.0396 | 0.9131 | relay |
| **D** | present | hard | 0.9914 | **+0.1493** (0.0218) | 0.1084 | 0.8001 | relay |

**The start codon is causally load-bearing in every cell** (8×–47× specificity over the decoy-Met
control). What varies is how the frame is *propagated*: counter → mixed → relay.

Three mechanisms, separated by the **assay pair**, not by either assay alone:

| regime | ATG ablation, far field | cue conflict, far field | mechanism |
|---|---|---|---|
| real DNA, μ = 0 | **decays** to +0.0085 | content 0.84 | bag of motifs |
| real DNA, μ = 1 (C, D) | **flat** +0.35 / +0.15 | content 0.91 / 0.80 | **relay** |
| uniform codons, μ = 1 (A) | collapse *below* guessing | **anchor 1.0000** | anchored counter |

**Increasing demand moves bag → relay → counter; shortcut availability decides where it stops.**

**D30's prediction (real CDS builds an anchored counter): HALF VINDICATED, HALF FALSIFIED.** The
anchor *is* used at long range — flat ablation, and the μ = 0 bag is the contrast — but the mechanism
is a relay, not a counter. Recorded as a failed prediction, not relabelled (D32).

**The frozen binary verdict cannot express a relay** and prints NO CIRCUIT for B, C and D. We
pre-registered that continuous values are reported regardless; the 2-D signature is the result.

**Cell B (D31, confirmed):** cut off mid-grok at 8k. At 24k it transitions at step 18–20k and reaches
aa 0.9735. It is the hardest cell by construction and needs ~2.5× cell A's budget.

**Both ceilings we predicted from were lower bounds.** Cell B reaches 0.9735 vs a 0.926 detector
ceiling; cell C reaches 0.9997 vs a 0.9509 pooled ceiling. Ceilings need a model-class-matched
estimator.

## 10. Retractions and corrections — kept, not deleted

| # | claim I made | what killed it |
|---|---|---|
| D24 | *"VERDICT: bag of motifs"* from a start-codon ablation | Ablation cannot detect a redundant circuit — the model falls back on the pooled route. Cam's objection. Cue conflict was required. |
| D28 | *"Genomic landmarks are unidentifiable from sequence; the circuit buys +0.0034"* | My Kozak heuristic scored 0.707. A **learned** detector reaches **0.9215** frame accuracy. The circuit buys **+0.125**. My rule was a strawman. |
| D28 | *"So the frame is anchor-free and the circuit is redundant"* | Stop-codon scan reaches 0.8242 at 240 nt, not 1.0. |
| D29 | *"Stop-codon depletion causes the mechanism switch"* | The registered sense-codon control (zero bias, zero in-frame stops) came out **ANCHORED** (anchor 0.8205, ΔaaAcc 0.3370). Falsified by its own control. |
| D30 | *"λ = 1 synthetic models real CDS"* | Same codon usage, but naive-Bayes independence closes its ceiling to 1.0000 where real DNA's is 0.9509. Real codons are correlated. **Nothing transfers from the λ arms to real DNA without re-measurement.** |
| D31 | Cell B's `NO CIRCUIT` verdict | Cut off mid-grok. |
| — | *"The λ sweep measures necessity"* | It measures **difficulty**. Every λ arm is μ = 1; codon bias never removes the frame's necessity. I built a difficulty dial and called it a necessity dial. |
| — | *"Attention magnitude is uncorrelated with causal importance"* | I measured attention to the **A** of ATG. The marker head attends the **T**. Measured correctly, top-attention heads *are* causal — the real failure is `L1H2` (§6). |

### The recurring failure class

**A saturated measurement reports nothing exactly where the effect is largest.** Four instances:

1. Saturated **softmax** — QK gradient annihilated (`3.7e-15`), D20
2. Element-wise **variance** on a sparse target — flatters any fit, D18
3. **Accuracy** pinned at 1.0000 — nearly published "L1H2 is inert" off a metric that could not detect an effect, D27
4. **Accuracy** cut off mid-grok — nearly published "cell B has no circuit," D31

---

## 11. Unmeasured / open

| item | why it matters |
|---|---|
| **LayerNorm faithfulness gap** | every tracer error rate came from LN-free planted models. Last hole in Gate 0. |
| Pooled ceilings for Tier 1–4 tasks (`REC.md` §4) | ours are codon-level naive Bayes = a *lower bound* on the shortcut. Re-measure with a receptive-field-limited CNN. Cuts against us: makes circuits rarer. |
| μ\* (objective-mixture threshold) | needs entropy-normalised loss terms (`ln 21 = 3.04` vs `ln 4 = 1.39`), ≥3 seeds, arms compared at **matched LM loss** not matched steps |
| Transplantation 3a/3b/3c | does the circuit survive contact with real DNA, or erode? bag-as-default vs bag-as-attractor |
| RC-equivariance blindness prediction | Caduceus identifies a motif with its reverse complement, so CTCF orientation is outside its hypothesis class. One afternoon to test. |
| Power simulation; pre-registration freeze | `PREREGISTRATION.md` still carries the power TODO. Test chromosomes stay locked until it is frozen. |
| Multiplicative-interaction plant | quantifies how fast edge recall degrades with nonlinearity — answers D2's Risk #1 |

---

## 12. Reproduce

```bash
conda activate gfm-thrust1
pip install -e ".[dev]"
pytest -q                                     # 102 tests

python scripts/validate_genome.py             # 0 in-frame stops in 117,481
python scripts/verdict_lambda.py              # §7, the core result
python scripts/verify_uniform.py --checkpoint checkpoints/nano_uniform_step12000.pt
python scripts/trace_uniform.py  --checkpoint checkpoints/nano_uniform_step12000.pt
python scripts/qk_ig_nano.py     --checkpoint checkpoints/nano_uniform_step12000.pt
python scripts/anchor_ceiling.py              # learned anchor detector: frame 0.9215
python scripts/verdict_2x2.py    --checkpoint checkpoints/cellB_s0_step8000.pt
```

Data: `~/Research/data/genomes` (hg38.fa, gencode.v50.basic.annotation.gtf.gz).
Checkpoints: `nano_{uniform,sense,lam*,ar_*,mlm_*}`, `cell{A,B,C,D}_s*`.
