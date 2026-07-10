# RESULTS — every measured number, with status

**Updated 2026-07-10** · test suite: run `uv run pytest -q` (see `artifacts/audit/baseline.md`)
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

## 0. Current organizing hypothesis

**HYPOTHESIS, NOT YET A LAW.** Across the controlled tasks completed so far, relational mechanisms
become more useful as the best **registered** shortcut leaves more objective value unclaimed. We call
this the **shortcut-gap hypothesis** (`shortcut_gap = loss(best registered shortcut) − loss(best
relational solution)`). Current experiments support the *direction* of the hypothesis but do **not**
establish necessity, sufficiency, or an if-and-only-if threshold, and the shortcut class is always the
specific pooled family we registered — never "all pooled statistics." The suggestive knife-edge
behaviour (a 96.5% shortcut not displacing a counter when the task demands 100%) is one observation
consistent with the hypothesis, not a proof of it.

_Terminology note._ Where this document says **"counter"** for a *trained* model (the λ arms, the 2×2
cells, GFM-nano), read it as **"anchor-relative modulo-3 mechanism"**: the model uses the anchor and
carries a recoverable relative phase, but whether it *sequentially counts* or reads an anchor/position
residue class is **not yet distinguished** (`docs/studies/reading_frame/COUNTER_VS_LOOKUP.md`). The
hand-built planted twin is a counter *by construction* and keeps that name.

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
start codon** (0.80 of its mass; 230× more than to a decoy ATG) → **MLPs develop the anchor-relative phase**
(a linearly recoverable phase representation builds L0 → L2; whether this is iterative counting or an
anchor/position residue computation is unresolved — see `COUNTER_VS_LOOKUP.md`).

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

**In this sweep the circuit formed only where the pooled route could not reach the ceiling** — the
direction the shortcut-gap hypothesis predicts. 0.965 was not enough to displace the mechanism when the
task demanded 1.0. The transition is sharp and looks like grokking; whether it is a true threshold (a
knife edge) or a steep-but-continuous dependence on the shortcut gap is not established by n = 8 arms.

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

## 9. The necessity 2×2 — PROVISIONAL / REPRODUCTION BLOCKED

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

All four cells, same footing (μ = 1). **PROVISIONAL / REPRODUCTION BLOCKED — one seed per cell.** The
numbers below are retained as historical measurements, not as a settled result: cells C/D require real
hg38+GENCODE windows that are not in a clean checkout (`docs/TODO.md` §1.1), each cell has a single
seed where the pre-registration required three on whichever cell the attribution turns on
(`PREREGISTRATION.md`), and the mechanism labels must be read at the model output (D35), which the
`mechanism_signature` code path does not yet do. This is **not** a headline contribution until those
artifacts are restored. Do not cite "three of four cells contain a counter" as settled.

> **Superseded by D35.** The `anchor` / `content` columns below were read from a linear probe at
> `L2.resid_post` — a layer that on real DNA does not yet carry the anchor. Both the mechanism labels
> and the cue-conflict numbers were wrong for cells B, C and D. **The table as first published is kept
> in `docs/DECISIONS.md` D33; the corrected table is below it.** The ATG-ablation column survives (D35,
> T3's OOD branch did not fire).

| cell | shortcut | detector | aa acc | ATG-abl Δ, far field (ctl) | anchor share, **output** | mechanism |
|---|---|---|---|---|---|---|
| **A** | absent | easy | 1.0000 | **+0.5891** (0.0193) | **1.0000** | anchored counter |
| **B** | absent | hard | 0.9605 | **+0.5553** (0.0219) | **0.8817** | anchored counter |
| **C** | present | easy | 0.9999 | **+0.3264** (0.0242) | **0.7106** | anchored counter |
| **D** | present | hard | 0.9892 | **+0.1126** (0.0335) | 0.2314 | **relay** |
| μ = 0 | present | n/a | — | +0.0025 (probe) | 0.0925 (probe) | bag of motifs |

**The start codon is causally load-bearing in every cell** (3.4×–30× specificity over the decoy-Met
control). What varies is whether the frame is *counted from* it.

Three mechanisms, separated by the **assay pair**, not by either assay alone. Ablation alone cannot
tell a counter from a relay (both lose the anchor). Cue conflict alone cannot tell a relay from a bag
(both follow content). **Neither axis is sufficient; the pair is** — that is why the 2×2 needed both.

|            | far-field ablation Δ | anchor share under conflict | reading |
|---|---|---|---|
| **counter** | large | > 0.5 | the frame is *counted from* the anchor |
| **relay**   | large | < 0.5 | the anchor is *used* but not counted from |
| **bag**     | ~0    | ~0    | the anchor is *unused* |

### The registered attribution logic resolves: the two factors interact

```
A -> B  (detector hard)     1.0000 -> 0.8817     an unreliable anchor is still counted from
A -> C  (shortcut on)       1.0000 -> 0.7106     a pooled route competes, and loses
A -> D  (both)              1.0000 -> 0.2314     only together do they displace the counter
```

**Neither factor alone displaces the circuit. Together they do** — which is, exactly, the condition of
real genomic self-supervision. Full trace and controls: `RELAY_TRACE.md`.

**D30's prediction (real CDS builds an anchored counter): VINDICATED for cell C, falsified for D.**
D33 recorded it as failed. The refutation was the artifact, not the prediction (D35).

**The frozen binary verdict** (anchor share > 0.50 **and** ablation > 0.10) now labels A, B and C as
CIRCUIT FORMED and D as the one cell that stops counting. Read at L2 it had printed NO CIRCUIT for
three of them. We pre-registered that continuous values are reported regardless of the threshold; that
clause is what let the error be found rather than buried.

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
| D32 | *"Cells C and D reveal a third mechanism, the relay"* | The cue-conflict anchor share was a probe on `L2.resid_post`, and in cells C/D the anchor does not enter until **L4** (D34's own table). Read at the model's output, **cell C follows the anchor 0.7106 vs content 0.3063** — an anchored counter. Two independent interventions agree (D35). The relay survives in **cell D alone**. |
| D33 | *"Cell B is a mixed / partial relay"* | Same fault, same layer. At the output, cell B's anchor share is **0.8817**. It is a counter. |
| D34 | *"Anchor-corrected pooling: the anchor supplies the residual 5%"* | T2: cell C's ablation costs **0.3153 on exactly the positions the pooled oracle already gets right**, where a correction term should have cost nothing. The pooled estimate is a scaffold the model **overrules**, not a summand the anchor corrects. D34's *depth profile* stands; its interpretation does not. |
| — | *"The λ sweep measures necessity"* | It measures **difficulty**. Every λ arm is μ = 1; codon bias never removes the frame's necessity. I built a difficulty dial and called it a necessity dial. |
| — | *"Attention magnitude is uncorrelated with causal importance"* | I measured attention to the **A** of ATG. The marker head attends the **T**. Measured correctly, top-attention heads *are* causal — the real failure is `L1H2` (§6). |
| — | T1's registered periodicity test (`ΔBIC > 6` ⇒ re-sync chain) | Δ(d) is autocorrelated (AR(1) ρ ≈ 0.5–0.6) and BIC assumes it is not. Under 2000 AR(1) surrogates the null's own 95th percentile of ΔBIC is **17–26**. The threshold was miscalibrated before it was ever used. Withdrawn (D35). |

### The recurring failure class

**A saturated measurement reports nothing exactly where the effect is largest.** Four instances:

1. Saturated **softmax** — QK gradient annihilated (`3.7e-15`), D20
2. Element-wise **variance** on a sparse target — flatters any fit, D18
3. **Accuracy** pinned at 1.0000 — nearly published "L1H2 is inert" off a metric that could not detect an effect, D27
4. **Accuracy** cut off mid-grok — nearly published "cell B has no circuit," D31

### The second recurring failure class

**A probe at a fixed layer measures whichever variable that layer happens to carry.** Both axes of the
2×2 were read at `L2.resid_post`, chosen when the only model we had was the counter — which has the
anchor at L2. Both axes were wrong for exactly the cells that produced a new claim.

1. The **x**-axis (anchor dependence) — found by D34, fixed by taking the max over depth
2. The **y**-axis (cue-conflict anchor share) — found by D35, fixed by reading the **model's output**

The lesson is not "probe more layers." It is that a probe is a claim about a representation, and the
mechanism question is a claim about behaviour. **Where a behavioural readout exists, use it.** Both
retractions vanish when the intervention is scored against the model's own predictions.

---

## 11. Unmeasured / open

| item | why it matters |
|---|---|
| **LayerNorm faithfulness gap** | every tracer error rate came from LN-free planted models. Last hole in Gate 0. |
| Pooled ceilings for Tier 1–4 tasks (`REC.md` §4) | ours are codon-level naive Bayes = a *lower bound* on the shortcut. Re-measure with a receptive-field-limited CNN. Cuts against us: makes circuits rarer. |
| μ\* (objective-mixture threshold) | needs entropy-normalised loss terms (`ln 21 = 3.04` vs `ln 4 = 1.39`), ≥3 seeds, arms compared at **matched LM loss** not matched steps |
| Transplantation 3a/3b/3c | does the circuit survive contact with real DNA, or erode? bag-as-default vs bag-as-attractor |
| Caduceus RC-equivariance vs orientation (later study) | Whether an RC-equivariant model represents CTCF orientation in an RC-*consistent* form and causally uses it. **Not** a blindness prediction — RC equivariance ≠ invariance; equivariance preserves orientation in a representation that transforms predictably. Test representation and causal use, do not assume blindness. |
| Power simulation; pre-registration freeze | `PREREGISTRATION.md` still carries the power TODO. Test chromosomes stay locked until it is frozen. |
| Multiplicative-interaction plant | quantifies how fast edge recall degrades with nonlinearity — answers D2's Risk #1 |

---

## 12. Reproduce

```bash
uv sync --extra lab --extra dev
pip install -e ".[dev]"
uv run pytest -q                              # run the suite (see artifacts/audit/baseline.md)

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
