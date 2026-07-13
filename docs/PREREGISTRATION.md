# Pre-registration

**[RECORD]** — what we predicted, *before* we looked. The document that lets a reader tell a prediction
from a story told afterwards.
**Read this if** you want to check whether a result was registered in advance — and, in two cases,
whether the prediction failed.
**Status: MIXED, and the distinction matters.**

| Part | Status |
|---|---|
| §Confirmatory tests (C1–C5), §Gates, §Power | **DRAFT — never frozen.** The power simulation was never run. No test chromosome may be touched until it is. |
| Addendum (a) — the necessity 2×2 | **FROZEN 2026-07-09**, before any cell ran. The verdict threshold and the attribution logic both held; the logic resolved as registered (`D → the two factors interact`). |
| Addendum (b) — tracing the relay (Phase C) | **FROZEN 2026-07-09**, before any measurement. Its T1 test was **withdrawn as miscalibrated** — see `RELAY_TRACE.md`. Registering a bad test in advance is still better than inventing a good one afterwards; the failure is recorded, not buried. |

> **The clause that did the work.** We registered that *"continuous values are reported regardless of
> which side of the threshold they land on; the threshold only labels the cell."* That single sentence is
> what let us find the L2-probe error (D35) rather than bury it. Keep it in every future registration.

---

> **Original status note (kept):** No test chromosome may be touched until this is frozen to a registry
> and the freeze recorded here with a commit hash and date. `data/genome.py` must enforce this in code,
> as `Config.assert_test_allowed` did on the previous project.

## Confirmatory tests (Holm–Bonferroni, family size 5)

A CLT over 6 layers × 4096 features yields edge counts in the millions. Correcting across all of
them leaves nothing standing. Only these five are confirmatory. Everything per-edge, per-feature and
per-layer is **exploratory, reported under FDR, and labelled as such.**

Scope is Rungs 0–1 and hypotheses H0, H1, H2, H4 (see `DECISIONS.md#d19`).

| ID | Hypothesis | Statistic | Prediction | Null |
|----|-----------|-----------|------------|------|
| **C1** | H0 | tracer recall & precision vs `PLANTED_EDGES`, **OV + QK** | both > 0.9 | random subgraph of matched size |
| **C2** | H1 | frame-circuit attribution on real sequence vs matched null | above null | dinucleotide-shuffled sequence; start-codon-scrambled sequence |
| **C3** | H2 | within-model completeness | monotone increasing across AR < MLM < MLM+phyloP < phyloP-only, **at matched eval-head accuracy** | permutation over arm labels |
| **C4** | H2 | routing share of the traced frame circuit | monotone increasing across the same ladder | permutation over arm labels |
| **C5** | H4 | routing share, traced vs matched control edges | traced ≫ matched, **sign-correct** under frameshift minimal pairs | matched random edges from the same node, same out-degree, same layer distance |

**Exploratory, not confirmatory:** H3 / JSAE Jacobian sparsity. D16 records that the statistic measures
in-degree, so it separates composition from a bag of weak features but not from a lone strong shortcut.
Reported as a caveat on the method, not as a test of the models.

All nulls are **permutation-based**, not parametric.

## Gates

- **C1 is a hard gate, and it has passed.** OV + QK edge recall against the planted circuit is 1.0;
  attribution conservation holds to 1e-21; dictionary recovery reaches MCC 0.945 where features are
  identifiable. Nothing downstream would have been reported had it failed.
- **Feature-level claims require an identifiability check** (D17). A concept absorbed into the decoder
  bias is a concept the circuit diagram silently omits. Report circuit *shape* from transcoder graphs;
  do not name a feature from its top-activating examples alone.

## Directional predictions (sign, not just magnitude)

| Rung | Latent variable | Sign prediction under edge patch |
|---|---|---|
| 1 | reading frame | frame-dependent readout shifts by exactly one codon position per nucleotide of frameshift; ablating the start codon collapses the readout to chance |

## Contamination

`config.assert_holdout_valid` **raises** for `splicebert` and `dnabert2`. Both were pretrained on the
whole human genome; no chromosome is held out from them. Mechanistic claims about these models are
in scope. **Generalisation claims are not, and cannot be made from any split.**

Only `gfm-nano-*` and the planted models admit an honest chromosome holdout.

## What would falsify the programme

- Tracer recall high on the compositional twin **and** on the heuristic twin → the tracer reports
  circuits on everything; the method is uninformative regardless of what it finds in real GFMs.
- Completeness rises monotonically with arm accuracy but **not** with objective at matched accuracy
  → H2 reduces to the behavioural literature it set out to open up.
- Frame circuit found in `gfm-nano` but not in any pretrained GFM → the phenomenon is an artifact of
  our training setup, and the paper is about toy models. Report it that way.
- Faithfulness gap large → the attribution graph is describing the transcoder, not the model, and
  every edge-level claim in the literature that omits this number is suspect. (This is a result.)

## Power

Not yet computed. **TODO before freeze:** simulate the C1/C2 monotone contrast across four arms under
plausible completeness variance to fix the number of seeds per arm.


---

## Addendum, frozen 2026-07-09 — the necessity 2×2

Registered **before** any cell was run. Rationale in `REC.md` §5–6 and `DECISIONS.md` D28.

All four cells use **μ = 1** (pure amino-acid readout — the frame is required, not merely useful).
Two factors, each at two levels:

```
                    detector EASY                     detector HARD
shortcut ABSENT     [A] uniform codons                [B] uniform + decoy ATGs + planted Kozak
                        (already run: circuit @3250)       (Kozak strength calibrated to d≈0.92)
shortcut PRESENT    [C] real CDS, no-upstream-ATG      [D] real CDS, natural
```

Cell B carries no ORF-integrity cue — with uniform codons, stops occur at rate 3/64 in *every* frame,
including the true one — so the planted Kozak motif is the only thing distinguishing the true start
from a decoy. Its strength is calibrated so the Bayes-optimal detector matches real data's measured
frame accuracy of 0.9215. That makes B a **matched-difficulty** control, not an arbitrary one.

### Verdict (frozen; both criteria required)

```
far-field cue-conflict anchor share (≥64 nt)   > 0.50
AND start-codon ablation Δ(aa accuracy)        > 0.10 beyond the decoy-Met control
```

Both are **causal**, both were validated on the planted twins (D3, D24) before being pointed at any
trained model, and **neither is the training accuracy** — which saturates, and has already fooled us
once (D27). Continuous values are reported regardless of which side of the threshold they land on;
the threshold only labels the cell.

### Attribution logic

```
D fails, B and C succeed   ->  the two factors interact
D fails, B fails           ->  detector difficulty did it
D fails, C fails           ->  the shortcut did it
D succeeds                 ->  neither factor prevents the circuit; proceed to the μ sweep on real CDS
```

### Seeds

One seed per cell for the factorial pass; **three seeds on whichever cell the attribution turns on.**
A 3-seed full factorial is 12 runs and 8 of them would confirm cells nobody disputes — cell A is
already replicated (λ=0 and λ=0.25 both grokked). μ\* is reported as an interval over ≥3 seeds.

### Expected payoffs, computed before running (`REC.md` §2a)

```
cell   bag ceiling   circuit ceiling   circuit buys
A         0.3768        1.0000            +0.6232
B         0.3768        ~0.926            +0.55
C        ~0.824         ~1.000            +0.18
D        ~0.824          0.9251           +0.10
```

---

## Addendum, frozen 2026-07-09 (b) — tracing the relay (Phase C)

Written **before** any Phase C measurement. D33 named a third mechanism from a behavioural signature;
D34 localised it by depth. Neither says what it *computes*. Three generative hypotheses, and the
interventions that separate them.

### The hypotheses

Let `a` be the index of the true start codon and `i` a far-field position (`i - a ≥ 64`).

```
H1  COUNTER                  frame(i) = (i - a) mod 3, read from an anchor-attending head.
                             Local content is irrelevant.

H2  RE-SYNC CHAIN            frame is propagated locally, position to position, seeded at the anchor
                             and periodically re-established from nearby codon evidence.
                             (This is what "relay" was named for.)

H3  ANCHOR-CORRECTED POOLING A position-local pooled estimator produces frame at the anchor-free
                             ceiling; a late layer spends the anchor to resolve the residual.
                             (What D34's depth profile suggests.)
```

### T1 — Anchor dependence as a function of distance (the plan's sawtooth test)

**The cue-conflict axis cannot run this test.** Cell C's anchor share is already at floor
(0.060 / 0.056 / 0.052 / 0.048 / 0.023 across the five distance bins); a saturated measurement reports
nothing where the effect is largest, which is the house rule we wrote after D27. The non-saturated axis
is **ATG-ablation Δ**, which D33 measured only at far field.

Measure `Δacc(d)` and `Δloss(d)` in 16-nt bins over `d ∈ [4, 255]`, each paired with a decoy-Met control.

```
H1  Δ flat-high or increasing in d   (a count is needed equally everywhere; errors may accumulate)
H2  Δ shows structure at a period > 3 nt: rises between re-sync points, drops at them
H3  Δ flat in d and equal to the aggregate  (a per-position correction has no distance scale)
```

Pre-committed statistic: fit `Δ(d)` with (a) a constant, (b) constant + slope in `log d`, (c) constant +
sinusoid at a fitted period ≥ 6 nt. Decide by BIC.
- **H3** if `|slope| < 0.02` per log2-decade **and** the periodic term's `ΔBIC < 6`.
- **H1** if the slope term is favoured (`ΔBIC > 6`) with slope ≥ 0.
- **H2** if the periodic term is favoured (`ΔBIC > 6`).
Any other outcome is reported as **undecided**, not narrated.

Power caveat, registered now: with 16 bins a period-3 effect is not identifiable against the codon
structure of the readout itself. T1 can find a *long*-period re-sync; it cannot find a codon-period one.

### T2 — Per-window conditioning on a pooled oracle (the decisive test)

Build the anchor-free pooled estimator that produced the 0.9509 ceiling (codon-level naive Bayes,
240 nt, no start codon) as a **per-window** oracle. It is computed from sequence alone; the model never
sees it. Partition held-out real-CDS windows:

```
ORACLE-RIGHT   the pooled scan recovers the true frame   (~95.1% of windows)
ORACLE-WRONG   it does not                               (~4.9%)
```

Measure far-field (`≥64 nt`) ATG-ablation `Δacc` on each subset, with the decoy-Met control on each.

```
H3  Δacc(ORACLE-RIGHT) < 0.10  AND  Δacc(WRONG) / Δacc(RIGHT) ≥ 3
H1, H2  ratio within 1.5x of 1    (the anchor is consumed everywhere)
```

**A pre-registered tension, stated with the arithmetic so it cannot be re-narrated afterwards.**
Cell C's aggregate far-field `Δacc` is **0.3529** (0.9997 → 0.6468, decoy control 0.0148). The oracle
errs on **0.0491** of windows. If the strong form of H3 held, the aggregate `Δacc` could not exceed
~0.05. **The strong form of H3 is therefore already close to falsified by a number in hand.** T2's job
is to say *how* it fails:

```
(i)   ablation is out-of-distribution and damages the pooling itself   -> T3 diagnoses this
(ii)  L4's representation is anchor-gated on every window, even though the pooled
      information exists intact at L2 (the L2 probe is unmoved by the ablation)
```

Outcome (ii) would be a **fourth** mechanism, not a variant of H3: information present and ignored
unless a gate is open. It is the outcome we consider most likely and we are recording that now.

### T3 — Anchor relocation, not deletion (the control that decides whether D33's ablation column means anything)

Replace the true ATG with CTC **and** insert an ATG elsewhere, in-frame or out-of-frame relative to the
original. All other content untouched.

```
far-field frame follows the relocated ATG      -> the anchor is read as a frame source (H1, H2)
far-field frame ignores it, tracks local content -> the anchor is a correction or a gate (H3, (ii))
far-field frame collapses to chance under both -> deletion was an OOD shock; D33's ablation
                                                  column measures distribution shift, not anchor use
```

The decoy-Met control already shows the *edit type* is nearly free (0.0148). T3 tests whether the
*position* is what is read. Registered as a falsifier of our own headline table.

### T4 — Component localisation, under the inert-head paranoia (D27)

Nominate heads by IG-QK attribution and by attention mass on the anchor. Then, for **every** nominated
head, report loss-based Δ with a paired decoy-Met control.

```
CARRYING   Δloss > 0.01 nats  AND  Δloss ≥ 10x its decoy-Met counterpart
INERT      otherwise
```

Every nominated head is reported, including those that fail — L1H2 in the forcing task had 406× decoy
specificity by attribution and a 2e-6 loss effect. Attribution ranks; only intervention decides.

### Seeds and status

Phase C runs on `cellC_s0_step8000` and `cellD_s0_step8000`, one seed each. **No Phase C claim may be
tagged `SETTLED`** until Phase D's three seeds. Everything below Phase D is `PENDING`.

### What would falsify the Phase C programme

- T3's third branch: deletion is an OOD shock. Then D33's ablation column, D34's depth localisation, and
  the "anchor is load-bearing in all four cells" headline all fall together, and the 2×2 reduces to the
  cue-conflict column alone.
- T2 returning a ratio near 1 **and** T1 returning flat: the anchor is used uniformly and distance-free,
  which no one of H1/H2/H3 predicts jointly. That would mean the mechanism is not any of the three.
