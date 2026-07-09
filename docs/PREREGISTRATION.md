# Pre-registration (DRAFT — not frozen)

> **Status: DRAFT.** No test chromosome may be touched until this is frozen to a registry and the
> freeze recorded here with a commit hash and date. `data/genome.py` must enforce this in code, as
> `Config.assert_test_allowed` did on the previous project.

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
