# Exp 3 — Carrier × genomic-score interaction: the design, and the two problems in it

**This is the bridge from coincidence to model use.** A positive dyad spectrum (Exp 2) establishes
provenance plausibility *only*. This experiment is what licenses the word "biological."

> Test whether carrier deletion causes greater damage in high-score than low-score windows:
> **DiD = ΔL(high score) − ΔL(low score)**

Written **before** Exp 2 reports, so the design cannot be tuned to its answer.

---

## ⚠️ PROBLEM 1 — the dyad map is MOUSE, and the splicing circuit was characterised on HUMAN

Voong is **mm9 mouse ESC**. There is no chemical-cleavage map of comparable resolution for human.
But every causal result in the record — L2H15's comb, L3H1's read, the donor–acceptor coupling
0.946 → 0.027, the 0.159-vs-0.042 nats exon/intron split — was measured on **hg38 human**.

So the interaction can only be tested where a dyad score exists: **on mouse sequence.**

That is legitimate — mouse is in SpliceBERT's multi-species training set, and the user flagged
this explicitly when choosing Voong — but it has consequences that must be stated up front:

| outcome | needs | available on mouse? |
|---|---|---|
| **MLM loss** (masked-base prediction) | nothing but sequence | **yes** — this is the primary |
| exon / intron / coding strata | mm9 gene annotation (refGene) | yes, small download |
| donor–acceptor coupling | mouse splice sites | yes, but the circuit was never characterised there |
| variant-effect scores | mouse variant sets | **no** — drop |

**⇒ The PRIMARY outcome is masked-base MLM loss on mouse windows.** It needs no annotation, it is
the same quantity the period-3 result used (0.159 vs 0.042 nats), and it is the only one that
transfers cleanly. The splicing outcomes become **secondary**, and any use of them must first
show the circuit *exists* in mouse — a control that can fail, and which has never been run.

**Do not quietly report a human-characterised circuit against a mouse-derived score.**

---

## ⚠️ PROBLEM 2 — the per-window score is NOISY, and R16 is exactly this mistake

The obvious score for a SpliceBERT window is its own phase-locked dyad projection. But:

- dyad spacing is **207 bp median** (measured, from the map)
- a SpliceBERT window is 510–1024 nt
- **⇒ a window contains only ~2.5–5 dyads**

Pooling five dyads' coefficients gives a *very* noisy `C`. **R16's lesson, verbatim: "At 1–2%
modulation, an individual short window contains essentially no usable phase estimate. Estimating
each window's phase and then averaging produces precise-looking noise."** A five-dyad window is
that mistake wearing a new hat.

Noise in a *predictor* attenuates the interaction toward zero (regression dilution), so the test
stays **conservative** — it will not manufacture a false positive. But it will happily manufacture
a false **null**, and an underpowered null is worth nothing (that is Assay 1's whole lesson).

### The fix: score the REGION, not the window

Score a **10 kb neighbourhood** (~50 dyads) and assign that score to every SpliceBERT window
inside it. This trades spatial resolution for SNR — and it is arguably the *right* object anyway,
because the hypothesis is about **regional chromatin organisation**, not about one nucleosome.

SNR of a pooled score grows as √(dyads), so:

| score window | dyads | SNR vs 1 dyad |
|---|---|---|
| 1 SpliceBERT window (1 kb) | ~5 | 2.2× |
| 10 kb | ~48 | 6.9× |
| 50 kb | ~240 | 15.5× |

**Before running Exp 3, compute the score's SNR from Exp 2's measured amplitude and null sd, then
derive the number of windows needed for 80% power at the effect size Exp 2 implies. On paper.**
That is D26's rule, and it is the reason Exp 1's grid was re-anchored instead of run.

If the required N is unreachable, **say so and report the sensitivity limit** — do not run it and
report the null.

---

## The comparison (4 arms, as specified)

| arm | what it deletes |
|---|---|
| intact | — |
| **10.67 carrier deleted** | bins 94–100 (final) / bin 48 (stage 1) |
| **energy-matched band deleted** | ⚠️ **NOT bins 94–100 — that IS the carrier** (D22). Must be a genuinely inert band, matched on energy. |
| **unrelated carrier deleted** | the period-3 codon clock (bins 339–345) |

⚠️ **The study's original "energy-matched control band" is the second carrier itself.** Any Exp 3
arm that reuses it is deleting the thing under test. A fresh, genuinely-inert energy-matched band
must be selected and *verified inert* (no head reads it, deleting it costs nothing) before use.

## Matching / adjustment (the user's list)

exon/intron · coding · GC · repeats · conservation · baseline prediction difficulty · dyad
confidence. **GC is the dangerous one**: WW-density and GC are mechanically anti-correlated, so a
"high dyad-score" stratum is at risk of simply being an "AT-rich" stratum. **The score must be
residualised on GC before stratifying**, and a GC-matched null must be reported alongside.

## What a positive means, and what it does not

- **Positive interaction** → the model *uses* the candidate biological statistic. This is the claim.
- **Positive dyad spectrum, null interaction** → provenance plausibility only. The carrier is real,
  the biology is real, and the model is not using one for the other.
- **Null interaction, underpowered** → **no conclusion.** Report the limit.

Also test whether **L0H11's 10.67 modulation and effective support** rise with the score — that is
the same claim at the level of the single head that demonstrably reads the carrier (D23).
