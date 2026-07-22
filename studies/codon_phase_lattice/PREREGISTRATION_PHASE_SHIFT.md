# PRE-REGISTRATION — the phase-shift intervention

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**v3, 2026-07-12.** v1 and v2 were both **killed before they were run** — v1 by an adversarial
design review, v2 by a memorial test that failed and by a check on the head set itself. The
amendment log (§9) records every change and why. Nothing in this document has been fitted to an
outcome.

> ## ⚠️ v3 EXISTS BECAUSE THE HEAD SET WAS WRONG (R14)
>
> v1 and v2 both tested "the 42-head codon-phase lattice." **There is no such lattice.** The
> purity metric that defined it is maximised by a head attending to **exactly one position** —
> one offset has one remainder mod 3, so a single-offset head scores 1.000 by arithmetic.
> `corr(purity, effective support) = −0.767`; all ten highest-purity heads are **single-offset
> copiers**; 24 of the 42 have effective support < 2.
>
> **Seven genuine stride-3 combs survive** (L0H0, L0H11, L0H13, L1H10, **L1H12**, L2H2,
> **L2H15**) — spread over 15–37 effective offsets, teeth all congruent mod 3, layers 0–2 only.
> **L2H15 is the coupling circuit's ADDRESSER**, found independently by an ablation scan that
> knew nothing about periodicity.
>
> **And this hands us the control the old design could not have had.**
> A single-tooth head has `w(τ) = δ_{τ,0}`, so `w(δ) = 0` for every δ ≠ 0:
> **its perturbation size is FLAT in δ and it CANNOT produce a comb-overlap sawtooth** —
> while scoring *identically* to the combs on the metric the lattice was built from.
>
> ### COMBS vs COPIERS: matched on the statistic that lied, opposite in structure.

---

## 0. Why this experiment exists: the retraction that precedes it (R13)

`run_phase_function.py` ablated the 42 lattice heads and measured the period-3 enrichment of the
model's masked-LM confidence. It reported that periodicity **does not fall** (3.34× → 4.52×,
add-one p = 0.625) and concluded the lattice is not what produces codon-periodic prediction.

**That conclusion is void.** The ablated model was at chance:

| arm | mean log-prob of the true base | probability |
|---|---|---|
| intact, exon | −1.120 | 0.326 |
| **42 lattice heads ablated, exon** | **−1.377** | **0.252** |
| 15 size-matched random 42-head nulls | −1.23 … −1.40 | ~0.25–0.29 |
| **chance (ln ¼)** | **−1.386** | **0.250** |

Intact headroom over chance is 0.076 nats. The ablation removes 0.074 of it — **97%**. The
period-3 statistic was computed on a model that was no longer predicting anything, and the
observed *rise* is the tell: a wrecked model falls back on its (trivially periodic) position
prior. That is also why 9 of 15 random nulls "did as much" — they were all equally dead.

**Third instance of one failure mode** (R8: probes never masked; R12: unmatched-health null).

> **An intervention that destroys the behaviour cannot measure the mechanism.**

**The function of the lattice is OPEN, not closed.**

---

## 1. The behaviour

Per-position log-probability of the **true base** under the native MLM head, at **masked**
positions, masked at stride 7 (**coprime with 3**, so the mask pattern cannot manufacture a
period-3 artefact and the density of masked tokens seen by a rolled read is 1/7 for every δ).

**Region = exon interior vs intron interior, restricted to `gene_type == "protein_coding"`.**
`gtf.py:113` hard-filters to `feature == "exon"` and never reads a CDS line, so nothing in the
pipeline distinguishes coding from UTR. Measured on chr8/9: **90.9%** of internal exons of
protein-coding transcripts are pure CDS, 3.8% pure UTR, 5.3% mixed. The residual **~9% UTR
dilution attenuates toward the null** and is stated, not hidden.

## 2. The intervention: ROLL THE ROUTING, KEEP THE VALUES

Roll one head's post-softmax attention along the **key** axis by δ; renormalise; leave the value
vectors, every weight, every other head and every MLP untouched. The head still emits a convex
combination of **real** value vectors with **its own** weight profile — just aimed δ nt off.

`δ ∈ {3, 6}` — still reads its own phase, from different positions. **IN PHASE.**
`δ ∈ {1, 2, 4, 5}` — reads a different phase. **OFF PHASE.**

δ = 0 runs the full code path and must reproduce the unhooked model. **Calibrated: 7.8e-09.**

---

## 3. ⛔ THE CENTRAL PROBLEM, AND WHY v1 WAS DEAD ON ARRIVAL

**The raw sawtooth is guaranteed by construction. It is not evidence of anything.**

Write the perturbation a roll delivers:

```
out_i^δ − out_i^0  =  Σ_k a_k ( v_{i+k+δ} − v_{i+k} )
E‖·‖²              =  Σ_τ w(τ) [ 2C(τ) − C(τ+δ) − C(τ−δ) ]
```

`C` = autocovariance of the value vectors along the sequence; `w(τ) = Σ_k a_k a_{k+τ}` = the
autocorrelation of **the head's own attention profile**.

For **i.i.d. values** (`C(τ) = σ²·1[τ=0]`) this collapses to

```
E‖ out^δ − out^0 ‖²  =  2σ² [ w(0) − w(δ) ]
```

**The sequence integrates out completely.** The perturbation size depends on the head's own comb
and on *nothing else*. A stride-3 comb has `w(δ)` large at δ ≡ 0 (mod 3) and zero otherwise — so
rolling it in phase barely moves its output, and rolling it off phase moves it a lot.

The elementary version: a comb on offsets {3,6,9,12,15}, rolled by 3, reads {6,9,12,15,18} —
**four of its five old positions.** Rolled by 1 it reads {4,7,10,13,16} — **none of them.** The
comb slides onto itself. That is all a raw sawtooth is.

**Consequences, all of them fatal to v1:**

1. **Any** stride-3 head sawtooths, on **any** input — coding, intronic, shuffled, random noise.
   Verified: `tests/test_phase_shift_memorials.py` reproduces the sawtooth on **white-noise value
   vectors** (off-phase 1.77 vs in-phase 0.95) and confirms the identity above to <6%.
2. The 42 lattice heads were **SELECTED for phase purity**. So `b_LATTICE > b_RANDOM` is a
   **selection effect**, not a finding.
3. `b_exon > b_intron` is likewise expected — coding DNA is period-3 autocorrelated, so `C`
   contributes on top of the comb term.
4. **Every one of v1's six pre-registered gates was therefore pre-satisfied.** A design whose
   signature is guaranteed by how the components were selected is **R7's shape** all over again.

### The fix: normalise by the perturbation actually delivered

```
damage  ≈  ∇L · Δout      ⟹      damage / ‖Δout‖  ≈  ‖∇L‖ · cos θ
```

The normalised quantity asks whether the model is hurt **more than the size of the perturbation
explains** — i.e. whether off-phase reads are *specifically* harmful rather than merely *bigger*.
It divides out **both** the comb-overlap term and the sequence-autocorrelation term.

**Every headline number is normalised. The raw sawtooth is reported and labelled
expected-by-construction.**

---

## 4. THE SECOND FIX: ONE HEAD AT A TIME

The pilot (n = 10) rolled all 42 lattice heads at once:

| δ | exon log-prob | headroom kept |
|---|---|---|
| 0 | −1.110 | 100% |
| 1 | **−1.495** | **−39%** |
| 2 | **−1.515** | **−47%** |

**Below chance.** The model is not damaged, it is *anti-predictive*: rolled by 1 it predicts the
base at i+1. Damage saturates, dynamic range is gone, and the experiment would return a **false
negative indistinguishable from a true one** — R13's exact failure, in a new costume. TOP-8 was
worse (−94%).

So **every head is rolled alone.** The model stays healthy, and every head becomes one point in:

## 5. THE MONEY TEST

> **normalised sawtooth amplitude `b_h`  vs  the head's PHASE PURITY, across all 96 heads,
> with layer fixed effects.**

`damage(δ) = c + a·δ + b·1[δ mod 3 ≠ 0]` on δ = 1…6. Design matrix rank 3 — `b` is identifiable
from `a` (asserted in the memorial tests). `a` absorbs any monotone displacement effect; no
monotone confound can dip at **both** 3 and 6.

| outcome | reading |
|---|---|
| **slope > 0** | the more phase-pure a head, the more the model *specifically* depends on it reading in phase, **beyond the perturbation that rolling it off-phase delivers**. **The lattice is USED.** |
| **slope ≈ 0** | every head that reads DNA shows this; the sawtooth is comb-overlap plus the genome's autocorrelation, and nothing SpliceBERT learned. **The lattice is inert routing** — and *this time the test has the power to say so.* |
| **slope < 0** | report as-is. Do not spin. |

Secondary: DiD = `b_norm(exon) − b_norm(intron)` per head.

## 6. Gates (in order; 1–2 license reading 3–5)

1. **CALIBRATION.** δ = 0 through the full roll path reproduces the unhooked model. **PASSED, 7.8e-09.**
2. **HEALTH.** Every reported head keeps ≥ 50% of intact headroom over chance at every δ.
   Unhealthy heads are **excluded and named**, not silently kept.
   Heads delivering ‖Δout‖ < 0.02 are excluded — 0/0 cannot be normalised.
3. **RAW sawtooth** reported, and labelled *expected by construction*. It is not a gate.
4. **NORMALISED `b`**, gene-grouped bootstrap (2000; resample genes, re-aggregate, **re-fit**).
5. **THE SLOPE** of `b_norm` on phase purity, layer-adjusted, bootstrap CI.

## 7. Controls, and how each can fail

| control | how it can fail |
|---|---|
| δ = 0 calibration | any hook bug → ≠ identity. (It did not fail. It could have.) |
| health gate | the pilot **already failed it** for LATTICE-42 and TOP-8, which is why the design changed. |
| intron region | can show the same normalised `b` as exons → kills codon-specificity. |
| non-lattice heads | can show the same normalised `b` → kills lattice-specificity. **In the pilot the healthy control arm (42 random non-lattice heads) showed a clean raw sawtooth — off-phase 0.087 vs in-phase 0.032 — and it survived normalisation (0.101 vs 0.036).** The control is already firing. |
| purity slope | can be flat. That is the null, and it is informative. |

## 8. REJECTED as unfalsifiable

- **A random-init model, same δ-scan.** Vacuous: it is already at chance, so it has no behaviour
  to damage, `b = 0` by construction. **A control that cannot fail is not a control.** Not run.

---

## 8b. THE PRIMARY STATISTIC — and why `damage / ‖Δout‖` is NOT it

Dividing damage by the perturbation delivered assumes **damage is linear in perturbation size**.
If it is convex — `damage ∝ ‖Δout‖^γ`, γ > 1 — then

```
damage / ‖Δout‖  ∝  ‖Δout‖^(γ−1)
```

and since a comb's off-phase ‖Δout‖ is ~2× its in-phase value **by construction**, the ratio is
larger off-phase *from convexity alone*. A copier, whose ‖Δout‖ is flat, shows nothing. The
result "b_norm(COMB) ≫ b_norm(COPIER)" would then reproduce itself with **no phase computation
anywhere in the model** — the control would fire, the algebra would look clean, and the answer
would be wrong. Same shape as R7 and R14: a statistic that cannot help but produce the expected
pattern.

**So the exponent is ESTIMATED, not assumed** (`analyse_comb_shift.py`):

```
log D_hd  =  a_h  +  γ·log N_hd  +  Σ_k b_k · offphase(δ) · 1[kind(h) = k]  +  ε
```

- `a_h` — head fixed effect (its own damage scale and downstream leverage)
- `γ` — the damage–perturbation exponent, **fitted**
- `b_k` — the phase effect for head-kind k: extra log-damage from reading **off phase, at
  matched perturbation size**

**THE CLAIM: `b_COMB − b_COPIER > 0`, in exons, with a head-level bootstrap CI excluding 0 —
and NOT in introns.** `damage/‖Δout‖` is retained as a secondary and is reported alongside `γ`,
so a reader can see whether the shortcut was safe.

## 9. Amendment log

| date | change | why |
|---|---|---|
| 2026-07-12 | **v2 → v3. The head set is now 7 COMBS vs 24 COPIERS vs DIFFUSE**, classified by *attention profile* (effective support + tooth congruence mod 3), not by the purity metric. | **R14.** The 42-head lattice does not exist: purity measures sharpness (`corr = −0.767`), and 24 of the 42 are single-offset copiers. |
| 2026-07-12 | **Primary statistic → the log-log model with `γ` fitted.** | `damage/‖Δout‖` is inflated by convexity of the damage–perturbation curve, which alone would reproduce COMB ≫ COPIER. |
| 2026-07-12 | **Self-attention restored after the roll.** | Otherwise a comb tooth slides onto the query's own **[MASK]** at one phase class (L1H10's teeth {−7,−4,−1,+2,+5} land on 0 at δ = 1,4,7) — a mod-3-correlated information loss that inflates the sawtooth for free. Also removes the separate "loss of self-attention" confound. |
| 2026-07-12 | **Bernoulli masking, not stride-7.** | A read at offset k is masked iff `k + δ ≡ 0 (mod 7)` — independent of the query, hence *systematic* per (head, δ). For L1H12 that is 1 masked read at every in-phase δ vs 0.5 off-phase. Bernoulli gives `P(masked) = 1/7` for every k and δ. |
| 2026-07-12 | **δ = 1…9 with a QUADRATIC phase-blind trend.** | With δ = 1…6 and a linear trend, `b` is a curvature detector, not a phase detector: the observed damage curve is visibly non-linear. |
| 2026-07-12 | **v1 → v2.** Raw damage demoted from headline to descriptive; **perturbation-normalised** damage is now the only reportable statistic. | The raw sawtooth is guaranteed by comb self-overlap (`E‖Δ^δ‖² = 2σ²[w(0) − w(δ)]`), reproduces on white noise, and is amplified by exactly the property the heads were selected on. All six v1 gates were pre-satisfied. |
| 2026-07-12 | **42-head roll → one head at a time.** | The 42-head roll drives the model **below chance** (−1.495 vs −1.386). Saturated damage cannot resolve a sawtooth. |
| 2026-07-12 | Added **protein_coding** filter; documented ~9% UTR dilution. | `gtf.py:113` never reads a CDS line; "exon (coding)" was never verified to be coding. |
| 2026-07-12 | Primary statistic changed from a 3-arm contrast (Δ+1, Δ+2 vs Δ+3) to a **δ-scan with a mod-3 indicator**. | A 3-arm contrast is confounded with displacement: +3 is a bigger shift than +1. Only dips at **both** 3 and 6 exclude a monotone confound. |

## 10. v1, superseded — kept because a pre-registration you can quietly rewrite is not one

v1 pre-registered six gates: calibration; health; `b_exon > 0`; `damage(3) < damage(2)`;
`b_LATTICE > b_MATCHED-RANDOM`; and `b_exon − b_intron > 0` — all on **raw** damage, with the 42
lattice heads rolled together. **Every one of those six would have passed on a model that had
learned nothing about codons.** The design was attacked before it was run, which is the only
reason it is being amended rather than retracted.
