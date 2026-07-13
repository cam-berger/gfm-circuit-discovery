# fig_phase_wave — the period-3 comb is a single cosine

**Files:** `fig_phase_wave.svg` / `.png`
**Built by:** `scripts/splicebert/make_fig_phase_wave.py` (renders purely from
`results/phase-lattice/fig_phase_data.json`, so the artwork cannot drift from the result)
**Maths:** `src/gfmcd/splicing/wave.py` · **Tripwires:** `tests/test_wave_memorials.py`

---

## The claim

The six comb heads' readout of the position clock,
`R(k) = mean_i (W_Q·h_i)·(W_K·h_{i+k})`, has always been drawn as a jagged period-3 comb. It
is not jagged. It is a **cosine**, and the jaggedness was an artefact of joining the integer
samples with straight lines.

This is not a smoothing. **A period-3 signal sampled at integers can carry exactly one
sinusoid.** Nyquist for an integer-sampled signal is 0.5 cycles/sample; the clock sits at 1/3.
Its second harmonic would sit at 2/3, which is *above* Nyquist and aliases straight back onto
the fundamental — `cos(2·2πk/3) ≡ cos(2πk/3)` for every integer `k`. Three real samples per
period carry exactly three real degrees of freedom, and DC + (amplitude, phase) uses all three.
So the curve through the dots is the **exact trigonometric interpolant**, with zero residual.

## The panels

**A. The jaggedness was the connector.** L2H15 (the addresser), raw `R(k)` at k = −12…+12.
The pale grey line is what we used to draw — straight lines between the samples. The blue curve
is `envelope(k) + A·cos(2πk/3 + φ)` on a continuous grid. Same dots. The corners were never in
the data.

**B. Six heads, three codon positions.** Each head's wave, amplitude-normalised, on a shared
offset grid; the faint columns mark the offsets that head peaks on. The six heads fall into
three groups of two, and the wave slides one nucleotide per group. *(See the caveats — this is
a description of the six survivors, not a claim about design.)*

**C. Delete the clock and the wave is gone.** The clock amplitude `A` per head, in attention
logits. Deleting the period-3 band from the position table (7 of 513 Fourier bins) leaves
**1.1%** of it. Deleting an **energy-matched control band** instead leaves **104.8%**.

## What is fitted, and what is not

`R(k)` also carries a slow envelope (attention decays with |k|) which is *not* the clock. Only
that envelope is fitted — as a low-order polynomial, jointly with the cosine by OLS, because
`cos(2πk/3)` is not exactly orthogonal to a linear trend over a finite window and a two-step fit
would bias `A`.

The envelope model is cross-checked against a **model-free** estimator that assumes nothing about
its shape: a 3-tap moving average has an exact zero at period 3 (`1 + 2cos(2π/3) = 0`), so it
annihilates the clock identically and leaves the envelope; the high-pass residual is then the
clock at gain 1. The two agree to **100.4%** (per head, within 2.2%). If they had diverged, the
envelope would have been stealing amplitude and `A` would be fiction.

| head | A (logits) | model-free A | codon position | one cosine explains | clock deleted | control deleted |
|---|---|---|---|---|---|---|
| L0H0  | 1.01 | 1.02 | 1 | 100.0% | 1.5% | 104.5% |
| L0H11 | 1.21 | 1.21 | 0 |  80.8% | 2.7% | 102.7% |
| L1H10 | 1.78 | 1.79 | 2 |  99.2% | 0.9% | 106.1% |
| L1H12 | 2.14 | 2.14 | 1 |  99.9% | 0.6% | 106.0% |
| L2H2  | 1.11 | 1.13 | 0 |  98.4% | 1.5% | 104.0% |
| L2H15 | 1.82 | 1.82 | 2 | 100.0% | 0.4% | 104.2% |

## Caveats — read these before quoting anything above

**The de-enveloping cannot manufacture a wave.** Fed a clock-free signal (a ramp, a parabola, a
sigmoid, or an adversarial slow wave at period 4 or 6), the estimator hallucinates at most
**A = 0.08**. The real amplitudes clear that floor by **13–20×**. And once the clock is deleted,
what is left (max A = 0.033) falls *back to the floor* — indistinguishable from a signal that
never had a clock. This is why the grey "clock deleted" trace in `fig_phase_clock` panel B is run
through the identical de-enveloping: it is a control that **can** fail, and does not.

**L0H11 is the ragged one.** One cosine explains only 80.8% of its ripple, against ≥98.4% for the
other five. It carries genuine non-clock structure, its points visibly leave the curve, and the
slow undulation left in its clock-deleted trace is envelope, not period 3 (its period-3 amplitude
is 2.7% of intact). Shown, not hidden.

**The integer phase-lock is PARTLY SELECTED FOR — it is not a discovery.** Every head's wave
peaks within 0.11 nt of an integer offset (five of six within 0.02 nt). That looks striking, and
it is largely an artefact of how these six heads were chosen: the comb classifier
(`run_comb_shift.py:130`) requires every attention tooth to be congruent mod 3, and a head whose
wave peaked at a *half-integer* offset would split its mass across two adjacent offsets — which
sit in different residue classes — and could never have passed. The phases are integer-locked
partly **by construction**. Report the phase as a descriptor of a head. Never as evidence.
(This is R14's mistake, and `test_wave_memorials.py` holds the tripwire.)

**The 2-2-2 split across codon positions is DESCRIBED, not claimed.** The classifier is
indifferent to *which* residue a head reads, so the balance is not selected for — but with six
heads a perfectly balanced split has p = 90/729 = **0.12** under random assignment. It is
unremarkable. Do not dress it up.

## What this figure does not add to the record

Nothing. It is a **re-rendering** of D21, not new evidence. The causal claims — that deleting the
clock collapses the heads' attention to chance, costs 0.159 nats in coding exons against 0.042 in
introns, and destroys the donor–acceptor coupling (0.946 → 0.027) — are unchanged, and live in
`studies/splicebert_partner_selection/CLAIMS.md`.
