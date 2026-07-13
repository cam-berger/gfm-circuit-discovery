# fig_two_carriers — SpliceBERT's position table carries two clocks

**Files:** `fig_two_carriers.svg` / `.png`
**Built by:** `scripts/splicebert/make_fig_two_carriers.py`
**Data:** `results/phase-lattice/second_carrier.json` (from `run_second_carrier.py`) +
`fig_phase_data.json` · **Claim:** D22 · **Tripwires:** `tests/test_second_carrier_memorials.py`

---

## Where this came from

L0H11 was the ragged comb. One period-3 cosine explained only **80.8%** of its readout, against
≥98.4% for the other five, and its points visibly left the curve in `fig_phase_wave`. The user
called it: *"L0H11 seems to have another frequency embedded, hence why the phase isn't matched."*

It does. It is P's **other** Fourier mode.

## The panels

**A. Two high-frequency carriers, and nothing else.** P's spectrum has exactly two fast modes: the
codon band (bins 339–345, period 3) and **bin 96 — period 10.69, at 546× the median bin, rank 7
of 513**. Above 200× the median, bin 96 is the *only* fast mode outside the codon band; every
other strong mode in P is slow global drift (period ≥ 22) — the smooth positional ramp any learned
position table has.

**B. L0H11 needed a second cosine.** One cosine: **80.8%** of the ripple. Add period 10.69:
**99.9%**, max residual 0.81 → 0.08. The raggedness was never noise — it was a second tone.
*(Panel B is the ±12 window, where the raggedness was seen and which every other figure plots.
Over the wider ±60 window the two tones take it from 70.2% to 89.5% — there is further structure
at long offsets, and this figure does not claim otherwise.)*

**C. The data names the period.** The frequency was **not imposed**. A free scan over T ∈ [4, 30]
on the wide window (k = −60…+60, 121 offsets), with the envelope *and* the period-3 term fitted
jointly so the scan cannot merely re-find either, lands on **T = 10.70** on its own — against P's
bin 96 at 10.6875. The peak is a sharp tone (FWHM ≈ 1.1 nt), not the broadband smear a spike
would give. **L2H15's periodogram is flat** (A = 0.02, no peak anywhere in the scan): the negative
control behaves as it must.

**D. Each carrier deletes on its own.** The double dissociation, in L0H11:

| | period-3 amplitude kept | period-10.69 amplitude kept |
|---|---|---|
| delete bins 339–345 (codon clock) | **1%** | **107%** |
| delete bins 94–100 (second carrier) | **105%** | **4%** |

**E. But only the codon clock does splicing.** Deleting the codon clock costs **0.159 nats** in
coding exons vs **0.042** in introns (DiD +0.117) and destroys the coupling (**2.8%** kept).
Deleting the second carrier costs **0.029 / 0.031** — *equal*, DiD −0.002 — and leaves the coupling
at **102.5%**. Two carriers, two different jobs.

## The methodological finding, which matters more than the figure

**The study's "energy-matched control band" IS the second carrier.** It is bins 94–100, and bin 96
alone is **64% of that band's entire energy**. It was energy-matched to the codon clock *because a
real carrier is sitting in it*. Every "control band deleted" arm in the period-3 study has been
deleting a strong, widely-read, causally-verified positional carrier.

**This does not weaken the period-3 result — it strengthens it.** The control was better than
advertised: deleting a comparably strong learned carrier that 25 heads demonstrably read *still*
leaves coding prediction and the splicing circuit intact. The period-3 damage is not "deleting
things from P hurts."

But the band was never *inert*, and any description that implies it was is wrong.

## Caveats — read before quoting

**⚠️ EVERYTHING HERE IS PRE-SOFTMAX.** `R(k)` is the QK *logit* as a function of offset. The
largest period-10.69 amplitude in the model belongs to **L0H5 (9.07 logits)** — and L0H5 is one of
**R14's single-offset copiers**. Its R(k) genuinely oscillates, hugely, across the whole ±60
window; its *attention* is still a spike at offset +2 (which its R(k) maximum confirms), because
softmax of *(a big smooth wave + one sharp peak)* is the peak. **"25 of 96 heads read the second
carrier" is a statement about logits and nothing else.** Whether any of them *uses* it in
attention is untested. This is R14's mistake in a new costume, and the tripwire is
`test_a_logit_tone_is_NOT_an_attention_tone`.

**⚠️ 10.69 ≈ 10.5, the helical repeat of B-form DNA. That is a coincidence until someone tests
it.** It is a seductive story — helical phase, nucleosome positioning, DNA bendability — and this
repo contains **zero** evidence for it. What we know is only: the carrier exists; 25 heads carry it
in their logits; it is causally independent of the codon clock; deleting it costs ~0.03 nats
uniformly and does nothing to splicing. **What it is for is open.** The tripwire is
`test_the_helical_repeat_reading_is_a_HYPOTHESIS_not_a_result`.

**The head counts are logit-level with a causal filter** (amplitude above the deleted-condition
floor, free-frequency peak within 0.3 nt of 10.69, *and* collapse below 25% when its band is
deleted). A floor-only count gives 40 heads and is too generous — several of those peak elsewhere.

**Split-half reliability is essentially perfect** (16 sequences vs the other 16: ratio 0.999–1.000),
because R(k) is by construction the position-conditional, content-independent component. The
measurement is not in doubt. Only its meaning is.
