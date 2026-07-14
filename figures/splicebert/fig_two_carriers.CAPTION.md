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

---

## What the second carrier IS — the evidence hierarchy (D24)

| claim | status |
|---|---|
| Transformers learn oscillator banks | **established** — RoBERTa-base, trained on **text**, has one |
| SpliceBERT selectively amplifies a **sharp** ~10.67 tone | **established** |
| L0H11 uses that tone | **established** (D23; post-softmax, not just in the logits) |
| Genomic/chromatin statistics **selected** the tone | **plausible, UNTESTED** |
| The model uses it to represent chromatin | **unproven** |
| The 512-grid **quantized** a broad nucleosome-scale band into bin 48 | **new, concrete, testable** |

**The bank is a Transformer artefact; the selected tone is not.** RoBERTa's 9–12 band is
**broadband** — it peaks at period 11.91, and at 10.67 it is *not even a local maximum*; only 20%
of its power sits in one bin. SpliceBERT's is a **sharp tone**: **76%** of the band in ONE bin,
15× over its neighbours. (BERT-base, trained mostly at length 128, has neither.)

**The carrier is ~10.67, and "10.69" is a Fourier grid, not a measurement.** It is learned in
**stage 1**: bin 48 of the 512-row table (period **10.6667**) at 370.5× the median; human-only:
**2.4×** (nothing); final 1024nt: bin 96 of 1026 at 545.8×; **carrier-plane overlap stage-1 vs
final = 0.997** — the same object, carried through the table extension. Do not quote either value
to two decimals as though it were exact.

⚠️ The selection is **model/data-specific**. It could still be driven by a *non-biological*
property of training — fixed length, optimisation, corpus size, repeats. **"Biological" is not
written until the bridge is crossed.**

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

**⚠️ THE CHROMATIN READING IS OPEN, NOT REFUTED (R16).** An earlier version of this caption said
it was "not supported". **That was retracted.** The WW/SS helical-phasing null came from an
estimator whose **detection floor is 5–10% modulation**, measured by injecting a known ~10.5-nt
periodicity into real sequence — against a literature effect of **a few percent**. The instrument
was validated only *after* it gave the expected answer. It fails. So is Assay 1, the unanchored
corpus spectrum: it is a **quadratic** detector (SNR ∝ depth²·√N), and at 120 Mb it can exclude
only large, globally incoherent effects. **Neither null is evidence against chromatin.**

The **α-helix** reading *is* refuted, and that one is powered — its positive control is the codon
clock through the identical model-side assay (DiD +0.1334 [+0.1118, +0.1561]; the carrier's is
+0.0122 [−0.0017, +0.0262]). And "the two clocks sharpen the address" is ruled out by **algebra**:
one added cosine of depth `a` can concentrate a comb by at most `1 − exp(−a²/4)` ≈ 8% at the
measured depth.

**The experiment that CAN see it** is the dyad-anchored assay (D25) — a **linear** detector
(SNR ∝ depth·√N), built and power-calibrated: 80% power at 2% needs only ~3,000 well-positioned
dyads. The binding constraint is **dyad precision, not count** (2 bp of jitter costs the same as
discarding four fifths of them), so it needs a **chemical-cleavage** map, not MNase-seq.

**The head counts are logit-level with a causal filter** (amplitude above the deleted-condition
floor, free-frequency peak within 0.3 nt of 10.69, *and* collapse below 25% when its band is
deleted). A floor-only count gives 40 heads and is too generous — several of those peak elsewhere.

**Split-half reliability is essentially perfect** (16 sequences vs the other 16: ratio 0.999–1.000),
because R(k) is by construction the position-conditional, content-independent component. The
measurement is not in doubt. Only its meaning is.
