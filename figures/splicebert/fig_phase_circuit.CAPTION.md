# fig_phase_circuit — the donor–acceptor circuit, and the clock that addresses it

**Files:** `fig_phase_circuit.svg` / `.png`
**Built by:** `scripts/splicebert/make_fig_phase_circuit.py` (renders purely from
`results/phase-lattice/fig_phase_data.json`)
**Claim:** D21 · **Supersedes:** `splicebert_donor_acceptor_binding.svg` (retracted, R15)

---

## The mechanism

SpliceBERT decides which acceptor belongs to a given donor by **addressing it with a learned
period-3 positional clock**, not by a position-free content lookup.

1. The position table `P` [1026 × 512] carries a **rank-2 period-3 rotary phase code** — one
   oscillation every three nucleotides. It is added to every token.
2. **L2H15, the addresser**, reads that clock and builds the acceptor's key `K` — its *address*.
   The address is therefore **phase-coded**.
3. **L3H1, the read**, uses the donor's query `Q` to select that address, and attention transports
   the acceptor's payload `V` back into the donor's residual stream.
4. The donor's output is now partner-informed.

**Delete the clock** — 7 of 513 Fourier bins in `P` — and the circuit dies: the six comb heads'
attention falls to chance (0.79 → 0.35, chance 0.333), and the donor–acceptor coupling goes
**0.946 → 0.027**. Restoring **L2H15 alone** inside the clock-deleted model brings back **36%** of
it (24 random single heads: 0.7% mean, 4.2% max; p = 0.040). An **energy-matched control band**,
deleted instead, leaves the coupling at **102.5%**.

## Two things changed on 2026-07-13

**It is generated now, not hand-drawn.** The previous version was a hand-authored SVG, and it had
already **drifted**: it printed the combs' phase mass as 0.786 where the run says **0.794**. Every
number in the figure is now read from `fig_phase_data.json` at render time, so it cannot drift
again. (This is why the raw `coupling` / `exon` / `intron` values were added to that file — a
figure that quotes an absolute number must read it, never retype it.)

**The clock glyph is a real cosine.** It used to be a hand-drawn *sawtooth*. It is now L2H15's
**actual fitted wave** — `A·cos(2πk/3 + φ)` with the measured amplitude and phase, and dots at the
integer offsets where the model is actually sampled. This is not a beautification: a period-3
signal sampled at integers carries **exactly one resolvable sinusoid** (its second harmonic is
above Nyquist and aliases onto the fundamental), so the smooth curve *is* the data and the
jaggedness was the straight-line connector. See `fig_phase_wave.CAPTION.md`.

## Reading the artwork

`Box` = token state · `Pill` = attention head · `Circle` = projected vector. The five colours are
identities, each also carrying a text label, so nothing is encoded by colour alone (validated:
all checks pass; worst adjacent CVD ΔE 28.4 protan).

The sequence strip is **illustrative** — a schematic GT…AG intron, not a real locus.

## Caveats

**"Comb attention"** in the causal box means the *post-softmax* attention profile, which really is
a comb (discrete teeth on one residue class mod 3). The *pre-softmax* readout `R(k)` is the
cosine. Both are true and they are not in tension: the softmax of a cosine is a comb.

**The 36% mediation is a single-head result with a size-matched null.** The six-comb arm (0.963 vs
random-6 max 0.939) is **suggestive only** — do not quote it as significant.

**Provenance is confounded with data volume.** "Learned with multi-species data only" is honest as
far as it goes (6.3× in the human-only checkpoint vs 477.7× multi-species), but the multi-species
checkpoint also saw more data, and the released checkpoints cannot separate the two.
