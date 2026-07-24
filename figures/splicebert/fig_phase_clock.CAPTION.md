# fig_phase_clock — a learned period-3 clock in SpliceBERT's position table

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Files:** `fig_phase_clock.svg` / `.png`
**Built by:** `scripts/splicebert/make_fig_phase_clock.py` (renders purely from
`results/phase-lattice/fig_phase_data.json`, so the artwork cannot drift from the result)
**Claim:** D21 · **Registry:** `studies/splicebert_partner_selection/CLAIMS.md`

---

## The claim

SpliceBERT's learned absolute position table `P` [1026 × 512] stores a **rank-2 period-3 rotary
phase code** — a codon clock. Six attention heads read it out. It is causally required for
coding-base prediction and for the donor–acceptor splicing circuit, and it is learned in stage-1
pretraining only with multi-species data.

The chain: **period-3 clock in `P` → L2H15's comb → the acceptor's address → L3H1's read →
donor–acceptor coupling.**

## The panels

**A. The clock is in the position table.** Fourier power of `P` along the position axis. 1026 =
3 × 342, so period 3 is an *exact* bin. The band (bins 339–345) carries **802× the median bin**,
is rank 9 of 513 modes, and is the only mode below period 10. The inset: **96%** of the carrier's
energy sits in two singular values — it is a rank-2 rotary code, not a scalar.

**B. Six heads read it as a period-3 wave.** `R(k) = mean_i (W_Q·h_i)·(W_K·h_{i+k})`, computed
from the weights and the position-conditional residual stream — no sequence content — with the
slow envelope removed. The curve is the **exact single-cosine interpolant**, not a smoothing;
see `fig_phase_wave.CAPTION.md` for why one cosine is all a period-3 integer signal can carry.
`A` is the clock amplitude in attention logits. The grey trace is the same head after the clock
is deleted from `P`, run through the **identical** de-enveloping — a control that can fail, and
does not (it keeps 1% of `A`; the energy-matched control band keeps 105%).

**L0H11 is drawn differently, and deliberately.** It is a **dual reader** (D22): it reads the
codon clock *and* a second carrier at ~10.69, so one cosine (the blue dashed line) visibly misses
its points. Its **complete wave function** — the orange curve — needs both:
`R(k) = env(k) + A₃·cos(2πk/3 + φ₃) + A₂·cos(2πk/10.69 + φ₂)`, with A₃ = 1.19 and
A₂ = 0.65. With it, 80.8% → **99.9%** of the ripple. See `fig_two_carriers` and
`fig_phase_wave` panel C.

**A NOTE ON WHAT "PHASE" MEANS HERE.** Both carriers are near-perfect **circles** (circularity
0.977 and 0.992) — rank-2 rotary codes — so **neither has a phase of its own**: every phase in
[0, 2π) is realised in *some* embedding direction. A head does not *read* a phase off the table;
it **chooses** one by choosing a direction in the carrier's plane. Verified in closed form from
`W_Q`, `W_K` alone: `phase = atan2(−Qs, Pc)` reproduces every comb head's measured phase to
**under a degree**, where a whole codon position is 120°.

**C. Delete it: attention falls to chance.** Empirical post-softmax attention, dominant phase mass.
Intact 0.794 → clock deleted **0.345** (chance = 0.333). The energy-matched control band leaves it
at 0.801.

**D. Delete it: coding and the circuit die.** Attenuating the clock by α. Coding-exon masked-base
prediction loses **0.159 nats** vs **0.042** in introns; the donor–acceptor coupling falls
**0.946 → 0.027**. Twenty-one energy-matched control bands damage exons and introns *equally*
(0.029 / 0.031, DiD −0.002; add-one p = 0.046). Monotonic in α.

**E. Learned only with multi-species data.** Period-3 enrichment by checkpoint: 477.7×
(multi-species 510nt), 226.2× (1024nt final), **6.3×** (human-only 510nt), 1.0× (random init).

## Caveats

**Provenance is confounded with data volume.** The multi-species checkpoint also saw more data.
The two released checkpoints cannot separate species diversity from sheer volume; a data-matched
human-only run would be needed. Panel E's headline is honest only as *"the clock is learned, and
the human-only checkpoint barely has it."*

**Panel C's metric is not "phase purity."** That statistic was retracted (R14) — it is maximised
by a head attending to exactly *one* position, so it ranked heads by sharpness, not by comb.
Panel C reports the dominant phase mass of the local attention profile, and the six heads shown
are the six that survive an explicit comb test (≥3 teeth, all congruent mod 3, span ≥ 6).

**Restoring L2H15 alone recovers 36.3% of the coupling** — against 0.7% mean / 4.2% max over 24
random single heads (p = 0.040). The six-head arm (0.963 vs random-6 max 0.939) is **suggestive
only**; do not quote it as significant.

## Superseded

This figure replaces `splicebert_donor_acceptor_binding.svg`, whose "content-addressed,
position-independent read" was retracted (**R15** — it was architecturally forced: BERT's
positions are absolute, so a relocated key/value pair attracts exactly the attention it attracted
at home, measured 9.9e-07). The address is **phase-coded**, not content-addressed.
