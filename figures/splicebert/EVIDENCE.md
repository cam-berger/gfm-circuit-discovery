# The evidence ladder

The point of this table is to stop evidence from silently migrating between levels. A component
can be **present in the position table (P)** without any head reading it; a head can **read it in
the QK logits** without that read surviving the **softmax (Attention)**; a head can carry the
signal in attention without its **OV output mediating** anything; and none of that is
**behaviour**. Each column is a separate experiment. Evidence never auto-promotes.

| Component | P | QK | Attention | OV / mediation | Behaviour | Interpretation |
|---|:--:|:--:|:--:|:--:|:--:|---|
| **Period 3** | ✓ | ✓ | ✓ | ✓ | ✓ | Coding-phase clock supporting donor–acceptor routing |
| **Donor–acceptor circuit** | — | ✓ | ✓ | ✓ | ✓ | Mechanistic information-routing circuit |
| **Multi-species ≈10.63** | ✓ | ✓ | ✓ | **?** | **?** | Real positional attention circuit; **function unknown** |
| **Human-only ≈10.2** | ✓ | provisional | **failed** | — | null | Does not form the proposed chromatin attention circuit |

## Reading the four rows

**Period 3** is the only component that carries all five. It is in the table (289× median bin,
circularity 0.977); six heads read it in QK with a predicted-to-measured phase error of 0.76°; it
survives the softmax (comb phase mass 0.68–0.92, falling to chance 1/3 when the band is deleted);
restoring **L2H15** alone recovers 36.3% of the lost coupling against a best-of-24 random
single-head null of 4.2%; and deleting it collapses donor–acceptor coupling 0.946 → 0.026 while
damaging exons (0.159 nats) far more than introns (0.042).

**The donor–acceptor circuit** has no entry under P because it is *not* a positional component —
it is the computation the clock serves. The donor arrives at L3H1 as content; the acceptor is
found by address.

**Multi-species ≈10.63** stops at Attention, and the two question marks are the honest state of
the work. Eight heads read it through the softmax with every control clean (carrier deletion →
27–44% retained; reciprocal band → 89–97%; low-leak → ~100%; **content shuffle → 93–107%**, the
positivity check a positional comb must pass). No OV-mediation experiment has been run, and no
behavioural effect has been demonstrated. **Do not read the three ✓ as four.**

**Human-only ≈10.2** is the row that used to say something else. The carrier is genuinely in the
table (12.7× median), and 14 QK readers were preregistered on the strength of that — hence
"provisional". At Level A, **0 of 14** reach significance. The preregistered primary, L1H3, puts
**0.19%** of its attention mass in the tested offsets: ten times below the 2% floor required even
to enter the scan. The stopping rule fires. The chromatin-circuit claim is **closed at QK**.

## Why the human-only null is a null and not a blind instrument

The identical estimator, with the identical frozen gates and the same 400 permutations, found
**eight** Level-A heads in the multi-species checkpoint — a model that differs from the human-only
one *only in training data*. See Figure 4A. An instrument that finds eight in one model and zero
in its twin is not blind.

The one real limit of the permutation budget, stated rather than buried: at 400 permutations,
BH-FDR requires **≥ 4 heads at the p-floor** before any head can be called significant, so a
*lone* reader would be undetectable. That is not what is happening here — the best human-only head
has m_T = 0.001.

## Statistical conventions used throughout

- **p-values are exact add-one permutation p-values**, `p = (hits + 1)/(nperm + 1)`. With 400
  permutations the floor is **0.0025**. **`p = 0.0000` is never printed** — 0 hits in 400 draws
  bounds a p-value, it does not measure one.
- **Multiplicity**: BH-FDR at q = 0.05, applied across the **whole scanned head family** (79
  heads multi-species, 75 human-only), not across a hand-picked subset.
- **Mediation nulls are size-matched.** A one-head restoration is scored only against random
  *one*-head sets; a six-head restoration only against random *six*-head sets. Pooling them would
  be a category error — the one-head nulls top out at 4.2%, the six-head nulls reach 93.9%.
- **Preregistered vs exploratory** is recorded per result in
  `results/phase-lattice/fig_report_data.json` under `status`. The 14 human-only QK readers were
  preregistered before Level A was measured. The eight multi-species Level-A heads were found by
  an *unpreregistered* scan of all heads clearing the band-mass floor, and are corrected for
  multiplicity across that family.
