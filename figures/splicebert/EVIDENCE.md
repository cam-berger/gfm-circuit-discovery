# The evidence ladder

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

The point of this table is to stop evidence from silently migrating between levels. A component
can be **present in the position table (P)** without any head reading it; a head can **read it in
the QK logits** without that read surviving the **softmax (Attention)**; a head can carry the
signal in attention without its **OV output mediating** anything; and none of that is
**behaviour**. Each column is a separate experiment. Evidence never auto-promotes.

| Component | P | QK | Attention | OV / mediation | Behaviour | Interpretation |
|---|:--:|:--:|:--:|:--:|:--:|---|
| **Period 3** | ✓ | ✓ | ✓ | ✓ | ✓ | Distributed relative-positional routing coordinate supporting donor–acceptor coupling |
| **Donor–acceptor circuit** | — | ✓ | ✓ | ✓ | ✓ | Mechanistic information-routing circuit |
| **Multi-species ≈10.63** | ✓ | ✓ | ✓ | **null** (donor) | **bounded** (−0.033) | Real positional attention circuit; **not part of the donor circuit**; general function unresolved |
| **Human-only ≈10.2** | ✓ | provisional | **failed** | — | null | Does not form the proposed chromatin attention circuit |

## Reading the four rows

**Period 3** is the only component that carries all five. It is in the table (289× median bin,
circularity 0.977); six heads read it in QK with a predicted-to-measured phase error of 0.76°; it
survives the softmax (comb phase mass 0.68–0.92, falling to chance 1/3 when the band is deleted);
restoring **L2H15** alone recovers 36.3% of the lost coupling against a best-of-24 random
single-head null of 4.2%; and deleting it collapses donor–acceptor coupling 0.946 → 0.026 while
damaging exons (0.159 nats) far more than introns (0.042).

Post-figure refinements C34/C35 localize this use: site-only restoration recovers only ~2–4%, broad-span
restoration is required, and coupling recovery runs mainly through L3H1's attention pattern rather than
carrier-dependent changes in its values. C34's confirm split is gate-qualified by R24; these results do
not identify the carrier's training origin or exclude upstream effects on payload construction.

C36 closes the clean current-use gap with a nucleotide-invariant intervention on 943 untouched genes:
the learned P3 direction shows a material phase-aligned coupling contrast of **+0.141
[0.110,0.172] raw logit units** and exceeds all 19 same-geometry hidden-axis controls. The matched
CDS-minus-non-CDS-exon contrast is inconclusive, so the evidence reaches model behavior but not coding
specificity or biological/training origin.

**The donor–acceptor circuit** has no entry under P because it is *not* a positional component —
it is the computation the clock serves. The donor arrives at L3H1 as content; the acceptor is
found by address.

**Multi-species ≈10.63** reaches Attention cleanly, and its OV and Behaviour columns are **measured, not
blank**. Eight heads read it through the softmax with every control clean (carrier deletion →
27–44% retained; reciprocal band → 89–97%; low-leak → ~100%; **content shuffle → 93–107%**, the
positivity check a positional comb must pass). Both downstream interventions ran on 2026-07-23 (B1 → C37,
folded into C12): restoring the readers' carrier-intact values moves the donor coupling margin **below the
pre-declared `SESOI_v` = 0.02 raw logit units** (`Mv = DI−DD`, −0.009 → +0.013 across nine readers), and deleting the
carrier moves coupling by only **−0.033 [−0.061, −0.005]** raw logit units — small, *suppressive*, sub-SESOI.
**Do not read the three ✓ as five**: these are a non-detection and a bound, not positives. Equally, do not
read them as "no experiment" — and note the scope, because it is narrower than it looks: the preregistered
`U_info`-projected write test (`S_ov`, `Ω`) and its planted-positive calibration were **never computed**, so
whether this carrier writes onto the donor readout is **OPEN** (**R31**).

**Human-only ≈10.2** is the row that used to say something else. The carrier is genuinely in the
table (12.7× median), and 14 QK readers were preregistered on the strength of that — hence
"provisional". At Level A, **0 of 14** reach significance. The preregistered primary, L1H3, puts
**0.19%** of its attention mass in the tested offsets: ten times below the 2% floor required even
to enter the scan. The stopping rule fires. The chromatin-circuit claim is **closed at the controlled
attention gate**.

## Why the human-only null is a null and not a blind instrument

The identical estimator, with the identical frozen gates and the same 400 permutations, found
**eight** Level-A heads in the multi-species checkpoint — a model that differs from the human-only
one *only in training data*. An instrument that finds eight in one model and zero
in its twin is not blind.

The one real limit of the permutation budget, stated rather than buried: at 400 permutations,
BH-FDR requires **≥ 4 heads at the p-floor** before any head can be called significant, so a
*lone* reader would be undetectable. That is not what is happening here — the best human-only head
is **L2H9**, at `m_T = 0.556` with `p_exact = 0.796`: 318 of its 400 permutations match or exceed
it, so its apparent signal sits **inside its own permutation null** rather than being a near-miss the
budget could not resolve (`results/phase-lattice/fig_report_data.json`, `levelA.human_only`; the
earlier "`m_T = 0.001`" here was L1H3's band mass, a different quantity — **R38**).

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
