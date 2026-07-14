# SpliceBERT figures — the state of the record

**Everything here is GENERATED.** The four publication figures read **one file**,
`results/phase-lattice/fig_report_data.json`, which is itself built by
`scripts/splicebert/make_fig_report_data.py` and **self-checks every recomputed number against
the recorded run before it writes anything**. A figure therefore cannot drift from the result it
claims to show. Each has a `.CAPTION.md`; **the narrative lives there, not in the artwork** — the
figures are held to a few words per component.

## The publication set

| figure | script | what it shows |
|---|---|---|
| `fig1_summary` | `make_fig1_summary.py` | the whole mechanism: two carriers, one with a job, one without |
| `fig2_oscillators` | `make_fig2_oscillators.py` | the spectrum, the carrier geometry, the orthogonality, and why "10.67" was a grid label |
| `fig3_clock_circuit` | `make_fig3_clock_circuit.py` | **the causal figure** — deletion, dose-response, and mediation against size-matched nulls |
| `fig4_carrier_circuit` | `make_fig4_carrier_circuit.py` | the ≈10.63 attention circuit: real, specific, unexplained |

**Start with [`EVIDENCE.md`](EVIDENCE.md)** — the evidence ladder (P → QK → Attention →
OV/mediation → Behaviour). It exists to stop evidence from silently migrating between levels.

## The one-paragraph version

SpliceBERT's learned position table `P` is **a small bank of oscillators**. Two are fast: a
**period-3 codon clock** and a **carrier at ≈10.630**. Both are near-perfect **circles**
(circularity 0.977 and 0.992) — rank-2 rotary codes — and therefore **have no phase of their
own**; each head *chooses* a phase by choosing a direction in the carrier's plane. Their planes
are **orthogonal** (subspace overlap 0.0003). The codon clock **carries the splicing circuit**:
deleting it collapses donor–acceptor coupling 0.946 → 0.026, and restoring L2H15 alone recovers
36.3%. The ≈10.63 carrier is **read by eight heads through the softmax**, with every control
clean — and **what it is for remains unknown.**

## Three claims, and nothing more

1. **SpliceBERT contains a donor–acceptor information-routing circuit.**
2. **A learned period-3 positional clock helps that circuit address coding sequence in codon
   phase.** (1 and 2 are parts of *one* mechanism.)
3. **A separate ≈10.63-nt positional carrier forms a genuine attention circuit, but its origin
   and function remain unknown.**

Call 10.63 a **carrier** or a **periodicity** — **never a phase**. The carrier plane contains
every phase; individual heads select one through their QK projections.

## Numbers that are RETIRED and must not be quoted

- **"10.67"** was never a measurement. It is the bin label 512/48. The sub-bin fit gives
  **10.630** (10.607 on the 510-row grid, 10.652 on the 1026-row grid). See Figure 2D.
- **"10.58 ± 0.35"** (Kogan) is not a measured constant. It is the spread of nine noisy
  periodogram maxima from 42-nt windows, which cannot locate a wavelength better than ±4 nt.
- **The ≈10.63 carrier is NOT the nucleosomal signal.** The nucleosomal periodicity measured in
  DNA is **10.298** (z = +249). The two differ by 0.33 nt against a fit uncertainty of ±0.005.
  This is a *powered refutation* (R17), not an open question.
- **The human-only ≈10.2 chromatin circuit is CLOSED at QK** (D35). 0 of 14 preregistered
  readers reach Level A; the preregistered primary L1H3 puts 0.19% of its attention mass in the
  tested offsets — ten times below the floor needed even to enter the scan.

## Conventions

- **Exact add-one permutation p-values**, floor 1/(400+1) = **0.0025**. `p = 0.0000` is never
  printed.
- **BH-FDR q = 0.05** across the whole scanned head family, not a hand-picked subset.
- **Mediation nulls are size-matched** — a one-head restoration is scored only against random
  one-head sets. Pooling them would be a category error.
- **The palette was computed, not chosen.** Every pair of colours co-occurring in a panel was
  checked for separation under simulated deuteranopia, protanopia and tritanopia (ΔE ≥ 12). The
  palette used by the earlier figures **failed** this on the pairs these figures actually compare
  — blue vs grey came out at ΔE = 0.9 under deuteranopia, and those are "intact" vs "deleted".
  See the docstring of `scripts/splicebert/figstyle.py`.

## Earlier figures (kept, superseded in part)

`fig_phase_circuit`, `fig_phase_clock`, `fig_phase_wave`, `fig_two_carriers` predate the
publication set. They remain correct on the period-3 mechanism but **quote "10.67"** and predate
Level A. Prefer the numbered figures above.

`splicebert_donor_acceptor_binding.svg` — asserts the **retracted** "content-addressed read"
(R15). Superseded by `fig1_summary.svg`. See `HAND_ASSETS.md`.
