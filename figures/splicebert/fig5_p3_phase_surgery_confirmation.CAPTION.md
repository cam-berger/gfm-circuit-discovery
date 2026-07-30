# Figure 5 — Nucleotide-invariant surgery confirms phase-aligned period-3 use

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**A.** Mean donor–acceptor coupling under local shifts of the learned period-3 position component.
Shifts `{3,6,9}` preserve modulo-3 register; adjacent offsets move it off register. Off-register surgery
reduces coupling only when applied inside CDS blocks, not under the identical upstream-intron control.
Ribbons are ±1 gene-level SE and are descriptive; inference uses the paired gene contrast in B.

**B.** The preregistered primary estimand, CDS modulo recovery minus upstream-intron modulo recovery, is
**+0.141 [0.110,0.172] logit units** across 943 untouched genes. The complete interval exceeds the
frozen 0.05-logit-unit SESOI.

**C.** The learned period-3 direction exceeds all 19 pre-seeded signed-hidden-axis controls. The controls
preserve the period-3 spatial waveform, Fourier support, spatial Gram matrix, row norms, block geometry,
and perturbation RMS; only alignment with learned hidden features changes. Paired learned-minus-control
excess is **+0.136 [0.051,0.222]**; exact add-one rank p=0.05.

**D.** The matched CDS-minus-non-CDS-exon boundary is **+0.006 [−0.080,+0.093]** across 52 pairs and is
inconclusive. The figure therefore supports present-day phase-aligned use of SpliceBERT's learned period-3
direction in CDS blocks, not coding specificity, reading-frame sensing, or training origin.

**Sources:** immutable independent analysis
`results/phase-lattice/p3_phase_surgery_confirmation_v1/confirmation_analysis.json` and frozen package
`results/phase-lattice/p3_phase_surgery_confirmation_package_v1.json`. Exact plotted data and source
hashes are recorded in the adjacent figure provenance and
`results/phase-lattice/p3_phase_surgery_confirmation_v1/figure_data.json`.
