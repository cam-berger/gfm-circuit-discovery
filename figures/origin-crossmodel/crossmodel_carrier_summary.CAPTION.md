# Cross-model positional-carrier transfer (Phase 2)

**(a)** Leakage-aware carrier band power, expressed as Z = (band power at the carrier period)
/ (median band power at control periods), for the two SpliceBERT positional carriers measured
in three single-nucleotide transformer MLMs. In SpliceBERT (trained on spliced pre-mRNA,
ORF-rich) the period-3 (codon-scale) carrier sits at Z = 241 (exact-bin peak 289x the median
bin) and the ≈10.63 carrier at Z = 456 (peak 546x). In RNA-FM and RNA-BERT — both trained on
ncRNA, which is largely ORF-free — both carriers sit at Z ≈ 1, i.e. band power indistinguishable
from arbitrary control periods (all rank p > 0.24). period-3, the ≈10.63 carrier, and the DNA
nucleosomal signal ≈10.298 nt are kept as distinct quantities throughout.

**(b)** Injection calibration establishes that the RNA-FM null is *powered*, not merely
insensitive: a synthetic carrier injected into a phase-scrambled position table is detected at
100% by depth 5% of the P scale for both periods, while the depth-0 false-positive rate is 0.
The observed carriers correspond to depth 0. RNA-BERT (not shown) has a coarser floor (20%) due
to its smaller table (440x120), but likewise a 0 false-positive rate and a Z ≈ 1 observation.

Estimator: joint least-squares fit of a degree-6 position envelope plus a continuous-frequency
cos/sin pair (leakage-free at any period), validated to machine-precision amplitude recovery and
0.01-nt period localization. Source data: results/origin-crossmodel/canonical/carrier_pspectrum.json
(+ manifest). n = 200 phase-scramble nulls per calibration point; seed 0.

**Scope.** This is a P-level (position-table) result. The companion reader-scan + would-be-band
surgery (reader_and_surgery.json, reader_freqcontrol.json) confirm the absence propagates into
post-softmax attention. No behavioural/splicing arm transfers to ncRNA models.
