# P3 shift diagnostic v2 — exploratory report

> **Tier: T2 — provenance.** Supporting record: preregistrations, decision logs, correction ledgers, audits.

**Status:** outcome-aware deterministic verification of an exploratory diagnostic. This is not a new
confirmed claim and does not establish the carrier's training origin or biological semantics.

## Question

Does L3H1's period-3 routing pattern remain fixed to absolute sequence classes `x%3`, move with the
acceptor, or respond specifically when the coding-exon register is broken?

The experiment used 24 coding examples from 23 genes and four paired `d={0,1,2,3}` perturbation families:
natural whole-window translation, deep-intron insertion, coding-exon insertion, and an upstream-intron
insertion sham. Every cell was evaluated under intact and C8 period-3-deleted position embeddings. See
[`EXPLORATORY_P3_SHIFT_DIAGNOSTIC_V1.md`](preregistrations/EXPLORATORY_P3_SHIFT_DIAGNOSTIC_V1.md) for the
design and its two transparent amendments.

## Instrument status

All corrected v2 gates pass:

- carrier construction error `1.49e-8`;
- fixed 1024-nt length and preserved donor `GT` / partner acceptor `AG` in every cell;
- protected site contexts unchanged and no new local `AG`, `GT`, or `GC` from insertion;
- live normalized attention;
- every carrier-deleted cell retains at least 76% of its corresponding intact masked-MLM headroom over
  chance, above the 50% floor.

The first smoke's `post_acceptor_sham` was invalid because it landed inside the next coding exon. It is
retained as a failed-control artifact and replaced in v2 by the upstream-intron sham.

## Results

### 1. The peak follows the site, not a fixed absolute class

Absolute class is defined exactly as `x%3 = 0,1,2,0,1,2,...`.

- Natural window translation advances the peak's absolute class by exactly `d mod 3` in 24/24 examples
  at every shift. Its donor- and acceptor-relative peak is exact in 23–24/24 examples.
- Deep-intron insertion fixes the donor and moves only the partner acceptor. The peak advances with the
  acceptor in 22/24 examples at +1 and +2 and 23/24 at +3; its donor-relative offset is preserved in only
  1/24. At +3, the absolute peak class returns to baseline in 24/24.
- The upstream-intron sham translates donor and acceptor together and reproduces the natural-translation
  tracking pattern, showing that generic insertion does not dislodge the site-attached peak.

This is a direct current-use result: L3H1 does not remain attached to one absolute `x%3` class. The
three-class pattern translates with the acceptor/content coordinate.

### 2. Coding-exon +1/+2 disrupts the pattern; +3 restores it

Mean intact coupling:

| edit | d=0 | d=1 | d=2 | d=3 |
|---|---:|---:|---:|---:|
| coding-exon insertion | 1.081 | 0.101 | 0.283 | 1.022 |
| deep-intron insertion | 1.081 | 1.112 | 1.063 | 1.036 |
| upstream-intron sham | 1.081 | 1.045 | 1.017 | 1.044 |

The exon effect survives the matched noncoding-insertion subtraction: the carrier-conditioned
in-phase-versus-off-phase coupling contrast is `+0.817 [0.396,1.378]`. It is positive in 20/24 examples,
and every leave-one-gene-out mean remains positive (`0.614–0.877`). +1 and +2 separately reduce intact
coupling by `−0.980 [−1.654,−0.485]` and `−0.799 [−1.390,−0.353]`; +3 changes it only
`−0.059 [−0.215,+0.045]`.

L3H1's acceptor-relative modular attention mass changes in parallel:

| edit | class 0 | class 1 | class 2 |
|---|---:|---:|---:|
| baseline | 0.058 | **0.844** | 0.098 |
| exon +1 | 0.096 | 0.452 | 0.451 |
| exon +2 | 0.351 | 0.519 | 0.130 |
| exon +3 | 0.063 | **0.837** | 0.100 |

The exact acceptor-relative maximum is retained in only 13/24 (+1) and 15/24 (+2), but returns in
23/24 at +3. Thus the off-register exon edit disrupts both coupling and the modular L3H1 routing pattern;
the in-register +3 edit restores both.

### 3. What is and is not licensed

**Supported as an exploratory current-use finding:** SpliceBERT is strongly sensitive to the three-base
register of coding-exon content upstream of a donor. L3H1's period-3 routing pattern is site-relative and
is disrupted by +1/+2 coding-exon insertions but restored by +3. Comparable intronic changes do not
reproduce the behavioral effect.

**Not established:** that the carrier was learned *because of* biological reading frame, or that P3 alone
mediates the exon-register effect. Full carrier deletion drives coupling from about `1.08` to `0.02`, so
the deleted behavior has little remaining headroom; disappearance of the frameshift effect in that state
cannot by itself prove mediation. The exonic insertion is also synthetic and changes coding content, not
just a hidden frame label.

## Next decisive test

1. Repeat the +1/+2/+3 pattern across several exon positions and with both insertions and deletions.
2. Replace binary carrier deletion with a graded carrier attenuation/restoration curve, requiring the
   off-register penalty to scale with carrier dose while baseline coupling remains measurable.
3. Add matched CDS, UTR, and noncoding-exon cohorts; the effect should be CDS-specific if it reflects coding
   register rather than generic triplet geometry.
4. Use frame-preserving synonymous recoding to distinguish an abstract register coordinate from native
   codon-composition statistics.
5. Freeze thresholds and run the final design on gene-disjoint, genuinely untouched examples.

## Artifacts

- `results/phase-lattice/p3_shift_diagnostic_v2.json` — complete model cells, gates, and bootstrap summaries.
- `results/phase-lattice/p3_shift_diagnostic_v2.mod3.json` — exact absolute `x%3` derivation.
- Corresponding `.manifest.json` files — checkpoint, data, command, code revision, and artifact hashes.
- `p3_shift_diagnostic_v2.manifest_correction.json` — transparent correction of the v2 manifest's
  inherited v1 experiment-ID label; no data or numerical value changed.
- v1 and both smoke artifacts are preserved for correction provenance.
