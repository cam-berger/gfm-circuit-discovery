# Statistical calibration specification

**Normative execution specification; frozen by the accompanying source-bound receipt.** This is a normative companion
to [PREREGISTRATION.md](PREREGISTRATION.md), to be frozen with the chosen
cohort. It tests software/statistics capability, not model behavior or biological
realism. Nothing here forecasts actual CC/max variance.

## Execution path and independent streams

Use actual selected N and stratum sizes, synthetic four-nucleotide logits, and the
production score-row serialization, reduction, margins, inference and classification.
Generate native, bridge and health rows with declared known truths. Production
numeric/phase/cell identity checks must run; fixture truth fields cannot stand in
for inference. Bind all sources and NumPy/bit-generator versions.

Use `SeedSequence(2026101601).spawn(256)` for development;
`SeedSequence(2026101602).spawn(1000)` for held-out null realizations; and
`SeedSequence(2026101603).spawn(256)` for paired plants. Development can reveal code
defects; it cannot tune amplitudes, thresholds, or the held-out acceptance rules.
Every analysis retains bootstrap 20,000/20260803 and the 99 protocol pseudo seeds.
Shared bootstrap indices/control coefficients may be cached without changing the
distribution or order. No selecting a new seed after failure.

## Known signal, noise, and raw-logit construction

Let `H` have diagonal 2/3 and off-diagonal −1/3. Adding `delta*H` increases Psi and
each directional contrast by exactly delta and leaves the phase-blind mean fixed.
Set `sigma_Psi = 0.035*sqrt(N)/(1.96+0.8416212335729143)`; this is a deliberately
capable fixture with nominal scalar MDE≈0.035, not an empirical noise estimate.

For each block draw independent row and column nuisance terms from N(0,0.10²),
three independent 3×3 standard-normal matrices Z0/Z1/Z2, and a block multiplier
chosen uniformly from `{0.5,1.5}/sqrt(1.25)`. Interaction residuals have marginal
SD `sqrt(2)*sigma_Psi`, multiplied by that block multiplier. CC uses Z0, AC uses
`0.9*Z0+sqrt(0.19)*Z1`, and AT uses `0.95*ACstandard+sqrt(0.0975)*Z2`. The same
row/column terms enter every operator. These correlations exercise paired inference.
Null coupling offsets are 0.10 for CC and 0.08 for AC/AT; positive fixtures use
0.05 for each. Bridge mean is zero with block SD 0.01. Native coupling mean is
0.30 with block SD 0.02. Intact margins satisfy the declared health floors.
Protected log-probability deltas have mean zero and SD 0.01, with identical source
and construct correctness counts and valid finite log normalizers. Tag all rows synthetic.

Convert desired max coupling `C_o` to raw logits: intact G=3, maximum alternative
logit zero, alternative-logit mean −K0, K0=2. Corrupted G=`3−C_o`; alternative mean
is −Ko, with `Ko=2−d_o`. Let `d_o=0.01*H+u_o`, with independent cell noise
`u_o~Uniform(−0.005,0.005)`; AC and AT share u, while CC has its own. A triple
`(0,−1.5Ko,−1.5Ko)` has the required mean/max; randomly permute alternative-base
identities. In a fixed 10% of requests instead use a randomly permuted
`(0,0,−3Ko)` triple to exercise ties. Then `C_linear=C_max+d_o`.
Record every known truth and the fixed request membership used for tie cases.
For coverage comparisons after random relabeling, use the generating population
truths, not noisy sample means.

Separate deterministic probes include unequal rivals, switched argmax, all-rival
ties and K=0. Equal-rival cases alone cannot validate the max readout. Synthetic
protected-health fixtures validate recorder arithmetic, not real-model health.

## Held-out nulls and coverage

For each of 1,000 null realizations independently permute acceptor labels with a
uniform S3 permutation in each block, carrying every state/arm together. The first
500 use only additive row/column terms plus a common N(0,0.10²) block offset; the
remaining 500 include the heterogeneous interaction residuals above. Never shuffle
individual cells or separate arms. The 99 balanced controls are generated separately
by the production procedure; do not reuse the generating permutation as a control.

Count any material, only, or detectable primary preference classification as a
full-decision false positive. Require **K≤38/1,000**. At the boundary, the exact
one-sided 95% Clopper–Pearson upper FPR bound is 0.0495129. This bounds the complete
decision at 5%; it does not change the pseudo-phase criterion from p≤0.01 to 0.05.
With K=0 report upper bound 0.00299125, not a proved zero rate.

Separately audit six nominal 95% CI families:

1. Scalar primary Psi.
2. Five simultaneous directional intervals.
3. Two simultaneous mismatch intervals.
4. Four simultaneous paired corruption/readout diagnostics.
5. Historical linear Psi_AC−Psi_AT interval.
6. Historical linear phase-blind AC−AT interval.

For each family count a failure when **any** true endpoint is outside its interval.
Use absolute tolerance 1e-10 only for this known-truth coverage comparison to avoid
roundoff failures on exact-zero fixtures. Production thresholds remain unchanged.
Require each noncoverage count **≤71/1,000**: the exact upper-tail binomial audit
of a 5% noncoverage rate at alpha=0.01/6 has per-family false-alarm probability
0.00154159 and union bound ≤0.00924953. Report observed rates and exact intervals;
acceptance is a calibrated coverage check, not proof that a measured 7.1% rate is
5%. A low full-classifier FPR cannot hide a broken interval family.

## Planted sensitivity and failure branches

Evaluate each of the 256 plant streams twice using identical noise, without the
generating null relabeling:

- `delta=0.05`: primary two-sided 95% lower bound must exceed zero.
- `delta=0.15`: the complete material-positive classifier must pass, including all
  gates and historical sensitivity conditions (either preferential or only).

Each success count must be **≥216/256**, with one-sided exact 95% lower power bound
0.80151. Do not require a true 0.05 plant to have its lower bound above 0.05; that
would test a boundary rather than sensitivity. At actual 90% success probability
the expected false failure of each acceptance gate is approximately 0.00179.

Deterministic branch fixtures must traverse production reductions and return the
prescribed result for inversion, bounded null, detectable sub-SESOI, conflicting
directions, failure of one mismatch equivalence interval, a tied pseudo statistic,
misassigned phase, duplicate/missing cells, and failed sensitivity/health gates.
A high-noise fixture with observed MDE>0.05 must return `INSTRUMENT_UNRESOLVED`;
record its actual MDE and require the branch to fail, rather than hard-coding PASS.

Any failed stochastic calibration stops real CC execution. Preserve all counts,
exact intervals, failure categories, seeds and raw fixture sufficient statistics.
Amendments to fix an instrument remain pre-outcome only and are logged; a new seed
does not erase a failed instrument. These calibrations have not yet been run.
