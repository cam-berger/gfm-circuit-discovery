# Amendment P3-A1 — executable staging and health estimands

> **Tier: T2 — prospective provenance. Frozen on 2026-08-03 before the first
> checkpoint forward from the direct-edge P3 intervention family.**

This amendment makes the staging and health language in
`PREREGISTRATION_DIRECT_EDGE_P3_RELATIONAL_MEDIATION.md` executable. It does
not change the biological hypothesis, handedness, carrier band, supports,
coordinates, cohorts, target size, null families, thresholds, or decision
table. No direct-edge P3 model output had been produced when this amendment
was written.

## 1. Deterministic roles

For a salt and block ID, define the ordering key as the lowercase hexadecimal
SHA-256 digest of the UTF-8 bytes `salt + "\0" + block_id`; ties are impossible
because block IDs are unique.

- Sort the 120 development blocks with salt `direct-edge-p3-smoke-v1`. The
  first six are plumbing smoke and the remaining 114 are the final-code
  development preflight.
- For the target specificity subset, partition the 384 target blocks by
  strand. Within each strand, group by chromosome, natural-sort chromosomes,
  and SHA-sort each chromosome with salt `direct-edge-p3-specificity-v1`.
  Traverse the chromosome groups round-robin, taking at most one block from
  each nonempty chromosome per cycle, until 12 blocks have been taken. The two
  strand-specific selections form the 24-block subset.
- For matched heads, use all 15 non-L3H1 layer-3 heads and L2H15. For every
  other layer, SHA-sort that layer's heads with salt
  `direct-edge-p3-matched-head-v1` applied to the canonical label `L{layer}H{head}`
  and take the first three. This yields 31 controls for the six-layer,
  16-head checkpoint.

All selected IDs and heads are serialized into the frozen package. Algorithms
and serialized assignments must reconstruct each other exactly.

The round-robin specificity rule intentionally selects at most one block from
each chromosome-by-strand cell before cycling and, for this cohort, the first
12-per-strand cycle fills the subset. Chromosome-by-strand bootstrap strata are
therefore singletons on the 24-block subset and cannot be resampled. Full
development and 384-block target inference retains the parent protocol's
chromosome-by-strand bootstrap. The two 24-block specificity families instead
use a paired whole-block bootstrap stratified by strand (12 `+`, 12 `-`), seed
`20260806`, with the same 20,000 draws. This is frozen as an executable
resolution of an internally impossible parent-protocol combination, not chosen
from outcomes.

## 2. Execution chronology

The exact chronology is:

1. unit and synthetic integration tests;
2. six-development-block plumbing smoke;
3. freeze and independently audit the exact code, inputs, assignments,
   checkpoint, runtime, stage paths, and completed smoke;
4. rerun the byte-identical frozen code on the 114-block development preflight;
5. independently analyze and audit preflight;
6. only if every preflight authorization gate passes, run all 384 target blocks
   for baseline, carrier, site-health, and protected-probe rows;
7. analyze and audit the target carrier result;
8. only if the carrier instrument remains capable, run the 24-block signed-axis
   controls, then L3H1 A/V and local Q/K target stages and the 24-block
   matched-head family;
9. independently analyze and audit the complete mechanism package.

Smoke may report only request counts, tensor shapes, finite flags, coordinate
and tag validation, forward determinism, projection round trips, no-op errors,
and nonzero projection-change RMS. It must not serialize or print coupling,
`G`, `S_r`, `Omega`, mediation effects, Q/K effects, or a biological/mechanistic
verdict.

Every post-freeze stage reserves an immutable attempt before loading the
checkpoint and writes one immutable shard per block. Resumption accepts a shard
only if its schema, package hash, stage, block ID, row count, coordinates, and
content hash validate. A conflicting shard or stale temporary file fails
closed. An arm manifest is written only after the exact expected shard family
is complete. No output is overwritten and a resumed process cannot silently
create a new attempt.

## 3. Baseline and carrier row definitions

For every biological cell, the no-surgery baseline records donor masked margins
for intact and `AG->AC` sequence states and the intact acceptor masked margin.
The carrier stage records the same three margins for every
`q_D x q_A x center x orientation` cell. Coupling is always computed by the
analyzer as the same-row intact donor margin minus AC donor margin; runners do
not choose or filter rows by coupling.

Site-health floors are the frozen fresh-replication floors (zero for both donor
and acceptor). Carrier site health passes only if simultaneous lower bounds for
the donor-intact and acceptor-intact margin families exceed those floors. The
phase-blind carrier-coupling floor is `+0.02`.

The q-zero fidelity endpoint is the paired block-level difference between the
biological diagonal excess under the carrier family restricted to
`q_D=q_A=0`, equally averaged over all centers and orientations, and the
no-surgery baseline `Psi`. Its ordinary paired 95% interval must lie wholly in
`[-0.05,+0.05]`.

## 4. Protected-probe health

Protected-probe health uses the 24 frozen nucleotide positions already defined
by `phase_crossed_runner.py`: 12 donor-module positions and 12 acceptor-module
positions. For every block it uses the `p_D=0,p_A=0` constructed cell, all nine
carrier-register pairs, center `b=6`, and both orientations. Each carrier probe
is paired with the no-surgery masked prediction of the same nucleotide in the
same constructed sequence; it is not compared with a different source
sequence.

For each block and carrier-register pair, average the 48
position-by-orientation log-probability changes. The simultaneous lower 95%
bound across the nine register pairs must exceed `-0.10` nat. Accuracy retention
is the fraction of probes predicted correctly under no surgery that remain
correct under surgery, pooled only after forming block-level numerators and
denominators; every register pair must be at least `0.80`. A block with no
baseline-correct protected probe fails closed.

Carrier, baseline, and protected-probe logits must be finite. Attention
normalization and projection finiteness are checked on the smoke blocks and on
the deterministic first 16 preflight requests; each checked eager-attention row
must sum to one within `1e-5`.

## 5. Preflight channel and power rows

The 114-block preflight runs:

- the complete six-geometry learned-carrier factorial;
- baseline, site-health, and protected-probe rows;
- anchor-geometry L3H1 A/V cells; and
- anchor-geometry L3H1 local Q/K true-window and decoy rows.

The A/V and Q/K point-direction requirements are channel-alive preflight gates,
not evidence about the target. Projected target `MDE80` is computed from the
centered chromosome-by-strand stratified bootstrap distribution of the paired
block estimand and multiplied by `sqrt(114/384)`. It must be at most `0.05` for
carrier `Omega`, `D_total`, `D_route`, `D_value`, and `Omega_QK`. The `D_value`
MDE gate is a resolution requirement and does not require a nonzero point
estimate.

## 6. Local Q/K and specificity endpoints

For local Q/K, the `q_D=q_A=0` same-state run is the host. Donor-only captures
provide the L3H1 query delta at token 393; acceptor-only captures provide the
key delta at tokens 510--513. The complete true-window grid applies both
deltas, with a zero delta giving an exact no-op. The decoy grid applies the
same four key-delta rows at tokens 435--438 and never edits the true key window.
Values and all other projections remain those of the q-zero host.

The true-minus-decoy specificity estimand is paired blockwise as
`Omega_QK,true - Omega_QK,decoy`; its simultaneous lower 95% bound must exceed
`+0.02`. Matched-head specificity uses the same local construction and compares
L3H1 with each control on the same 24 blocks. The learned-versus-signed-axis
endpoint analogously compares learned-carrier `Omega` with each signed-axis
carrier `Omega` on those same blocks. The 31-member families remain separate.

For the target carrier decision, the previously named but underspecified
geometry and common-mode gates are executable as follows: all three
center-specific `Omega` endpoints and both orientation-specific `Omega`
endpoints must have simultaneous lower 95% bounds above zero within their
respective frozen families, and all three common-mode endpoints must have
simultaneous lower 95% bounds above zero. Strand- and donor-phase-specific
families remain mandatory reports rather than additional pass/fail filters.
These robustness gates supplement, and do not replace, the primary
`Omega > +0.05` and all-`S_r > 0` gates.

## 7. Target-stage stopping and negative scope

Failure of a target carrier gate stops route-localization execution and yields
the carrier-level classification allowed by the original decision table; it
does not authorize changing the instrument. A failed development gate stops
all target execution as `INSTRUMENT_UNRESOLVED`.

As in the parent preregistration, any negative conclusion is scoped to this
checkpoint, masked-donor direct-edge behavior, crossed-sequence context, and
the declared intervention family. It is not a biological no-go, and
"insufficiently embedded" remains a separate trajectory/contrast/training
hypothesis rather than a post hoc explanation.
