# Preregistration — SpliceBERT positional boundary coherence

> **Tier: T2 — provenance.** Frozen design; no new model outcome is reported here.

**Status: FROZEN DESIGN 2026-08-10.** The scientific question, estimands, thresholds, test families,
authorization graph, and outcome language below are frozen before any outcome from this successor study.
The executable target package is **not yet frozen**: implementation may be tested on synthetic fixtures and
the declared development role, but no target or Phase-III reserve may be opened until a freeze package binds
the final code and inputs and an independent conformance audit passes. A change after this date is an
additive, dated amendment; this file is never silently edited to match observed behavior.

## 1. Question, hypotheses, and claim boundary

### 1.1 Primary question

At the official base `SpliceBERT.1024nt` checkpoint, after prespecified balance of sequence features that
covary with annotated coding-boundary phase, does the learned period-3 position-table component causally
support the remaining preference for phase-matched donor/acceptor boundaries?

If so, can a new span-wide period-3 register intervention relocate that preference and can a frozen internal
path mediate and rescue the relocation?

### 1.2 Primary and ancillary hypotheses

- **H3-behavior (primary):** the clean composition-balanced contrast is materially positive and
  health-preserving period-3 attenuation materially reduces it relative to intervention-matched nulls.
- **H3-register (primary successor):** a span-wide differential period-3 register intervention cyclically
  relocates the preferred annotated phase relation, and a frozen model path is necessary and rescuable.
- **H10-behavior (ancillary):** if a capable ≈10.63 instrument exists, attenuation has a material effect and
  may interact non-additively with period-3. Its null, failure, or unresolved biological origin does not
  negate H3-behavior or block the P3-only register path.
- **Hcomposition (leading rival):** the prior `C43` effect is accounted for, within the prespecified measured
  feature family, by phase-correlated nucleotide structure. A capable balanced-cohort equivalence/null is a
  scientifically useful terminal outcome.

### 1.3 What a positive may mean

The maximum intended biological interpretation is a **sequence-encoded coding-boundary coherence prior** in
a masked-language model. Even a complete positive cannot show that the model reads a translated frame, that
the spliceosome checks frame, that phase match is necessary for splicing, that the model implements U2AF or
another spliceosomal factor, or that ≈10.63 has a particular biological referent.

Annotated boundary phase and carrier register are separate variables. “Phase” without a qualifier is
forbidden in outcome prose.

## 2. Model, geometry, and notation

### 2.1 Checkpoint firewall

- Model: official `SpliceBERT.1024nt`.
- `pytorch_model.bin` sha256:
  `2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`.
- Expected learned position table: 1,026 rows × 512 hidden dimensions.
- Every tokenized construct has length 1,024 including `[CLS]` and `[SEP]`; nucleotide-to-token offset is
  `+1`.
- Target nucleotide sequence length is 1,022. The masked donor `G` is nucleotide `d=392`, token `393`.
- The true acceptor boundary is nucleotide `a=512`; its motif is `AG` at nucleotides `[510,512)` and tokens
  `[511,513)`.
- The primary donor→acceptor span is `a-d=120` nt.

Only this checkpoint produces Phase-II or Phase-III outcomes. The stage-1 510-nt pair and the five GTEx
fine-tuned checkpoints are provenance, not outcome sources.

### 2.2 Model-native readout

For a construct with the donor `G` masked, define

`f = logit(G) - mean(logit(A), logit(C), logit(T))`.

The primary corruption replaces the true acceptor `AG→AC` at `[a-2,a)`. The intact and corrupted sequences
are otherwise identical. Define cell coupling

`C_b(p_D,p_A; w) = f_intact - f_AG→AC`,

where `b` is a seven-source block, `p_D,p_A∈{0,1,2}` are GENCODE-annotated donor and acceptor coding-boundary
phases, and `w` names a position-table world.

For `δ=(p_A-p_D) mod 3`, let `M0`, `M+`, and `M-` be equal-cell means for `δ=0,+1,-1`. The per-block and
equal-block behavior is

`Ψ_b(w) = M0_b(w) - [M+_b(w)+M-_b(w)]/2`,

`Ψ(w) = mean_b Ψ_b(w)`.

The five directional checks are fixed as `D+=M0−M+`, `D−=M0−M−`, and, for donor phase
`p∈{0,1,2}`, `Ψ_p=C(p,p)−[C(p,p+1 mod 3)+C(p,p−1 mod 3)]/2`. The mismatch-equivalence
endpoints are the raw `M+` and `M−` means themselves. These definitions are applied blockwise before
equal-block aggregation; no orientation is selected from outcomes.

`AG→AT` is a prespecified operator-sensitivity arm. It cannot replace `AG→AC` after outcomes.

### 2.3 Materiality scale

The smallest effect size of interest is `0.05` raw masked-MLM logit units for `Ψ`, `N3`, `N10`, `J`, and
Phase-III register/mediation contrasts unless a section explicitly gives a stricter threshold. “Null” means
an interval wholly inside `[-0.05,+0.05]`; a point estimate near zero is not a bounded null.

## 3. Provenance firewall and freshness registry

### 3.1 Historical inputs

Before cohort enumeration, construct a sorted exclusion registry from source artifacts, never from memory.
Gene versions are removed before comparison (`ENSG...` before the first dot). The union includes:

1. every gene in every predecessor, development, confirmation-reserve, and replication cohort underlying
   `C43`, including all source genes in the 384-block fresh replication;
2. every gene in `C44` smoke and 114-block development preflight, plus its reserved 384-block target (the
   `C43` replication role);
3. every gene in the stopped `GFS25` V1 cohort and completed `GFS25` V2 cohort; and
4. every earlier exon-edge gene named by the source registry frozen in the `C43` protocol.

For archived-only inputs, extraction uses `git show fd26cc1b:<path>` or an equivalent read-only object
lookup. The derived registry records source revision, path, source sha256, extraction command, normalized
count, and the final sorted-list sha256. No archived runner, module, test, or cohort is restored to the active
tree. If a required source cannot be obtained or independently rederived, construction stops.

No `C43`, `C44`, or `GFS25` model outcome, sign, activation, rank, or target covariate enters selection,
balance, role assignment, or dose choice. The only numerical legacy input allowed for sizing is a sign-free
block-variance estimate, recorded separately.

### 3.2 Chromosomes

The default allowed set is `chr1`–`chr7`, `chr10`–`chr22`, and `chrX`. `chrY` is excluded. `chr8` and `chr9`
remain globally sealed and are forbidden to every census, fallback, development, target, and reserve command.
Insufficient capacity is a stop, not authority to use them.

Use of `chr8` or `chr9` requires all four before enumeration: explicit written owner approval, an additive
pre-outcome amendment, a newly named split, and a repository-level record that the earlier seal is spent.

### 3.3 Input pins

The execution freeze package records and verifies:

- hg38 FASTA path, byte size, sha256, and `.fai` sha256;
- GENCODE v50 basic and comprehensive GTF paths, byte sizes, sha256 values, coordinate convention, and
  phase convention; the basic set defines transcript eligibility and the comprehensive set audits local
  coding/strand conflicts;
- tokenizer/config/vocabulary/model file inventory and sha256 values;
- the freshness registry and every input used to derive it;
- this protocol's sha256;
- exact executable and reusable-module sha256 values; and
- Python, PyTorch, Transformers, NumPy, SciPy, scikit-learn, and platform versions.

A recorder that merely prints these values is not a gate. The package audit compares each measured value to
the frozen value and fails on inequality.

## 4. Phase I-A — fresh composition-balanced 3×3 lattice

Phase I-A is model-free. No SpliceBERT checkpoint may be loaded by the builder, balancer, role assigner,
sizer, or construction auditor.

### 4.1 Eligible boundaries and seven-source blocks

Starting from the pinned reference and annotation, enumerate internal protein-coding exon boundaries with a
valid GENCODE CDS phase in `{0,1,2}`, unambiguous reference sequence, canonical donor `GT` or acceptor `AG`
as required by the source role, adequate flanking support for the frozen geometry, and no freshness-registry
gene. Duplicate loci, pseudoautosomal duplicates, conflicting strand/phase annotations, and any sequence with
non-ACGT bases in a measured window are excluded with reason codes.

`p_D` and `p_A` use the conventional **intron-phase** convention: `0` is an interruption between codons,
`1` after the first coding nucleotide, and `2` after the second. For a candidate source edge, derive the
upstream CDS end phase and downstream CDS start phase independently from transcript-oriented GTF CDS frames;
both must agree, and every retained GENCODE-basic transcript supporting the coordinate-identical edge must
assign the same value. The comprehensive-annotation audit must find no conflicting local coding feature or
strand across the measured boundary flanks. The artifact records the raw GTF frame, derived start/end phases,
supporting transcript IDs, and the equality checks; a caller-supplied phase label is never trusted alone.

Each block contains seven globally unique version-normalized genes:

- three authentic donor modules, exactly one for each `p_D`;
- three authentic acceptor modules, exactly one for each `p_A`; and
- one bridge/scaffold source.

Crossing the three donors with the three acceptors yields all nine cells. Every source sequence is authentic
hg38, but every joined target is **chimeric** and must be called chimeric. Each source gene is used in one
block and one study role only. All nine cells travel together through every later state and intervention.

The primary geometry fixes `d=392`, `a=512`, and the 120-nt span. Module seams, bridge bases, motifs, masked
position, sequence length, and tokenized coordinates are identical across the nine cells except for the
authentic source-module sequence. The construction artifact stores source coordinates, strand, versioned and
normalized gene IDs, annotated phases, full sequence sha256, module sha256 values, seam coordinates, and
eligibility/exclusion reasons.

### 4.2 Roles and sample size

Role assignment is a deterministic hash of the normalized seven-gene block identity under a frozen salt,
with relative role weights `development:confirmation:phase3_reserve = 2:2:1`. Assignment uses no sequence
score beyond eligibility, no phase-balance result, and no model quantity. A twelve-block hash-selected subset
of development is the plumbing-smoke role; it is not a fourth independent split.

The exact confirmation count is frozen at **600 complete blocks**. This is the first integer-rounded target
above the sign-free `C43` replication-variance projection needed for `MDE80≤0.04` at the 0.05 SESOI, with a
small guard against landing again at the Monte-Carlo boundary. Development contains at least 120 complete
blocks. Phase-III reserve contains exactly 384 complete blocks. If these counts cannot be met with global
gene disjointness and all balance gates, the study stops before model access; target size is not reduced.

### 4.3 Frozen composition family

All sequence features are computed in transcript orientation. Relative windows use half-open nucleotide
coordinates:

- donor core `[d-3,d+6)` and donor exonic flank `[d-90,d)`;
- donor proximal intron `[d+6,d+36)`;
- acceptor core `[a-20,a+3)`, acceptor proximal intron `[a-50,a-20)`, and acceptor exonic flank
  `[a,a+90)`; and
- the complete joined coding segment between the authenticated exonic module edges.

The feature vector contains:

1. role-prefixed nucleotide one-hot values at every donor/acceptor core position;
2. mono-, dinucleotide-, and trinucleotide frequencies in each exonic and proximal-intronic flank;
3. base and trinucleotide frequencies stratified by annotated codon position in each exonic flank;
4. GC fraction, CpG count/rate, module length, source-exon length, strand, chromosome stratum, and exact
   numeric geometry;
5. donor 9-mer and acceptor 23-mer scores from the execution-package-pinned MaxEnt implementation/tables;
6. joined in-frame stop count, stop rate, mismatch-minus-match stop excess, and counts in the two alternate
   reading registers as diagnostics; and
7. for every aligned donor/acceptor numeric family, absolute-difference and elementwise-product features.

Caller-supplied numeric sequence scores are allowed only if named, implemented, and hash-frozen before the
first cohort is selected; after freeze they join the complete balance and classifier family. No feature may
be dropped because it is difficult to balance. Constant features are recorded; their SMD is defined as zero
only when the compared means are exactly equal.

### 4.4 Balance algorithm and estimand-preserving weights

Block assembly uses deterministic role-wise standardized nearest-triplet matching around the scaffold,
followed by deterministic global selection to minimize the complete frozen imbalance objective. Ties use
lexical block identity. The outcome estimand remains equal-block and uses exact within-block weights `1/3`
for each diagonal cell and `1/6` for each off-diagonal cell in the opposing mean. No outcome-derived or
post-model propensity weight is allowed.

For every feature, compute standardized contrasts for `M0−M+`, `M0−M-`, and `M+−M-`, using the pooled SD
on the complete candidate role. The maximum absolute SMD over every feature and contrast must be `≤0.10`
separately in development, confirmation, and Phase-III reserve.

### 4.5 Classifier and calibration gates

Independently of the optimizer, predict `L=1[p_D=p_A]` from the complete pair-feature vector. Use nested
five-fold group cross-validation, grouping all nine cells of a block, with chromosome-by-strand-balanced
outer folds. Inner folds choose L2 logistic regularization from the frozen grid
`C∈{10^-4,10^-3,10^-2,10^-1,1,10,100}`. Preprocessing is fit inside each training fold. The primary
classifier statistic is pooled out-of-fold ROC AUC. A 2,000-draw chromosome-by-strand whole-block bootstrap
gives its 95% interval.

All of the following must pass separately in confirmation and reserve:

1. maximum absolute SMD `≤0.10` over the complete three-contrast family;
2. classifier AUC 95% upper bound `≤0.55`;
3. every 3×3 cell, role×phase source stratum, strand, and permitted chromosome stratum meets its frozen
   count and no source is duplicated;
4. adding the frozen planted label feature makes AUC lower bound `≥0.70`;
5. 99 balanced whole-block pseudo-label assignments produce family FPR `≤0.05`, and the primary permuted
   label interval contains `0.50`; and
6. a separate construction audit reconstructs every feature, match, role, source, sequence, motif, phase,
   seam, blacklist exclusion, count, statistic, and hash from pinned inputs.

A failed gate is `PHASE1_INSTRUMENT_UNRESOLVED`. It does not license “composition explains the effect” or
“phase is not sequence encoded.” Identifiers, sequences, roles, features, and hashes are sealed before any
model output.

## 5. Phase I-B — optional ≈10.63 sequence-referent branch

This branch is independent and non-blocking. `C39` already supplies the checkpoint-level comparative
association with multi-species pretraining; this arm asks the narrower, still-open question of whether a
specific sequence statistic is a plausible referent. It runs only if the actual 72-species training
transcripts can be recovered and hash-bound; otherwise an explicitly named proxy corpus may be used, and a
training-corpus sequence-referent claim is forbidden.

Analyze transcripts/windows independently without concatenation or artificial phase resets. Compare CDS to
length/GC/species-matched UTR, intron, and noncoding sequence; structure-annotated α-helical CDS to matched
β/coil CDS; native CDS to codon-order shuffles and synonymous recodings; and multispecies to token-count- and
class-matched human sequence. Structure labels must come from an annotation independent of the nucleotide
spectrum; sequence-predicted labels are sensitivity only.

Freeze direct tests at periods `3`, `10.63`, `10.8`, and `32/3`, plus corrected scans over `2.8–3.2` and
`9.5–11.5`. Plant tones across the target effect scale and require at least 80% corrected power to distinguish
`10.63` from `32/3` before claiming wavelength separation. Correct across wavelength, channel, sequence
class, structure class, and control family. A positive is a candidate sequence referent only if it is
CDS-enriched, helix-enriched, stable in held-out species, and behaves as prespecified under recoding controls.
A calibrated null demotes that referent; it does not positively establish an optimization/table-length
ruler. No Phase-I-B outcome changes Phase-II or P3-only Phase-III authorization.

## 6. Phase II — period-3-first attenuation factorial

### 6.1 Position-table decomposition

Let `P` be the raw 1,026×512 learned position table. Mean-center each hidden column across rows, take the
row-axis real FFT, edit the declared coefficients with conjugate symmetry, invert in float64, and restore
the column mean. The NumPy reference and Torch implementation must agree at the frozen tolerance.

- Period-3 band `B3`: rFFT bins `339..345` inclusive.
- ≈10.63 band `B10`: rFFT bins `94..100` inclusive.

The band names are operational. The resulting finite-table periods are reported rather than overwritten by
the prose labels. No sub-bin fit changes the nominated bins.

For attenuation dose `d∈{0.25,0.50,0.75,1.00}`, multiply nominated complex coefficients by `1-d` and leave
every other coefficient unchanged. The same table is used for a cell's intact and `AG→AC` sequences. Table
worlds are:

| World | Period-3 state | ≈10.63 state |
|---|---|---|
| `Y00` | intact | intact |
| `Y10` | selected P3 attenuation | intact |
| `Y01` | intact | selected P10 attenuation |
| `Y11` | selected P3 attenuation | selected P10 attenuation |

Let `Yab=Ψ(Yab)`. Define

- `N3 = Y00-Y10`;
- `N10 = Y00-Y01`;
- `Nboth = Y00-Y11`; and
- `J = Y11-Y10-Y01+Y00 = N3+N10-Nboth`.

With this arm coding, `J<0` is super-additive attenuation damage and `J>0` is sub-additive attenuation
damage on the raw `Ψ` scale. `J=0` is additivity on that scale only. `Nboth` is descriptive and does not
join a confirmatory family unless an amendment is frozen before target exposure.

### 6.2 Outcome-blind dose selection

Dose selection occurs only on the development role. Annotated boundary-phase labels, `Ψ`, `N3`, `N10`, `J`,
and any sign-equivalent target statistic are unavailable to the dose selector.

1. Evaluate P3 doses from smallest to largest using the P3 capability, specificity, and health gates below;
   select the largest passing dose.
2. If no P3 dose passes, return `P3_INSTRUMENT_UNRESOLVED`; no target world opens.
3. Evaluate P10 doses independently and select the largest individually passing dose.
4. If a P10 dose passes individually, test it jointly with the selected P3 dose. If joint health fails,
   step P10 down monotonically while keeping the P3 dose fixed. Use the largest P10 dose that passes both
   individual and joint gates.
5. If no P10 dose survives, return `OPTIONAL_P10_INSTRUMENT_UNRESOLVED`. Only `Y00/Y10` may later open;
   `Y01/Y11/N10/J` remain unexecuted and uninterpretable.

This P3-first rule is fixed because period-3 is the primary question. It is not an after-outcome preference.

### 6.3 Structured null families

Nulls have the same table shape, nominated frequency support, changed-table RMS, dose, sequence states, and
cohort as their named intervention. Separate intervention sizes are never pooled.

For each carrier, freeze two 31-member families before model access:

1. **Same-band phase controls.** Rotate each nominated complex coefficient by seeded sign-balanced phases
   whose coefficient-space displacement has exactly the named attenuation's RMS while preserving nominated
   coefficient magnitude and band power.
2. **Hidden-axis controls.** Right-transform the complete band component by a seeded orthogonal signed
   permutation in hidden space, then subtract dose `d` of that transformed component. This preserves the
   band waveform's spatial Gram matrix, row norms, support, and changed-table RMS while leaving the native
   learned carrier present.

The learned intervention must beat both families separately. Their p-values are not combined. The joint
family contains 31 independently seeded P3/P10 control pairs with the exact combined support and per-band
doses; it supplies a null distribution for `J`. P3, P10, and joint families use RNG seeds `2026081003`,
`2026081004`, and `2026081005`, respectively, under NumPy `PCG64`; the freeze artifact materializes every
matrix and hashes it. A control generated after target outcomes is invalid.

### 6.4 Exact replay and rescue controls

Raw-table replay must be byte-identical to the captured checkpoint table and reproduce all logits within
`1e-8`. Replacing an attenuated band with its original raw-checkpoint coefficients is the prespecified
table-level rescue and must reconstruct `Y00` within `1e-6` at the logit level. The rescue establishes
intervention reversibility; it is not Phase-III path sufficiency. Composition order in `Y11` must be exactly
commutative.

## 7. Phase-II development preflight

The final target code is run first on the 120-block development role. Smoke uses only its hash-selected
twelve-block subset and may emit shapes, request counts, construction checks, numerical errors, and
planted-control results. Smoke may not emit a target verdict or write to a target path.

Recorders—checkpoint name, table size, requested dose, and sample count—are not counted as passing gates.
Every gate below is implemented as a failable comparison against the frozen value.

### 7.1 Mechanical gates — required for every interpreted table

1. Captured raw table equals the checkpoint file tensor exactly; the checkpoint and tokenizer hashes match.
2. NumPy and Torch band decompositions agree with maximum absolute error `≤1e-5`.
3. Intact/replay forward error is `≤1e-8`; attenuation-plus-exact-band-rescue error is `≤1e-6`.
4. The selected-band coefficient multiplier equals `1-d` within `1e-10` relative; nonselected rFFT leakage
   is `≤1e-10` relative to the maximum raw coefficient.
5. Named and corresponding control edits match changed-table RMS within `1e-10` relative. Phase controls
   preserve band power and hidden-axis controls preserve the spatial Gram matrix within `1e-10` relative.
6. P3 edits leave `B10` coefficients unchanged exactly; P10 edits leave `B3` unchanged exactly; joint edit
   order differs by at most `1e-12`.
7. Token IDs, nucleotide text, sequence/corruption pairing, masks, motifs, source IDs, cell labels, role tags,
   and coordinates are identical across table worlds. Intact and corrupt states never share a captured
   activation by accident.
8. Logits, losses, attentions, projections, and every derived statistic are finite. Repeated deterministic
   forwards differ by at most `1e-8`.

### 7.2 P3 capability gates — mandatory

On the fresh development lattice, discard annotated phase labels and equal-average all nine cells inside
each block. Let `K3` be this phase-blind ordinary paired-acceptor coupling under intact minus P3-attenuated
tables. It is a positive-control computation, not `Ψ` or a sign-equivalent target statistic. The selected P3
dose must satisfy all:

- `K3` 95% lower bound `>+0.05`;
- learned `K3` exceeds every member of both 31-member P3 control families, exact add-one one-sided
  `p=1/32≤0.05`, and its paired learned-minus-mean-control lower bound is `>0` in each family;
- the aggregate frozen nine-reader P10 attention score retains at least `80%` of its intact magnitude, with
  the retention lower bound `>0.80`; and
- exact band rescue recovers at least `90%` of `K3`, with recovery lower bound `>0.80`.

This is a new capability reading, not a replication or reopening of the completed `C34`–`C36` present-use
family.

### 7.3 P10 capability gates — optional

The base-checkpoint reader set is frozen as

`{L2H6,L2H12,L3H3,L3H5,L3H8,L3H9,L3H11,L3H13,L4H13}`.

Using the same post-softmax carrier-mass statistic and deletion audit recorded for `C37`, the selected P10
dose must:

- have an intact aggregate score with 95% lower bound `>0` and a finite, sign-stable denominator in at
  least 95% of gene-bootstrap draws;
- reduce the aggregate carrier-specific score by at least 50%, with 95% lower bound on proportional
  reduction `>0.25`;
- exceed every member of both 31-member P10 capability-control families, with exact add-one `p≤1/32` and
  paired learned-minus-mean-control lower bound `>0`;
- retain at least `80%` of intact `K3` magnitude, with retention lower bound `>0.80`; and
- recover at least 90% of its attention-score change under exact band rescue, with recovery lower bound
  `>0.80`.

Failure is `OPTIONAL_P10_INSTRUMENT_UNRESOLVED`, not a P10 null and not a reason to stop P3.

### 7.4 Behavior-independent health and specificity gates

Every named, control, and joint table that may be interpreted must pass:

1. native-source `AG→AC` coupling 95% lower bound `>+0.05`;
2. phase-blind crossed `AG→AC` coupling 95% lower bound `>+0.02`;
3. fixed-bridge `AG→AC` effect interval wholly inside `[-0.05,+0.05]`;
4. phase-blind crossed `AC−AT` interval wholly inside `[-0.05,+0.05]`;
5. protected true-base log-probability-change lower bound `>-0.10` nat and masked-base accuracy retention
   `≥0.80` relative to the intact table;
6. local motif-clean donor `GT→AC` and acceptor `AG→AC` destruction-response retention `≥0.80`, with intact
   response lower bound `>0.50` native-margin units on the corresponding **base-checkpoint** source-site
   assay; this threshold is not imported from the fine-tuned `GFS25` classifier;
7. no target sequence, block, or gene fails in only one world. A numerical failure excludes the complete
   block from no world: it stops the run and is diagnosed before any target interpretation.

These pooled checks do not use annotated match labels or `Ψ` and may be used for dose selection.

### 7.5 Resolution and false-positive gates

Using `B=20,000` centered chromosome-by-strand whole-block development bootstraps, project the fixed
600-block target:

- `MDE80≤0.04` for clean `Ψ00` and `N3`;
- if P10 is capable, `MDE80≤0.05` for `N10` and `J`;
- bootstrap Monte-Carlo SE of every projected MDE `≤0.0002`; and
- under 1,000 sign-symmetric synthetic null replicates, empirical familywise false-positive rate `≤0.06`
  for the exact analysis/decision functions; under planted effects `0.05`, `0.075`, and `0.10`, corrected
  power at `0.075` is `≥0.80` for `N3` and for every optional endpoint claimed capable.

Identity/no-op fixtures, planted positives, sign-reversed fixtures, and null fixtures must traverse the real
analysis and verdict functions. Reimplementing the decision rule inside the planted test is forbidden.

### 7.6 Authorization states

- Any failed Phase-I gate: `PHASE1_INSTRUMENT_UNRESOLVED`; no model target.
- Any failed mechanical, P3 capability, shared-health, P3 resolution, or FPR gate:
  `P3_INSTRUMENT_UNRESOLVED`; no target world.
- P3 passes and P10 fails: `P3_ONLY_TARGET_AUTHORIZED`; only `Y00/Y10` may execute.
- P3 and P10 plus joint health/power pass: `FULL_FACTORIAL_TARGET_AUTHORIZED`; all four worlds must execute.

The authorization artifact contains no target effect and is independently audited before a target command
can read the confirmation cohort.

## 8. Phase-II target inference

### 8.1 Unit, resampling, and missingness

- Independent unit: the complete seven-source block.
- Point estimates: equal-block means; all nine cells and all executed worlds of a block travel together.
- Bootstrap: 20,000 chromosome-by-strand whole-block draws using NumPy `PCG64` seed `2026081001`.
- Pseudo-phase controls: 999 balanced whole-block assignments generated with seed `2026081002`; because
  `N=600`, all six `S3` acceptor-label permutations occur exactly 100 times in each assignment.
- No runtime exclusion, winsorization, imputation, or per-world missingness is allowed. A failed forward
  makes the one-shot run anomalous and blocks interpretation.

### 8.2 Confirmatory families

The **primary P3 family** is always `{Ψ00,N3}`. Use max-absolute-studentized simultaneous 95% bounds over
the two endpoints, with one-sided positive decision limits. Keeping the optional family separate is frozen
to prevent an ancillary P10 arm from taxing the primary P3 question.

If the full factorial is authorized, the **ancillary P10 family** is `{N10,J}`. `N10` uses a one-sided
positive simultaneous bound. `J` uses a two-sided simultaneous interval; it is material only if that entire
corrected interval lies above `+0.05` or below `-0.05`. The sign is reported using the arm-coding map in
§6.1 and is never relabelled after outcomes.

The clean directional family `{D+,D-,Ψ0,Ψ1,Ψ2}` uses max-studentized simultaneous lower bounds. The mismatch
family `{M+,M-}` uses simultaneous two-sided intervals at the `±0.05` equivalence bound. Failure of mismatch
equivalence permanently forbids “only,” but does not negate a preferential positive.

### 8.3 Permutation and structured-null requirements

- `Ψ00` must exceed all 999 balanced pseudo-phase controls; report exact add-one one-sided p and the paired
  target-minus-mean-control 95% lower bound. The p-value floor is reported as granularity, not strength.
- `N3` must exceed every P3 same-band phase control and every P3 hidden-axis control separately; each exact
  add-one p must be `≤1/32` and each paired learned-minus-mean-control lower bound must be `>0`.
- If executed, `N10` has the analogous two-family requirement.
- A joint interaction claim requires `|J|` to exceed every absolute joint-control `|J_s|`, exact add-one
  `p≤1/32`, plus a paired signed excess in the observed direction whose corrected lower bound is `>0`.

Exact add-one p-values are `(1+#controls at least as extreme)/(1+#controls)`. No Gaussian tail is substituted
for a finite control family. P3, P10, and joint structured-null families remain separate in artifacts,
inference, and prose.

### 8.4 Target classifications

All classifications require the relevant preflight authorization, construction audit, health gates,
target resolution, code/input hashes, one-shot chronology, and independent result audit.

| Conditions | Classification | Licensed statement |
|---|---|---|
| Clean `Ψ00` simultaneous interval wholly inside `[-0.05,+0.05]` | `BALANCED_PREFERENCE_BOUNDED_NULL` | No material residual annotated-phase preference is detected after the prespecified measured composition balance in this fresh chimeric domain; no mediation target exists |
| `Ψ00` lower bound `>0` but not `>+0.05`, or directional/pseudo-phase criteria fail | `BALANCED_PREFERENCE_INCONCLUSIVE` | No material directional conclusion |
| `Ψ00` lower bound `>+0.05`; all five directional lower bounds `>0`; pseudo-phase criteria pass | `BALANCED_PREFERENCE_MATERIAL` | A material composition-balanced preferential behavior exists in this domain; “only” still requires mismatch equivalence |
| Material `Ψ00`; `N3` interval inside `[-0.05,+0.05]`; both P3 null-family construction/audits and power pass | `P3_ATTENUATION_BOUNDED_NULL` | The tested health-preserving P3 attenuation does not materially support the residual preference at the 0.05 scale |
| Material `Ψ00`; corrected `N3` lower bound `>+0.05`; both P3 null families and rescue pass | `P3_NECESSITY_SUPPORTED` | Period-3 causally supports the residual preference under this intervention; P3-only Phase III is authorized |
| Material `Ψ00`; corrected `N3` upper bound `<-0.05` | `P3_SUPPRESSIVE` | P3 attenuation materially increases the preference; supportive-use wording and Phase III are not authorized |
| P3 necessity supported; P10 interval bounded/null or P10 unresolved | `P3_ONLY_CLOSURE_PATH` | P3 advances independently; P10 remains separate and cannot block P3-only Phase III |
| P3 necessity supported; corrected `N10` lower bound `>+0.05`; corrected `J` interval outside `±0.05`; all P10/joint nulls pass | `DUAL_SCALE_INTERACTION_SUPPORTED` | A scoped P3×P10 interaction is supported and the optional P10-modulated Phase-III arm is authorized |
| Any required gate/audit fails | `INSTRUMENT_UNRESOLVED` | No positive, null, suppressive, or inverted target conclusion |
| Any other pattern | `INCONCLUSIVE` | Report estimates and boundaries without mechanism promotion |

A material `J` with null/unresolved `N10` is a context-dependent interaction diagnostic, not dual-scale
support and not authority for the P10-modulated Phase-III arm. A P10 positive with corrected `J` equivalent
to zero supports parallel contributions only; “integrated code” is forbidden.

## 9. Phase III — span-wide P3 register relocation

### 9.1 Authorization and roles

The P3-only Phase-III path opens if and only if the audited Phase-II target is
`P3_NECESSITY_SUPPORTED` and its clean behavior is `BALANCED_PREFERENCE_MATERIAL`. This authorization is
independent of P10.

Phase-II development may be used for Phase-III instrument development and candidate selection. The
384-block `phase3_reserve` remains unread until the exact register operator, direction, candidate set,
patch sites, controls, inference, output paths, and code hashes are sealed and independently audited. No
Phase-II confirmation block is reused as Phase-III confirmation.

The optional P10-modulated Phase-III arm additionally requires the audited Phase-II class
`DUAL_SCALE_INTERACTION_SUPPORTED`. A P10 capability failure, bounded/null `N10`, or corrected-null `J`
keeps that arm closed while leaving the P3-only arm open.

### 9.2 Full-span analytic phase-ramp operator

The register intervention uses the full donor-to-acceptor boundary span, derived from each cell:

- nucleotide support: `[d,a)=[392,512)`;
- token support: `[d+1,a+1)=[393,513)`; and
- support length: exactly 120 tokens, including the masked donor `G` and acceptor `AG`, excluding the first
  acceptor-exon nucleotide.

It uses the complete P3 band `k=339..345`, not a fitted single sinusoid. For nucleotide `t`, let
`j=t+1` be its position-table/token row and let the positive-frequency analytic P3 component be the complex
512-vector

`Z3(j) = (2/N) Σ_{k=339}^{345} P̂_k exp(i 2πkj/N)`, with `N=1026`.

The implementation evaluates this expression continuously from the frozen Fourier coefficients; it does
not interpolate or roll position-table rows. For register `r∈{-1,0,+1}` and nucleotide
`t∈[d,a)`, define the phase ramp

`φ_r(t) = −(2πr/3) · (t-d)/(a-d-1)`

and replace only the real P3 component by

`C3_r(j) = Re[Z3(j) exp(iφ_r(t))]`,

`P_r(j) = P(j) - Re[Z3(j)] + C3_r(j)`.

Outside `[d,a)`, `P_r=P` exactly. Thus the ramp is zero at the donor `G` and totals `−2πr/3` at the
acceptor `G` (`t=a-1`). The executable direction label is frozen as `right`; `r=0` is an exact no-op. A
uniform phase shift across both endpoints is a common-mode control, not the relational intervention.

The frozen sign convention predicts preferred biological offset

`δ=(p_A-p_D) mod 3 = -r mod 3`.

With the positive-frequency convention above, the negative ramp evaluates a pure center-bin P3 carrier as
`C3(j-r)` at the acceptor endpoint: source displacement `s=-r`, carrier-register label `q=-s=r`, and
effective-state equality gives `δ+q=0`. The frozen truth table is therefore
`r=-1 → δ=+1`, `r=0 → δ=0`, and `r=+1 → δ=-1` (represented modulo three as `2`).

An independently implemented symbolic/index audit must establish this handedness before development. An
opposite-handed target is a falsification; the sign may not be flipped after outcomes.

### 9.3 Register estimands

For P10 state `z∈{intact,attenuated}`, equal-weight blocks, phases, and strands to form

`G(δ,r,z)=E[C | (p_A-p_D) mod 3=δ, register=r, P10=z]`.

For each `r`, define

`S_r(z)=G(-r mod 3,r,z)-mean_{δ≠-r}G(δ,r,z)`

and

`Ω3(z)=mean_{r∈{-1,0,+1}} S_r(z)`.

`Ω3` is insufficient by itself: all three `S_r` simultaneous lower bounds must exceed zero. The `r=0`
cell must reproduce the Phase-III reserve's no-ramp `Ψ` inside `[-0.05,+0.05]`. Report ramp sign, strand,
annotated donor phase, chromosome, and center-free full-span sensitivity; none can substitute for the
primary all-register result.

If the P10-modulated arm is authorized, define

`Jreg = Ω3(P10 intact)-Ω3(P10 attenuated)`.

The Phase-III freeze sets `Jreg`'s expected direction to the sign of the independently confirmed,
multiplicity-corrected Phase-II `J`. This is coherent because Phase-II `J` equals P3 attenuation necessity
with P10 intact minus that necessity with P10 attenuated. A reserve effect in the opposite direction is
reported as such and cannot be renamed supportive. `Jreg` is an effect of P10 attenuation on P3 relocation,
not a directed `P3→P10` arrow.

### 9.4 Register controls and development authorization

Before reserve exposure, every register state must pass the Phase-II mechanics and health gates plus:

1. `r=0` table and logits are exact no-ops within `1e-8`;
2. off-support rows are byte-identical, the pointwise non-P3 residual `P-Re[Z3]` and independently
   reconstructed P10 analytic component are unchanged within `1e-10` relative on support, and P10 reader
   score retains at least 80%; the expected global spectral leakage of a windowed phase ramp is reported and
   matched by controls, not mislabelled a failure;
3. the measured analytic phase at the donor and acceptor endpoints matches the requested ramp within
   `1e-6` rad and all intermediate rows are monotone in signed phase;
4. ramp-forward order, batch order, and corruption state change no result above `1e-8`;
5. an independently rescaled same-RMS common-mode uniform P3 phase rotation, a same-RMS ramp on 31 frozen
   signed hidden-axis controls, and an independently rescaled same-RMS upstream decoy span `[152,272)` nt /
   `[153,273)` tokens are executed for every nonzero register (`r=0` remains an exact no-op); the decoy is
   disjoint from `[392,512)` and exists in every construct;
6. development `Ω3` lower bound is `>+0.05`, every development target `S_r` simultaneous lower bound is
   `>0`, learned `Ω3` exceeds all 31 hidden-axis controls (`p≤1/32`), and the paired
   `{target−common-mode Ω3, target−decoy Ω3}` specificity family has simultaneous lower bounds `>0`.
   Neither common-mode nor decoy may itself have simultaneous lower bounds `>0` for all three of its
   `S_{-1},S_0,S_{+1}` contrasts. This is a failure-to-reproduce criterion plus a positive paired separation,
   not an equivalence or bounded-null claim;
7. projected reserve `MDE80≤0.05` for `Ω3`, every `S_r`, primary path necessity, primary rescue, and—if
   authorized—`Jreg`; and
8. planted positive, opposite-handed, no-ramp, common-mode, wrong-support, and null fixtures traverse the
   real register analysis and decision functions with familywise false-positive rate `≤0.06` and corrected
   power `≥0.80` at a planted `0.075` effect.

Failure is `P3_REGISTER_INSTRUMENT_UNRESOLVED`; the reserve remains sealed. This outcome cannot be reported
as “the ridge does not move.” The `C44` development preflight, its local two-support surgery, `Ω`, A/V
signs, and Q/K signs are blacklisted as calibration and are never pooled with this development reading.

## 10. Phase-III localization and mediation

### 10.1 Candidate generation and freeze

Candidate generation is explicitly discovery-level. On development, measure carrier-register structure at
separate evidence levels:

- pre-softmax QK/logit decomposition;
- post-softmax attention;
- per-head output and per-MLP residual changes; and
- behavior under one-component same-state scrub and restoration.

The candidate universe is all 96 attention heads and all six MLP outputs. The screen patches the full
`[393,513)` token support, not a site-local window. Intact and `AG→AC` states are patched only from the same
sequence and corruption state. L3H1 and L2H15 receive no automatic positive label; prior evidence is stated
as rationale, not used to waive selection or null tests.

Rank components by the development same-state scrub loss in `Ω3`. Freeze the smallest descending prefix
whose cumulative jointly remeasured loss reaches 80% of the full register effect, capped at eight
components. Static singleton effects are not summed. If no prefix of at most eight reaches 80%, freeze the
top eight as a **distributed candidate set** and forbid sparse/minimal wording. Candidate identities,
prefix size, layers, patch tensors, token support, signs, and all matched sets are hash-frozen before the
reserve opens.

Thirty-one component-null sets match candidate count, layer counts, head/MLP type, patch-token count, and
development output-delta norm within ±10%. A set that cannot be matched is an instrument failure; looser
matching is not chosen after outcomes.

### 10.2 Causal cells

For each register and corruption state, let `A_r` be the native register-specific candidate activation and
`A_0` the same-sequence `r=0` activation. Define:

- **raw:** native `P_r` forward;
- **scrub:** run `P_r` but replace the frozen candidate activation by same-state `A_0`;
- **restore:** run the health-preserving Phase-II P3-attenuated table with the requested analytic ramp and
  replace the frozen candidate activation by same-state `A_r` from the intact-band source;
- **component-null scrub/restore:** the identical operations on each matched component set;
- **wrong-state prohibition:** no intact activation enters an `AG→AC` host and no `AG→AC` activation enters
  an intact host; and
- **sufficiency:** in a separate arm, inject `A_r-A_0` into the no-ramp intact host at the frozen site and
  compare with norm-matched null directions. This arm is not inferred from restoration.

Compute `Ω3` and all `S_r` in every cell. Define

- path necessity `D_path=Ω3(raw)-Ω3(scrub)`;
- path restoration `R_path=Ω3(restore)-Ω3(attenuated host)`; and
- path sufficiency `Q_path=Ω3(sufficiency)-Ω3(no-ramp host)`.

All three use raw logit units. No percentage mediation is reported unless the denominator's simultaneous
interval excludes zero and the 0.05 SESOI; even then the primary report is the controlled absolute effect.

### 10.3 QK, attention, and OV firewall

A candidate may be called a **selector** only if the carrier-register relation appears in pre-softmax QK
scores relative to matched heads. It may be called an **attention router** only if the relation survives
softmax and a controlled attention intervention changes `Ω3`. It may be called an **output mediator** only
if same-state output scrub and restoration pass. A QK or attention result alone cannot carry the mediation
claim.

For any head promoted to a route, run a frozen attention×value factorial over the full span. Report routing,
value, and interaction cells separately. A value-equivalence claim requires its simultaneous interval wholly
inside `[-0.05,+0.05]` with `MDE80≤0.05`; otherwise the route is mixed. Bilinear regrouping identities are
marked forced and never interpreted as binding.

Directed edge language additionally requires an edge/path intervention from the frozen upstream candidate
to the downstream output mediator, plus necessity and restoration. Failure of local Q/K after a successful
full-output path leaves fine localization unresolved; it does not erase the broader mediation result.

### 10.4 Reserve inference and classifications

Use equal blocks and 20,000 chromosome-by-strand whole-block bootstrap draws, seed `2026081006`. Use
simultaneous max-studentized lower bounds over `{Ω3,S_-1,S_0,S_+1}` and separately over
`{D_path,R_path,Q_path}`. Each learned path contrast must exceed all 31 matched component sets with exact
add-one `p≤1/32` and paired learned-minus-mean-null lower bound `>0`. Correct the optional `{Jreg, P10-path
necessity, P10-path restoration}` family independently; it never alters P3-family thresholds.

| Conditions | Classification | Licensed statement |
|---|---|---|
| `Ω3` lower bound `>+0.05`; every `S_r` simultaneous lower bound `>0`; target−common-mode and target−decoy paired `Ω3` simultaneous lower bounds `>0`; neither control reproduces all three positive `S_r`; handedness, health, no-ramp, hidden-axis, and audit gates pass | `P3_REGISTER_RELOCATION_SUPPORTED` | The span-wide learned-P3 register causally relocates the composition-balanced preference in this assay |
| Relocation passes; path instrument capable; `D_path`, `R_path`, and separate `Q_path` lower bounds `>+0.05`; all matched-set tests pass | `P3_REGISTER_MEDIATED` | The frozen component path is necessary, rescuable, and separately sufficient for a scoped portion of the relocation |
| Relocation passes; path instrument capable; simultaneous intervals for all of `D_path`, `R_path`, and `Q_path` lie inside `[-0.05,+0.05]` | `P3_ROUTE_BOUNDED_NULL` | The declared frozen path does not materially mediate relocation or supply a material sufficiency effect; alternative routes remain open |
| Relocation passes but path gate/power fails | `P3_ROUTE_INSTRUMENT_UNRESOLVED` | Register relocation may be reported; route interpretation is withheld |
| Opposite-handed material result with all gates pass | `P3_REGISTER_HANDEDNESS_FALSIFIED` | The frozen biological orientation is falsified; it is not renamed positive |
| P3 mediation passes; optional `Jreg` and P10 path family pass | `P10_MODULATES_P3_REGISTER_PATH` | P10 attenuation modulates the scoped P3 register computation; no biological identity or directed carrier-to-carrier arrow is licensed |
| Any required register gate/audit fails | `P3_REGISTER_INSTRUMENT_UNRESOLVED` | No positive, null, or inverted register conclusion |
| Any other pattern | `INCONCLUSIVE` | Stop at the strongest lower evidence level |

A directed `P3→P10` statement would require an additional asymmetric intervention: P3 changes a later
P10-specific state, restoring that state rescues the interaction, and the reciprocal path is bounded. That
experiment is not part of this protocol. Write “joint use” or “P10 modulation,” never an arrow.

## 11. Execution, sealing, and audit contract

### 11.1 New-tree rule

Successor implementation is confined to:

- `scripts/splicebert_positional_boundary_coherence/`;
- flat active modules named `src/gfmcd/splicing/positional_boundary_coherence.py`,
  `src/gfmcd/splicing/pbc_carrier_factorial.py`, and `src/gfmcd/splicing/pbc_register_*.py`;
- `results/splicebert_positional_boundary_coherence/`; and
- new tests under `tests/`.

No successor runner is added to `scripts/splicebert/`. No file imports from a deleted or archived
`phase_*`, `p3_direct_edge_*`, or exon-edge package. A generic active primitive may be reused only through
its live path and with its behavior covered by a successor test.

### 11.2 Immutable stage graph

The executable graph has four namespaces:

1. `construction/` — freshness registry, census, role assignment, sequence package, balance report, sizing,
   and independent construction audit; no checkpoint access;
2. `development/` — smoke, dose/capability/health/power preflight, null materialization, candidate discovery,
   and authorization audit;
3. `target/` — one-shot Phase-II confirmation worlds, immutable per-block endpoints, analysis, and result
   audit; and
4. `phase3/` — register development, separate reserve freeze, one-shot reserve endpoints, mediation, and
   result audit.

Every writer calls the repository overwrite guard before the first forward and refuses an existing path.
Smoke, development, target, and Phase-III outputs have distinct filenames and directories. A smoke flag may
never reduce `N` while retaining a target filename or verdict field.

### 11.3 Freeze packages

Before Phase-II target exposure, write one machine-readable freeze package containing:

- protocol and additive-amendment hashes;
- all executable, module, schema, and test hashes;
- checkpoint, tokenizer, reference, annotation, blacklist, cohort, feature, role, null, and environment
  hashes;
- exact commands, seeds, thresholds, sample counts, expected request counts, and immutable output paths;
- development gate values and a pure authorization class containing no target outcome;
- explicit booleans for `p3_target_authorized` and `full_factorial_authorized`; and
- a declaration that `chr8/chr9` were neither enumerated nor read, or the required owner-approved amendment.

The target runner recomputes every binding, refuses a dirty tree, refuses an unauthorized world, and exits
before checkpoint loading on any mismatch. The Phase-III reserve gets a separate post-Phase-II package with
the exact candidate and ramp bindings. Phase-II authorization never implicitly authorizes Phase III.

### 11.4 Artifact and manifest minimum

Every result-producing command writes immutable JSON plus a sibling manifest. The result contains the
smallest per-block endpoint table sufficient to rederive all point estimates, intervals, gates, p-values,
and decision branches. The manifest contains command, code revision, dirty state, input/output sha256 values,
checkpoint hash, parameters, seeds, device/dtype, package versions, sample/exclusion counts, stage, status,
and chronology timestamps.

Fitted matchers, raw logits, and activation captures are not committed. If retained locally, their hashes
and derivation role are recorded, but they cannot be the only support for a claim. No artifact is hand-edited
after production. A changed interpretation is a new audit/correction artifact pointing at the immutable
result.

### 11.5 Independent audits

The construction, preflight authorization, Phase-II result, Phase-III register, and Phase-III mediation each
receive an independent audit. An audit:

- reads recorded primitive endpoints and recomputes statistics without importing the analysis verdict
  function;
- verifies exact protocol-to-code threshold and endpoint coverage;
- distinguishes fireable gates, forced identities, and recorders;
- checks full-family multiplicity, add-one denominators, null size matching, and sign conventions;
- verifies role/gene disjointness, checkpoint firewall, chronology, hashes, and output immutability;
- fails if a required endpoint, arm, control, manifest field, or raw row is absent; and
- declares whether it is clean-checkout re-executable.

Target interpretation requires audit `PASS`. A recorded gate assertion that the auditor cannot rederive is
an attestation, not an independently audited pass, and blocks confirmatory wording.

## 12. Anomaly, amendment, and stopping rules

1. A failed construction, capability, health, sensitivity, resolution, false-positive, seal, or audit gate
   stops its dependent stage. The corresponding target is not opened when still sealed.
2. An anomalous control freezes interpretation. Diagnose on synthetic fixtures or development; never tune on
   target rows.
3. A pre-target implementation defect may be repaired only with a dated additive amendment, new code hash,
   rerun of the complete affected preflight, and a new freeze/audit package.
4. A defect found after target exposure preserves the target artifact, records the exposure and defect, and
   cannot be repaired into confirmation on the same target. Any rerun requires a genuinely fresh role.
5. A failed P10 branch stops only P10-dependent cells. It does not relax P3 gates or convert P10 into a null.
6. A failed Phase-III route does not weaken a valid Phase-II P3 necessity result or a valid register
   relocation; evidence stops at the last passed rung.
7. No sample size, dose, carrier bin, sign, feature, candidate, null, interval type, or endpoint changes after
   its dependent outcomes are visible.

## 13. Strongest wording by terminal outcome

Before any new result, the only licensed statement remains:

> SpliceBERT has a causal period-3 relative-position carrier and a separate behavioral preference for
> matched annotated coding-boundary phase in a prior fixed-span chimeric assay; whether period-3 implements
> that preference is unresolved.

If Phase II alone passes for P3:

> In a fresh composition-balanced fixed-span chimeric lattice at `SpliceBERT.1024nt`, health-preserving
> attenuation of the learned period-3 band materially reduces the model's annotated boundary-phase
> preference relative to intervention-matched controls. Period-3 therefore causally supports this scoped
> sequence-encoded boundary-coherence behavior.

If P3 register relocation and mediation also pass:

> In the same checkpoint and assay family, a full-span analytic period-3 register intervention cyclically
> relocates the preferred annotated boundary relation, and a frozen internal path is necessary, rescuable,
> and separately sufficient for a scoped portion of that relocation. SpliceBERT therefore uses a learned
> relative period-3 coordinate to evaluate a sequence-encoded coding-boundary coherence prior.

If the optional joint arm also passes, one additional sentence is allowed:

> Attenuating the separate ≈10.63 carrier changes the period-3 register effect through a corrected
> non-additive interaction in this assay.

That sentence does not require, and does not imply, resolution of ≈10.63's biological origin. None of the
three wordings licenses reading-frame sensing, translation, a spliceosomal frame checker, U2AF/spliceosome
implementation, native junction usage, transcript abundance, or generalization beyond the frozen domain.

## 14. Pre-run completion checklist

No target command is valid until every item is machine-checked:

- [ ] protocol and any amendments hash-bound;
- [ ] archived provenance used only for the declared registry/variance roles;
- [ ] complete C43/C44/GFS25 freshness registry independently rederived;
- [ ] chr8/chr9 absent, or explicit owner-approved amendment present;
- [ ] exact development/confirmation/reserve counts and global gene disjointness pass;
- [ ] full feature/SMD/AUC/planted/permuted construction gates pass;
- [ ] checkpoint, tokenizer, FASTA, GTF, environment, code, schema, and null hashes pass;
- [ ] P3 mechanical, capability, health, specificity, resolution, FPR, and power gates pass;
- [ ] P10 and joint authorization recorded independently, without changing P3 authorization;
- [ ] target world allow-list agrees with authorization class;
- [ ] independent preflight conformance audit passes; and
- [ ] immutable output paths do not exist.

No checked box is evidence about a target effect. It certifies only that the declared question can be asked.

## Amendment A1 — executable source-site health and authorization chain (2026-08-10)

**Timing:** additive before any successor development or target model outcome. **Scope:** data retention and
execution conformance only. No hypothesis, endpoint, threshold, dose, null family, or decision wording is
changed.

The Phase-I candidate package must retain enough reference-derived sequence to execute every Section 7.4
gate natively for every exact table. For each donor and acceptor candidate, the enumerator derives from the
pinned hg38 FASTA and GENCODE-v50 transcript geometry a 1,022-nt transcript-oriented `native_sequence`, its
`native_donor_pos`, and its `native_acceptor_pos`; the latter is aligned to nucleotide 512. The runner checks
the embedded `GT` and `AG` motifs and the complete source geometry against sequence, rather than accepting a
caller-supplied pass flag. Each scaffold similarly records a reference-derived `bridge_acceptor_pos`, frozen
at nucleotide 437 for this geometry, and construction verifies an `AG` at `[435,437)` survives identically
in all nine cells. These sequences and anchors are outcome-blind inputs, hash-bound through construction,
its independent audit, preflight, and target freeze. Missing sequence or a failed reconstruction stops
before a health forward.

For every exact target, single-band control, and joint-control position table that could be carried into the
confirmation run, preflight records 120 raw whole-block rows sufficient to rederive native-source `AG→AC`,
phase-blind crossed `AG→AC`, fixed-bridge `AG→AC`, crossed `AC−AT`, protected-base log-probability and
accuracy, reciprocal donor `GT→AC`, acceptor `AG→AC`, and numerical-completeness gates. Donor destruction is
read at the native acceptor and acceptor destruction at the native donor; retention is relative to the
unaltered checkpoint table on the identical native windows. The independent preflight auditor recomputes
all intervals, ratios, table hashes, family coverage, and thresholds from those rows. Shared intact health
is not a substitute for an intervention table's own pass.

Authorization is three-stage. The development runner writes measurements and an embedded raw/analysis
content hash but cannot write a target seal. A separate auditor, which does not import the preflight verdict
function, must emit `gfmcd.pbc.carrier_factorial_preflight_audit.v1` with status `PASS`. Only a separate
freezer may consume the raw preflight, its manifest, that audit and its manifest; it requires a clean tracked
Git revision, regenerates and hash-matches every authorized position table, and binds the checkpoint,
candidate package, cohort, protocol, code, audit chain, exact shard count, every shard path, and the assembly
path. The target runner refuses a missing or self-issued audit chain and refuses any output path or shard
partition not named in that audited seal.

## Amendment A2 — P3-only closure completion and compact evidence (2026-08-10)

**Timing:** additive before any successor development or target model outcome. **Scope:** makes already
declared Phase-III requirements executable and bounds the route language; it changes no scientific
hypothesis, target sign, SESOI, register direction, or target sample.

Section 9.4 gate 7 includes projected `MDE80≤0.05` for **all three** primary path endpoints:
`D_path`, `R_path`, and the separately measured `Q_path`. Missing or underpowered sufficiency keeps the
reserve sealed even when necessity and restoration are adequately powered. The candidate-freeze audit must
recompute all three projections from the jointly remeasured development prefix.

For every target, signed hidden-axis control, common-mode control, and decoy-span register table,
"passes the Phase-II health gates" means the complete Section 7.4 / Amendment-A1 source-site table, not a
masked-probe shorthand. The register preflight therefore consumes the hash-bound candidate package and its
construction audit, records the primitive whole-block health rows for each exact table, and receives a
separate audit that recomputes every inherited interval, ratio, table identity, family count, and P10
retention quantity. A masked-base probe is one health endpoint, not an authorization by itself.

Raw activation and per-cell mediation shards may remain local. The immutable assembled mediation artifact
normalizes the three phase-pair cells sharing each `(block, component-set, state, register, delta)` into
mean intact margin, mean `AG→AC` margin, and mean coupling, while retaining a complete cell-identity ledger,
deduplicated patch-provenance ledger, raw-shard file records, counts, and tensor-stream hashes. Those compact
endpoints are sufficient to rederive every `S_r`, `Ω3`, `D_path`, `R_path`, `Q_path`, interval, and matched-set
test. The independent result audit must operate from the compact artifact and must replay both raw-state
margins—not only their difference—against the separately audited relocation artifact.

The primary mediation class licenses only a **frozen component-output path**. A selected attention head is
not thereby called a selector, attention router, or value carrier. The attention×value factorial in
Section 10.3 becomes mandatory before any head-specific routing/value subtype is promoted in a claim or
paper figure; if it is not run, those subtype fields remain explicitly unadjudicated. This bound does not
waive same-state output necessity, restoration, separate sufficiency, matched component-set controls, or
the QK/attention/output evidence-level firewall required for the primary component-path result.

## Amendment A3 — primary scoped bridge and partial register concordance (2026-08-10)

**Timing:** additive before any successor development or target model outcome. **Scope:** changes the paper
priority, adds one directional Phase-II tier and one partial-concordance Phase-III tier, and amends only the
development-relocation success clause in Section 9.4 gate 6 as stated below. It does not change a primary
endpoint, intervention, sign, SESOI, null family, construction/health/false-positive/power threshold,
sample, or original material/full-relocation/mediation rule. It adds only the derived handedness guard `H`
defined below.

### A3.1 Primary paper target

The primary paper target is the Phase-II P3 attenuation bridge, reported at the strongest of two positive
tiers. Section 8.4 gains this lower tier:

| Conditions | Classification | Licensed statement |
|---|---|---|
| Every construction, mechanical, P3-capability, health, false-positive, resolution, seal, and independent-audit gate passes; the primary-family simultaneous lower bounds for both `Ψ00` and `N3` are `>0`; all five clean directional point estimates `{D+,D−,Ψ0,Ψ1,Ψ2}` are `>0`; the clean pseudo-phase exact-p and paired-specificity requirements pass; both separate P3 structured-null exact-p and paired-specificity requirements pass; rescue passes; and observed `MDE80≤0.04` for both `Ψ00` and `N3` | `P3_DIRECTIONAL_SCOPED_SUPPORT` | Period-3 attenuation shifts a composition-balanced boundary preference in the prespecified aggregate direction relative to both intervention-matched control families. The link is statistically resolved but is not shown to exceed the `0.05` materiality threshold. |

This is scoped directional causal support for a rough P3–boundary-preference link, not material mechanism
closure. It does not license “material,” “only,” a universal orientation statement, register relocation,
route mediation, or Phase-III exposure. If either simultaneous lower bound is not above zero, any clean
directional point estimate is non-positive, a pseudo/structured-null specificity requirement fails, or an
instrument/audit gate fails, this class is unavailable.

The stronger Phase-II class remains `P3_NECESSITY_SUPPORTED`: material composition-balanced `Ψ00`, material
corrected `N3`, both P3 structured-null families, rescue, health, resolution, seal, and independent-audit
gates all pass under the original Section-8 rules. That result closes the material scoped bridge and alone
authorizes Phase III. Phase III is strengthening evidence; it is not required for either Phase-II wording.

### A3.2 Partial-concordance register tier

For each register, define the frozen opposite-handed contrast by substituting `δ=+r mod 3` for the expected
`δ=−r mod 3` relation:

`Sopp_r = G(+r mod 3,r,P10 intact) − mean_{δ≠+r mod 3} G(δ,r,P10 intact)`,

`Ωopp = mean_r Sopp_r`, and `H = Ω3(P10 intact) − Ωopp`.

At block level, evaluate an expected-orientation **partial-only three-endpoint max-T family**:

`{H_b, Ω3_b−Ωcommon_b, Ω3_b−Ωdecoy_b}`.

Use the chromosome-by-strand whole-block bootstrap with 20,000 draws and seed `2026081010`. The simultaneous
lower bounds for all three endpoints must exceed zero. This family jointly guards handedness and separation
from the two named geometric controls. For the opposite-only falsification path, evaluate the mirrored
three-endpoint family

`{−H_b, Ωopp_b−Ωcommon,opp,b, Ωopp_b−Ωdecoy,opp,b}`

with 20,000 draws and seed `2026081012`; all three simultaneous lower bounds must exceed zero. The two
partial-family seeds and bootstrap count are bound in the pre-reserve seal and emitted in the analysis and
manifest. The original Section-9.4/Section-10.4 two-endpoint specificity family for full relocation is
unchanged; neither partial-only family replaces or relaxes it.

Section 10.4 gains one classification below `P3_REGISTER_RELOCATION_SUPPORTED`:

| Conditions | Classification | Licensed statement |
|---|---|---|
| Phase II is `P3_NECESSITY_SUPPORTED`; every Phase-III mechanical, health, replay, seal, and independent-audit gate passes; the simultaneous lower bound for `Ω3` is `>+0.05`; all three frozen-orientation `S_r` point estimates are `>0`; at least two of their three simultaneous lower bounds are `>0`; learned `Ω3` exceeds all 31 hidden-axis controls with exact `p≤1/32` and a positive paired learned-minus-control lower bound; all three simultaneous lower bounds in the partial-only `{H,target−common-mode,target−decoy}` family are `>0`; neither common-mode nor decoy satisfies the same raw partial rule; and the opposite diagnostic defined below does not pass | `P3_REGISTER_PARTIAL_CONCORDANCE_SUPPORTED` | The aggregate register response is material and all three estimates point in the prespecified direction, with at least two orientations individually resolved. This is partial register concordance that strengthens the rough, assay-scoped P3–boundary-coherence link; it is not complete cyclic relocation. |

Classification is by strongest passing tier. If all three `S_r` simultaneous lower bounds exceed zero and
the original Section-10.4 requirements pass, the result is
`P3_REGISTER_RELOCATION_SUPPORTED`, not the partial class. If fewer than two `S_r` lower bounds exceed
zero, any point estimate is non-positive, the aggregate is non-material, a named control reproduces the
partial rule, or any instrument/specificity/audit gate fails, the supported class is unavailable.

The frozen opposite diagnostic requires `Ωopp` lower bound `>+0.05`, positive point estimates for all three
`Sopp_r`, at least two simultaneous `Sopp_r` lower bounds `>0`, opposite `Ωopp` superiority to all 31
hidden-axis controls and to their paired mean, non-reproduction by common-mode and decoy, and all three
lower bounds in the mirrored opposite-only family above `>0`. If that diagnostic passes while the expected
raw partial pattern does not, classify `P3_REGISTER_PARTIAL_HANDEDNESS_FALSIFIED`. If both raw orientations
satisfy their 2-of-3 pattern, or either otherwise qualifies but its orientation-specific handedness family
is unresolved, classify
`P3_REGISTER_PARTIAL_HANDEDNESS_AMBIGUOUS`. Neither class licenses a rough P3–boundary link. If neither
orientation passes, the original bounded-null, failed-instrument, or inconclusive rule applies as
appropriate.

For development authorization only, Amendment A3 replaces Section 9.4 gate 6's requirement that every
development `S_r` lower bound exceed zero with the partial-concordance rule above. Thus either audited
`P3_REGISTER_PARTIAL_CONCORDANCE_SUPPORTED` or the stronger audited
`P3_REGISTER_RELOCATION_SUPPORTED` may open the reserve. Gates 1–5, 7–8 and all mechanical, health,
specificity, fixture, and projected-power requirements remain unchanged. A partial reserve result still
cannot open the route-mediation arm; only the original full-relocation classification can do so.

`P3_REGISTER_PARTIAL_CONCORDANCE_SUPPORTED` does not authorize “cyclically relocates across all three registers,” a
universal orientation claim, route mediation, necessity/restoration/sufficiency language, or a directed
circuit. Those promotions still require the original three-of-three relocation class and the separate
Section-10 mediation gates. Optional P10 results never alter this hierarchy.

### A3.3 Training-data limitation language

A finite or uneven pretraining corpus is one plausible reason that a learned relation might be approximate
rather than perfectly symmetric across the three annotated orientations. An unresolved orientation may be
described as **compatible with uneven training support**, but not as caused by it. This compatibility
hypothesis never waives a construction, capability, health, replay, specificity, resolution, seal, or audit
gate and never converts an inconclusive or failed-instrument result into support.

A positive training-data explanation requires a separate, outcome-independent census of the identifiable
training corpus and, for causal attribution, a controlled corpus or training intervention. Without that
evidence, the paper must say that finite-corpus support is a possible limitation—not that training-data
scarcity explains the observed pattern.

### A3.4 Outcome wording

If Phase II reaches only `P3_DIRECTIONAL_SCOPED_SUPPORT`:

> In a fresh composition-balanced fixed-span chimeric assay at `SpliceBERT.1024nt`, health-preserving
> attenuation of the learned period-3 band shifts the model's annotated boundary-phase preference in the
> prespecified direction relative to both intervention-matched control families. The effect is statistically
> resolved but is not shown to exceed the `0.05` materiality threshold; this is a rough, assay-scoped causal
> link, not material mechanism closure.

If Phase II reaches `P3_NECESSITY_SUPPORTED` and no Phase-III strengthening tier passes:

> In a fresh composition-balanced fixed-span chimeric assay at `SpliceBERT.1024nt`, health-preserving
> attenuation of the learned period-3 band materially reduces the model's annotated boundary-phase
> preference relative to intervention-matched controls. Period-3 therefore causally supports this scoped
> sequence-encoded boundary-coherence behavior.

If the partial-concordance tier also passes, add:

> A full-span analytic period-3 register intervention produces a material aggregate response in the
> prespecified direction: all three register estimates are concordant and at least two are individually
> resolved. This partial register concordance strengthens the functional P3–boundary-preference link while
> leaving complete three-register relocation and its internal mediation unresolved.

The stronger Section-13 full-relocation and mediation wording remains available only when its original
three-of-three and route-specific requirements pass.
