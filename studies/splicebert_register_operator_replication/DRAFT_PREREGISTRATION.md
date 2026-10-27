# Draft preregistration — crossed register under CC and max margin

**DRAFT, NOT FROZEN. Owner approved the 342-block clean fallback and 120 nt only on
2026-10-16. No CC outcome has been computed.** The exact selected IDs are bound below.
Implementation and independent conformance review precede the execution freeze.
The operator, cohort, and span decisions are fixed; no additional span is authorized.

## 1. Intent and historical boundary

Does the annotated donor–acceptor phase-matching preference in the fixed-span
crossed chimeric assay persist with the paper's primary coupling operator:
true-acceptor `AG→CC` and masked-donor max margin?

This is a new study with new immutable output paths. C43's row, protocol, raw rows,
analysis, audit, and original verdict are immutable. The source implementation is
available only from local Git revision
`fd26cc1b6c107dc8476a4b3d117ae897e2e86437` in `hidden_GFM`; its archived prose is
superseded by R45. Construction and analysis are newly transcribed and tested in
this study's own code. No archived module is restored or imported. The staged R49
reproduction package in `practical-hellman-b38d21` is a read-only reference.

The boundary is behavior in chimeric model inputs. Neither a period-3 implementation,
an equality comparator, translation, natural splice-site choice, nor spliceosomal
recognition is established. This study does not authorize any C44 rerun.

## 2. Fixed operators and estimands

At nucleotide 392 (zero-based; tokenizer position 393 after CLS), mask the donor G.
For every cell retain the four nucleotide logits in every scored state and compute
both margins from those same logits:

`m_max = l_G − max(l_A,l_C,l_T)`

`m_linear = l_G − (l_A+l_C+l_T)/3`.

For operator `o` in `{CC,AC,AT}`, define
`C_b,o,r(p,q) = m_r(intact) − m_r(true acceptor AG→o)`.
CC/max is the sole primary combination. AC/max, CC/linear, and AC/linear are
prespecified bridge endpoints; AT is retained for the original AC/AT checks.
Corruptions act at `[a−2,a)`, with `a=512` and `a−d=120`.

For each complete block, indices are modulo three:

```
M0 = mean_p C(p,p)
M+ = mean_p C(p,p+1)
M− = mean_p C(p,p−1)
Psi = M0 − (M+ + M−)/2
D+ = M0 − M+
D− = M0 − M−
Psi_p = C(p,p) − (C(p,p+1)+C(p,p−1))/2, for donor rows p=0,1,2
```

The estimand is the equal-block mean, in raw logit units. The five directional
checks are `D+`, `D−`, and the **donor-row** `Psi_p` contrasts. Acceptor-column
contrasts may be descriptive but do not substitute for this family.

Record paired changes in Psi for CC versus AC at each margin, and max versus linear
at each corruption. Use a four-endpoint simultaneous interval family. These are
operator-dependence diagnostics, never tests that select the primary operator or
require operators to agree. Report the margins' competing-base identities/ties to
make changes induced by the nonlinear max readout inspectable.

## 3. Geometry, cohort, and motif hygiene

The owner-selected span is **120 nt only**. Preserve 1,022 sequence nucleotides / 1,024 tokens,
seven distinct source genes per block, the full 3×3 annotated phase lattice, native
source windows, authentic fragment boundaries, bridge sequence, seam constraints,
phase convention, and protected positions from C43. All genes are version-normalized.
Source genes are globally disjoint across blocks. Sampling strata are chromosome ×
strand; chr8, chr9, and chrY are excluded for this study.

Preferred fresh option: exclude C43's 2,688 source genes, every predecessor role
(including the unopened confirmation role), the older C43 source blacklist, and
all genes in the PBC candidate census/reserve. A missing exclusion source is fatal.
The model-free census must determine capacity before selecting N. Do not release
PBC candidates or spend excluded chromosomes to repair a deficit. Historical AC
effects may supply centered variance only, never sequence or block selection.

If fresh supply cannot support the chosen N, owner approval is required for the
fallback. It is explicitly **outcome-aware verification**, because C43 AC/linear
outcomes are known. The new CC result is prospective with respect to its operator;
that does not make the cohort an independent replication.

### CC-specific safety

Writing CC creates an upstream GC donor exactly when the preceding base is G.
For the proposed clean design, exclude a block if any true acceptor or any of its
six native-source acceptors has `seq[a−3]=='G'`. Exclusion is sequence-only and occurs
before any new model forward. Require zero newly planted `AG`, `GT`, or `GC` motifs
in the three overlapping dinucleotide registers for every CC-edited request.
Enumerate all 16 flanking contexts against the production corruption function;
cross-check [the repository motif memorial](../../tests/test_motif_cleanliness.py).

Keep the historical bridge corruption **AG→AC** as the primary bridge negative
control. Substituting CC at that bridge would introduce an additional design change
and a much larger hygiene exclusion; it is not silently added. Describe this as an
AC bridge specificity control, not a corruption-matched CC sham. The native CC
positive control and motif audit separately check channel health and motif safety.

The 99 pseudo-phase controls use the **same CC/max matrices and two-nucleotide
edits** as the target, so they are the matched-size structured null family for
label alignment. The AC bridge is kept separate and never pooled with that family.
This design does not certify spatial specificity of CC perturbations in chimeras.

Frozen-C43 census: 19/1,152 target acceptor modules are unsafe, affecting 18 blocks
and 57/3,456 cells. Another 16 donor-role native acceptors are unsafe, in 16 further
blocks. Thus 366 blocks are target-clean; 350 are clean for both target and all
native controls (172 plus-strand, 178 minus-strand). These are model-free counts.
The familiar ~0.4% genomic frequency is not the rate in this cohort.

A strict fallback that also preserves equal strand counts and exact six-permutation
balance can contain at most **342 blocks, 171 per strand**. Its exact selection must
discard singleton chromosome × strand strata and then use a fixed SHA-256 order,
with a constraint that every retained stratum has at least two blocks. There is no
selection on AC effects. The manifest must bind the final IDs, excluded IDs and
reasons, salt, and algorithm. This is a declared departure from 384, not a claim to
have repeated the old sample exactly. The model-free selection is now recorded in
[fallback342_selection_v1.json](planning/fallback342_selection_v1.json), with sorted-ID
hash `972d388cc031b6debdbb5b1c531afb7a8619630462f6bf4bb380d6a989417436`.
Final feasibility is in [DECISIONS.md](DECISIONS.md): the preserved fresh design has
zero balanced-block capacity; only an explicitly approved fallback can proceed.

For a fresh cohort use a fixed hash-sorted chromosome round robin within each strand,
with the same minimum stratum size. No stopping at a favorable prefix, replacement
after scoring, or retrospective enlargement is allowed.

## 4. Freeze and exposure order

1. Obtain the owner's cohort/span decision; complete the draft accordingly.
2. Independently audit the design, construction, and protocol-to-code conformance.
   Freeze protocol, source/data/checkpoint hashes, cohort IDs, all RNGs, request
   order/batch, schemas, numerical thresholds, and unused output paths. Pin actual
   file hashes because HEAD alone cannot identify a dirty working tree. No commit
   or push is authorized by this protocol.
3. Run smoke **memorials only**, including synthetic logits, planted-positive and
   permuted-null calibrations. Do not score a new CC target for a smoke example.
4. Reconstruct and replay the complete historical 384-block AC/linear assay as a
   fidelity memorial. Historical numerical targets are allowed only in this gate.
   No CC model request can be issued until this gate is audited PASS.
5. Run native controls, then the frozen crossed target once, after the required
   preflight checks pass. Preserve all outputs, including a capability failure.
6. Analyze all prespecified endpoints and run an independent result audit. Only
   then assign the final classification and draft a new claim/R-number entry.

Observed target resolution and some phase-blind controls require target rows, so
their final gates run after scoring; they are not represented as pre-target
preflight successes. Preflight calibration and the old-fidelity gate must pass
first. A target-dependent gate failure cannot trigger a new sample or new operator.

## 5. Fidelity and falsifiable memorials

The checkpoint is `SpliceBERT.1024nt`, SHA-256
`2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d`.
Hash tokenizer/configuration and hg38/GENCODE v50 basic files as well. Bind the
comprehensive annotation only where a specified safety predicate uses it; never
substitute it for the original transcript selection.

Before CC, reconstructed fragment hashes, all 3,456 historical cell sequence hashes,
all 2,304 native source windows, source identities, phases, seams, masks, and the
original 99 pseudo assignments must match exactly. Inference from stored rows must
match the independent C43 audit at relative/absolute tolerance `1e-10`, reporting
bit identity separately. This offline match alone is not model/construction fidelity.

For the full model replay, require maximum absolute discrepancy **≤1e-4 logit units**
over every stored intact/AC/AT/bridge donor margin, acceptor-site margin, and native
margin. Protected true-base log probabilities must agree within 1e-4 nat and all
discrete correctness flags must match exactly. All comparisons are required, not
merely reported. No missing cell,
mean-only comparison, correlation threshold, or selected subset can pass this gate.
The 1e-4 tolerance is frozen before replay and is 500 times smaller than the 0.05
SESOI; its worst-case induced error is ≤2e-4 for a coupling and ≤4e-4 for Psi
(0.8% of the SESOI). Record actual hardware and dtypes. Preserve the original
CPU float32 operation order for the historical margin: `target − (sum(four bases)
− target)/3`; an algebraically equivalent sum of the other three bases may differ
by an ulp. Deterministic repeat forwards in the
execution environment must still agree within the original `1e-8` threshold.
Failure stops CC; do not widen either tolerance after seeing discrepancies.

Memorials use the **production** construct→edit→mask→score→row→matrix path. Deliberate
faults must fail: swapped p/q, reversed corruption subtraction, +1 mask offset,
wrong max competitor set, a dropped/duplicated cell, a misassigned source, a G at
the CC flank, altered pseudo assignment, and a protected probe with the wrong source
coordinate. Test numeric return schemas and argument names before any model job.

Synthetic logits must include examples where max and linear differ and where their
CC-minus-AC differences differ. A fixture with three identical alternative logits
cannot validate that the correct readout was used. Exact reconstruction identities
are numerical anchors; they are not biological sensitivity evidence.

## 6. Statistical calibration before CC

Use the final N, frozen stratum sizes, complete nine-cell blocks, both production
margin functions, and the complete production analysis and decision paths. Generate
synthetic raw four-logit score rows and serialize/reload them through the same schema.
Include correlated cells, heterogeneous block variances, max-competitor changes,
and a phase-blind coupling offset. Calibrate the statistical instrument, not the model.

Freeze disjoint development and test RNG streams, with seeds `2026101601` and
`2026101602`; planted-positive seed `2026101603`. No calibration uses any new CC
model output. Null repetitions permute acceptor labels at whole-lattice/block level,
retaining all states/readouts of a block together. Include main-effect-only nulls
and heterogeneous nulls; never shuffle individual cell outcomes independently.

The complete normative recipe is [STATISTICAL_CALIBRATION_DRAFT.md](STATISTICAL_CALIBRATION_DRAFT.md)
and must be hash-frozen with this protocol. It uses 256 development realizations,
1,000 held-out null realizations, and 256 paired planted realizations. Full-classifier
false positives must number at most 38/1,000 (exact one-sided 95% upper FPR bound
below 5%). Each of six interval families separately must pass its prespecified
binomial coverage audit. The 0.05 detection plant and 0.15 material-preference plant
each require at least 216/256 successes, giving a one-sided 95% lower sensitivity
bound above 80%. The synthetic capable noise is a known fixture, not an estimate
of CC/max noise. A deliberately noisy fixture must fail the resolution gate.
An unsuccessful calibration produces `INSTRUMENT_UNRESOLVED`; do not search seeds
or alter thresholds against the held-out results.

## 7. Inference and power

Bootstrap the **whole seven-source block** within chromosome × strand, preserving
all nine cells, arms, probes, and spans together. The primary analysis retains
20,000 draws and seed `20260803`. No cross-block gene reuse is allowed.

- Psi: ordinary two-sided percentile 95% CI.
- Resolution: `(1.96+0.8416212335729143) × SD(centered-stratum bootstrap means) ≤0.05`.
- Five directions: max-absolute-studentized simultaneous 95% lower bounds.
- Mismatch family: two simultaneous max-studentized 95% intervals, both within
  `[-0.05,+0.05]` to license the word **only**.
- Pseudo phases: 99 frozen balanced whole-lattice acceptor relabelings, seeds
  `2026080401..2026080499`; assign each of the six S3 permutations N/6 times by the
  documented SHA-256 ordering. At N=342 the count is 57, not C43's 64. Require
  exact regeneration and a phase orientation truth table before model loading.
- Add-one p: `(1 + number(control ≥ target))/100`, with ties counted against the
  target. Strict superiority to all controls yields p=0.01, the test's floor.
  Also require a positive lower 95% bound for target minus mean control.

Do not pool past and new outcomes. These balanced pseudo labels test the declared
label-alignment null; they do not randomize biological phase independently of all
sequence features. A label-specific result remains a composite sequence effect.

Planning uses only C43 within-stratum residual variance. The recorded finite-draw
MDE is 0.04976964 at N=384. The exact empirical variance of that same fixed-stratum
bootstrap gives MDE 0.05008128; this is a planning diagnostic and **does not change
C43's frozen verdict or the new fixed-seed gate**. Record both the finite-draw and
analytic resolution diagnostics in the new run. No choosing seeds or more draws
because the first gate fails.

Square-root projections and SD inflation scenarios are in
[power_v1.json](planning/power_v1.json). They address scalar resolution, not the
joint power of all capability/directional/equivalence tests. CC/max variance is
unknown until the one-shot run; a favorable AC projection is not a guarantee.

## 8. Ten capability gates and decision rule

The ten gate headings retain C43's structure; several inputs necessarily change to
the new primary operator. This is a prospective specification, not an attestation
that these gates have run.

| Gate | Frozen requirement |
|---|---|
| 1. Construction | Final chosen N, complete gene-disjoint lattices, hashes, exclusion registry, CC hygiene, and independent construction/conformance audit PASS; all smoke/fidelity/calibration prerequisites PASS. |
| 2. Native positive | Equal-block native-source CC/max coupling lower 95% bound >+0.05. Six clean sources per block; no source dropping after scoring. |
| 3. Crossed capability | Phase-blind mean of all nine CC/max cells lower 95% bound >+0.02. |
| 4. Bridge negative | Historical bridge AG→AC effect on the max margin has its 95% CI wholly within ±0.05; also report its linear-margin result. |
| 5. Resolution | Primary CC/max centered-stratified MDE80≤0.05, with the frozen 20,000 draws/seed. |
| 6. Historical alt check | Phase-blind AC-minus-AT **linear** simultaneous interval wholly within ±0.05; max-margin counterpart is diagnostic, not a hidden new veto. |
| 7. Protected health | Intact-construct versus aligned-native true-base log-probability lower 95% bound >−0.10 nat; masked-base accuracy retention ≥0.80; a zero source-correct denominator fails. |
| 8. Site health | Intact donor/acceptor max-margin lower bounds exceed floors `min(0,native lower95)` computed from the same frozen six-source blocks before crossed scoring. Report linear counterparts. |
| 9. Numerics | Every score finite; 1,024 tokens; correct mask/base mapping; deterministic repeated forward max difference≤1e-8; recorded four-logit margins reconstruct. |
| 10. Independent audit | Separately rederive rows, gates, bootstrap, controls, diagnostics, pins, and attempt chronology; no copying producer summaries as checks. |

Any failed capability gate gives `INSTRUMENT_UNRESOLVED`; preserve and report raw
endpoints but make no positive, null, or inverted target claim. The historical
paired Psi_AC−Psi_AT linear interval within ±0.05 remains a sensitivity condition
for a positive preference classification. CC−AC and max−linear equality are never
required. Every failed or unrun gate is listed.

For a capable instrument, apply mutually explicit outcomes:

- Material preference: primary lower CI>+0.05, all five directional lower bounds>0,
  strict superiority to all 99 controls, add-one p≤0.01, positive paired excess
  lower bound, and the historical AC/AT sensitivity condition. If both mismatch
  intervals are within ±0.05, add `ONLY_WHEN_COHERENT_IN_FROZEN_DOMAIN`; otherwise
  use `MATERIAL_PREFERENTIAL_POSITIVE` and forbid **only**.
- Detectable preference/materiality unresolved: lower CI>0 but not >0.05 and the
  same directional, pseudo, and historical sensitivity conditions pass. If its
  entire CI is also within ±0.05, explicitly report it as a detectable but bounded
  sub-SESOI effect, rather than implying materiality remains unknown.
- Bounded null: primary CI wholly within ±0.05 (report any nonzero sub-SESOI
  estimate as above); no conclusion about another operator or geometry.
- Material inversion: primary upper CI<−0.05; report directional/control patterns
  without relabeling the expected diagonal.
- All other capable patterns: `INCONCLUSIVE`.

## 9. Stop codons, sensitivity, and additional spans

Before outcomes, reconstruct the annotated-frame junction stop-count 3×3 matrix,
`omega = mean(off-diagonal stop load) − mean(diagonal stop load)`, and the cohort
median omega cutpoint. Report the matrix, Spearman association of omega with block
Psi, and estimates for omega>median and omega≤median, using the frozen cutpoint.
This is diagnostic; it never selects blocks, changes a gate, or proves that phase
is separable from coding/stop composition. Report leave-one-block, chromosome, and
strand sensitivity without selecting a preferred estimate.

The owner declined additional spans. For the recorded design comparison, candidate
spans 144 and 168 nt retain the
120-nt span's modulo-three class; shorter 90/96-nt spans cannot retain the 32+64-nt
authentic intronic flanks and nonempty bridge. Longer spans require new bridge
availability, seam/motif audits, shifted acceptor coordinates/probes, and a new
common-support cohort. They are not automatically available on frozen C43 blocks.

Any future study of extra spans would need its own common block set and familywise
inference plan. This study will not pool spans to improve the primary gate or assume
3N independent blocks. The authorized scope is one span; no generalization over
separations will be claimed.

## 10. Artifacts, audit, and reporting

Run jobs in the background only after the smoke/fidelity gate. Every output and
sibling manifest has an explicit new path and an exclusive-create/overwrite guard.
Use `gfmcd.provenance.ExperimentManifest` with checkpoint/data/source hashes, dirty
code digest, parameters, seeds, command, dependency/hardware versions, and stage
authorization hashes. Preserve raw four-base logits, sufficient probe statistics,
gene/module IDs, and compact cohort/reconstruction recipes; do not commit model
weights or activation binaries. A compressed raw-row package must retain a hash
of its uncompressed bytes and be re-executable from the recorded sources.
For protected probes also retain the full-vocabulary log normalizer and argmax
(or full masked logits): four nucleotide logits alone do not reconstruct
full-vocabulary true-base log probabilities and accuracy. Validate the six pinned
checkpoint/tokenizer/config files by an explicit allowlist; unrelated local
metadata such as `.hf_revision` is not a reason to delete files or change a pin.

After independent audit, draft a **new** claim row and the next unused R-number
(R49 is already occupied in this checkout). Record the owner's pre-outcome operator
decision, cohort tier, actual differences from C43, all failed gates, and the result
regardless of sign. Publication/commit/push remains on the owner's cadence.

The paper-ready paragraph must give CC/max Psi with CI, N and genes, 120-nt span,
M0/M+/M−, minimum of the five directional bounds, maximum pseudo control and exact
p, gate status/MDE, paired AC bridge, and the in-frame stop-load diagnostic. Replace
the live manuscript's L131 numbers only after that audit; never call an
outcome-aware fallback a fresh blinded replication. A numerical template is not a
result and no paper change is made at this design gate.
