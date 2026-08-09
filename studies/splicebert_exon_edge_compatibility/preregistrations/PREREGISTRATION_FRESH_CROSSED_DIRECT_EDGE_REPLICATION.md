# Preregistration — fresh crossed direct-edge replication

> **Tier: T2 — provenance.** Frozen before any model output on the replication blocks.

**Status: FROZEN 2026-08-03.** This protocol is a new one-shot replication. It does not
amend, reopen, or repurpose the sealed confirmation role of
`PREREGISTRATION_CROSSED_DIRECT_EDGE_LATTICE.md`, which stopped after its frozen
development-resolution gate failed.

**Pre-model amendment R1 (2026-08-03; no replication outcomes existed):** the exact count
was increased from the initial 336-block ledger-remainder proposal to 384 newly constructed
blocks after a one-sided upper-variance calculation and an outcome-blind all-eligible-
chromosome census. The amendment also strengthened freshness from crossed-study non-overlap
to a version-normalized blacklist of every gene in every prior exon-edge cohort. The original
336-block reasoning was never model-scored.

## 1. Question and claim boundary

At the official `SpliceBERT.1024nt` checkpoint, does masked-donor direct-partner coupling
remain materially larger when authentic donor and acceptor coding-boundary phases match than
when they differ, in a fresh gene-disjoint replication of the fixed 120-nt crossed lattice?

The primary block estimand is unchanged:

`Psi_b = M0_b - (M+_b + M-_b)/2`,

where `M0`, `M+`, and `M-` are the three perfect-matching means of the complete donor-phase
by acceptor-phase `3 x 3` lattice under true-acceptor `AG->AC` corruption. The primary
replication estimand is the equal-block mean `Psi`.

A positive result licenses only:

> At `SpliceBERT.1024nt`, in the frozen crossed authentic-module domain, direct-partner
> coupling is materially greater for annotated phase-matched than phase-mismatched coding
> boundaries.

It does not establish that the spliceosome reads phase, that mismatched junctions cannot be
spliced, that the effect is an exact equality comparator, that the period-3 carrier implements
the behavior, or that the result generalizes to another checkpoint, context, species, model,
or objective. The stronger word **only** remains a separate mismatch-equivalence endpoint.

## 2. Fresh replication domain

The construction reuses the frozen eligibility and geometry predicates, but performs a new
model-free census and block assignment with a new hash salt. The allowed chromosomes are
`chr1`--`chr7`, `chr10`--`chr22`, and `chrX`. `chr8` and `chr9` remain excluded to preserve
the repository's global sealed-test boundary; `chrY` is excluded because of sparse and
pseudoautosomal duplication-sensitive capacity.

Before enumeration, the builder derives a version-normalized (`ENSG...` before the first
dot) blacklist from every prior cohort artifact in this study: all phase-edge attempts, both
phase-state cohorts, the optimal exact-distance view, crossed-lattice v1, and the final compact
crossed cohort. Each source artifact and the resulting sorted blacklist are hash-bound. No
prior target magnitude, sign, per-block effect, or activation is read.

The new replication:

- excludes every version-normalized gene present in any prior exon-edge cohort, including all
  genes selected into the permanently sealed prior confirmation role;
- constructs new blocks from eligible genes outside that registry rather than repurposing a
  prior block;
- uses exactly 384 blocks, 192 per strand, hence 2,688 mutually distinct source genes;
- preserves the original sequence length, donor/acceptor coordinates, bridge geometry,
  authentic modules, phase convention, readout, corruption operators, protected probes, and
  native-source controls;
- treats chromosome and strand as sampling strata and reports leave-one-chromosome,
  leave-one-strand, and leave-one-block sensitivity.

Selection is model-free. Within each strand, newly constructed candidate blocks are grouped
by chromosome, ordered within stratum by SHA-256 under the new construction salt and selected
one per stratum in a SHA-256-ordered round robin until 192 are obtained. The exact salt,
namespaces, preimages, and collision tie-break are emitted by the builder and frozen before
model access. Lexical text is the
explicit collision tie-break. No sequence covariate, stop statistic, model output, prior
`Psi_b`, or target-direction information enters selection.

An independent construction audit must rederive the blacklist and selection from the pinned
inputs, confirm exact gene/module non-overlap, reconstruct every sequence and hash from
GENCODE v50 and hg38, and verify the complete lattice, motifs, seams, phases, bridge,
tokenizer-independent geometry, and pseudo-phase family before checkpoint loading.

## 3. Power and frozen sample size

The stopped 120-block development role is used only as a variance source. Its centered
stratified bootstrap gave `MDE80=0.0728239160` and bootstrap SD `0.0259934909`; neither its
target sign nor target magnitude enters sample-size selection. The pooled within-stratum
residual SD is `0.301741` (`df=106`); its one-sided 95% upper bound is `0.340503`, an inflation
factor of `1.12846`. Under square-root scaling:

- naive projected `MDE80(384) = 0.040710`;
- upper-variance projected `MDE80(384) = 0.045939`;
- projected `MDE80` under a 20% SD increase is `0.048852`.

All are below the unchanged `0.05` SESOI. The exact sample size is frozen at 384 and may not
be changed after any replication model output. The observed, sign-independent centered
stratified `MDE80` must still be `<=0.05`; the projection is not a substitute for that gate.

## 4. Readout, controls, and estimands

For every construct, nucleotide 392 is masked and

`f = logit(G) - mean(logit(A), logit(C), logit(T))`.

The primary cell effect is

`C^AC_b(p_D,p_A) = f(intact) - f(true acceptor AG->AC)`.

The prespecified sensitivity arm substitutes `AG->AT`. The fixed bridge `AG->AC` effect is
the negative control. Native authentic source edges are evaluated in separate native windows.
All nine cells and every state of a block travel together through inference.

The unchanged target family contains:

- primary `Psi` and `M0`, `M+`, `M-`;
- directional `D+`, `D-`, and phase-specific `Psi_0`, `Psi_1`, `Psi_2`;
- simultaneous mismatch-equivalence intervals for `M+` and `M-` at `+/-0.05`;
- paired AC-versus-AT `Psi` sensitivity;
- 99 balanced whole-lattice pseudo-phase assignments using seeds
  `2026080401..2026080499`, with all six `S3` assignments appearing 64 times per seed;
- the sequence-derived junction stop-load matrix, Spearman association of `omega_b` with
  `Psi_b`, and estimates above/below the frozen replication-median `omega_b` cutpoint.

Stop-load analyses are prespecified scope diagnostics, not selection variables and not a
way to waive the primary result. A concentration of the effect in high-stop-load blocks
narrows the interpretation toward a stop-avoidance/coding-plausibility prior; it does not
convert the model behavior into a spliceosomal phase rule.

## 5. Inference

- Unit: the seven-source block.
- Point estimates: equal-block means.
- Bootstrap: 20,000 whole-block draws within frozen chromosome-by-strand strata, seed
  `20260803`.
- `Psi`: ordinary two-sided percentile 95% interval.
- Resolution: `(1.96 + 0.8416212335729143) * SD` of bootstrap means after centering
  `Psi_b` within stratum.
- Directional family: max-absolute-studentized simultaneous 95% lower bounds over `D+`,
  `D-`, and the three `Psi_p` endpoints.
- Mismatch family: simultaneous two-sided max-studentized 95% intervals.
- Pseudo-phase test: exact add-one one-sided p-value across all 99 controls plus a paired
  target-minus-mean-control lower 95% bound.
- No prior and replication target outcomes are pooled for the verdict.

## 6. Capability gates

All gates are required. A failed gate returns `INSTRUMENT_UNRESOLVED` and licenses no
positive, null, inverted, or biological conclusion.

1. Construction audit passes for exactly 384 fresh gene-disjoint complete blocks.
2. Native-source `AG->AC` coupling lower 95% bound is above `+0.05`.
3. Phase-blind crossed `AG->AC` coupling lower 95% bound is above `+0.02`.
4. Bridge-effect 95% interval lies wholly inside `[-0.05,+0.05]`.
5. Observed centered-stratified `MDE80<=0.05`.
6. Phase-blind AC-minus-AT interval lies wholly inside `[-0.05,+0.05]`.
7. Protected true-base log-probability lower bound is above `-0.10` nat and masked-base
   accuracy retention is at least 0.80.
8. Donor and acceptor intact-margin lower bounds exceed floors recalibrated from the fresh
   native-source windows.
9. Logits and derived scores are finite, tokenized length is 1024 including special tokens,
   and repeat-forward maximum difference is `<=1e-8`.
10. A model-free result auditor rederives raw-row arithmetic, all gates, inference, hashes,
    manifests, checkpoint/code pins, and the one-shot output chronology.

## 7. One-shot execution and decision table

The cohort, construction audit, protocol, checkpoint files, executable code, schemas,
thresholds, seeds, and exact output paths are hash-frozen and independently audited before
the checkpoint is loaded. Native and crossed-cell outputs are immutable stages. There is no
adaptive development opening: all 384 replication blocks are run once, and the target is
interpreted only after every capability gate and the independent audit are complete.

| Replication outcome | Classification | Licensed interpretation |
|---|---|---|
| All gates pass; `Psi` lower 95% bound `>+0.05`; all five simultaneous directional lower bounds `>0`; target exceeds all 99 pseudo-phase controls with exact p `<=0.01` and paired excess lower bound `>0`; AC/AT passes; both mismatch intervals lie inside `[-0.05,+0.05]` | `ONLY_WHEN_COHERENT_IN_FROZEN_DOMAIN` | Material phase-matched preference, with mismatch coupling equivalent to zero only in this frozen model domain. |
| Same preferential criteria, but either mismatch interval fails equivalence | `MATERIAL_PREFERENTIAL_POSITIVE` | Material phase-matched preference in this frozen model domain; **only** is forbidden. |
| `Psi` lower bound `>0` but not `>+0.05`, with directional, pseudo-phase, AC/AT, and capability criteria passing | `DETECTABLE_MATERIALITY_UNRESOLVED` | Detectable preference; materiality at 0.05 unresolved. |
| All gates pass and the entire two-sided `Psi` interval lies in `[-0.05,+0.05]` | `BOUNDED_NULL` | This checkpoint/domain excludes a material annotated phase-matching preference of 0.05 or larger. |
| All gates pass and the upper `Psi` bound is `<-0.05` | `MATERIAL_INVERTED` | The prespecified compatible diagonal is materially disfavored in this checkpoint/domain. |
| Any gate fails | `INSTRUMENT_UNRESOLVED` | No target conclusion. |
| Any other pattern | `INCONCLUSIVE` | No directional phase-compatibility conclusion. |

## 8. Capability-conditioned and checkpoint-scoped interpretation

Every conclusion is indexed by checkpoint, behavior, context, and intervention family. A
capable null closes only the 0.05-scale crossed-lattice behavior at `SpliceBERT.1024nt` in
this 120-nt authentic-module geometry. It is not a biological no-go and does not show that
phase information is absent elsewhere in the network. A failed capability gate is an
instrument failure, not evidence that the model failed to learn the biology.

Conversely, “insufficiently embedded” is not an allowed post-hoc rescue. It becomes a testable
training hypothesis only under a preregistered checkpoint trajectory, matched sibling-corpus
contrast, or paired coherent-versus-scrambled targeted-training experiment in which both the
behavior and its proposed causal route strengthen with the relevant biological variation.

## 9. Gate to mechanism work

No new P3-carrier implementation result may be exposed unless this replication is classified
`MATERIAL_PREFERENTIAL_POSITIVE` or `ONLY_WHEN_COHERENT_IN_FROZEN_DOMAIN` after independent
audit. A passing behavioral result licenses a separate preregistration of carrier-register,
common-mode/differential, and mediation interventions. Historical P3 necessity is a candidate
and preflight source, not proof that P3 implements this newly replicated behavior.
