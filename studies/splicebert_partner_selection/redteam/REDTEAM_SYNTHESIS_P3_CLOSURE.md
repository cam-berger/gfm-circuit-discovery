# P3 closure — final pre-freeze red-team synthesis

**Date:** 2026-07-21

**Reviewed:** `PREREGISTRATION_P3_CLOSURE.md`, DRAFT v2
**Outcome:** **freeze-after-engineering**; the estimand and controls are adequate, but no untouched
forward pass is licensed until the package, independent audit, sealed runner, and analysis in §9 of the
preregistration exist and pass on development data.

## End-goal relevance

The study has one purpose: determine whether the frozen SpliceBERT checkpoint causally uses its learned
period-3 positional component to support donor–acceptor coupling, and whether that use is enriched in
coding sequence. It does not test the evolutionary origin of the carrier, a spliceosomal reading-frame
sensor, the 10.63-period component, another model, or a broader UTR/noncoding taxonomy. Those questions
must not delay or enter this confirmation.

## Why the sequence-edit path was retired

The insertion diagnostic and fixed-length sequence rotations were informative development failures, not
confirmatory evidence. Insertions alias phase with edit dose and alter coding content. Strict rotations
had zero usable construction yield, while relaxed rotations changed local MaxEnt splice-site landscapes;
their apparent coupling effect moved toward zero after nuisance adjustment. No sequence edit in that
family cleanly isolates the learned carrier phase.

The replacement surgery changes only the selected component of the learned position embedding inside a
frozen block. Nucleotides, token IDs, genomic coordinates, length, codons, annotated motifs, MaxEnt
scores, and every other sequence-derived variable are identical across cells. This removes the
load-bearing confound rather than statistically adjusting for it.

## Lens 1 — estimand and inference

### Accepted design

- The endpoint is the already-established model-native coupling behavior, not attention alone:
  `C = donor_margin(clean) - donor_margin(partner_acceptor_destroyed)`.
- The primary contrast compares near-native shifts `{3,6,9}` with their adjacent off-register shifts,
  then subtracts the identical manipulation in an annotation-clear upstream-intron block. This identifies
  phase-aligned use in CDS beyond a generic local position-embedding perturbation.
- Blocks, directions, and repeated shifts are averaged within gene before inference. The gene is the
  inferential unit.
- There is one primary endpoint. The L3H1 phasor is descriptive because it failed development stability;
  it cannot increase multiplicity or rescue the behavioral result.
- The complete 95% interval determines material, detectable, bounded-null, or inconclusive status. A
  nonsignificant mean is never called a null.
- The 943-gene census is fixed with no result-dependent extension. Development variance gives a projected
  80% MDE of about 0.052 nat, close to the inherited 0.05-nat SESOI.

### Remaining scope boundary

`Psi_C` is an interaction between block context and modulo phase. It supports present-day causal use of
the learned component; it does not prove that coding frame caused the component to arise during training.
The context contrast is secondary and can license only **coding-context enrichment**, not frame origin.

## Lens 2 — biological and perturbation validity

### Accepted design

- CDS blocks are fully coding, have a single comprehensive-GENCODE codon register, are splice-boundary
  clear, and are safe for every frozen shift.
- The intronic control is within example and uses the same block length, direction, offsets, surgery
  magnitude, and model behavior.
- A matched non-CDS-exon arm addresses the main residual biological alternative: a CDS-versus-intron
  difference could reflect exon/intron architecture rather than coding context. Matching is outcome-blind
  and the frozen 52-pair match satisfies the predeclared balance threshold.
- Because the available non-CDS examples are exonic regions of protein-coding genes, the correct label is
  **non-CDS exon**, not noncoding gene, 5′ UTR, or 3′ UTR.

### Interpretation boundary

The intervention is artificial hidden-state surgery. A positive result shows that SpliceBERT's learned P3
direction is usable by its existing computation. It does not show that real mutations perturb splicing by
the same mechanism or that cells use a triplet carrier to choose splice partners.

## Lens 3 — component specificity

The strongest remaining alternative is that any same-size periodic perturbation would produce the
effect. The signed-permutation family is an appropriate structured null because every control preserves
the P3 spatial waveform, full spatial Gram matrix, row norms, Fourier support, block geometry, and
intervention RMS. Only alignment to the model's learned hidden features changes.

The direction-specificity claim requires both frozen conditions: the learned direction must exceed all 19
controls (`p=1/20`) and the paired learned-minus-mean-control interval must exclude zero. The 19 seeds and
150-gene subset are fixed in advance. Failure does not negate phase sensitivity; it narrows the result to
a same-geometry position-embedding effect with learned-direction specificity unresolved.

The global P3 deletion is the substantive positive control. Reconstruction and full restore are only
plumbing checks and must not be described as scientific sensitivity evidence.

## Lens 4 — instrument and implementation

The development instruments currently support a freeze:

- reconstruction, determinism, no-op, attention normalization, and mechanical modulo-recovery checks pass;
- continuous masked-base health passes in CDS and non-CDS contexts at the frozen −0.05-nat floor;
- signed-permutation controls preserve geometry within the frozen numerical tolerances;
- global P3 deletion nearly abolishes native coupling, demonstrating a live endpoint;
- development behavior separates the learned P3 direction from all 19 structured controls.

One apparent non-CDS coarse-accuracy failure came from only 32 discrete probes. The predeclared continuous
784-probe true-base log-probability assay passed and supersedes that noisy diagnostic. Both artifacts must
remain preserved; the coarse result may not be silently rewritten.

The untouched runner must still enforce package hashes, a clean frozen revision, exact checkpoint and data
identities, immutable resumable shards, and no outcome summaries before all planned shards complete. The
independent analyzer must reconstruct gene-level contrasts without importing runner summaries.

## Lens 5 — exposure, multiplicity, and publication risk

- The honest cohort label is **P3-analysis-untouched**, not globally never-seen: historically used
  validation/test chromosomes and every recorded P3/C8/C34/C35/shift gene are excluded.
- The primary has one endpoint. Direction specificity and matched context are gatekept by a material
  primary and cannot rescue it.
- The context arm uses all 52 pre-matched pairs. Its smaller sample can establish enrichment only if its
  frozen interval clears zero; otherwise coding enrichment remains unresolved.
- Confirmation is one-shot. Any post-outcome bug repair or design change is labeled outcome-aware
  verification and cannot replace the clean result.
- Figures must be generated from immutable recorded artifacts, and failed rotation/phasor routes remain in
  the record.

## Freeze verdict

**Freeze-after-engineering; no redesign required.** The final untouched run is licensed only after:

1. one immutable package freezes all IDs, pairs, seeds, thresholds, source hashes, and development evidence;
2. an independent audit reconstructs counts, roles, sequences, exposure exclusions, and match balance;
3. the exact final runner and analyzer pass development-only smoke tests and all instrument gates;
4. code, package, audit, and hashes are committed with a clean worktree; and
5. the runner refuses any mismatch and writes no partial biological interpretation.

If those conditions pass, the next scientifically relevant action is the untouched run itself. Additional
cohorts, mechanisms, carrier bands, or visualizations before that run would be scope drift.
