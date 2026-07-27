# DNABERT-2 Reconstruction Atlas decision log

> **Tier: T2 — provenance.**

## 2026-07-27 — Reset from favored hypotheses to a biological census

- Trigger: the atlas request and the prior DNABERT-2 instrument records.
- Evidence available: the unmerged core-promoter branch reached a qualified preflight but found the
  forward estimand ceiling below its precision target; the coupling branch stopped after BPE
  geometry and donor-specificity power made the assay non-adjudicable.
- Decision: start from current `origin/master` in a separate clean worktree and carry forward only
  the pinned adapter and explicit instrument lessons.
- Alternatives considered: extend either old branch; merge both; retain their scripts as active.
- Effect: no legacy target hypothesis or outcome is a live atlas claim. Legacy commits are indexed,
  not copied into the active study.

## 2026-07-27 — Make per-covered-nucleotide log probability primary

- Trigger: variable-length BPE tokens can overhang an annotated nucleotide interval.
- Decision: normalize the summed true-token log probability by the full minimal token cover.
- Rejected: normalize only by annotated length; average tokens equally; retokenize a string
  containing literal mask text.
- Effect: every result records annotated length, covered length, overhang, offsets, and token count.

## 2026-07-27 — Permit alternative margins only under exact geometry

- Trigger: editing a target can change BPE segmentation and ALiBi token distance.
- Decision: an alternative margin is eligible only if the full offset map is unchanged and every
  unmasked token ID is identical.
- Effect: ineligible alternatives are recorded rather than forced into a biased comparison.

## 2026-07-27 — Gate mechanisms on both excess behavior and context specificity

- Trigger: raw motif likelihood can reflect token frequency or local spelling.
- Decision: internal screens are prohibited until the site class beats empirical controls and a
  biologically structured, token-invariant source-region intervention.
- Effect: an attractive attention map cannot create an atlas mechanism row.

## 2026-07-27 — Keep the preflight threshold file as a draft

- Trigger: no atlas-wide development variance or positive-control calibration exists yet.
- Decision: implement the gate schema and lock candidate, but leave site-specific SESOIs and minimum
  cluster counts unset until outcome-blind/calibration evidence supports them.
- Effect: no target runner can qualify from the draft alone; a frozen owner-approved config and
  matching preflight report are required.

## 2026-07-27 — Bound census v1 before target outcomes

- Trigger: GENCODE v50 basic is locally available and has an implemented source adapter; the other
  declared source families still lack admitted release manifests or source-specific adapters.
- Decision: the first census slice uses protein-coding genes on `chr1`–`chr22`, `chrX`, and `chrY`;
  it samples at most one target per gene × site class and 512 clusters per split.
- Split: 30% development, 20% baseline training, and 50% sealed confirmation by one deterministic
  cluster hash. This replaces the internally inconsistent draft wording that allocated 40% + 60%
  while also requiring a third disjoint corpus.
- Controls: one per empirical family, drawn from 256 outcome-blind positions per admitted window
  plus exhaustive exact-motif occurrences. GENCODE exon boundaries and start/stop annotations are
  streamed across all transcripts overlapping the sampled windows so they cannot masquerade as
  “unannotated” controls.
- Effect: splice donor, splice acceptor, start codon, Kozak, and stop codon enter outcome-blind
  supply qualification. Every other catalog class remains explicitly deferred, not negative.

## 2026-07-27 — Separate calibration, qualification, and target execution

- Trigger: the foundation enforced a frozen preflight report but did not yet implement a runner
  capable of producing one.
- Decision: run a development-only calibration artifact first, using 1/2/4-copy plants,
  irrelevant-region exchanges, geometry checks, matching balance, replay/batch invariance, and
  cluster precision. Freeze thresholds and admitted classes only from those controls; then rerun
  the same instrument as the qualified preflight.
- Effect: no annotated-target versus empirical-control contrast may be computed until the frozen
  qualified report exists and binds the exact collection, matches, matching audit, and baseline
  corpus by SHA-256.

## 2026-07-27 — Resolve empirical-control count from outcome-blind supply

- Trigger: the initial three-controls-per-family supply audit retained 460/460 acceptors,
  332/342 donors, 127/101 start codons, and 229/208 stop codons on development/confirmation.
  Kozak retained none. No MLM score or target contrast had been computed.
- Decision: use the foundation's original one-control-per-family default. On the identical
  candidate pool it retains 501/503 acceptors, 451/445 donors, 354/350 start codons, and
  419/393 stop codons. Kozak retains only 5/6 and is quarantined at Phase A.
- Rationale: every family remains separate and exactly geometry/composition matched, while the
  one-control design preserves enough independent gene clusters for the frozen precision
  calibration. The rejected three-control audit remains immutable runtime evidence.
- Effect: four GENCODE classes enter preflight calibration. Kozak and all non-GENCODE catalog
  classes remain untested, not negative.

## 2026-07-27 — Freeze after calibration-v1 positive-control anomaly

- Trigger: calibration v1 passed every mechanical/invariant check but the random downstream
  copy plant was not reliably positive across classes. Donor's four-copy mean was −0.0043;
  acceptor/start saturated or reversed from dose 2 to dose 4; only stop codon was strictly
  monotone.
- Decision: record `INSTRUMENT_UNRESOLVED`, keep the lock draft, and compute no empirical
  target-control outcomes.
- Diagnosis: v1 chose a random eligible region anywhere downstream after a ≥32-nt buffer. That
  design conflates whether the score can detect supplied target information with whether
  arbitrary distal context is used. The positive control should exercise the nearest eligible
  non-overlapping context while retaining the same target and geometry invariants.
- Effect: v1 remains immutable. Any placement amendment must be committed before a second
  calibration and cannot change after target outcomes.

## 2026-07-27 — Amend copy-plant placement before calibration v2

- Trigger: the calibration-v1 diagnosis above; still no empirical target-control outcome.
- Frozen amendment: choose the nearest BPE-geometry-eligible downstream region after the complete
  scored cover, with a four-nucleotide buffer. Keep the same 1/2/4 copy doses, exact target-cover
  word, class sample, negative exchange, score, normalization, and monotonicity requirement.
- Rationale: the positive control asks whether supplied sequence information can affect the masked
  target at all. Arbitrary distal placement tests a different biological-context question and
  belongs after behavior qualification.
- Stop rule: if the amended control still fails for a class, that class is
  `INSTRUMENT_UNRESOLVED` and cannot enter target execution.

## 2026-07-27 — Correct monotonicity and resolution definitions before calibration v3

- Trigger: v2 produced strongly ordered copy effects for all four classes, but donor's one-copy
  effect was negative. The implementation required `0 < dose1 < dose2 < dose4`; the draft only
  required monotonicity across the graded doses.
- Correction: evaluate `dose1 < dose2 < dose4`; the separately frozen positive-effect threshold
  determines whether the control is actually positive.
- Resolution correction: estimate the detection floor and CI half-width from the matched
  irrelevant-exchange null noise, not heterogeneous positive-copy response variance. The positive
  plant remains the sensitivity injection and must exceed its threshold.
- Effect: both changes are frozen before v3 and before any empirical target-control outcome.
  V1 and v2 artifacts remain unchanged.
