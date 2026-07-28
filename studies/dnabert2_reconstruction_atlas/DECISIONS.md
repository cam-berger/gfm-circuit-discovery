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

## 2026-07-27 — Freeze preflight v1 from corrected calibration v3

- Evidence: all four supply-qualified classes have ordered copy doses and four-copy means
  +0.090 to +0.206 nats/covered-nt. Irrelevant-exchange MDEs are 0.0077–0.0139 and CI
  half-widths are 0.0051–0.0096. All geometry, balance, split, replay, batch, and finite-score
  invariants pass.
- Frozen gates: positive effect ≥0.05 with dose ordering; irrelevant-exchange 95% interval
  inside ±0.02; null-noise MDE ≤0.015; CI half-width ≤0.01.
- Behavior thresholds: SESOI 0.02 nats/covered-nt, at least 350 gene clusters on each split,
  add-one permutation values, and Holm correction over all admitted class × contrast tests.
- Effect: splice acceptor, splice donor, start codon, and stop codon may enter the qualified
  preflight. No target outcome may run until that report passes unchanged.

## 2026-07-27 — Authorize development behavior screen

- Evidence: `preflight_qualified_v1.json` matches frozen config hash
  `e7efe5a2a2e1c7c571047466f442de6ae67cdc02f3a8e93ad3ae0471bca9e16c`,
  binds all four runtime inputs, and returns `PASS` for every mandatory gate.
- Decision: authorize the development split for the four admitted classes. The confirmation
  split remains sealed until the development artifact is immutable and its thresholds remain
  unchanged.
- Boundary: this is an instrument decision only; it creates no atlas row and no behavior claim.

## 2026-07-27 — Stop census v1 at the development behavior gate

- Evidence: all four admitted classes beat order-0/1/2 baselines, but none exceeds all three
  empirical controls. Acceptor, start, and stop are negative against every empirical family.
  Donor is +0.023 against composition/token matching (`p_add_one=0.1324`,
  `p_Holm=1.0`) and negative against same-motif and same-sequence controls.
- Decision: record D1, keep the confirmation split sealed, and do not enter context-source,
  candidate, or causal stages.
- Interpretation: DNABERT-2 reconstructs these short words better than low-order nucleotide
  models, but census v1 provides no evidence of annotation-specific excess recoverability.
- Boundary: this is not a confirmatory `NULL`, because the sealed split was not opened after
  the development gate failed. A future census must preregister a genuinely new source/class
  scope rather than retune v1.

## 2026-07-27 — Replace the four-class pilot with a multisource atlas amendment

- Trigger: the requested atlas spans promoters, poly(A) elements, TF-binding sites, branch points,
  polypyrimidine tracts, and enhancers; census v1 covered none of those classes.
- Sources: EPDnew human v006, PolyASite 2.0 GRCh38.96, UCSC’s multi-experiment ENCODE TF ChIP
  clusters, Mercer et al. high-confidence lariat-seq branch points, ENCODE Registry v4 GRCh38
  cCREs, and GENCODE v50/GRCh38 for original classes and tract derivation.
- Coordinate decisions: EPDnew contributes seven-nt TSS-centered cores; PolyASite contributes its
  declared signal hexamer and representative cleavage nucleotide; broad TF peaks and dELS cCREs
  contribute deterministic seven-nt centers explicitly labeled as derived assay anchors; Mercer
  calls require unique forward liftover, exact reverse round trip, and source-base agreement.
- Split decision: replace source-local gene clusters with shared 1-Mb locus blocks across every
  feature class. Exclude any complete assay window crossing a block boundary. This prevents
  overlapping loci from entering different splits through different databases.
- Supply amendment: a model-free build showed that 1,536 provisional GENCODE genes per provisional
  split left four baseline classes 3–17 loci below the frozen 384 quota. Increase only that
  upstream source pool to 2,048; keep the final quota and every model/statistical threshold fixed.
- Effect: 13 v2 classes are preregistered. V1 remains a preserved narrow development result and
  cannot authorize or answer v2. No v2 model outcome existed when this amendment was frozen.

## 2026-07-27 — Amend multisource interval anchors from outcome-blind control supply

- Trigger: the immutable v2 supply audit retained fewer than 32 complete control sets per split for
  seven-nt EPDnew promoter, ENCODE TF, and ENCODE dELS anchors. The first failure was usually an
  absent unannotated exact word; among survivors, random local candidates often lacked exact BPE
  geometry. No DNABERT-2 score or target outcome had been computed.
- Preserved result: `collection_qualification_v2.json` quarantines promoter, enhancer, and all
  three TF classes. It is not rewritten or described as a biological negative.
- V3 amendment: use symmetric five-nucleotide centers for those broad interval sources. Preserve
  PolyASite hexamers, exact cleavage/branch nucleotides, seven-nucleotide PPTs, and GENCODE
  splice/codon targets unchanged. Select the 128 local candidate positions by minimum nucleotide
  composition distance followed by the existing seeded hash; exhaustive exact-word occurrences
  remain included.
- Unchanged: all three empirical families, exact token/cover geometry, match calipers, split,
  quotas, preflight thresholds, SESOI, permutations, and multiple-testing rule.
- Effect: rebuild and re-audit as v3 before any model score. V2 remains a supply failure, not a
  target result.

## 2026-07-27 — Expand local candidate search to the frozen behavior floor

- Trigger: v3 makes all 13 classes preflight-eligible, but known complete-control supply remains
  below the frozen 250-locus behavior minimum for enhancer (210/216), poly(A) signal (198/195),
  promoter (222/234), PPT (87/98), CEBPB (196/215), CTCF (134/135), REST (146/159), and start
  codon (244/263). No model score had been computed.
- Diagnosis: after five-nt exact-word supply largely resolves, almost every remaining failure is
  the within-window family. The 128 composition-prioritized positions are a search bound, not a
  biological exclusion.
- V4 amendment: increase the same frozen candidate ordering from 128 to 512 positions per window.
  Preserve all target intervals, annotations, source filters, empirical families, exact token
  geometry, calipers, split, final quotas, and statistical thresholds.
- Stop rule: if a class still supplies fewer than 250 complete sets on either target split, score
  it only as explicitly underpowered or quarantine it; do not lower the behavior minimum from
  source-supply evidence.

## 2026-07-27 — Close supply amendments and preserve underpowered classes

- Evidence: v4 evaluates 512 rather than 128 composition-ranked local positions but retains exactly
  the same 6,448 complete matched targets as v3. Candidate depth therefore does not explain the
  remaining attrition; exact within-window BPE geometry and composition constraints do.
- Decision: make no further source, width, candidate, control, or threshold amendment. All 13
  classes meet the 32-locus preflight floor and enter instrument preflight. Only cleavage region,
  splice acceptor, branch point, splice donor, and stop codon meet the frozen 250-locus behavior
  floor on both target splits.
- Interpretation rule: development estimates may be recorded for every preflight-qualified class,
  but an underpowered class cannot behavior-qualify regardless of effect size or p-value. It is
  labeled underpowered, not null. Confirmation opens only development-qualified classes.

## 2026-07-27 — Authorize the 13-class multisource development census

- Evidence: `preflight_qualified_multisource_v4.json` is `QUALIFIED` against the exact v4
  collection, matches, audit, baseline corpus, config, and pinned checkpoint. Every class has 64
  geometry-eligible preflight examples. All graded-copy means are +0.082 to +0.226
  nats/covered-nt and monotone; all irrelevant-exchange, sensitivity, resolution, balance,
  invariant, geometry, replay, batch, and finiteness gates pass.
- Decision: authorize development target-control scoring for all 13 classes. Preserve the
  pre-scoring power labels from IF14. The confirmation split remains sealed.
- Boundary: this authorizes an assay; it is not evidence that any biological class has excess
  reconstruction.

## 2026-07-27 — Stop multisource v4 at the development behavior gate

- Evidence: the qualified screen scored 3,189 complete matched targets across 78
  class × contrast tests using 9,999 add-one permutations and Holm correction over the full
  family. Every class exceeds all order-0/1/2 baselines, but no class passes any of the three
  empirical-control contrasts at both the +0.02 SESOI and corrected-alpha gates.
- Powered boundary: poly(A) cleavage regions, splice acceptors, branch points, splice donors, and
  stop codons have 296–384 independent locus blocks and therefore meet the frozen 250-cluster
  floor. All five fail empirical specificity.
- Underpowered boundary: enhancer, poly(A) signal, promoter, PPT, CEBPB, CTCF, REST, and start
  codon have 87–244 complete development clusters. Their estimates are retained, but none can
  qualify and none is a biological null.
- Decision: record D2; keep confirmation sealed; stop context-source, internal-candidate, and
  causal stages. A direct confirmation invocation was checked and refused before model loading
  with `no development-qualified classes; confirmation remains sealed`.
- Interpretation: across this multisource development cohort, DNABERT-2 reconstructs the assayed
  words better than low-order sequence models, but the effect is not specific to biological
  annotation under the preregistered empirical comparisons.

## 2026-07-27 — Replace annotation-excess entry with a geometry-qualified context margin

- Trigger: annotation-specific excess reconstruction was a poor mechanistic entry gate, and
  SpliceBERT-style alternative margins needed a DNABERT-2 BPE fairness audit.
- Measurement: score the true simultaneously masked target against the log-sum-exp of every
  Hamming-one target alternative that preserves the complete BPE offset map and masked input;
  divide by covered nucleotides. Use the identical alternative intersection in reference,
  structured-context, and irrelevant-context variants.
- Primary effect: `margin(irrelevant distal shuffle) − margin(class-specific proximal shuffle)`.
  Both shuffles preserve the target and complete scored cover, sequence length, mononucleotide
  composition, and exactly the same number of changed positions.
- Frozen target gate: +0.01 nats/covered-nt, positive 95% cluster-bootstrap lower bound, add-one
  cluster sign permutation, and Holm correction across the full preregistered class family.
- Boundary: the distal null controls composition and intervention dose, not distance. A positive
  result is a proximal-context model behavior and cannot by itself be called cellular biology or
  an internal mechanism.

## 2026-07-27 — Quarantine acceptor and donor from the margin instrument

- Outcome-blind supply retained at least 82 targets per class and split with zero invariant
  violations. Control-only calibration used 64 development blocks per class and no
  structured-versus-irrelevant target contrast.
- Eleven classes passed the graded copy-margin control. Splice acceptor's four-copy 95% lower
  bound was negative; donor's lower bound was negative and its dose means were non-monotonic.
- Decision: freeze acceptor and donor as `INSTRUMENT_UNRESOLVED`. Freeze the remaining preflight
  ceilings at 0.007 irrelevant equivalence, 0.006 MDE, and 0.004 CI half-width.
- A first qualified artifact incorrectly required the quarantined classes to meet the admitted
  class count gate. Preserve it, correct only that membership check, and rerun under the unchanged
  config. The corrected report is `QUALIFIED` for all 11 admitted classes.

## 2026-07-27 — Open confirmation for ten development-qualified context effects

- Development scored 2,059 targets across 11 classes. Ten met all gates: enhancer, poly(A)
  cleavage and signal, promoter, branch point, PPT, CEBPB/CTCF/REST peak anchors, and start codon.
- Stop codon did not qualify (+0.00330; interval crossed zero; Holm `p=0.2401`) and remained sealed.
- Decision: open only the ten qualifying classes on untouched confirmation locus blocks, preserving
  the exact score, contexts, alternatives, SESOI, and inferential family.

## 2026-07-27 — Record nine replicated context-margin classes

- Confirmation replicated nine of ten opened effects. Start codon was strongest (+0.0479), followed
  by enhancer anchor (+0.0384), poly(A) cleavage (+0.0348), and branch point (+0.0304).
  Poly(A) signal, PPT, and all three TF peak-anchor classes also passed.
- Promoter did not replicate (+0.00066; CI crossed zero; Holm `p=0.4548`) and is recorded as a
  development-only failure.
- Decision: record D3 and allow a future, separately frozen internal candidate stage for only the
  nine replicated classes. No attention, attribution, head, MLP, mediation, or causal claim is
  made by this census.
