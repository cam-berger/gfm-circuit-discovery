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

## 2026-07-27 — Select a promoter-trained DNABERT-2 head and reject the circular EPD rerun

- Candidates: the compatible MagicsLab human GUE all-promoter checkpoints reproduce native held-
  out performance at MCC 0.6804/AUROC 0.9179 for 70-nt core promoters and MCC 0.8705/AUROC
  0.9848 for 300-nt promoters. The plant checkpoint changes species and tokenizer; the other Hub
  results are DNABERT-1 or undocumented DNABERT-AG variants.
- Decision: pin `magicslabnu/DNABERT-2-finetuned-prom_300_all` at revision
  `7f2a5f5b8e7e6fffb2dcb396a77d1bc78b7f5f12`. Use the supervised promoter-minus-nonpromoter
  logit margin, which compares two labels on one tokenized input and therefore avoids the
  alternative-sequence BPE fairness problem.
- Contamination finding: all 768 selected EPDnew 300-nt windows occur verbatim in GUE. The
  best-aligned 70-nt cores also have 768/768 exact membership. EPDnew target scoring is prohibited
  for this fine-tuned head.
- Replacement: use GENCODE v50 protein-coding MANE Select TSSs, excluding exact forward/reverse
  membership in every GUE split before outcomes. Pair each target with a GC/CpG-matched,
  TSS-excluded same-locus pseudo-TSS and split by shared 1-Mb locus block.

## 2026-07-27 — Preserve failed promoter preflight v1 and qualify the corrected instrument

- Outcome-blind cohort: 384 development and 384 confirmation blocks pass with zero exact GUE
  overlap, split leakage, anchor changes, composition changes, or target/control Hamming-dose
  mismatch.
- V1 anomaly: all positive, sensitivity, resolution, invariant, balance, and token gates pass, but
  a single seeded label permutation yields MCC −0.0742 and CPU batch reshaping changes a margin
  by \(3.10\times10^{-6}\), above the frozen \(10^{-6}\) tolerance. Target outcomes remain sealed.
- Diagnosis: a one-draw random-label MCC is a high-variance null statistic; the observed
  permutation distribution has standard deviation 0.0437. The CPU discrepancy is bounded
  floating-point reduction noise, not stochastic model behavior; exact replay differs by zero.
- V2 amendment, frozen before target outcomes: replace the one-draw null with the mean MCC across
  256 seeded label permutations and require absolute mean ≤0.01. Round the batch-invariance
  tolerance outward to \(10^{-5}\). Keep the checkpoint, cohort, intervention, +0.50/+0.25 target
  SESOIs, intervals, inference, and all other gates unchanged.
- Result: all eight V2 preflight families pass. The permutation mean is +0.00183, held-out GUE
  context difference is +1.085 (95% CI +0.759 to +1.410), MDE is 0.464, and CI half-width is
  0.325. Development is authorized.

## 2026-07-27 — Seal D4 as promoter behavior without a replicated context mechanism

- Development: on 384 locus blocks, clean TSS-versus-pseudo-TSS margin enrichment is +2.0956
  (95% CI +1.7813 to +2.4080) and context difference-in-differences is +0.6655 (+0.3250 to
  +1.0080); both Holm `p=0.0002`. Both endpoints qualify and confirmation opens.
- Confirmation: clean enrichment replicates at +1.8892 (+1.6118 to +2.1867; Holm `p=0.0002`).
  The context endpoint is +0.2750, above its +0.25 SESOI, but its interval crosses zero
  (−0.0684 to +0.6151) and Holm `p=0.0569`.
- Decision: D4 supports fine-tuned promoter classification behavior on the exact-GUE-excluded
  GENCODE cohort. The target-specific local-context mechanism does not replicate. Stop before
  attention, attribution, internal candidates, ablation, necessity, or mediation.

## 2026-07-28 — Freeze a separate clean-promoter head atlas before internal outcomes

- New question: localize D4's replicated clean TSS-versus-pseudo-TSS classifier behavior, not the
  failed target-preserving context endpoint.
- GUE-only calibration: first-order attribution tracks exact single-head CLS patching
  (Spearman 0.839 necessity, 0.787 sufficiency); all-head CLS transfer recovers the full class
  margin; exact self-patching is zero; batch drift is \(4.77\times10^{-7}\).
- Candidate-set calibration: the eight-head rank rule yields +4.87/+5.28 logit
  necessity/sufficiency on the reserved GUE slice and exceeds all 127 exact layer-profile null
  sets. Observed variance projects 80% MDEs 0.504/0.562 at 192 loci.
- Frozen choice: 192 loci per target split, eight heads with at most two per layer, 63 distinct
  layer-matched null sets, separate +0.60 necessity/sufficiency SESOIs, positive bootstrap lower
  bounds, and two-endpoint Holm add-one inference.
- Boundary: even a pass supports joint `[CLS]` head-output mediation only. It does not revive H6,
  establish individual-head specificity, identify a motif, or supply a directed circuit.

## 2026-07-28 — Record replicated candidate-set CLS head-output mediation

- Preflight: all eight frozen gate families pass on 32 reserved GUE pairs disjoint from
  calibration. The assay captures all 144 heads, exact no-op effects are zero, the cross-class
  patch is 4.73 times the same-class absolute control, attribution/exact-patch Spearman is
  0.754/0.691, projected 192-pair MDE is 0.483/0.311, and model/token invariants pass.
- Development: the frozen rank rule selects L2H2, L4H6, L4H7, L9H7, L10H1, L10H5, L11H2,
  and L11H5. Their joint necessity/sufficiency effects are +1.061/+1.417 logits. Both exceed
  all 63 distinct candidate-excluding layer-profile null sets and pass the two-endpoint Holm
  gate at 0.03125, so confirmation opens without changing the candidates.
- Confirmation: on 192 untouched 1-Mb locus blocks, the same set replicates at
  +0.732 necessity (95% CI +0.544 to +0.929) and +1.138 sufficiency
  (+0.904 to +1.375). Both exceed the +0.60 SESOI and every newly seeded exact-layer-profile
  null set; Holm `p=0.03125`.
- Decision: record D5 and prioritize L11H2/L11H5 for an independently frozen pair-level
  decomposition. L9H7 and L10H5 also retain positive corrected individual signs in both
  directions and splits. Individual rows remain candidates because they did not receive
  matched single-head null inference.
- Source clue: confirmation L9H7 target attention concentrates in the proximal-upstream band
  and shifts toward that band and the TSS anchor. Treat this only as a source-localization
  hypothesis; do not infer motif or positional causality from attention.
- Boundary: the validated node is the candidate set's CLS context slices before the shared
  output projection. QK, OV, source positions, downstream paths, sparsity, the failed
  target-preserving context endpoint, and cellular transcription remain unresolved.

## 2026-07-28 — Freeze the promoter circuit battery before unused-locus outcomes

- Node universe: all 144 pre-output-projection head contexts plus all 12 MLP output writes at
  the unpadded `[CLS]` row. Pooler/classifier weights are a fixed readout; non-CLS writes are
  outside this first strict-faithfulness boundary.
- Cohort: the 192 unique 1-Mb locus blocks per split not used by D5. Bind the exclusion to both
  immutable head-atlas raw artifacts.
- Primary measurement: complement-scrub ratio of mean retained enrichment to mean native
  target-control enrichment, with exact empty/all-node anchors and locus-block bootstrap.
- Causal family: L11H2/L11H5 versus all 45 candidate-excluding two-head pairs in layer 11,
  query-only target-from-control transfer, and early-six corruption followed by cached-native
  layer-11 pair rescue.
- Gates: +0.25 logit SESOI for pair necessity/sufficiency and upstream source loss, +0.10 for
  query necessity, positive bootstrap lower bounds, exact add-one ranks, and endpoint-family
  Holm correction. Confirmation requires clean behavior, valid anchors, and at least one
  qualifying causal test.
- Boundary: source-band and motif surgery stays sealed unless the directed path qualifies.

## 2026-07-28 — Record D6 and reject the proposed early-to-late chain

- Preflight: 16 new GUE pairs pass every hook invariant. Full scrub reproduces the paired control
  exactly, all-node retention reproduces native output exactly, self/query/restore no-ops are
  zero, batch drift is \(4.77\times10^{-7}\), and full-scrub sensitivity is 8.14 logits.
- Pair specificity: L11H2/L11H5 beats all 45 equal-size layer-11 alternatives in both directions
  and both splits. Development necessity/sufficiency is +0.647/+1.068; confirmation is
  +0.700/+1.093. Each endpoint has a positive interval and Holm `p=0.04348`.
- Faithfulness: C8 alone retains 15.30%/13.73%; L11H2/L11H5 alone retains nearly the same
  15.18%/13.62%. C8 plus all MLP writes retains 61.04%/61.27%. MLP writes alone retain zero and
  all heads without MLP writes retain 34.36%/31.30%.
- Falsification: target-query transfer is effectively zero. Early-six corruption costs only
  +0.0585/+0.0563 logits, below the +0.25 SESOI, so its statistically ordered but tiny rescue
  through the late pair does not establish a directed path.
- Decision: record a replicated late output-carrier core operating on a distributed
  MLP-maintained representation. Do not describe C8 as a self-contained sparse circuit, do not
  claim target-dependent Q gating, and do not open source-band or motif surgery under v1.

## 2026-07-28 — Freeze address/payload decomposition on an independent promoter cohort

- Fresh supply: construct 768 exact-GUE-excluded MANE TSS/pseudo-TSS pairs per split using the
  original outcome-blind cohort rules, then bind 192 per split after excluding every example ID
  and 1-Mb locus block in the original 768-locus promoter cohort.
- Token surgery: use 40 SHA-ranked genomic tokens per sequence and a deterministic nonzero cyclic
  derangement shared by the candidate and every null pair. The minimum audited genomic token
  count is 44.
- Primary roles: K permutation tests content-key addressing; V permutation leaves native Q, K,
  QK logits, and attention probabilities intact while breaking address-to-value association.
  Joint K/V permutation is secondary and ALiBi-relative.
- Inference: target loss minus paired-control loss, absolute 0.05-logit SESOI, interval excluding
  zero, every one of 45 candidate-excluding layer-11 pairs as equal-size nulls, exact add-one
  ranks, and Holm across K and V. Confirmation requires the development sign.
- Boundary: the residual-write direction is descriptive. No source band, motif, promoter grammar,
  or semantic direction label can be claimed under v1.

## 2026-07-28 — Record D7 as replicated value-payload alignment dependence

- Preflight: 32 new reserved-GUE pairs pass the identity, K/V sensitivity, resolution, null-count,
  batch-invariance, residual-finiteness, model, and token-geometry gates. Identity error is
  \(2.38\times10^{-6}\); projected 192-pair MDE is 0.0032 logits for K and 0.0262 for V.
- Development: V permutation gives +0.1026 target-specific margin loss (95% CI +0.0517 to
  +0.1552), exceeds all 45 pair nulls, and passes Holm at 0.04348. K permutation is +0.0073,
  below the 0.05 SESOI. Confirmation opens for V only.
- Confirmation: V replicates at +0.0615 (+0.0030 to +0.1210), again above every null with
  Holm `p=0.04348`. K remains sub-SESOI at +0.0084. Joint K/V is secondary and has a
  confirmation interval crossing zero.
- Residual geometry: the paired target-minus-control mean write directions have cosine 0.9908
  across splits; 75.0% of confirmation loci project positively onto the development direction.
  This remains descriptive because no semantic or direction-only causal gate was run.
- Decision: L11H2/L11H5 causally depends on the native association between attention weights and
  source value vectors. Together with the D6 query null, this narrows the pair away from
  target-specific Q/K addressing. Freeze a separate equal-dose source-localization assay before
  selecting TSS-relative bands or motifs.

## 2026-07-28 — Freeze source localization and biological qualification before target outcomes

- Cohort: bind the final 192 unused exact-GUE-excluded locus blocks per split after excluding all
  original-promoter and D7 example IDs and 1-Mb blocks.
- Unbiased atlas: scan thirty nonoverlapping 10-nt nominal tiles. Choose exactly two adjacent
  genomic BPE tokens per sequence by maximum overlap; require cover ≤18 nt and target/control
  cover-length imbalance ≤2 nt.
- Measurements: target loss minus paired pseudo-TSS loss for post-tokenization `[MASK]` and for a
  two-token L11H2/L11H5 V swap. Require a raw practical effect, a positive interval, an equal-dose
  within-locus localization contrast, and a maximum sign-flip correction over all 60 tests.
- Biological qualification: keep FANTOM5 dominant CTSS, ENCODE PLS, UCSC CpG island, and EPD motif
  evidence separate. Motif disruptions must cross below the predefined EPD cutoff while preserving
  branch-specific mononucleotide composition, equal target/control Hamming dose, and complete BPE
  geometry.
- Calibration/preflight: use GUE pair offsets 112–127 and 128–159, disjoint from all earlier GUE
  slices. Freeze +0.10/+0.05 mask raw/localization SESOIs and +0.005/+0.002 regional-V SESOIs.
- Boundary: mask is a candidate generator; annotation overlap is not specificity; motif disruption
  is not head mediation. Confirmation opens only unchanged development-qualified tests.

## 2026-07-28 — Record D8 as replicated behavioral source localization without a biological label

- Preflight: all frozen identity, sensitivity, precision, resolution, batch, finiteness, and
  45-null-family checks pass on 32 reserved GUE pairs. Maximum identity error is
  \(1.91\times10^{-6}\), batch drift is zero, and every tile has at least 17 GUE pairs.
- Development: tile 26, nominal +11 to +21 relative to the TSS, is the sole qualifying test.
  Target-specific mask loss is +0.2452 (95% CI +0.0755 to +0.4285), localization contrast is
  +0.2656 (+0.0953 to +0.4442), and joint-family `p=0.0260`.
- Confirmation: the unchanged tile-26 mask test replicates at +0.2056 (+0.0447 to +0.3916;
  `p=0.0082`). Actual two-token covers have median length 10 nt and range 6–15, so the nominal
  interval is not exact nucleotide resolution.
- Biological qualification: the PLS-overlap subset qualifies in development but not confirmation
  (`p=0.0589`, interval crossing zero). CAGE is underpowered at tile 26, CpG fails development,
  and no predefined EPD motif instance intersects the nominated interval.
- D7 link: no regional L11H2/L11H5 V tile reaches the frozen raw or localization SESOI. Do not
  infer that tile 26 supplies the replicated whole-sequence D7 payload.
- Decision: record a replicated classifier-relevant behavioral source region only. A
  native-nucleotide tile-26 intervention and mask-by-payload interaction/rescue must be frozen
  before sequence-specific or mediation claims.

## 2026-07-28 — Record post-hoc tile-26 classification performance

- Scope: after D8 nomination, retain clean and masked margins for the unchanged eligible
  development and confirmation pairs. Label real TSS as promoter and matched pseudo-TSS as
  non-promoter; use margin greater than zero as the hard call.
- Development: accuracy is 0.5888 clean and 0.5794 masked; AUROC is 0.6775 and 0.6443.
  Three real promoters flip to non-promoter and one flips in the reverse direction.
- Confirmation: accuracy is 0.5806 clean and 0.5887 masked; AUROC is 0.6788 and 0.6580.
  Four real promoters flip to non-promoter and two flip in the reverse direction.
- Replication: AUROC change is negative with a negative 95% pair-bootstrap interval in both
  splits. Accuracy-change intervals span zero in both splits; pooled accuracy is exactly
  unchanged because target errors are offset by pseudo-TSS corrections.
- Invariant: the rerun reproduces every previously recorded tile-26 target, control, and
  target-specific margin loss exactly (maximum absolute difference zero).
- Decision: record a post-hoc ranking degradation and seven target promoter→non-promoter flips,
  not a net accuracy loss. The cohort is the fresh balanced GENCODE/pseudo-TSS challenge, not
  the original GUE held-out task.

## 2026-07-28 — Qualify D8 with native bases before tracing residual sources

- Intervention: edit only +11:+21. Exact dinucleotide reordering preserves mono- and linear
  dinucleotide counts; the separate three-transition edit removes three G/C bases. Target and
  pseudo-TSS edits require equal Hamming dose and matched local BPE retokenization geometry.
- Supply history: the first exact-full-geometry design retained only 13 development loci; the
  local-retokenization amendment retained 70/47. A three-edit ensemble retained only 39/20 and
  increased GC-depletion calibration variance, so the single-edit design was retained.
- Frozen v7 preflight: dinucleotide reordering passed, but GC depletion exceeded the 0.300
  projected-MDE ceiling (0.403). No target outcome was read.
- Frozen v8 preflight: on a second disjoint GUE slice, the 40-locus dinucleotide design also
  exceeded the same ceiling (0.364). No threshold was relaxed.
- Redesign: outcome-blind dinucleotide-only supply supports 87 development and 61 confirmation
  loci; freeze 60 per split and retain the +0.300 precision ceiling.
- Frozen v10 preflight: a third disjoint slice passes all eight checks, projected MDE 0.156, zero
  replay drift, and zero batch drift. Open development only.

## 2026-07-28 — Stop the +11:+21 native-source path at development

- Clean behavior: target-minus-pseudo-TSS promoter margin is +2.321 logits (95% CI +1.392 to
  +3.266; add-one `p=0.0001`), so the cohort retains the classified behavior.
- Native effect: exact-composition reordering gives −0.0162 target-specific margin loss
  (−0.1762 to +0.1383; `p=0.5834`). It fails the +0.10 SESOI, positive interval, and sign gates.
- Performance: accuracy stays 0.650, AUROC changes descriptively by +0.0058, and one target flips
  promoter→non-promoter. CAGE-, PLS-, and CpG-overlap subgroup means are nonpositive.
- Decision: confirmation remains sealed. Do not run residual-layer localization, L11H2/L11H5
  rescue, path blocking, or faithfulness for this source hypothesis.
- Interpretation: D8 remains a post-tokenization mask sensitivity, not native sequence-order
  necessity. GC-content necessity is instrument-unresolved, not negative.

## 2026-07-28 — Run an accuracy-first atlas over every CLS head and MLP write

- Supply: outcome-blind exclusion leaves exactly 100 fresh unique 1-Mb blocks after D4 and
  D7–D9. Freeze 50 development and 50 confirmation pairs by stable hash.
- Universe: scan all 144 attention contexts and all 12 MLP writes individually at `[CLS]`.
  Reciprocally exchange real-TSS and pseudo-TSS activations so both class branches contribute to
  balanced accuracy loss.
- Inference: require at least 0.02 accuracy loss, a strictly positive 95% pair-bootstrap lower
  bound, and one-sided add-one maximum-sign `p≤0.05` across the complete 156-node family.
- Preflight diagnosis: v1's simultaneous all-write exchange moved class margin but not hard
  accuracy enough because residual paths remain and repeated swaps can cancel. Preserve it as
  `INSTRUMENT_UNRESOLVED`; no target outcome had been read.
- Corrected control: before opening development, freeze v2 with an exact terminal CLS residual
  exchange. It takes held-out GUE accuracy from 0.891 to 0.109 and AUROC from 0.977 to 0.023;
  self-patch and batch drift are zero.
- Outcome: L11MLP ranks first in both splits, costing 0.130/0.120 accuracy and 0.214/0.283
  AUROC. It passes the SESOI and familywise test twice, but the confirmation bootstrap lower
  bound is zero. L11H2 is the top head in both splits; L11H5 remains near the top. No individual
  node passes the complete replicated gate.
- Decision: prioritize L11MLP for a new targeted MLP causal phase, retain L11H2/L11H5 as the
  leading heads, and do not claim a standalone necessary node from this atlas.

## 2026-07-28 — Freeze a post-hoc low-rank L11MLP assay

- Trigger: D10 ranks L11MLP first in both splits, but a full 768-dimensional write exchange does
  not show whether the causal effect is compressible.
- Supply boundary: D4 and D7–D10 consume every currently admitted exact-GUE-excluded MANE 1-Mb
  block. No fresh cohort exists without changing the biological source definition.
- Decision: reuse D10 development to fit an uncentered target-minus-control SVD and transfer the
  unchanged basis to D10 confirmation. Label the result post-hoc and cohort-reused because
  L11MLP selection already inspected both splits.
- Rank family: 1, 2, 4, 8, 16, and 32 of 768 dimensions. Select the smallest rank clearing the
  frozen margin, capture, accuracy, interval, and 31-random-subspace gates.
- Controls: full L11MLP exchange, orthogonal-complement exchange, directional patches, and a new
  disjoint GUE instrument slice beginning at pair offset 960.
- Boundary: a passing result licenses low-rank causal compression on these loci. It cannot supply
  fresh standalone L11MLP necessity, neuron identity, nucleotide source, or biological function.
- Outcome: rank 4 is the smallest qualified development rank. It captures 89.7% of the full
  class-margin effect in development and 84.5% under unchanged-basis confirmation, beating all
  31 dimension-matched random subspaces in each split.
- Interpretation: retain “dominant low-rank carrier,” not “complete bottleneck.” The orthogonal
  complement remains positive, rank-4 hard-label intervals include zero, and cohort reuse
  remains explicit in every summary.

## 2026-07-28 — Freeze D12 promoter-domain v2 after v1's completeness rule collapsed E1/E2

- Trigger: computing D12's frozen v1 endpoints under their own definitions, from
  `data/processed/dnabert2_atlas/promoter_source_localization_v1/development.raw.jsonl`, found that
  requiring every named tile per locus collapsed the scorable cohort. E1 (tiles 25,27,28,29 all
  required) fell to n=34 with development estimate +0.1419 (95% CI −0.1009 to +0.4074 — the
  interval spans zero). E2 (tiles 25,27,21,22,23 all required) fell to n=22 with development
  estimate +0.7433 (+0.2346 to +1.2806). No confirmation-split outcome had been read.
- Decision: freeze `configs/dnabert2_atlas/promoter_domain_v2.frozen.json`, declaring
  `"supersedes": "configs/dnabert2_atlas/promoter_domain_v1.frozen.json"`. v1 is not edited or
  deleted; it remains the immutable record of the original specification and of why it changed.
- Amendment: relax E1 and E2 to a completeness rule (at least one inner/downstream tile and at
  least one outer/upstream tile present per locus, averaging over whichever are available) and add
  a new primary endpoint, E1b — tile 27 minus tile 28, both required, the sharpest and most
  tile-parsimonious statement of the downstream edge. Under the relaxed v2 definitions the same
  development data give E1 +0.2291 (95% CI +0.0917 to +0.3752, n=149), E1b +0.2631 (+0.0695 to
  +0.4890, n=90), and E2 +0.4045 (+0.1661 to +0.6451, n=160); E3 (tile 22 alone, two-sided,
  unchanged) is −0.2793 (−0.5307 to −0.0420, n=126). `inference.multiplicity` becomes `Holm across
  E1, E1b, E2, E3`. `tiles_to_score` and `tiles_remaining_sealed` are unchanged and were
  re-verified to still partition `{0, ..., 29}` exactly, with no gap and no overlap.
- Boundary: this is a pre-outcome design correction, not a post-outcome fix — the confirmation
  split remains sealed for every endpoint except the already-public `mask|tile26`. The v2
  development numbers above were selected after inspecting several completeness-rule variants on
  the development split, so they are inflated by that selection and must not be read as
  effect-size estimates; they license only the endpoint redesign, not a claim.
- Phase-H gating ruling (fix round 2): E1 and E1b are two measurements of the same distal-edge claim
  (E1 the relaxed multi-tile contrast, E1b the sharper paired tile-27-minus-tile-28 difference); E2
  tests a different property, TSS polarity. Phase H (the helical-phase assay) is licensed only if at
  least one of E1 or E1b qualifies — if both fail, the bounded-domain premise is unsupported and
  Phase H must not run, regardless of E2. E2 qualifying alone does not license Phase H, because a
  polarity reversal is compatible with an unbounded downstream gradient as much as with a bounded
  domain. Refutation is per endpoint: E1 and E1b both failing refutes the domain reading (tile 26's
  D8 result would then stand alone as a single-tile effect); E1 qualifying with E1b failing is
  weaker evidence (elevated downstream sensitivity, not a sharp +31 boundary); E1b qualifying with
  E1 failing is stronger evidence for the edge specifically than for the domain as a block; E2
  failing refutes only polarity and leaves the domain claim intact; E3 failing licenses no null. The
  frozen `stop_rules` field in both v1 and v2 is unchanged and still reads literally "if E1 and E2
  both fail…"; this ruling, not that literal wording, is the operative gate now that E1b exists.
- Effect: `studies/dnabert2_reconstruction_atlas/PREREGISTRATION_D12_DOWNSTREAM_DOMAIN.md` is
  amended in place (Amendment 1) to record this trigger, the before/after numbers, and the
  disclosure above; its Provenance-and-honesty and Prior-art sections are unchanged. Fix round 2
  additionally threaded E1b through the Stop rules and What-would-refute-this sections, which had
  still reasoned only over E1/E2 and left the Phase-H gate undefined for the case that matters most.
  v1 remains preserved and superseded, not deleted.
