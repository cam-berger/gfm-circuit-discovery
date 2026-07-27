# DNABERT-2 Reconstruction Atlas claim ledger

> **Tier: T1 — canonical for this study's claim wording.**

## Live scientific claims

**D1 — development behavior boundary.** In the frozen bounded GENCODE v50 protein-coding
development cohort, none of splice acceptor, splice donor, start codon, or stop codon meets the
predeclared excess-reconstruction gate over every empirical control. All four beat n-gram
baselines, but empirical advantages range from −0.130 to +0.023 nats per covered nucleotide;
the only positive value (donor versus composition/token matched, +0.023) is not
Holm-significant and fails the other empirical families.

This licenses no `NULL` verdict and no claim about the sealed confirmation split. It stops
context and internal localization for these classes under census v1.

**D2 — multisource development behavior boundary.** In the frozen multisource-v4 development
cohort, none of the 13 tested classes meets the predeclared excess-reconstruction gate over every
empirical control. All classes beat order-0/1/2 nucleotide baselines, with advantages from +0.060
to +0.251 nats per covered nucleotide. Empirical advantages instead range from −0.150 to +0.020;
the largest, PPT versus a same-sequence non-site (+0.020), has add-one
`p=0.2445` and full-family Holm `p=1.0`.

Poly(A) cleavage regions, splice acceptors, branch points, splice donors, and stop codons meet the
frozen 250-cluster behavior floor and still do not qualify. Promoters, poly(A) signals, CTCF/REST/
CEBPB peak cores, distal enhancers, polypyrimidine tracts, and start codons are development
estimates only because control-complete supply is below that floor. This licenses no `NULL` verdict
for any class and no claim about the sealed confirmation split. It stops context and internal
localization under multisource v4.

## Established infrastructure facts

These are implementation properties and prior instrument constraints, not claims about biological
reconstruction.

| ID | Fact | Evidence | Boundary |
|---|---|---|---|
| IF1 | The adapter pins DNABERT-2-117M checkpoint, tokenizer, config, and vendored MosaicBERT code by revision and SHA-256; an incomplete MLM head is a hard failure. | `src/gfmcd/adapters/dnabert2/` and adapter tests | Does not show the model reconstructs any biological element. |
| IF2 | Atlas masking selects the minimal BPE token cover from nucleotide offsets and masks token IDs without retokenizing. | `test_atlas_tokenization.py` | An assay invariant, not model behavior. |
| IF3 | Alternative margins are rejected unless full offsets and all unmasked token IDs remain identical. | `test_atlas_tokenization.py` | Does not guarantee adequate alternative supply. |
| IF4 | Context exchanges are ineligible if they change target nucleotides, length, target token indices, offsets, or cover. | `test_atlas_context.py` | Does not establish any informative region. |
| IF5 | Causal specs reject null sets whose component count differs from the candidate intervention. | `test_atlas_mechanism.py` | Does not establish necessity or mediation. |
| IF6 | The bounded GENCODE v50 collection has zero target/baseline cluster overlap and exact declared matching balance; splice donor, splice acceptor, start codon, and stop codon meet outcome-blind control supply, while Kozak does not. | `results/dnabert2_reconstruction_atlas/collection_qualification_v1.json` | Supply qualification is not a reconstruction outcome; Kozak is untested, not null. |
| IF7 | Preflight calibration v1 passed geometry, split, balance, finiteness, deterministic replay, and batch-invariance checks, but its randomly located ≥32-nt-buffer copy plant failed class-wide positive-control monotonicity and sensitivity. | `results/dnabert2_reconstruction_atlas/preflight_calibration_v1.json` | `INSTRUMENT_UNRESOLVED`; no target outcome or biological null is licensed. |
| IF8 | Calibration v2 validates the nearest eligible copy-plant placement with four-copy means of +0.09 to +0.21 nats per covered nucleotide, while exposing an over-strict monotonicity implementation and a positive-effect rather than null-noise resolution statistic. | `results/dnabert2_reconstruction_atlas/preflight_calibration_v2.json` | Development-only instrument diagnosis; the lock remains draft and no target outcome is licensed. |
| IF9 | Corrected calibration v3 shows all four admitted classes exceed the positive-control floor with ordered doses; null-noise MDEs are ≤0.0139 and CI half-widths ≤0.0096, supporting the frozen v1 thresholds. | `results/dnabert2_reconstruction_atlas/preflight_calibration_v3.json` and `configs/dnabert2_atlas/preflight_v1.frozen.json` | Calibrates the instrument; target execution still requires a separate frozen `QUALIFIED` report. |
| IF10 | Frozen preflight v1 is `QUALIFIED`: every mandatory positive, negative, sensitivity, resolution, invariant, balance, geometry, and model-health gate passes against the exact bound input hashes. | `results/dnabert2_reconstruction_atlas/preflight_qualified_v1.json` | Authorizes development behavior scoring only; it is not evidence that any biological class reconstructs excessively. |
| IF11 | Multisource-v2 adapters and frozen definitions cover 13 classes from GENCODE, EPDnew, PolyASite, ENCODE TF clusters, Mercer lariat-seq, and ENCODE cCREs. A model-free full build rehearsal fills 384 locus blocks per class in each split after strict coordinate/source attrition. | `SOURCE_MANIFEST_V2.json`, `preflight_v2.frozen.json`, adapter tests | Outcome-blind source and collection fact only; v2 target scoring still requires a new qualified preflight. |
| IF12 | V2 outcome-blind control supply is insufficient for seven-nt promoter, enhancer, and TF peak-core anchors; those classes are supply-quarantined, not biologically negative. | `results/dnabert2_reconstruction_atlas/collection_qualification_v2.json` | No model score was computed. V3 changes only target width for broad interval anchors and outcome-blind local candidate prioritization. |
| IF13 | V3 five-nt anchors make all 13 classes preflight-eligible, but eight classes remain below the frozen 250-locus behavior floor under 128 local candidates. | `results/dnabert2_reconstruction_atlas/collection_qualification_v3.json` | Outcome-blind supply fact only; V4 expands candidate search without relaxing controls or the behavior minimum. |
| IF14 | V4’s 512-position search retains the same 6,448 complete matched targets as v3. All 13 classes meet the preflight floor; five meet the frozen behavior floor and eight are known underpowered before model scoring. | `results/dnabert2_reconstruction_atlas/collection_qualification_v4.json` | Final outcome-blind supply boundary; underpowered does not mean biological null. |
| IF15 | Frozen multisource-v4 preflight is `QUALIFIED` for all 13 classes against the exact bound data, controls, config, checkpoint, and model-health checks. | `results/dnabert2_reconstruction_atlas/preflight_qualified_multisource_v4.json` | Authorizes development scoring only; not a biological reconstruction result. |

## Open hypotheses

| ID | Hypothesis | Current status | First required bridge |
|---|---|---|---|
| H1 | At least one annotated site class has positive excess reconstruction over every required control. | not supported in multisource-v4 development; five adequately powered classes fail and eight classes remain underpowered | A newly preregistered behavior design or expanded control-complete supply; v4 confirmation stays sealed. |
| H2 | For at least one H1-qualified class, a biologically related context region outperforms matched irrelevant-region exchanges. | blocked by the v4 H1 gate | H1 plus token-geometry-invariant context interventions. |
| H3 | A bounded set of heads, MLPs, positions, or residual directions causally supports an H2-qualified behavior. | blocked by the v4 H1 gate | Frozen candidate screen, matched-size necessity, mediation, and path tests. |

Negative, retracted, and superseded rows will remain here permanently once evidence exists.
