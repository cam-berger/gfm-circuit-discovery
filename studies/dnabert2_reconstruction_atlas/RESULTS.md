# DNABERT-2 Reconstruction Atlas results index

> **Tier: T1 — canonical for result status.**

## Scientific results

| Run | Question | Status | Artifact | Manifest | Claim IDs |
|---|---|---|---|---|---|
| GENCODE behavior development v1 | Does any admitted class exceed all empirical and low-order controls? | development gate failed for every class; confirmation sealed | `../../results/dnabert2_reconstruction_atlas/behavior_development_v1.json` | `../../results/dnabert2_reconstruction_atlas/behavior_development_v1.manifest.json` | D1 |
| Multisource behavior development v4 | Do promoter, poly(A), TFBS, branch-point, PPT, enhancer, splice, or translation classes exceed every control? | development gate failed for every class; five powered, eight underpowered; confirmation sealed | `../../results/dnabert2_reconstruction_atlas/behavior_development_multisource_v4.json` | `../../results/dnabert2_reconstruction_atlas/behavior_development_multisource_v4.manifest.json` | D2 |
| Context-margin atlas v1 | Does a class-specific, target-preserving proximal-context corruption reduce DNABERT-2's geometry-qualified alternative margin more than an equal-dose distal corruption? | nine classes replicated; promoter did not replicate; stop failed development; acceptor/donor preflight-unresolved | `../../results/dnabert2_reconstruction_atlas/context_margin_v1_replication.json` | `../../results/dnabert2_reconstruction_atlas/context_margin_v1_replication.manifest.json` | D3 |
| Fine-tuned promoter classifier v2 | Does a promoter-trained DNABERT-2 margin distinguish exact-GUE-excluded GENCODE TSSs, and is its local context effect target-specific? | clean promoter margin replicated; context difference did not replicate; context-mechanism search stopped | `../../results/dnabert2_reconstruction_atlas/promoter_classifier_replication_v2.json` | `../../results/dnabert2_reconstruction_atlas/promoter_classifier_replication_v2.manifest.json` | D4 |
| Promoter clean-behavior head atlas v1 | Does a bounded set of CLS head outputs causally support the replicated clean promoter margin? | eight-head candidate set replicated in both patch directions against exact layer-profile nulls | `../../results/dnabert2_reconstruction_atlas/promoter_head_atlas_replication_v1.json` | `../../results/dnabert2_reconstruction_atlas/promoter_head_atlas_replication_v1.manifest.json` | D5 |
| Promoter circuit reverse engineering v1 | Is the candidate set faithful, pair-specific, query-gated, or arranged as an early-to-late path? | L11H2/L11H5 pair specificity replicated; about 61% faithfulness requires C8 plus all MLP writes; query and early-to-late models failed | `../../results/dnabert2_reconstruction_atlas/promoter_circuit_replication_v1.json` | `../../results/dnabert2_reconstruction_atlas/promoter_circuit_replication_v1.manifest.json` | D6 |
| Promoter address/payload decomposition v1 | Does the late pair depend on content-key addressing or correct value-payload association? | V-payload alignment dependence replicated against all 45 pair nulls; K-address effect stayed below the practical threshold | `../../results/dnabert2_reconstruction_atlas/promoter_address_payload_replication_v1.json` | `../../results/dnabert2_reconstruction_atlas/promoter_address_payload_replication_v1.manifest.json` | D7 |

### Promoter address/payload decomposition v1

The frozen assay uses a newly constructed set of 192 development and 192 confirmation locus
blocks, excluding every example and 1-Mb block in the original promoter cohort. For L11H2/L11H5
and all 45 candidate-excluding layer-11 pairs, it deterministically permutes K or V across the
same 40 genomic token positions. The score is target margin loss minus paired pseudo-TSS margin
loss.

| Primary operation | Development target-specific loss (95% CI) | Dev null max | Confirmation target-specific loss (95% CI) | Conf null max | Holm p, each split | Verdict |
|---|---:|---:|---:|---:|---:|---|
| K permutation | +0.0073 (+0.0003, +0.0144) | 0.0122 | +0.0084 (+0.0028, +0.0140) | 0.0098 | 0.1957 / 0.1087 | below 0.05 SESOI |
| V permutation | +0.1026 (+0.0517, +0.1552) | 0.0401 | +0.0615 (+0.0030, +0.1210) | 0.0301 | 0.04348 | replicated |

V permutation leaves Q, K, the QK logits, and post-softmax attention probabilities unchanged.
The replicated loss therefore establishes that the attention weights must remain associated with
the correct source value vectors for the promoter margin. It is a causal payload-association
result, not a claim that the values encode a known promoter motif.

The target-minus-control residual-write mean direction is highly stable across splits
(cosine 0.9908); confirmation projections average +3.111 (95% CI +2.287 to +3.904), with
75.0% positive loci. This direction is descriptive only. Joint K/V permutation is a secondary
ALiBi-relative positional diagnostic and does not replicate a separate effect: confirmation is
+0.0481 with a confidence interval crossing zero.

D7 narrows the late pair to a value/source-state payload-routing role. It does not identify the
causal source positions, represented sequence feature, residual semantics, a complete circuit,
or cellular promoter biology. Those questions require a new frozen source-localization phase.

### Promoter circuit reverse engineering v1

The frozen assay uses the 192 atlas-unused unique locus blocks in each split. Its strict
faithfulness universe contains all 144 attention-head context slices and all 12 MLP writes at
`[CLS]`. Every unretained node is patched from the paired pseudo-TSS; the score is the ratio of
mean retained target-control enrichment to mean native enrichment.

| Retained nodes | Development faithfulness (95% CI) | Confirmation faithfulness (95% CI) |
|---|---:|---:|
| L11H2/L11H5 | 15.18% (13.48%, 17.06%) | 13.62% (11.98%, 15.39%) |
| eight D5 heads | 15.30% (13.57%, 17.25%) | 13.73% (12.07%, 15.56%) |
| all heads, no MLP writes | 34.36% (30.98%, 37.96%) | 31.30% (28.05%, 35.00%) |
| all MLP writes, no head contexts | 0.00% | 0.00% |
| eight D5 heads plus all MLP writes | 61.04% (57.30%, 65.51%) | 61.27% (57.30%, 65.61%) |
| all 156 nodes | 100.00% | 100.00% |

The exact empty/all-node anchors are 0/1 on every example. The pair-specific test is stronger
than the original candidate-set localization:

| L11H2/L11H5 endpoint | Development effect (95% CI) | Null max | Confirmation effect (95% CI) | Null max | Holm p, each split |
|---|---:|---:|---:|---:|---:|
| necessity | +0.647 (+0.428, +0.877) | +0.329 | +0.700 (+0.483, +0.926) | +0.318 | 0.04348 |
| sufficiency | +1.068 (+0.825, +1.309) | +0.608 | +1.093 (+0.831, +1.363) | +0.640 | 0.04348 |

Both endpoints beat every one of the 45 candidate-excluding two-head pairs in layer 11. In
contrast, transferring only the late pair's target queries has essentially zero effect in both
splits. The proposed early-six corruption also has only +0.0585/+0.0563 logits of effect, below
the frozen +0.25 SESOI, so its small 30.2%/31.6% rescue fraction through the late pair is not a
qualified directed path.

D6 supports a late L11H2/L11H5 output-carrier core that depends on an MLP-maintained
representation. It rejects a target-dependent late-query gate and does not support the proposed
early-six-to-late chain. The eight heads are not a self-contained sparse circuit. K/V content,
source states, non-CLS computation, motifs, completeness, and cellular biology remain unresolved.

### Promoter clean-behavior head atlas v1

The frozen assay uses 192 unique 1-Mb locus blocks per split from D4's exact-GUE-excluded
GENCODE cohort. It scans all 144 heads by swapping one attention-context slice at the unpadded
`[CLS]` destination before the shared output projection. Development attribution plus exact
patching froze eight candidates: L2H2, L4H6, L4H7, L9H7, L10H1, L10H5, L11H2, and L11H5.
Confirmation reused that set unchanged.

| Joint endpoint | Development effect (95% CI) | Development null max | Confirmation effect (95% CI) | Confirmation null max | Holm p, each split | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Necessity: target gets paired pseudo-TSS head contexts | +1.061 (+0.818, +1.315) | +0.410 | +0.732 (+0.544, +0.929) | +0.331 | 0.03125 | replicated |
| Sufficiency: pseudo-TSS gets paired target head contexts | +1.417 (+1.149, +1.686) | +0.754 | +1.138 (+0.904, +1.375) | +0.600 | 0.03125 | replicated |

Effects are promoter-minus-nonpromoter logit-margin units. Each endpoint exceeds the frozen
+0.60 SESOI, has a positive locus-block-bootstrap lower bound, and beats all 63 distinct
candidate-excluding random head sets with the exact same per-layer counts. Attribution/exact
patch rankings remain associated in development (Spearman 0.747/0.876) and confirmation
(0.602/0.799).

| Candidate | Dev necessity / sufficiency | Confirmation necessity / sufficiency | Corrected positive signs in both directions and splits? |
|---|---:|---:|---|
| L11H2 | +0.506 / +0.765 | +0.315 / +0.627 | yes |
| L11H5 | +0.305 / +0.729 | +0.208 / +0.660 | yes |
| L9H7 | +0.0186 / +0.0545 | +0.0169 / +0.0538 | yes |
| L10H5 | +0.0101 / +0.0140 | +0.0080 / +0.0126 | yes |
| L4H6 | +0.0261 / +0.0088 | +0.0294 / +0.0148 | no |
| L10H1 | +0.00027 / +0.00221 | +0.00017 / +0.00197 | no |
| L2H2 | +0.00349 / +0.00038 | +0.00117 / −0.00116 | no |
| L4H7 | +0.00030 / +0.00167 | +0.00072 / +0.00057 | no |

The individual exact patches are priorities, not individual-specific claims: only the eight-head
group was compared with a matched head family. L11H2 and L11H5 are the leading decomposition
targets. L9H7 is the strongest source-localization lead because its confirmation target attention
is concentrated proximal-upstream (0.548) and shifts +0.0458 into that band and +0.0108 into the
TSS anchor relative to controls. Those attention values nominate a hypothesis only; they do not
show causal source positions or a motif.

D5 licenses candidate-set `[CLS]` head-output mediation of the clean supervised contrast. QK
selection, OV content, downstream paths, sparsity, individual-head specificity, the failed
target-preserving context endpoint, and cellular transcription remain unresolved.

### Fine-tuned promoter classifier v2

Checkpoint selection was independent of atlas outcomes. On the native GUE test split, the
MagicsLab 300-nt all-promoter head reached MCC 0.8705 and AUROC 0.9848, versus MCC 0.6804 and
AUROC 0.9179 for its 70-nt core-promoter sibling. Both original EPDnew cohorts were ineligible:
the best-aligned 70-nt and 300-nt windows each had 768/768 exact GUE memberships. The target run
therefore used GENCODE v50 protein-coding MANE Select TSSs after excluding any target or control
with exact forward or reverse-complement membership in any GUE split.

| Endpoint | Development, n=384 | Confirmation, n=384 | Confirmation Holm p | Verdict |
|---|---:|---:|---:|---|
| Clean promoter margin enrichment | +2.0956 (1.7813, 2.4080) | +1.8892 (1.6118, 2.1867) | 0.0002 | replicated |
| Target-specific context difference | +0.6655 (0.3250, 1.0080) | +0.2750 (−0.0684, 0.6151) | 0.0569 | not replicated |

Effects are promoter-minus-nonpromoter classifier logit margins. The first row compares true TSS
windows with composition-matched, TSS-excluded pseudo-TSS windows from the same locus. The second
is the true-TSS clean-minus-shuffle loss minus the identical, equal-dose pseudo-TSS loss. Its
confirmation mean clears the +0.25 SESOI, but its interval crosses zero and its corrected p-value
misses the gate. The licensed result is promoter classification behavior only; no local-context
mechanism or internal component search is authorized.

The first frozen preflight remains preserved as `INSTRUMENT_UNRESOLVED`: a single random-label
null was too volatile and a \(3.10\times10^{-6}\) CPU batch-shape difference exceeded a
\(10^{-6}\) tolerance. V2 changed no biological threshold or target outcome. It uses the mean of
256 label permutations and a \(10^{-5}\) numerical tolerance; all eight gates pass.

### Replicated context-margin atlas v1

The frozen assay scored 2,059 development targets across 11 preflight-admitted classes, then
opened only ten development-qualified classes on 1,781 untouched confirmation targets.

| Site class | Dev n | Dev effect | Confirmation n | Confirmation effect (95% CI) | Holm p | Verdict |
|---|---:|---:|---:|---:|---:|---|
| start codon | 219 | +0.0468 | 222 | +0.0479 (+0.0306, +0.0673) | 0.0010 | replicated |
| distal enhancer anchor | 85 | +0.0395 | 98 | +0.0384 (+0.0101, +0.0730) | 0.0144 | replicated |
| poly(A) cleavage | 112 | +0.0298 | 108 | +0.0348 (+0.0125, +0.0601) | 0.0068 | replicated |
| branch point | 145 | +0.0412 | 138 | +0.0304 (+0.00443, +0.0585) | 0.0270 | replicated |
| REST peak anchor | 208 | +0.0136 | 230 | +0.0213 (+0.0126, +0.0302) | 0.0010 | replicated |
| CEBPB peak anchor | 216 | +0.0215 | 213 | +0.0183 (+0.00795, +0.0295) | 0.0035 | replicated |
| CTCF peak anchor | 217 | +0.0141 | 198 | +0.0159 (+0.00643, +0.0255) | 0.0036 | replicated |
| poly(A) signal | 241 | +0.0127 | 245 | +0.0127 (+0.00555, +0.0208) | 0.0036 | replicated |
| polypyrimidine tract | 250 | +0.0109 | 228 | +0.0125 (+0.00830, +0.0172) | 0.0010 | replicated |
| promoter TSS anchor | 85 | +0.0247 | 101 | +0.00066 (−0.0117, +0.0126) | 0.4548 | not replicated |
| stop codon | 281 | +0.00330 | — | — | — | development failed |

Effects are nats per complete BPE-covered nucleotide:
`margin(distal equal-dose corruption) − margin(class-specific proximal corruption)`.
Splice acceptor and donor were not target-scored because their margin positive controls failed
before outcomes. The nine replicated rows license a DNABERT-2 proximal-context behavior, not a
cellular functional claim or an internal mechanism. For TF/enhancer rows the five-nucleotide
target remains a derived peak/cCRE anchor, not a resolved motif. The distal control is exactly
matched for width, composition, and changed-base count but not target distance; this bounds the
claim to proximal-context dependence.

### Development behavior census v1

The frozen, qualified development screen scored 1,725 matched targets across 24
class × contrast tests with add-one permutation values and full-family Holm correction.

| Site class | Clusters | Same motif | Composition/token | Same sequence | N-gram range | Development verdict |
|---|---:|---:|---:|---:|---:|---|
| splice acceptor | 501 | −0.110 | −0.106 | −0.099 | +0.089 to +0.156 | not qualified |
| splice donor | 451 | −0.032 | +0.023 | −0.023 | +0.186 to +0.225 | not qualified |
| start codon | 354 | −0.068 | −0.080 | −0.042 | +0.106 to +0.126 | not qualified |
| stop codon | 419 | −0.130 | −0.117 | −0.123 | +0.079 to +0.110 | not qualified |

Effects are nats per complete BPE-covered nucleotide. Every n-gram contrast passes the
0.02 SESOI and full-family correction; no class passes its empirical families. D1 is therefore
a bounded development result. Confirmation remains sealed, no `NULL` verdict is issued, and
P4–P6 are stopped.

### Multisource development behavior census v4

The frozen, qualified development screen scored 3,189 complete matched targets across 78
class × contrast tests with 9,999 add-one permutations and Holm correction across the full
family.

| Site class | Clusters | Power | Same motif | Composition/token | Same sequence | N-gram range | Verdict |
|---|---:|---|---:|---:|---:|---:|---|
| distal enhancer core | 210 | underpowered | −0.012 | −0.070 | −0.020 | +0.191 to +0.241 | not qualified |
| poly(A) cleavage region | 383 | powered | −0.068 | −0.080 | −0.069 | +0.144 to +0.174 | not qualified |
| poly(A) signal | 198 | underpowered | −0.044 | −0.037 | −0.015 | +0.138 to +0.197 | not qualified |
| EPD promoter TSS core | 222 | underpowered | −0.021 | −0.023 | −0.043 | +0.133 to +0.170 | not qualified |
| splice acceptor | 373 | powered | −0.048 | −0.078 | −0.066 | +0.118 to +0.177 | not qualified |
| branch point | 384 | powered | −0.092 | −0.123 | −0.103 | +0.131 to +0.156 | not qualified |
| splice donor | 316 | powered | −0.073 | −0.080 | −0.057 | +0.156 to +0.176 | not qualified |
| polypyrimidine tract | 87 | underpowered | −0.004 | +0.004 | +0.020 | +0.124 to +0.246 | not qualified |
| CEBPB peak core | 196 | underpowered | −0.012 | −0.087 | −0.087 | +0.231 to +0.251 | not qualified |
| CTCF peak core | 134 | underpowered | +0.001 | −0.066 | −0.037 | +0.178 to +0.212 | not qualified |
| REST peak core | 146 | underpowered | −0.082 | −0.014 | −0.066 | +0.175 to +0.223 | not qualified |
| start codon | 244 | underpowered | −0.099 | −0.087 | −0.098 | +0.060 to +0.089 | not qualified |
| stop codon | 296 | powered | −0.150 | −0.127 | −0.117 | +0.077 to +0.100 | not qualified |

Effects are nats per complete BPE-covered nucleotide. Every n-gram contrast passes the +0.02
SESOI and full-family correction. No empirical contrast passes both gates. The largest positive
empirical estimate, PPT versus same-sequence non-site (+0.020), has add-one `p=0.2445` and
Holm `p=1.0`. D2 is therefore a bounded development result, not a confirmatory null. The
confirmation runner refuses to open because no class development-qualified; P4–P6 are stopped.

## Infrastructure verification

The test suite verifies coordinate schemas, BPE cover selection, masking, scoring normalization,
alternative eligibility, empirical-family separation, context invariants, gate failure behavior,
add-one inference, matched-size causal nulls, adapter unpadding maps, and immutable writes.
Infrastructure tests license only the IF rows in `CLAIMS.md`.

### Phase-A bounded collection qualification

`results/dnabert2_reconstruction_atlas/collection_qualification_v1.json` records 5,120 target
examples and 2,560 baseline-only examples with zero cluster overlap. Under the frozen
one-control-per-family supply design, splice acceptor (501/503 development/confirmation
clusters), splice donor (451/445), start codon (354/350), and stop codon (419/393) enter
preflight calibration. Kozak (5/6) is quarantined before model outcomes. The rejected
three-control supply audit is preserved as a bound comparison in the same artifact.

This is a collection/instrument result only. It licenses IF6 and no biological behavior claim.

### Preflight calibration v1 — instrument unresolved

`results/dnabert2_reconstruction_atlas/preflight_calibration_v1.json` used 64 development
clusters per supply-qualified class and no empirical target-control contrast. Geometry, matching
balance, split invariants, finite scores, deterministic replay, and batch invariance passed.

The positive control did not qualify. Four-copy effects were +0.0051 acceptor, −0.0043 donor,
+0.0089 start codon, and +0.0088 stop codon nats per covered nucleotide; only stop codon was
strictly dose-monotone. The v1 plant was randomly located anywhere downstream after a ≥32-nt
buffer, so it mixed sensitivity with arbitrary-distance context use. Status is
`INSTRUMENT_UNRESOLVED`; interpretation is frozen while placement is diagnosed. This licenses
IF7 and no behavior claim.

### Preflight calibration v2 — placement validated, gate implementation amended

`results/dnabert2_reconstruction_atlas/preflight_calibration_v2.json` used the preregistered
nearest eligible placement. Four-copy effects were +0.090 acceptor, +0.161 donor, +0.113 start
codon, and +0.206 stop codon nats per covered nucleotide. Every dose curve increased from one
to two to four copies; donor was labeled non-monotone only because the implementation also
required the one-copy mean to be positive, a requirement not present in the draft.

The same review found that sensitivity/resolution were computed from heterogeneous positive-effect
variance. Before freezing, calibration v3 uses the irrelevant-exchange null-noise MDE and CI
half-width for those gates, while retaining the positive effect and dose ordering for the
sensitivity positive control. V2 remains development-only diagnosis and licenses IF8, not behavior.

### Preflight calibration v3 — lock calibration complete

`results/dnabert2_reconstruction_atlas/preflight_calibration_v3.json` corrects the two v2 gate
definitions without changing any sequence or score. All four classes have ordered copy doses and
four-copy effects above +0.09 nats per covered nucleotide. Irrelevant-exchange null-noise MDEs
range 0.0077–0.0139 and CI half-widths 0.0051–0.0096.

These controls freeze `preflight_v1.frozen.json`: positive floor 0.05, negative equivalence
0.02, MDE ceiling 0.015, CI-half-width ceiling 0.01, behavior SESOI 0.02, and 350 clusters
per split. This licenses IF9. It still does not authorize target execution until the same
instrument reruns against the frozen config and emits `QUALIFIED`.

### Frozen preflight v1 — qualified

`results/dnabert2_reconstruction_atlas/preflight_qualified_v1.json` reruns calibration v3
against the frozen config hash and exact collection, match, matching-audit, and baseline hashes.
All eight mandatory gate families pass and the report decision is `QUALIFIED`.

This licenses IF10 and authorizes the development behavior screen. It is still an instrument
result, not evidence for H1.

### Multisource collection qualification v2–v4

V2 adds 13 frozen classes from GENCODE v50, EPDnew human v006, PolyASite 2.0 GRCh38.96,
UCSC’s multi-experiment ENCODE TF ChIP clusters, Mercer high-confidence lariat-seq branch points,
and ENCODE Registry v4 GRCh38 cCREs. Its seven-nucleotide promoter, enhancer, and TF centers fail
outcome-blind exact-control supply and remain preserved in
`collection_qualification_v2.json`.

V3 changes only those broad-interval assay anchors to symmetric five-nucleotide centers and
prioritizes local control candidates by composition, making all 13 classes preflight-eligible.
V4 expands that same candidate search from 128 to 512 positions without changing targets,
controls, calipers, or statistical gates. It retains the same 6,448 complete matched targets,
showing that exact BPE geometry—not candidate depth—limits the remaining supply. Five classes meet
the 250-locus behavior floor on both splits; eight are frozen as underpowered. These are
collection facts (IF12–IF14), not biological outcomes.

### Frozen multisource preflight v4 — qualified

`preflight_qualified_multisource_v4.json` binds the exact v4 collection, matches, matching audit,
baseline corpus, frozen config, and pinned checkpoint. Every class has 64 preflight-eligible
examples. All graded-copy curves are monotone and their four-copy means range from +0.082 to
+0.226 nats per covered nucleotide. All negative-equivalence, sensitivity, resolution, balance,
split, coordinate, geometry, replay, batch, finiteness, and model-health gates pass.

This licenses IF15 and authorized the multisource development screen. It is not biological
evidence by itself.

Bulk runtime collections and per-site shards will remain ignored. Every settled compact result must
be immutable, indexed here, and accompanied by a manifest containing checkpoint, data inputs,
parameters, seeds, code revision, and command.
