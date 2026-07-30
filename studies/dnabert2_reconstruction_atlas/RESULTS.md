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
| Promoter source localization v1 | Where is classifier-relevant promoter information, and does the replicated late-pair V payload localize to the same region? | nominal +11 to +21 mask effect replicated; regional V, independent biological annotations, and predefined motifs did not replicate/qualify | `../../results/dnabert2_reconstruction_atlas/promoter_source_localization_replication_v1.json` | `../../results/dnabert2_reconstruction_atlas/promoter_source_localization_replication_v1.manifest.json` | D8 |
| Promoter source classification performance v1 | Does the nominated mask alter hard-label performance and ranking? | post-hoc AUROC loss repeats in both splits; 7/231 target promoter→non-promoter flips; balanced accuracy unchanged by offsetting control corrections | `../../results/dnabert2_reconstruction_atlas/promoter_source_classification_performance_v1.json` | `../../results/dnabert2_reconstruction_atlas/promoter_source_classification_performance_v1.manifest.json` | D8 |
| Promoter native-source development v10 | Is the +11:+21 mask nomination reproduced by a native exact-composition sequence reorder? | development positive gate failed; confirmation, residual tracing, and mediation sealed | `../../results/dnabert2_reconstruction_atlas/promoter_native_source_development_v10.json` | `../../results/dnabert2_reconstruction_atlas/promoter_native_source_development_v10.manifest.json` | D9 |
| Promoter all-node accuracy atlas v1 | Which of all 144 CLS head contexts and 12 MLP writes matter most for correct labels? | L11MLP ranks first with 13/12-point accuracy loss and 0.214/0.283 AUROC loss; confirmation CI touches zero, so no node passes the complete replicated gate | `../../results/dnabert2_reconstruction_atlas/promoter_node_accuracy_replication_v1.json` | `../../results/dnabert2_reconstruction_atlas/promoter_node_accuracy_replication_v1.manifest.json` | D10 |
| Promoter L11MLP low-rank assay v1 | Can a small causal subspace capture the L11MLP exchange effect? | a fixed rank-4 basis captures 89.7%/84.5% of the class-margin effect and beats 31 matched random subspaces in both reused D10 splits | `../../results/dnabert2_reconstruction_atlas/promoter_low_rank_replication_v1.json` | `../../results/dnabert2_reconstruction_atlas/promoter_low_rank_replication_v1.manifest.json` | D11 |
| Promoter downstream-domain confirmation v1 | Is the D8 mask profile a bounded downstream domain with a distal edge and a TSS polarity reversal, rather than a single 10-nt tile? | TSS polarity (E2) replicated; bounded domain and distal edge (E1/E1b) refuted, tile 26 stands alone; upstream reversal (E3) is a near-miss that licenses no null | `../../results/dnabert2_reconstruction_atlas/promoter_domain_confirmation_v1.json` | `../../results/dnabert2_reconstruction_atlas/promoter_domain_confirmation_v1.manifest.json` | D12 |

### Promoter source localization v1

The outcome-blind supply binds the 192 remaining unused locus blocks per split after excluding
both the original promoter cohort and D7. Thirty 10-nt nominal tiles receive the same two-token
dose on each target and pseudo-TSS branch. Development scans all tiles; confirmation opens only
the development-qualified operation/tile.

| Endpoint | Development | Confirmation | Verdict |
|---|---:|---:|---|
| Tile 26 mask target-specific loss | +0.245 (+0.076, +0.428), n=107 | +0.206 (+0.045, +0.392), n=124 | replicated |
| Tile 26 localization contrast | +0.266 (+0.095, +0.444), max-family p=0.026 | frozen from development; raw-effect p=0.0082 | qualified |
| Best regional L11H2/L11H5 V raw effect | +0.00124 | not opened | below +0.005 SESOI |
| Best regional L11H2/L11H5 V localization contrast | +0.00119 | not opened | below +0.002 SESOI |

Tile 26 is nominally +11 to +21 nt relative to the TSS. The selected BPE covers have median length
10 nt, range 7–14 in development and 6–15 in confirmation. The result therefore does not license
single-base or exact ten-nucleotide resolution.

Independent qualification does not replicate. The ENCODE PLS-overlapping subset is positive in
development (+0.261, 95% CI +0.063 to +0.484; corrected `p=0.0412`) but misses confirmation
(+0.170, −0.025 to +0.383; `p=0.0589`). CAGE overlap is below the frozen sample floor at tile 26,
and the CpG subset does not qualify. No geometry-qualified predefined TATA, Inr, CCAAT, or GC-box
instance intersects the nominated tile, so no motif was relabeled or disrupted after seeing the
outcome.

D8 licenses a replicated **behavioral source-region candidate** for this fine-tuned classifier.
It does not identify a biological promoter element, localize D7's V payload, or establish
mask/motif-to-head mediation.

### Promoter source classification performance v1

This user-requested secondary analysis was specified after D8 and is therefore explicitly
post-hoc. It reruns the unchanged two-token tile-26 mask and retains clean/masked margins for the
same eligible examples.

| Endpoint | Development (107 pairs) | Confirmation (124 pairs) |
|---|---:|---:|
| Accuracy, clean → masked | 0.589 → 0.579 (Δ −0.009; 95% CI −0.033 to +0.014) | 0.581 → 0.589 (Δ +0.008; −0.020 to +0.036) |
| AUROC, clean → masked | 0.678 → 0.644 (Δ −0.033; −0.060 to −0.007) | 0.679 → 0.658 (Δ −0.021; −0.043 to −0.002) |
| Real promoter→non-promoter flips | 3/107 | 4/124 |
| Reverse target flips | 1/107 | 2/124 |
| Pseudo-TSS false-positive→correct-negative flips | 1/107 | 5/124 |

Pooled accuracy is exactly unchanged at 0.584 because the seven target errors are balanced by
control-side corrections and three reverse target flips/two reverse control flips. Pooled AUROC
falls by 0.026 (bootstrap 95% CI −0.043 to −0.011). Thus the mask reliably weakens ranking and
changes a small subset of hard labels, but it does not reduce net balanced accuracy on this
challenge cohort. These values do not estimate performance on GUE.

### Promoter native-source qualification v10

The native intervention reorders +11:+21 while preserving exact mononucleotide and linear
dinucleotide counts. It changes no nucleotide outside that interval and uses equal target/control
Hamming dose with locally matched BPE retokenization geometry. The final outcome-blind supply
contains 60 fresh unique 1-Mb locus blocks per split, with 37/36 CAGE-supported TSSs, 46/47
promoter-like cCREs, and 27/25 CpG-island overlaps in development/confirmation.

The two-operation v7 preflight rejected GC depletion for excess variance. A fresh 40-locus
dinucleotide-only v8 preflight also missed the unchanged precision ceiling. After increasing the
outcome-blind cohort to 60 loci, the third disjoint preflight passed all checks with projected MDE
0.156 logits and zero identity/batch drift.

| Development endpoint | Estimate | Frozen verdict |
|---|---:|---|
| Native-reorder target-specific margin loss | −0.016 (95% CI −0.176 to +0.138; `p=0.5834`) | fails +0.10 SESOI, interval, and sign gate |
| Clean target−control promoter margin | +2.321 (+1.392 to +3.266; `p=0.0001`) | classifier behavior healthy |
| Accuracy, clean → edited | 0.650 → 0.650 | unchanged |
| AUROC, clean → edited | 0.7075 → 0.7133 | descriptive change +0.0058 |
| Real promoter→non-promoter flips | 1/60 | descriptive |

CAGE-, PLS-, and CpG-overlap subgroup means are all nonpositive and descriptive; the one
DCE-II-like instance lies outside development. D9 does not support native sequence-order
necessity for this edit and cohort. Because the interval includes the +0.10 SESOI, it is not an
equivalence null. It does not test every possible substitution or establish that the region is
biologically inert, or qualify GC-content necessity. Confirmation and all internal tracing remain
sealed.

### Promoter all-node accuracy atlas v1

The accuracy-first atlas uses the last 100 outcome-unseen, unique 1-Mb locus blocks after
excluding all examples and blocks used by D4 and the D7–D9 phases. Fifty pairs are assigned to
development and fifty to confirmation by a stable outcome-blind hash. Every one of the 144
attention-head contexts and 12 MLP writes is exchanged individually between the real TSS and its
paired pseudo-TSS at `[CLS]`. The primary endpoint is the loss in balanced paired accuracy;
AUROC, class-margin loss, and directional label flips are secondary.

| Node | Dev rank; accuracy loss (95% CI) | Dev AUROC loss; maxT p | Confirmation rank; accuracy loss (95% CI) | Confirmation AUROC loss; maxT p | Verdict |
|---|---:|---:|---:|---:|---|
| L11MLP | 1; +0.130 (+0.030, +0.230) | +0.2136; 0.0100 | 1; +0.120 (+0.000, +0.240) | +0.2832; 0.0401 | strongest replicated candidate; confirmation CI gate fails |
| L11H2 | 2; +0.040 (−0.020, +0.090) | +0.0416; 0.2758 | 2; +0.040 (−0.010, +0.100) | +0.0832; 0.3819 | top head; family gate fails |
| L11H5 | 3; +0.030 (−0.020, +0.080) | +0.0600; 0.2758 | 4; +0.020 (−0.030, +0.070) | +0.0488; 0.3819 | stable late-head candidate; family gate fails |

Clean balanced accuracy is 0.600 in each split; clean AUROC is 0.6612/0.7104. L11MLP exchange
causes 6/5 target promoter→non-promoter flips and 10/13 control non-promoter→promoter flips in
development/confirmation, partly offset by reverse transitions. It alone clears the 0.02
accuracy SESOI and full-family maximum-sign test in both splits, but its confirmation bootstrap
lower bound is exactly zero. No individual component therefore passes all three frozen gates on
both splits.

The first preflight is preserved as `INSTRUMENT_UNRESOLVED`: simultaneous exchange of all
layerwise writes was not an accuracy positive control because residual paths remain and repeated
exchanges can cancel. Before target outcomes, v2 replaced only that control with an exact
terminal `[CLS]` residual exchange. It reduced GUE accuracy from 0.891 to 0.109 and AUROC from
0.977 to 0.023; self-patch and batch drift were zero.

D10 identifies L11MLP as the highest-priority missing component and independently returns
L11H2/L11H5 as the leading heads. These are reciprocal paired-exchange results, not zero
ablations, source localization, a complete circuit, or biological promoter mechanisms.

### Promoter L11MLP low-rank assay v1

All currently admitted exact-GUE-excluded MANE locus blocks were already used by D4 and D7–D10.
The low-rank assay therefore reuses the D10 development and confirmation splits. L11MLP had
already been selected after both splits were read, so this is a post-hoc structural-transfer
test rather than fresh independent replication.

An uncentered SVD of development target-minus-control L11MLP `[CLS]` writes defines a maximum
32-dimensional basis. The rank scan tests 1, 2, 4, 8, 16, and 32 dimensions. Each rank is
evaluated by projected reciprocal exchange; the selected rank is compared with full exchange,
its orthogonal complement, both directional transfers, and 31 Gaussian-QR random subspaces of
the same dimension.

| Endpoint | Development | Unchanged-basis confirmation |
|---|---:|---:|
| Selected rank | 4/768 (0.52%) | 4/768 (0.52%) |
| Rank-4 class-margin loss | +0.828 (+0.270, +1.401) | +0.923 (+0.300, +1.538) |
| Full L11MLP class-margin loss | +0.922 (+0.360, +1.530) | +1.092 (+0.433, +1.721) |
| Full margin effect captured | 89.7% | 84.5% |
| Rank-4 accuracy loss | +0.110 (0.000, +0.220) | +0.070 (−0.030, +0.170) |
| Rank-4 / full AUROC loss | 0.184 / 0.214 | 0.236 / 0.283 |
| Random-subspace add-one p | 0.03125 | 0.03125 |
| Complement class-margin loss | +0.079 (+0.002, +0.155) | +0.156 (+0.070, +0.246) |

The rank-4 subspace is a dominant causal carrier of the reciprocal L11MLP class-margin effect,
but it is not exhaustive: the orthogonal complement remains positive. The hard-label accuracy
interval includes zero in both splits. SVD variance is descriptive; causal evidence comes from
projected activation exchange and the matched random-subspace comparison. The result does not
identify neurons, late-head mediation, sequence content, or promoter biology.

### Promoter downstream-domain confirmation v1

D12 reuses D8's own sealed 192-pair confirmation split under a protocol frozen before any of its
five endpoints were read beyond the already-public `mask|tile26` value. Development predictions are
computed from D8's own development records and require no model run
(`promoter_domain_development_predictions_v1.json`, `FROZEN`). Amendment 1 relaxed the original
all-tile-required completeness rule for E1/E2 — which had collapsed E1 to n=34 with a
zero-spanning development interval — to a looser per-locus completeness rule, and added a second,
sharper primary endpoint, E1b (tile 27 minus tile 28, both required).

| Endpoint | Development (point estimate) | Confirmation (95% CI, n) | Holm p | Verdict |
|---|---:|---:|---:|---|
| E1 downstream edge | +0.2291 | +0.0375 (−0.0964, +0.1666), n=138 | 0.47 | failed |
| E1b sharp edge | +0.2631 | +0.0496 (−0.0746, +0.1815), n=77 | 0.47 | failed |
| E2 TSS polarity | +0.4045 | +0.3262 (+0.1110, +0.5564), n=149 | 0.0096 | qualified |
| E3 upstream anti-evidence | −0.2793 | −0.2164 (−0.4189, +0.0005), n=128 | 0.1476 | failed; licenses no null |
| E4 full domain (descriptive only) | +0.2524 | +0.0846 (−0.0442, +0.2295), n=162 | — | never eligible |

Only E2 qualifies: masking downstream tiles 25/27 costs more target-specific promoter margin than
masking upstream tiles 21/22/23, exceeding the +0.15 SESOI with a positive lower bound and passing
Holm correction. E1 and E1b both keep the development-predicted sign but fail their SESOI, interval,
and Holm gates; per the preregistered refutation rule this refutes the bounded-domain and
distal-edge reading, so D8's tile 26 result stands alone as a single-tile effect. E4 was never
eligible to qualify because it includes the already-opened tile 26.

Shrinkage from development to confirmation tracks exactly which endpoints touch the +31:+51
downstream flank. E1, E1b, and E4 — all defined against that flank — shrink about 6.1×, 5.3×, and
3.0× respectively and fail; E2 and E3, which never reference it, shrink only about 1.24× and 1.29×
and hold sign and direction. Amendment 1 discloses that the v2 endpoint definitions were chosen
after inspecting several completeness-rule variants on the development split, so this shrinkage
pattern is consistent with selection-inflated development estimates for the flank-referencing
endpoints specifically, not with sampling noise alone.

E3 clears its 0.10 absolute SESOI and keeps the predicted sign, but its 95% interval
(−0.4189, +0.0005) misses excluding zero by 4.5e-04, and Holm `p=0.1476` fails. E3 is preregistered
as powered only for effects at or above 0.10 absolute, so per D12's own protocol this failure
licenses no null about the upstream reversal — it is a near-miss, not evidence of absence.

The frozen preflight (`promoter_domain_preflight_v1.json`) is `QUALIFIED`: M1 (identity), M2 (record
anchor), M3 (cover balance), M4 (equal dose), and a separate batch-invariance check all pass. M2
re-derives all 124 recorded `mask|tile26` confirmation values through an independent code path
(worst absolute difference 1.133e-05 against a 1e-4 tolerance) and is the historical-fidelity check.
M1 is a Q/K/V-style no-op patch (worst absolute difference 0.0, worst absolute value 2.38e-06) that
is largely architecture-forced — correctly wired, it returns near-zero almost regardless of
checkpoint or tokens — so it demonstrates patch-machinery consistency, not historical fidelity, and
should not be counted as evidence independent of M2.

Amendment 2, authorized before any confirmation outcome was read, permits the planned downstream
Phase H (helical-phase) assay to proceed regardless of the E1/E1b verdict. Because E1 and E1b both
failed here, that authorization is now live: Phase H may run, but per Amendment 2's own wording
constraint, no Phase H result may be described as a bounded domain's phase dependence or used to
argue backwards that the domain exists.

D12 licenses a replicated classifier-level downstream-vs-upstream mask-sensitivity polarity. It does
not license a bounded domain, a distal edge, a biological promoter element, a nucleotide motif,
native-sequence necessity, mediation by any internal component, or transcriptional function in
cells. Tiles 0–20 and 24 (22 of the 30 scanned tiles) remain sealed. Whether the polarity reflects
GC/CpG composition rather than position is tested by D13.

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

### D14 — arrangement vs composition behind the D12 TSS polarity

`results/dnabert2_reconstruction_atlas/promoter_arrangement_composition_confirmation_v4.json`
(sealed confirmation split, three-arm intersection n=83; per-locus values embedded inline).
Protocol `PREREGISTRATION_D14_ARRANGEMENT_VS_COMPOSITION.md`. Smoke gate
`promoter_arrangement_composition_smoke_v1.json`; exploratory development-split Phase 0
`promoter_arrangement_composition_phase0_v1.json`.

Three interventions at D12's own E2 tiles, applied per 10-nt tile:

| arm | E2 | 95% CI | mean Hamming |
|---|---:|---|---:|
| `mask` (replayed from D12, not re-estimated) | +0.4791 | [+0.2288, +0.7358] | token cover |
| `gc_depletion` (dose 3, real bases) | +0.2625 | [−0.0622, +0.5934] | 3.000 nt |
| `dinucleotide_reorder` (composition exactly fixed) | +0.0619 | [−0.0719, +0.1986] | 4.917 nt |

| endpoint | mean | 95% CI | Holm p |
|---|---:|---|---:|
| **P1 `mask − reorder`** | **+0.4172** | **[+0.1705, +0.6608]** | **0.0066** |
| P3 arrangement contribution | +0.0619 | [−0.0719, +0.1986] | 0.4506 |
| P2 `gc_depletion − reorder` | +0.2006 | [−0.1191, +0.5238] | 0.4506 |
| S1 `mask − gc_depletion` | +0.2166 | [−0.1307, +0.5561] | — |

Gates: exact replay of D12's stored per-locus mask values, worst absolute difference **0.0** across
93 rows; batch invariance 0.0; clean-behaviour anchor +1.9568; inert-tile (tile 5) specificity
−0.0623, inside the 0.15 SESOI; M1/M2/M3/M4 pass, each reported only on the arm where it can fail.

**Reading.** The polarity survives exact composition-preserving scrambling — arrangement carries
12.9% of the mask effect, median −0.0070, 41/83 loci positive. It is not a dose artifact (3 nt
yields +0.2625 while 4.917 nt yields +0.0619) and not a `[MASK]` artifact (S1 spans zero,
`gc_depletion` uses only real bases). **Not licensed:** "arrangement contributes nothing" (bound
0.1986 > SESOI 0.15) or "composition is the carrier" (P2 declared underpowered pre-run). P1's
magnitude is skew-inflated — robust estimates ≈0.25–0.32, sign test 51/83, exact binomial
`p=0.0475`; direction robust across five bootstrap seeds. Partial seal: loci opened by D12 for
`mask`, never scored under either native arm.

### D15 — regional information density vs the ALiBi-forced attention prior

Track B. Protocol `PREREGISTRATION_D15_ATTENTION_CAUSALITY_DISSOCIATION.md` (frozen 2026-07-29,
Amendment 1 pre-outcome). Instrument: equal-dose contiguous span masking — 7 BPE tokens in target and
control with matched cover, so any artifact that depends only on "seven tokens were masked" cancels
exactly in a between-arm difference.

**Architecture (zero forward passes, not evidence).** `far_upstream` is `(−249,−50)`: 200 of 300 nt and
the band nearest `[CLS]` at index 0. The ALiBi-only prior therefore places 49%–89% of *every* head's
`[CLS]` mass there, at all twelve slopes. Far-upstream attention dominance is forced; only
measured-minus-prior is interpretable.

| head | slope | excess `far_up` | excess `prox_up` | excess `tss` | excess `prox_dn` | necessity |
|---|---:|---:|---:|---:|---:|---:|
| L11H2 | 0.1250 | +0.046 | +0.000 | +0.000 | +0.000 | 0.3154 |
| L11H5 | 0.0156 | +0.032 | +0.002 | +0.000 | +0.001 | 0.2084 |
| L9H7 | 0.0039 | −0.525 | +0.401 | +0.051 | +0.082 | 0.0169 |

Head reach does **not** predict causal effect within layer (mean Spearman 0.062–0.284, sign-test
p=0.146–1.000, four tests, nothing qualifies) — a **failed prediction** of the D14 prereg §2.

**Confirmation split, n=186.**

| arm | TSS-relative | dose | mean target-specific loss |
|---|---|---:|---:|
| `S` | −9:+31 | 7 | +0.4675 |
| `F1`…`F5` | −249:−49, five 40-nt windows | 7 | +0.146, +0.089, +0.066, +0.038, +0.226 |
| `U` | −49:−9 | 7 | −0.018 |
| `B` | −249:−49 whole band | 30 | +0.8168 |

| endpoint | n | mean | 95% CI | Holm p | verdict |
|---|---:|---:|---|---:|---|
| **E_reg** | 186 | **+0.3490** | **[+0.1153, +0.5892]** | **0.0058** | **QUALIFIES** |
| E_band | 114 | −0.3319 | [−0.9109, +0.2488] | 0.265 | fails; underpowered by pre-run declaration |
| N1 null control | 178 | +0.0806 | [−0.1153, +0.2910] | — | passes |
| spec `S − U` | 175 | +0.4504 | [+0.0747, +0.8169] | — | predicted sign holds |

Per-token density 0.0668 (`S`) vs 0.0162 (far window) = **4.13×**, both at identical dose.
Development→confirmation shrinkage 1.29×. Memorials: clean anchor +1.866, cover balance 0.828 nt,
non-degeneracy 0.817 vs 0.113, batch invariance 5.25e-06 — all pass.

**Bounds.** E_reg is **regional, not positional** (composition confound, D14's P2 still open). The
magnitude is skew-inflated and the per-locus sign test **fails**: median +0.1464 is below the SESOI and
only 104/186 loci are positive (exact p=0.1234); read the magnitude as ≈0.15–0.29 and the effect as
distributional. **The "far-upstream is inert" hypothesis is refuted by this study's own data** —
E_band's point estimate is negative, masking the whole band costs +0.8168 against the signal window's
+0.4675, and the band arm is super-additive (5× a window = +0.566), so far-upstream is diffuse and
low-density rather than unimportant, and dose-30 masking is a different regime. No mediation is shown;
attention and behaviour come from different cohorts; "relay" remains an inference with no path patched.

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
