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

**D3 — replicated DNABERT-2 context-margin behavior.** Under the frozen context-margin-v1
amendment, nine classes replicate a positive target-preserving proximal-context effect on the
geometry-qualified alternative margin: start codon, distal-enhancer anchor, poly(A) cleavage
region, branch point, REST/CEBPB/CTCF peak anchors, poly(A) signal, and polypyrimidine tract.
Development/confirmation effects are respectively +0.0468/+0.0479, +0.0395/+0.0384,
+0.0298/+0.0348, +0.0412/+0.0304, +0.0136/+0.0213, +0.0215/+0.0183,
+0.0141/+0.0159, +0.0127/+0.0127, and +0.0109/+0.0125 nats per covered nucleotide.
Every confirmation result exceeds the frozen +0.01 SESOI, has a positive cluster-bootstrap lower
bound, and passes Holm correction over the opened family.

This licenses a replicated **model-behavior context effect**: the preregistered proximal context
supports the true target over its fixed eligible Hamming-one alternatives more than an
equal-width, equal-composition, equal-change-count distal corruption. It does not establish
biological function, a resolved motif, mediation, or a causal model component. The null is distal,
so proximity-sensitive encoding remains part of the licensed effect.

Promoter qualified in development (+0.0247) but did not replicate (+0.00066, 95% CI
−0.0117 to +0.0126, Holm `p=0.4548`). Stop codon failed development (+0.00330).
Splice acceptor and donor were quarantined before target scoring because the margin positive
control was unresolved; none of these four outcomes is silently pooled with the replicated set.

**D4 — fine-tuned promoter behavior replicates; context mechanism does not.** The selected
`prom_300_all` DNABERT-2 classifier replicates a positive clean promoter margin on
exact-GUE-excluded GENCODE v50 protein-coding MANE Select TSS windows relative to
composition-matched, TSS-excluded same-locus pseudo-TSS windows. Development and confirmation
effects are +2.0956
(95% CI +1.7813 to +2.4080) and +1.8892 (+1.6118 to +2.1867) classifier-logit-margin units;
both Holm-adjusted add-one values are 0.0002.

The target-preserving local-context difference-in-differences does **not** replicate. It is
+0.6655 (+0.3250 to +1.0080, Holm `p=0.0002`) in development and +0.2750
(−0.0684 to +0.6151, Holm `p=0.0569`) in confirmation. The confirmation point estimate exceeds
the +0.25 SESOI, but its interval and corrected p-value fail. D4 licenses supervised promoter
classification behavior on the declared external cohort. It licenses no context-specific
mechanism, internal component, necessity, or mediation claim.

**D5 — a development-frozen eight-head set causally supports the clean promoter contrast.**
The frozen promoter head atlas scanned all 144 attention heads at the unpadded `[CLS]`
destination on 192 unique GENCODE locus blocks per split. Development selected
L2H2, L4H6, L4H7, L9H7, L10H1, L10H5, L11H2, and L11H5 without inspecting confirmation.
Jointly replacing those target head-context slices with their paired pseudo-TSS values reduced
the target promoter margin by +1.0614 logits in development (95% CI +0.8179 to +1.3146) and
+0.7322 in confirmation (+0.5443 to +0.9289). The reciprocal target-to-pseudo-TSS transfer
raised the pseudo-TSS margin by +1.4166 (+1.1490 to +1.6863) and +1.1384
(+0.9041 to +1.3746). All four estimates exceed the frozen +0.60 SESOI. In each split and
direction, the candidate set exceeds all 63 distinct candidate-excluding random sets with the
exact same per-layer head counts; two-endpoint Holm-adjusted add-one `p=0.03125`.

Attribution/exact-patch head rankings remain positively associated in development
(Spearman 0.747/0.876 for necessity/sufficiency) and confirmation (0.602/0.799).
L11H2 and L11H5 have the largest individual exact-patch effects; L9H7 and L10H5 also have
positive corrected signs in both directions on both splits. Those individual rows remain
candidate priorities because only the eight-head group received a matched-head null test.
L9H7's target `[CLS]` attention is concentrated proximal-upstream (0.548 mass) and shifts
+0.0458 relative to controls, with a +0.0108 TSS-anchor shift; this is an attention clue, not
causal source localization.

D5 licenses replicated **candidate-set `[CLS]` head-output mediation** of D4's clean supervised
TSS-versus-pseudo-TSS contrast. It does not revive D4's failed local-context endpoint, establish
individual-head specificity, distinguish QK selection from OV content, identify a motif or
source position, resolve a downstream path, prove a sparse circuit, or establish transcriptional
function in cells.

**D6 — L11H2/L11H5 is a replicated late output-carrier core with a distributed MLP
scaffold.** The frozen circuit assay uses the 192 unique 1-Mb locus blocks per split that were
not used in D5. In development, exact joint L11H2/L11H5 patches have +0.6467 necessity
(95% CI +0.4278 to +0.8772) and +1.0680 sufficiency (+0.8251 to +1.3089). Confirmation
replicates +0.7004 necessity (+0.4833 to +0.9261) and +1.0932 sufficiency
(+0.8310 to +1.3631). In every split and direction, the pair exceeds all 45
candidate-excluding two-head pairs in layer 11; two-endpoint Holm-adjusted exact add-one
`p=0.04348`.

Strict complement scrub over all 144 `[CLS]` head contexts and 12 `[CLS]` MLP writes shows that
the eight D5 heads retain only 15.30% (95% CI 13.57% to 17.25%) in development and 13.73%
(12.07% to 15.56%) in confirmation. L11H2/L11H5 alone retains nearly the same
15.18%/13.62%. Retaining all MLP writes with the eight heads raises faithfulness to 61.04%
(57.30% to 65.51%) and 61.27% (57.30% to 65.61%). All MLP writes without native head writes
retain exactly zero, while all native head writes without MLP writes retain 34.36%/31.30%.
Empty and all-node anchors are exactly zero and one.

Two proposed mechanisms fail their frozen gates. Paired-control transfer of only the
L11H2/L11H5 target queries is effectively zero (−0.00004 development, −0.00034 confirmation)
and is not specific against the 45-pair family. Corrupting the six earlier D5 heads costs only
+0.0585/+0.0563 logits, below the +0.25 SESOI; therefore the small late-pair rescue cannot
license an early-six-to-late directed path even though its descriptive rescue fraction is
30.2%/31.6%.

D6 licenses a replicated **layer-specific two-head causal core at the late `[CLS]` head-output
node** and a continuously measured dependence on distributed MLP writes. The supported model is
a late output-carrier pair operating on an MLP-maintained representation. It is not a
target-dependent query gate, a demonstrated early-to-late chain, a complete sparse circuit, a
resolved K/V source or motif, or a claim about transcription in cells.

**D7 — The late L11H2/L11H5 core has replicated value-payload alignment dependence, not
detectable target-specific key-address dependence.** This separately frozen assay uses a newly
constructed cohort of 192 development and 192 confirmation locus blocks after excluding every
example and 1-Mb block in the original 768-locus promoter cohort. At a fixed dose of 40 genomic
tokens per sequence, deterministic V permutation leaves Q, K, QK logits, and post-softmax
attention probabilities native while reassigning which source value vector each probability
weights.

V permutation of L11H2/L11H5 causes a target-specific promoter-margin loss of +0.1026 logits in
development (95% CI +0.0517 to +0.1552) and +0.0615 in confirmation (+0.0030 to +0.1210). In
each split it exceeds every one of the 45 candidate-excluding two-head pairs in layer 11 and
passes the two-endpoint Holm gate (`p=0.04348`). By contrast, K permutation is only
+0.0073/+0.0084 logits, below the frozen absolute 0.05-logit SESOI in both splits. It therefore
does not support a target-specific content-key address role under this intervention.

The paired target-minus-control residual-write direction is stable between splits
(mean-direction cosine 0.9908), with a positive confirmation projection of +3.111
(95% CI +2.287 to +3.904) and 75.0% positive loci. These direction measurements are descriptive:
they received no semantic or direction-only causal test. Joint K/V permutation is also secondary;
its confirmation estimate is +0.0481 and its interval crosses zero.

D7 licenses a **causal value/source-state payload-routing dependence** for the replicated late
pair: the model's native attention probabilities must remain associated with the correct value
vectors for the promoter margin. Together with D6's null query-transfer result, it narrows the
pair away from target-specific Q or K addressing. It does not identify a source band, nucleotide
motif, promoter grammar, semantic residual direction, complete circuit, or transcriptional
function in cells.

**D8 — Equal-dose masking localizes a replicated classifier-relevant source region over the
nominal +11 to +21 nt interval, but the late-pair V payload and independent biological labels do
not localize there.** The outcome-blind assay uses the final 192 unused exact-GUE-excluded locus
blocks per split. All thirty 10-nt nominal tiles receive exactly two adjacent genomic BPE tokens per target
and pseudo-TSS sequence. Candidate generation is independent of CAGE, cCRE, CpG, and motif labels.

Tile 26 (+11 to +21 relative to the TSS) is the sole development-qualified region. Its
target-specific mask loss is +0.2452 logits (95% CI +0.0755 to +0.4285), its equal-dose
localization contrast is +0.2656 (+0.0953 to +0.4442), and its joint 60-test maximum sign-flip
`p=0.0260`. The unchanged confirmation test replicates at +0.2056 (+0.0447 to +0.3916;
`p=0.0082`). Actual BPE covers have median length 10 nt and range 6–15 across the experiment, so
this is not exact ten-nucleotide resolution.

The regional L11H2/L11H5 V swap does not qualify at any tile: the largest raw and localization
effects are +0.00124 and +0.00119, below the frozen +0.005/+0.002 SESOIs. ENCODE PLS overlap
qualifies in development but not confirmation. CAGE is underpowered at tile 26, CpG does not
qualify, and no geometry-qualified EPD TATA, Inr, CCAAT, or GC-box instance intersects the
nominated interval.

D8 licenses a replicated **classifier-level behavioral source region**. It does not license a
canonical promoter motif, a replicated biological annotation, a regional source for D7, a
mask-to-payload mediation claim, or transcriptional function in cells.

**D8 secondary performance boundary — the nominated mask reduces ranking quality and causes a
small number of target label flips, but not net balanced-accuracy loss.** In a user-requested
post-hoc evaluation of the unchanged tile-26 intervention, AUROC falls by 0.0332 in development
(95% pair-bootstrap CI −0.0604 to −0.0069) and 0.0207 in confirmation (−0.0431 to −0.0019).
Three of 107 and four of 124 real promoters flip from promoter to non-promoter. Accuracy changes
by −0.0093 and +0.0081, with both intervals spanning zero, because six pseudo-TSS false positives
also become correct negatives and other reverse transitions occur.

This supports a bounded hard-label and ranking consequence on the fresh GENCODE/pseudo-TSS
challenge. It does not show an accuracy loss on the original GUE task, independently validate
the selected region, or identify its biological content.

**D9 — the +11:+21 `[MASK]` nomination does not survive native sequence-order
qualification.** A fresh, outcome-blind cohort admits 60 unique locus blocks per split under an
exact mononucleotide- and linear-dinucleotide-preserving nonidentity reorder. Target and
pseudo-TSS edits have equal Hamming dose and matched local BPE retokenization geometry. The final
disjoint GUE preflight passes sensitivity, precision, identity, batch, geometry, and finiteness
checks.

Development target-specific promoter-margin loss is −0.0162 logits (95% CI −0.1762 to +0.1383;
add-one cluster-sign and Holm `p=0.5834`), failing the frozen +0.10 SESOI, positive interval, and
sign gates. Clean target-minus-control behavior remains strong at +2.321 (+1.392 to +3.266;
`p=0.0001`). Accuracy is unchanged at 0.650, AUROC changes descriptively from 0.7075 to 0.7133,
and one of 60 targets flips promoter→non-promoter.

D9 licenses a **failed development positive gate for exact-composition sequence-order
necessity under this intervention**. The interval still includes the +0.10 SESOI, so this is not
an equivalence null. D8 remains a post-tokenization mask-sensitive candidate, not established
native-sequence evidence. D9 does not show that +11:+21 is biologically inert, exclude every
native substitution, or resolve GC content: the GC-depletion instrument failed its frozen
precision gate. Confirmation, residual-source tracing, and head mediation remain sealed.

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
| IF16 | Context-margin supply retains 4,799 target-preserving structured/irrelevant variant pairs across all 13 classes with zero target-cover, geometry, dose, or split violations; every class supplies at least 82 loci on each target split. | `results/dnabert2_reconstruction_atlas/context_margin_v1_supply_audit.json` | Outcome-blind supply only; attrition is substantial and retained rather than repaired. |
| IF17 | The corrected frozen context-margin preflight is `QUALIFIED` for 11 admitted classes. Copy-margin positive controls, irrelevant-control equivalence, ≤0.006 MDE, ≤0.004 CI half-width, exact geometry/dose, and model health pass. Acceptor and donor remain preflight-quarantined. | `results/dnabert2_reconstruction_atlas/context_margin_v1_preflight_qualified_rerun.json` | Authorizes the context-margin development run; does not itself establish D3. The prior failed qualified artifact is preserved as a gate-membership implementation failure. |
| IF18 | Among compatible human all-promoter DNABERT-2 heads, `prom_300_all` has the stronger native GUE test result (MCC 0.8705, AUROC 0.9848) and a complete pinned two-class head. | `results/dnabert2_reconstruction_atlas/promoter_classifier_checkpoint_audit_v2.json` | Checkpoint selection only; it does not establish promoter behavior on an external cohort. |
| IF19 | Every selected 300-nt EPDnew atlas window occurs verbatim in GUE; the 70-nt core task also has 768/768 exact membership at its best one-base alignment. | `results/dnabert2_reconstruction_atlas/promoter_classifier_checkpoint_audit_v2.json` | EPDnew is excluded from the fine-tuned target run; scoring it would be training-source contaminated. |
| IF20 | The replacement GENCODE cohort contains 384 development and 384 confirmation MANE TSS locus blocks, zero exact forward/reverse GUE memberships, zero split leakage, exact anchor/composition/dose invariants, and bounded GC/CpG imbalance. | `results/dnabert2_reconstruction_atlas/promoter_classifier_cohort_audit_v1.json` | Outcome-blind cohort fact only. Exact exclusion does not rule out undocumented near-duplicates. |
| IF21 | Promoter preflight v1 is `INSTRUMENT_UNRESOLVED` because a one-draw permuted-label MCC and an over-tight CPU batch tolerance fail; no GENCODE target outcome was opened. | `results/dnabert2_reconstruction_atlas/promoter_classifier_preflight_v1.json` | Preserved instrument failure, not a promoter null. |
| IF22 | Promoter preflight v2 is `QUALIFIED` after replacing the volatile one-draw null with a 256-permutation center and rounding the numerical tolerance to \(10^{-5}\), while retaining all scientific SESOIs. | `results/dnabert2_reconstruction_atlas/promoter_classifier_preflight_v2.json` | Authorizes the D4 target run; preflight itself is not external-cohort behavior evidence. |
| IF23 | Reserved-GUE calibration validates the promoter head instrument: exact no-op patching is zero, batch drift is \(4.77\times10^{-7}\), all-head CLS transfer recovers the class contrast, attribution tracks exact patches, and the calibrated group exceeds exact layer-profile nulls. | `promoter_head_atlas_calibration_v1.json` and `promoter_head_group_calibration_v1.json` | Calibration only; its GUE-ranked heads are not target candidates and create no GENCODE internal claim. |
| IF24 | Frozen promoter-head preflight v1 is `QUALIFIED` on 32 GUE pairs disjoint from calibration: GUE MCC/AUROC, attribution agreement, full/no-op/same-class patches, projected sensitivity, null-set resolution, token geometry, head capture, and model health all pass. | `promoter_head_atlas_preflight_v1.json` | Authorizes the development head atlas; it is not a GENCODE component outcome. |
| IF25 | Frozen promoter-circuit preflight v1 is `QUALIFIED` on 16 new GUE pairs: all 156 declared CLS write nodes are covered, full scrub reproduces paired-control margins exactly, empty complement and self/query/restore no-ops are exactly zero, batch drift is \(4.77\times10^{-7}\), and full-scrub sensitivity is 8.14 logits. | `promoter_circuit_preflight_v1.json` | Authorizes D6 development only; GUE controls are not evidence about unused GENCODE loci. |
| IF26 | The expanded promoter-decomposition cohort supplies 768 exact-GUE-excluded MANE TSS/pseudo-TSS pairs per split. Outcome-blind selection binds 192 fresh example IDs and unique 1-Mb blocks per split after excluding the complete original promoter cohort; every sequence has at least 44 genomic tokens, supporting a fixed surgery dose of 40. Reserved-GUE calibration gives projected 192-pair MDEs of 0.0170 logits for K and 0.0456 for V at that dose. | `promoter_decomposition_cohort_audit_v1.json`, `promoter_decomposition_supply_audit_v1.json`, and `promoter_address_payload_calibration_v1.json` | Cohort, token-geometry, and calibration facts only; no fresh GENCODE model outcome was read before the lock. |
| IF27 | Frozen promoter address/payload preflight v1 is `QUALIFIED` on 32 reserved GUE pairs: identity error is \(2.38\times10^{-6}\), all-layer-11 K/V target-specific sensitivity is 0.0372/0.2049 mean absolute logits, projected MDE is 0.0032/0.0262, batch drift is zero, all 45 matched null pairs are present, and residual writes are finite. | `promoter_address_payload_preflight_v1.json` | Authorizes D7 development only; reserved-GUE controls are not evidence about fresh GENCODE loci. |
| IF28 | The source-localization supply binds 192 final unused locus blocks per split, 107–140/111–133 eligible pairs per tile, 105 CAGE-supported TSSs per split, 138/142 PLS TSSs, 70/73 CpG-island TSSs, and geometry-qualified motif disruptions without reading a target outcome. Reserved-GUE calibration gives identity error \(9.54\times10^{-7}\), mask sensitivity 0.463 mean absolute logits, and candidate regional-V projected MDE 0.00297. | `promoter_source_localization_supply_audit_v1.json` and `promoter_source_localization_calibration_v1.json` | Outcome-blind supply and calibration only. |
| IF29 | Frozen promoter source-localization preflight v1 is `QUALIFIED` on 32 new reserved-GUE pairs: all identity, mask/V sensitivity, precision, tile-resolution, batch, finiteness, and 45-null-family checks pass. | `promoter_source_localization_preflight_v1.json` | Authorizes D8 development only; not evidence about fresh GENCODE loci. |
| IF30 | Native-source preflights v7 and v8 are `INSTRUMENT_UNRESOLVED`: three-base GC depletion has projected MDE 0.403 at 40 loci, and a disjoint dinucleotide-only slice has projected MDE 0.364. No target outcome was read in either run. | `promoter_native_source_preflight_v7.json` and `promoter_native_source_preflight_v8.json` | Preserved instrument failures; neither is a biological null. |
| IF31 | Dinucleotide-only supply v9 binds 60 fresh unique locus blocks per split; frozen preflight v10 passes all eight checks on a third disjoint GUE slice with projected 60-locus MDE 0.156 and zero replay/batch drift. | `promoter_native_source_supply_audit_v9.json` and `promoter_native_source_preflight_v10.json` | Authorizes D9 development only. |

## Open hypotheses

| ID | Hypothesis | Current status | First required bridge |
|---|---|---|---|
| H1 | At least one annotated site class has positive excess reconstruction over every required control. | not supported in multisource-v4 development; five adequately powered classes fail and eight classes remain underpowered | A newly preregistered behavior design or expanded control-complete supply; v4 confirmation stays sealed. |
| H2 | For at least one H1-qualified class, a biologically related context region outperforms matched irrelevant-region exchanges. | blocked by the v4 H1 gate | H1 plus token-geometry-invariant context interventions. |
| H3 | A bounded set of heads, MLPs, positions, or residual directions causally supports an H2-qualified behavior. | blocked by the v4 H1 gate | Frozen candidate screen, matched-size necessity, mediation, and path tests. |
| H4 | A geometry-qualified DNABERT-2 alternative margin has a replicated, class-specific target-preserving proximal-context effect. | supported for the nine D3 classes; promoter did not replicate, stop failed development, acceptor/donor are instrument-unresolved | A distance-matched or orthogonal context null would narrow the proximal-context interpretation further. |
| H5 | A bounded internal DNABERT-2 component set causally mediates a D3 context-margin effect. | open but untested for the nine D3 classes | Freeze candidates and matched-size necessity/mediation tests before inspecting internal effects. |
| H6 | A promoter-fine-tuned DNABERT-2 has a replicated target-specific local-context effect at training-corpus-excluded human TSSs. | not supported: clean promoter margin replicated, but the context difference failed confirmation | A new independently justified context intervention/null and a fresh sealed cohort; D4 does not authorize internal localization. |
| H7 | A bounded set of head outputs at `[CLS]` mediates the replicated clean promoter-classifier contrast. | refined by D7: L11H2/L11H5 is a replicated late output-carrier core with value-payload alignment dependence and a distributed MLP scaffold; target-query, content-key-address, and early-six-to-late models fail their gates | Freeze an equal-dose source-localization phase for the replicated V estimand; do not infer bands or motifs from the stable residual direction alone. |
| H8 | A bounded TSS-relative source band or sequence feature supplies the late pair's replicated value payload. | not supported at +11:+21: regional V was null and exact-composition native reordering failed development; the `[MASK]` effect remains classifier-level only | A new independently justified native intervention/source band and fresh cohort; D8/D9 do not authorize mask-by-payload tracing. |

Negative, retracted, and superseded rows will remain here permanently once evidence exists.
