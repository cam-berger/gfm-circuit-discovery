# Results and manuscript audit

This file summarizes the repository's quantitative results, each with its controlling artifact, and
audits one manuscript draft against them.

The draft audited is the 411-line workshop manuscript supplied as an external attachment, SHA-256
`0439a1c73f97679e151083ace0fbf4dc85e089d3e71bf9e98967240f18a6b8cf`. The checkout has no live `tex/`
source for it, so line numbers below refer to the attachment. The evidence audited is the state
inherited from commit `409e154`.

Audit date: 2026-08-22. Checkpoint: SpliceBERT.1024nt, SHA-256 `2ad91428…0149d`. Primary study:
[`studies/splicebert_partner_selection/CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md) and
[`studies/splicebert_partner_selection/CORRECTIONS.md`](../studies/splicebert_partner_selection/CORRECTIONS.md).

When sources disagree, the claim and correction ledgers and
[`docs/PREREG_CONFORMANCE.md`](PREREG_CONFORMANCE.md) take precedence over immutable artifacts and
manifests, which take precedence over results indices, captions, and narrative summaries. This file
replaces `RESULTS_VERIFICATION.md`, which covered only C1--C26 as of 2026-07-16 (Section 7).

## 1. Verdict

The paper's central behavioral number and most intervention values reproduce their controlling
artifacts. The strongest synthesis safe to publish is:

> SpliceBERT contains a structured, top-ranked causal core for paired-acceptor-dependent masked-donor
> prediction. Positional geometry supplies an address, acceptor-local values supply a distinct payload,
> L3H1 combines them into a causally useful donor-side state, and later heads deliver that state. The named
> core is highly injectable but not a faithful autonomous generator; held-out static-ranking results show
> a gradual, distributed faithfulness boundary rather than a compact circuit.

The manuscript is not fact-check complete as written; Section 4.1 lists the required changes. The most
serious, by draft line: 195 reverses the sign of the C38 transfer contrast; 87 defines `\ell_b` as a
log-probability, not a raw pre-softmax logit; 103--105 runs the 96-head scan, the sealed two-head test,
and the 400 random two-head null interventions together; 114 attaches an all-16-Layer-0-head recovery
number to a displayed three-head set (R54); 35, 201, and 245 read minimality into C33; and 55 links the
period-3 carrier, L3H1's predominantly positional address, and C43's annotated boundary-register
preference, which have not been causally connected.

## 2. Repository verification

| Check | Result |
|---|---|
| `uv run python scripts/verify_manifest.py` | PASS after cleanup: exact coverage of all 1,108 `results/MANIFEST.csv` rows |
| Manifest status census | 651 canonical, 256 development, 180 superseded, 21 infrastructure |
| Manifest polarity census | 534 positive, 23 null, 5 refutation, 111 bounded, 435 not applicable |
| `uv run --extra splicebert --extra biology --extra dev pytest -q` | PASS: 1,023 passed, 22 skipped in 221.40 s |
| Focused manifest, ledger, hygiene, figure-provenance, and DNABERT checks | PASS: 93 passed |

The skips are declared data or optional-environment skips, not test failures. No model was rerun;
headline arithmetic was taken or recomputed directly from the controlling artifacts, the C33
interpolation from the committed held-out curve rather than its grid-snapped summary.
`verify_manifest.py` checks registry, path, and schema closure only. It does not establish scientific
truth, verify every output hash, fill missing producers, or make an artifact clean-checkout
reproducible. Passing memorial tests are regression locks on recorded values, not independent reruns.

Do not call the repository uniformly preregistered or say all gates passed. The conformance audit covers
25 SpliceBERT-arc families and records 188 deviations, 157 of them medium or worse; it states that none
overturns a scientific result, but they limit evidence tier, provenance language, and claim strength.
The unmanifested legacy `confirmation.json` was archived off the active figure path. Remaining
paper-facing gaps: post-hoc or unmanifested parts of C27, the C33 grid-snap deviation, and C36's
missing analysis code pin and sensitivity reporting.

## 3. Data splits and example counts

### 3.1 Donor-coupling population

The donor assay uses hg38 and GENCODE v50 basic (GTF
`5b9ad9aa71f6ab5d55123c51f74723644a895bc0cbaa9cee8fbc11bb9dec0bf7`, FASTA
`5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51`). A `full-build` window is a
deduplicated, sequence-valid 1,024-nt internal-exon window that passed the builder's length, motif,
boundary, and ACGT checks. An `eligible` window also has a canonical `GT` donor, its downstream
intron-mate `AG` acceptor inside the window, and a 60--150-nt donor-to-acceptor gap. The assay
population is the 22,694 eligible rows, a subset of the 232,867 built windows. Validation is
`{chr11, chr16, chr19}`, the sealed test is `{chr8, chr9}`, and the other 20 chromosomes (chr1--chr7,
chr10, chr12--chr15, chr17, chr18, chr20--chr22, chrX, chrY, and chrM, which has no windows) are train;
the obsolete v1 config in `configs/splicebert_partner_selection/base.json` assigns different chromosomes
and does not control.

| Chromosome | Split | Full-build windows | Eligible sequences | Eligible genes |
|---|---|---:|---:|---:|
| chr8 | sealed test | 7,878 | 783 | 205 |
| chr9 | sealed test | 9,377 | 785 | 249 |
| chr11 | validation | 13,273 | 1,650 | 446 |
| chr16 | validation | 10,202 | 1,682 | 394 |
| chr19 | validation | 14,133 | 2,598 | 676 |
| All 20 train chromosomes | train | 178,004 | 15,196 | 4,933 |
| Total validation | validation | 37,608 | 5,930 | 1,516 |
| Total sealed test | sealed test | 17,255 | 1,568 | 454 |
| All splits | | 232,867 | 22,694 | — |

| Split | Full-build | Canonical GT | Partner in window | Gap 60--150 (eligible) |
|---|---:|---:|---:|---:|
| Train | 178,004 | 175,475 | 52,606 | 15,196 |
| Validation | 37,608 | 37,032 | 15,885 | 5,930 |
| Test | 17,255 | 17,002 | 4,895 | 1,568 |

### 3.2 Frozen draws, and what "400" means

| Cohort or null family | Rows / interventions | Genes | By chromosome | Use |
|---|---:|---:|---|---|
| Unconditional validation behavior | 400 rows | 334 | chr11 98/83 genes; chr16 94/82; chr19 208/169 | Validation behavior |
| Unconditional sealed test behavior | 400 rows | 211 | chr8 181/94 genes; chr9 219/117 | C1/C2 behavior |
| Conditional validation mechanism | 150 rows | 140 | chr11 33; chr16 48; chr19 69 | Mechanism development |
| Conditional sealed test mechanism | 150 rows | 100 | chr8 77; chr9 73 | C3 sealed cohort; health and null use its first 100 rows |
| Corrected exhaustive head scan | 160 rows | 146 | chr11 48; chr16 39; chr19 73 | Development candidate ranking across 96 heads |
| Matched two-head null | 400 head-pair interventions | first 100 mechanism rows | not 400 new sequences | Add-one `p=1/401` for C3 |

The per-chromosome mechanism counts are exact deterministic re-derivations that reproduce the stored
aggregate means; most compact artifacts do not store row IDs. Label them "re-derived," and do not
rescore or reopen chr8/9 to enrich old metadata.

### 3.3 Other studies

These are separate assays: do not pool them, and different rows alone do not make them independent
replications.

| Study | Cohort | Counts and status |
|---|---|---|
| Acceptor coupling | GENCODE v50 comprehensive; val chr11/16/19; intended target chr8/9 | AC1 draw 0: 974 pairs/645 genes. AC5--AC8 target: 500 rows/304 genes (validation chromosomes). AC9 preflight: 400/321, then schema stop. No chr8/9 acceptor target scored. |
| Head-context atlas | GENCODE v50 comprehensive | Legacy usable train/val/test 32,054/11,275/3,014 rows, 4,864/1,567/444 genes. Fresh gene-hash discovery 11,807 packets/5,089 genes; internal validation 4,019/1,486. Sealed chr8/9 confirmation not run. |
| GTEx fine-tune shift | GENCODE v50 comprehensive | Fixed validation cohort of 400 rows/334 genes (chr11 98/83, chr16 94/82, chr19 208/169), reused across five seeds (not 2,000 rows). Fresh train discovery/confirmation families use 128/200 rows; chr8/9 sealed. Fine-tuning training-set identities not in this repo. |
| C43 register assay | Designed fixed-span chimeras | Replication: 384 gene-disjoint blocks, 3,456 lattice cells, 2,304 native rows. C44 ran a 114-block development preflight; its 384-block target was never opened. |
| Positional boundary coherence | Model-free capacity stop | Required 1,104 blocks; donor/acceptor gene ceilings 809/854; zero blocks selected, no model loaded. |
| Positional proxy modifiability | Model-free capacity stop | Required 718 blocks; exact ceiling 726, usable 25%-margin ceiling 544; no confirmatory cohort selected. |
| DNABERT-2 promoter | GUE, not GENCODE | Raw train/dev/test 47,356/5,920/5,920; 47,149/5,856/5,865 after exact duplicate, conflict, and cross-split cleaning. No chromosome or gene metadata, so no chromosome-level breakdown. |

The exploratory or superseded legacy `acceptor-interpretability/`, `ntv3-interpretability/`, and
top-level `dnabert/` cohorts are recoverable at `archive/unsupported-exploratory-2026-08-22` and are
not paper evidence.

## 4. Manuscript claim audit

### 4.1 Corrections required before submission

| Line(s) | Problem | Fix |
|---|---|---|
| 31, 37, 51, 55, 65--67, 247, 251 | Biological meaning or wet-lab hypotheses beyond sequence statistics | Model-internal hypothesis generation. No external assay identifies payload semantics, splicing activity, U2AF/branch-point biology, or a uniquely implied experiment. |
| 49, 59, 78 | "First," "has not yet been applied systematically," literature-wide rarity | Remove, or hold for a current literature review. |
| 55 | Address equated with "a phase-aligned period-3 register" | Separate them. Frozen A11 says MIXED (result and manifest disagree) despite the artifact's "predominantly relative" verdict string (R55); the live C32 reads a substantial relative-offset signal with absolute/edge modulation. C36 and C43 are separate results; C43 is a behavior-level construct result. |
| 59, 245 | "No sparse faithful set exists," "irreducibly distributed" | No sparse knee in the tested held-out static-ranking curve: a structured top-ranked core within distributed computation, and an upper bound, not an impossibility proof. |
| 83 | "Five causal hypotheses," not enumerated | List them, or "a sequence of causal hypotheses." |
| 35, 241--243 | Four "components"; whole account replicated | Four causal roles, each with several components. C26 reproduces the donor-side `U_info` state, not the whole account. |
| 87, 318--321 | "Masked dinucleotide"; `\ell_b` a log-probability | Only the donor `G` is masked; `\ell_b` is its raw pre-softmax logit. |
| 92 | "All splits are gene-disjoint" | Analysis cohorts and split construction only; no claim about SpliceBERT pretraining genes. |
| 103--105 | 96-head scan, sealed paired test, and 400-pair null blended | Scan on 160 validation rows; C3 on 150 sealed-test rows; health and the 400 random-pair null on the first 100. `99.2%` is a retention ratio (`0.51065/0.51457`), not 99.2 accuracy points; `1/401` is an add-one null p-value, not scan-wide. L1H12 also contributes, so "two dominant" is not completeness. |
| 108 | Attention "reads," values "write" | Attention is the address, acceptor-local values the payload, the donor-side post-L3H1 state the write. |
| 111 | A1's `2.718` read as relative position | `2.718` is positional over local-token influence. A11 keeps a large preference under common translations, with up to 23.7% nonmonotonic, edge-sensitive modulation; frozen decision MIXED. |
| 114, 151--154 | `+0.136` as "the complete set" or "six heads," in implied logit units | `0.348/0.160/0.116` are dimensionless conditional-necessity ratios. `R=0.136` restores all 16 Layer-0 heads; adding MLP0 gives `R=0.356` (R54). `{L0H7,H8,H10,H11,H12,H13}` pass the held-out rule; their joint restoration was not measured. H7/H12/H10 are the largest, not a demonstrated complete or minimal set (sufficiency arm not run). |
| 117 | Rank 8 at 92% vs "a random rank-8 subspace" | "versus a maximum of 5.30% across the matched random rank-8 controls." Rank 8 is tested and useful, not uniquely selected or saturating. |
| 117, 159, 342 | `+0.875` shown with `[0.853,0.901]` | The estimate is the A→B/B→A mean; the interval is A→B only. |
| 195 | Mismatched source said to exceed matched by 0.158 | Reversed: matched exceeds margin-mismatched by `+0.158`, CI `[-0.110,0.331]`, unresolved. C38 is verification tier and does not establish example identity. |
| 197 | "Common to all donors"; 551 "targets" | 1,012 target examples from 551 source-disjoint genes. Bound to C41's additive final-residual assay (six of eight draws generic, two instrument-limited), consistent with a gene-generic readout axis in the tested linear regime; not addressing, routing, payload, or nonlinear specificity. |
| 199 | One "nine-node core," injectable and isolated | Two sets: post-hoc `D(C\\embed)=0.986` (nine non-embedding spine nodes, corrupt embedding kept) and preregistered `F(C)=0.00539` (ten nodes with the embedding). Reconcile artifact unmanifested; `N(C\\embed)=0.9888` has no CI or null. |
| 199 | Workflow Figure 1B cited for faithfulness | Cite the C27/C33 panel or supplementary curve; the workflow panel is not quantitative evidence. |
| 35, 201, 245 | "90% requires 64--72 of 103," as minimality proof | Interpolated `64.2`; first sampled crossing `72/103`. Upper-bound construction; non-monotone at high `k`. |
| 210, 391--392 | Early P3 full-band deletion as a clean lesion | Supportive dose/control evidence: health field false, control result at the `1/22` floor, seven of 21 control bands overlap the second carrier, "CDS" label not annotation-derived. C36 is cleaner. |
| 208--214, 391--392 | P3 readable as reading frame or training origin | C8/C36 show model use; C36's CDS-minus-non-CDS contrast is inconclusive. C39 does not isolate species diversity, corpus size, composition, sampling, or optimization. |
| 225 | C44 "was not run" | A 114-block development preflight ran and failed four authorization gates; the 384-block target did not run. An instrument stop, not a positive, null, or inverted P3-to-register result. |
| 232 | "No internal mechanism has been traced" | No reverse four-role mechanism. AC5 (L1H12 mean-ablation specificity) and AC6 (L4H11/H9/H7 same-example noising) are component effects, not address, write, delivery, mediation, symmetry, or chr8/9 evidence. |
| 234 | DBP8 fails only after correction; five-node effect "analogous" | `0.036` (matched `0.879`, same-class `0.843`) is nonsignificant already at raw `p=0.252`. State one seed, no preregistration, no calibration pair, incomplete manifest and producer provenance. |
| 397--398 | DNABERT effect as "normalized coupling" | "Normalized promoter-evidence corruption effect"; not donor-acceptor coupling. |
| 403--407 | "Only within coding-sequence blocks" | Under phase-aligned carrier shifts, more coupling is preserved in CDS than in matched upstream-intron blocks; keep the inconclusive CDS-minus-non-CDS-exon contrast. |

### 4.2 Verified quantities

Each row's supported wording and scope are in its ledger:
[donor partner selection](../studies/splicebert_partner_selection/CLAIMS.md) for C rows,
[acceptor coupling](../studies/splicebert_acceptor_coupling/CLAIMS.md) for AC rows, and
[DNABERT-2 promoter](../studies/dnabert_promoter/CLAIMS.md) for DBP rows. Section 4.1 lists the
corrections that apply to them.

| Result | Artifact-backed quantity | Controlling artifact |
|---|---|---|
| Behavior (C1/C2) | `+0.872 [0.771,0.969]`, n=400; controls `+0.007/+0.005` | `results/splicebert_coupling/sealed_test.json` |
| Corruption robustness (C42; behavior level, acceptor operator only; A0–A12 remain CC-only, R44) | n=1,000/648 genes, val; every admissible partner-acceptor alternative to `AG->CC` inside the frozen `[0.75,1.33]` equivalence band (`0.861--0.982`) | `results/splicebert_coupling/ar1_corruption_operator_sensitivity.json` |
| Dominant heads `{L3H1,L2H15}` (C3) | pair loss `0.8161`; health retained `0.9924`; 400 matched pairs, `p=1/401` | `sealed_test.json` |
| Early K/V conjunction (C4; not the later address/value factorial) | key `0.0320`, value `0.1005`, joint `0.6633`, interaction `0.5307 [0.4774,0.5833]` | `sealed_test.json` |
| Address-payload factorial (C23) | address `0.0104`, payload `0.0514`, joint `0.3645`; interaction `0.3028` | `a2_address_payload.json` |
| Address (C22/C32/C25) | position `2.7176 [2.2229,3.1535]`; content `0.3973 [0.1808,0.6198]`; L2H15 address/value rescue `0.4536/0.0194` | `a1_token_position.json`; `a11_address_geometry.json`; `a3_l2h15_address_payload.json` |
| Payload localization (C24) | four-token values `+0.3820 [0.3490,0.4153]`; `U_info` `0.0604`, random range `0.0275--0.0699`, `p=0.2195` | `a2b_payload_localization.json` |
| Payload generator (C28/C29/C31) | dimensionless held-out conditional-necessity ratios H7 `0.3481`, H12 `0.1597`, H10 `0.1157`; conditional path `{H7,H12,H10} -> MLP0 -> MLP1 -> MLP2` | `a7_source_localization.json`; `a8_generator_path.json`; `a10_generator_edges.json` |
| Donor-side state (C16) | rank-8 `R=0.9179 [0.8999,0.9357]`; mean/max over 200 matched random rank-8 controls `0.0155/0.0530`, `p=1/201` | `donor_anchor.json` |
| Write and route (C17) | L3H1 `R=0.8288`; `U_info R=0.7697`; routed fraction `0.9287`; donor±5 fraction `0.9620`; conditional, and fractions are not additive across serial components | `donor_forward.json` |
| Delivery (C18) | L5H11 `0.1251 [0.1044,0.1433]`; L4H2 `0.1222 [0.1047,0.1421]`; L4H0 `0.0812 [0.0646,0.0994]` | `donor_heads.json` |
| Disjoint-gene replication (C26) | rank-1 overlap `0.970`; held-out two-direction mean `0.8747` | `a5_uinfo_replication.json` |
| Cross-example acceptor transfer (C38, verification tier) | matched `0.9308`; decoy `-0.0490`; scramble `0.3394`; matched minus mismatched `0.1582 [-0.1102,0.3309]` | `b3_cross_example_transduction_n512.json` |
| Donor generic axis (C41, verification tier) | mean `tau_REAL=0.6585`; mean `Delta_spec=-0.3197`; 8 source draws, 1,012 target examples/551 target genes | `g1_cross_example_transduction.json` |
| Injectability vs generation (C27) | `D(C\\embed)=0.9864 [0.9815,0.9906]`; `F(C)=0.005394 [0.003150,0.007794]` | `a6_circuit_faithfulness.json`; `a6_reconcile.json` |
| Faithfulness boundary (C33) | interpolated 90% `64.2`; sampled crossing `72/103`; `F(64)=0.8992`, `F(72)=0.9375` | `a12_faithfulness_size.json`; conformance correction |
| Period-3 deletion and mediation (C8/C9) | coupling `0.946 -> 0.026` (R53); matched band `0.970`; L2H15/L3H1 restore `0.363/0.348`; null max `0.042` | `results/phase-lattice/carrier_surgery.json`; `carrier_dose.json`; `carrier_mediation.json` |
| P3 span and readout (C34/C35; C34's confirm gate missed, R24) | site-local restoration recovers only about 2--4%; confirm-stage L3H1-attention recovery `+0.284`, value recovery `-0.005` | `results/p3_carrier_locus.json`; `results/p3_av_decomposition.json` |
| Phase-aligned use (C36) | primary `Psi_C=+0.141 [0.110,0.172]`, n=943; learned minus control `+0.136 [0.051,0.222]` on a separately frozen n=150 direction-control subset, 19 controls; CDS-minus-non-CDS-exon `+0.006 [-0.080,+0.093]`, inconclusive | `p3_phase_surgery_confirmation_v1/confirmation_analysis.json` |
| Corpus association (C39) | median P3 power: 72-species `435x`, human-only `6.2x` | `results/origin-siblings/canonical/sibling_pspectrum.json` |
| Separate ≈10.63 carrier (C11--C13) | eight controlled post-softmax readers; donor-coupling deletion `-0.0333 [-0.0606,-0.0053]`, underpowered; general OV value-main below the P3 router-null scale | `results/phase-lattice/postsoftmax.json`; `carrier_p10_behaviour_confirm.json`; `carrier_ov_av.json`; `carrier_ov_general.json` |
| Boundary-register behavior (C43) | `Psi=+0.3901 [0.3553,0.4252]`, 384 blocks; 99 controls; `p=0.01` floor | `results/splicebert_exon_edge_compatibility/crossed_direct_edge_replication_audit.json` |
| Reverse behavior (AC1) | `+0.9763 [0.9060,1.0526]`, 974 pairs/645 genes; controls `-0.0009/+0.0147` | `results/splicebert_acceptor_coupling/r1b_reverse_coupling.json` |
| Reverse components (AC5/AC6) | L1H12 specific `+0.2447`; L4H11/H9/H7 `+0.0934/+0.0905/+0.0788`; Holm `p=0.00175` | AC compact attestation and R3c decision audit |
| DNABERT representation (DBP3, DBP4) | BA `0.8102`; AUC `0.8919`; layer-2 lead `0.0019`, single seed, not localization (DBP4) | `results/dnabert_promoter/frozen_claim_tests/` artifacts |
| DNABERT core and boundary (DBP6--DBP8) | five-node necessity `0.630`, delivery `0.882`, standalone `0.194`; matched/same-class `0.879/0.843`, `p=0.252` | DBP6--DBP8 controlling artifacts; Figure 9 sources |

## 5. Paper-safe narrative

Order the main results as behavior, nomination and necessity, address, payload, write, delivery, and
the faithfulness boundary, ending on "a structured, top-ranked causal core within distributed
computation." Then give C36 and, separately, C43 (C44 does not causally connect them), and last the
bounded extensions: the reciprocal endpoint, with no four-role mechanism or symmetry result, and
DNABERT-2, which reproduces the shape of the faithfulness boundary, not the SpliceBERT circuit. Keep the
negative, inconclusive and stopped results: C24/C30, C27/C33, C38, C41, C36's CDS-minus-non-CDS-exon
contrast, C44, AC7, AC8, AC9, ED1, and DBP7/DBP8. AC7, C38 and the C36 contrast are inconclusive, AC8 is
health-ineligible rather than null, and ED1 licenses no interpretation in either direction.

Do not write any of these:

- "first systematic application" or "first causally supported account" without a literature review;
- a sparse, minimal, autonomous, or complete circuit, or proof of irreducible distribution;
- address `=` period-3, period-3 `=` coding frame, or period-3 `=` C43 register mechanism;
- the payload as splice-site strength, donor identity, U2AF/branch-point state, or any biological semantic;
- biological splicing, molecular mechanism, or wet-lab validation from model behavior alone;
- reciprocal symmetry, the donor mechanism running backward, or masked-acceptor chr8/9 confirmation;
- DNABERT sharing the SpliceBERT circuit or providing an example-specific transferable mechanism;
- multispecies origin, instead of comparative training-condition association;
- all studies preregistered, all gates passed, or all artifacts clean-checkout reproducible.

## 6. Figure audit

The active `figures/` directory holds exactly the four raster assets the workshop manuscript uses: the
conceptual workflow, the SpliceBERT headline, the DNABERT-2 workflow-transfer composite, and the
positional/faithfulness composite, each with a repository-relative, hash-checked provenance sidecar.
Historical and supplementary figure suites are only in Git history.

The captions keep the required boundaries. The workflow is conceptual; the period-3 and
register-preference findings are separate; the 90% faithfulness statement distinguishes the 64.2-node
interpolation from the first sampled crossing at 72/103; and the DNABERT-2 panel claims workflow transfer,
not mechanism transfer. The C43 panel stays behavior-level and states the fixed 120-nt chimeric span, the
384 blocks, the add-one resolution floor, and the stop-load rival.

## 7. The old verification file

The deleted `RESULTS_VERIFICATION.md` is recoverable from Git history at SHA-256
`3300fc6af0e8b00ff9abca8116ced018e25f2fb4538342ce5443e6442fa0ae15`. Its historical `52/52` claim is not
retained: two C8 rows compare different cohorts or endpoints while marking them exact, and the C14
biology wavelength cites `carrier_subbin.json` instead of the dyad artifacts. Its table is provenance,
not current verification. Since then, A11/C32 superseded A4, though A11's frozen classifier,
post-hoc result wording, and manifest remain inconsistent. A6 has run, and C27 and C33 now define the
injectability/faithfulness boundary. The live ledger extends through C46 plus IC1/ED1, with permanent
corrections R1--R59. "Information-specific circuit" is overstated after C38/C41, and `U_info` does not
support paired-intron semantics or donor identity. The 120-pass/33-skip test baseline is obsolete; the
current one is 1,023/22. Acceptor AC9 is terminal under its old lifecycle, so
address-payload-write-delivery and symmetry work needs a new successor.

## 8. Live-ledger coverage index

The audit gave a disposition to 123 claim rows in seven active ledgers. The donor ledger now has 47
rows; C45 and C46 were added on 2026-10-26, after the audit, and are not covered. Labels give each
row's place in the paper; the linked ledger holds its wording and scope.

| Ledger | Rows audited | IDs by disposition |
|---|---|---|
| [Donor partner selection](../studies/splicebert_partner_selection/CLAIMS.md) | 45 of 47 | Main text (C19's phase-flat arm is descriptive, not equivalence; C42 is behavior-level and acceptor-operator only, A0–A12 remain CC-only; C5/C6 marginal effects are not additive shares; C20 is geometry, not partner identity, R20): C1, C2, C42, C19, C3, C4, C5, C6, C20, C21, C22, C32, C25, C24, C28, C29, C31, C23, C16, C17, C18, C26, C27, C33. Limitations or supplement: C30, IC1. Discussion or supplement, with no origin, universality, or species-diversity cause: C39, C40. P3 family: C8, C9, C10 (bounded matched-null power stop), C34 (confirm gate-qualified, M1 failed, R24), C35, C36 (the clean phase-aligned-use headline). ≈10.63 carrier: C11 (attention established), C12 (bounded; `U_info` write open, R31/R36). Human-only ≈10.2 proposal closed at attention: C13. Mouse ES-cell/mm9 nucleosomal comparator and powered wavelength refutation, not a human measurement or an explanation of ≈10.63: C14, C15. Verification tier: C41, C38. Closed: C7, ED1. Bounded behavior and separate instrument stop: C43, C44. Not audited: C45, C46. |
| [Acceptor coupling](../studies/splicebert_acceptor_coupling/CLAIMS.md) | 11 of 11 | Bounded: AC1, AC5, AC6, AC10, AC11. Operator screen: AC2. Retracted: AC3. Scan with a retracted/instrument-unresolved partition: AC4. Inconclusive: AC7. Health-ineligible: AC8. Terminal schema stop: AC9. |
| [DNABERT-2 promoter](../studies/dnabert_promoter/CLAIMS.md) | 10 of 10 | Main bounded comparison: DBP6, DBP7. Class-level transfer: DBP8. Provenance: DBP1, DBP2. Representation: DBP3, DBP4. Ranking only: DBP5. Enrichment and TATA null: DBP9, DBP10. |
| [GTEx fine-tune shift](../studies/splicebert_gtex_finetune_shift/CLAIMS.md) | 34 of 34; conformance self-reported | Discussion or supplement: GFS8, GFS9, GFS10, GFS11, GFS12, GFS13, GFS14, GFS15, GFS16, GFS17, GFS18. Bounded nulls: GFS24, GFS25. Exploratory: GFS19, GFS20. Provenance and preflight: GFS1, GFS2. Retracted coordinate instrument and corrected preflight: GFS6, GFS7. Failed route: GFS3, GFS4, GFS5. Spectral retention: GFS23. Representation instrument and replication: GFS28, GFS29. Failed, stopped, or unresolved instruments: GFS21, GFS22, GFS26, GFS27, GFS30, GFS31, GFS32, GFS33, GFS34. |
| [Head-context atlas](../studies/splicebert_head_context_atlas/CLAIMS.md) | 21 of 21 | Discussion or supplement, disclosing its dirty-run provenance: HCA19. Do not headline until its off-repo artifacts, produced after a 100-fold numerical-gate relaxation, are accessible and independently audited: HCA20. Positive control: HCA9. Discovery classification with an empty context shortlist: HCA18. Context signals closed as underpowered leads (design-limit prose only): HCA23. Cohort feasibility: HCA1. Infrastructure: HCA11, HCA12, HCA13, HCA14, HCA15, HCA16, HCA17. Visualization only: HCA21, HCA22. Closed competition branch: HCA2, HCA3, HCA4, HCA6, HCA7, HCA10. |
| [Positional boundary coherence](../studies/splicebert_positional_boundary_coherence/CLAIMS.md) | 1 of 1 | Model-free capacity stop: PBC1. |
| [Positional proxy modifiability](../studies/splicebert_positional_proxy_modifiability/CLAIMS.md) | 1 of 1 | Model-free max-flow/capacity result only; its archived exploratory pilots cannot support a training conclusion: PPM1. |

The prospective `splicebert_acceptor_context` study and the template study have no live result rows.
Archived exon-edge prose is superseded except for the consolidated C43/C44. Excluded legacy trees,
unledgered notebooks, the PPM pilots, and the notebook-only fine-tuned DNABERT-2 extension are recoverable
at `archive/unsupported-exploratory-2026-08-22` and are not part of the active evidence tree.

## 9. Scope of this audit

This internal audit checks artifact arithmetic, provenance, cohort definitions, and claim wording. It
includes no external literature review, biological validation, or model rerun. The fixes above must be
made in the manuscript source and re-audited once its line numbers, captions, citations, and asset
paths settle.
