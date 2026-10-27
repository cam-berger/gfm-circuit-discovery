# Results and manuscript verification

> **Current audit.** This document replaces the historical `RESULTS_VERIFICATION.md` snapshot, which
> covered only C1--C26 as of 2026-07-16. It audits the evidence state inherited from commit
> `409e154` and the supplied 411-line manuscript attachment with
> SHA-256 `0439a1c73f97679e151083ace0fbf4dc85e089d3e71bf9e98967240f18a6b8cf`.
>
> **Authority order.** Current claim and correction ledgers plus
> [`docs/PREREG_CONFORMANCE.md`](docs/PREREG_CONFORMANCE.md) control over immutable artifacts and
> manifests, which control over results indices, captions, and narrative summaries. Line numbers below
> refer to the fingerprinted attachment and will change when the manuscript is edited.

**Audit date:** 2026-08-22

**Primary checkpoint:** SpliceBERT.1024nt, SHA-256 `2ad91428…0149d`

**Primary study:** [`studies/splicebert_partner_selection/CLAIMS.md`](studies/splicebert_partner_selection/CLAIMS.md)
**Correction history:** [`studies/splicebert_partner_selection/CORRECTIONS.md`](studies/splicebert_partner_selection/CORRECTIONS.md)

## 1. Executive verdict

The paper's central behavioral number and most intervention values reproduce their controlling
machine-readable artifacts. The strongest publication-safe synthesis is:

> SpliceBERT contains a structured, top-ranked causal core for paired-acceptor-dependent masked-donor
> prediction. Positional geometry supplies an address, acceptor-local values supply a distinct payload,
> L3H1 combines them into a causally useful donor-side state, and later heads deliver that state. The named
> core is highly injectable but not a faithful autonomous generator; held-out static-ranking results show
> a gradual, distributed faithfulness boundary rather than a compact circuit.

The manuscript is **not fact-check complete as written**. Before submission it needs the corrections in
Section 4, especially these:

1. `\ell_b` is a raw pre-softmax logit, not a log-probability.
2. The 96-head development scan, the sealed two-head test, and the 400 random two-head null
   interventions are three different objects. The number 400 in the null test is not an example count.
3. The payload paragraph attaches a six-head recovery number to a displayed three-head set.
4. The transfer paragraph reverses the sign: matched exceeds margin-mismatched by `+0.158`, not the
   reverse, and the interval crosses zero.
5. The preregistered C33 interpolation reaches 90% faithfulness at **64.2 nodes**; **72** is merely the
   first evaluated grid point above 90%. Neither proves that no smaller alternative circuit exists.
6. The period-3 carrier, L3H1's predominantly positional address, and C43's annotated boundary-register
   preference are separate results. They have not been causally connected.
7. The reverse SpliceBERT direction now has bounded component-level findings (AC5/AC6), although no
   address--payload--write--delivery mechanism, symmetry result, or chr8/9 acceptor endpoint exists.
8. DNABERT-2 is a single-seed, non-preregistered, uncalibrated, provenance-limited comparison. DBP8 is
   already nonsignificant before correction (`p=0.252`), and the task effect is not “coupling.”
9. “First,” “systematic,” and literature-wide novelty claims require a separate current literature
   review. Model-internal results do not establish biological function or a uniquely motivated wet-lab
   experiment.

## 2. Repository verification

### 2.1 Integrity checks run for this audit

| Check | Result |
|---|---|
| `uv run python scripts/verify_manifest.py` | **PASS** after cleanup: exact coverage of all **1,108** `results/MANIFEST.csv` rows |
| Manifest status census | 651 canonical, 256 development, 180 superseded, 21 infrastructure |
| Manifest polarity census | 534 positive, 23 null, 5 refutation, 111 bounded, 435 not applicable |
| `uv run --extra splicebert --extra biology --extra dev pytest -q` | **PASS**: **1,023 passed, 22 skipped** in 221.40 s |
| Focused manifest, ledger, hygiene, figure-provenance, and DNABERT checks | **PASS**: **93 passed** |

The skips are declared data/optional-environment skips, not test failures. No model intervention was
rerun merely to restate a recorded result. Headline arithmetic was taken from, or recomputed directly
from, the controlling machine-readable artifacts. In particular, the C33 linear interpolation was
recomputed from the committed held-out curve rather than copied from its grid-snapped summary.

`verify_manifest.py` verifies registry/path/schema closure. It does **not** establish scientific truth,
verify every output hash, fill missing producers, or make an artifact clean-checkout reproducible.
Passing memorial tests are regression locks on recorded values, not independent reruns of the model.

### 2.2 Preregistration and provenance boundary

Do not describe the repository as uniformly preregistered or say that all gates passed. The independent
conformance audit covers 25 SpliceBERT-arc families and records **188 deviations, 157 medium-or-worse**;
it states that none overturns a scientific result. Those deviations still constrain the evidential tier,
provenance language, and strength of a claim. The unmanifested legacy `confirmation.json` was archived
and removed from the active figure path in this cleanup. Remaining paper-facing gaps include
post-hoc/unmanifested parts of C27, the grid-snap deviation in C33, and the C36 analysis code-pin and
sensitivity-reporting omissions.

## 3. Data splits and example counts

### 3.1 Canonical donor-coupling population

The active donor assay uses GENCODE v50 **basic** annotation and hg38. The pinned input hashes are:

- GTF: `5b9ad9aa71f6ab5d55123c51f74723644a895bc0cbaa9cee8fbc11bb9dec0bf7`
- FASTA: `5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51`

`full-build` counts are deduplicated, sequence-valid 1,024-nt internal-exon windows after the active
builder's length, motif, boundary, and ACGT checks. `eligible` additionally requires a canonical `GT`
donor, its downstream intron-mate `AG` acceptor inside the window, and a 60--150-nt donor-to-acceptor
gap. The assay population is the **22,694 eligible rows**, not all 232,867 built windows.

| Chromosome | Split | Full-build windows | Assay-eligible sequences | Eligible genes |
|---|---:|---:|---:|---:|
| chr1 | train | 24,029 | 2,035 | 678 |
| chr2 | train | 17,650 | 1,334 | 420 |
| chr3 | train | 14,169 | 1,303 | 360 |
| chr4 | train | 8,835 | 412 | 213 |
| chr5 | train | 10,007 | 602 | 235 |
| chr6 | train | 11,289 | 1,094 | 314 |
| chr7 | train | 11,065 | 846 | 316 |
| chr8 | sealed test | 7,878 | 783 | 205 |
| chr9 | sealed test | 9,377 | 785 | 249 |
| chr10 | train | 9,423 | 531 | 239 |
| chr11 | validation | 13,273 | 1,650 | 446 |
| chr12 | train | 13,031 | 1,172 | 396 |
| chr13 | train | 3,840 | 185 | 94 |
| chr14 | train | 7,163 | 657 | 209 |
| chr15 | train | 8,142 | 635 | 213 |
| chr16 | validation | 10,202 | 1,682 | 394 |
| chr17 | train | 13,780 | 2,061 | 509 |
| chr18 | train | 3,424 | 146 | 68 |
| chr19 | validation | 14,133 | 2,598 | 676 |
| chr20 | train | 5,879 | 604 | 175 |
| chr21 | train | 2,357 | 100 | 49 |
| chr22 | train | 5,268 | 614 | 167 |
| chrM | train | 0 | 0 | 0 |
| chrX | train | 8,034 | 794 | 244 |
| chrY | train | 619 | 71 | 34 |
| **Total train** |  | **178,004** | **15,196** | **4,933** |
| **Total validation** |  | **37,608** | **5,930** | **1,516** |
| **Total sealed test** |  | **17,255** | **1,568** | **454** |
| **All splits** |  | **232,867** | **22,694** | — |

The eligibility funnel is:

| Split | Full-build | Canonical GT | Partner in window | Gap 60--150 (eligible) |
|---|---:|---:|---:|---:|
| Train | 178,004 | 175,475 | 52,606 | 15,196 |
| Validation | 37,608 | 37,032 | 15,885 | 5,930 |
| Test | 17,255 | 17,002 | 4,895 | 1,568 |

The obsolete v1 config in `configs/splicebert_partner_selection/base.json` assigns different
chromosomes and is not controlling. The active split is validation `{chr11, chr16, chr19}`, sealed test
`{chr8, chr9}`, and all listed remaining chromosomes as train.

### 3.2 Frozen draws and the meaning of “400”

| Cohort or null family | Rows / interventions | Genes | Chromosome breakdown | Use |
|---|---:|---:|---|---|
| Unconditional validation behavior | 400 rows | 334 | chr11 98/83 genes; chr16 94/82; chr19 208/169 | Validation behavior |
| Unconditional sealed test behavior | 400 rows | 211 | chr8 181/94 genes; chr9 219/117 | C1/C2 behavior |
| Conditional validation mechanism | 150 rows | 140 | chr11 33; chr16 48; chr19 69 | Mechanism development |
| Conditional sealed test mechanism | 150 rows | 100 | chr8 77; chr9 73 | C3 sealed intervention cohort; health/null use its first 100 rows |
| Corrected exhaustive head scan | 160 rows | 146 | chr11 48; chr16 39; chr19 73 | Development candidate ranking across 96 heads |
| Matched two-head null | **400 head-pair interventions** | first 100 mechanism rows | not 400 new sequences | Add-one `p=1/401` for C3 |

The per-chromosome mechanism counts above are exact deterministic re-derivations that reproduce the
stored aggregate means; most compact artifacts do not serialize row IDs. They should be labeled
“re-derived,” not “stored in the sealed artifact.” Do not rescore or reopen chr8/9 merely to enrich old
metadata.

### 3.3 Other active study split schemes

These cohorts are separate assays and must not be pooled or described as independent replications merely
because they use different rows.

| Study | Split/cohort record | Counts and status |
|---|---|---|
| Acceptor coupling | GENCODE v50 comprehensive; val chr11/16/19; intended target chr8/9 | AC1 draw 0: 974 pairs/645 genes. AC5--AC8 target: 500 rows/304 genes on validation chromosomes. AC9 preflight: 400/321, then schema stop. **No chr8/9 acceptor target was scored.** |
| Head-context atlas | GENCODE v50 comprehensive | Legacy usable train/val/test: 32,054/11,275/3,014 rows and 4,864/1,567/444 usable genes. Fresh gene-hash discovery: 11,807 packets/5,089 genes; internal validation: 4,019 packets/1,486 genes. Sealed chr8/9 confirmation not run. |
| GTEx fine-tune shift | GENCODE v50 comprehensive assay cohorts | Fixed validation cohort 400/334 genes, reused across five seeds (not 2,000 rows): chr11 98/83, chr16 94/82, chr19 208/169. Fresh train discovery/confirmation families use 128/200 rows; chr8/9 remain sealed. Fine-tuning training-set identities are not present in this repo. |
| C43 register assay | Designed fixed-span chimeras | Fresh replication 384 gene-disjoint blocks, 3,456 lattice cells, 2,304 native rows. C44 ran a 114-block development preflight; its 384-block target was never opened. |
| Positional boundary coherence | Model-free capacity stop | Required 1,104 blocks; donor/acceptor gene ceilings 809/854; zero blocks selected and no model loaded. |
| Positional proxy modifiability | Model-free capacity stop | Required 718 blocks; exact ceiling 726 but usable 25%-margin ceiling 544; no confirmatory cohort selected. |
| DNABERT-2 promoter | GUE, not GENCODE | Raw train/dev/test 47,356/5,920/5,920; after exact duplicate/conflict/cross-split cleaning 47,149/5,856/5,865. No chromosome or gene metadata exists, so a chr-level breakdown cannot be recovered. |

Legacy `acceptor-interpretability/`, `ntv3-interpretability/`, and top-level `dnabert/` cohorts were
exploratory or superseded and were removed from the active tree. They remain recoverable at
`archive/unsupported-exploratory-2026-08-22` and are not paper evidence.

## 4. Manuscript claim audit

### 4.1 Corrections required before submission

| Attachment line(s) | Current text/problem | Required disposition |
|---|---|---|
| 31, 37, 51, 55, 65--67, 247, 251 | Says the mechanism produces biological meaning or wet-lab hypotheses unavailable from sequence statistics. | Recast as **model-internal hypothesis generation**. No external assay identifies payload semantics, splicing activity, U2AF/branch-point biology, or a uniquely implied experiment. |
| 49, 59, 78 | “First,” “has not yet been applied systematically,” or literature-wide rarity. | Remove or hold pending a separate current literature review. The repository cannot verify novelty. |
| 55 | Equates the positional address with “a phase-aligned period-3 register.” | Separate them. The live C32 ledger uses “predominantly relative-offset,” but the frozen A11 classifier returned MIXED and its result/manifest disagree; report substantial relative-offset retention with absolute/edge modulation. C36 establishes present-day phase-aligned use of a learned P3 direction. C43 is a third, behavior-level construct result. |
| 59, 245 | “No sparse faithful set exists” / “irreducibly distributed.” | Use: **the tested held-out static-ranking curve has no sparse knee and supports a structured top-ranked core within distributed computation**. It is an upper bound, not an impossibility proof. |
| 83 | Says five causal hypotheses but the presented structure is behavior plus four roles, followed by additional specificity and positional studies. | Enumerate the five explicitly or change to “a sequence of causal hypotheses.” |
| 35, 241--243 | Calls address, payload, write, and delivery four “components” and says the whole account has a disjoint-gene replication. | Call them four **causal roles**, each implemented by multiple components. C26 independently reproduces the donor-side `U_info` state, not the entire four-role account. |
| 87, 318--321 | Calls the donor target a “masked dinucleotide” and defines `\ell_b` as log-probability. | Only the donor `G` position is masked. Define `\ell_b` as the model's raw pre-softmax logit at that position. |
| 92 | “All splits are gene-disjoint.” | Scope this to the analysis cohorts/split construction. It is not a claim that SpliceBERT pretraining excluded those genes. |
| 103--105 | Blends a 96-head development scan with the sealed paired intervention and its 400-pair null. | State the stages separately. The corrected scan used 160 validation rows; C3 used 150 conditional sealed-test rows; health and the 400 random-pair interventions use the first 100. `99.2%` is a **retention ratio** (`0.51065/0.51457`), not 99.2 percentage-point accuracy, and `1/401` is an add-one empirical null p-value rather than a scan-wide correction. L1H12 is also a validated contributor, so “two dominant” must not imply completeness. |
| 108 | Calls attention “where it reads” and the value vector “what it writes.” | In this factorial, attention is the address and acceptor-local values are the payload; the donor-side post-L3H1 state is the write. |
| 111 | Uses A1's `2.718` to infer relative position. | `2.718` establishes positional over local-token influence. A11 retains a large preference under common translations but modulates it by as much as 23.7% in a nonmonotonic, edge-sensitive pattern; the frozen absolute-vs-relative decision is MIXED. |
| 114, 151--154 | Names three L0 heads, then calls `+0.136` “the complete set” / “six heads,” with the table caption implying raw logit units. | `0.348/0.160/0.116` are **dimensionless conditional-necessity ratios**, not donor-margin logits. `R=0.136` belongs to the held-out six-head set `{L0H7,H8,H10,H11,H12,H13}`; adding MLP0 gives `R=0.356`. H7/H12/H10 are the three largest, not a demonstrated complete/minimal set, because the registered minimal-set sufficiency arm was not run. |
| 117 | Says rank 8 recovers 92% versus “a random rank-8 subspace.” | Say **versus a maximum of 5.30% across the matched random rank-8 controls**. Rank 8 is tested and useful, not uniquely selected or a saturation point. |
| 117, 159, 342 | `+0.875` is presented near `[0.853,0.901]`. | The point estimate is the A→B/B→A mean; the stored interval applies only to A→B. Never decorate the two-direction mean with that one-direction interval. |
| 195 | “A margin-mismatched source exceeds the matched source by 0.158.” | Reverse the direction: **matched exceeds margin-mismatched by `+0.158`, CI `[-0.110,0.331]`; unresolved**. C38 is verification-tier and does not establish example identity. |
| 197 | Calls the final-residual direction “common to all donors” and describes 551 “targets.” | The cohort has **1,012 target examples from 551 source-disjoint target genes**. Bound the result to C41's additive final-residual assay: it is consistent with a gene-generic readout axis in the tested linear regime. Six of eight draws were generic and two instrument-limited; this does not cover earlier addressing, routing, payload, or nonlinear specificity. |
| 199 | Describes one “nine-node core” as both injectable and isolated. | These are different sets/interventions. Post-hoc `D(C\\embed)=0.986` supplies clean outputs to nine non-embedding spine nodes while retaining the corrupt embedding; preregistered `F(C)=0.00539` isolates the ten-node set including embedding. The reconcile artifact is unmanifested, and `N(C\\embed)=0.9888` has no CI/null. |
| 199 | Points to conceptual workflow Figure 1B for quantitative faithfulness. | Point to the actual C27/C33 injectability/faithfulness panel or supplementary curve; the workflow panel is not quantitative evidence. |
| 35, 201, 245 | “90% requires 64--72 of 103” and uses it as minimality proof. | Report both definitions: preregistered linear interpolation `64.2`; first sampled grid crossing `72/103`. Static ranking is an upper-bound construction and the curve is non-monotone at high `k`. |
| 210, 391--392 | Presents the early P3 full-band deletion as a clean causal lesion. | Keep it as supportive dose/control evidence: the artifact's registered health field is false, the add-one control result is at the `1/22` floor, seven of 21 control bands overlap the second carrier, and the runner's “CDS” label was not annotation-derived. C36 is the cleaner current-use result. |
| 208--214, 391--392 | Risks reading P3 as biological reading frame or training origin. | C8/C36 establish model use; C36's CDS-minus-non-CDS contrast is inconclusive. C39 is comparative corpus association and does not isolate species diversity, corpus size, composition, sampling, or optimization. |
| 225 | Says the C44 experiment “was not run.” | A **114-block development preflight ran** and failed four authorization gates; the **384-block target was not run**. This is an instrument stop, not a positive, null, or inverted P3-to-register result. |
| 232 | “No internal mechanism has been traced.” | Say no reverse four-role mechanism has been traced. AC5 supports L1H12 mean-ablation specificity; AC6 supports L4H11/H9/H7 same-example noising. These are component effects, not address, write, delivery, mediation, symmetry, or chr8/9 evidence. |
| 234 | Says DBP8 fails only after multiple-comparison correction and calls the five-node effect analogous without provenance caveats. | Matched `0.879` vs same-class `0.843`, difference `0.036`, is nonsignificant already at raw `p=0.252`. State one seed, no preregistration, no calibration pair, and incomplete manifest/producer provenance. |
| 397--398 | Calls DNABERT's promoter effect “normalized coupling.” | Use **normalized promoter-evidence corruption effect**. It is not donor--acceptor coupling. |
| 403--407 | “Only within coding-sequence blocks” can be read as coding specificity. | Say phase-aligned shifts preserve more coupling in CDS blocks **relative to matched upstream-intron blocks**. Retain the explicit inconclusive CDS-minus-non-CDS-exon contrast. |

### 4.2 Verified quantitative statements and licensed wording

| Role/result | Artifact-backed quantity | Licensed statement and first boundary | Controlling artifact |
|---|---|---|---|
| Paired-site behavior (C1/C2) | `+0.872 [0.771,0.969]`, n=400; controls `+0.007/+0.005` | Paired-acceptor corruption selectively changes the native masked-donor margin on sealed chr8/9 canonical short introns. Not a splicing assay. | `results/splicebert_coupling/sealed_test.json` |
| Corruption robustness (C42) | n=1,000/648 genes on val; every admissible partner-acceptor alt ratio lies in the frozen `[0.75,1.33]` equivalence band (`0.861--0.982`) | The behavior is not an artifact of choosing `AG->CC`, but this is acceptor-operator behavior evidence only; it does not transfer to A0--A12 mechanisms or to donor corruption. | `results/splicebert_coupling/ar1_corruption_operator_sensitivity.json` |
| Dominant heads (C3) | pair loss `0.8161`; health retained `0.9924`; 400 matched pairs, `p=1/401` | `{L3H1,L2H15}` is a dominant necessary subcircuit, not complete or minimal. | `sealed_test.json` |
| Early conjunction (C4) | key `0.0320`, value `0.1005`, joint `0.6633`, interaction `0.5307 [0.4774,0.5833]` | L3H1's sealed acceptor read is conjunctive under the tested patch. Do not equate this older K/V factorial with the later address/value factorial. | `sealed_test.json` |
| Address--payload factorial (C23) | address `0.0104`, payload `0.0514`, joint `0.3645`; interaction `0.3028` | In the non-forced reconstruction, L3H1's contribution requires positional attention and acceptor-local values jointly. | `a2_address_payload.json` |
| Address (C22/C32/C25) | position `2.7176 [2.2229,3.1535]`; content `0.3973 [0.1808,0.6198]`; L2H15 address/value rescue `0.4536/0.0194` | L3H1 selection is predominantly positional; L2H15 amplifies that channel. Common translations retain a large preference but show up to 23.7% absolute/edge modulation; A11's frozen classifier is MIXED despite the stronger current-ledger wording. | `a1_token_position.json`; `a11_address_geometry.json`; `a3_l2h15_address_payload.json` |
| Payload localization (C24) | four-token values `+0.3820 [0.3490,0.4153]`; `U_info` `0.0604`, random range `0.0275--0.0699`, `p=0.2195` | Payload is acceptor-local and not concentrated in donor-derived rank-8 `U_info`; intrinsic payload dimension and semantics remain unresolved. | `a2b_payload_localization.json` |
| Payload generator (C28/C29/C31) | dimensionless conditional-necessity ratios: H7 `0.3481`, H12 `0.1597`, H10 `0.1157` on held-out genes | Six L0 heads pass the held-out positive-interval rule; H7/H12/H10 are the largest. Results are consistent with conditional `{H7,H12,H10} -> MLP0 -> MLP1 -> MLP2` wiring, but the unrun minimal-set sufficiency arm precludes a complete/minimal-set claim. | `a7_source_localization.json`; `a8_generator_path.json`; `a10_generator_edges.json` |
| Donor-side state (C16) | rank-8 `R=0.9179 [0.8999,0.9357]`; mean/max across 200 matched random rank-8 controls `0.0155/0.0530`, `p=1/201` | A stable low-rank final-residual direction mediates the donor readout. It is a delivered state, not payload semantics. | `donor_anchor.json` |
| Write/route (C17) | L3H1 `R=0.8288`; `U_info R=0.7697`; routed fraction `0.9287`; donor±5 fraction `0.9620` | L3H1 writes a useful state whose tested effect is conditionally mediated through donor-local `U_info`. Fractions across serial components are not additive shares. | `donor_forward.json` |
| Delivery (C18) | L5H11 `0.1251 [0.1044,0.1433]`; L4H2 `0.1222 [0.1047,0.1421]`; L4H0 `0.0812 [0.0646,0.0994]` | These late heads are selective OV/value transporters under the tested intervention; QK damage is approximately zero. | `donor_heads.json` |
| Disjoint-gene state replication (C26) | rank-1 overlap `0.970`; held-out two-direction mean `0.8747` | Independently derived donor-side directions generalize across disjoint genes. Stored `[0.853,0.901]` is A→B only. | `a5_uinfo_replication.json` |
| Cross-example acceptor transfer (C38) | matched `0.9308`; decoy `-0.0490`; scramble `0.3394`; matched-minus-mismatched `0.1582 [-0.1102,0.3309]` | Position-specific causal transfer is supported; example identity is unresolved. Verification tier. | `b3_cross_example_transduction_n512.json` |
| Donor generic axis (C41) | mean `tau_REAL=0.6585`; mean `Delta_spec=-0.3197`; 8 source draws, 1,012 target examples/551 target genes | The additive final-residual state behaves as a gene-generic readout axis in the tested linear regime. Verification tier; brackets across draws are ranges, not CIs. | `g1_cross_example_transduction.json` |
| Injectability vs generation (C27) | `D(C\\embed)=0.9864 [0.9815,0.9906]`; `F(C)=0.005394 [0.003150,0.007794]` | The tested delivery spine is highly injectable but not a faithful autonomous generator. | `a6_circuit_faithfulness.json`; `a6_reconcile.json` |
| Faithfulness boundary (C33) | interpolated 90% `64.2`; sampled crossing `72/103`; `F(64)=0.8992`, `F(72)=0.9375` | The frozen static ranking has a gradual curve with no sparse knee. It is an upper bound on minimal size, not a proof of nonexistence. | `a12_faithfulness_size.json`; conformance correction |
| Period-3 deletion/mediation (C8/C9) | coupling `0.946 -> 0.027`; matched band `0.970`; L2H15/L3H1 restore `0.363/0.348`; null max `0.042` | The learned P3 position component supports the model behavior, but the full-lesion health gate fails and its control-band p-value is the `1/22` floor. Treat as supportive dose/control evidence; no biological-origin or reading-frame claim. | `results/phase-lattice/carrier_surgery.json`; `carrier_dose.json`; `carrier_mediation.json` |
| P3 span/readout refinement (C34/C35) | site-local restoration recovers only about 2--4%; confirm L3H1-attention recovery `+0.284`, value recovery `-0.005` | P3 contribution is span-distributed and reaches coupling mainly through L3H1's attention pattern in the tested decomposition. C34's confirmation precision gate missed its threshold, and neither result proves that P3 is the entire address. | `results/p3_carrier_locus.json`; `results/p3_av_decomposition.json` |
| Phase-aligned use (C36) | primary `Psi_C=+0.141 [0.110,0.172]`, n=943; learned-minus-control `+0.136 [0.051,0.222]` on a separately frozen n=150 direction-control subset with 19 controls | Present-day checkpoint use of the learned P3 direction is phase-aligned in the tested CDS-vs-upstream-intron contrast. Coding specificity and origin unresolved. | `p3_phase_surgery_confirmation_v1/confirmation_analysis.json` |
| Corpus association (C39) | P3 power 72-species `435x` vs human-only `6.2x` median | Comparative association with the multi-species training condition; species diversity is not isolated. | `results/origin-siblings/canonical/sibling_pspectrum.json` |
| Separate ≈10.63 carrier (C11--C13) | eight controlled post-softmax readers; donor-coupling deletion `-0.0333 [-0.0606,-0.0053]`, verdict underpowered; general OV value-main below the P3 router-null scale | Routing/attention is supported. The tested donor behavior is a small suppressive bound, general OV is bounded, and the preregistered `U_info` write was never computed/not evaluable under the scoped L0-reader requirement. Function and biology remain open; the human-only reader proposal fails at attention. | `results/phase-lattice/postsoftmax.json`; `carrier_p10_behaviour_confirm.json`; `carrier_ov_av.json`; `carrier_ov_general.json` |
| Boundary-register behavior (C43) | `Psi=+0.3901 [0.3553,0.4252]`, 384 blocks; 99 controls; `p=0.01` floor | Fixed-span chimeric constructs preferentially favor annotated phase matches. Behavior-level only; mismatch equivalence fails for one direction, stop avoidance remains a rival. | `results/splicebert_exon_edge_compatibility/crossed_direct_edge_replication_audit.json` |
| Reverse behavior (AC1) | `+0.9763 [0.9060,1.0526]`, 974 pairs/645 genes; controls `-0.0009/+0.0147` | Validation authorization-gate observation, not sealed confirmation or symmetry. | `results/splicebert_acceptor_coupling/r1b_reverse_coupling.json` |
| Reverse components (AC5/AC6) | L1H12 specific `+0.2447`; L4H11/H9/H7 `+0.0934/+0.0905/+0.0788`; Holm `p=0.00175` | Bounded, intervention-specific component effects on a held-out-within-validation target. No reverse route or mediation claim. | AC compact attestation and R3c decision audit |
| DNABERT representation (DBP3) | BA `0.8102`; AUC `0.8919` | The frozen layer-2 endpoint carries promoter-class information. Thin layer lead (`0.0019`) is not localization. | `results/dnabert_promoter/frozen_claim_tests/` artifacts |
| DNABERT core/boundary (DBP6--DBP8) | five-node necessity `0.630`, delivery `0.882`, standalone `0.194`; matched/same-class `0.879/0.843`, `p=0.252` | A conditional causal core sits inside an MLP-heavy distributed computation. Not a compact circuit, example-specific mechanism, or SpliceBERT replication. | DBP6--DBP8 controlling artifacts; Figure 9 sources |

## 5. Paper-safe narrative

### 5.1 Main result sequence

1. **Behavior.** Paired-acceptor corruption produces a selective `+0.872` masked-donor margin change on
   400 sealed chr8/9 examples; matched controls are near zero.
2. **Candidate nomination and necessity.** An exhaustive validation scan ranks individual heads by
   partner-specific mean-ablation damage. A separate sealed intervention establishes `{L3H1,L2H15}` as
   a dominant necessary pair against 400 matched random two-head interventions.
3. **Address.** Token--position surgery shows that L3H1's acceptor selection is predominantly positional,
   with L2H15 amplifying this attention-side address. Translation surgery retains a large relative-offset
   signal but also shows absolute/edge modulation, leaving the frozen absolute-vs-relative decision mixed.
4. **Payload.** With the address fixed, clean values in the four-token paired-acceptor window restore
   the effect. Backward-pruned held-out interventions trace their construction through
   `{L0H7,L0H12,L0H10} -> MLP0 -> MLP1 -> MLP2`; semantics remain unidentified.
5. **Write.** Joint address and payload restoration is much larger than either alone. L3H1 restoration
   writes a donor-local state whose effect is largely captured by stable low-rank `U_info`.
6. **Delivery.** L5H11, L4H2, and L4H0 carry the state through OV/value components toward the masked
   donor readout.
7. **Boundary.** The named spine is highly injectable but generates almost none of the behavior when
   isolated. The held-out static-ranking curve has no sparse knee, so the safe governing formulation is
   **a structured, top-ranked causal core within distributed computation**.
8. **Independent positional result.** C36 supports phase-aligned present-day use of a learned period-3
   direction. C43 separately supports register-preference behavior in designed fixed-span chimeras.
   C44 does not causally connect them.
9. **Bounded extensions.** The reciprocal SpliceBERT endpoint has behavior and selected component
   effects but no four-role mechanism or symmetry result. DNABERT-2 reproduces the *shape of the
   faithfulness boundary*, not the SpliceBERT circuit.

### 5.2 Claims to retain as negative or stopped results

- C24/C30: payload semantics and intrinsic dimensionality are unresolved; “splice-site strength” is
  withdrawn.
- C27/C33: the small named core is injectable but not autonomously faithful; the static curve has no
  sparse knee.
- C38: matched versus same-task margin-mismatched transfer is unresolved.
- C41: example-specific off-generic-axis final-residual content is inert only for the tested additive
  donor-margin assay; two of eight draws are instrument-limited.
- C36: CDS-minus-non-CDS-exon contrast `+0.006 [-0.080,+0.093]` is inconclusive.
- C44: a 114-block preflight failed authorization; no 384-block target result exists.
- AC7: L3H1 reuse and distinct-route conjunctions are inconclusive, not absent.
- AC8: L2H15 is health-ineligible, not supported and not null.
- AC9: schema mismatch stops the mechanism handoff before chr8/9 science.
- ED1: preregistered gates failed; neither a positive nor negative definition-switch interpretation is
  licensed.
- DBP7/DBP8: no compact DNABERT circuit and no detected matched-over-same-class specificity; TATA
  follow-ups are also null/bounded.

### 5.3 Unsupported or prohibited formulations

- “first systematic application” or “first causally supported account” without a separate literature
  review;
- sparse/minimal/autonomous/complete circuit, or proof of irreducible distribution;
- address `=` period-3, period-3 `=` coding frame, or period-3 `=` C43 register mechanism;
- payload as splice-site strength, donor identity, U2AF/branch-point state, or another biological
  semantic;
- biological splicing, molecular mechanism, or wet-lab validation from model behavior alone;
- reciprocal symmetry, the donor mechanism running backward, or masked-acceptor chr8/9 confirmation;
- DNABERT sharing the SpliceBERT circuit or providing an example-specific transferable mechanism;
- multispecies **origin** rather than comparative training-condition association;
- all studies preregistered, all gates passed, or all artifacts clean-checkout reproducible.

## 6. Figure audit

The active `figures/` directory contains exactly the four raster assets used by the
current workshop manuscript: the conceptual workflow, the SpliceBERT headline, the DNABERT-2 workflow
transfer composite, and the positional/faithfulness composite. Each asset has a repository-relative,
hash-checked provenance sidecar. Historical and supplementary figure suites remain recoverable from Git
history but are no longer present in the active tree.

The current captions preserve the required boundaries: the workflow is conceptual; the period-3 and
register-preference findings are separate; the 90\% faithfulness statement distinguishes the 64.2-node
interpolation from the first sampled crossing at 72/103; and the DNABERT-2 panel claims workflow transfer,
not mechanism transfer. The C43 panel remains behavior-level evidence with the fixed 120-nt chimeric span,
384 blocks, add-one resolution floor, and stop-load rival.

## 7. Supersession of the old verification file

The deleted `RESULTS_VERIFICATION.md` remains recoverable from Git history at SHA-256
`3300fc6af0e8b00ff9abca8116ced018e25f2fb4538342ce5443e6442fa0ae15`. Its historical `52/52` claim is
**not retained**: two C8 rows compare different cohort/endpoints while marking them exact, and the C14
biology wavelength cites the wrong source (`carrier_subbin.json` rather than the dyad artifacts). The old
table is provenance, not a current scientific verification:

- A4 is superseded by A11/C32, but A11's frozen classifier, post-hoc result wording, and manifest remain
  inconsistent; use the bounded geometry description above.
- A6 is no longer unrun; C27 and C33 now define the injectability/faithfulness boundary.
- The live ledger now extends through C46 plus IC1/ED1 and permanent corrections R1--R52.
- “Information-specific circuit” is overstrong after C38/C41.
- `U_info` does not license paired-intron semantics or donor identity.
- The prior 120-pass/33-skip test baseline is obsolete; the current baseline is 1,023/22.
- The forward program has changed: acceptor AC9 is terminal by its old lifecycle, and a new successor is
  required for address--payload--write--delivery and symmetry work.

## 8. Complete live-ledger coverage index

This index proves disposition coverage of the **123 claim rows in seven active ledgers**: 45 donor
partner-selection rows (including IC1/ED1), 34 GTEx rows, 21 HCA rows, 10 DNABERT rows, 11 acceptor
rows, PBC1, and PPM1. It is an index, not a substitute for the exact licensed wording in each ledger.

### 8.1 Donor partner-selection ledger: 45/45 rows

| IDs | Disposition in this audit |
|---|---|
| C1, C2, C42, C19 | Main behavior/specificity. C42 validates behavior-level acceptor-alt robustness only; A0--A12 remain CC-only. C19 is a behavioral signature, not payload semantics; its phase-flat arm is descriptive and does not establish equivalence. |
| C3, C4, C5, C6 | Main necessity/anatomy. Dominant and conditional, not complete; marginal component effects are not additive shares. |
| C20, C21, C22, C32, C25 | Main address evidence with the A11 frozen-rule/result-manifest conflict disclosed. QK/attention preference is geometry, not partner identity. |
| C24, C28, C29, C31, C30, IC1 | Main payload localization/construction; C30 semantic stop and IC1 distal-arrangement sub-SESOI result belong in limitations/supplement. No minimal L0 set or biological payload name. |
| C23, C16, C17, C18, C26 | Main conjunction, donor-side state, write, and delivery. Preserve conditional estimands and C26 interval direction. |
| C27, C33, C7 | Main faithfulness boundary; C7 is closed/superseded and its damaged-model null is not completeness evidence. |
| C8, C9, C10, C34, C35, C36 | P3 result family. C8/C9 are supportive lesion/mediation evidence with health/control limitations; C10 is a matched-null power stop; C34 is span-distributed with a missed confirmation gate; C35 supports L3H1-attention mediation; C36 is the clean phase-aligned-use headline. |
| C39, C40 | Discussion/supplement comparative association and scoped cross-model absence; no origin, universality, or species-diversity cause. |
| C11, C12, C13 | Separate ≈10.63 carrier. Position/QK/attention are supported; donor-side general OV/function remains bounded/open after B1. The human-only proposal fails at attention. |
| C14, C15 | Mouse ES-cell/mm9 nucleosomal comparator and powered wavelength refutation. Do not call it a human measurement or an explanation of ≈10.63. |
| C41, C38 | Verification-tier cross-example boundaries: generic additive final-residual axis and unresolved content identity. |
| ED1 | Closed gate failure: no positive or negative intron/exon-definition switch interpretation. |
| C43, C44 | Bounded chimeric register preference and separate instrument stop. No P3-to-register mediation result. |

### 8.2 Acceptor coupling ledger: 11/11 rows

| IDs | Disposition in this audit |
|---|---|
| AC1 | Bounded validation behavior gate, not sealed confirmation. |
| AC2 | Post-outcome donor-base operator screen; useful for designing matched substitutions, not biology. |
| AC3 | Retracted stratum contrast. |
| AC4 | Candidate-generating component scan; shared/specific partition retracted/instrument-unresolved. |
| AC5, AC6 | Bounded held-out-within-validation component interventions; no route, write, or mediation. |
| AC7 | L3H1 reuse and distinct-route conjunctions inconclusive, not absent. |
| AC8 | L2H15 health-ineligible, neither positive nor null. |
| AC9 | Terminal old-program schema stop before target science; chr8/9 masked-acceptor endpoint unopened. |
| AC10, AC11 | Bounded natural-GC behavior and attention modifiers with quasi-observational/provenance limits. |

### 8.3 DNABERT-2 promoter ledger: 10/10 rows

| IDs | Disposition in this audit |
|---|---|
| DBP1, DBP2 | Task definition and leakage-cleaning provenance; chromosome/gene metadata unavailable. |
| DBP3, DBP4 | Strong promoter representation at the working Layer-2 endpoint; the `0.0019` layer lead is not localization. |
| DBP5 | Candidate ranking only; no matched-size calibration family. |
| DBP6, DBP7 | Main bounded comparison: conditional five-node core plus no-knee, MLP-dominated standalone faithfulness. |
| DBP8 | Class-level transfer; matched-versus-same-class difference nonsignificant (`p=0.252`). |
| DBP9, DBP10 | Positional enrichment only and frozen TATA null; no TATA mechanism. |

### 8.4 GTEx fine-tune-shift ledger: 34/34 rows

| IDs | Disposition in this audit |
|---|---|
| GFS1, GFS2 | Source/checkpoint provenance and instrument preflight. |
| GFS3, GFS4, GFS5 | Failed original-MLM mechanism route; GFS4 is readout incompatibility under collapsed MLM health, not encoder erasure. |
| GFS6, GFS7 | Retracted coordinate instrument and corrected native-classifier preflight. |
| GFS8, GFS9, GFS10, GFS11, GFS12, GFS13, GFS14, GFS15, GFS16, GFS17, GFS18 | Strong complete extension for Discussion/supplement: native paired dependence, late integration, L5H6 minority carrier, final-MLP dominance, nonlocal higher-order sequence regions, several late heads, and triple-to-final-MLP mediation. The triples carry only 28.6--34.4%; no complete/minimal or biological grammar claim. Conformance is self-reported rather than independently audited. |
| GFS19, GFS20 | Exploratory reused-partition alignment and native-unit decomposition; not independent confirmation or monosemantic biology. |
| GFS21 | Sparse-transcoder representation gate failed; no semantics/steering. |
| GFS22 | Outcome-blind biological-feature construction stopped before model output; RBP candidates remain unconfirmed. |
| GFS23 | Positional spectral retention; dominant-plane sensitivity is post-hoc and no full-rank coordinate-retention label is licensed. |
| GFS24 | ≈2.2-nt amplified bands are causal bounded nulls for native donor coupling. |
| GFS25 | P3 and ≈10.63 support local native margins but are bounded nulls for paired coupling; this separates carrier retention/use from reuse of the base mechanism. |
| GFS26, GFS27 | Jacobian-SAE instrument and AuxK repair failures. |
| GFS28, GFS29 | Outcome-blind representation instrument and five-seed representation replication; no semantics or causal feature identity. |
| GFS30, GFS31, GFS32 | Behavioral/reconstruction instruments remain unresolved despite target-path approximations; feature identity and steering stay closed. |
| GFS33, GFS34 | Larger and model-difference sparse instruments stop at unresolved pilots; compressible final writes do not sparsely explain upstream change. |

### 8.5 Head-context atlas ledger: 21/21 rows

| IDs | Disposition in this audit |
|---|---|
| HCA1 | Cohort feasibility, strongly short-intron biased. |
| HCA11, HCA12, HCA13, HCA14, HCA15, HCA16, HCA17 | Cohort, deployment, analysis, mechanism-handoff, validation, conditional-stage, and visualization instruments; infrastructure, not biological results. |
| HCA18 | Four general-use candidates plus 20 discovery context signals; all context signals fail the frozen precision gate, so the context shortlist is empty. |
| HCA19 | Discussion/supplement: all four general-use components validate for both roles on internal validation. Component reuse only, not QK/write/path/biology; dirty-run provenance must be disclosed. |
| HCA20 | Large conditional-path results are off-repo/Drive and followed a 100× numerical-gate relaxation; do not headline until artifacts are accessible and independently audited. |
| HCA23 | Context signals closed as underpowered leads, but implementation/conformance deviations make this design-limit prose only. |
| HCA9 | Valid donor L3H1 positive control; useful as rediscovery, not reciprocal mechanism evidence. |
| HCA21, HCA22 | Visualization transport only; no new scientific finding. |
| HCA2, HCA3, HCA4, HCA6, HCA7, HCA10 | Closed competition branch: behavior is not competition-specific; HCA6 never executed; bounded exploratory remnants do not support a competition circuit. |

### 8.6 Capacity-stop ledgers: 2/2 rows

| IDs | Disposition in this audit |
|---|---|
| PBC1 | Model-free necessary-capacity stop: zero blocks selected, no model loaded, no biological/model result. |
| PPM1 | Model-free max-flow/capacity result only; exploratory pilots were archived and cannot support a training conclusion. |

Prospective `splicebert_acceptor_context` and the template study contain no live result rows. Archived
exon-edge prose is superseded except for consolidated C43/C44. Excluded legacy trees, unledgered
notebooks, the PPM pilots, and the notebook-only fine-tuned DNABERT-2 extension are recoverable at
`archive/unsupported-exploratory-2026-08-22`, not part of the active evidence tree.

## 9. Audit boundary

This is an internal repository and supplied-manuscript audit. It verifies artifact arithmetic,
provenance, cohort definitions, and licensed claim language. It does not perform an external literature
review, biological validation, or model rerun. The current checkout has no live `tex/` manuscript; the
paper reviewed here is an external attachment. Consequently, the fixes above must be integrated into
the actual manuscript source and re-audited after line numbers, captions, citations, and asset paths
stabilize.
