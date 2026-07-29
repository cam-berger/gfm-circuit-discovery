# Preregistration v1: biological features in the fine-tuned splice-classification code

> **Tier: T2 — provenance.** Frozen before any model outcome on the
> `gtex-native-biological-features-v1` cohorts, before fitting a sparse
> dictionary, and before opening any COMPASS splicing outcome. Existing GFS1–GFS20
> results motivate the assay but may not select a new gene, sequence feature, sparse
> latent, external variant, threshold, or biological label.

## 1. Question and evidence boundary

GFS14 localized two distal sequence-order dependencies of the released GTEx
classifier:

- donor evidence uses `[-80,-48)` relative to the donor;
- acceptor evidence uses `[+16,+48)` relative to the acceptor.

Those coordinates are sequence-defined. The original broad cohort did not establish
that every window was exonic or coding. The present study applies a new comprehensive-
annotation gate before using either biological label.

GFS15–GFS20 show that these signals arrive late, travel through several layer-5
attention routes, and are synthesized by hundreds of final-MLP units into a compact
checkpoint-rotated write. They do not name the sequence variable.

This study asks:

> Does fine-tuning construct causal features for exonic splicing-regulatory grammar,
> generic coding/exon identity, codon preference, or local RNA folding, and can a
> sequence feature be traced through a sparse final-MLP feature to the native
> classifier and an independent splicing assay?

The competing explanations are frozen before outcomes:

1. **RBP-affinity grammar:** sequence changes alter an independently measured RBNS
   affinity model, including possible ESE/ESS-like signals;
2. **codon grammar:** synonymous codon preference or coding periodicity is used as a
   boundary cue;
3. **generic exon identity:** the sequence supports an annotation-trained
   exon-versus-intron discriminator without requiring a specific regulatory factor;
4. **local RNA folding:** synonymous changes alter local ViennaRNA ensemble
   accessibility;
5. **unresolved higher-order order:** the confirmed effect survives these explanations.

A model feature, sequence grammar, RBP-affinity match, eCLIP peak, and reporter effect
are separate evidence levels. Motif similarity alone never licenses an RBP name.

## 2. Frozen cohorts and leakage exclusion

All examples come from the annotation-only `train` chromosome split. The sealed chr8/9
test family remains closed. Chromosomes `chr8`, `chr9`, `chr11`, `chr16`, `chr19`,
`chrX`, `chrY`, and mitochondrial contigs are forbidden.

Before cohort construction, collect the union of every gene ID present in every
existing machine-readable artifact under
`results/splicebert_gtex_finetune_shift/`. Exclude that complete union. The exclusion
list and source artifacts are written to every result.

Every remaining gene is assigned with

`SHA256("gtex-native-biological-features-v1:" + gene_id) mod 100`.

- buckets `0–49`: discovery;
- buckets `50–99`: confirmation.

Within-gene example order is deterministic and outcome-blind.

Required cohorts:

- **preflight:** 16 discovery-partition examples, tag
  `gtex-native-biological-features-preflight-v1`;
- **discovery:** seed 2020, 128 examples whose genes exclude preflight, tag
  `gtex-native-biological-features-discovery-v1`;
- **confirmation:** all checkpoints 2020–2024, 200 examples, tag
  `gtex-native-biological-features-confirmation-v1`.

Both target regions must fall entirely inside one unambiguous CDS register in the
comprehensive GENCODE v50 annotation, in transcript orientation, and contain exactly
ten complete codons. Partial edge codons are never edited. Annotation filtering occurs
before deterministic sampling. The donor and acceptor arms use the same CDS-qualified
examples unless construction feasibility requires a prespecified per-arm subset; genes
remain disjoint across stages and all raw identities are recorded.

The annotation-only audit before freezing found 2,040 eligible unexposed genes with
both regions fully covered by a single CDS register under the comprehensive annotation.
This is a feasibility count, not a model result.

Frozen reference inputs:

- comprehensive GENCODE v50 GTF SHA256:
  `83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453`;
- hg38 FASTA SHA256:
  `5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51`.

The gene is the resampling and sign-flip unit. Checkpoint seeds are a required
replication family, not independent samples.

## 3. External sequence-feature baselines

### 3.1 RBNS affinity

Use the 79 RBPamp PSAM files distributed with the published Sparse Adjusted Motif
implementation:

- repository: `https://github.com/kavigupta/sam`;
- source commit inspected before freeze:
  `8dbf38d1855112505ef4152cebebf416054ced7d`;
- source directory: `spliceai/data/motifs/`;
- sorted-file aggregate SHA256:
  `cc515bdd38d2fe777d17b468fece07943f0d2ef942b0493348838194aae4d35e`;
- local files remain ignored external inputs and are hash-recorded.

For a PSAM component, score every fully contained placement by its published `A0`
scale times the product of its base affinities. Sum placements and components within
the 32-nt region. No model outcome modifies a PSAM, threshold, width, or RBP name.

### 3.2 Codon preference

Estimate synonymous codon frequencies from GENCODE v50 CDS sequence outside every
preflight/discovery/confirmation gene and forbidden chromosome. Apply add-one
smoothing within each amino-acid family. A variant's codon-preference delta is the
clean-minus-variant log conditional frequency. Stop codons are ineligible.

### 3.3 Generic exon identity

Fit an annotation-only logistic baseline on 1–4-mer counts from 32-nt CDS exon
segments versus same-gene intronic segments matched by chromosome, GC decile, and
distance-to-junction band. Training genes exclude all study genes. Its hyperparameters
are selected without any SpliceBERT activation or output. The baseline produces an
exon-identity logit and is never itself called a biological mechanism.

### 3.4 Local RNA folding

Pin ViennaRNA `>=2.7,<3`. Use RNAplfold ensemble unpaired probability on a 201-nt
transcript-oriented context with frozen settings `W=150`, `L=100`, `u=7`. The primary
descriptor is the clean-minus-variant mean probability that each 7-mer beginning in
the target region is unpaired. Minimum free energy is reported only as a sensitivity
descriptor. This is a computational structural candidate, not experimental RNA
structure.

## 4. Outcome-blind sequence interventions

### 4.1 Instrument and region controls

Reuse the exact directed-dinucleotide-preserving shuffle from GFS14 as the positive
region-order control. Reuse its endpoint-specific equal-width sham region as the
negative region control. Local `GT→CC` and `AG→CC` destruction remain classifier
positive controls, not biological-feature outcomes.

### 4.2 Synonymous saturation

For every complete codon in the target region, enumerate every alternative standard
genetic-code codon encoding the same amino acid. Each variant changes exactly one
codon. Record changed positions, Hamming dose, RBNS-affinity deltas, codon-preference
delta, exon-identity delta, and MFE delta.

Also construct two deterministic maximum-Hamming synonymous recodings of every
mutable complete codon. The two draws differ only through a hash tie-break and must
preserve:

- amino-acid sequence;
- reading phase and length;
- every base outside the 32-nt region;
- canonical target splice dinucleotides;
- no newly created canonical `AG` or `GT` dinucleotide inside the edited region;
- every partial edge codon.

Synonymous edits test nucleotide grammar conditional on protein sequence; they do not
by themselves distinguish an RBP motif from codon bias, another motif, or structure.

### 4.3 Candidate-specific destruction, rescue, and gain

Discovery may freeze at most three nonredundant RBNS candidates per endpoint and the
three always-carried macro baselines: codon preference, exon identity, and MFE.

For each frozen candidate, an outcome-blind beam search over synonymous codons creates:

- **low/destroy:** minimum candidate score within four changed bases;
- **high/rescue-or-gain:** maximum candidate score within four changed bases;
- **matched controls:** three noncandidate PSAMs matched on width, information content,
  natural score variance, eligible-example count, and delivered edit count.

Every constructed arm is additionally audited for aggregate off-target PSAM affinity,
codon usage, GC3, codon-pair score, MaxEnt splice-site landscape, canonical `AG`/`GT`
counts, and RNA accessibility. The target-versus-control nuisance bands are frozen
from discovery before confirmation. If an edit cannot isolate the target descriptor,
it is labeled a composite synonymous edit rather than candidate-specific.

RBNS candidates with absolute discovery score-profile correlation above `0.90` are one
redundancy cluster; retain the discovery winner only. The target feature, controls,
construction success rate, and eligible examples are frozen before confirmation.

High-versus-low is the primary transferable feature contrast. Natural clean→low
destruction and low→sequence-distinct-high rescue are secondary directions. A
candidate must have at least 40 eligible examples from at least 30 genes in discovery
and 60 examples from at least 45 genes in confirmation.

### 4.4 Cross-example transduction

For a confirmed RBP-affinity candidate, construct peptide-preserving gain edits in
naturally low-affinity carriers. For generic exon identity, run a separate explicitly
synthetic within-recipient replacement assay: comprehensive-annotation real-exon
32-mers versus intronic 32-mers matched on endpoints, Hamming dose, GC/dinucleotides,
canonical motifs, MaxEnt landscape, aggregate PSAM affinity, RNA accessibility, and
donor/recipient gene disjointness. These replacements are portability tests and are
not claimed to preserve protein sequence. An intronic segment is never called a
pseudoexon without a separate motif-and-annotation qualification.

## 5. Hard preflight

No discovery outcome may open unless all gates pass.

### Sequence and annotation gates

1. every synonymous variant translates identically over every edited complete codon;
2. every target edit is confined to the declared 32-nt region;
3. maximum synonymous recoding changes at least 20% of editable bases in at least 90%
   of preflight examples;
4. target and sham dinucleotide shuffles preserve their five GFS14 invariants and
   deliver at least 25% Hamming dose;
5. PSAM parser replays published matrix widths, component counts, and finite
   nonnegative values for all 79 files;
6. identity PSAM, codon, exon-identity, and MFE calculations are exact;
7. GTF-to-sequence strand reconstruction, codon slots, and translated codons agree for
   both strands;
8. preflight, discovery, confirmation, prior-exposed, and forbidden-chromosome sets
   obey every disjointness invariant.

### Native-model gates

1. identity input and identity residual interventions change margins by at most
   `2e-5`;
2. local splice-dinucleotide destruction lowers its endpoint margin by at least `5.0`;
3. target-region dinucleotide shuffle has mean damage at least `0.05` for each
   endpoint;
4. its 95% gene-bootstrap interval excludes zero, the effect is positive in all five
   checkpoints, candidate-minus-sham has an interval above zero, and the add-one
   grouped sign-flip test has family-corrected `q<=0.05`;
5. final-layer replay and complete MLP composition remain within the previously
   calibrated precision bounds;
6. all model outputs are finite and the checkpoint family loads exactly.

Any anomaly stops interpretation and is preserved. Numerical implementation
tolerances may be calibrated only before target-feature outcomes and must be documented
without changing scientific effect thresholds.

## 6. Discovery and feature freeze

For seed 2020, record native donor/acceptor margins, target-token final-MLP inputs and
writes, all 16 layer-5 head outputs, and every feature descriptor for clean, target
shuffle, sham shuffle, synonymous saturation, and maximum-recoding sequences.

Discovery is candidate generation:

1. Fit a gene-grouped elastic net predicting single-codon margin change from
   standardized RBNS, codon-preference, exon-identity, MFE, and edit-count deltas.
2. Use five fixed gene folds and a fixed regularization grid. A descriptor is eligible
   only if its coefficient has the same sign in at least four folds and the
   descriptor-containing model improves held-out squared error over the nuisance-only
   model.
3. Rank eligible RBNS descriptors by median absolute standardized coefficient, cluster
   redundant score profiles, and freeze at most two per endpoint.
4. Carry codon preference, exon identity, and MFE regardless of discovery rank so a
   targeted null is recorded rather than silently omitted.

No discovery regression, logo, top sequence, or sparse latent establishes causality or
biological identity.

## 7. Six-view sparse final-MLP feature model

### 7.1 Architecture and data

Fit a paired multi-view transcoder on the same ordered, label-free target-token states
from:

`{base SpliceBERT.1024nt, fine-tuned seeds 2020, 2021, 2022, 2023, 2024}`.

For view `s`,

`x_s = exact input to the final BertIntermediate`,

`z_s = sparse nonnegative detector_s(x_s)`,

`m_hat_s = writer_s(z_s)`.

Each view has its own detector and writer. Fine-tuned views share feature indices
through matched-sequence activation consistency; the base view is a contrast and is
not forced to activate a fine-tuning-specific feature.

Training may use clean, shuffled, synonymous, high/low, and annotation-matched non-site
states from discovery genes only. It may not access classifier weights, labels, or
margins.

The objective contains:

- complete native MLP-write reconstruction;
- clean/counterfactual write-delta reconstruction;
- base/fine view-difference reconstruction;
- fine-seed activation consistency;
- BatchTopK sparsity.

### 7.2 Frozen model-selection grid and controls

- feature widths: `{128,256,512,1024}`;
- active features per example: `{8,16,32}`;
- three dictionary initializations per setting;
- choose the smallest setting passing every preflight; break ties by validation
  write-delta error, then feature width, then active count.

Required baselines:

1. exact native 2,048-unit curves from GFS20;
2. PCA/SVD at matched dimensions;
3. Haar-random subspaces and random dictionaries matched for dimension, writer norm,
   and activation count;
4. independent per-seed transcoders with post-hoc matching;
5. the joint model without view-difference loss;
6. the joint model without seed-consistency;
7. the complete native MLP ceiling.

The sparse instrument passes only if, on outcome-blind held-out discovery genes:

- all-feature write reconstruction explains at least 95% of total write-delta energy;
- replacing the native write by the reconstruction preserves clean and counterfactual
  endpoint margins within mean absolute `0.05`;
- the sign and at least 90% of each full-MLP target-shuffle effect are preserved;
- a planted sparse-feature simulation is recovered above its calibrated floor;
- permuting cross-seed example pairings destroys activation agreement without
  spuriously improving per-view reconstruction;
- selected feature families match across all three dictionary fits.

Failure stops feature interpretation; the native-unit result remains valid.

### 7.3 Biological-feature assignment

For each frozen sequence candidate, rank sparse latents on discovery by the
clean/high/low activation contrast and their decoded contribution to the corresponding
native write. Freeze the smallest nested latent set through 32 features that reaches
80% of the full decoded candidate contrast. Compare with three activation-frequency-,
feature-count-, and writer-norm-matched random sets.

Classify the representational shift descriptively as:

- **acquired detector:** candidate activation is absent/sub-SESOI in base and stable in
  tuned seeds;
- **amplified detector:** corresponding base feature exists but tuned activation
  increases;
- **rewritten feature:** detector responses match but tuned writers change;
- **readout-only use:** feature and writer are stable while classifier sensitivity
  changes.

These labels require causal confirmation below.

## 8. Confirmation: sequence to head to feature to classifier

Confirmation uses frozen sequences, candidates, control descriptors, sparse-feature
sets, signs, and thresholds across all five tuned checkpoints.

Primary quantities for each candidate and endpoint:

1. **sequence behavior:** native margin(high) minus native margin(low);
2. **feature damage:** on the native high run, add only the selected decoded
   `z_low-z_high` write;
3. **feature recovery:** on the native low run, add only the selected decoded
   `z_high-z_low` write;
4. **head→feature path:** patch the frozen GFS18 head triple, recompute its selected
   feature delta, and add only that decoded feature write;
5. **compact-code overlap:** energy of selected decoded writes inside the target
   checkpoint's frozen aligned rank-4 GFS19 subspace;
6. **natural-range steering:** replace selected activation from its observed 10th,
   25th, 50th, 75th, and 90th percentile while retaining the native reconstruction
   error and all unselected feature contributions.

Steering doses may not exceed the discovery activation range. Monotonicity is evaluated
in the discovery-frozen sign. Feature patching uses native-write-plus-decoded-delta, so
an autoencoder reconstruction error cannot itself create recovery.

A sequence-linked sparse feature confirms only if:

- high-versus-low behavior has absolute family mean at least `0.05`, a 95% grouped
  bootstrap interval excluding zero, and the frozen sign in all five checkpoints;
- both feature damage and recovery have the frozen sign, intervals excluding zero,
  corrected `q<=0.05`, and at least 50% of the full high/low behavior;
- candidate effects exceed every matched control family in the frozen sign with
  corrected `q<=0.05`;
- at least 80% of the selected decoded-write energy lies in the aligned rank-4
  task-code subspace or the failure is explicitly reported;
- the natural-range dose response is monotone by Spearman `|rho|>=0.8` with the frozen
  sign and exceeds matched features;
- results are positive in the frozen sign in all five checkpoints.

Head→feature mediation is a separate verdict. It requires a positive directed path
above matched head/feature controls but need not explain the complete sequence effect.

All primary candidate, direction, control, path, and steering tests form one declared
family per stage. Use exact add-one gene sign flips with 9,999 permutations and BH
correction across the complete scanned family. Use 5,000 gene bootstraps. Do not pool
null sets of different size.

## 9. Independent biological validation

### 9.1 Primary peer-reviewed functional anchor: Vex-seq

Freeze Vex-seq rows and sequence mappings before opening delta-PSI outcomes:

- study: Adamson et al., Genome Biology 2018;
- GEO: `GSE113163`;
- code/data mirror: `https://github.com/scottiadamson/Vex-seq`, inspected commit
  `36d9c4a6e5e74def46b3cada69d3166f08c54bd5`;
- frozen inputs:
  `processed_files/Vex-seq_positions.tsv`,
  `processed_files/delta_PSI_values.tsv`, and
  `preprocess_data/68_97_final_sequences_fixed.tsv`.

Files are downloaded at runtime and hash-recorded, not vendored. Because the assay is
GRCh37.p13, the primary mapping uses author-provided transcript-oriented sequences.
Any genomic liftover is a sensitivity analysis requiring a unique lift and reference-
allele validation. Primary cell lines are K562 and HepG2. The exact donor and acceptor
bins contain hundreds of SNVs in the annotation-only feasibility audit.

### 9.2 Temporally independent functional replication: COMPASS

Freeze COMPASS before opening its outcome columns:

- study: Koplik et al. 2025 exon-splicing MPRA;
- GEO: `GSE307247`;
- processed outcomes:
  `GSE307247_Processed_PSIs_All_Cells.csv.gz`;
- column dictionary and exact 161-nt library FASTA from the same GEO supplementary
  directory;
- code reference:
  `https://github.com/skoplik/ESL_MPRA_2025`, inspected commit
  `67791aa433bf66ac0b7b095ede01d650c5de9cb1`.

Files are downloaded at runtime and hash-recorded, not vendored. Before reading PSI or
delta columns:

1. reconstruct reference alleles and 161-nt sequences;
2. convert the documented one-based inclusive event and exon coordinates explicitly
   to internal half-open intervals;
3. retain single-SNV constructs only;
4. map transcript-oriented positions and strands exactly;
5. retain SNVs in the two frozen relative windows;
6. separate CDS and UTR;
7. freeze candidate-carrier, matched-noncarrier, cell-line, and grouping rows.

Require `len(full_seq)==event_end-event_start+1`,
`len(exon)==exon_end-exon_start+1`, strand-aware genomic reference agreement, and
canonical `AG`/`GT` checks before outcomes. COMPASS `full_seq` is transcript-oriented.

Primary assay outcome: WT-matched delta-logit PSI. Primary matched cell line is K562;
the other four assayed cell lines are a replication family. Compare experimental
delta-logit PSI with the frozen model margin delta, sparse-feature delta, and external
feature-score delta. Group by exon and gene.

A reporter bridge requires:

- directionally consistent model/assay effects in K562 and at least three of five cell
  lines;
- grouped interval excluding zero for the frozen association or carrier contrast;
- an effect exceeding edit/position/GC-matched noncarrier variants;
- corrected `q<=0.05` across all frozen candidates and endpoints.

Vex-seq and COMPASS are transient reporter/minigene assays in transformed cell lines.
COMPASS is presently a preprint. A positive result licenses `reporter-supported
cis-regulatory feature`, not endogenous spliceosome biochemistry.

### 9.3 Binding anchors

For a frozen RBP candidate:

- use ENCODE-released RNA Bind-n-Seq 6-mer enrichment as an independent affinity
  sensitivity check when the target exists;
- use ENCODE GRCh38 merged-replicate eCLIP peaks, prioritizing K562 and then HepG2,
  only when the target exists.

Candidate availability may not change selection. eCLIP tests enrichment of candidate
sites inside the frozen windows against position-, GC-, expression-, and
affinity-matched noncandidate sites. RBNS is cell-free affinity; eCLIP is binding, not
regulatory sign.

GTEx sQTL and MFASS are sensitivity analyses only. GTEx shares the fine-tune's data
ecosystem; MFASS was available to the original SpliceBERT study.

## 10. Licensed claims

Evidence wording is frozen by the highest rung reached:

| Evidence | Maximum wording |
|---|---|
| feature descriptor predicts discovery variants | sequence-feature candidate |
| controlled sequence high/low plus sparse-feature mediation | causal classifier feature |
| independent RBNS affinity only | RBP-affinity-linked classifier feature |
| plus matched eCLIP binding | RBP-associated binding feature |
| plus Vex-seq and temporally independent COMPASS direction | replicated reporter-supported cis-regulatory feature |

No computational result establishes recruitment of an RBP, spliceosome assembly,
endogenous tissue regulation, or a protein-level consequence. A coding/exon-identity
shortcut or a fully unresolved outcome is retained as a publishable alternative rather
than renamed as regulatory biology.
