# Reconstruction Atlas protocol

> **Tier: T1 — canonical for the study design.** Current, authoritative. Read this before a run.

## Scope and coordinate contract

- Checkpoint: `zhihan1996/DNABERT-2-117M` at revision
  `7bce263b15377fc15361f52cfab88f8b586abda0`.
- Stored sequence coordinates are zero-based, half-open nucleotide intervals.
- Directional examples are stored transcription 5′→3′; genomic coordinates and strand remain
  explicit.
- Unstranded regulatory intervals are stored in genomic-forward orientation and retain
  `source_strand=.`.
- Multisource split assignment occurs on shared 1-Mb genomic blocks before any outcome is
  computed. A target is excluded if its complete 512-nt-flanked window crosses a block boundary.
  This makes blocks disjoint across every source and feature class rather than only within one
  annotation database.
- Development, baseline-training, and confirmation clusters must be disjoint.
- Every element carries source, release, evidence type, and source-record identity.

Computational motif calls are never silently described as experimentally supported. Derived splice
dinucleotides, codons, and polypyrimidine tracts retain
`derived_from_curated_annotation`; direct assay records retain `experimentally_supported`.
The five-nucleotide centers of broad ENCODE TF peaks and dELS cCREs are deterministic assay
anchors, not claims that the complete functional element is five nucleotides or that its causal
base is known.

## Phase A — collection qualification

For every source and site class:

1. verify assembly and coordinate conventions against source memorial records;
2. verify plus/minus strand sequence identity;
3. reject incomplete windows, unsupported symbols, duplicate identifiers, and target intervals
   outside the sequence;
4. cluster before splitting and audit cluster leakage;
5. record site count, cluster count, strand, chromosome, sequence, GC, and token-geometry
   distributions;
6. quarantine any class that cannot supply all empirical control families.

Counts are instrument facts, not model behavior.

The qualified preflight report binds the exact collection, matched-control file, matching audit,
and baseline-training corpus by SHA-256. A target runner refuses a byte-different replacement even
when the release label is unchanged.

## Phase B — mask and native score

The mask is the minimal consecutive BPE token cover. It is applied to token IDs after tokenization.
The label IDs never appear in the masked input. Required per-site diagnostics are:

- annotated nucleotide interval;
- target token indices and offsets;
- total token-cover nucleotides;
- BPE overhang;
- true-token log probability sum and per-covered-nucleotide score;
- token accuracy and exact token reconstruction;
- eligible and rejected alternative margins, with rejection reason.

Never call the summed score a joint sequence probability. All target tokens are masked
simultaneously.

## Phase C — excess biological recoverability

Each target receives the frozen number of controls from each empirical family. Match balance is
reported separately by family. The primary effect for family \(f\) is
\(\Delta_f=\ell_\mathrm{target}-\mathrm{mean}(\ell_{\mathrm{control},f})\).

A site class is behavior-qualified only if:

1. its predeclared primary control-family effects exceed the frozen SESOI;
2. the same-motif-unannotated contrast is positive, excluding token frequency alone;
3. balance, precision, model-health, and negative-control gates pass;
4. the corrected add-one permutation test passes over the complete site-class × contrast family;
5. the effect replicates on the untouched split with the same direction and threshold.

Failure of a positive, sensitivity, geometry, or negative-control gate produces
`INSTRUMENT_UNRESOLVED`, not a null.

### Context-margin-v1 amendment

The multisource-v4 annotation-excess result remains valid for its question, but it is not the
entry gate for context-margin v1. The amended model-native behavior is the true target score minus
the log-sum-exp of all fair Hamming-one alternative scores, normalized by the complete BPE cover.
An alternative is fair only when the full tokenizer offset map and every unmasked input token are
identical. Reference and both intervention variants must share the same nonempty alternative
intersection.

Each site class has one frozen proximal context shape. A seeded joint mononucleotide shuffle
corrupts that context without touching the target or any nucleotide in its scored cover. The null
shuffles equal-width spans 128 nt farther from the cover on the same side or sides. The pair must
preserve composition and change exactly the same number of nucleotide positions. The primary
effect is

\[
C_M=M(\text{distal equal-dose corruption})-
M(\text{class-specific proximal corruption}).
\]

The secondary true-score contrast is diagnostic only. Development and confirmation each require
at least 64 locus blocks, mean \(C_M\ge0.01\) nats/covered-nt, a positive 95% cluster-bootstrap
lower bound, and a Holm-adjusted one-sided add-one cluster sign-permutation value ≤0.05.

The distal null does not match distance to the target. Therefore a qualified result licenses
proximal-context dependence under the frozen class shape; it does not isolate a particular motif,
prove cellular function, or separate biological context from generic distance-sensitive model
encoding. Broad TF/cCRE anchors retain their explicit precision boundary.

### Promoter-classifier-v2 amendment

The promoter-specific assay uses
`magicslabnu/DNABERT-2-finetuned-prom_300_all` at revision
`7f2a5f5b8e7e6fffb2dcb396a77d1bc78b7f5f12`. The loader verifies every checkpoint file, the
vendored MosaicBERT implementation, the complete two-class head, and semantic equivalence between
the checkpoint's newer tokenizer serialization and the original pinned DNABERT-2 BPE. Remote code
is never executed.

The existing EPDnew target cohort is prohibited: all selected 300-nt windows occur verbatim in
GUE. The replacement cohort uses GENCODE v50 protein-coding MANE Select transcript TSSs, one per
shared 1-Mb locus block, after excluding exact forward or reverse-complement membership in every
GUE split. A same-locus pseudo-TSS has no annotated transcript TSS within 300 nt and must meet the
frozen GC/CpG calipers.

For an input \(x\), the supervised measurement is
\(L(x)=z_1(x)-z_0(x)\), with label 1 fixed to promoter. The clean effect is
\(L(\mathrm{TSS})-L(\mathrm{pseudoTSS})\). The local-context effect is

\[
[L(T)-L(T_{\mathrm{shuffle}})]-
[L(C)-L(C_{\mathrm{shuffle}})].
\]

The same relative 50-nt upstream and 10-nt downstream spans are shuffled around a preserved
five-nucleotide TSS or pseudo-TSS anchor. Target/control shuffles preserve length and local
mononucleotide composition and are selected to have identical Hamming dose. Clean and context
effects use SESOIs +0.50 and +0.25 logit-margin units. Each must clear its SESOI, positive 95%
locus-block bootstrap lower bound, and Holm-adjusted one-sided add-one sign-permutation gate in
development and confirmation. Failure of the context endpoint stops internal localization even
when the clean classifier behavior replicates.

## Phase D — informative source regions

For behavior-qualified targets, exchange or perturb:

- immediate 5′ and 3′ flanks;
- wider upstream context;
- wider downstream context;
- annotated paired elements;
- matched irrelevant regions.

The target nucleotides must remain unchanged. Sequence length, target token indices, full target
offset map, cover, and every target geometry field must remain identical. Non-invariant
interventions are ineligible, not adjusted after the fact.

The source-region claim is the candidate region's effect relative to equal-length irrelevant-region
exchanges, with all regions analyzed separately. A broad sequence-class effect and a
relationship-specific effect are different outcomes.

## Phase E — internal candidate atlas

Only site classes with qualified behavior and source specificity enter this phase. Candidate
generation may use:

- residual and layer patching screens;
- target-to-source attention and pre-softmax QK decomposition;
- gradients or integrated attribution;
- head-output and MLP-output attribution;
- low-rank residual directions.

DNABERT-2 MosaicBERT encoder states are unpadded. Every internal row must be mapped back through the
recorded valid-token index; an unpadded row number is never reported as a nucleotide position.
Candidate artifacts stop at their actual evidence level.

## Phase F — causality and circuit

For a frozen candidate set:

1. necessity: candidate ablation against same-size random and structured null sets;
2. output: head/MLP write and protected-health effects;
3. mediation: clean↔counterfactual activation patching and rescue;
4. information specificity: source-region-matched versus irrelevant-source patching;
5. direction: path or edge interventions;
6. faithfulness: size–faithfulness curve at declared node granularity;
7. replication: gene-disjoint sealed data and independently sampled null sets.

Null families of different intervention sizes are never pooled. Attention or attribution alone can
never populate the “Causal support” column.

## Stopping rules

- No preflight pass: stop before target outcomes.
- Raw reconstruction without either the historical empirical-excess gate or a separately frozen,
  replicated model-native behavior gate: do not localize internals.
- A context-margin result against a distal null may name proximal-context dependence but not a
  resolved biological source element.
- Context specificity without controlled causal intervention: report candidate components only.
- An anomalous control at any stage: freeze interpretation and diagnose the instrument.
- A failed hypothesis remains in `CLAIMS.md` and `DECISIONS.md`; its artifact is never rewritten.
