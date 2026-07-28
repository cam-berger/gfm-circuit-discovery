# DNABERT-2 Biological Reconstruction Atlas

> **Tier: T2 — provenance.** Study overview. `CLAIMS.md` controls scientific wording.

## Central question

> Across annotated biological sequence elements, which masked elements can DNABERT-2
> reconstruct unusually well, what surrounding biological regions provide that information,
> and which internal components implement the reconstruction?

This study is a census, not a motif-first hypothesis. A site class reaches internal
reverse-engineering only after it passes all three behavioral gates:

1. reconstructability under the native MLM;
2. excess reconstructability over every declared control family;
3. a target-preserving, token-geometry-invariant context effect.

Attention, gradients, and attribution nominate components only after those gates. Matched-size
ablation, patching, rescue, mediation, and path interventions decide causal support.

## Current status

| Stage | State | What is licensed |
|---|---|---|
| P0 study architecture | implemented | The data, mask, score, control, context, gate, intervention, and artifact contracts can be tested. |
| P1 collection assembly | v4 complete: all 13 classes preflight-eligible; five behavior-powered and eight explicitly underpowered | Collection/supply facts only; no biological census claim. |
| P2 assay preflight | multisource v4 `QUALIFIED` across all 13 classes | Instrument authorization only; no behavioral claim. |
| P3 excess-reconstruction census | multisource v4 development complete; no class qualified | Five classes were adequately powered and eight remain explicitly underpowered; confirmation is sealed. |
| P4 context-source census | stopped at the P3 gate | No informative-region claim. |
| P5 internal candidate atlas | stopped at the P3 gate | No component claim. |
| P6 causal tests | stopped at the P3 gate | No causal claim. |
| Promoter classifier amendment | sealed confirmation complete | Fine-tuned promoter margin replicated on exact-GUE-excluded GENCODE MANE TSSs; the target-preserving context difference did not replicate. |
| Promoter clean-behavior head atlas | sealed confirmation complete | A development-frozen eight-head set at `[CLS]` replicated bidirectional causal patch effects against exact layer-profile-matched null sets. |
| Promoter circuit reverse engineering | sealed confirmation complete | L11H2/L11H5 is a replicated late output-carrier core; strict faithfulness requires a distributed MLP scaffold, while query-gating and early-to-late path models failed. |
| Promoter address/payload decomposition | sealed confirmation complete | L11H2/L11H5 has replicated value-payload alignment dependence; target-specific content-key addressing stayed below the frozen practical threshold. |
| Promoter source localization | sealed confirmation complete | Equal-dose masking localizes a replicated classifier-relevant source region to nominal +11 to +21 nt; no regional late-pair V effect or independent biological annotation replicated. |
| Promoter source performance follow-up | post-hoc descriptive complete | Masking +11 to +21 lowers AUROC in both splits and flips 7/231 real promoters to non-promoter, but balanced accuracy is unchanged because pseudo-TSS corrections offset target errors. |
| Promoter native-source qualification | development gate failed; confirmation and tracing sealed | Exact mono/linear-dinucleotide-preserving reordering of +11:+21 is not supported as necessary at 60 fresh loci (−0.016 logits, 95% CI −0.176 to +0.138). |
| Promoter all-node accuracy atlas | sealed confirmation complete; no node fully qualified | L11MLP ranks first in both splits (13/12-point accuracy loss; 0.214/0.283 AUROC loss), but its confirmation CI touches zero. L11H2/L11H5 remain the leading heads. |

Multisource v4 found no annotation-specific excess reconstruction over all empirical controls.
The later context-margin-v1 amendment asks a different, model-native question: whether a
preregistered proximal context supports the true target over fair Hamming-one alternatives more
than an equal-width, equal-composition, equal-change-count distal corruption. Nine classes
replicate on untouched confirmation blocks: start codon, enhancer anchor, poly(A) cleavage and
signal, branch point, polypyrimidine tract, and CEBPB/CTCF/REST peak anchors. Promoter does not
replicate, stop codon fails development, and splice acceptor/donor are preflight-unresolved.

These are DNABERT-2 context-behavior rows. They do not establish biological function in cells,
resolved motifs for the broad regulatory anchors, or internal causal mechanisms. The distal null
does not match target distance, so proximity-sensitive encoding remains inside the licensed
effect. All prior negative and failed artifacts remain preserved.

The promoter-specific amendment replaces the MLM readout with the supervised class margin from
`magicslabnu/DNABERT-2-finetuned-prom_300_all`. The obvious EPDnew rerun was rejected: all 768
300-nt EPDnew atlas windows occur verbatim in GUE, the fine-tuning source. On a replacement cohort
of 384 development and 384 confirmation GENCODE v50 MANE TSS locus blocks with exact
forward/reverse GUE exclusion, the clean promoter margin replicates against composition-matched
same-locus pseudo-TSSs. The stricter target-preserving context difference qualifies in development
but not confirmation. D4 therefore licenses promoter classification behavior, not a
context-specific mechanism.

A separately frozen head atlas localizes only D4's replicated clean contrast. It scans all
144 attention heads at the unpadded `[CLS]` destination, ranks development candidates using
gradient agreement with exact paired head-output patches, and tests the eight-head group against
63 candidate-excluding random sets with identical per-layer head counts. The development-frozen
set L2H2/L4H6/L4H7/L9H7/L10H1/L10H5/L11H2/L11H5 passes both necessity and sufficiency gates on
192 development and 192 untouched confirmation locus blocks. D5 supports candidate-set CLS
head-output mediation. It does not establish individual-head specificity, causal source
positions, QK/OV roles, a directed circuit, or the failed context hypothesis.

The next frozen circuit battery uses the 192 head-atlas-unused loci per split. Against every
candidate-excluding pair in layer 11, L11H2/L11H5 replicates exact necessity and sufficiency.
Strict complement scrub shows that the pair and the full eight-head set retain only about
14–15% of the clean contrast on their own; retaining all MLP writes with the eight heads raises
this to about 61% in both splits. Query-only transfer is effectively zero, and corrupting the six
earlier candidate heads is below the frozen effect threshold. D6 therefore supports a late
output-carrier core on an MLP-maintained representation, not a target-specific query gate or a
demonstrated early-to-late chain.

An accuracy-first follow-up scans all 144 head contexts and all 12 MLP writes on the final 100
unused locus blocks. Reciprocal exchange of L11MLP is the largest effect in both 50-pair splits:
balanced accuracy falls by 0.130/0.120 and AUROC by 0.214/0.283. L11H2 is the top head in both
splits, followed closely by L11H5/L11H6. L11MLP clears the SESOI and full-family maximum-sign
test twice, but its confirmation bootstrap interval begins at zero; no node passes the complete
replicated gate. This prioritizes the last-layer MLP for targeted reverse engineering without
promoting it to a standalone necessary circuit element.

A fresh-cohort address/payload assay then separates the late pair's K and V roles at fixed
40-token dose. Permuting V while leaving Q, K, QK logits, and native attention probabilities
unchanged causes a target-specific margin loss of +0.1026 in development and +0.0615 in
confirmation; both beat all 45 equal-size layer-11 null pairs. K permutation is only
+0.0073/+0.0084, below the 0.05-logit practical threshold. D7 therefore supports value/source-state
payload routing, while the stable residual direction remains unlabeled and causal source bands
or motifs remain untested.

The next frozen phase scans thirty nonoverlapping 10-nt nominal tiles without biological labels.
Each intervention uses exactly two adjacent genomic BPE tokens on both the real-TSS and pseudo-TSS
branches; actual covers are retained because BPE boundaries can overhang the nominal tile. One
behavioral region replicates: nominal +11 to +21 relative to the TSS. Its target-specific mask loss
is +0.245 in development (95% CI +0.076 to +0.428; familywise `p=0.026`) and +0.206 in confirmation
(+0.045 to +0.392; `p=0.0082`). The actual cover length has median 10 nt and ranges from 6 to
15 nt across the two splits.

This D8 region is not yet biologically labeled. A PLS-overlapping subset qualifies only in
development and misses confirmation; CAGE is underpowered at this downstream tile, CpG does not
qualify, and no predefined EPD TATA/Inr/CCAAT/GC-box instance intersects it. The regional
L11H2/L11H5 V-swap scan also stays below both frozen effect thresholds at every tile. D8 therefore
supports a replicated **classifier-relevant source region**, not a canonical motif, a source for
the D7 payload, mediation, or endogenous promoter function.

A user-requested post-hoc follow-up reruns the unchanged tile-26 mask while retaining each clean
and masked decision margin. On the balanced real-TSS/pseudo-TSS challenge cohort, AUROC falls
from 0.678 to 0.644 in development (change −0.033, bootstrap 95% CI −0.060 to −0.007) and from
0.679 to 0.658 in confirmation (−0.021, −0.043 to −0.002). Three of 107 and four of 124 real
promoters cross from promoter to non-promoter. Balanced accuracy changes by −0.9 and +0.8
percentage points, respectively, with both intervals spanning zero; six pseudo-TSS false
positives also become correct negatives, offsetting the target errors. This is performance on the
fresh GENCODE/pseudo-TSS challenge, not the original GUE held-out test set.

Native-base qualification then replaced `[MASK]` with a nonidentity reorder of +11:+21 that
preserves exact mononucleotide and linear-dinucleotide counts. Target and pseudo-TSS edits have
matched Hamming dose and locally matched BPE retokenization geometry. After two preserved
precision failures, an outcome-blind 60-locus design passed a third disjoint GUE preflight.
Development failed the positive gate: target-specific margin loss was −0.016 logits (95% CI −0.176 to +0.138,
add-one `p=0.5834`), accuracy was unchanged, and one target flipped promoter→non-promoter.
Confirmation, residual tracing, and head mediation therefore remain sealed. GC-content necessity
is separately instrument-unresolved because its preflight variance exceeded the frozen limit.

## Historical reconstruction estimand

For annotated interval \(S=[a,b)\), tokenize the complete sequence once and select the minimal
consecutive non-special BPE tokens whose nucleotide offsets overlap \(S\). Replace those token IDs
with `[MASK]` without retokenizing. The primary score is

\[
\ell(S)=\frac{\sum_{j\in C(S)}\log p_\theta(t_j\mid x_{\setminus C(S)})}
{|\,\mathrm{nt\_cover}(C(S))\,|}.
\]

This is a simultaneous-mask pseudo-log-likelihood. It is normalized by the complete BPE cover,
including unavoidable overhang, not by only the annotated bases. Alternative-sequence margins are
eligible only when full offset geometry and every unmasked token ID are identical.

The primary biological quantity is not \(\ell(S)\), but its within-match advantage over each
control family separately:

- unannotated occurrences of the same sequence;
- composition-, covered-length-, and token-count-matched genomic sites;
- matched non-sites in the same sequence;
- held-out order-0/1/2 nucleotide baselines.

Control families are not pooled.

## Context-margin-v1 estimand

For each target, retain every Hamming-one nucleotide alternative whose edited sequence has the
identical full BPE offset map and identical unmasked token IDs. The raw alternative margin is

\[
M(S)=\frac{s_\mathrm{true}-\operatorname{logsumexp}_{a\in A(S)}s_a}
{|\mathrm{nt\_cover}(S)|}.
\]

The same nonempty alternative intersection scores reference, proximal-context-shuffled, and
distal-context-shuffled variants. Both shuffles leave the target and full scored cover unchanged,
preserve mononucleotide composition, and change exactly the same number of positions. The primary
effect is \(M_\mathrm{distal}-M_\mathrm{proximal}\). A class must exceed +0.01, have a positive
95% cluster-bootstrap lower bound, and pass add-one cluster sign inference with Holm correction
in development and confirmation.

## Promoter-classifier-v2 estimand

For a 300-nt window from −249 through +50 around a TSS, the model-native supervised margin is

\[
L(x)=z_\mathrm{promoter}(x)-z_\mathrm{nonpromoter}(x).
\]

This margin is BPE-valid because both logits come from one classification head on the same
tokenized input. The clean effect is \(L(\mathrm{TSS})-L(\mathrm{pseudoTSS})\). The context
effect is a difference-in-differences: clean-minus-shuffled margin loss at the true TSS minus the
same loss at its same-locus pseudo-TSS. Both shuffles preserve a five-nucleotide anchor,
mononucleotide composition, sequence length, and exact target/control Hamming dose. Each split
requires the frozen SESOI, a positive 95% locus-block bootstrap lower bound, and Holm-adjusted
add-one sign inference for both endpoints. Exact GUE membership is excluded before scoring.

## Promoter clean-behavior head-atlas estimand

For one head \(h\), let \(c_h(x)\) be its attention context slice at the unpadded `[CLS]` row
immediately before the shared attention output projection. For paired target \(t\) and pseudo-TSS
control \(u\), exact necessity is

\[
N_h=L(t)-L\!\left(t;c_h(t)\leftarrow c_h(u)\right),
\]

and exact sufficiency is

\[
S_h=L\!\left(u;c_h(u)\leftarrow c_h(t)\right)-L(u).
\]

Attribution nominates heads with the corresponding first-order
gradient–activation-difference dot product. Confirmation freezes the eight development-selected
heads unchanged. The joint set must exceed +0.60 logits in both directions, have positive
95% locus-block-bootstrap lower bounds, and beat 63 distinct candidate-excluding null sets with
the exact same layer profile under two-endpoint Holm correction. Attention source bands are
reported descriptively and cannot establish source-position causality.

## Promoter source-localization estimand

For each nominal 10-nt tile \(R\), choose two adjacent genomic BPE tokens per sequence by maximum
nucleotide overlap, subject to an 18-nt cover ceiling and at most two-nucleotide target/control
cover-length imbalance. The behavioral measurement is

\[
\Delta_{\mathrm{mask}}(R)
=\bigl[L(t)-L(t_{\mathrm{mask}(R)})\bigr]
-\bigl[L(u)-L(u_{\mathrm{mask}(R)})\bigr].
\]

Localization additionally requires the per-locus effect to exceed that locus's mean effect over
its other eligible equal-dose tiles. A joint maximum sign-flip null corrects over all 30 tiles and
both mask and regional-V operations. The V operation swaps only the two selected value vectors for
L11H2/L11H5, leaving Q, K, QK logits, ALiBi, and native attention probabilities unchanged.

Biological qualification is separate. FANTOM5 dominant CTSSs, ENCODE PLS cCREs, UCSC CpG islands,
and EPD motif instances are never pooled. Motif interventions must remove the published EPD call,
preserve each branch's local mononucleotide composition, match target/control Hamming dose, retain
the complete BPE offset map, and leave every token outside the affected cover unchanged.

## Promoter native-source estimand

For the nominated +11:+21 interval, construct a nonidentity nucleotide reorder with exact
mononucleotide and linear-dinucleotide counts. Apply an equal Hamming dose to the paired target and
pseudo-TSS, and admit only pairs whose changed BPE segment is local and matched in token-count
delta, affected-token count, and affected-cover length. The primary endpoint is

\[
\Delta_{\mathrm{native}}
= [L(t)-L(t_{\mathrm{edit}})]-[L(u)-L(u_{\mathrm{edit}})].
\]

Development requires mean \(\Delta_{\mathrm{native}}\ge0.10\) logits, a positive 95% locus-block
bootstrap lower bound, and one-sided add-one cluster-sign `p≤0.05`. Confirmation opens only for
an unchanged qualified operation. Failure seals residual-source tracing and mediation.

## Repository map

| Path | Purpose |
|---|---|
| `src/gfmcd/adapters/dnabert2/` | Pinned checkpoint loader, token/output access, MosaicBERT unpadding map, named hook points |
| `src/gfmcd/reconstruction_atlas/` | Validated schemas, BPE masking, scoring, controls, baselines, context exchanges, gates, statistics, causal records, immutable artifacts |
| `scripts/dnabert2_atlas/` | Collection validation/build and phase-gated behavioral runner |
| `configs/dnabert2_atlas/` | Versioned site catalogs and frozen preflight locks |
| `studies/dnabert2_reconstruction_atlas/` | Protocol, claims, decisions, results, source catalog, and archive index |
| `results/dnabert2_reconstruction_atlas/` | Settled compact artifacts and manifests only; bulk shards remain ignored |

## Execution

Model-bearing runs use the isolated pinned environment:

```bash
uv run --isolated --no-project --python 3.11 \
  --with-requirements studies/dnabert2_reconstruction_atlas/requirements-dnabert2.txt \
  --with-editable . \
  python scripts/dnabert2_atlas/<runner>.py ...
```

Every producing command requires an explicit new `--out`; no runner defaults to a scientific
record. `run_behavioral_screen.py` refuses to run unless a frozen preflight report has all mandatory
gates at `PASS` and matches the exact config and checkpoint hashes.

For multisource v4, `build_multisource_collection.py` also emits the conservative annotation BED
used to prevent source-overlapping sites from serving as “unannotated” controls. Confirmation can
open only the classes that qualified in the immutable development report.

## Read next

1. [`PROTOCOL.md`](PROTOCOL.md)
2. [`DATA_CATALOG.md`](DATA_CATALOG.md)
3. [`PREREGISTRATION.md`](PREREGISTRATION.md)
4. [`CLAIMS.md`](CLAIMS.md)
5. [`RESULTS.md`](RESULTS.md)
6. [`PROMOTER_CLASSIFIER_SOURCE_MANIFEST.json`](PROMOTER_CLASSIFIER_SOURCE_MANIFEST.json)
7. [`SOURCE_MANIFEST_PROMOTER_LOCALIZATION_V1.json`](SOURCE_MANIFEST_PROMOTER_LOCALIZATION_V1.json)
8. [`archive/LEGACY_DNABERT2_STUDIES.md`](archive/LEGACY_DNABERT2_STUDIES.md)
