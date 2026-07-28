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
context-specific mechanism or internal circuit search.

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
7. [`archive/LEGACY_DNABERT2_STUDIES.md`](archive/LEGACY_DNABERT2_STUDIES.md)
