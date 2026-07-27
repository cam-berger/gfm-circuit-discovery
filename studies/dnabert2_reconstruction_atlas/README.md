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

The behavior-qualified atlas table remains empty after the requested multisource census.
Multisource v4 tested EPDnew promoters, PolyASite poly(A) elements, ENCODE TF-binding and distal
enhancer annotations, Mercer lariat-seq branch points, GENCODE-derived polypyrimidine tracts, and
the original splice and translation classes. All 13 classes beat order-0/1/2 sequence baselines,
but none beat all three empirical controls at the frozen effect and familywise-error gates. The
five adequately powered classes were poly(A) cleavage regions, splice acceptors, branch points,
splice donors, and stop codons. Estimates for the other eight classes are retained but
underpowered. Confirmation, context localization, and component work therefore remain sealed.
The definitions, source filters, shared locus split, and thresholds were frozen before model
outcomes; superseded outcome-blind supply audits remain preserved.

## Primary estimand

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
6. [`archive/LEGACY_DNABERT2_STUDIES.md`](archive/LEGACY_DNABERT2_STUDIES.md)
