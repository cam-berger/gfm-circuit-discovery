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
| P1 collection assembly | multisource v2 seven-nt interval cores supply-quarantined; v3 five-nt amendment frozen for 13 classes | Collection/supply facts only; no biological census claim. |
| P2 assay preflight | v1 `QUALIFIED`; multisource v3 preflight not yet run | The v1 authorization does not authorize v3 target outcomes. |
| P3 excess-reconstruction census | narrow v1 development complete; multisource v3 not yet scored | The v1 nonqualification does not generalize to the new classes. |
| P4 context-source census | blocked on P3 | No informative-region claim. |
| P5 internal candidate atlas | blocked on P4 | No component claim. |
| P6 causal tests | blocked on frozen candidates | No causal claim. |

The existing behavior-qualified atlas table is empty because census v1 was a narrow four-class
GENCODE pilot. It does not answer the requested atlas question. Multisource v3 adds EPDnew
promoters, PolyASite poly(A) elements, ENCODE TF-binding and distal enhancer annotations, Mercer
lariat-seq branch points, and GENCODE-derived polypyrimidine tracts while retaining the original
splice and translation classes. Its definitions, source filters, shared locus split, and thresholds
are frozen before v3 model outcomes. Its superseded seven-nt v2 design and failed outcome-blind
control-supply audit remain preserved.

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

For multisource v3, `build_multisource_collection.py` also emits the conservative annotation BED
used to prevent source-overlapping sites from serving as “unannotated” controls. Confirmation can
open only the classes that qualified in the immutable development report.

## Read next

1. [`PROTOCOL.md`](PROTOCOL.md)
2. [`DATA_CATALOG.md`](DATA_CATALOG.md)
3. [`PREREGISTRATION.md`](PREREGISTRATION.md)
4. [`CLAIMS.md`](CLAIMS.md)
5. [`RESULTS.md`](RESULTS.md)
6. [`archive/LEGACY_DNABERT2_STUDIES.md`](archive/LEGACY_DNABERT2_STUDIES.md)
