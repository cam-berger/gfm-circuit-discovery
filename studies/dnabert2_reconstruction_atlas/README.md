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
| P1 collection assembly | bounded GENCODE v50 slice qualified for four classes; Kozak supply quarantined; other sources pending | Collection/supply facts only; no biological census claim. |
| P2 assay preflight | protocol/config drafted; not frozen or run | No behavioral claim. |
| P3 excess-reconstruction census | blocked on qualified preflight | No atlas ranking. |
| P4 context-source census | blocked on P3 | No informative-region claim. |
| P5 internal candidate atlas | blocked on P4 | No component claim. |
| P6 causal tests | blocked on frozen candidates | No causal claim. |

The correct current biological atlas table is therefore empty. This is not a negative result:
the target experiment has not run. The Phase-A supply record admits splice donor, splice
acceptor, start codon, and stop codon to calibration; its Kozak quarantine is an instrument
boundary, not a DNABERT-2 reconstruction result.

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
| `configs/dnabert2_atlas/` | Versioned site catalog and preflight lock candidate |
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

## Read next

1. [`PROTOCOL.md`](PROTOCOL.md)
2. [`DATA_CATALOG.md`](DATA_CATALOG.md)
3. [`PREREGISTRATION.md`](PREREGISTRATION.md)
4. [`CLAIMS.md`](CLAIMS.md)
5. [`RESULTS.md`](RESULTS.md)
6. [`archive/LEGACY_DNABERT2_STUDIES.md`](archive/LEGACY_DNABERT2_STUDIES.md)
