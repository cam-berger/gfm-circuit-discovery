# `tex/` — NeurIPS 2026 paper draft

> Not authoritative for any claim. Where this directory and a claim ledger disagree, **the ledger wins**
> and the paper is wrong.

## Canonical draft

[`splicebert_neurips_2026.tex`](splicebert_neurips_2026.tex) is the current anonymous NeurIPS 2026
main-track manuscript, **“Mechanistic Discovery in Genomic Foundation Models.”** The source and its author-kit
dependencies were imported unchanged from the supplied NeurIPS bundle. The older
[`neurips_2026_gfm_causal_discovery.tex`](neurips_2026_gfm_causal_discovery.tex) is a superseded predecessor
and is not a build or editing target.

The paper asks whether causal interpretability can turn a regularity learned by a genomic foundation model
into an experimentally distinguishable discovery hypothesis. SpliceBERT is the worked case study. The paper
is organized around four steps:

1. establish a selective, model-native paired-boundary behavior on sealed gene-disjoint data;
2. resolve an ordered **address → payload → write → delivery** computation with nested interventions;
3. use cross-gene transfer and complement scrubbing to delimit what that account explains; and
4. use two still-unconnected positional findings to motivate a testable positional-scaffold/content-gating
   hypothesis and the experiments that could distinguish it from alternatives.

The intended contribution is methodological and hypothesis-generating. The draft does **not** claim that
SpliceBERT predicts cellular splicing, that the acceptor-local payload has a known biological identity, that
the compact causal core is minimal or autonomously faithful, that period-3 geometry implements annotated
coding-register preference, or that the SpliceBERT mechanism transfers wholesale to another GFM.

## Author-kit contents

The following files are the intact payload of the supplied NeurIPS 2026 bundle. Finder metadata from
`__MACOSX/` is intentionally excluded.

The import baseline is `Formatting_Instructions_For_NeurIPS_2026-2.zip`, SHA-256
`38fd47e787451caef89c368bfa172b6f827622d13bc16bac5bca610eb4074937`. Every listed payload matched its
archive member byte for byte at import. Later manuscript edits should be ordinary reviewed changes and
must not be described as bundle-identical.

| File | Role |
|---|---|
| `splicebert_neurips_2026.tex` | Canonical anonymous main-track draft and appendices. |
| `fig_headline.png` | Headline paired-boundary and positional-content figure. |
| `figure_2.png` | Positional-results and scaffold-hypothesis figure. |
| `refs.bib` | Bibliography used by the canonical draft. |
| `checklist.tex` | Official NeurIPS paper checklist, included at the end of the manuscript. |
| `neurips_2026.sty` | NeurIPS 2026 formatting package supplied with the author kit. |
| `neurips_2026.tex` | Official author-kit example and formatting instructions. |

These third-party formatting files remain subject to their upstream terms; tracking them here does not
relicense them under the repository’s MIT license. See
[`docs/THIRD_PARTY_ASSETS.md`](../docs/THIRD_PARTY_ASSETS.md).

## Draft state

The imported draft is current but not submission-ready:

- the reciprocal-acceptor paragraph and DNABERT-2 comparison each end in a literal `TODO`;
- the official checklist still contains its unanswered `[TODO]` fields; and
- the acceptor-context paragraph describes branch-local source-audit and cohort-builder work that is not
  included as a checkable implementation or registered artifact in this tree; only its draft protocol is
  preserved here;
- the anonymous appendix names `origin/padmaja/dnabert-clean-pipeline`, which requires an anonymity review
  before submission;
- scientific language must continue to be checked against the live ledgers and
  [`docs/PREREG_CONFORMANCE.md`](../docs/PREREG_CONFORMANCE.md).

Those markers are retained because the bundle is the source of truth for this import. Resolve them in a
subsequent manuscript-editing change; do not silently substitute text from an older branch draft.

## Figure provenance status

The imported PNGs are the figures used by the current draft, but neither currently has an in-repository
generator or provenance sidecar. Their scientific panels should be checked against these records before
submission:

- `fig_headline.png` draws on the sealed paired-acceptor behavior and matched stage-1 carrier-origin
  comparison; its curves and attention intensity are explicitly schematic in the manuscript caption.
- `figure_2.png` combines C36's phase-aligned period-3 intervention with C43's fixed-span chimeric
  register preference; its bridge panel is a hypothesis, and C44 supplies no mediation result.

Until source declarations or generators are added, these bitmaps do not satisfy the repository's normal
generated-figure provenance standard. This is a packaging limitation, not new scientific evidence.

## Building

Run the standard BibTeX sequence from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error splicebert_neurips_2026.tex
bibtex splicebert_neurips_2026
pdflatex -interaction=nonstopmode -halt-on-error splicebert_neurips_2026.tex
pdflatex -interaction=nonstopmode -halt-on-error splicebert_neurips_2026.tex
```

The submission source uses `\usepackage[main]{neurips_2026}` for anonymous main-track review. Do not enable
`final` or `preprint` for the review PDF. LaTeX products (`.aux`, `.bbl`, `.blg`, `.log`, `.out`, and `.pdf`)
are ignored and should not be committed.

## Claim authority and editing guardrails

Use these records when checking or revising scientific prose:

1. [`studies/splicebert_partner_selection/CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md)
   and its [`CORRECTIONS.md`](../studies/splicebert_partner_selection/CORRECTIONS.md) control the donor
   mechanism, period-3 carrier, transfer, faithfulness, and coding-register claims.
2. [`MECHANISM.md`](../MECHANISM.md) gives the current one-file mechanism and limitation narrative.
3. [`docs/EVIDENCE_LADDER.md`](../docs/EVIDENCE_LADDER.md) controls promotion between observation,
   intervention, mediation, directed-circuit, and biological claims.
4. [`docs/PREREG_CONFORMANCE.md`](../docs/PREREG_CONFORMANCE.md) controls whether a preregistered gate may
   be described as enforced.
5. The claim-to-artifact map in the manuscript appendix identifies the compact results behind selected
   statements; it is not a replacement for the ledgers.

Keep these boundaries explicit in every revision:

- masked-token prediction is a model behavior, not an RNA-processing measurement;
- position-specific transfer is established, but content specificity is not;
- the final additive donor effect is captured by a gene-generic support axis in the tested regime;
- the period-3 intervention and register-matching assay are separate results without a licensed causal
  bridge;
- reciprocal acceptor coupling is a validation authorization gate, not a reverse mechanism; and
- the DNABERT-2 material is a bounded historical probe study, not a live cross-model replication.
