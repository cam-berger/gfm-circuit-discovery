# Figures and provenance

Each figure is accompanied by a `.provenance.json` file identifying its source documents or recorded
artifacts and their SHA-256 hashes. Claim and correction ledgers control interpretation; a visualization
does not promote observational evidence to a causal claim.

| Figure | Purpose and scope | Sources |
|---|---|---|
| [Workflow](experiment_workflow.png) | Conceptual overview of behavior, intervention, causal-role testing, and bounded hypotheses; no quantitative result. | [Provenance](experiment_workflow.provenance.json) |
| [SpliceBERT headline](headline.png) | Paired-acceptor coupling, positional spectra, phase-surgery effects, and a separate fixed-span chimeric register result. The phase-surgery and register panels do not establish a mechanistic bridge between them. | [Provenance](headline.provenance.json) |
| [Positional and faithfulness results](fig_3.png) | Period-3 use, the address/payload factorial, the static-ranking faithfulness curve, and the two-dimensional Fourier plane. No experiment connects the period-3 carrier to biological reading frame. | [Provenance](fig_3.provenance.json) |
| [DNABERT-2 comparison](paper_figure_workflow_transfer.png) | A bounded workflow comparison with a conditional, MLP-heavy causal core. | [Provenance](paper_figure_workflow_transfer.provenance.json) |

The [study index](../studies/README.md) links to each scientific record. The manuscript audit's
[figure discussion](../RESULTS.md#6-figure-audit) describes the assets present at its stated audit date.

## Reading the mechanism figure

The [positional and faithfulness figure](fig_3.png) uses these panel-specific conventions:

- **A — Period-3 use:** points are recorded mean coupling values; bars show ±1 gene standard error,
  not 95% confidence intervals. Straight segments connect measured shifts. This is positional-phase
  surgery, not a test of biological reading frame.
- **B — Address and payload:** values are normalized recovery fractions, not raw logit units. Intervals
  are those stored in the factorial artifact; no interval is supplied for the interaction estimate.
- **C — Faithfulness:** the 90% crossing is **64.2 nodes by linear interpolation**; **72/103** is the
  first sampled point above 90%. The static ranking does not establish a minimal circuit.
- **D — Fourier plane:** only the two recorded singular values are plotted. A single-frequency
  sine/cosine reconstruction has rank at most two by construction; this is not independent evidence
  for a learned low-rank causal mechanism or a measured ten-component energy spectrum.

## Checking and regenerating figures

Run the source-hash and sidecar checks in the project environment:

```bash
uv run --locked --extra splicebert --extra biology --extra dev pytest \
  tests/test_figure_provenance.py tests/test_figure_provenance_freshness.py
```

These checks verify that cited source bytes still match the sidecars. They do not independently
validate a plot's arithmetic, labels, or interpretation. Review those against the controlling artifacts
and ledgers before using a figure in a manuscript.

| Rendered asset | Available renderer |
|---|---|
| Positional and faithfulness figure | [make_mechanism_figure.py](../scripts/splicebert/make_mechanism_figure.py) |
| DNABERT-2 comparison | [make_paper_figure_transfer.py](../scripts/dnabert_promoter/make_paper_figure_transfer.py) |

To regenerate the committed mechanism figure from its recorded sources:

```bash
uv run --locked python scripts/splicebert/make_mechanism_figure.py --overwrite
```

The renderer writes PNG, SVG, and provenance together. To inspect a separate rendering, use
`--out /tmp/gfmcd-figures/fig_3.png` without `--overwrite`.

Regenerate an affected figure from its sources when those sources change, and inspect the rendering
and provenance together. Source sidecars for the workflow and headline assets are retained, but an
in-tree renderer for their current layouts is not available. Historical figure names and paths in
immutable manifests remain part of the execution record.
