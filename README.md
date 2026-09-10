# Mechanistic Discovery in Genomic Foundation Models

Methods, code, and evidence for studying what genomic foundation models compute. The workflow starts
with a biological computation and a measurable model behavior, then uses controlled interventions to
identify the information, components, and paths that support it. SpliceBERT is the main worked example.

[Methodology](docs/METHODOLOGY.md) · [Mechanism](MECHANISM.md) · [Study index](studies/README.md) ·
[Reproduction guide](REPRODUCE.md)

![Conceptual workflow: define model behavior, intervene on internal states, test causal roles, and formulate bounded hypotheses.](figures/experiment_workflow.png)

*Conceptual overview; the full [seven-stage method](docs/METHODOLOGY.md#the-seven-stage-workflow)
specifies assay preflight, controls, and replication. [Figure sources](figures/experiment_workflow.provenance.json).*

## Main result

On 400 canonical short-intron examples from the sealed chromosome 8/9 test split, replacing the paired
acceptor's `AG` with `CC` lowers the masked-donor margin. The mean clean-minus-corrupt difference is
**0.872 logit units** (95% CI **[0.771, 0.969]**); matched non-partner controls are near zero.
The margin is `logit(G) − max(logit(A), logit(C), logit(T))` at the masked donor `G`.
See [C1–C2](studies/splicebert_partner_selection/CLAIMS.md#1-the-behaviour-and-its-specificity) and the
[recorded result](results/splicebert_coupling/sealed_test.json).

Causal interventions resolve this behavior into four roles:

| Role | Supported account |
|---|---|
| **Address** | L3H1 selects the paired acceptor mainly through positional information; L2H15 strengthens the attention-side signal. |
| **Payload** | Earlier heads and MLPs construct causally relevant values in a four-token acceptor window. Their biological meaning remains unresolved. |
| **Write** | L3H1 combines address and payload into a donor-local state captured well by a tested rank-8 subspace. |
| **Delivery** | Later heads, including L4H0, L4H2, and L5H11, carry that state toward the masked-token output through their value-output paths. |

These components form a structured, top-ranked causal core inside a broader distributed computation. They
are not a complete, minimal, or autonomous circuit. The held-out static-ranking curve has no sparse knee,
but it does not prove that no smaller alternative circuit exists.

The learned period-3 positional direction and the boundary-register preference seen in fixed-span chimeras
are separate findings. No current experiment connects them, identifies reading frame, or establishes a
cellular splicing mechanism.

Reciprocal SpliceBERT experiments provide validation behavior and partial component evidence. DNABERT-2
promoter experiments provide a bounded comparison with a different, MLP-heavy conditional causal core.
Neither establishes a replication or symmetry of the donor mechanism. The
[study index](studies/README.md) also covers context studies, fine-tuning comparisons, and stopped
experiments.

## Reading the evidence

The claim ledgers and immutable result artifacts control the scientific record. Narrative summaries do not
override them.

| To inspect… | Start with… |
|---|---|
| The mechanism and its limits | [MECHANISM.md](MECHANISM.md) |
| Individual donor claims and corrections | [Claim ledger](studies/splicebert_partner_selection/CLAIMS.md) and [correction ledger](studies/splicebert_partner_selection/CORRECTIONS.md) |
| Quantitative results and the manuscript audit | [RESULTS.md](RESULTS.md), dated 2026-08-22 |
| Evidence from each study | [Study index](studies/README.md) and [artifact registry](results/MANIFEST.csv) |
| Methods, evidence levels, and protocol deviations | [Documentation index](docs/README.md) |
| Figures and their sources | [Figure index](figures/README.md) |

Attention, attribution, probes, and spectra nominate candidates. Intervention establishes causal use. A
causal model mechanism is not automatically a biological mechanism.

## Reproduce and validate

Validate the artifact registry without model weights or third-party Python packages:

```bash
python3 scripts/verify_manifest.py
```

Install the locked environment and run the test suite:

```bash
uv sync --locked --extra splicebert --extra biology --extra dev
uv run --locked --extra splicebert --extra biology --extra dev pytest
```

The full provenance checks require Git history and archive tags. Model checkpoints, hg38 FASTA, and
GENCODE annotation are obtained separately. Tests requiring unavailable assets report skips;
passing repository checks does not constitute a rerun of the model experiments. See
[REPRODUCE.md](REPRODUCE.md) for dependencies, data, commands, and limits on exact reproduction.

## Repository layout

| Directory | Contents |
|---|---|
| [`src/gfmcd/`](src/gfmcd/) | Model adapters, reusable circuit tools, and study assays |
| [`scripts/`](scripts/README.md) | Experiment, audit, and figure entry points by study |
| [`studies/`](studies/README.md) | Claim ledgers, protocols, decisions, and study guides |
| [`results/`](results/) | Immutable results, manifests, and audits |
| [`figures/`](figures/README.md) | Publication figures and source provenance |
| [`tests/`](tests/) | Analysis tests, evidence-integrity checks, and regression guards |
| [`docs/`](docs/README.md) | Methodology, evidence standards, and terminology |

For changes or a new model study, see [CONTRIBUTING.md](CONTRIBUTING.md). The live manuscript source is
maintained separately; [RESULTS.md](RESULTS.md) audits a fingerprinted draft and records its outstanding
corrections.

Repository code is available under the [MIT License](LICENSE). External model and data terms are listed in
[`docs/THIRD_PARTY_ASSETS.md`](docs/THIRD_PARTY_ASSETS.md).
