# DNABERT-2 Reconstruction Atlas claim ledger

> **Tier: T1 — canonical for this study's claim wording.**

## Live scientific claims

**None.** No atlas preflight, behavioral census, context experiment, or causal intervention has
been run on this branch.

## Established infrastructure facts

These are implementation properties and prior instrument constraints, not claims about biological
reconstruction.

| ID | Fact | Evidence | Boundary |
|---|---|---|---|
| IF1 | The adapter pins DNABERT-2-117M checkpoint, tokenizer, config, and vendored MosaicBERT code by revision and SHA-256; an incomplete MLM head is a hard failure. | `src/gfmcd/adapters/dnabert2/` and adapter tests | Does not show the model reconstructs any biological element. |
| IF2 | Atlas masking selects the minimal BPE token cover from nucleotide offsets and masks token IDs without retokenizing. | `test_atlas_tokenization.py` | An assay invariant, not model behavior. |
| IF3 | Alternative margins are rejected unless full offsets and all unmasked token IDs remain identical. | `test_atlas_tokenization.py` | Does not guarantee adequate alternative supply. |
| IF4 | Context exchanges are ineligible if they change target nucleotides, length, target token indices, offsets, or cover. | `test_atlas_context.py` | Does not establish any informative region. |
| IF5 | Causal specs reject null sets whose component count differs from the candidate intervention. | `test_atlas_mechanism.py` | Does not establish necessity or mediation. |

## Open hypotheses

| ID | Hypothesis | Current status | First required bridge |
|---|---|---|---|
| H1 | At least one annotated site class has positive excess reconstruction over every required control. | proposed | Frozen, qualified preflight followed by development and sealed behavior runs. |
| H2 | For at least one H1-qualified class, a biologically related context region outperforms matched irrelevant-region exchanges. | proposed | H1 plus token-geometry-invariant context interventions. |
| H3 | A bounded set of heads, MLPs, positions, or residual directions causally supports an H2-qualified behavior. | proposed | Frozen candidate screen, matched-size necessity, mediation, and path tests. |

Negative, retracted, and superseded rows will remain here permanently once evidence exists.
