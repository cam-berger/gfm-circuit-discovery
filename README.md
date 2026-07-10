# Circuit Discovery in Genomic Foundation Models

Tracing how a training objective installs a latent representation of genome structure.

Built from scratch: none of our three model families (planted, GFM-nano, SpliceBERT) are supported
by TransformerLens / `circuit-tracer`, which target GPT-style decoder-only models. See
`docs/DECISIONS.md#d5`.

## Where to start

| | |
|---|---|
| `docs/EXPERIMENTS.md` | **Plain-language account of every experiment and what it showed.** Start here. |
| **`RESULTS.md`** | **Every measured number, with status: SETTLED / PENDING / WITHDRAWN / RETRACTED.** §10 is the retraction record. |
| **`REC.md`** | The argument: critique of Proposal D2, and the sufficiency criterion that replaces it. |
| `Proposal_Sufficiency_Circuit_Discovery.md` | The PI-facing writeup. |
| `RELAY_TRACE.md` | Phase C: what actually separates a counter from a relay from a bag. |
| `docs/DECISIONS.md` | Why the design is what it is, with the numbers. D1–D35. |
| `docs/PREREGISTRATION.md` | Frozen predictions, registered before each measurement. |
| `scripts/README.md` | Which script produces which number. |

## Repository map

```
README.md                       you are here
RESULTS.md   REC.md             the ledger, and the argument
RELAY_TRACE.md                  Phase C deliverable (reading-frame track)
Proposal_Sufficiency_*.md       PI-facing writeup
PLAN_reading_frame.md           active plan, track 1
PLAN_repeat_copy.md             active plan, track 2

docs/
  DECISIONS.md                  D1-D35, the reasoning with the numbers
  PREREGISTRATION.md            frozen before each measurement
  EXPERIMENTS.md                plain-language account
  PLAN.md                       phases and gates, across both tracks
  explainers/                   standalone HTML for non-specialists
  archive/                      superseded prose, kept not deleted

src/gfmcd/                      the library: models, data, tracer, oracle, signatures
scripts/                        one script per measurement -- see scripts/README.md
tests/                          113 tests; Rung 0 is ground truth, not a smoke test

results/reading-frame/
  fig_data/                     the measured numbers -- TRACKED, so a retraction leaves a diff
  fig_data/retracted/           dumps whose measurement was withdrawn
  figures/                      gitignored; regenerate with scripts/fig_phase_*.py
branches/codex-repeat-induction/  track 2's findings and figures

data/cache/                     window caches, gitignored: the filename is the recipe
checkpoints/                    gitignored
logs/archive/                   completed run logs
```

Two tracks share this repo and share no files: **reading frame** (this document's ladder) and
**repeat-copy / induction** (`PLAN_repeat_copy.md`).

## The ladder

```
Rung 0  planted twins    edge set known BY CONSTRUCTION   -> tracer validation   [gate]
Rung 1  reading frame    period-3, anchored to an ATG     -> proof of method
Rung 2  splice sites     local composition, GENCODE       -> first real circuit
Rung 3  G-quadruplex     cross-strand, loop-window        -> flagship
H2      objective arms   AR < MLM < MLM+phyloP < phyloP   -> the payload
```

Rung 0 exists because a trained model gives you a known *task*, not a known *edge set*. Two planted
twins — one compositional, one a lookup table — have identical behaviour and different mechanism:

```
                    accuracy    after marker ablation
compositional         0.586            0.383
heuristic             0.583            0.583
                    |Δ| 0.0034
```

The tracer must find a circuit in one and fail in the other. A positive control alone cannot falsify
a tracer that reports compact circuits on everything.

## Quickstart

```bash
conda activate gfm-thrust1
pip install -e ".[dev]"
pytest -q
```

## Firewalls

`config.py` raises rather than warns:

- `assert_holdout_valid("splicebert")` → pretrained on the whole human genome; no chromosome is held
  out. Mechanistic claims fine, generalisation claims impossible.
- `assert_per_nucleotide_readout("dnabert2")` → BPE; a frameshift is not a minimal pair.
- `assert_cross_strand_capable("splicebert")` → pre-mRNA, sense strand only; cannot represent G4's
  cross-strand axis.
