# Circuit Discovery in Genomic Foundation Models

Tracing how a training objective installs a latent representation of genome structure.

Built from scratch: none of our three model families (planted, GFM-nano, SpliceBERT) are supported
by TransformerLens / `circuit-tracer`, which target GPT-style decoder-only models. See
`docs/DECISIONS.md#d5`.

## Where to start

| | |
|---|---|
| **`RESULTS.md`** | **Every measured number, with status: SETTLED / PENDING / WITHDRAWN / RETRACTED.** |
| **`REC.md`** | The argument: critique of Proposal D2, and the criterion that replaces it. |
| `docs/EXPERIMENTS.md` | **Plain-language account of every experiment and what it showed.** Start here. |
| `docs/DECISIONS.md` | Why the design is what it is, with the numbers. |
| `docs/PLAN.md` | Phases and gates. |
| `docs/PREREGISTRATION.md` | The five confirmatory tests. **Draft — not frozen.** |

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
