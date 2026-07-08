# Pre-registration (DRAFT — not frozen)

> **Status: DRAFT.** No test chromosome may be touched until this is frozen to a registry and the
> freeze recorded here with a commit hash and date. `data/genome.py` must enforce this in code, as
> `Config.assert_test_allowed` did on the previous project.

## Confirmatory tests (Holm–Bonferroni, family size 5)

A CLT over 6 layers × 4096 features yields edge counts in the millions. Correcting across all of
them leaves nothing standing. Only these five are confirmatory. Everything per-edge, per-feature and
per-layer is **exploratory, reported under FDR, and labelled as such.**

| ID | Hypothesis | Statistic | Prediction | Null |
|----|-----------|-----------|------------|------|
| **C1** | H2 | within-model completeness | monotone increasing across AR < MLM < MLM+phyloP < phyloP-only, **at matched eval-head accuracy** | permutation over arm labels |
| **C2** | H2 | routing share of the traced circuit | monotone increasing across the same ladder | permutation over arm labels |
| **C3** | H3 | JSAE Jacobian sparsity (Hoyer) | AUC > 0.8 separating compositional from heuristic planted twin | twin labels shuffled |
| **C4** | H4 | routing share, traced vs matched control edges | traced ≫ matched, **sign-correct** vs biophysics | matched random edges from the same node, same out-degree, same layer distance |
| **C5** | H0 | tracer recall & precision vs `PLANTED_EDGES` | both > 0.9 | random subgraph of matched size |

All nulls are **permutation-based**, not parametric.

## Gates

- **C5 is a hard gate.** If the tracer cannot recover a circuit it was handed, no downstream result
  is interpretable. Nothing in Phases 1–4 is reported if C5 fails.
- **C3 is a gate on H3 only.** If the discriminator cannot separate twins that differ *by
  construction*, H3 is withdrawn. It is not reinterpreted post hoc.

## Directional predictions (sign, not just magnitude)

| Rung | Latent variable | Sign prediction under edge patch |
|---|---|---|
| 1 | reading frame | frame-dependent readout shifts by exactly one codon position per nucleotide of frameshift |
| 2 | splice site | disrupting donor→acceptor spacing reduces acceptor readout; preserving spacing while shuffling the intervening sequence does not |
| 3 | G-quadruplex | extending a loop beyond the ~7-nt window reduces the readout; G-density-matched decoys do not activate the composition node |

## Contamination

`config.assert_holdout_valid` **raises** for `splicebert` and `dnabert2`. Both were pretrained on the
whole human genome; no chromosome is held out from them. Mechanistic claims about these models are
in scope. **Generalisation claims are not, and cannot be made from any split.**

Only `gfm-nano-*` and the planted models admit an honest chromosome holdout.

## What would falsify the programme

- Tracer recall high on the compositional twin **and** on the heuristic twin → the tracer reports
  circuits on everything; the method is uninformative regardless of what it finds in real GFMs.
- Completeness rises monotonically with arm accuracy but **not** with objective at matched accuracy
  → H2 reduces to the behavioural literature it set out to open up.
- Frame circuit found in `gfm-nano` but not in any pretrained GFM → the phenomenon is an artifact of
  our training setup, and the paper is about toy models. Report it that way.
- Faithfulness gap large → the attribution graph is describing the transcoder, not the model, and
  every edge-level claim in the literature that omits this number is suspect. (This is a result.)

## Power

Not yet computed. **TODO before freeze:** simulate the C1/C2 monotone contrast across four arms under
plausible completeness variance to fix the number of seeds per arm.
