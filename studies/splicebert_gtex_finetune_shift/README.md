# SpliceBERT GTEx fine-tuning shift

## Purpose and design

This study asks what happens to SpliceBERT's donor–acceptor coupling when the same `SpliceBERT.1024nt`
encoder is fine-tuned for three-class splice-site prediction on SpliceAI's GTEx dataset. The checkpoints
are the five published seeds (2020–2024) from
[`chenkenbio/SpliceBERT-analysis`](https://github.com/chenkenbio/SpliceBERT-analysis), trained with
learning rate `1e-4` and R-Drop weight `10`, treated as one family on the existing gene-split validation
cohort. Coupling is measured two ways: through the original masked-LM donor margin, by transplanting each
fine-tuned encoder into the base masked LM, and through the released classifier's native margin, a raw
logit difference. Follow-ups freeze components on training genes and test them on held-out genes with
patching, path decomposition, and positional-carrier surgery. This is a matched-checkpoint transfer
study, not a claim about biological splicing.

## Findings

Through the original MLM readout, coupling falls from `+0.958` to near zero in every seed, but MLM health
falls too, so this supports broad readout incompatibility, not evidence of erasure (GFS4–GFS5). The released
classifier still uses the paired acceptor: destroying it lowers native donor evidence by
`0.646 [0.563,0.733]` logit units, positive in every seed, and the signal becomes readable almost
entirely in the final encoder block (GFS7–GFS9). `L5H6` carries part of that coupling signal, and other
layer-5 heads carry part of two nonlocal sequence-order signals, one upstream of the donor and one
downstream of the acceptor. All of them converge on the final MLP, which carries almost all of each
effect (GFS11–GFS18). In exploratory follow-ups, the MLP writes a shared compact code, rotated
differently in each seed and built from hundreds of units (GFS19–GFS20). The inherited period-3 and
≈10.63 carriers survive fine-tuning and support local splice-site evidence but not paired-site coupling
(GFS23–GFS25).

## Status

Open. The fixed-MLM arm is closed at its behavior gate, so the base circuit was not traced. The
native-classifier coupling behavior (GFS8, verification tier) and an incomplete many-head-to-final-MLP
causal route (GFS11–GFS18) replicate across the five seeds. The seed alignment and unit census await a new
analysis-sealed replication, the biological identity of the context signals is unresolved (GFS21–GFS22),
and the J-SAE family stopped before feature identity, classifier behavior, biology, or steering
(GFS26–GFS34). Claims are scoped to this validation instrument. [`FINDINGS.md`](FINDINGS.md) lists the
missing bridges.

## Files

- [`FINDINGS.md`](FINDINGS.md): start here; groups GFS1–GFS34 into six findings.
- [`CLAIMS.md`](CLAIMS.md): the authoritative claim ledger, with every number and caveat.
- [`DECISIONS.md`](DECISIONS.md): design decisions and preserved stops.
- [`RESULTS.md`](RESULTS.md) and the [result
  index](../../results/splicebert_gtex_finetune_shift/RESULTS_INDEX.md): run map and provenance.
- [`PREREGISTRATION.md`](PREREGISTRATION.md) (derived checkpoint and stage gates),
  [`PREREGISTRATION_NATIVE_CLASSIFIER.md`](PREREGISTRATION_NATIVE_CLASSIFIER.md), and the later
  `PREREGISTRATION*.md` and `IMPLEMENTATION_ADDENDUM_*.md` files: the frozen designs.
