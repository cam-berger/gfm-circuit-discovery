# Preregistration: SpliceBERT GTEx fine-tuning shift

> **Tier: T1 — canonical.** Current, authoritative. Read this.

Status: **FROZEN AND EXECUTED THROUGH THE BEHAVIOR GATE. MECHANISM CLOSED BY THE STOPPING RULE.**

This document was frozen before any coupling result from a fine-tuned checkpoint was inspected. Parameter
displacement and checkpoint structure were inspected during instrument construction; model behavior was
opened only after the preflight passed. The seed-2020 smoke then failed the behavior/denominator gate, so
attention and activation tracing were not opened. See [`DECISIONS.md`](DECISIONS.md) for the dated execution
record and the outcome-aware status of the subsequent five-seed behavior-only verification.

## 1. Question and estimands

The primary question is whether supervised GTEx splice-site fine-tuning changes the base model's native
masked-LM donor–acceptor coupling behavior and whether the already established base-model circuit transfers
without component reselection.

For each seed `s`, on the same annotation-defined validation examples:

- `C_base` and `C_s`: mean masked-donor coupling, clean minus paired-acceptor corruption.
- `ΔC_s = C_s - C_base`: paired behavioral shift.
- specificity shifts against the distance-matched other-donor and non-partner-AG controls.
- transfer of the frozen `{L3H1, L2H15}` necessity, K/V interaction, relocated binding, and denoise
  recovery arms.

Parameter norms, representation similarity, probes, spectra, and attention changes are candidate
generators. They do not establish that fine-tuning changed a causal computation.

## 2. Checkpoint family and readout boundary

All five published seeds (2020–2024) are required. Seeds may not be selected using behavior. The raw
checkpoint hashes and archive hash are frozen in
[`../../configs/splicebert_gtex_finetune_shift/base.json`](../../configs/splicebert_gtex_finetune_shift/base.json).

The authors released token-classification state dictionaries, not masked-LM checkpoints. The primary
instrument is therefore a derived checkpoint:

1. load the exact base `SpliceBERT.1024nt` masked LM;
2. require exact, finite, shape-compatible coverage of every `bert.*` tensor;
3. reject every non-encoder tensor except the documented three-class classifier weight and bias;
4. replace the complete encoder;
5. retain the base MLM transform and output bias;
6. keep the decoder tied to the transplanted nucleotide embedding table.

Claims are scoped to this fixed-MLM-readout instrument. The native three-class head is a distinct future
behavioral arm and may not be silently substituted into the primary estimand.

## 3. Preflight gates

All gates must pass before a target run:

- **P1 provenance:** base, archive, and all five checkpoint hashes match the frozen config.
- **P2 architecture:** 6 layers, 16 heads, hidden size 512, 1026 positions, vocabulary size 10.
- **P3 exact transplant:** all 101 encoder tensors load strictly; classifier shapes are `(3, 512)` and
  `(3,)`; no other keys occur.
- **P4 numerical integrity:** every tensor and a short forward pass are finite.
- **P5 tokenizer/coordinate:** exactly one token per nucleotide with one prefix token; donor and acceptor
  coordinates match the existing coupling task.
- **P6 readout construction:** the MLM transform and output bias match the base; decoder and transplanted
  nucleotide embeddings are tied.
- **P7 positive control:** a deliberately perturbed encoder is detected above float64 numerical
  resolution, and each published seed has a nonzero recorded displacement.
- **P8 negative control:** identity/base state has zero displacement and a checkpoint missing one encoder
  tensor is rejected.
- **P9 mechanics:** the unchanged base model reproduces a small validation smoke cohort within deterministic
  tolerance before any fine-tuned smoke output is interpreted.

Any anomaly freezes interpretation and returns the study to instrument diagnosis.

## 4. Target sequence

1. Run an outcome-limited smoke on seed 2020 only (`n≤32`, `n_mech≤12`, `n_null≤20`) to test mechanics.
   Smoke outcomes are permanently non-confirmatory.
2. If the smoke passes mechanically, run the unchanged validation protocol at its recorded sizes on all
   five seeds, one checkpoint at a time.
3. Do not open the sealed chr8/9 test split for this study.
4. Only after behavior and specificity are established per seed may the frozen mechanism arms be read.
5. Only after those gates may donor-anchor/address/payload/carrier comparisons be run.

## 5. Controls and inference

- The base checkpoint is the paired reference.
- Distance-matched other-donor and non-partner-AG controls remain mandatory.
- Component interventions use the same-size random null family as the base study; null families of
  different sizes are not pooled.
- Add-one permutation values are used exactly; the full five-seed family is reported.
- A meaningful family-level shift requires a paired effect with uncertainty plus directionally consistent
  evidence across at least four of five seeds. A visually large parameter displacement is insufficient.
- Failure of the frozen heads means only that the base circuit did not transfer at that granularity. It
  does not establish absence of a fine-tuned circuit; new discovery would be a separate exploratory stage.

## 6. Stopping and claims

If base replay, specificity, health, tokenizer, attention mechanics, or transplant invariants fail, stop.
No fine-tuning interpretation is licensed. If behavior survives but the frozen circuit fails, the strongest
claim is a cross-checkpoint circuit-transfer failure, not circuit destruction. If both survive, the
strongest claim is preservation under the tested encoder fine-tuning and derived readout.
