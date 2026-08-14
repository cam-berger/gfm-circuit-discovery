# DNABERT-2 fine-tuned versus frozen — concise comparison

> This document summarizes the completed fine-tuned assays and the licensed
> descriptive comparison with the canonical frozen DNABERT-2 promoter study. Canonical claims live in
> [`CLAIMS.md`](CLAIMS.md); corrections and boundaries live in [`CORRECTIONS.md`](CORRECTIONS.md).

## Why this file exists

The fine-tuned experiment package is complete, but the frozen and fine-tuned analyses are not a matched
before/after intervention. This file therefore separates:

1. what is established within the fine-tuned state; from
2. what may only be compared descriptively across states.

## Conditions compared

| Property | Frozen canonical study | Fine-tuned analysis |
|---|---|---|
| Model state | `quietflamingo/dnabert2-no-flashattention@813031b2…`, frozen | `magicslabnu/DNABERT-2-finetuned-prom_300_all@7f2a5f5b…` loaded into the compatible no-FlashAttention implementation |
| GUE task | `prom_core_all` | `prom_300_all` |
| Sequence length | 70 nt | 300 nt |
| Data treatment | Explicit leakage cleaner; retained `47,149 / 5,856 / 5,865` | Official `47,356 / 5,920 / 5,920`; exact-overlap audit and strict-unseen test sensitivity |
| Primary behavioural readout | Fixed masked-mean linear probe | Checkpoint 2-class classifier |
| Layer representation assay | Frozen layer sweep; Layer 2 is a thin working choice | Fine-tuned layer sweep; Layer 10 selected on development |
| Main comparison status | — | Descriptive only across states (DBP17) |

Because these columns differ in more than weights, cross-state numerical deltas do **not** identify a
causal effect of fine-tuning.

## Fine-tuned predictive result

On the official 5,920-example `prom_300_all` test split:

| Metric | Value |
|---|---:|
| Balanced accuracy | **0.935205** |
| MCC | **0.870487** |
| ROC-AUC | **0.984785** |
| PR-AUC | **0.984351** |
| F1 | **0.934871** |

Removing exact test sequences appearing in train or development leaves `n=5,900` and balanced accuracy
`0.935136`, with ROC-AUC `0.984781`. The small exact-overlap set therefore does not materially explain
the fine-tuned checkpoint's test performance (DBP12).

## Fine-tuned representational accessibility

A masked-mean linear probe is fit independently at every encoder layer and selected by development
balanced accuracy, with development ROC-AUC as tie-breaker.

- layer returned by the stated rule: **Layer 10**
- development BA: `0.942919`
- development ROC-AUC: `0.985708`
- descriptive test BA: `0.932418`
- descriptive test ROC-AUC: `0.983282`

**The rule returns a layer; the data do not separate one.** Layer 10 leads its runner-up (Layer 7) by
`0.000998` development BA, which is `0.33` standard errors on this split. Layers **7–11** lie within one
standard error of the argmax and layers **5–11** within two
(`results/dnabert_promoter/finetuned/layer_localization/layer_selection_band.json`). Layer 8 has the
numerically highest test BA (`0.935252`), but test was not used to select the endpoint.

The licensed statement is that promoter information is strongly linearly accessible across a band of
late layers, and that Layer 10 is the layer the stated development rule returned on one seed. This is
neither causal localization nor a localized endpoint (DBP13, DBPR11). The seeded probe repeat that
would settle the choice has not been run.

## Fine-tuned early causal dependence

A locked balanced 256-example test cohort (`128` non-promoters, `128` promoters) is used for targeted
interventions.

| Condition | Balanced accuracy | MCC | ROC-AUC |
|---|---:|---:|---:|
| Baseline | **0.957031** | 0.914090 | 0.991272 |
| `L0H8` ablated | **0.511719** | 0.108893 | 0.685669 |
| `L0_MLP` ablated | **0.816406** | 0.671858 | 0.900757 |
| Both ablated | **0.503906** | 0.036298 | 0.665222 |

The `L0H8` effect is strongly class-asymmetric: on non-promoters, ablation raises mean promoter
probability by `+0.819336`, while the promoter-class mean shift is `+0.003715`.

### Restoration

Starting from joint ablation:

| Condition | Balanced accuracy |
|---|---:|
| Joint ablation | 0.503906 |
| Restore `L0H8` | 0.816406 |
| Restore `L0_MLP` | 0.925781 |
| Restore both | **0.957031** |

Restoring both reproduces baseline logits exactly (`max |Δlogit| = 0`). The causal conclusion is strong
dependence on the targeted early pathway in this fine-tuned state; compactness or identity with the frozen
mechanism is not established (DBP14).

## Fine-tuned activation-source compatibility

Activation sources are aligned by nucleotide-overlap-weighted tokenizer offsets rather than raw BPE token
index.

| Component | Same-class mean abs ΔP | Opposite-class mean abs ΔP | Opposite / same |
|---|---:|---:|---:|
| `L0H8` | 0.092880 | 0.175941 | **1.89×** |
| `L0_MLP` | 0.090191 | 0.773006 | **8.57×** |
| Both | 0.114631 | 0.815616 | **7.12×** |

Opposite-class swaps are much less compatible, especially for the MLP and combined pathway. This supports
class-conditioned early states; it does not establish example-specific payload or biological semantics
(DBP15).

## Fine-tuned TATA perturbation

Among 354 promoters with `TATA[AT][AT]`, the earliest motif occurrence is replaced with `CGCGCG` and
compared with an equal-length mirrored control replacement.

- mean motif damage: `+0.025684`
- mean control damage: `+0.047558`
- mean excess motif damage: `-0.021874`
- fraction motif > control: `0.483051`
- Wilcoxon p: `0.162760`
- bootstrap mean-effect interval: `[-0.047613, +0.003288]`

The assay does not support positive TATA-specific dependence in the fine-tuned state (DBP16).

## What may be said across states

The canonical frozen study establishes that promoter-discriminative information is recoverable from a
frozen encoder; Layer 2 is the strongest tested working endpoint only by a `0.0019` validation-BA margin
on one seed, and the frozen causal study identifies a structured but non-compact, MLP-dominated early
computation (DBP3–DBP8).

The completed fine-tuned state shows:

- strong `prom_300_all` predictive performance;
- high late-layer linear accessibility;
- strong causal dependence on `L0H8` and `L0_MLP`;
- class-conditioned early activation compatibility;
- no positive TATA-specific effect under the completed paired assay.

A safe synthesis is:

> **The analyzed fine-tuned DNABERT-2 state combines late-layer promoter accessibility with strong
> early Layer-0 causal dependence. Relative to the frozen study, the representational profiles differ,
> but the present design does not isolate fine-tuning as the cause of that difference.**

## What may not be said

Do not write, without a new matched experiment:

- “fine-tuning improved balanced accuracy from 0.810 to 0.935” as a causal effect;
- “fine-tuning moved the promoter representation from Layer 2 to Layer 10”;
- “fine-tuning preserved/strengthened the exact same frozen circuit”;
- “fine-tuning erased or forgot TATA-box biology”;
- “the fine-tuned mechanism is compact”;
- “Layer 10 is the causal promoter layer.”

The first two numbers/endpoints may appear in a descriptive table only if the task/readout mismatch is
stated next to them.

## What a matched fine-tuning comparison would require

The cleanest next comparison would hold fixed:

1. GUE task/configuration and sequence length;
2. train/dev/test examples and duplicate policy;
3. tokenizer and parent checkpoint identity;
4. representation extraction and layer-probe protocol;
5. intervention sites and causal cohort;
6. metrics and selection rules.

Ideally the exact pre-fine-tune parent checkpoint and the fine-tuned checkpoint would be evaluated under
the same pipeline. Only then can a cross-state delta be attributed primarily to supervised fine-tuning.

## Canonical references

- Frozen claims: `DBP1–DBP10`
- Fine-tuned claims: `DBP11–DBP16`
- Cross-state boundary: `DBP17`
- Historical missing-artifact correction: `DBPR6`
- Fine-tuned artifact supersession/status update: `DBPR9`
- Unmatched-comparison boundary: `DBPR10`
- Fine-tuned design decisions: `D14–D22`
