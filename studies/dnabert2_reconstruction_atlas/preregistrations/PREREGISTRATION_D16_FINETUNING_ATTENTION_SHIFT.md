# D16 preregistration — did fine-tuning move the attention, or only the readout?

> **Tier: T1 — frozen protocol.** Frozen 2026-07-29 before any base-model measurement.

> **Status: FROZEN 2026-07-29, before any base-model measurement is taken.** The interpretation
> rule in §4 is fixed here precisely so that it cannot be chosen after seeing the numbers.

Follow-up to D15. Short, descriptive, and deliberately narrow.

---

## 1. Intent

> **Were the promoter classifier's carrier heads already sitting at the ALiBi positional prior
> before fine-tuning, or did fine-tuning move them there?**

D15 established that in the **fine-tuned** model `L11H2`/`L11H5` — the replicated causal core — do
essentially no content-driven `[CLS]` source selection (excess over the ALiBi-only prior +0.046 and
+0.032, ≈0.000 in every other band), while the one real TSS reader `L9H7` (excess `prox_up` +0.401)
is ~19× weaker causally. D15 could not say whether that is a *property of DNABERT-2* or a *product
of fine-tuning*, because no experiment in this study has ever loaded both checkpoints.

**This is the fill-in for a blank the record cannot currently fill.** D1–D3 used base
`zhihan1996/DNABERT-2-117M`; D4–D15 used `magicslabnu/DNABERT-2-finetuned-prom_300_all`. No script
loads both, and the two arcs measure different quantities (MLM nats per nucleotide vs classifier
logit margin), so the study has **no before/after measurement of anything**.

---

## 2. Why this comparison is licensed at all

The two checkpoints share an architecture and a tokenizer. Tokenizer equivalence is not assumed: it
is already a **hard load-time gate** — `load_dnabert2_promoter_classifier` calls
`verify_tokenizer_equivalence` (`promoter_classifier.py:191`), which raises unless BPE model type,
full vocabulary, normalised merge order, added tokens, pre-tokenizer and post-processor all match the
pinned base tokenizer. §5 adds a **runtime** check on the actual cohort, which is stronger than the
spec comparison.

Because the tokenization is identical, the BPE offsets are identical, so the nucleotide-band
allocation and the ALiBi-only prior are **the same for both models**. Only the attention changes.

---

## 3. Estimand

For every one of the 144 heads `h`, and each model `m ∈ {base, fine_tuned}`, measure the `[CLS]`-row
attention mass over the frozen `PROMOTER_SOURCE_BANDS`, averaged over the cohort, and subtract the
ALiBi-only prior for that head index:

```
excess_m(h, band) = measured_m(h, band) − prior(head_index(h), band)
```

`prior` is computed exactly as in D15 (`analyze_promoter_alibi_reach.py`) on the same real BPE
offsets, so it is common to both models and cancels in any base-vs-fine-tuned difference.

Primary quantity: **`Δexcess(h, band) = excess_fine_tuned − excess_base`**.

Population summaries: mean `|excess|` over all 144 heads per model; Pearson and Spearman correlation
of per-head `excess_far_upstream` between models.

Cohort: all 384 `promoter_source_localization_v1` loci, target and control sequences. This is a
**descriptive measurement with no selection, no split, and no hypothesis test gating a claim** —
attention is a deterministic function of the weights and the input, so the risk here is instrument
error, not inference error. §5 is therefore where the rigour sits.

---

## 4. Frozen interpretation rule — fixed before any base number is read

Let `E_base` be the base model's excess `far_upstream` for `L11H2` and `L11H5`. Threshold
**0.10**, chosen because `L9H7`'s +0.401 is the scale of genuine content-driven selection while
D15's carriers sit at 0.03–0.05; 0.10 lies between "at the prior" and "really selecting".

| outcome | reading |
|---|---|
| `\|E_base\| < 0.10` for both carriers | **ALREADY AT PRIOR.** The positional blindness is a property of pretrained DNABERT-2, not of fine-tuning. Fine-tuning changed *what is read out of* a fixed attention geometry, not *where the model looks*. |
| `\|E_base\| ≥ 0.10` for either carrier, and `\|excess_fine_tuned\| < \|E_base\|` | **MOVED ONTO PRIOR.** Fine-tuning actively gave up content-driven source selection at the carriers. |
| `\|E_base\| < 0.10` and `\|excess_fine_tuned\| ≥ 0.10` | **MOVED OFF PRIOR.** Effectively excluded in advance: D15 already measured the fine-tuned excess at +0.046/+0.032. If this fires, something is wrong with the instrument, not with the biology. |

Secondary, same rule applied to `L9H7`: was it already the TSS reader, or did fine-tuning make it one?

**No causal claim follows from any outcome.** Attention is descriptive; this compares two descriptive
maps. It cannot show that a changed head caused a changed behaviour.

---

## 5. Memorials — each must be able to fail, on the measurement path

| id | check | can it fail? |
|---|---|---|
| **M1** runtime tokenization identity | for every one of the 768 cohort sequences, base and classifier tokenizers produce **identical `input_ids` and identical offset mappings** | **yes** — and it is the load-bearing check: if it fails, the offsets differ, the band allocation differs, and no comparison is valid |
| **M2** batch invariance | each model's per-head band masses reproduce to < 1e-5 when recomputed at a different batch size | **yes** |
| **M3** cross-cohort consistency | the fine-tuned profile measured here is compared against the committed D5 head-atlas values for the eight candidate heads | **reported, not gated** — D5 used a *different* cohort, so exact agreement is not expected; a gross divergence would indicate a capture bug |
| **M4** band-mass normalisation | per-head band masses sum to 1 | **architecture-forced** — asserted inside `capture_cls_attention_bands`. **Flagged, not counted as evidence** |
| **M5** base checkpoint identity | base provenance reports `has_learned_position_table == False` and an MLM head present | **yes for the head; forced for the position table** — the position-table absence is an architecture fact and is flagged as such |

---

## 6. Scope

1. **Descriptive only.** No causal claim, no mediation, no behaviour.
2. **Attention mass is not information transfer.** A head at its positional prior can still move
   information; this measures where it looks, not what it carries.
3. **This does not fill the "representation changed from X to Y" blank for behaviour.** The base model
   has no promoter readout, so the D15 behavioural endpoint has no base-model counterpart. Any
   behavioural before/after needs a separate, differently-designed study, and its estimand is not
   obvious — that limitation is stated, not worked around.
4. The comparison covers the `[CLS]` row only, which is what the classifier reads; it says nothing
   about token-to-token attention elsewhere in the network.
