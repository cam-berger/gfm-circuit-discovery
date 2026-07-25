# Carrier origin — convergence vs idiosyncrasy synthesis

> **Tier: T1 — canonical.** Current, authoritative. Read this.

**Tier:** T1 (canonical & live). Date 2026-07-23. Controlling claims: C39, C40 (with C8, C12, C36, C37)
in [`studies/splicebert_partner_selection/CLAIMS.md`](../studies/splicebert_partner_selection/CLAIMS.md);
correction history R25 in
[`CORRECTIONS.md`](../studies/splicebert_partner_selection/CORRECTIONS.md). This doc states the verdict;
exact numbers and controlling artifacts live in the ledger rows.

## The question

Why did training build SpliceBERT's two positional carriers — the **period-3** codon clock and the
distinct **≈10.63 carrier**? A single frozen checkpoint cannot answer this. The prior record deferred it
to a future project needing a second model or training-time evidence. Two published-checkpoint natural
experiments, runnable now, convert "deferred" into a *comparative* origin attribution.

## What was run

Three independent lines, all on frozen public checkpoints / public sequence — no retraining:

1. **Sibling corpus contrast** (C39). The SpliceBERT authors released two stage-1 checkpoints differing
   *only* in corpus: `SpliceBERT.510nt` (72 vertebrates) vs `SpliceBERT-human.510nt` (human only) — same
   6L×16H×512 architecture, same stage, same 510-nt length. A leakage-free P-spectrum estimator
   (validated on the final 1024-nt checkpoint against the known period-3 289× / ≈10.63 bin-96 results)
   measured both carriers in both siblings. A band-deletion (carrier-transfer) test then measured whether
   period-3 is *used* in the sibling where it exists.
2. **Cross-model transfer** (C40). The same instruments were ported to two single-nucleotide transformer
   MLMs with learned absolute position embeddings — RNA-FM and RNA-BERT — both trained on largely
   ORF-free ncRNA. A DNABERT-2 arm serves as a tokenizer-aliasing contrast.
3. **Model-free corpus spectrum** — 60 Mb of human genome at the preregistered ≈10.63 period, as the
   input-side anchor.

## The verdict

**Both carriers are corpus-built features of multi-species pre-training, and period-3 is codon-linked.
Neither is a universal genomic-LM feature.**

| Line | period-3 | ≈10.63 carrier |
|---|---|---|
| 72-species stage-1 (P-spectrum) | **435× median** (present) | **369×** (present) |
| human-only stage-1 (P-spectrum) | **6.2×**, undetected (powered) | **2.4×**, undetected (powered) |
| used coding-specifically? (72-sp band-deletion) | **pattern reproduced, verification level** — DiD +0.138, intron health 87.6%; add-one p floored at 0.077 (12 of 30 control bands), so not a passed confirmation (R33) | not tested (donor-scoped, C37) |
| RNA-FM / RNA-BERT (ncRNA, P-spectrum) | **absent** (≈1× median, powered null) | **absent** (≈1× median, powered null) |
| human genome (model-free, 60 Mb) | n/a (codon signal is coding-only) | **no ≈10.63 periodicity** (Z≈1, n.s.) |
| DNABERT-2 (BPE, no learned P) | **not representable** (codon = 0.64 token-steps < Nyquist) | — |

Three convergent reads:

- **period-3 is corpus-dependent and codon-linked, not a generic codon read-out.** The load-bearing
  result is its *absence* in the human-only sibling: codon structure is present in human coding sequence,
  yet human-only training did not build the code. It appears only with multi-species pre-training, and it
  is absent from ORF-free ncRNA models — the parsimonious account is that period-3 encodes reading-frame
  structure that becomes learnable/useful under cross-species coding-sequence pressure.
- **≈10.63 is a multi-species artifact, not a human-sequence statistic.** Present in 72-species, absent
  in human-only, **not detectable** in the human genome's dinucleotide spectrum above that unanchored
  assay's 5–10% modulation floor (a sensitivity limit, not an established absence — R16/C15/**R30**), and
  absent from the ncRNA models. This makes MECHANISM item 4's "multispecies" label a measured attribution
  rather than an assertion. It does **not** reopen the nucleosomal reading: the ≈10.298-nt signal —
  measured in **mouse** ES-cell chromatin (Voong 2016, mm9), not in human — remains a powered refutation
  (C14, R30).
- **Single-nucleotide tokenization is necessary to see these codes at all.** DNABERT-2 has no learned
  position table (ALiBi) and a BPE tokenizer averaging 4.66 nt/token, so a 3-nt period maps to 0.64
  token-steps — below the 2-step Nyquist period, i.e. mathematically unrepresentable. A null in a BPE
  model is a tokenizer artifact, not evidence about the model.

## Scope and honest limits

- **Attribution, not dynamics.** These are endpoint natural experiments. They show a feature tracks a
  training factor (corpus); they do not trace when in training it forms or prove causation the way
  controlled retraining would. Training *dynamics* remain a genuine next-project question.
- **Cross-model set is RNA-domain.** No public **DNA-domain** single-nt transformer MLM with a learned P
  exists, so the cross-model absence is demonstrated on ncRNA models; the behavioral coupling arm does
  not transfer there (no splice sites in ncRNA), so C40 is structural presence/read only.
- **One caught false positive.** An initial "16 RNA-FM heads read ≈10.63" was shown to be an
  envelope/permutation artifact by a frequency control and is not claimed. See R25.

## What this closes

MECHANISM items 3 and 4 move from "origin deferred, needs a new model" to **corpus-attributed origin**.
Origin *dynamics* and any *biological function* for either carrier remain open and correctly deferred.
