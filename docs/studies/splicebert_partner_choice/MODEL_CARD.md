# MODEL_CARD — splicebert_partner_choice

**Model.** SpliceBERT (Chen et al.) — a BERT-style masked-LM over primary RNA. Provenance class:
**pretrained** ⇒ no chromosome holdout; mechanistic claims only (enforced by `config.assert_holdout_valid`).

**Source (verified, `scripts/verify_model_source.py`).**
- HF id in use: `yangheng/SpliceBERT-510nt` — a **community mirror**, `is_official=False`.
- The **official** weights ship via the `biomed-AI/SpliceBERT` GitHub repo / Zenodo, not an
  anonymously-accessible HF repo (`chenkenbio/SpliceBERT` requires auth). Verify against the official
  release before any published claim.

**Architecture (empirically confirmed).** 6 layers · hidden 512 · 16 heads · absolute position
embeddings · **19.45M** parameters · this mirror is the **510nt** variant (`max_position_embeddings`
512 = 510 nt + [CLS] + [SEP]; the official 1024nt variant differs). Post-LayerNorm BERT.

**Tokenizer (the trap the empirical check caught).** Single-nucleotide, but requires **space-separated**
input: `"ACGT"` → one `[UNK]`; `"A C G T"` → `[6,7,8,9]`. Vocab `{A:6, C:7, G:8, T:9, N:5, [CLS]:2,
[SEP]:3, [PAD]:0, [MASK]:4, [UNK]:1}`. `SpliceBERTAdapter.encode` inserts the spacing and records the
nt→token offset (nt `i` → token `i+1`).

**Adapter.** `gfmcd.models.adapters.SpliceBERTAdapter` — exposes nt↔token alignment, per-layer
residual/Q/K/V/attention/per-head context/MLP, MLM logits, and self-restoring head ablation. Smoke
tests: `tests/test_splicebert_adapter.py` (`-m network`), all passing.
