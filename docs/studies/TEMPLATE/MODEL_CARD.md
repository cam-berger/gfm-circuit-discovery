# MODEL_CARD — <study name>

- **Model + provenance.** Official repo / Zenodo / verified HF conversion (with checksum) — community
  mirror only as a clearly-labeled fallback. Record source_type, source_uri, revision, sha256.
- **Architecture facts.** Layers, hidden size, heads, position encoding, tokenization, max length,
  parameter count. Verify empirically (`scripts/verify_model_source.py`), do not trust the card.
- **Provenance class.** planted / trained-by-us / pretrained. Pretrained ⇒ no chromosome holdout;
  mechanistic claims only (enforced by `gfmcd.config` firewalls).
