# Vendored DNABERT-2 modelling code

> **Tier: T2 — provenance.**

These three files are **verbatim copies** of the remote code published with
`zhihan1996/DNABERT-2-117M`, pinned to Hugging Face revision
`7bce263b15377fc15361f52cfab88f8b586abda0`.

## Why vendored rather than `trust_remote_code=True`

1. **It does not otherwise load.** `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)`
   fails on both transformers versions available here:
   - `transformers 5.13.0` — the remote-code *import scanner* hard-fails on `flash_attn_triton.py`'s
     unguarded `import triton`, even though `bert_layers.py` already guards its own import of that module
     in a `try/except` and falls back to PyTorch attention.
   - `transformers 4.44.2` — `AutoModel.register` raises
     `config_class not consistent`, because the remote `BertModel` inherits the *builtin*
     `BertPreTrainedModel.config_class` rather than the remote `BertConfig`.

   Direct import of the vendored modules bypasses both, because neither the auto-registry nor the
   remote-code import scanner is involved.

2. **Reproducibility.** The study's execution does not depend on a network fetch of mutable remote code.
   Hashes below are checked by the loader at every run.

3. **Hookability.** Later phases require named hooks on residual/attention/QKV. A local, readable copy is
   a prerequisite; MosaicBERT also unpads internally (per-block activations are `[n_tok, H]`, not
   `[B, T, H]`), which must be handled explicitly.

## What was changed

**Nothing.** All three files are byte-identical to the upstream revision. `flash_attn_triton.py` is
**deliberately not vendored**: `bert_layers.py` imports it inside a `try/except ImportError` and sets
`flash_attn_qkvpacked_func = None` on failure, which selects the PyTorch attention path. Omitting the file
therefore reproduces exactly the code path that runs on this machine anyway (no Triton on Apple Silicon),
and removes the only unguarded Triton dependency.

Consequence to record with any result: **attention is computed by the PyTorch reference path**, not the
Triton kernel. Attention probabilities are materialised and hookable.

## Provenance

| Artifact | sha256 |
|---|---|
| `bert_layers.py` | `317ad7e9667980ac724c07f0174ba265c8446ca5230f6b473068ff1b911edcf9` |
| `bert_padding.py` | `44d1c68afb1f585fdc66c150d4c60f1ed44a89c006abc57d50531d71940d7421` |
| `configuration_bert.py` | `95fc868641b87bbcd7a32d2cd7b9f4769c27592e129daf167d14b5b8c74ec4c5` |
| `config.json` (upstream, not vendored) | `ba9bdafaff0cc3e30556927474d4a179519a9864012bed2628e9f1bc23c84bfd` |
| `tokenizer.json` (upstream, not vendored) | `5d178e8ce2ba55df97fff197f4b30f40133b95d7096be398c2df6b526c5d8cd3` |
| `tokenizer_config.json` (upstream, not vendored) | `f9d18c81f4dd9dd7db02e9f27cc1203228147d890bfce9167c3af6465ff5b769` |
| `pytorch_model.bin` (upstream, not vendored) | `7ff39ec77a484dd01070a41bfd6e95cdd7247bec80fe357ab43a4be33687aeba` |

Verified on load: `load_state_dict(..., strict=False)` returns `missing=[]`, `unexpected=[]` — the
checkpoint carries a **complete, trained MLM head** (`cls.predictions.decoder [4096, 768]`), which is the
precondition for the native-MLM assay.

`VENDOR.sha256` in this directory is the machine-readable form and is re-checked by the loader.
