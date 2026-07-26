"""Pinned, hash-verified loader for DNABERT-2-117M.

Every load is a provenance event: the vendored modelling code, the checkpoint, the tokenizer,
and the library versions are all hashed/recorded and returned in `identity_record()`, which the
runners embed verbatim in their result manifests.

Design notes that matter downstream
-----------------------------------
* The checkpoint carries a **complete trained MLM head**; `load_state_dict` is asserted to report
  no missing and no unexpected keys. A partial head would silently turn the primary assay into
  noise, so this is a hard failure, not a warning.
* There is **no learned position table** (ALiBi in the attention bias). `config.position_embedding_type`
  reads `"absolute"` and is *wrong*; the state dict has no `position_embeddings` key. Any
  position-table method from the SpliceBERT arc is inapplicable — see
  `results/origin-crossmodel/canonical/dnabert2_aliasing.json`.
* Triton is absent, so MosaicBERT selects its PyTorch attention path. Attention probabilities are
  materialised (hookable); throughput is lower. This is recorded, not worked around.
* MosaicBERT **unpads internally**: per-block activations are `[n_valid_tokens, H]`, not `[B, T, H]`.
  Anything hooking encoder internals must map back through the padding indices.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DNABERT2_HF_ID = "zhihan1996/DNABERT-2-117M"
DNABERT2_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"

#: Library versions this study is pinned to. See
#: `studies/dnabert2_core_promoter/requirements-dnabert2.txt` for why.
PINNED_TRANSFORMERS = "4.44.2"

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"

#: Upstream artifacts that are *not* vendored (too large / binary) but are hash-pinned here.
UPSTREAM_SHA256 = {
    "pytorch_model.bin": "7ff39ec77a484dd01070a41bfd6e95cdd7247bec80fe357ab43a4be33687aeba",
    "tokenizer.json": "5d178e8ce2ba55df97fff197f4b30f40133b95d7096be398c2df6b526c5d8cd3",
    "config.json": "ba9bdafaff0cc3e30556927474d4a179519a9864012bed2628e9f1bc23c84bfd",
    "tokenizer_config.json": "f9d18c81f4dd9dd7db02e9f27cc1203228147d890bfce9167c3af6465ff5b769",
}


class PinnedEnvironmentError(RuntimeError):
    """Raised when the execution environment is not the pinned one."""


def require_pinned_env() -> dict[str, str]:
    """Hard-stop unless the pinned execution environment is active.

    A version drift here does not raise at load time, it changes the numbers. Fail loudly.
    """
    import torch
    import transformers

    if transformers.__version__ != PINNED_TRANSFORMERS:
        raise PinnedEnvironmentError(
            f"transformers=={transformers.__version__} but this study is pinned to "
            f"{PINNED_TRANSFORMERS}. Run via:\n"
            "  uv run --isolated --no-project --python 3.11 \\\n"
            "    --with-requirements studies/dnabert2_core_promoter/requirements-dnabert2.txt \\\n"
            "    --with-editable . python <runner>"
        )
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_vendor() -> dict[str, str]:
    """Re-check the vendored modelling code against VENDOR.sha256. Any drift is a hard stop."""
    expected: dict[str, str] = {}
    for line in (_VENDOR_DIR / "VENDOR.sha256").read_text().splitlines():
        digest, name = line.split()
        expected[name.strip()] = digest
    observed = {name: sha256_file(_VENDOR_DIR / name) for name in expected}
    drifted = {n: (expected[n], observed[n]) for n in expected if expected[n] != observed[n]}
    if drifted:
        raise PinnedEnvironmentError(f"vendored DNABERT-2 code has drifted from its pin: {drifted}")
    return observed


def resolve_snapshot(revision: str = DNABERT2_REVISION) -> Path:
    """Locate the pinned local HF snapshot. Never downloads."""
    pat = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/{revision}"
    )
    hits = glob.glob(pat)
    if not hits:
        raise FileNotFoundError(
            f"DNABERT-2 snapshot {revision} not found in the local HF cache. "
            "Fetch it once with huggingface-cli, then re-run; runs must not download."
        )
    return Path(hits[0])


@dataclass
class Dnabert2Bundle:
    """A loaded, verified DNABERT-2 MLM plus everything needed to reproduce it."""

    model: Any
    tokenizer: Any
    config: Any
    snapshot: Path
    provenance: dict[str, Any] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        return dict(self.provenance)


def load_dnabert2_mlm(
    revision: str = DNABERT2_REVISION,
    device: str = "cpu",
    verify_upstream_hashes: bool = True,
) -> Dnabert2Bundle:
    """Load DNABERT-2-117M for masked-LM scoring, verifying every pin on the way.

    Raises on: wrong library version, drifted vendored code, drifted upstream artifact,
    or an incomplete MLM head.
    """
    import torch
    from transformers import AutoTokenizer

    env = require_pinned_env()
    vendor_hashes = _verify_vendor()
    snap = resolve_snapshot(revision)

    upstream_hashes: dict[str, str] = {}
    for name, expected in UPSTREAM_SHA256.items():
        path = snap / name
        if not path.exists():
            raise FileNotFoundError(f"missing pinned upstream artifact: {path}")
        if verify_upstream_hashes:
            got = sha256_file(path)
            if got != expected:
                raise PinnedEnvironmentError(f"{name}: sha256 {got} != pinned {expected}")
            upstream_hashes[name] = got
        else:
            upstream_hashes[name] = "<unverified>"

    # Direct import of the vendored modules: bypasses both the auto-registry config_class clash
    # (transformers 4.44) and the remote-code import scanner (transformers 5.x).
    from .vendor.bert_layers import BertForMaskedLM
    from .vendor.configuration_bert import BertConfig

    config = BertConfig.from_pretrained(str(snap))
    model = BertForMaskedLM(config)
    state = torch.load(snap / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise PinnedEnvironmentError(
            f"checkpoint/architecture mismatch: missing={list(missing)} unexpected={list(unexpected)}"
        )
    if not any(k.startswith("cls.predictions.decoder") for k in state):
        raise PinnedEnvironmentError("checkpoint has no MLM head; the native-MLM assay is impossible")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(str(snap))

    provenance = {
        "hf_id": DNABERT2_HF_ID,
        "revision": revision,
        "snapshot": str(snap),
        "device": device,
        "dtype": str(next(model.parameters()).dtype),
        "env": env,
        "vendor_sha256": vendor_hashes,
        "upstream_sha256": upstream_hashes,
        "n_layers": config.num_hidden_layers,
        "n_heads": config.num_attention_heads,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "has_learned_position_table": any("position_embeddings" in k for k in state),
        "attention_impl": "pytorch_reference (triton absent; flash_attn_triton.py not vendored)",
        "mlm_head_complete": True,
        "state_dict_keys": len(state),
    }
    return Dnabert2Bundle(
        model=model, tokenizer=tokenizer, config=config, snapshot=snap, provenance=provenance
    )


def identity_record(revision: str = DNABERT2_REVISION) -> dict[str, Any]:
    """Provenance block without paying for a full model load (hashes + versions only)."""
    env = require_pinned_env()
    snap = resolve_snapshot(revision)
    return {
        "hf_id": DNABERT2_HF_ID,
        "revision": revision,
        "env": env,
        "vendor_sha256": _verify_vendor(),
        "upstream_sha256": {n: sha256_file(snap / n) for n in UPSTREAM_SHA256},
    }


if __name__ == "__main__":  # pragma: no cover - provenance dump
    print(json.dumps(identity_record(), indent=2, sort_keys=True))
