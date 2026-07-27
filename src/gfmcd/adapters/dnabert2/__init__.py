"""DNABERT-2 adapter: pinned, hash-verified load of the vendored MosaicBERT code.

Architecture-specific access only. Biological assay logic belongs in the reconstruction-atlas
package and study record, per `docs/TRANSFER_GUIDE.md`.
"""

from .adapter import Dnabert2Adapter, Dnabert2HookPoint
from .loader import (
    DNABERT2_HF_ID,
    DNABERT2_REVISION,
    Dnabert2Bundle,
    Dnabert2TokenizerBundle,
    identity_record,
    load_dnabert2_mlm,
    load_dnabert2_tokenizer,
    require_pinned_env,
)

__all__ = [
    "DNABERT2_HF_ID",
    "DNABERT2_REVISION",
    "Dnabert2Adapter",
    "Dnabert2Bundle",
    "Dnabert2HookPoint",
    "Dnabert2TokenizerBundle",
    "identity_record",
    "load_dnabert2_mlm",
    "load_dnabert2_tokenizer",
    "require_pinned_env",
]
