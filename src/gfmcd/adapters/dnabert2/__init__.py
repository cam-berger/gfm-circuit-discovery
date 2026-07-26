"""DNABERT-2 adapter: pinned, hash-verified load of the vendored MosaicBERT code.

Architecture-specific access only. Biological assay logic belongs in the study package
(`studies/dnabert2_core_promoter/`), per `docs/TRANSFER_GUIDE.md`.
"""

from .loader import (
    DNABERT2_HF_ID,
    DNABERT2_REVISION,
    Dnabert2Bundle,
    identity_record,
    load_dnabert2_mlm,
    require_pinned_env,
)

__all__ = [
    "DNABERT2_HF_ID",
    "DNABERT2_REVISION",
    "Dnabert2Bundle",
    "identity_record",
    "load_dnabert2_mlm",
    "require_pinned_env",
]
