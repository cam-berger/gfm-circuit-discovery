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
from .promoter_classifier import (
    PROMOTER_CHECKPOINTS,
    PROMOTER_GUE_DATASET_ID,
    PROMOTER_GUE_DATASET_REVISION,
    Dnabert2PromoterBundle,
    Dnabert2PromoterClassifier,
    PromoterCheckpoint,
    PromoterMarginScore,
    load_dnabert2_promoter_classifier,
    resolve_promoter_snapshot,
    verify_tokenizer_equivalence,
)

__all__ = [
    "DNABERT2_HF_ID",
    "DNABERT2_REVISION",
    "PROMOTER_CHECKPOINTS",
    "PROMOTER_GUE_DATASET_ID",
    "PROMOTER_GUE_DATASET_REVISION",
    "Dnabert2Adapter",
    "Dnabert2Bundle",
    "Dnabert2HookPoint",
    "Dnabert2PromoterBundle",
    "Dnabert2PromoterClassifier",
    "Dnabert2TokenizerBundle",
    "PromoterCheckpoint",
    "PromoterMarginScore",
    "identity_record",
    "load_dnabert2_mlm",
    "load_dnabert2_promoter_classifier",
    "load_dnabert2_tokenizer",
    "require_pinned_env",
    "resolve_promoter_snapshot",
    "verify_tokenizer_equivalence",
]
