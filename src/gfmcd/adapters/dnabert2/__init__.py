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
from .promoter_head_atlas import (
    PROMOTER_SOURCE_BANDS,
    Head,
    PairedPromoterBatch,
    all_promoter_heads,
    bidirectional_patch_effects_for_batch,
    capture_cls_attention_bands,
    head_label,
    paired_head_attributions,
    patch_effect_for_batch,
    patch_paired_cls_heads,
    promoter_margin_tensor,
    tokenize_promoter_pairs,
)

__all__ = [
    "DNABERT2_HF_ID",
    "DNABERT2_REVISION",
    "PROMOTER_CHECKPOINTS",
    "PROMOTER_GUE_DATASET_ID",
    "PROMOTER_GUE_DATASET_REVISION",
    "PROMOTER_SOURCE_BANDS",
    "Dnabert2Adapter",
    "Dnabert2Bundle",
    "Dnabert2HookPoint",
    "Dnabert2PromoterBundle",
    "Dnabert2PromoterClassifier",
    "Dnabert2TokenizerBundle",
    "Head",
    "PairedPromoterBatch",
    "PromoterCheckpoint",
    "PromoterMarginScore",
    "all_promoter_heads",
    "bidirectional_patch_effects_for_batch",
    "capture_cls_attention_bands",
    "head_label",
    "identity_record",
    "load_dnabert2_mlm",
    "load_dnabert2_promoter_classifier",
    "load_dnabert2_tokenizer",
    "paired_head_attributions",
    "patch_effect_for_batch",
    "patch_paired_cls_heads",
    "promoter_margin_tensor",
    "require_pinned_env",
    "resolve_promoter_snapshot",
    "tokenize_promoter_pairs",
    "verify_tokenizer_equivalence",
]
