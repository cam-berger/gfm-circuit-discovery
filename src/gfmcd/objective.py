"""Objective accounting for the copy-position-weighted AR loss (TO_FIX.md §4.4).

The repeat-copy model is trained with copy positions upweighted (default 8x). That is a forcing
signal, so the objective must be reported, not described as "ordinary." These helpers make the weight
auditable: the effective share of the loss that copy positions carry, and the canonical weighted loss
(shared by the training script and the tests, so weight=1 is provably ordinary mean cross-entropy).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def effective_copy_loss_share(copy_mask, copy_weight: float) -> float:
    """Fraction of total loss weight carried by copy positions under `copy_weight`.

    Monotone increasing in `copy_weight`; equals the raw copy-token fraction at weight 1.
    Accepts a torch tensor or anything with `.sum()`/`.numel()` (e.g. numpy bool array).
    """
    copy_tokens = float(copy_mask.sum())
    total_tokens = float(copy_mask.size if hasattr(copy_mask, "size") and not callable(copy_mask.size) else copy_mask.numel())
    background = total_tokens - copy_tokens
    denom = copy_weight * copy_tokens + background
    return (copy_weight * copy_tokens) / denom if denom else 0.0


def weighted_ar_loss(
    logits: torch.Tensor, targets: torch.Tensor, copy_mask: torch.Tensor, copy_weight: float
) -> torch.Tensor:
    """Causal CE with copy positions upweighted by `copy_weight`.

    At `copy_weight == 1` this is exactly the ordinary mean cross-entropy (see test).
    """
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).view_as(targets)
    weights = torch.ones_like(loss)
    weights[copy_mask] = copy_weight
    return (loss * weights).sum() / weights.sum()


def objective_accounting(
    logits: torch.Tensor, targets: torch.Tensor, copy_mask: torch.Tensor, copy_weight: float
) -> dict[str, float]:
    """Per-run objective breakdown for logging (TO_FIX.md §4.4)."""
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").view_as(targets)
    bg_mask = ~copy_mask
    return {
        "copy_weight": float(copy_weight),
        "copy_token_fraction": float(copy_mask.float().mean()),
        "effective_copy_loss_share": effective_copy_loss_share(copy_mask, copy_weight),
        "copy_ce": float(ce[copy_mask].mean()) if copy_mask.any() else float("nan"),
        "background_ce": float(ce[bg_mask].mean()) if bg_mask.any() else float("nan"),
        "all_token_ce": float(ce.mean()),
    }
