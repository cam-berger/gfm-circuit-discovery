"""Score a learned dictionary against planted feature directions.

Traced edges live in feature space; planted edges live in weight space. There is no
identity map between them, which is why a single end-to-end recall number confounds
dictionary error with attribution error. This module measures the dictionary half alone:
match learned decoder directions to the planted ones, then ask whether the matched
features fire on the same inputs.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def hungarian_match(learned: torch.Tensor, planted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign each planted direction its best learned decoder row.

    Returns (index into `learned` per planted feature, |cosine| at that assignment).
    Sign is ignored: a dictionary may recover a direction with either polarity.
    """
    l = learned / learned.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    p = planted / planted.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos = (p @ l.T).abs()

    rows, cols = linear_sum_assignment(-cos.detach().numpy())
    idx = torch.as_tensor(cols, dtype=torch.long)
    return idx, cos[torch.as_tensor(rows), idx].detach()


def mcc(true_active: torch.Tensor, pred_active: torch.Tensor) -> float:
    """Matthews correlation over the flattened support masks."""
    t = true_active.bool().flatten()
    p = pred_active.bool().flatten()

    tp = float((t & p).sum())
    tn = float((~t & ~p).sum())
    fp = float((~t & p).sum())
    fn = float((t & ~p).sum())

    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom
