"""Objective accounting for the repeat-copy weighted loss (TO_FIX.md §4.5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gfmcd.objective import effective_copy_loss_share, objective_accounting, weighted_ar_loss


def _fixture(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(4, 16, 4, generator=g)
    targets = torch.randint(0, 4, (4, 16), generator=g)
    copy_mask = torch.zeros(4, 16, dtype=torch.bool)
    copy_mask[:, 8:] = True  # half the positions are copy positions
    return logits, targets, copy_mask


def test_weight_one_is_ordinary_mean_cross_entropy():
    logits, targets, mask = _fixture()
    w1 = weighted_ar_loss(logits, targets, mask, copy_weight=1.0)
    plain = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
    assert torch.allclose(w1, plain, atol=1e-6), "copy_weight=1 must equal ordinary mean CE"


def test_effective_share_monotone_in_weight():
    _, _, mask = _fixture()
    shares = [effective_copy_loss_share(mask, w) for w in (1, 2, 4, 8)]
    assert shares == sorted(shares)
    assert all(shares[i] < shares[i + 1] for i in range(len(shares) - 1))
    # At weight 1 the effective share equals the raw copy-token fraction.
    assert abs(shares[0] - float(mask.float().mean())) < 1e-6


def test_effective_share_formula():
    mask = torch.tensor([True, True, False, False, False, False, False, False])  # 2/8 copy
    # weight 3: (3*2)/(3*2 + 6) = 6/12 = 0.5
    assert abs(effective_copy_loss_share(mask, 3.0) - 0.5) < 1e-9


def test_accounting_keys_and_weighting_effect():
    logits, targets, mask = _fixture()
    acc = objective_accounting(logits, targets, mask, copy_weight=8.0)
    for k in ("copy_weight", "copy_token_fraction", "effective_copy_loss_share", "copy_ce", "background_ce", "all_token_ce"):
        assert k in acc
    assert acc["copy_weight"] == 8.0
    # Upweighting copy positions raises the loss above the unweighted mean iff copy CE > background CE.
    w8 = float(weighted_ar_loss(logits, targets, mask, 8.0))
    w1 = float(weighted_ar_loss(logits, targets, mask, 1.0))
    if acc["copy_ce"] > acc["background_ce"]:
        assert w8 > w1
