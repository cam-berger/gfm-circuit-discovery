"""Fit a dictionary to activations. Deliberately boring; Test B is about what it recovers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gfmcd.dicts.base import Transcoder


def train_sae(
    acts: torch.Tensor,
    n_features: int,
    k: int,
    steps: int = 4000,
    lr: float = 1e-3,
    seed: int = 0,
) -> Transcoder:
    """TopK SAE. `acts` is [N, d]; the reconstruction target is `acts` itself.

    The decoder is *not* renormalised between steps. Unit-norming decoder rows is standard
    for L1 SAEs, where it stops the sparsity penalty being gamed by shrinking features and
    inflating codes. A TopK SAE has no L1 term, so the renormalisation buys nothing and
    fights the optimiser: with it, recovered directions reach |cos| = 0.39 against planted
    ground truth; without it, 0.998.
    """
    torch.manual_seed(seed)
    d = acts.shape[-1]
    sae = Transcoder(d_in=d, d_out=d, n_features=n_features, k=k)

    with torch.no_grad():
        sae.W_dec.copy_(sae.W_enc.T / sae.W_enc.T.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        sae.b_dec.copy_(acts.mean(dim=0))

    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(sae(acts), acts).backward()
        opt.step()

    return sae


def explained_variance(sae: Transcoder, acts: torch.Tensor) -> float:
    with torch.no_grad():
        resid = acts - sae(acts)
    return float(1.0 - resid.var() / acts.var())
