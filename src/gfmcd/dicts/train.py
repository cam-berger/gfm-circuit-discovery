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


def explained_variance(model: Transcoder, x: torch.Tensor, y: torch.Tensor | None = None) -> float:
    """1 - FVU. Element-wise variance flatters a target whose dimensions are mostly zero."""
    y = x if y is None else y
    with torch.no_grad():
        sse = (y - model(x)).pow(2).sum()
        sst = (y - y.mean(dim=0)).pow(2).sum()
    return float(1.0 - sse / sst)


def train_transcoder(
    x: torch.Tensor,
    y: torch.Tensor,
    n_features: int,
    k: int,
    steps: int = 4000,
    lr: float = 1e-3,
    seed: int = 0,
) -> Transcoder:
    """Encode `x` (an MLP's input), decode `y` (that MLP's output). Nodes of the graph."""
    torch.manual_seed(seed)
    tc = Transcoder(d_in=x.shape[-1], d_out=y.shape[-1], n_features=n_features, k=k)
    with torch.no_grad():
        tc.b_dec.copy_(y.mean(dim=0))

    opt = torch.optim.Adam(tc.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(tc(x), y).backward()
        opt.step()
    return tc
