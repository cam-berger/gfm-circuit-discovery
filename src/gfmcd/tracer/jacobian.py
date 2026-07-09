"""H3's discriminator: how sparse is the computation, not the activations.

For a ReLU MLP the input-feature -> output-feature Jacobian is analytic and cheap:

    d(mlp_out) / d(mlp_in) = W_in @ diag(1[pre > 0]) @ W_out

Run at neuron level (identity dictionary) the "features" are residual dimensions, and no
dictionary is trained -- which is what we want here, because the planted twins' neurons are
monosemantic by construction and their residual code is not identifiable anyway (see D14).

Hoyer sparsity is used rather than a raw L0 count so the measure is scale-free and reacts to
how mass is *distributed* across edges, not merely how many are nonzero.
"""

from __future__ import annotations

import torch

from gfmcd.models.tiny import TinyTransformer


@torch.no_grad()
def neuron_jacobian(model: TinyTransformer, tokens: torch.Tensor, layer: int = 1) -> torch.Tensor:
    """[batch, pos, d_model, d_model] -- d(MLP output) / d(MLP input), per token position."""
    _, cache = model.run_with_cache(tokens)
    pre = cache[f"L{layer}.resid_mid"] @ model.mlp[layer].W_in + model.mlp[layer].b_in
    gate = (pre > 0).to(pre.dtype)
    return torch.einsum("ij,bpj,jk->bpik", model.mlp[layer].W_in, gate, model.mlp[layer].W_out)


def hoyer_sparsity(jac: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """1.0 when one edge carries everything; 0.0 when mass is spread uniformly."""
    flat = jac.flatten(start_dim=-2)
    n = flat.shape[-1]
    l1 = flat.abs().sum(dim=-1)
    l2 = flat.pow(2).sum(dim=-1).sqrt()
    return (n**0.5 - l1 / (l2 + eps)) / (n**0.5 - 1)


def active_edge_count(jac: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (jac.abs() > eps).flatten(start_dim=-2).sum(dim=-1)


def auc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    """P(a random positive scores above a random negative); ties count a half."""
    p, n = positive.flatten(), negative.flatten()
    greater = (p.unsqueeze(1) > n.unsqueeze(0)).sum()
    equal = (p.unsqueeze(1) == n.unsqueeze(0)).sum()
    return float((greater + 0.5 * equal) / (p.numel() * n.numel()))
