"""One interface, several dictionaries.

The proposal's method table lists SAEs, transcoders, cross-layer transcoders, JSAEs and
crosscoders as five methods from five papers. In code they are one abstraction:
encode activations to sparse features, decode back, expose a Jacobian. Building the
interface once is what makes the ladder tractable for one person.

    SAE          encode/decode the same activation
    Transcoder   encode MLP input, decode MLP output       <- ship this first
    JSAE         Transcoder with a sparsified output side, Jacobian penalised
    CLT          Transcoder whose decoders write to layers l..L
    Crosscoder   CUT -- see docs/DECISIONS.md

Transcoder is first because it is the simplest object yielding feature->feature edges
through an MLP, and it validates against the planted circuit immediately. Starting at
CLT means debugging the hardest dictionary against an unvalidated tracer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dictionary(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Dense activations -> sparse feature activations."""

    @abstractmethod
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Sparse feature activations -> dense reconstruction of the target."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @property
    @abstractmethod
    def decoder_weight(self) -> torch.Tensor:
        """[n_features, out_dim]. Row i is feature i's write direction."""


class Transcoder(Dictionary):
    """Reads an MLP's input, predicts its output. Its features are the graph's nodes."""

    def __init__(self, d_in: int, d_out: int, n_features: int, k: int | None = None):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(d_in, n_features) / d_in**0.5)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.randn(n_features, d_out) / n_features**0.5)
        self.b_dec = nn.Parameter(torch.zeros(d_out))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        if self.k is None:
            return F.relu(pre)
        vals, idx = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, F.relu(vals))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    @property
    def decoder_weight(self) -> torch.Tensor:
        return self.W_dec

    def loss(self, x: torch.Tensor, target: torch.Tensor, l1: float = 0.0) -> torch.Tensor:
        f = self.encode(x)
        recon = F.mse_loss(self.decode(f), target)
        return recon + l1 * f.abs().sum(dim=-1).mean()


class JSAE(nn.Module):
    """A transcoder whose output side is also sparse, with a penalised Jacobian.

    H3's discriminator. For ReLU dictionaries the feature-to-feature Jacobian is analytic
    and cheap -- that is the whole point of Farnik et al.:

        out_feats = relu(W_enc_out^T · mlp_out + b)
        mlp_out   = W_dec_in^T · in_feats
        d out_feats / d in_feats = diag(1[out_pre > 0]) · W_enc_out^T · W_dec_in

    A compositional circuit routes through few feature pairs, so this Jacobian is sparse.
    A bag-of-motifs diffuses across many, so it is dense. `jacobian_sparsity` is the
    scalar we ROC-calibrate on the planted twins BEFORE applying it to trained models.
    """

    def __init__(self, transcoder: Transcoder, out_sae: Dictionary):
        super().__init__()
        self.transcoder = transcoder
        self.out_sae = out_sae

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """[..., n_out_features, n_in_features]"""
        in_feats = self.transcoder.encode(x)
        mlp_out = self.transcoder.decode(in_feats)
        out_pre = mlp_out @ self.out_sae.W_enc + self.out_sae.b_enc
        gate = (out_pre > 0).to(x.dtype)
        core = self.out_sae.W_enc.T @ self.transcoder.W_dec.T
        return gate.unsqueeze(-1) * core

    @staticmethod
    def jacobian_sparsity(jac: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Normalised L1/L2 ratio: 1.0 when one edge carries everything, low when diffuse."""
        flat = jac.flatten(start_dim=-2)
        l1 = flat.abs().sum(dim=-1)
        l2 = flat.pow(2).sum(dim=-1).sqrt()
        n = flat.shape[-1]
        hoyer = (n**0.5 - l1 / (l2 + eps)) / (n**0.5 - 1)
        return hoyer


class CLT(Dictionary):
    """Cross-layer transcoder: encode layer l's MLP input, decode into layers l..L.

    Deferred until Transcoder passes the planted-circuit tests. Sizing for SpliceBERT
    (6 layers, d=512): 4096 features ~= 44M params; 16384 features ~= 175M. Both fit in
    68 GB unified memory.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CLT lands after Transcoder clears tests A/B/C. See docs/PLAN.md")

    def encode(self, x): ...
    def decode(self, f): ...
    @property
    def decoder_weight(self): ...
