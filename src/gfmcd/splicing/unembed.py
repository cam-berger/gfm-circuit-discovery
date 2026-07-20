"""Logit-lens for the donor readout: apply the MLM head to any residual.

``logits = cls(hidden_states[-1])`` exactly (verified bit-exact on SpliceBERT.1024nt), so
the donor margin reads off any final-layer residual with no re-forward. This is the
backward-arm primitive of the L3H1 -> masked-donor-logit trace.

Prereg: ``studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_DONOR_ANCHOR.md``.
"""
from __future__ import annotations

import torch


def donor_logits(cls: torch.nn.Module, h: torch.Tensor) -> torch.Tensor:
    """Apply the MLM head ``cls`` to residuals ``h`` [..., hidden] -> logits [..., vocab]."""
    return cls(h)


def margin_hard(logits: torch.Tensor, bid: dict) -> torch.Tensor:
    """logit(G) - max(logit A,C,T). The frozen readout (matches ``CouplingTask.margin``)."""
    alt = torch.stack([logits[..., bid[b]] for b in "ACT"], -1).max(-1).values
    return logits[..., bid["G"]] - alt


def margin_smooth(logits: torch.Tensor, bid: dict) -> torch.Tensor:
    """logit(G) - logsumexp(logit A,C,T). Smooth surrogate for gradient direction-finding."""
    alt = torch.logsumexp(torch.stack([logits[..., bid[b]] for b in "ACT"], -1), -1)
    return logits[..., bid["G"]] - alt


def read_gradient(cls: torch.nn.Module, h: torch.Tensor, bid: dict) -> torch.Tensor:
    """d(margin_smooth)/dh per row -- the local readout direction. Returns [N, hidden]."""
    with torch.enable_grad():
        h = h.detach().clone().requires_grad_(True)
        margin_smooth(cls(h), bid).sum().backward()
        return h.grad.detach()
