"""v2 readouts.

RETRACTED v1/v2 SCAFFOLDING — no importers; retained deliberately as negative-result evidence.
    This module belongs to the retracted v1/v2 partner-selection line. The live donor-side readout
    is the low-rank `U_info` axis (C16-C18, replicated C26, bounded by C41), implemented in
    `readout.py` / `subspace.py`. Nothing here should be imported by new work.

v1's readout could not express partner binding at all:

    score(a,d) = f(h_a) + g(h_d) + L(len)     # no joint term

``g(h_d)`` is identical no matter which acceptor it is paired with, so the ranking of
donors never reads ``h_a``.  That is why "the bilinear readout does not beat the additive
one" was NOT evidence that partner information is linearly accessible -- the task simply
never required an interaction.

v2 provides:
  * ``BilinearPartnerReadout``  -- CAN express binding.  The hypothesis.
  * ``AdditiveDonorReadout``    -- CANNOT.  The falsification control (see below).
  * ``AcceptorQueryReadout``    -- reads h_a and the offset, never h_d.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class CandidateBatch:
    """Padded candidate donors for a batch of examples."""

    acceptor: torch.Tensor  # [B, D]      hidden state at the acceptor
    donor: torch.Tensor     # [B, C, D]   hidden states at each candidate donor
    offset: torch.Tensor    # [B, C]      donor_pos - acceptor_pos
    mask: torch.Tensor      # [B, C] bool
    labels: torch.Tensor    # [B]         index of the true donor

    def to(self, device) -> "CandidateBatch":
        return CandidateBatch(*[t.to(device) for t in (
            self.acceptor, self.donor, self.offset, self.mask, self.labels)])


def _offset_features(offset: torch.Tensor) -> torch.Tensor:
    x = offset.clamp_min(1.0)
    return torch.stack((torch.log(x), torch.sqrt(x) / 10.0), dim=-1)


class AdditiveDonorReadout(nn.Module):
    """score(d) = g(h_d) + L(offset).   NO dependence on the acceptor.

    ** THE FALSIFICATION CONTROL. **

    On the v2 task every candidate is a REAL annotated splice site, so ``g(h_d)`` --
    which can only express "how much does this look like a donor" -- has nothing to
    discriminate.  On the ``adversarial`` stratum the offset term is actively
    misleading (that stratum is defined as "the length prior prefers a decoy").

    Therefore this readout MUST land at or below chance.  If it does not, the dataset
    still contains a shortcut and the experiment is NOT identified -- stop and find it.
    Do not proceed to the bilinear result until this control has failed as predicted.
    """

    def __init__(self, hidden_size: int, site_hidden: int = 128):
        super().__init__()
        self.donor = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, site_hidden),
            nn.GELU(), nn.Linear(site_hidden, 1),
        )
        self.length = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, b: CandidateBatch) -> torch.Tensor:
        logits = self.donor(b.donor).squeeze(-1) + self.length(_offset_features(b.offset)).squeeze(-1)
        return logits.masked_fill(~b.mask, torch.finfo(logits.dtype).min)


class BilinearPartnerReadout(nn.Module):
    """Additive terms PLUS a low-rank acceptor x donor interaction.

    The interaction term ``<W_a h_a, W_d h_d>`` is the smallest form that can express
    "donor d is the partner OF acceptor a".  This is the hypothesis: if the frozen
    representation encodes partner identity, this readout can extract it and the
    additive control cannot.
    """

    def __init__(self, hidden_size: int, rank: int = 64, site_hidden: int = 128):
        super().__init__()
        self.additive = AdditiveDonorReadout(hidden_size, site_hidden)
        self.la = nn.LayerNorm(hidden_size)
        self.ld = nn.LayerNorm(hidden_size)
        self.wa = nn.Linear(hidden_size, rank, bias=False)
        self.wd = nn.Linear(hidden_size, rank, bias=False)
        self.scale = rank ** -0.5

    def interaction(self, b: CandidateBatch) -> torch.Tensor:
        a = self.wa(self.la(b.acceptor)).unsqueeze(1)   # [B,1,r]
        d = self.wd(self.ld(b.donor))                   # [B,C,r]
        return (a * d).sum(-1) * self.scale

    def forward(self, b: CandidateBatch) -> torch.Tensor:
        logits = self.additive(b) + self.interaction(b)
        return logits.masked_fill(~b.mask, torch.finfo(logits.dtype).min)


class AcceptorQueryReadout(nn.Module):
    """score(d) = phi(h_a, offset_d).  Uses the ACCEPTOR representation and the offset,
    but NEVER h_d.

    Donor site-detection is therefore unusable as a feature.  This asks directly:
    does the acceptor's own representation encode WHERE its partner is?  Its rival is
    the length prior (the offset alone).
    """

    def __init__(self, hidden_size: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, b: CandidateBatch) -> torch.Tensor:
        C = b.donor.shape[1]
        a = self.proj(b.acceptor).unsqueeze(1).expand(-1, C, -1)   # [B,C,H]
        x = torch.cat([a, _offset_features(b.offset)], dim=-1)
        logits = self.head(x).squeeze(-1)
        return logits.masked_fill(~b.mask, torch.finfo(logits.dtype).min)


def loss_fn(logits: torch.Tensor, b: CandidateBatch) -> torch.Tensor:
    return F.cross_entropy(logits, b.labels)


def top1(logits: torch.Tensor, b: CandidateBatch) -> torch.Tensor:
    return (logits.argmax(-1) == b.labels).float().mean()
