"""Planted MOCK Caduceus for LOCAL dry-run validation (no CUDA, no mamba_ssm).

Mimics the Caduceus interface: forward(input_ids) -> object with .logits of shape (B, L, vocab).
Plants KNOWN behaviour so the T1 harness's memorials/estimands provably FIRE end-to-end before any GPU time:
  * DONOR recognition: a masked G whose next base is T ("G|T" donor) gets a strong logit(G) boost
    => margin_donor(clean) > 0; destroying the donor GT collapses it => A_donor > 0.
  * PARTNER coupling: an intact downstream AG in [p+MIN, p+MAX] adds a DISTANCE-DECAYING logit(G) boost at p
    => corrupting the partner AG (AG->CC) drops the donor margin => coupling C > 0, partner-specific,
       and the long-range transmission curve decays with offset.
  * weak MLM competence: local-composition copy (predict the commonest base in [p-5,p+5]) => mlm_health > 0.
This is a PLANTED-POSITIVE oracle, NOT Caduceus; it validates plumbing + decision logic, not biology.
"""
from __future__ import annotations
import math
import torch


class _Out:
    def __init__(self, logits): self.logits = logits


class MockCaduceus:
    VOCAB = 16

    def __init__(self, base_ids=None, donor_boost=3.0, couple_amp=2.0, lam=90.0,
                 dmin=8, dmax=250, copy_amp=0.5, win=5, seed=0):
        # base_ids resolved from the real tokenizer (A=7,C=8,G=9,T=10 for Caduceus)
        self.bid = base_ids or {"A": 7, "C": 8, "G": 9, "T": 10}
        self.A, self.C, self.G, self.T = (self.bid[b] for b in "ACGT")
        self.donor_boost, self.couple_amp, self.lam = donor_boost, couple_amp, lam
        self.dmin, self.dmax, self.copy_amp, self.win = dmin, dmax, copy_amp, win
        self.g = torch.Generator().manual_seed(seed)
        self._params = [torch.zeros(1)]                 # so .parameters() is non-empty / requires_grad works

    def parameters(self):
        return iter(self._params)

    def eval(self):
        return self

    def to(self, *a, **k):
        return self

    @torch.no_grad()
    def __call__(self, input_ids=None, **kw):
        if input_ids is None:
            input_ids = kw["input_ids"]
        ids = input_ids
        B, L = ids.shape
        V = self.VOCAB
        logits = torch.zeros(B, L, V, dtype=torch.float32)   # deterministic (no noise; planted signal dominates)

        is_A = (ids == self.A); is_G = (ids == self.G); is_T = (ids == self.T)
        # --- DONOR recognition: boost logit(G) at p if base at p+1 is T (localizes the donor readout) ---
        Tnext = torch.zeros(B, L, dtype=torch.bool)
        Tnext[:, :-1] = is_T[:, 1:]
        logits[:, :, self.G] += self.donor_boost * Tnext.float()

        # --- PARTNER coupling: only an AG carrying the PARTNER TAG (TGCA immediately upstream) couples, so
        #     the TRUE partner is distinguishable from a plain downstream AG (partner-specificity); decaying;
        #     gated to donor positions. (A plain acceptor AG has recognition but no partner tag.) ---
        AG0 = torch.zeros(B, L, dtype=torch.bool)
        AG0[:, :-1] = is_A[:, :-1] & is_G[:, 1:]        # AG dinucleotide starts at q
        Tg, Gg, Cg, Ag = (ids == self.T), (ids == self.G), (ids == self.C), (ids == self.A)
        tag = torch.zeros(B, L, dtype=torch.bool)       # TGCA at [q-4..q-1]
        tag[:, 4:] = Tg[:, :-4] & Gg[:, 1:-3] & Cg[:, 2:-2] & Ag[:, 3:-1]
        AG = AG0 & tag
        couple = torch.zeros(B, L, dtype=torch.float32)
        for delta in range(self.dmin, self.dmax + 1):
            w = self.couple_amp * math.exp(-delta / self.lam)
            if w < 1e-3:
                break
            src = torch.zeros(B, L, dtype=torch.bool)
            src[:, :L - delta] = AG[:, delta:]          # AG located delta downstream of p
            couple += w * src.float()
        logits[:, :, self.G] += couple * Tnext.float()  # coupling modulates DONOR confidence only

        # --- weak MLM competence: local-composition copy (commonest base in [p-win, p+win]) ---
        for b, bid in self.bid.items():
            occ = (ids == bid).float()
            k = 2 * self.win + 1
            pad = torch.zeros(B, self.win)
            padded = torch.cat([pad, occ, pad], dim=1)               # (B, L+2win)
            cs = torch.cat([torch.zeros(B, 1), torch.cumsum(padded, dim=1)], dim=1)  # (B, L+2win+1)
            local = cs[:, k:] - cs[:, :-k]                            # (B, L) window count of base b
            logits[:, :, bid] += self.copy_amp * local
        return _Out(logits)
