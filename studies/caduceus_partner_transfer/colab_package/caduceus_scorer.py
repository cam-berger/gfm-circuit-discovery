"""Coordinate-correct masked-LM readout for Caduceus-PS (T1 transfer assay).

Mirrors gfmcd.splicing.coupling.CouplingScorer, but fixes the three Caduceus tokenizer facts that a naive
SpliceBERT port gets WRONG (red-team E2/E3/E6, empirically confirmed on the real tokenizer 2026-07-18):

  E2  prefix_tokens = 0 and a TRAILING [SEP] (id 1): tok('ACGT') -> [7,8,9,10,1].  Readout index = p, NOT p+1.
  E3  tokenizer padding_side = 'left' by default -> left-pad shifts every nucleotide index under batching.
      An SSM's REVERSE pass also makes right-padding unsafe (pads are processed before position p in reverse).
      => we forbid padding entirely: all constructs are FIXED 1024 nt, we batch only equal-length sequences.
  E6  A/C/G/T/[MASK] ids resolved dynamically via convert_tokens_to_ids (A=7 C=8 G=9 T=10 N=11 [MASK]=3).

margin(seq, p, base) = logit(base) - max(logit of the other three bases) at masked position p.
donor_margin masks the donor G; acceptor_margin masks the acceptor AG's G (at acceptor_pos-1).
"""
from __future__ import annotations
from typing import Sequence
import numpy as np


class CaduceusScorer:
    def __init__(self, model, tokenizer, device: str = "cuda", batch_size: int = 16,
                 window_length: int = 1024):
        import torch
        self.m, self.tok, self.dev, self.bs = model, tokenizer, device, batch_size
        self.torch = torch
        self.W = window_length
        # --- E6: dynamic ids ---
        self.base_id = {b: tokenizer.convert_tokens_to_ids(b) for b in "ACGT"}
        self.mask_id = tokenizer.mask_token_id
        assert self.mask_id is not None, "tokenizer has no [MASK]"
        assert all(v is not None and v >= 0 for v in self.base_id.values()), f"bad base ids {self.base_id}"
        # --- E2: verify prefix_tokens == 0 and a single trailing [SEP] on a raw probe ---
        probe = tokenizer("ACGT")["input_ids"]
        assert probe[:4] == [self.base_id[b] for b in "ACGT"], f"prefix!=0 or reordered: {probe}"
        assert len(probe) == 5 and probe[4] == tokenizer.sep_token_id, f"expected trailing [SEP]: {probe}"
        self.prefix = 0
        self.suffix_len = len(probe) - 4          # trailing specials (1 = [SEP])
        # --- E3: force right padding as defence; but we also assert equal-length per batch below ---
        try:
            tokenizer.padding_side = "right"
        except Exception:
            pass
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.m.eval()

    def _encode(self, seqs: Sequence[str]):
        """Tokenize an EQUAL-LENGTH batch (no padding) -> input_ids on device. Asserts equal length (SSM-safe)."""
        torch = self.torch
        lens = {len(s) for s in seqs}
        assert len(lens) == 1, f"CaduceusScorer batches must be equal-length (no padding on an SSM); got {sorted(lens)}"
        ids = [self.tok(s)["input_ids"] for s in seqs]     # each: [nt ids...] + [SEP]
        L = {len(x) for x in ids}
        assert len(L) == 1, f"tokenized lengths differ: {sorted(L)}"
        t = torch.tensor(ids, dtype=torch.long, device=self.dev)
        return t

    def _logits(self, input_ids):
        """Forward -> per-position vocab logits (float32). Caduceus (Mamba) takes input_ids; no attention_mask."""
        torch = self.torch
        with torch.no_grad():
            try:
                out = self.m(input_ids=input_ids)
            except TypeError:
                out = self.m(input_ids)
        lg = getattr(out, "logits", out)
        return lg.float()

    def _margin(self, seqs: Sequence[str], positions: Sequence[int], base: str) -> np.ndarray:
        others = [b for b in "ACGT" if b != base]
        out: list[float] = []
        for i in range(0, len(seqs), self.bs):
            S, P = list(seqs[i:i + self.bs]), list(positions[i:i + self.bs])
            ids = self._encode(S).clone()
            for j, p in enumerate(P):
                ids[j, p + self.prefix] = self.mask_id            # E2: prefix=0
            lg = self._logits(ids)
            for j, p in enumerate(P):
                v = lg[j, p + self.prefix]
                hit = float(v[self.base_id[base]])
                alt = max(float(v[self.base_id[b]]) for b in others)
                out.append(hit - alt)
        return np.asarray(out)

    def donor_margin(self, seqs, donor_pos) -> np.ndarray:
        return self._margin(seqs, list(donor_pos), "G")

    def acceptor_margin(self, seqs, acceptor_pos) -> np.ndarray:
        return self._margin(seqs, [p - 1 for p in acceptor_pos], "G")

    def mlm_health(self, seqs, n_probe: int = 20, seed: int = 0) -> float:
        import random
        rng = random.Random(seed)
        ps, pp, pb = [], [], []
        for s in seqs:
            for _ in range(n_probe):
                p = rng.randrange(10, len(s) - 10)
                ps.append(s); pp.append(p); pb.append(s[p])
        vals = []
        for b in "ACGT":
            idx = [i for i, x in enumerate(pb) if x == b]
            if idx:
                vals.append(self._margin([ps[i] for i in idx], [pp[i] for i in idx], b))
        return float(np.concatenate(vals).mean()) if vals else float("nan")
