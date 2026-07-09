"""A small single-nucleotide transformer, trained by us, on which the objective can be varied.

This is the substrate that carries Rung 1 and, later, H2's four objective arms. Pretrained GFMs
cannot serve H2: it needs matched architecture and scale with only the loss changing, and no such
family exists on the Hub (D1).

Two things differ from `TinyTransformer` on purpose:

  * **LayerNorm is present.** Rung 0's planted models are LN-free so Test A could measure attribution
    error and nothing else. Real backbones have LN, the replacement model must freeze it, and that
    introduces an approximation whose cost is currently *unmeasured*. Everything the tracer knows
    about its own error rate was learned in a world without LayerNorm.
  * **Absolute learned position embeddings**, as SpliceBERT has. This is exactly why `data/genome.py`
    randomises window offsets: with ATG-aligned windows the model would read codon phase straight off
    `pos_embed` and learn a lookup table (D21).

`causal=True` gives the AR arm, `causal=False` the MLM arm. Everything else is identical, so the H2
contrast is a contrast of objectives and not of architectures.

Cache keys mirror `TinyTransformer` so the tracer generalises without a second code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

A, C, G, T, N, MASK = range(6)
VOCAB = "ACGTN?"


@dataclass(frozen=True)
class NanoConfig:
    vocab: int = 6
    d_model: int = 256
    n_heads: int = 8
    d_mlp: int = 1024
    n_layers: int = 6
    max_len: int = 256
    causal: bool = False

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


class Attention(nn.Module):
    def __init__(self, cfg: NanoConfig):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_O = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: dict, layer: int) -> torch.Tensor:
        b, n, _ = x.shape
        h, dh = self.cfg.n_heads, self.cfg.d_head
        shape = lambda t: t.view(b, n, h, dh)

        q, k, v = shape(self.W_Q(x)), shape(self.W_K(x)), shape(self.W_V(x))
        scores = torch.einsum("bphc,bqhc->bhpq", q, k) / dh**0.5
        if self.cfg.causal:
            mask = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))

        cache[f"L{layer}.attn.scores"] = scores
        cache[f"L{layer}.attn.q"] = q
        pattern = scores.softmax(dim=-1)
        cache[f"L{layer}.attn.pattern"] = pattern

        z = torch.einsum("bhpq,bqhc->bphc", pattern, v)
        cache[f"L{layer}.attn.z"] = z
        return self.W_O(z.reshape(b, n, -1))


class MLP(nn.Module):
    def __init__(self, cfg: NanoConfig):
        super().__init__()
        self.W_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.W_out = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x: torch.Tensor, cache: dict, layer: int) -> torch.Tensor:
        cache[f"L{layer}.mlp.in"] = x
        acts = F.gelu(self.W_in(x))
        cache[f"L{layer}.mlp.acts"] = acts
        out = self.W_out(acts)
        cache[f"L{layer}.mlp.out"] = out
        return out


class Block(nn.Module):
    def __init__(self, cfg: NanoConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, resid: torch.Tensor, cache: dict, layer: int) -> torch.Tensor:
        resid = resid + self.attn(self.ln1(resid), cache, layer)
        cache[f"L{layer}.resid_mid"] = resid
        resid = resid + self.mlp(self.ln2(resid), cache, layer)
        cache[f"L{layer}.resid_post"] = resid
        return resid


class GFMNano(nn.Module):
    def __init__(self, cfg: NanoConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        cache = {} if cache is None else cache
        n = tokens.shape[-1]
        pos = torch.arange(n, device=tokens.device)

        resid = self.embed(tokens) + self.pos_embed(pos)
        cache["resid.embed"] = resid
        for layer, block in enumerate(self.blocks):
            resid = block(resid, cache, layer)

        resid = self.ln_f(resid)
        cache["resid.final"] = resid
        return self.unembed(resid)

    def run_with_cache(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cache: dict = {}
        return self(tokens, cache), cache

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def ar_batch(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-nucleotide prediction. Inputs are the sequence; targets are it shifted left."""
    return tokens[:, :-1], tokens[:, 1:]


def mlm_batch(
    tokens: torch.Tensor,
    rate: float = 0.15,
    generator: torch.Generator | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT-style corruption: 80% [MASK], 10% random nucleotide, 10% left alone.

    The 10%-unchanged branch is what stops the model learning "[MASK] means predict, anything else
    means copy" -- it must model every position, not only the corrupted ones.
    """
    device = tokens.device
    inputs = tokens.clone()
    targets = torch.full_like(tokens, ignore_index)

    r = torch.rand(tokens.shape, device=device, generator=generator)
    selected = r < rate
    targets[selected] = tokens[selected]

    branch = torch.rand(tokens.shape, device=device, generator=generator)
    inputs[selected & (branch < 0.8)] = MASK
    random_branch = selected & (branch >= 0.8) & (branch < 0.9)
    inputs[random_branch] = torch.randint(
        0, 4, (int(random_branch.sum()),), device=device, generator=generator, dtype=tokens.dtype
    )
    return inputs, targets
