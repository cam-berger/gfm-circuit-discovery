"""A minimal transformer with every internal activation exposed.

Deliberately has no LayerNorm. Planted circuits must be exactly analysable, and LN couples
every residual dimension to every other one. Real backbones (SpliceBERT, GFM-nano) do have
LN; the tracer freezes it, which is the standard replacement-model assumption. Keeping L0
LN-free means Test A measures attribution error and nothing else.

`attn_bias` lets a planted head hold a fixed pattern (e.g. "attend one position left")
without pretending to derive it from QK. Freezing patterns is what the tracer does anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9


@dataclass(frozen=True)
class TinyConfig:
    vocab: int = 5
    d_model: int = 36
    n_heads: int = 5
    d_head: int = 5
    d_mlp: int = 25
    n_layers: int = 2
    n_out: int = 3
    max_len: int = 256
    layernorm: bool = False  # off by default -> byte-identical to the LN-free model (Test A substrate)
    ln_eps: float = 1e-5


class FreezableLayerNorm(nn.Module):
    """Pre-LN with optionally *frozen* statistics.

    The tracer's replacement model freezes LN — it reuses the mean/std from a clean pass rather than
    recomputing them under intervention. That is the approximation whose cost §16 measures. Passing
    `frozen=(mean, std)` reproduces the replacement model; passing `frozen=None` recomputes (ground
    truth). Init is exact identity (gamma=1, beta=0), so a frozen (mean=0, std=1) LN is a no-op.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x: torch.Tensor, frozen: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        if frozen is not None:
            mean, std = frozen
        else:
            mean = x.mean(-1, keepdim=True)
            std = (x.var(-1, keepdim=True, unbiased=False) + self.eps).sqrt()
        return (x - mean) / std * self.gamma + self.beta


class Attention(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.W_Q = nn.Parameter(torch.zeros(cfg.d_model, cfg.n_heads, cfg.d_head))
        self.W_K = nn.Parameter(torch.zeros(cfg.d_model, cfg.n_heads, cfg.d_head))
        self.W_V = nn.Parameter(torch.zeros(cfg.d_model, cfg.n_heads, cfg.d_head))
        self.W_O = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.register_buffer("attn_bias", torch.zeros(cfg.n_heads, cfg.max_len, cfg.max_len))

    def forward(self, resid: torch.Tensor, cache: dict, layer: int) -> torch.Tensor:
        n = resid.shape[1]
        q = torch.einsum("bpd,dhc->bphc", resid, self.W_Q)
        k = torch.einsum("bqd,dhc->bqhc", resid, self.W_K)
        v = torch.einsum("bqd,dhc->bqhc", resid, self.W_V)

        scores = torch.einsum("bphc,bqhc->bhpq", q, k) + self.attn_bias[:, :n, :n]
        cache[f"L{layer}.attn.scores"] = scores
        cache[f"L{layer}.attn.q"] = q
        pattern = scores.softmax(dim=-1)
        cache[f"L{layer}.attn.pattern"] = pattern

        z = torch.einsum("bhpq,bqhc->bphc", pattern, v)
        cache[f"L{layer}.attn.z"] = z
        return torch.einsum("bphc,hcd->bpd", z, self.W_O)


class MLP(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.W_in = nn.Parameter(torch.zeros(cfg.d_model, cfg.d_mlp))
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.zeros(cfg.d_mlp, cfg.d_model))

    def forward(self, resid: torch.Tensor, cache: dict, layer: int) -> torch.Tensor:
        acts = F.relu(resid @ self.W_in + self.b_in)
        cache[f"L{layer}.mlp.acts"] = acts
        out = acts @ self.W_out
        cache[f"L{layer}.mlp.out"] = out
        return out


class TinyTransformer(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Parameter(torch.zeros(cfg.vocab, cfg.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(cfg.max_len, cfg.d_model))
        self.attn = nn.ModuleList(Attention(cfg) for _ in range(cfg.n_layers))
        self.mlp = nn.ModuleList(MLP(cfg) for _ in range(cfg.n_layers))
        if cfg.layernorm:
            self.ln1 = nn.ModuleList(FreezableLayerNorm(cfg.d_model, cfg.ln_eps) for _ in range(cfg.n_layers))
            self.ln2 = nn.ModuleList(FreezableLayerNorm(cfg.d_model, cfg.ln_eps) for _ in range(cfg.n_layers))
        self.W_U = nn.Parameter(torch.zeros(cfg.d_model, cfg.n_out))
        self.register_buffer("readout_gate", torch.ones(cfg.max_len))

    def _norm(self, resid, which: str, layer: int, cache: dict, frozen: dict | None):
        """Apply pre-LN if enabled, cache the stats used, and honour frozen stats."""
        if not self.cfg.layernorm:
            return resid
        ln = (self.ln1 if which == "ln1" else self.ln2)[layer]
        key = f"L{layer}.{which}"
        frz = frozen.get(key) if frozen else None
        out = ln(resid, frozen=frz)
        # cache the stats actually used, so a later pass can freeze them.
        if frz is not None:
            mean, std = frz
        else:
            mean = resid.mean(-1, keepdim=True)
            std = (resid.var(-1, keepdim=True, unbiased=False) + ln.eps).sqrt()
        cache[f"{key}.mean"], cache[f"{key}.std"] = mean, std
        cache[f"{key}.out"] = out
        return out

    def forward(self, tokens: torch.Tensor, cache: dict | None = None, ln_frozen: dict | None = None) -> torch.Tensor:
        cache = {} if cache is None else cache
        n = tokens.shape[-1]
        resid = self.embed[tokens] + self.pos_embed[:n]
        cache["resid.embed"] = resid
        for layer in range(self.cfg.n_layers):
            resid = resid + self.attn[layer](self._norm(resid, "ln1", layer, cache, ln_frozen), cache, layer)
            cache[f"L{layer}.resid_mid"] = resid
            resid = resid + self.mlp[layer](self._norm(resid, "ln2", layer, cache, ln_frozen), cache, layer)
            cache[f"L{layer}.resid_post"] = resid
        resid = resid * self.readout_gate[:n].unsqueeze(-1)
        cache["resid.final"] = resid
        return resid @ self.W_U

    def capture_ln_stats(self, tokens: torch.Tensor) -> dict:
        """Run a clean pass and return {site: (mean, std)} to freeze into a later `ln_frozen` pass."""
        cache: dict = {}
        self(tokens, cache)
        out = {}
        for layer in range(self.cfg.n_layers):
            for which in ("ln1", "ln2"):
                key = f"L{layer}.{which}"
                if f"{key}.mean" in cache:
                    out[key] = (cache[f"{key}.mean"], cache[f"{key}.std"])
        return out

    def run_with_cache(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cache: dict = {}
        return self(tokens, cache), cache
