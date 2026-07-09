"""Test A's instrument: direct-path attribution with the dictionary set to identity.

With attention patterns frozen and ReLU gates frozen, a LayerNorm-free transformer is exactly
linear in its activations. So first-order attribution is not an approximation here -- every
edge weight is the true linear coefficient times the source activation, and incoming edges
must reconstruct each node's pre-activation exactly. That equality is the assertion: it tests
the attribution machinery and nothing else.

Freezing attention is what buys the linearity, and it is also the method's blind spot. A frozen
pattern means the QK circuit -- the computation that decides *where* to look -- has no edges at
all. Attribution can only ever see the OV path. Our compositional twin locates its start marker
through QK, so that edge is invisible here while being causally necessary under ablation.

Residual-dim groups double as node names (see models/planted.py for the layout).
"""

from __future__ import annotations

import torch

from gfmcd.models.tiny import TinyTransformer

GROUPS: dict[str, slice] = {
    "pos_embed.phase": slice(0, 3),
    "resid.start_phase": slice(3, 6),
    "resid.phase_feature": slice(6, 9),
    "embed.tok": slice(9, 14),
    "ctx.tok_m2": slice(14, 19),
    "ctx.tok_m1": slice(19, 24),
    "ctx.tok_p1": slice(24, 29),
    "ctx.tok_p2": slice(29, 34),
    "embed.const": slice(35, 36),
}


@torch.no_grad()
def _state(model: TinyTransformer, tokens: torch.Tensor):
    _, cache = model.run_with_cache(tokens)
    W_O, W_in = model.attn[0].W_O, model.mlp[1].W_in
    return cache, torch.einsum("hcd,dj->hcj", W_O, W_in)


@torch.no_grad()
def conservation_residuals(model: TinyTransformer, tokens: torch.Tensor) -> dict[str, float]:
    """Max absolute error when incoming edges are summed back into each node."""
    cache, W_Oin = _state(model, tokens)
    n = tokens.shape[-1]
    x0, z, acts = cache["resid.embed"], cache["L0.attn.z"], cache["L1.mlp.acts"]

    z_hat = torch.einsum("bhpq,bqd,dhc->bphc", cache["L0.attn.pattern"], x0, model.attn[0].W_V)
    pre_hat = torch.einsum("bphc,hcj->bpj", z, W_Oin) + x0 @ model.mlp[1].W_in + model.mlp[1].b_in
    logit_hat = torch.einsum("bpj,jk,p->bpk", acts, model.mlp[1].W_out @ model.W_U, model.readout_gate[:n])

    pre = cache["L1.resid_mid"] @ model.mlp[1].W_in + model.mlp[1].b_in
    logits = cache["resid.final"] @ model.W_U

    return {
        "L0.attn.z": float((z_hat - z).abs().max()),
        "L1.mlp.pre": float((pre_hat - pre).abs().max()),
        "logits": float((logit_hat - logits).abs().max()),
    }


@torch.no_grad()
def abstract_edges(model: TinyTransformer, tokens: torch.Tensor) -> dict[tuple[str, str], float]:
    """Collapse per-dimension attributions into the named edges of `PLANTED_EDGES_OV`."""
    cache, W_Oin = _state(model, tokens)
    n = tokens.shape[-1]
    x0, z, acts = cache["resid.embed"], cache["L0.attn.z"], cache["L1.mlp.acts"]
    pattern, W_V, W_in = cache["L0.attn.pattern"], model.attn[0].W_V, model.mlp[1].W_in

    out: dict[tuple[str, str], float] = {}
    for name, sl in GROUPS.items():
        to_z = torch.einsum("bhpq,bqd,dhc->bhpqdc", pattern, x0[..., sl], W_V[sl])
        out[(name, "L0.attn.z")] = float(to_z.abs().mean())
        to_mlp = torch.einsum("bpd,dj->bpdj", x0[..., sl], W_in[sl])
        out[(name, "L1.mlp.acts")] = float(to_mlp.abs().mean())

    out[("L0.attn.z", "L1.mlp.acts")] = float(torch.einsum("bphc,hcj->bphcj", z, W_Oin).abs().mean())
    out[("L1.mlp.acts", "logits")] = float(
        torch.einsum("bpj,jk,p->bpjk", acts, model.mlp[1].W_out @ model.W_U, model.readout_gate[:n])
        .abs()
        .mean()
    )
    return out


def recovered_edges(edges: dict[tuple[str, str], float], eps: float = 1e-8) -> set[tuple[str, str]]:
    return {k for k, v in edges.items() if v > eps}


@torch.no_grad()
def transcoder_edges(
    model: TinyTransformer,
    transcoder,
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> dict[tuple[str, str], float]:
    """Test C's graph: residual groups -> transcoder features -> logits.

    The nodes are now learned features rather than neurons, so this composes dictionary error
    with attribution error. Test A and Test B say which of the two moved.
    """
    _, cache = model.run_with_cache(tokens)
    n = tokens.shape[-1]
    x = cache["L1.resid_mid"].reshape(-1, model.cfg.d_model)[mask]
    acts = transcoder.encode(x)
    gate = model.readout_gate[:n].unsqueeze(0).expand(tokens.shape[0], n).reshape(-1)[mask]

    out: dict[tuple[str, str], float] = {}
    for name, sl in GROUPS.items():
        contrib = torch.einsum("nd,df->ndf", x[:, sl], transcoder.W_enc[sl])
        out[(name, "transcoder.feature")] = float(contrib.abs().mean())

    to_logits = torch.einsum("nf,fk,n->nfk", acts, transcoder.W_dec @ model.W_U, gate)
    out[("transcoder.feature", "logits")] = float(to_logits.abs().mean())
    return out


PLANTED_EDGES_TRANSCODER: frozenset[tuple[str, str]] = frozenset(
    {
        ("pos_embed.phase", "transcoder.feature"),
        ("resid.start_phase", "transcoder.feature"),
        ("transcoder.feature", "logits"),
    }
)


def recovered_edges_relative(edges: dict[tuple[str, str], float], frac: float = 0.05) -> set[tuple[str, str]]:
    """Learned encoders are never exactly zero; threshold relative to the strongest edge."""
    cut = frac * max(edges.values())
    return {k for k, v in edges.items() if v > cut}
