"""Test A's instrument: direct-path attribution with the dictionary set to identity.

With attention patterns frozen and ReLU gates frozen, a LayerNorm-free transformer is
exactly linear in its activations. So first-order attribution is not an approximation
here -- every edge weight is the true linear coefficient times the source activation, and
incoming edges must reconstruct each node's pre-activation exactly. That equality is the
assertion: it tests the attribution machinery and nothing else.

Freezing attention is what buys the linearity, and it is also the method's blind spot. A
frozen pattern means the QK circuit -- the computation that decides *where* to look -- has
no edges at all. Attribution can only ever see the OV path. Our compositional twin locates
its start marker through QK, so we expect that edge to be invisible here while being
causally necessary under ablation. Test A measures exactly that.

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
    "embed.const": slice(15, 16),
}


@torch.no_grad()
def edge_tensors(model: TinyTransformer, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """The four direct-path edge families of the planted architecture.

    E1  resid.embed[q]  -> L0.attn.z[p]     via the frozen pattern and the OV matrix
    E2  L0.attn.z[p]    -> L1.mlp.acts[p]   via W_O then W_in
    E3  resid.embed[p]  -> L1.mlp.acts[p]   the residual skip into the MLP
    E4  L1.mlp.acts[p]  -> logits[p]        via W_out then W_U
    """
    _, cache = model.run_with_cache(tokens)
    x0 = cache["resid.embed"]
    z0 = cache["L0.attn.z"]
    a1 = cache["L1.mlp.acts"]
    pattern = cache["L0.attn.pattern"]

    W_V, W_O = model.attn[0].W_V, model.attn[0].W_O
    W_in, W_out = model.mlp[1].W_in, model.mlp[1].W_out
    gate = model.readout_gate[: tokens.shape[-1]]

    e1 = torch.einsum("bpq,bqd,dc->bpqdc", pattern, x0, W_V)
    e2 = torch.einsum("bpc,cj->bpcj", z0, W_O @ W_in)
    e3 = torch.einsum("bpd,dj->bpdj", x0, W_in)
    e4 = torch.einsum("bpj,jk,p->bpjk", a1, W_out @ model.W_U, gate)
    return {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "cache": cache}


@torch.no_grad()
def conservation_residuals(model: TinyTransformer, tokens: torch.Tensor) -> dict[str, float]:
    """Max absolute error when incoming edges are summed back into each node."""
    e = edge_tensors(model, tokens)
    cache = e["cache"]
    n = tokens.shape[-1]

    z_hat = e["E1"].sum(dim=(2, 3))
    pre_hat = e["E2"].sum(dim=2) + e["E3"].sum(dim=2) + model.mlp[1].b_in
    logit_hat = e["E4"].sum(dim=2)

    pre = cache["L1.resid_mid"] @ model.mlp[1].W_in + model.mlp[1].b_in
    logits = (cache["resid.final"]) @ model.W_U

    return {
        "L0.attn.z": float((z_hat - cache["L0.attn.z"]).abs().max()),
        "L1.mlp.pre": float((pre_hat - pre).abs().max()),
        "logits": float((logit_hat - logits).abs().max()),
    }


@torch.no_grad()
def abstract_edges(model: TinyTransformer, tokens: torch.Tensor) -> dict[tuple[str, str], float]:
    """Collapse per-dimension attributions into the named edges of `PLANTED_EDGES`."""
    e = edge_tensors(model, tokens)
    out: dict[tuple[str, str], float] = {}

    for name, sl in GROUPS.items():
        out[(name, "L0.attn.z")] = float(e["E1"][:, :, :, sl, :].abs().mean())
        out[(name, "L1.mlp.acts")] = float(e["E3"][:, :, sl, :].abs().mean())

    out[("L0.attn.z", "L1.mlp.acts")] = float(e["E2"].abs().mean())
    out[("L1.mlp.acts", "logits")] = float(e["E4"].abs().mean())
    return out


def recovered_edges(edges: dict[tuple[str, str], float], eps: float = 1e-8) -> set[tuple[str, str]]:
    return {k for k, v in edges.items() if v > eps}
