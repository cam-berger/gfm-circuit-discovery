"""Three levels of intervention, and the gap between them.

An edge does not exist in the real model. `MLP_m` is a nonlinear function of a residual
vector; there is no wire from an upstream feature into it. Edges exist only in the
*replacement model*, where the MLP output decomposes additively into feature decoder
vectors, so u's contribution to v's pre-activation is an isolable summand.

A real-model edge intervention nonetheless exists, as a path patch: take the vector
w = dec_u * a_u that the replacement model attributes to u, subtract it from the residual
input *as seen only by MLP_m*, leave attention and every other consumer untouched, and let
it propagate through the real nonlinear MLP.

The difference between the two is the faithfulness gap. Here it has exactly one source:
the replacement model freezes the ReLU gate, the real model recomputes it. When an
intervention flips a gate, linear attribution mispredicts the effect. This is the proposal's
Risk #1 in its smallest possible form, and on the planted model we can measure it exactly.

H4 is stated in terms of routing share, not magnitude. Ablating a node removes all of its
outgoing edges, so |eff(node u)| >= |eff(any single edge from u)| always. The original
"edge-patch effect > node-only control" predicts the part exceeding the whole.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gfmcd.models.tiny import TinyTransformer


@dataclass(frozen=True)
class PatchResult:
    effect_replacement: float
    effect_real: float

    @property
    def faithfulness_gap(self) -> float:
        return abs(self.effect_replacement - self.effect_real)


def _logits_from(model: TinyTransformer, resid_mid: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
    n = resid_mid.shape[1]
    x = resid_mid + acts @ model.mlp[1].W_out
    return (x * model.readout_gate[:n].unsqueeze(-1)) @ model.W_U


@torch.no_grad()
def _mlp1_state(model: TinyTransformer, tokens: torch.Tensor):
    logits, cache = model.run_with_cache(tokens)
    m1 = cache["L1.resid_mid"]
    pre1 = m1 @ model.mlp[1].W_in + model.mlp[1].b_in
    return logits, cache, m1, pre1


@torch.no_grad()
def edge_patch(
    model: TinyTransformer,
    tokens: torch.Tensor,
    head_dim: int,
    freeze_gate: bool,
) -> torch.Tensor:
    """Remove the layer-0 head's component `head_dim` from MLP1's input, and nothing else.

    `freeze_gate=True` is the replacement model: the ReLU mask is held at its clean value,
    so the computation is linear. `freeze_gate=False` is the real model: ReLU recomputes.
    """
    logits, cache, m1, pre1 = _mlp1_state(model, tokens)

    z = cache["L0.attn.z"]
    contribution = z[..., head_dim : head_dim + 1] * model.attn[0].W_O[head_dim, :]
    pre_patched = (m1 - contribution) @ model.mlp[1].W_in + model.mlp[1].b_in

    if freeze_gate:
        acts = pre_patched * (pre1 > 0).to(pre1.dtype)
    else:
        acts = torch.relu(pre_patched)

    return _logits_from(model, m1, acts) - logits


@torch.no_grad()
def node_ablate_head_dim(model: TinyTransformer, tokens: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Zero the head component everywhere it is read, not just into MLP1."""
    logits, cache, m1, pre1 = _mlp1_state(model, tokens)

    z = cache["L0.attn.z"]
    contribution = z[..., head_dim : head_dim + 1] * model.attn[0].W_O[head_dim, :]
    m1_ablated = m1 - contribution
    acts = torch.relu(m1_ablated @ model.mlp[1].W_in + model.mlp[1].b_in)

    return _logits_from(model, m1_ablated, acts) - logits


@torch.no_grad()
def faithfulness(model: TinyTransformer, tokens: torch.Tensor, head_dim: int) -> PatchResult:
    repl = edge_patch(model, tokens, head_dim, freeze_gate=True)
    real = edge_patch(model, tokens, head_dim, freeze_gate=False)
    return PatchResult(float(repl.abs().mean()), float(real.abs().mean()))


@torch.no_grad()
def routing_share(model: TinyTransformer, tokens: torch.Tensor, head_dim: int) -> float:
    """|eff_real(edge u -> MLP1)| / |eff_real(node u)| in [0, 1].

    H4a: much larger for the traced edge than for matched random edges out of u.
    """
    edge = edge_patch(model, tokens, head_dim, freeze_gate=False).abs().mean()
    node = node_ablate_head_dim(model, tokens, head_dim).abs().mean()
    return float(edge / node) if node > 0 else 0.0
