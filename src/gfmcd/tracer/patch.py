"""Three levels of intervention, and the gap between them.

An edge does not exist in the real model. `MLP_m` is a nonlinear function of a residual
vector; there is no wire from an upstream feature into it. Edges exist only in the
*replacement model*, where the MLP output decomposes additively into feature decoder
vectors, so u's contribution to v's pre-activation is an isolable summand.

A real-model edge intervention nonetheless exists, as a path patch: take the vector
w = dec_u * a_u that the replacement model attributes to u, subtract it from the residual
input *as seen only by MLP_m*, leave attention and every other consumer untouched, and let
it propagate through the real nonlinear MLP.

The difference between the two is the faithfulness gap. It is the proposal's Risk #1
(attribution is first-order-linear and misses cooperativity) turned into a number. No GFM
interpretability paper currently reports it.

H4 is stated in terms of routing share, not magnitude. Ablating a node removes all of its
outgoing edges, so |eff(node u)| >= |eff(any single edge from u)| always. The original
"edge-patch effect > node-only control" predicts the part exceeding the whole.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Node:
    layer: int
    feature: int
    position: int


@dataclass(frozen=True)
class Edge:
    src: Node
    dst_layer: int


@dataclass(frozen=True)
class PatchResult:
    effect_replacement: float
    effect_real: float

    @property
    def faithfulness_gap(self) -> float:
        return abs(self.effect_replacement - self.effect_real)


def node_ablate(model, dictionaries, node: Node, tokens: torch.Tensor) -> float:
    """Zero feature `node` wherever it is read. Removes all of its outgoing edges."""
    raise NotImplementedError


def edge_ablate_replacement(model, dictionaries, edge: Edge, tokens: torch.Tensor) -> float:
    """Drop u's summand from MLP_m's pre-activation, inside the replacement model. Exact."""
    raise NotImplementedError


def edge_patch_real(model, dictionaries, edge: Edge, tokens: torch.Tensor) -> float:
    """Subtract dec_u * a_u from MLP_m's residual input only; real nonlinear MLP recomputes."""
    raise NotImplementedError


def routing_share(model, dictionaries, edge: Edge, tokens: torch.Tensor) -> float:
    """|eff_real(edge u->m)| / |eff_real(node u)| in [0, 1].

    H4a: this is much larger for the traced edge than for matched random edges out of u
    (same out-degree, same layer distance). A real circuit routes through few edges; a
    bag-of-motifs diffuses across many.
    """
    raise NotImplementedError
