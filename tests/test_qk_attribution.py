"""Test A2 — recovering the edges frozen attention deletes.

Test A measured the blind spot: the marker-finding edge scored exactly 0.000 under OV
attribution while its ablation cost 0.617 accuracy. Rung 1's circuit is anchored by that edge,
so this instrument is on the critical path, not a refinement.

The planted twins give a known answer on both sides:

  compositional  W_K reads only the M-token dimension, W_Q only the constant dimension.
                 -> key-side attribution must land on `embed.tok`, and only at the marker.
  bag of motifs  W_Q = W_K = 0; its heads are driven by a fixed positional bias.
                 -> QK attribution must be identically zero. No content decides where it looks.

The second is the falsification control: an instrument that reports QK edges on a model whose
attention ignores content is measuring itself.
"""

import pytest
import torch

from gfmcd.data.synthetic import M, ablate_marker, make_batch
from gfmcd.models.planted import (
    PLANTED_EDGES,
    PLANTED_EDGES_OV,
    PLANTED_EDGES_QK,
    build_compositional,
    build_heuristic,
)
from gfmcd.tracer.graph import abstract_edges, recovered_edges
from gfmcd.tracer.qk import key_attribution_by_position, qk_edges

EPS = 1e-8


@pytest.fixture(scope="module")
def batch():
    return make_batch(n_seq=8, body_len=60, seed=21)


@pytest.fixture(scope="module")
def comp():
    return build_compositional(match_accuracy=False)


def test_key_side_attribution_lands_on_token_identity(comp, batch):
    """W_K reads only the M-token dimension, so only `embed.tok` may carry key-side weight."""
    tokens, labels = batch
    edges = qk_edges(comp, tokens, labels)

    key_edges = {k: v for k, v in edges.items() if k[1].endswith("attn.pattern")}
    live = {k[0] for k, v in key_edges.items() if v > EPS}
    assert live == {"embed.tok"}, f"key-side attribution leaked to {live - {'embed.tok'}}"


def test_query_side_attribution_lands_on_the_constant(comp, batch):
    """W_Q reads only the constant dimension: every position asks the same question."""
    tokens, labels = batch
    edges = qk_edges(comp, tokens, labels)

    q_edges = {k: v for k, v in edges.items() if k[1].endswith(".query")}
    live = {k[0] for k, v in q_edges.items() if v > EPS}
    assert live == {"embed.const"}, f"query-side attribution leaked to {live - {'embed.const'}}"


def test_key_attribution_concentrates_on_the_marker_position(comp, batch):
    """Not just the right feature -- the right token."""
    tokens, labels = batch
    attr = key_attribution_by_position(comp, tokens, labels, head=0)

    starts = (tokens == M).float().argmax(dim=-1)
    total = attr.sum(dim=-1)
    at_marker = attr[torch.arange(len(tokens)), starts]
    share = (at_marker / total).min()
    assert float(share) > 0.99, f"only {float(share):.4f} of key attribution sits on the marker"


def test_marker_ablation_destroys_the_qk_edge(comp, batch):
    """The attribution tracks the cause, not the position."""
    tokens, labels = batch
    intact = qk_edges(comp, tokens, labels)[("embed.tok", "L0.attn.pattern")]
    ablated = qk_edges(comp, ablate_marker(tokens, seed=3), labels)[("embed.tok", "L0.attn.pattern")]
    assert ablated < 0.1 * intact, f"intact={intact:.6f} ablated={ablated:.6f}"


def test_bag_of_motifs_has_no_qk_edges_at_all(batch):
    """Falsification control. Its heads are positional bias; no content decides where it looks."""
    tokens, labels = batch
    bag = build_heuristic(seq_len=tokens.shape[-1])
    edges = qk_edges(bag, tokens, labels)
    assert max(edges.values()) == 0.0, f"invented QK structure: {[k for k,v in edges.items() if v>0]}"


def test_combined_graph_recovers_every_planted_edge(comp, batch):
    """OV attribution plus QK attribution: recall 1.0 over the full planted edge set."""
    tokens, labels = batch

    ov = recovered_edges(abstract_edges(comp, tokens))
    qk = {k for k, v in qk_edges(comp, tokens, labels).items() if v > EPS and k[1].endswith("attn.pattern")}
    found = ov | qk

    assert PLANTED_EDGES_OV <= found
    assert PLANTED_EDGES_QK <= found
    recall = len(found & PLANTED_EDGES) / len(PLANTED_EDGES)
    assert recall == 1.0, f"missed {PLANTED_EDGES - found}"
