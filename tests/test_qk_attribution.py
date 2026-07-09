"""Test A2 — recovering the edges frozen attention deletes.

Test A measured the blind spot: the marker-finding edge scored exactly 0.000 under OV
attribution while its ablation cost 0.617 accuracy. Rung 1's circuit is anchored by that edge,
so this instrument is on the critical path, not a refinement.

The planted twins give a known answer on both sides:

  compositional  W_K reads only the M-token dimension, W_Q only the constant dimension.
                 -> key-side attribution must land on `embed.tok`, and only at the marker.
  paired control W_Q and W_K bit-identical to the compositional twin -- the head still finds the
                 marker -- but W_O = 0, and the readout is carried by the bag pathway instead, so
                 the model still works and the loss is not constant.
                 -> attribution must collapse 7.798 -> 0.000. Everything the instrument could use
                    to hallucinate an edge is held fixed; only causal relevance moves.

The bag twin also reports zero, but only because W_Q = W_K = 0 makes the coefficients vanish
before the instrument does any work. That is a weaker statement and is asserted as such.

An earlier control simply zeroed W_O on the compositional twin. That silences the MLP too, so the
loss goes constant and *every* gradient vanishes -- zero attribution then proves nothing.
"""

import pytest
import torch

from gfmcd.data.synthetic import M, ablate_marker, make_batch
from gfmcd.models.planted import (
    PLANTED_EDGES,
    PLANTED_EDGES_OV,
    PLANTED_EDGES_QK,
    build_bag_with_unread_marker_head,
    build_compositional,
    build_heuristic,
    phase_accuracy,
)
from gfmcd.tracer.qk import integrated_score_gradient, qk_attribution
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


def test_paired_control_only_causal_relevance_moves_the_attribution(comp, batch):
    """The falsification control, as a paired ablation.

    Head 0's QK is bit-identical in both models: same W_Q, same W_K, same 584 content-bearing
    score entries, same coefficients. Only W_O differs, so only whether the head is *read*
    differs. The unread model still works (0.80) via the bag pathway, so the loss is not
    constant. Everything the instrument could use to hallucinate an edge is held fixed.

    Attribution goes 7.798 -> 0.000. An instrument reading its own coefficients could not do this.
    """
    tokens, labels = batch
    unread = build_bag_with_unread_marker_head(seq_len=tokens.shape[-1])

    assert torch.equal(comp.attn[0].W_K.detach(), unread.attn[0].W_K.detach())
    assert torch.equal(comp.attn[0].W_Q.detach(), unread.attn[0].W_Q.detach())

    _, content_read, _ = integrated_score_gradient(comp, tokens, labels)
    _, content_unread, _ = integrated_score_gradient(unread, tokens, labels)
    assert int((content_read[:, 0].abs() > 0).sum()) == int((content_unread[:, 0].abs() > 0).sum()) > 0

    assert phase_accuracy(unread, tokens, labels) > 0.7, "control must remain a working model"

    read_attr = float(qk_attribution(comp, tokens, labels)[0][:, 0].abs().max())
    unread_attr = float(qk_attribution(unread, tokens, labels)[0][:, 0].abs().max())
    assert read_attr > 1.0
    assert unread_attr == 0.0, f"invented a QK edge into an unread head: {unread_attr:.3e}"


def test_bag_of_motifs_reports_zero_but_only_trivially(batch):
    """Weaker than it looks: W_Q = W_K = 0, so the coefficients vanish before the instrument runs."""
    tokens, labels = batch
    bag = build_heuristic(seq_len=tokens.shape[-1])

    assert float(bag.attn[0].W_K.detach().abs().sum()) == 0.0
    assert max(qk_edges(bag, tokens, labels).values()) == 0.0


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
