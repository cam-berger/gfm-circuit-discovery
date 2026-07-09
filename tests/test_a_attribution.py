"""Test A — attribution machinery, dictionary set to identity.

With frozen attention and frozen ReLU gates this model is exactly linear, so attribution
is exact, not approximate. Any error here is a bug in the tracer, never in the dictionary.

The interesting assertion is the last one: the tracer recovers the OV subgraph perfectly
and is *structurally blind* to the QK edge that makes the circuit work.
"""

import pytest
import torch

from gfmcd.data.synthetic import ablate_marker, make_batch
from gfmcd.models.planted import (
    PLANTED_EDGES_OV,
    build_compositional,
    phase_accuracy,
)
from gfmcd.tracer.graph import abstract_edges, conservation_residuals, recovered_edges
from gfmcd.tracer.patch import edge_patch, faithfulness, node_ablate_head_dim, routing_share


@pytest.fixture(scope="module")
def setup():
    tokens, labels = make_batch(n_seq=8, body_len=60, seed=3)
    return build_compositional(match_accuracy=False), tokens, labels


def test_incoming_edges_reconstruct_every_node(setup):
    """A1: attribution is exact under frozen attention. No approximation to excuse."""
    model, tokens, _ = setup
    for node, err in conservation_residuals(model, tokens).items():
        assert err < 1e-4, f"{node} attribution does not sum to its pre-activation: {err:.2e}"


def test_ov_subgraph_is_recovered_exactly(setup):
    """A2: recall and precision of 1.0 against the planted OV edges."""
    model, tokens, _ = setup
    found = recovered_edges(abstract_edges(model, tokens))

    recall = len(found & PLANTED_EDGES_OV) / len(PLANTED_EDGES_OV)
    precision = len(found & PLANTED_EDGES_OV) / len(found)
    assert recall == 1.0, f"missed {PLANTED_EDGES_OV - found}"
    assert precision == 1.0, f"spurious {found - PLANTED_EDGES_OV}"


def test_token_identity_has_zero_ov_attribution(setup):
    """The value path never reads token dims, so tokens attribute exactly zero."""
    model, tokens, _ = setup
    edges = abstract_edges(model, tokens)
    assert edges[("embed.tok", "L0.attn.z")] == 0.0
    assert edges[("embed.tok", "L1.mlp.acts")] == 0.0


def test_frozen_attention_is_blind_to_a_causally_necessary_edge(setup):
    """A3: the headline limitation, measured rather than asserted.

    Attribution assigns the start marker exactly zero weight. Ablating it destroys the
    circuit. Any edge-level claim that freezes attention inherits this blind spot.
    """
    model, tokens, labels = setup
    edges = abstract_edges(model, tokens)

    attribution_to_marker = edges[("embed.tok", "L0.attn.z")]
    intact = phase_accuracy(model, tokens, labels)
    ablated = phase_accuracy(model, ablate_marker(tokens, seed=4), labels)

    assert attribution_to_marker == 0.0
    assert intact - ablated > 0.5, "marker must be causally necessary for this to bite"


def test_head_edge_carries_most_of_the_nodes_influence(setup):
    """H4a in miniature: the traced edge routes nearly all of the node's causal effect."""
    model, tokens, _ = setup
    shares = [routing_share(model, tokens, c) for c in range(3)]
    assert min(shares) > 0.9, f"routing shares {shares}"


def test_node_ablation_dominates_edge_ablation(setup):
    """The part cannot exceed the whole. H4's original inequality was backwards."""
    model, tokens, _ = setup
    for c in range(3):
        edge = edge_patch(model, tokens, c, freeze_gate=False).abs().mean()
        node = node_ablate_head_dim(model, tokens, c).abs().mean()
        assert edge <= node + 1e-6, f"edge {edge:.4f} > node {node:.4f} at head dim {c}"


def test_faithfulness_gap_is_nonzero_and_gate_flips_explain_it(setup):
    """Freezing the ReLU gate mispredicts the intervention. Quantified, not assumed away."""
    model, tokens, _ = setup
    gaps = [faithfulness(model, tokens, c) for c in range(3)]
    assert all(g.faithfulness_gap > 1e-3 for g in gaps), [g.faithfulness_gap for g in gaps]
    assert all(g.effect_replacement > g.effect_real for g in gaps), (
        "linear attribution should overstate the effect: ReLU clamps what it predicts"
    )
