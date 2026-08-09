"""Exactness of the post-LayerNorm edge-reconstruction kernel (A14 section 2).

Genome-free and checkpoint-free: a tiny randomly-initialised BERT with the same post-LN
structure exercises every path, so these run in CI.

WHY THESE ARE MEMORIALS AND NOT FORCED IDENTITIES.  A14's kernel does not assume residual
additivity -- SpliceBERT is post-LN, so that assumption is false -- and instead replays the
LayerNorm recursion per destination.  A wrong routing, a wrong write order, or a wrong LN
association breaks the two reduction identities below.  They can fail, unlike A2b's
``decomposition`` check, which `docs/PREREG_CONFORMANCE.md:139` records as a forced arithmetic
identity with no discriminating power.

``test_partial_mask_actually_changes_the_output`` is the one that makes the other three mean
something: all of them pass on a kernel that ignores the mask entirely and just runs the model.
"""

from __future__ import annotations

import pytest
import torch
from transformers import BertConfig, BertForMaskedLM

from gfmcd.circuits.acdc_graph import EdgeGraph
from gfmcd.circuits.acdc_kernel import AcdcForward, capture_node_writes

TOL_LOGITS = 1e-4          # A14 section 5, kernel_exact_full / kernel_exact_empty
TOL_HIDDEN = 1e-5          # A14 section 5, per_head_input_identity


def tiny_model(seed: int = 0):
    """Two layers, four heads -- same post-LN structure as SpliceBERT, 1/1000 the size."""
    torch.manual_seed(seed)
    config = BertConfig(
        vocab_size=32, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=64, layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    model = BertForMaskedLM(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config


def make_inputs(seed: int, batch: int = 2, length: int = 16):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 32, (batch, length), generator=generator)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.fixture(scope="module")
def setup():
    model, config = tiny_model()
    graph = EdgeGraph(config.num_hidden_layers, config.num_attention_heads)
    return model, config, graph, AcdcForward(model, graph)


# --------------------------------------------------------------------------------------
# The two reduction identities.  H = ALL must be the clean forward; H = empty the corrupt one.
# --------------------------------------------------------------------------------------
def test_full_mask_reproduces_the_clean_forward(setup):
    model, _config, graph, forward = setup
    clean, corrupt = make_inputs(1), make_inputs(2)
    reference = model(**clean).logits
    got = forward.logits(clean, capture_node_writes(model, corrupt), graph.full_mask())
    assert torch.max((got - reference).abs()).item() < TOL_LOGITS


def test_empty_mask_reproduces_the_corrupt_forward(setup):
    model, _config, graph, forward = setup
    clean, corrupt = make_inputs(1), make_inputs(2)
    reference = model(**corrupt).logits
    empty = torch.zeros_like(graph.full_mask())
    got = forward.logits(clean, capture_node_writes(model, corrupt), empty)
    assert torch.max((got - reference).abs()).item() < TOL_LOGITS


def test_per_head_reconstructed_input_equals_the_true_hidden_state(setup):
    model, config, graph, forward = setup
    clean = make_inputs(1)
    true_hidden = model(**clean, output_hidden_states=True).hidden_states
    inputs = forward.destination_inputs(
        clean, capture_node_writes(model, clean), graph.full_mask()
    )
    for layer in range(config.num_hidden_layers):
        for head in range(config.num_attention_heads):
            got = inputs[((layer, head), "in")]
            assert torch.max((got - true_hidden[layer]).abs()).item() < TOL_HIDDEN


def test_mlp_reconstructed_input_equals_the_post_attention_state(setup):
    """The MLP of layer l reads the state after layer l's attention block, not before."""
    model, config, graph, forward = setup
    clean = make_inputs(1)
    inputs = forward.destination_inputs(
        clean, capture_node_writes(model, clean), graph.full_mask()
    )
    hidden = model(**clean, output_hidden_states=True).hidden_states
    for layer in range(config.num_hidden_layers):
        got = inputs[((layer, "mlp"), "in")]
        assert torch.max((got - hidden[layer]).abs()).item() > TOL_HIDDEN   # not the input
        assert got.shape == hidden[layer].shape


# --------------------------------------------------------------------------------------
# The test that makes the three above meaningful.
# --------------------------------------------------------------------------------------
def test_partial_mask_actually_changes_the_output(setup):
    model, _config, graph, forward = setup
    clean, corrupt = make_inputs(1), make_inputs(2)
    writes = capture_node_writes(model, corrupt)
    mask = graph.full_mask()
    baseline = forward.logits(clean, writes, mask)

    cut = mask.clone()
    cut[:, graph.node_index[(1, 0)], :] = False       # every in-edge of one head
    assert torch.max((forward.logits(clean, writes, cut) - baseline).abs()).item() > 1e-3


def test_cutting_the_readout_edge_of_every_node_gives_the_corrupt_readout(setup):
    """Severing all edges into the readout must leave the readout reading corrupt state."""
    model, _config, graph, forward = setup
    clean, corrupt = make_inputs(1), make_inputs(2)
    mask = graph.full_mask()
    mask[:, graph.node_index["readout"], :] = False
    got = forward.logits(clean, capture_node_writes(model, corrupt), mask)
    assert torch.max((got - model(**corrupt).logits).abs()).item() < TOL_LOGITS


# --------------------------------------------------------------------------------------
# Write decomposition.  b_O is architecture, so per-head writes plus the bias must recover
# the true attention write exactly.
# --------------------------------------------------------------------------------------
def test_per_head_writes_plus_bias_recover_the_true_attention_write(setup):
    model, config, _graph, _forward = setup
    clean = make_inputs(1)
    writes = capture_node_writes(model, clean)

    captured: dict[int, torch.Tensor] = {}
    handles = [
        model.bert.encoder.layer[l].attention.output.dense.register_forward_hook(
            lambda _m, _i, out, l=l: captured.__setitem__(l, out.detach())
        )
        for l in range(config.num_hidden_layers)
    ]
    try:
        model(**clean)
    finally:
        for handle in handles:
            handle.remove()

    for layer in range(config.num_hidden_layers):
        summed = writes.attention[layer].sum(dim=0) + writes.attention_bias[layer]
        assert torch.max((summed - captured[layer]).abs()).item() < TOL_HIDDEN


def test_determinism_same_inputs_give_bitwise_identical_logits(setup):
    """ACDC is a sequential greedy: one nondeterministic call changes every later decision."""
    model, _config, graph, forward = setup
    clean, corrupt = make_inputs(1), make_inputs(2)
    writes = capture_node_writes(model, corrupt)
    first = forward.logits(clean, writes, graph.full_mask())
    second = forward.logits(clean, writes, graph.full_mask())
    assert torch.equal(first, second)
