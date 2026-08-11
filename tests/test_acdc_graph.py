"""Edge-graph enumeration for ACDC (A14).

Genome-free and model-free: these tests pin the graph the A14 protocol is defined over, so a
change in node ordering, upstream relation, or edge count fails here rather than silently
redefining the circuit ACDC searches.
"""

from __future__ import annotations

import torch

from gfmcd.circuits.acdc_graph import EdgeGraph


# --------------------------------------------------------------------------------------
# Counts.  Pinned by PREREGISTRATION_A14_EDGE_ACDC.md section 1 as an `edge_count` memorial.
# --------------------------------------------------------------------------------------
def test_node_count_matches_a12_granularity():
    g = EdgeGraph(num_layers=6, num_heads=16)
    assert g.n_nodes == 103          # embed + 96 heads + 6 MLPs, identical to A12/A13


def test_head_granular_edge_count_is_4636():
    g = EdgeGraph(num_layers=6, num_heads=16)
    assert g.n_edges == 4636         # includes the 103 node -> readout edges


def test_qkv_split_edge_count_is_12988():
    g = EdgeGraph(num_layers=6, num_heads=16, qkv_split=True)
    assert g.n_edges == 12988


# --------------------------------------------------------------------------------------
# The upstream relation.  Write order is embed, then per layer: heads, then that layer's MLP.
# --------------------------------------------------------------------------------------
def test_head_is_upstream_of_its_own_layer_mlp():
    g = EdgeGraph(6, 16)
    assert ((0, 3), (0, "mlp"), "in") in g.edge_set


def test_mlp_is_upstream_of_the_next_layer_heads():
    g = EdgeGraph(6, 16)
    assert ((0, "mlp"), (1, 0), "in") in g.edge_set


def test_later_node_is_not_upstream_of_earlier():
    g = EdgeGraph(6, 16)
    assert ((1, 0), (0, "mlp"), "in") not in g.edge_set


def test_heads_in_the_same_layer_are_parallel_not_connected():
    g = EdgeGraph(6, 16)
    assert ((0, 3), (0, 5), "in") not in g.edge_set


def test_every_node_writes_to_the_readout():
    g = EdgeGraph(6, 16)
    for node in g.nodes:
        assert (node, "readout", "in") in g.edge_set


def test_embed_is_upstream_of_everything():
    g = EdgeGraph(6, 16)
    for node in g.nodes:
        if node != "embed":
            assert ("embed", node, "in") in g.edge_set


# --------------------------------------------------------------------------------------
# Ordering.  ACDC iterates output-first; a parent must never be visited before its child.
# --------------------------------------------------------------------------------------
def test_reverse_topological_starts_at_readout_and_ends_at_embed():
    order = EdgeGraph(6, 16).reverse_topological()
    assert order[0] == "readout"
    assert order[-1] == "embed"


def test_reverse_topological_visits_every_child_before_its_parents():
    g = EdgeGraph(6, 16)
    seen: set = set()
    for node in g.reverse_topological():
        for src, _slot in g.parents(node):
            assert src not in seen, f"{src} was visited before its child {node}"
        seen.add(node)


def test_reverse_topological_covers_every_node_exactly_once():
    g = EdgeGraph(6, 16)
    order = g.reverse_topological()
    assert len(order) == g.n_nodes + 1                 # + readout
    assert len(set(order)) == len(order)


# --------------------------------------------------------------------------------------
# Slots.  With qkv_split a head destination has three separately ablatable input slots;
# MLPs and the readout have one.  This is what separates address from payload at edge level.
# --------------------------------------------------------------------------------------
def test_head_destination_has_three_slots_when_split():
    g = EdgeGraph(6, 16, qkv_split=True)
    slots = {slot for src, dst, slot in g.edges if dst == (3, 1) and src == (2, 15)}
    assert slots == {"q", "k", "v"}


def test_mlp_destination_has_one_slot_even_when_split():
    g = EdgeGraph(6, 16, qkv_split=True)
    slots = {slot for src, dst, slot in g.edges if dst == (2, "mlp") and src == (1, 0)}
    assert slots == {"in"}


def test_head_destination_has_one_slot_when_not_split():
    g = EdgeGraph(6, 16)
    slots = {slot for src, dst, slot in g.edges if dst == (3, 1) and src == (2, 15)}
    assert slots == {"in"}


# --------------------------------------------------------------------------------------
# The mask tensor.  D7.2: the edge set is a boolean tensor so the graph is structurally
# static and edge removal is a flip rather than a Python-set rebuild.
# --------------------------------------------------------------------------------------
def test_full_mask_is_boolean_and_has_exactly_n_edges_set():
    g = EdgeGraph(6, 16)
    mask = g.full_mask()
    assert mask.dtype == torch.bool
    assert int(mask.sum()) == g.n_edges


def test_full_mask_is_antisymmetric_because_the_graph_is_a_dag():
    g = EdgeGraph(6, 16)
    mask = g.full_mask()
    i, j = g.node_index[(0, 3)], g.node_index[(0, "mlp")]
    assert bool(mask[i, j, 0])
    assert not bool(mask[j, i, 0])


def test_edge_slot_index_round_trips_every_edge():
    g = EdgeGraph(6, 16)
    mask = g.full_mask()
    for edge in g.edges:
        assert bool(mask[g.edge_slot_index(edge)])


def test_qkv_mask_separates_key_from_value_edges():
    g = EdgeGraph(6, 16, qkv_split=True)
    mask = g.full_mask()
    k_idx = g.edge_slot_index(((2, 15), (3, 1), "k"))
    v_idx = g.edge_slot_index(((2, 15), (3, 1), "v"))
    assert k_idx != v_idx
    mask[k_idx] = False
    assert not bool(mask[k_idx]) and bool(mask[v_idx])


def test_parents_agrees_with_the_edge_list():
    g = EdgeGraph(6, 16)
    key = str                                          # nodes mix str and tuple; sort by repr
    for node in list(g.nodes) + ["readout"]:
        from_parents = sorted(g.parents(node), key=key)
        from_edges = sorted(
            ((src, slot) for src, dst, slot in g.edges if dst == node), key=key
        )
        assert from_parents == from_edges
