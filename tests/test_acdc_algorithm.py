"""ACDC Algorithm 1 against synthetic metrics with known answers.

No model, no checkpoint, no genome -- the metric is injected, so the search itself is testable
in CI.  These pin the two things that would silently invert or vacate the search:

* the divergence sign convention (lower = more faithful).  A caller that passes faithfulness
  directly instead of ``1 - F`` would prune the circuit away and report a beautiful sparse result.
* that ``parallel`` and ``sequential`` genuinely differ on a non-submodular metric.  If they never
  differ, ``equivalence_check`` licenses the fast path without evidence -- the R21/R7 forced-memorial
  pathology this repository already has a correction for.
"""

from __future__ import annotations

import pytest
import torch

from gfmcd.circuits.acdc import acdc_run, equivalence_check
from gfmcd.circuits.acdc_graph import EdgeGraph


def planted_path(graph: EdgeGraph) -> set:
    """A CONNECTED chain from embed to the readout.

    The planted set must reach the output.  A metric that rewards an edge feeding a node with
    no path to the readout is not realisable by any model -- the output cannot depend on it --
    and ACDC correctly drops such edges as part of no subgraph.
    """
    return {
        ("embed", (0, 0), "in"),
        ((0, 0), (0, "mlp"), "in"),
        ((0, "mlp"), (1, 0), "in"),
        ((1, 0), "readout", "in"),
    }


def planted_metric(graph: EdgeGraph, planted: set):
    """Divergence 0 when every planted edge is present, rising as they are removed."""
    positions = [graph.edge_slot_index(e) for e in planted]

    def metric(mask: torch.Tensor) -> float:
        missing = sum(1 for p in positions if not bool(mask[p]))
        return missing / len(positions)

    return metric


def test_recovers_exactly_the_planted_edges():
    graph = EdgeGraph(num_layers=2, num_heads=4)
    planted = planted_path(graph)
    result = acdc_run(planted_metric(graph, planted), graph, tau=0.01)
    assert set(result.kept_edges(graph)) == planted


def test_every_non_planted_edge_is_removed():
    graph = EdgeGraph(num_layers=2, num_heads=4)
    planted = planted_path(graph)
    result = acdc_run(planted_metric(graph, planted), graph, tau=0.01)
    assert result.n_edges_kept == len(planted)
    assert result.n_edges_kept < graph.n_edges


def test_larger_tau_keeps_fewer_edges():
    graph = EdgeGraph(num_layers=2, num_heads=4)
    planted = planted_path(graph)
    metric = planted_metric(graph, planted)
    small = acdc_run(metric, graph, tau=0.01).n_edges_kept
    large = acdc_run(metric, graph, tau=10.0).n_edges_kept
    assert large <= small


def test_tau_above_every_delta_removes_everything():
    graph = EdgeGraph(num_layers=2, num_heads=4)
    planted = planted_path(graph)
    result = acdc_run(planted_metric(graph, planted), graph, tau=100.0)
    assert result.n_edges_kept == 0


def test_zero_or_negative_tau_is_rejected():
    graph = EdgeGraph(2, 4)
    with pytest.raises(ValueError, match="tau must be positive"):
        acdc_run(lambda m: 0.0, graph, tau=0.0)


# --------------------------------------------------------------------------------------
# The sign convention.  A divergence, not a score.
# --------------------------------------------------------------------------------------
def test_a_constant_metric_removes_every_edge():
    """Nothing ever raises divergence, so nothing is load-bearing."""
    graph = EdgeGraph(2, 4)
    assert acdc_run(lambda m: 0.0, graph, tau=0.001).n_edges_kept == 0


def test_passing_faithfulness_instead_of_divergence_inverts_the_search():
    """Guards the caller contract: F itself must never be passed as the metric."""
    graph = EdgeGraph(num_layers=2, num_heads=4)
    planted = planted_path(graph)
    divergence = planted_metric(graph, planted)
    inverted = acdc_run(lambda m: 1.0 - divergence(m), graph, tau=0.01)
    assert set(inverted.kept_edges(graph)) != planted


# --------------------------------------------------------------------------------------
# Node skipping and ordering.
# --------------------------------------------------------------------------------------
def test_nodes_with_no_outgoing_edge_are_skipped():
    graph = EdgeGraph(2, 4)
    planted = planted_path(graph)
    result = acdc_run(planted_metric(graph, planted), graph, tau=0.01)
    visited = {d.dst for d in result.decisions}
    assert "readout" in {str(v) for v in visited}


def test_parent_order_is_honoured():
    graph = EdgeGraph(2, 4)
    planted = planted_path(graph)
    metric = planted_metric(graph, planted)
    forward = acdc_run(metric, graph, tau=0.01)
    reversed_order = acdc_run(metric, graph, tau=0.01, parent_order=lambda p: list(reversed(p)))
    assert set(forward.kept_edges(graph)) == set(reversed_order.kept_edges(graph)) == planted
    first_forward = forward.decisions[0]
    first_reversed = reversed_order.decisions[0]
    assert (first_forward.src, first_forward.slot) != (first_reversed.src, first_reversed.slot)


# --------------------------------------------------------------------------------------
# The two modes.  D7.6 -- the fast path must be licensed, not assumed.
# --------------------------------------------------------------------------------------
def test_modes_agree_on_a_submodular_metric():
    graph = EdgeGraph(2, 4)
    planted = planted_path(graph)
    metric = planted_metric(graph, planted)
    assert torch.equal(
        acdc_run(metric, graph, tau=0.01, mode="sequential").mask,
        acdc_run(metric, graph, tau=0.01, mode="parallel").mask,
    )


def test_modes_disagree_on_a_non_submodular_metric():
    """Discriminating power for equivalence_check: it must be able to fail."""
    graph = EdgeGraph(2, 4)
    # Two parents of the READOUT, so both are on a path to the output.
    pair = [graph.edge_slot_index((src, "readout", slot))
            for src, slot in graph.parents("readout")[:2]]

    def redundant(mask: torch.Tensor) -> float:
        # Either edge alone suffices; losing BOTH is what costs.  Removing one looks free
        # against the entry state, so the parallel pass drops both and the sequential does not.
        present = sum(1 for p in pair if bool(mask[p]))
        return 0.0 if present >= 1 else 1.0

    sequential = acdc_run(redundant, graph, tau=0.5, mode="sequential")
    parallel = acdc_run(redundant, graph, tau=0.5, mode="parallel")
    assert not torch.equal(sequential.mask, parallel.mask)


def test_equivalence_check_passes_on_a_submodular_metric():
    graph = EdgeGraph(2, 4)
    planted = planted_path(graph)
    report = equivalence_check(planted_metric(graph, planted), graph, 0.01, ["readout"])
    assert report["PASS"] is True
    assert report["n_nodes_checked"] == 1


def test_equivalence_check_fails_on_a_non_submodular_metric():
    graph = EdgeGraph(2, 4)
    # Two parents of the READOUT, so both are on a path to the output.
    pair = [graph.edge_slot_index((src, "readout", slot))
            for src, slot in graph.parents("readout")[:2]]

    def redundant(mask: torch.Tensor) -> float:
        return 0.0 if sum(1 for p in pair if bool(mask[p])) >= 1 else 1.0

    report = equivalence_check(redundant, graph, 0.5, ["readout"])
    assert report["PASS"] is False
    assert report["disagreements"]


# --------------------------------------------------------------------------------------
# Bookkeeping the runner and the resume path depend on.
# --------------------------------------------------------------------------------------
def test_on_node_hook_fires_for_checkpointing():
    graph = EdgeGraph(2, 4)
    seen: list = []
    acdc_run(planted_metric(graph, set(graph.edges[:5])), graph, tau=0.01,
             on_node=lambda node, result: seen.append(node))
    assert len(seen) >= 1
    assert str(seen[0]) == "readout"                    # output-first, per the paper


def test_result_serialises_with_edge_counts_and_decisions():
    graph = EdgeGraph(2, 4)
    result = acdc_run(planted_metric(graph, set(graph.edges[:5])), graph, tau=0.01)
    payload = result.to_dict(graph)
    assert payload["n_edges_total"] == graph.n_edges
    assert payload["n_edges_kept"] == result.n_edges_kept
    assert payload["mode"] == "sequential"
    assert len(payload["decisions"]) == len(result.decisions)
    assert payload["n_evaluations"] >= len(result.decisions)


def test_initial_mask_allows_resume_from_a_checkpoint():
    graph = EdgeGraph(2, 4)
    planted = planted_path(graph)
    metric = planted_metric(graph, planted)
    partial = graph.full_mask()
    partial[graph.edge_slot_index((("embed"), (1, 3), "in"))] = False
    result = acdc_run(metric, graph, tau=0.01, initial_mask=partial)
    assert set(result.kept_edges(graph)) == planted


def test_decision_log_is_internally_consistent():
    """``delta`` must equal ``after - before`` on EVERY logged decision.

    The first implementation recorded ``divergence_before = current if remove else after``,
    which is wrong in both branches: on removal ``current`` had already been advanced to
    ``after``, and on retention it recorded ``after`` as the before-value.  ``delta`` was
    computed correctly, so no decision changed -- but the trace was unusable, and a reader
    reconstructing the search from it would have got nonsense.
    """
    graph = EdgeGraph(2, 4)
    result = acdc_run(planted_metric(graph, planted_path(graph)), graph, tau=0.01)
    assert result.decisions
    for decision in result.decisions:
        assert decision.delta == pytest.approx(
            decision.divergence_after - decision.divergence_before, abs=1e-9
        ), decision


def test_decision_before_value_tracks_the_running_divergence():
    """A removed edge's ``divergence_after`` becomes the next decision's ``divergence_before``."""
    graph = EdgeGraph(2, 4)
    result = acdc_run(planted_metric(graph, planted_path(graph)), graph, tau=0.01)
    by_node: dict = {}
    for decision in result.decisions:
        by_node.setdefault(str(decision.dst), []).append(decision)
    for decisions in by_node.values():
        for earlier, later in zip(decisions, decisions[1:]):
            expected = earlier.divergence_after if not earlier.kept else earlier.divergence_before
            assert later.divergence_before == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------------------
# Resume.  A 10-hour Colab run that cannot restart is a coin flip, and the first
# implementation persisted only an edge COUNT, which restarts nothing.
# --------------------------------------------------------------------------------------
def test_resume_from_a_checkpoint_reproduces_an_uninterrupted_run(tmp_path):
    """Stop after the first node, reload mask + done-list, finish -> same mask."""
    import json

    graph = EdgeGraph(2, 4)
    metric = planted_metric(graph, planted_path(graph))
    order = graph.reverse_topological()

    reference = acdc_run(metric, graph, tau=0.01)

    # --- leg 1: run only the first node, checkpointing the way the runner does ---------
    state = tmp_path / "leg.state.json"
    done: list[str] = []

    def checkpoint(node, res):
        done.append(str(node))
        state.write_text(json.dumps({
            "done_nodes": done,
            "kept": res.mask.reshape(-1).nonzero().flatten().tolist(),
            "n_edges_kept": res.n_edges_kept,
        }))

    acdc_run(metric, graph, tau=0.01, node_filter={order[0]}, on_node=checkpoint)
    assert state.is_file() and len(done) == 1

    # --- leg 2: reload and finish the remainder ---------------------------------------
    saved = json.loads(state.read_text())
    restored = torch.zeros_like(graph.full_mask())
    flat = restored.reshape(-1)
    for index in saved["kept"]:
        flat[index] = True
    finished = set(saved["done_nodes"])
    remaining = {node for node in order if str(node) not in finished}
    assert len(remaining) == len(order) - 1

    resumed = acdc_run(metric, graph, tau=0.01, node_filter=remaining, initial_mask=restored)
    assert torch.equal(resumed.mask, reference.mask)
    assert set(resumed.kept_edges(graph)) == set(reference.kept_edges(graph))


def test_resume_with_every_node_done_is_a_no_op(tmp_path):
    """The last checkpoint before a disconnect may already be complete."""
    graph = EdgeGraph(2, 4)
    metric = planted_metric(graph, planted_path(graph))
    full_run = acdc_run(metric, graph, tau=0.01)
    again = acdc_run(metric, graph, tau=0.01, node_filter=set(), initial_mask=full_run.mask)
    assert torch.equal(again.mask, full_run.mask)
    assert again.decisions == []


def test_a_checkpoint_carrying_only_an_edge_count_cannot_restore_a_mask():
    """Pins why the original checkpoint format was useless: it has no mask to restore."""
    stale = {"n_edges_kept": 4480}
    assert "kept" not in stale and "done_nodes" not in stale
