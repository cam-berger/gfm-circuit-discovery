"""ACDC Algorithm 1 over an edge mask (arXiv:2304.14997, section 3).

    H <- G
    for v in reverse_topological_sort(H):
        for w in parents(v):
            H_new <- H \\ {w -> v}
            if D(G || H_new) - D(G || H) < tau:
                H <- H_new

SIGN CONVENTION.  ``metric_fn`` returns a **divergence**: lower is more faithful, 0 is perfect
agreement with the full model.  The paper uses KL, which is a divergence already.  A14 prunes on
faithfulness ``F``, so its runner passes ``lambda mask: 1.0 - F(mask)``.  Algorithm 1 then applies
verbatim in both cases and the threshold ``tau`` means the same thing -- an admissible *increase*
in divergence.  Getting this backwards silently inverts the search, so the convention is asserted
in ``tests/test_acdc_algorithm.py`` rather than left to the caller.

THE METRIC IS INJECTED.  ``acdc_run`` never touches a model.  That is what lets the algorithm be
tested against synthetic metrics with known answers, with no checkpoint and no genome.

TWO EVALUATION MODES (D7.6).  ``sequential`` is Algorithm 1 exactly: after an edge is removed,
later candidates at the same node are tested against the already-pruned ``H``.  ``parallel``
tests every candidate at a node against that node's *entry* ``H``, which is what allows one
batched forward over a candidate axis on a GPU -- but it is **not** Algorithm 1.  It is licensed
only by ``equivalence_check`` (A14 section 5) passing on a sample of nodes, and the mode that
produced a reported circuit is recorded in the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Sequence

import torch

from gfmcd.circuits.acdc_graph import READOUT, Edge, EdgeGraph, Node

Mode = Literal["sequential", "parallel"]
MetricFn = Callable[[torch.Tensor], float]
BatchMetricFn = Callable[[Sequence[torch.Tensor]], Sequence[float]]


@dataclass(frozen=True)
class EdgeDecision:
    src: Node
    dst: Node
    slot: str
    divergence_before: float
    divergence_after: float
    delta: float
    kept: bool

    def to_dict(self) -> dict:
        return {
            "src": str(self.src), "dst": str(self.dst), "slot": self.slot,
            "divergence_before": self.divergence_before,
            "divergence_after": self.divergence_after,
            "delta": self.delta, "kept": self.kept,
        }


@dataclass
class AcdcResult:
    mask: torch.Tensor
    decisions: list[EdgeDecision] = field(default_factory=list)
    tau: float = 0.0
    mode: Mode = "sequential"
    n_evaluations: int = 0

    @property
    def n_edges_kept(self) -> int:
        return int(self.mask.sum())

    def kept_edges(self, graph: EdgeGraph) -> list[Edge]:
        return [e for e in graph.edges if bool(self.mask[graph.edge_slot_index(e)])]

    def to_dict(self, graph: EdgeGraph) -> dict:
        return {
            "tau": self.tau, "mode": self.mode,
            "n_edges_kept": self.n_edges_kept, "n_edges_total": graph.n_edges,
            "n_evaluations": self.n_evaluations,
            "kept_edges": [[str(s), str(d), sl] for s, d, sl in self.kept_edges(graph)],
            "decisions": [d.to_dict() for d in self.decisions],
        }


def _has_outgoing(mask: torch.Tensor, node_row: int) -> bool:
    return bool(mask[node_row].any())


def acdc_run(
    metric_fn: MetricFn,
    graph: EdgeGraph,
    tau: float,
    *,
    mode: Mode = "sequential",
    parent_order: Callable[[list[tuple[Node, str]]], list[tuple[Node, str]]] | None = None,
    batch_metric_fn: BatchMetricFn | None = None,
    initial_mask: torch.Tensor | None = None,
    node_filter: set | None = None,
    on_node: Callable[[Node, AcdcResult], None] | None = None,
) -> AcdcResult:
    """Run ACDC.  ``on_node`` fires after each destination node -- the checkpoint hook.

    ``parent_order`` reorders the parents of each node.  ACDC's paper records (Appendix J) that
    parent order changed a result in one case, so the order is a recorded hyperparameter and A14
    runs one arm with it permuted.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if mode not in ("sequential", "parallel"):
        raise ValueError(f"unknown mode {mode!r}")

    mask = graph.full_mask() if initial_mask is None else initial_mask.clone()
    index = graph.node_index
    result = AcdcResult(mask=mask, tau=tau, mode=mode)

    current = metric_fn(mask)
    result.n_evaluations += 1

    for node in graph.reverse_topological():
        if node_filter is not None and node not in node_filter:
            continue
        if node != READOUT and not _has_outgoing(result.mask, index[node]):
            # No path from this node to the readout, so nothing it reads can reach the output.
            # Its in-edges are not part of the subgraph and must not be counted as kept.
            # Reverse-topological order guarantees every consumer is already decided, so this
            # verdict is final and the fixed point needs no second pass.
            result.mask[:, index[node], :] = False
            continue
        parents = graph.parents(node)
        if parent_order is not None:
            parents = parent_order(parents)

        live = [
            (src, slot) for src, slot in parents
            if bool(result.mask[graph.edge_slot_index((src, node, slot))])
        ]
        if not live:
            if on_node is not None:
                on_node(node, result)
            continue

        if mode == "sequential":
            for src, slot in live:
                position = graph.edge_slot_index((src, node, slot))
                candidate = result.mask.clone()
                candidate[position] = False
                after = metric_fn(candidate)
                result.n_evaluations += 1
                delta = after - current
                remove = delta < tau
                if remove:
                    result.mask = candidate
                    current = after
                result.decisions.append(EdgeDecision(
                    src=src, dst=node, slot=slot, divergence_before=current if remove else after,
                    divergence_after=after, delta=delta, kept=not remove,
                ))
        else:
            candidates = []
            for src, slot in live:
                candidate = result.mask.clone()
                candidate[graph.edge_slot_index((src, node, slot))] = False
                candidates.append(candidate)
            if batch_metric_fn is not None:
                scores = list(batch_metric_fn(candidates))
            else:
                scores = [metric_fn(c) for c in candidates]
            result.n_evaluations += len(candidates)

            entry = current
            for (src, slot), after in zip(live, scores):
                delta = after - entry                  # every candidate vs the ENTRY H
                remove = delta < tau
                if remove:
                    result.mask[graph.edge_slot_index((src, node, slot))] = False
                result.decisions.append(EdgeDecision(
                    src=src, dst=node, slot=slot, divergence_before=entry,
                    divergence_after=after, delta=delta, kept=not remove,
                ))
            if any(d.dst == node and not d.kept for d in result.decisions):
                current = metric_fn(result.mask)
                result.n_evaluations += 1

        if on_node is not None:
            on_node(node, result)

    return result


def equivalence_check(
    metric_fn: MetricFn,
    graph: EdgeGraph,
    tau: float,
    nodes: Iterable[Node],
    *,
    batch_metric_fn: BatchMetricFn | None = None,
) -> dict:
    """A14 section 5 / D7.6.  Do the two modes retain the same edges on a sample of nodes?

    The parallel path is a ~45x reduction in sequential steps and is the whole GPU case, but it
    is not Algorithm 1.  This is what licenses it -- or refuses to.
    """
    subset = set(nodes)
    start = graph.full_mask()
    sequential = acdc_run(metric_fn, graph, tau, mode="sequential",
                          initial_mask=start, node_filter=subset)
    parallel = acdc_run(metric_fn, graph, tau, mode="parallel", initial_mask=start,
                        batch_metric_fn=batch_metric_fn, node_filter=subset)
    differing = (sequential.mask != parallel.mask).nonzero().tolist()
    return {
        "PASS": not differing,
        "n_nodes_checked": len(subset),
        "n_differing_edges": len(differing),
        "n_edges_sequential": sequential.n_edges_kept,
        "n_edges_parallel": parallel.n_edges_kept,
        "disagreements": [
            {"src_row": d[0], "dst_row": d[1], "slot": d[2]} for d in differing[:32]
        ],
    }
