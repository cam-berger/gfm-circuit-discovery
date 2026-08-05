"""The computational graph ACDC searches over.  Pure structure -- no model, no genome.

WHY THIS IS A SEPARATE MODULE
-----------------------------
ACDC (arXiv:2304.14997, Algorithm 1) is defined over a DAG of model units.  Which units, and
which of them count as connected, *is* the definition of the circuit being searched -- a change
in write order or in the upstream relation silently redefines the answer.  Keeping the graph
free of any model dependency means the definition is tested in CI without a checkpoint or a
genome, and a regression fails in `tests/test_acdc_graph.py` rather than inside a 4-hour sweep.

NODES.  Exactly A12/A13's 103: ``embed`` + 96 attention heads + 6 MLPs.  Node identity is shared
across A12, A13 and A14 on purpose, so the node-level and edge-level faithfulness curves are
comparable without a re-derivation.

WRITE ORDER.  ``embed``, then for each layer ``l``: the 16 heads of ``l``, then ``(l, "mlp")``.
``u`` is upstream of ``v`` iff ``u`` writes strictly before ``v``.  Heads within a layer are
parallel -- they read the same layer input -- so no head is upstream of another in its layer.

BIASES ARE NOT NODES.  ``attention.output.dense`` is linear over the concatenated per-head
contexts, so the attention write decomposes exactly as ``sum_h W_O^(h) c_h + b_O``.  The ``b_O``
and MLP output biases are architecture, present on every forward pass and never ablated.  They
have no node and no edge; a "circuit" here is a set of *learned routing* choices, not a claim
that the biases are inert.

SLOTS.  Without ``qkv_split`` every destination has a single input slot ``"in"``.  With it, a
head destination has three separately ablatable slots ``q``/``k``/``v`` -- which is what lets
A14 separate L3H1's address edges (key) from its payload edges (value).  MLP and readout
destinations keep one slot either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import torch

Node = str | tuple[int, int] | tuple[int, str]
Slot = Literal["in", "q", "k", "v"]
Edge = tuple[Node, Node, Slot]

EMBED: Node = "embed"
READOUT: Node = "readout"

_SLOT_ORDER: tuple[Slot, ...] = ("in", "q", "k", "v")
_SLOT_INDEX: dict[Slot, int] = {slot: i for i, slot in enumerate(_SLOT_ORDER)}


def is_head(node: Node) -> bool:
    return isinstance(node, tuple) and isinstance(node[1], int)


def is_mlp(node: Node) -> bool:
    return isinstance(node, tuple) and node[1] == "mlp"


@dataclass(frozen=True)
class EdgeGraph:
    """The DAG A14 prunes.  Immutable; the *live* subgraph is a boolean mask over it."""

    num_layers: int
    num_heads: int
    qkv_split: bool = False

    # ---------------------------------------------------------------------------------
    # Nodes, in write order.
    # ---------------------------------------------------------------------------------
    @property
    def nodes(self) -> list[Node]:
        out: list[Node] = [EMBED]
        for layer in range(self.num_layers):
            out += [(layer, head) for head in range(self.num_heads)]
            out.append((layer, "mlp"))
        return out

    @property
    def n_nodes(self) -> int:
        return 1 + self.num_layers * (self.num_heads + 1)

    @property
    def node_index(self) -> dict[Node, int]:
        """``READOUT`` sits last so it is a valid destination in the same square mask."""
        index = {node: i for i, node in enumerate(self.nodes)}
        index[READOUT] = self.n_nodes
        return index

    def _write_order(self) -> dict[Node, int]:
        """Write *stage*, not node position.

        All 16 heads of a layer share one stage because they read the same layer input and
        write in parallel -- none is upstream of another.  Indexing by node position instead
        would make ``(0,3)`` a parent of ``(0,5)`` and inflate the graph by 6 * C(16,2) = 720
        edges that do not exist in the architecture.
        """
        order: dict[Node, int] = {EMBED: 0}
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                order[(layer, head)] = 2 * layer + 1
            order[(layer, "mlp")] = 2 * layer + 2
        order[READOUT] = 2 * self.num_layers + 1
        return order

    # ---------------------------------------------------------------------------------
    # Slots.
    # ---------------------------------------------------------------------------------
    def slots_for(self, dst: Node) -> tuple[Slot, ...]:
        if self.qkv_split and is_head(dst):
            return ("q", "k", "v")
        return ("in",)

    @property
    def n_slots(self) -> int:
        return 4 if self.qkv_split else 1

    # ---------------------------------------------------------------------------------
    # Edges.
    # ---------------------------------------------------------------------------------
    def _iter_edges(self) -> Iterator[Edge]:
        order = self._write_order()
        nodes = self.nodes
        for dst in nodes[1:] + [READOUT]:            # embed has no parents
            for src in nodes:
                if order[src] >= order[dst]:
                    continue
                for slot in self.slots_for(dst):
                    yield (src, dst, slot)

    @property
    def edges(self) -> list[Edge]:
        return list(self._iter_edges())

    @property
    def edge_set(self) -> set[Edge]:
        return set(self._iter_edges())

    @property
    def n_edges(self) -> int:
        return sum(1 for _ in self._iter_edges())

    def parents(self, dst: Node) -> list[tuple[Node, Slot]]:
        if dst == EMBED:
            return []
        order = self._write_order()
        return [
            (src, slot)
            for src in self.nodes
            if order[src] < order[dst]
            for slot in self.slots_for(dst)
        ]

    # ---------------------------------------------------------------------------------
    # Ordering.  ACDC iterates output-first (paper section 3).
    # ---------------------------------------------------------------------------------
    def reverse_topological(self) -> list[Node]:
        return list(reversed(self.nodes + [READOUT]))

    # ---------------------------------------------------------------------------------
    # The mask.  D7.2 -- edge removal is a flip, not a set rebuild, so the graph is
    # structurally static across the whole sweep and the inner loop stays on-device.
    # ---------------------------------------------------------------------------------
    def edge_slot_index(self, edge: Edge) -> tuple[int, int, int]:
        src, dst, slot = edge
        index = self.node_index
        return (index[src], index[dst], _SLOT_INDEX[slot])

    def full_mask(self, device: torch.device | str | None = None) -> torch.Tensor:
        """``[n_nodes + 1, n_nodes + 1, n_slots]`` bool; True exactly on real edges."""
        size = self.n_nodes + 1
        mask = torch.zeros((size, size, self.n_slots), dtype=torch.bool, device=device)
        for edge in self._iter_edges():
            mask[self.edge_slot_index(edge)] = True
        return mask
