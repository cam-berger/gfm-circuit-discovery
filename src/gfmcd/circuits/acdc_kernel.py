"""Exact edge-level subgraph evaluation for post-LayerNorm BERT encoders.

WHY THE STANDARD ACDC KERNEL DOES NOT APPLY
-------------------------------------------
ACDC (arXiv:2304.14997) evaluates a subgraph ``H`` by giving each node an input equal to the
sum of its in-``H`` parents' live writes plus its out-of-``H`` parents' cached corrupt writes.
That construction needs ``r_v = sum_u write(u)``, which holds in the pre-LayerNorm decoders
ACDC was built on.

**SpliceBERT is post-LayerNorm.**  ``BertLayer`` computes ``LayerNorm(dense(context) + input)``
and then ``LayerNorm(dense(intermediate) + attn_output)``, so the residual stream is
renormalised twice per layer and is not a sum of upstream writes at any depth.

This module does not linearise, fold, or freeze LayerNorm -- an approximation whose error is
worst exactly at the small-``dF`` decisions ACDC's threshold rule turns on.  It replays the
true recursion once per destination:

    r_v = sel_v(embed)
    for each block b upstream of v, in write order:
        r_v = LN_b(r_v + sel_v(b))

    sel_v(b) = live_write(b)            if edge b -> v is in H
             = corrupt_cached_write(b)  otherwise

This is exact.  It reduces to the true clean forward when every edge is present and to the
true corrupt forward when none is, and both reductions gate the run (A14 section 5).

BIASES ARE ARCHITECTURE.  ``attention.output.dense`` is linear over the concatenated per-head
contexts, so the attention write decomposes exactly as ``sum_h W_O^(h) c_h + b_O``.  ``b_O``
and the MLP output bias are added on every forward pass and are never ablated: a circuit here
is a set of learned routing choices, not a claim that the biases are inert.

COST.  Heads in a layer no longer share an input, so their projections are computed per head.
They are batched over the head axis rather than looped (D7.3); ``acdc_device`` handles the
precision policy that keeps the decision path in fp32 (D7.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from gfmcd.circuits.acdc_graph import EMBED, READOUT, EdgeGraph, Node, Slot, is_head


@dataclass(frozen=True)
class NodeWrites:
    """Every node's residual write from one forward pass, decomposed to head granularity.

    ``attention[l]`` is ``[H, B, S, D]`` -- the per-head contribution *after* ``W_O`` and
    *before* ``b_O``.  ``mlp[l]`` is ``[B, S, D]`` before its bias.  ``embed`` is the
    post-LayerNorm embedding output, i.e. the initial residual.
    """

    embed: torch.Tensor
    attention: dict[int, torch.Tensor]
    mlp: dict[int, torch.Tensor]
    attention_bias: dict[int, torch.Tensor]
    mlp_bias: dict[int, torch.Tensor]


def _encoder(model: Any):
    return model.bert.encoder.layer


def _head_dim(model: Any) -> int:
    config = model.config
    return config.hidden_size // config.num_attention_heads


def _split_heads(context: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``[B, S, D] -> [H, B, S, D_head]``."""
    b, s, d = context.shape
    return context.view(b, s, num_heads, d // num_heads).permute(2, 0, 1, 3).contiguous()


def _per_head_write(model: Any, layer: int, context: torch.Tensor) -> torch.Tensor:
    """``[B, S, D] -> [H, B, S, D]``: each head's contribution through ``W_O``, no bias."""
    dense = _encoder(model)[layer].attention.output.dense
    num_heads = model.config.num_attention_heads
    dh = _head_dim(model)
    heads = _split_heads(context, num_heads)                       # [H, B, S, Dh]
    # HF Linear: y = x @ W.T; head h owns input columns [h*Dh, (h+1)*Dh).
    weight = dense.weight.view(dense.weight.shape[0], num_heads, dh).permute(1, 2, 0)
    return torch.einsum("hbsd,hdo->hbso", heads, weight)           # [H, B, S, D]


def capture_node_writes(model: Any, inputs: Mapping[str, torch.Tensor]) -> NodeWrites:
    """Run one forward pass and decompose it into per-node residual writes."""
    contexts: dict[int, torch.Tensor] = {}
    intermediates: dict[int, torch.Tensor] = {}
    embed_box: dict[str, torch.Tensor] = {}
    layers = _encoder(model)
    handles = []

    def attn_hook(layer: int):
        def hook(_module, _inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            contexts[layer] = tensor.detach()
        return hook

    def mlp_hook(layer: int):
        def hook(_module, inputs, _output):
            intermediates[layer] = inputs[0].detach()
        return hook

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.attention.self.register_forward_hook(attn_hook(index)))
            handles.append(layer.output.dense.register_forward_hook(mlp_hook(index)))
        handles.append(
            model.bert.embeddings.register_forward_hook(
                lambda _m, _i, out: embed_box.__setitem__("e", out.detach())
            )
        )
        with torch.no_grad():
            model(**dict(inputs))
    finally:
        for handle in handles:
            handle.remove()

    attention, mlp, attn_bias, mlp_bias = {}, {}, {}, {}
    for index, layer in enumerate(layers):
        attention[index] = _per_head_write(model, index, contexts[index])
        dense = layer.output.dense
        mlp[index] = intermediates[index] @ dense.weight.T
        attn_bias[index] = layer.attention.output.dense.bias.detach()
        mlp_bias[index] = dense.bias.detach()

    return NodeWrites(
        embed=embed_box["e"], attention=attention, mlp=mlp,
        attention_bias=attn_bias, mlp_bias=mlp_bias,
    )


class AcdcForward:
    """Evaluate ``H(x, x')`` for an arbitrary edge mask over a post-LN BERT encoder."""

    def __init__(self, model: Any, graph: EdgeGraph):
        self.model = model
        self.graph = graph
        self.layers = _encoder(model)
        self.num_heads = model.config.num_attention_heads
        self.head_dim = _head_dim(model)
        self.index = graph.node_index
        self._stage = graph._write_order()

    # ---------------------------------------------------------------------------------
    # Selection and the LN replay.
    # ---------------------------------------------------------------------------------
    def _build_inputs(
        self,
        destinations: list[Node],
        slot: Slot,
        live: NodeWrites,
        corrupt: NodeWrites,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replay the LayerNorm recursion for many destinations at once -> ``[n_dst, B, S, D]``.

        D7.2.  An earlier version selected per head with a Python ``bool(mask[...])``, which
        issues ~10,000 device syncs per forward pass and measured **9.2x** a plain forward on
        real SpliceBERT -- the cost was Python and synchronisation, not arithmetic.  Selection
        is therefore a tensor op over a destination axis, and the mask never leaves the device.

        All destinations must share a write stage, which is what makes one shared recursion
        valid: they see the same set of upstream blocks and differ only in their parent masks.
        """
        from gfmcd.circuits.acdc_graph import _SLOT_INDEX

        slot_index = _SLOT_INDEX[slot]
        stages = {self._stage[d] for d in destinations}
        if len(stages) != 1:
            raise ValueError(f"destinations span several write stages: {sorted(stages)}")
        stage_dst = stages.pop()

        device = mask.device
        n = len(destinations)
        j = torch.tensor([self.index[d] for d in destinations], device=device)

        keep_embed = mask[self.index[EMBED], j, slot_index].view(n, 1, 1, 1)
        residual = torch.where(
            keep_embed, live.embed.unsqueeze(0), corrupt.embed.unsqueeze(0)
        )

        for layer_index, layer in enumerate(self.layers):
            if self._stage[(layer_index, 0)] < stage_dst:
                head_rows = torch.tensor(
                    [self.index[(layer_index, h)] for h in range(self.num_heads)],
                    device=device,
                )
                live_heads = live.attention[layer_index]                 # [H, B, S, D]
                corrupt_heads = corrupt.attention[layer_index]
                keep = mask[head_rows][:, j, slot_index].to(residual.dtype)   # [H, n]
                # sum_h sel(h, dst) == sum_h corrupt_h + sum_h keep[h,dst] * (live_h - corrupt_h)
                write = corrupt_heads.sum(0).unsqueeze(0) + torch.einsum(
                    "hn,hbsd->nbsd", keep, live_heads - corrupt_heads
                )
                residual = layer.attention.output.LayerNorm(
                    residual + write + live.attention_bias[layer_index]
                )
            if self._stage[(layer_index, "mlp")] < stage_dst:
                keep_mlp = mask[self.index[(layer_index, "mlp")], j, slot_index].view(n, 1, 1, 1)
                write = torch.where(
                    keep_mlp,
                    live.mlp[layer_index].unsqueeze(0),
                    corrupt.mlp[layer_index].unsqueeze(0),
                )
                residual = layer.output.LayerNorm(
                    residual + write + live.mlp_bias[layer_index]
                )
        return residual

    # ---------------------------------------------------------------------------------
    # Live writes, computed stage by stage from each node's own reconstructed input.
    # ---------------------------------------------------------------------------------
    def _attention_context(
        self, layer_index: int, head_inputs: torch.Tensor, attention_bias: torch.Tensor
    ) -> torch.Tensor:
        """``head_inputs`` ``[H, B, S, D]`` -> per-head context ``[H, B, S, D_head]``."""
        selfattn = self.layers[layer_index].attention.self
        h, dh = self.num_heads, self.head_dim

        def project(module) -> torch.Tensor:
            weight = module.weight.view(h, dh, -1)                 # [H, Dh, D]
            bias = module.bias.view(h, dh)                         # [H, Dh]
            return torch.einsum("hbsd,hed->hbse", head_inputs, weight) + bias[:, None, None, :]

        query, key, value = project(selfattn.query), project(selfattn.key), project(selfattn.value)
        # [H, B, S, Dh] -> [B, H, S, Dh] for scaled_dot_product_attention.
        query, key, value = (t.permute(1, 0, 2, 3) for t in (query, key, value))
        context = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_bias
        )
        return context.permute(1, 0, 2, 3)                          # [H, B, S, Dh]

    def _extended_attention_mask(
        self, inputs: Mapping[str, torch.Tensor], dtype: torch.dtype
    ) -> torch.Tensor | None:
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            return None
        extended = attention_mask[:, None, None, :].to(dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    def _run(
        self,
        clean_inputs: Mapping[str, torch.Tensor],
        corrupt: NodeWrites,
        mask: torch.Tensor,
    ) -> tuple[NodeWrites, dict[tuple[Node, Slot], torch.Tensor]]:
        with torch.no_grad():
            clean_embed = self.model.bert.embeddings(
                input_ids=clean_inputs["input_ids"],
                token_type_ids=clean_inputs.get("token_type_ids"),
            )
        # Masks are built by EdgeGraph on CPU by default and the model may live anywhere.
        # Moving once here keeps every caller -- runner, equivalence check, floor panel --
        # from having to know the device, and the mask is small enough that the copy is free
        # relative to the forward pass it gates.
        mask = mask.to(clean_embed.device)
        attn_bias = self._extended_attention_mask(clean_inputs, clean_embed.dtype)

        live = NodeWrites(
            embed=clean_embed, attention={}, mlp={},
            attention_bias=corrupt.attention_bias, mlp_bias=corrupt.mlp_bias,
        )
        destination_inputs: dict[tuple[Node, Slot], torch.Tensor] = {}

        with torch.no_grad():
            for layer_index, layer in enumerate(self.layers):
                head_slots = self.graph.slots_for((layer_index, 0))
                heads = [(layer_index, head) for head in range(self.num_heads)]
                per_slot: dict[Slot, torch.Tensor] = {}
                for slot in head_slots:
                    stacked = self._build_inputs(heads, slot, live, corrupt, mask)
                    per_slot[slot] = stacked
                    for head in range(self.num_heads):
                        destination_inputs[((layer_index, head), slot)] = stacked[head]

                if head_slots == ("in",):
                    context = self._attention_context(layer_index, per_slot["in"], attn_bias)
                else:
                    context = self._attention_context_split(layer_index, per_slot, attn_bias)

                merged = context.permute(1, 2, 0, 3).reshape(
                    context.shape[1], context.shape[2], self.num_heads * self.head_dim
                )
                live.attention[layer_index] = _per_head_write(self.model, layer_index, merged)

                mlp_input = self._build_inputs(
                    [(layer_index, "mlp")], "in", live, corrupt, mask
                )[0]
                destination_inputs[((layer_index, "mlp"), "in")] = mlp_input
                intermediate = layer.intermediate(mlp_input)
                live.mlp[layer_index] = intermediate @ layer.output.dense.weight.T

            readout_input = self._build_inputs([READOUT], "in", live, corrupt, mask)[0]
            destination_inputs[(READOUT, "in")] = readout_input

        return live, destination_inputs

    def _attention_context_split(
        self, layer_index: int, per_slot: dict[Slot, torch.Tensor], attention_bias
    ) -> torch.Tensor:
        """Q/K/V each read their own reconstructed input (the qkv_split arm)."""
        selfattn = self.layers[layer_index].attention.self
        h, dh = self.num_heads, self.head_dim

        def project(module, head_inputs: torch.Tensor) -> torch.Tensor:
            weight = module.weight.view(h, dh, -1)
            bias = module.bias.view(h, dh)
            return torch.einsum("hbsd,hed->hbse", head_inputs, weight) + bias[:, None, None, :]

        query = project(selfattn.query, per_slot["q"]).permute(1, 0, 2, 3)
        key = project(selfattn.key, per_slot["k"]).permute(1, 0, 2, 3)
        value = project(selfattn.value, per_slot["v"]).permute(1, 0, 2, 3)
        context = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_bias
        )
        return context.permute(1, 0, 2, 3)

    # ---------------------------------------------------------------------------------
    # Public API.
    # ---------------------------------------------------------------------------------
    def destination_inputs(
        self,
        clean_inputs: Mapping[str, torch.Tensor],
        corrupt: NodeWrites,
        mask: torch.Tensor,
    ) -> dict[tuple[Node, Slot], torch.Tensor]:
        return self._run(clean_inputs, corrupt, mask)[1]

    def logits(
        self,
        clean_inputs: Mapping[str, torch.Tensor],
        corrupt: NodeWrites,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        _live, inputs = self._run(clean_inputs, corrupt, mask)
        with torch.no_grad():
            return self.model.cls(inputs[(READOUT, "in")])
