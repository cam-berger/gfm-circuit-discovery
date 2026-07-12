"""QK/OV diagnostics for BERT attention heads.

These are reverse-engineering aids after causal localization.  A large QK logit
or an interpretable OV write is not itself causal evidence; the study order is
behavior -> ablation/patching -> QK/OV analysis.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F

from gfmcd.adapters.splicebert import SpliceBertAdapter


def _qkv_modules(adapter: SpliceBertAdapter, layer: int):
    attn = adapter.attention_self_modules[layer]
    for name in ("query", "key", "value"):
        if not hasattr(attn, name):
            raise AttributeError(
                f"attention module at layer {layer} has no {name!r}; "
                "add an architecture-specific adapter before using QK/OV tools"
            )
    return attn.query, attn.key, attn.value


def _head_slice(adapter: SpliceBertAdapter, head: int) -> slice:
    if not 0 <= head < adapter.num_heads:
        raise ValueError("head out of range")
    d = adapter.head_dim
    return slice(head * d, (head + 1) * d)


def qk_logits(
    adapter: SpliceBertAdapter,
    layer_input: torch.Tensor,
    *,
    layer: int,
    head: int,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> torch.Tensor:
    """Scaled query-key logits for aligned position pairs.

    ``layer_input`` is ``hidden_states[layer]`` from a Hugging Face output:
    embeddings for layer 0, then each preceding layer's output.
    ``query_positions`` and ``key_positions`` are ``[batch, n_pairs]``.
    """
    query, key, _value = _qkv_modules(adapter, layer)
    if query_positions.shape != key_positions.shape:
        raise ValueError("query/key position shapes disagree")
    if layer_input.ndim != 3:
        raise ValueError("layer_input must be [B,S,D]")
    batch = layer_input.shape[0]
    if query_positions.shape[0] != batch:
        raise ValueError("position batch does not match hidden states")
    b = torch.arange(batch, device=layer_input.device)[:, None]
    q_in = layer_input[b, query_positions]
    k_in = layer_input[b, key_positions]
    hs = _head_slice(adapter, head)
    q = query(q_in)[..., hs]
    k = key(k_in)[..., hs]
    return (q * k).sum(-1) / math.sqrt(adapter.head_dim)


def attention_output_dense(adapter: SpliceBertAdapter, layer: int):
    layer_module = adapter.encoder_layers[layer]
    candidates = (
        getattr(getattr(layer_module, "attention", None), "output", None),
        getattr(layer_module, "attention_output", None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "dense"):
            return candidate.dense
    raise AttributeError(f"cannot locate attention output projection at layer {layer}")


def ov_write_from_context(
    adapter: SpliceBertAdapter,
    head_context: torch.Tensor,
    *,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Exact bias-free residual write of one head after BERT's output matrix.

    ``head_context`` has shape ``[..., D_head]`` and is the weighted sum of value
    vectors for the selected head.  BERT's output bias is shared by all heads and
    is intentionally excluded from per-head attribution.
    """
    if head_context.shape[-1] != adapter.head_dim:
        raise ValueError("head_context has wrong width")
    dense = attention_output_dense(adapter, layer)
    hs = _head_slice(adapter, head)
    weight = dense.weight[:, hs]  # [D_model, D_head]
    return F.linear(head_context, weight, bias=None)


def top_qk_pairs(
    logits: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    k: int = 20,
) -> list[dict]:
    flat = logits.detach().cpu().flatten()
    top = torch.topk(flat, min(k, flat.numel()))
    width = logits.shape[1]
    out = []
    qp = query_positions.detach().cpu()
    kp = key_positions.detach().cpu()
    for value, index in zip(top.values, top.indices):
        b = int(index // width)
        p = int(index % width)
        out.append(
            {
                "batch": b,
                "pair_index": p,
                "query_position": int(qp[b, p]),
                "key_position": int(kp[b, p]),
                "qk_logit": float(value),
            }
        )
    return out
