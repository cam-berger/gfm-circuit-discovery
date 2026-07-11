"""Attribution patching (linear/Taylor) for head outputs — a fast pre-screen.

Standard denoising attribution: the effect of patching head ``h``'s output from a
clean run into a corrupt run is approximated to first order by

    attr(h) ~= sum_over(batch, seq, d_head) [ (clean_h - corrupt_h) * d(metric)/d(corrupt_h) ]

so a single corrupt forward + one backward ranks all heads at once (Nanda 2023;
Syed, Rager & Conmy 2023). It is an APPROXIMATION and is only used to shortlist
candidates; the pre-registered gates are always evaluated with exact ablation /
patching downstream.

Works with a frozen backbone: activations from frozen ops carry no grad by
default, so each attention-context tensor is tagged ``requires_grad_(True)`` in a
forward hook (grad still flows through frozen weights to the activation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.circuits.hf_bert import _context_from_output, _split_context


@dataclass(frozen=True)
class AttributionEffect:
    layer: int
    head: int
    attribution: float

    def to_dict(self) -> dict:
        return {"layer": self.layer, "head": self.head, "attribution": self.attribution}


def head_attribution_scan(
    adapter: SpliceBertAdapter,
    source_cache: Mapping[int, torch.Tensor],
    corrupt_metric_tensor_fn: Callable[[], torch.Tensor],
) -> list[AttributionEffect]:
    """Rank every head by its first-order clean->corrupt denoising attribution.

    ``source_cache``: clean per-head outputs ``[B,H,S,Dh]`` (from
    ``capture_head_outputs``). ``corrupt_metric_tensor_fn`` must run the *corrupt*
    forward and return a **scalar tensor** metric (higher = better), differentiable
    back to the attention-context tensors. Positive attribution = patching the
    clean head output raises the metric (mirrors positive exact recovery).
    """
    acts: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            ctx = _context_from_output(output)
            ctx.requires_grad_(True)
            ctx.retain_grad()
            acts[layer] = ctx
            return output

        return hook

    try:
        for layer, module in enumerate(adapter.attention_self_modules):
            handles.append(module.register_forward_hook(make_hook(layer)))
        with torch.enable_grad():
            metric = corrupt_metric_tensor_fn()
            if not (torch.is_tensor(metric) and metric.ndim == 0):
                raise ValueError("corrupt_metric_tensor_fn must return a scalar tensor")
            adapter.model.zero_grad(set_to_none=True)
            metric.backward()
        missing = set(range(adapter.num_layers)) - set(acts)
        if missing:
            raise RuntimeError(f"no attention output captured for layers {sorted(missing)}")
        results: list[AttributionEffect] = []
        for layer in range(adapter.num_layers):
            ctx = acts[layer]
            if ctx.grad is None:
                raise RuntimeError(f"no gradient reached layer {layer}; metric is not differentiable")
            corrupt = _split_context(ctx.detach(), adapter.num_heads)  # [B,H,S,Dh]
            grad = _split_context(ctx.grad, adapter.num_heads)
            clean = source_cache[layer].to(device=corrupt.device, dtype=corrupt.dtype)
            if clean.shape != corrupt.shape:
                raise ValueError(
                    f"clean/corrupt head-cache shapes disagree at layer {layer}: "
                    f"{tuple(clean.shape)} vs {tuple(corrupt.shape)}"
                )
            attr = ((clean - corrupt) * grad).sum(dim=(0, 2, 3))  # per head
            for head in range(adapter.num_heads):
                results.append(AttributionEffect(layer, head, float(attr[head])))
    finally:
        for handle in handles:
            handle.remove()
    return sorted(results, key=lambda x: x.attribution, reverse=True)
