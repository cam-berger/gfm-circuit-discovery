"""Residual-stream capture and patching for BERT-style encoder layers.

These utilities complement head-output patching.  They locate the earliest
layer at which clean information can restore a corrupt behavior, which helps
bound the circuit before QK/OV or MLP reverse engineering.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable, Iterator, Mapping

import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter


def _hidden_from_output(output) -> torch.Tensor:
    hidden = output if torch.is_tensor(output) else output[0]
    if not torch.is_tensor(hidden):
        raise TypeError("layer output does not begin with a hidden-state tensor")
    return hidden


def _replace_hidden(output, hidden: torch.Tensor):
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + output[1:]
    raise TypeError(f"unsupported layer output type: {type(output)!r}")


def capture_layer_outputs(
    adapter: SpliceBertAdapter,
    run_forward: Callable[[], object],
) -> dict[int, torch.Tensor]:
    cache: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            cache[layer_index] = _hidden_from_output(output).detach().clone()
        return hook

    try:
        for index, layer in enumerate(adapter.encoder_layers):
            handles.append(layer.register_forward_hook(make_hook(index)))
        run_forward()
    finally:
        for handle in handles:
            handle.remove()
    if len(cache) != adapter.num_layers:
        raise RuntimeError("failed to capture every encoder layer")
    return cache


@contextmanager
def patch_layer_output(
    adapter: SpliceBertAdapter,
    source_cache: Mapping[int, torch.Tensor],
    layer: int,
    *,
    token_mask: torch.Tensor | None = None,
) -> Iterator[None]:
    if not 0 <= layer < adapter.num_layers:
        raise ValueError("layer out of range")
    if layer not in source_cache:
        raise KeyError(f"source cache lacks layer {layer}")

    def hook(_module, _inputs, output):
        target = _hidden_from_output(output).clone()
        source = source_cache[layer].to(target.device, target.dtype)
        if source.shape != target.shape:
            raise ValueError("clean/corrupt layer-output shapes disagree")
        if token_mask is None:
            replacement = source
        else:
            mask = token_mask.to(target.device)
            if mask.shape != target.shape[:2]:
                raise ValueError("token_mask must have shape [batch, sequence]")
            replacement = torch.where(mask[..., None], source, target)
        return _replace_hidden(output, replacement)

    handle = adapter.encoder_layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@dataclass(frozen=True)
class LayerPatchEffect:
    layer: int
    clean_score: float
    corrupt_score: float
    patched_score: float
    recovery_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def layer_patch_scan(
    adapter: SpliceBertAdapter,
    *,
    source_cache: Mapping[int, torch.Tensor],
    clean_score: float,
    corrupt_score: float,
    corrupt_score_fn: Callable[[], float],
    token_mask: torch.Tensor | None = None,
) -> list[LayerPatchEffect]:
    gap = clean_score - corrupt_score
    if abs(gap) < 1e-12:
        raise ValueError("clean/corrupt gap is zero")
    results = []
    for layer in range(adapter.num_layers):
        with patch_layer_output(adapter, source_cache, layer, token_mask=token_mask):
            patched = float(corrupt_score_fn())
        results.append(
            LayerPatchEffect(
                layer=layer,
                clean_score=clean_score,
                corrupt_score=corrupt_score,
                patched_score=patched,
                recovery_fraction=(patched - corrupt_score) / gap,
            )
        )
    return results
