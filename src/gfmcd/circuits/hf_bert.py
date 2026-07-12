"""Causal head interventions for BERT-style genomic models.

These utilities edit the **per-head context vectors before the attention output
projection**.  They therefore intervene on a well-defined head output rather
than zeroing an arbitrary slice of the residual stream.

The first real-model pass intentionally starts with head-level interventions.
MLP features and edge-level mediation are added only after a reproducible
behavioral effect exists; the protocol forbids tracing an unproven behavior.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import random
from typing import Callable, Iterable, Iterator, Literal, Mapping

import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter

Head = tuple[int, int]
EditMode = Literal["zero", "mean", "resample"]


def _split_context(context: torch.Tensor, num_heads: int) -> torch.Tensor:
    if context.ndim != 3:
        raise ValueError(f"attention context must be [B,S,D], got {tuple(context.shape)}")
    b, s, d = context.shape
    if d % num_heads:
        raise ValueError("context width is not divisible by num_heads")
    return context.view(b, s, num_heads, d // num_heads).permute(0, 2, 1, 3)


def _merge_context(heads: torch.Tensor) -> torch.Tensor:
    b, h, s, d = heads.shape
    return heads.permute(0, 2, 1, 3).contiguous().view(b, s, h * d)


def _replace_output(output, new_context: torch.Tensor):
    if torch.is_tensor(output):
        return new_context
    if isinstance(output, tuple):
        return (new_context,) + output[1:]
    if isinstance(output, list):
        return [new_context] + output[1:]
    raise TypeError(f"unsupported attention output type: {type(output)!r}")


def _context_from_output(output) -> torch.Tensor:
    context = output if torch.is_tensor(output) else output[0]
    if not torch.is_tensor(context):
        raise TypeError("first attention output is not a tensor")
    return context


def _validate_heads(adapter: SpliceBertAdapter, heads: Iterable[Head]) -> list[Head]:
    unique = sorted(set(heads))
    for layer, head in unique:
        if not 0 <= layer < adapter.num_layers:
            raise ValueError(f"layer out of range: {layer}")
        if not 0 <= head < adapter.num_heads:
            raise ValueError(f"head out of range: {head}")
    return unique


@contextmanager
def edit_heads(
    adapter: SpliceBertAdapter,
    heads: Iterable[Head],
    *,
    mode: EditMode = "zero",
) -> Iterator[None]:
    """Temporarily zero/mean/resample selected head context vectors."""
    selected = _validate_heads(adapter, heads)
    by_layer: dict[int, list[int]] = {}
    for layer, head in selected:
        by_layer.setdefault(layer, []).append(head)
    handles = []

    def make_hook(layer_heads: list[int]):
        def hook(_module, _inputs, output):
            context = _context_from_output(output)
            split = _split_context(context, adapter.num_heads).clone()
            for head in layer_heads:
                source = split[:, head]
                if mode == "zero":
                    replacement = torch.zeros_like(source)
                elif mode == "mean":
                    replacement = source.mean(dim=(0, 1), keepdim=True).expand_as(source)
                elif mode == "resample":
                    if source.shape[0] < 2:
                        raise ValueError("resample ablation requires batch size >= 2")
                    replacement = source.roll(shifts=1, dims=0)
                else:  # pragma: no cover
                    raise ValueError(f"unknown mode: {mode}")
                split[:, head] = replacement
            return _replace_output(output, _merge_context(split))

        return hook

    try:
        for layer, layer_heads in by_layer.items():
            handles.append(
                adapter.attention_self_modules[layer].register_forward_hook(make_hook(layer_heads))
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


def capture_head_outputs(
    adapter: SpliceBertAdapter,
    run_forward: Callable[[], object],
) -> dict[int, torch.Tensor]:
    """Run a forward pass and cache ``[B,H,S,D_head]`` for every layer."""
    cache: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            context = _context_from_output(output)
            cache[layer] = _split_context(context, adapter.num_heads).detach().clone()

        return hook

    try:
        for layer, module in enumerate(adapter.attention_self_modules):
            handles.append(module.register_forward_hook(make_hook(layer)))
        run_forward()
    finally:
        for handle in handles:
            handle.remove()
    missing = set(range(adapter.num_layers)) - set(cache)
    if missing:
        raise RuntimeError(f"failed to capture attention outputs for layers {sorted(missing)}")
    return cache


@contextmanager
def patch_heads(
    adapter: SpliceBertAdapter,
    source_cache: Mapping[int, torch.Tensor],
    heads: Iterable[Head],
    *,
    token_mask: torch.Tensor | None = None,
) -> Iterator[None]:
    """Patch selected clean head outputs into a corrupt forward pass.

    ``token_mask`` is an optional boolean ``[B,S]`` mask.  Without it, the whole
    head output is patched.  Clean and corrupt examples must have aligned token
    coordinates; the conflict builder records and tests this precondition.
    """
    selected = _validate_heads(adapter, heads)
    by_layer: dict[int, list[int]] = {}
    for layer, head in selected:
        by_layer.setdefault(layer, []).append(head)
    handles = []

    def make_hook(layer: int, layer_heads: list[int]):
        def hook(_module, _inputs, output):
            context = _context_from_output(output)
            target = _split_context(context, adapter.num_heads).clone()
            source = source_cache[layer].to(device=target.device, dtype=target.dtype)
            if source.shape != target.shape:
                raise ValueError(
                    f"clean/corrupt head-cache shapes disagree at layer {layer}: "
                    f"{tuple(source.shape)} vs {tuple(target.shape)}"
                )
            if token_mask is not None:
                mask = token_mask.to(target.device)
                if mask.shape != (target.shape[0], target.shape[2]):
                    raise ValueError("token_mask must have shape [batch, sequence]")
            for head in layer_heads:
                if token_mask is None:
                    target[:, head] = source[:, head]
                else:
                    target[:, head] = torch.where(
                        mask[..., None], source[:, head], target[:, head]
                    )
            return _replace_output(output, _merge_context(target))

        return hook

    try:
        for layer, layer_heads in by_layer.items():
            if layer not in source_cache:
                raise KeyError(f"source cache has no layer {layer}")
            handles.append(
                adapter.attention_self_modules[layer].register_forward_hook(
                    make_hook(layer, layer_heads)
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


@dataclass(frozen=True)
class HeadEffect:
    layer: int
    head: int
    baseline_score: float
    intervened_score: float
    effect: float
    mode: str

    def to_dict(self) -> dict:
        return asdict(self)


def head_ablation_scan(
    adapter: SpliceBertAdapter,
    score_fn: Callable[[], float],
    *,
    heads: Iterable[Head] | None = None,
    mode: EditMode = "zero",
) -> list[HeadEffect]:
    """Measure score loss from ablating each head independently.

    ``score_fn`` must return a scalar where **higher means better behavior**
    (usually mean true-vs-best-decoy margin).  Use a separate loss scan when the
    scientific metric is loss; do not change signs ad hoc in downstream prose.
    """
    candidates = _validate_heads(
        adapter,
        heads
        if heads is not None
        else [
            (layer, head)
            for layer in range(adapter.num_layers)
            for head in range(adapter.num_heads)
        ],
    )
    baseline = float(score_fn())
    results: list[HeadEffect] = []
    for layer, head in candidates:
        with edit_heads(adapter, [(layer, head)], mode=mode):
            intervened = float(score_fn())
        results.append(
            HeadEffect(layer, head, baseline, intervened, baseline - intervened, mode)
        )
    return sorted(results, key=lambda x: x.effect, reverse=True)


@dataclass(frozen=True)
class PatchEffect:
    layer: int
    head: int
    clean_score: float
    corrupt_score: float
    patched_score: float
    recovery_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def head_patch_scan(
    adapter: SpliceBertAdapter,
    *,
    clean_score: float,
    corrupt_score: float,
    source_cache: Mapping[int, torch.Tensor],
    corrupt_score_fn: Callable[[], float],
    heads: Iterable[Head] | None = None,
    token_mask: torch.Tensor | None = None,
) -> list[PatchEffect]:
    gap = clean_score - corrupt_score
    if abs(gap) < 1e-12:
        raise ValueError("clean/corrupt behavior gap is zero; patching is not identifiable")
    candidates = _validate_heads(
        adapter,
        heads
        if heads is not None
        else [
            (layer, head)
            for layer in range(adapter.num_layers)
            for head in range(adapter.num_heads)
        ],
    )
    out: list[PatchEffect] = []
    for layer, head in candidates:
        with patch_heads(
            adapter,
            source_cache,
            [(layer, head)],
            token_mask=token_mask,
        ):
            patched = float(corrupt_score_fn())
        out.append(
            PatchEffect(
                layer=layer,
                head=head,
                clean_score=clean_score,
                corrupt_score=corrupt_score,
                patched_score=patched,
                recovery_fraction=(patched - corrupt_score) / gap,
            )
        )
    return sorted(out, key=lambda x: x.recovery_fraction, reverse=True)


def matched_random_head_sets(
    adapter: SpliceBertAdapter,
    selected: Iterable[Head],
    *,
    n: int = 1000,
    seed: int = 0,
    exclude_selected: bool = False,
) -> list[tuple[Head, ...]]:
    """Distinct matched-random head sets.

    THE SETS ARE DE-DUPLICATED, and when the universe is small enough they are ENUMERATED.

    Mean-ablation is deterministic, so evaluating the same head set twice yields the same
    number.  Sampling ``n`` sets WITH replacement from a small universe therefore does not
    buy resolution -- it buys copies.  With ``k=1`` and 95 candidate heads, ``n=1000`` draws
    produced 95 distinct values duplicated ~10.5x, and the add-one p-value then divided by
    n+1 = 1001 instead of by the 95 exchangeable units.  The result was p = 1/1001 =
    0.000999 -- a number determined by the ``--n-null`` flag rather than by the data, and it
    overstated the evidence by 10.4x.  The honest value for that design is 1/96 = 0.0104.

    Returning distinct sets makes the denominator of the empirical p-value equal to the
    number of things actually being compared.
    """
    from itertools import combinations
    from math import comb

    selected_tuple = tuple(_validate_heads(adapter, selected))
    k = len(selected_tuple)
    if k == 0:
        raise ValueError("selected head set cannot be empty")
    universe = [
        (layer, head)
        for layer in range(adapter.num_layers)
        for head in range(adapter.num_heads)
        if not exclude_selected or (layer, head) not in selected_tuple
    ]
    if len(universe) < k:
        raise ValueError("not enough heads for matched random sets")

    total = comb(len(universe), k)
    if total <= n:
        # the null is exhaustible -- enumerate it and stop pretending it is a sample
        return [tuple(sorted(c)) for c in combinations(universe, k)]
    rng = random.Random(seed)
    seen: set[tuple[Head, ...]] = set()
    while len(seen) < n:
        seen.add(tuple(sorted(rng.sample(universe, k))))
    return sorted(seen)
