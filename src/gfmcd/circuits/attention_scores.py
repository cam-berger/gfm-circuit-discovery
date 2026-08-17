"""Read-only capture of BERT pre-softmax attention scores.

The capture deliberately uses a forward pre-hook while the model remains in its native eager
attention implementation.  A custom attention-registry implementation can silently receive no
padding mask in Transformers 5.13; recomputing QK scores from the eager module's actual inputs avoids
that measurement-induced change.  The returned scores are QK/logit geometry.  By themselves they do
not distinguish query changes from target-key or competitor-key changes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter


class CapturedScores:
    """Box holding the most recent forward pass's pre-softmax attention scores."""

    __slots__ = ("scores",)

    def __init__(self) -> None:
        self.scores: torch.Tensor | None = None

    def require(self) -> torch.Tensor:
        """Return captured scores, failing loudly if no forward pass reached the hook."""

        if self.scores is None:
            raise RuntimeError(
                "no attention scores were captured -- did a forward pass run inside the "
                "capture_attention_scores(...) context?"
            )
        return self.scores


@contextmanager
def capture_attention_scores(
    adapter: SpliceBertAdapter,
    layer: int,
    *,
    head: int | None = None,
) -> Iterator[CapturedScores]:
    """Capture one layer's post-mask, pre-softmax score tensor without changing its forward.

    The yielded tensor has shape ``[B,H,S,S]`` when ``head`` is ``None`` and ``[B,S,S]`` for a
    selected head.  Each forward overwrites the prior value.  The model must be running eager
    attention when callers compare these scores with ``output_attentions=True``.
    """

    if not 0 <= layer < adapter.num_layers:
        raise ValueError(f"layer out of range: {layer}")
    if head is not None and not 0 <= head < adapter.num_heads:
        raise ValueError(f"head out of range: {head}")

    module = adapter.attention_self_modules[layer]
    head_dim = int(getattr(module, "attention_head_size", adapter.head_dim))
    scaling = float(getattr(module, "scaling", head_dim**-0.5))
    box = CapturedScores()

    def pre_hook(mod, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        if hidden is None:
            raise RuntimeError("attention module received no hidden states to capture from")
        shape = (*hidden.shape[:-1], -1, head_dim)
        with torch.no_grad():
            query = mod.query(hidden).view(*shape).transpose(1, 2)
            key = mod.key(hidden).view(*shape).transpose(1, 2)
            if head is not None:
                # Slice before matmul: a full [B,H,S,S] capture is unnecessary for one head.
                query = query[:, head : head + 1]
                key = key[:, head : head + 1]
            scores = torch.matmul(query, key.transpose(2, 3)) * scaling
            if mask is not None:
                scores = scores + mask
        box.scores = scores[:, 0] if head is not None else scores
        return None

    handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield box
    finally:
        handle.remove()
