"""Safe interventions: ablations that always restore, and three baselines for what to ablate to.

The scripts ablate by hand -- `model.blocks[l].attn.head_mask[h] = 0.0` then `= 1.0` after -- which
leaks a zeroed head into every later cell if an exception fires in between. Here every intervention is
a context manager that restores on exit, exception or not:

    with ablate_heads(m.model, [(1, 3)]):
        loss_ablated = ...              # L1H3 is off only inside this block
    # L1H3 is on again here, guaranteed

**What to ablate a head TO is a real methodological choice, not a detail.** Zero-ablation pushes the
residual stream off-distribution -- downstream components see an input they were never trained on, so
the measured effect mixes "this head mattered" with "this input is broken." The interpretability
literature prefers a baseline that stays on the activation manifold: resample ablation (patch the
head's output from a different, shuffled batch) and mean ablation (patch it to its dataset mean). Both
are provided; the repo previously had only zero. Denoising (start corrupted, restore the circuit) is
generally a stronger test than noising (start clean, break the circuit), because a corrupted baseline
lacks the compensatory structure a clean run has -- but that requires a paired clean/corrupt input and
is left to the assay layer, which knows the task.

The knob these wrap is `Attention.head_mask`, a per-head multiplier on `z` (the attention output,
pre-W_O) registered as a non-persistent buffer. Zeroing an entry removes that head's write to the
residual stream. resample/mean instead overwrite `z` via a forward hook, so the head still writes --
just a distribution-preserving value rather than its real one.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from gfmcd.models.gfm_nano import GFMNano

Head = tuple[int, int]  # (layer, head)


@contextmanager
def ablate_heads(model: GFMNano, heads: list[Head] | Head):
    """Zero-ablate one or more heads for the duration of the block, then restore.

        with ablate_heads(model, [(1, 3), (1, 2)]):
            ...

    Zero-ablation is the simplest baseline and the one the existing scripts use. It is off-distribution;
    prefer `resample_heads` / `mean_heads` when the downstream effect is what you care about.
    """
    if isinstance(heads, tuple):
        heads = [heads]
    saved = {(l, h): float(model.blocks[l].attn.head_mask[h]) for l, h in heads}
    try:
        for l, h in heads:
            model.blocks[l].attn.head_mask[h] = 0.0
        yield
    finally:
        for (l, h), v in saved.items():
            model.blocks[l].attn.head_mask[h] = v


@contextmanager
def ablate_all_but(model: GFMNano, keep: list[Head], layer: int | None = None):
    """Ablate every head EXCEPT `keep` (optionally only within `layer`). Restores on exit.

    Useful for the completeness question: does the circuit run on the kept heads alone?
    """
    keep_set = set(keep)
    heads = [
        (l, h)
        for l in range(model.cfg.n_layers)
        for h in range(model.cfg.n_heads)
        if (layer is None or l == layer) and (l, h) not in keep_set
    ]
    with ablate_heads(model, heads):
        yield


class _HeadPatch:
    """Overwrite specific heads' attention output `z` with supplied tensors, via a forward hook.

    Registered on the `Attention` module for a layer. `replacement[h]` is a tensor broadcastable to
    z[..., h, :] = [batch, pos, d_head]. Restores by removing the hook.
    """

    def __init__(self, model: GFMNano, layer: int, replacement: dict[int, torch.Tensor]):
        self.attn = model.blocks[layer].attn
        self.replacement = replacement
        self.handle = None

    def __enter__(self):
        attn = self.attn
        repl = self.replacement
        # Record whether the instance already had its own `forward` override, to restore exactly.
        self._had_own = "forward" in vars(attn)
        self._prev = vars(attn).get("forward")

        def patched(x, cache, layer):
            b, n, _ = x.shape
            h, dh = attn.cfg.n_heads, attn.cfg.d_head
            shape = lambda t: t.view(b, n, h, dh)
            q, k, v = shape(attn.W_Q(x)), shape(attn.W_K(x)), shape(attn.W_V(x))
            scores = torch.einsum("bphc,bqhc->bhpq", q, k) / dh**0.5
            if attn.cfg.causal:
                mask = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
                scores = scores.masked_fill(~mask, float("-inf"))
            cache[f"L{layer}.attn.scores"] = scores
            cache[f"L{layer}.attn.q"] = q
            pattern = scores.softmax(dim=-1)
            cache[f"L{layer}.attn.pattern"] = pattern
            z = torch.einsum("bhpq,bqhc->bphc", pattern, v)
            for head_i, rep in repl.items():
                z[..., head_i, :] = rep.to(z.dtype).to(z.device)
            cache[f"L{layer}.attn.z"] = z
            z = z * attn.head_mask.view(1, 1, -1, 1)
            return attn.W_O(z.reshape(b, n, -1))

        attn.forward = patched
        return self

    def __exit__(self, *exc):
        if self._had_own:
            self.attn.forward = self._prev
        else:
            # Remove the instance-level override so `forward` falls back to the class method.
            del self.attn.forward
        return False


@contextmanager
def resample_heads(model: GFMNano, heads: list[Head] | Head, source_tokens: torch.Tensor):
    """Resample-ablate heads: patch their z from a forward pass on `source_tokens` (shuffled).

    This keeps each head's output on the activation manifold -- the downstream components see a
    plausible value rather than a hole. The standard resample-ablation baseline.

        idx = torch.randperm(tokens.shape[0])
        with resample_heads(model, [(1,3)], tokens[idx]):
            ...
    """
    if isinstance(heads, tuple):
        heads = [heads]
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)

    # Capture z on the source batch for the needed layers.
    with torch.no_grad():
        _, src_cache = model.run_with_cache(source_tokens.to(next(model.parameters()).device))

    patches = []
    try:
        for l, hs in by_layer.items():
            src_z = src_cache[f"L{l}.attn.z"]  # [b, pos, head, d_head]
            repl = {h: src_z[..., h, :] for h in hs}
            p = _HeadPatch(model, l, repl).__enter__()
            patches.append(p)
        yield
    finally:
        for p in reversed(patches):
            p.__exit__()


@contextmanager
def mean_heads(model: GFMNano, heads: list[Head] | Head, ref_tokens: torch.Tensor):
    """Mean-ablate heads: patch their z to the per-position mean over `ref_tokens`.

    The dataset-mean baseline: removes the head's input-specific signal while preserving its average
    contribution to the residual stream.
    """
    if isinstance(heads, tuple):
        heads = [heads]
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)

    with torch.no_grad():
        _, ref_cache = model.run_with_cache(ref_tokens.to(next(model.parameters()).device))

    patches = []
    try:
        for l, hs in by_layer.items():
            z = ref_cache[f"L{l}.attn.z"]  # [b, pos, head, d_head]
            mean_z = z.mean(dim=0, keepdim=True)  # [1, pos, head, d_head]
            # [1, pos, d_head] broadcasts across the real batch when assigned in the patch.
            repl = {h: mean_z[..., h, :] for h in hs}
            p = _HeadPatch(model, l, repl).__enter__()
            patches.append(p)
        yield
    finally:
        for p in reversed(patches):
            p.__exit__()
