"""Component-necessity intervention utilities for NTv3's conv/deconv U-Net.

Reframed from the NTv2/DNABERT phase03-08 notebooks' attention-head-level
ablation: NTv3's Transformer bottleneck collapses to a single token at the
70nt promoter length this study uses (``2**num_downsamples == 128 ==
config.PAD_LENGTH``), so head-level localization there is close to vacuous
-- see README.md's "Architecture, verified against the real checkpoint"
section. "Component" here instead means a conv/deconv tower stage, a
channel group within one, the Transformer bottleneck's per-token FFN, or a
skip connection -- the parts of NTv3 that actually do position-resolved
work on a 70nt input.

Ablation is **mean-ablation**: replace a component's output with its
per-channel mean computed over a reference cohort, not zero. Zero is an
off-distribution value to whatever consumes that activation, which would
conflate "component removed" with "component fed a value it never
produces in practice" -- mean-ablation removes the component's
sequence-specific signal while leaving downstream computation on a
realistic input.

Skip-connection ablation needs a second mechanism: the residual add
(``x = y + r``) in ``Core.forward`` happens in plain Python between two
module calls, not inside a hookable submodule, so it can't be intercepted
with a forward hook without also corrupting the conv tower's own forward
path (which reuses the same tensor as its next block's input). Instead,
``forward_with_ablated_skip`` is a parallel reimplementation of
``Core.forward`` verified line-for-line against the real
``modeling_ntv3_pretrained.py`` (see that comparison in this module's
tests / phase03 script), simplified by dropping the autocast contexts:
every ``*_compute_dtype`` field in the ``NTv3_8M_pre`` config is
``"float32"``, verified against the real config, so ``_autocast_to`` is a
no-op for this checkpoint. ``_assert_all_dtypes_float32`` checks that
assumption at runtime rather than silently reusing it for a checkpoint
where it might not hold.

``capture_activation``/``inject_activation`` support phase04's
reconstruct-inject experiments (a different causal question from
necessity: not "does removing this hurt" but "does transplanting a clean
run's activation into a corrupted run's forward pass restore the clean
readout"). The DNABERT-2/NTv2 original needs a nucleotide-span projection
matrix here because its BPE tokenizer can produce a different token count
for a sequence and its shuffled corruption. NTv3 doesn't: tokenization is
strictly one nucleotide per token (see ``ntv3/model.py``), and this study's
corruptions preserve length, so a clean run and its corrupted pair always
tokenize to the identical shape -- direct tensor substitution, no
projection needed.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class ChannelMeans:
    """Per-channel mean activation of one module's output, over a reference
    cohort. ``channel_dim`` is the tensor axis channels live on: 1 for conv
    modules (``B, C, L``), -1 / 2 for transformer ``fc1``/``fc2`` (``B, L, C``)."""

    mean: torch.Tensor  # shape (C,)
    channel_dim: int


def collect_channel_means(module: torch.nn.Module, forward_fn: Callable[[], None], *, channel_dim: int) -> ChannelMeans:
    """Run ``forward_fn`` (expected to trigger one or more forward passes
    over a reference cohort) with a hook capturing every call to
    ``module``'s output, and return the per-channel mean across all of
    them (batch and position dims collapsed together)."""

    captured: list[torch.Tensor] = []

    def _capture(_module, _inputs, output):
        captured.append(output.detach())

    handle = module.register_forward_hook(_capture)
    try:
        forward_fn()
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("forward_fn did not trigger the hooked module")
    stacked = torch.cat(captured, dim=0)
    reduce_dims = [d for d in range(stacked.dim()) if d != channel_dim % stacked.dim()]
    return ChannelMeans(mean=stacked.mean(dim=reduce_dims), channel_dim=channel_dim % stacked.dim())


@contextmanager
def mean_ablate(module: torch.nn.Module, means: ChannelMeans, *, channels: list[int] | None = None):
    """While active, ``module``'s forward output has the given channels
    (default: all) replaced by their reference-cohort mean, broadcast over
    every other dimension. Everything downstream of ``module`` then runs on
    the ablated tensor exactly as it would on a real activation."""

    def _hook(_module, _inputs, output):
        n_channels = output.shape[means.channel_dim]
        idx = channels if channels is not None else list(range(n_channels))
        shape = [1] * output.dim()
        shape[means.channel_dim] = n_channels
        mean_full = means.mean.view(*[1] * means.channel_dim, n_channels, *[1] * (output.dim() - means.channel_dim - 1))
        mean_full = mean_full.expand(output.shape)
        out = output.clone()
        sel = [slice(None)] * output.dim()
        sel[means.channel_dim] = idx
        out[tuple(sel)] = mean_full[tuple(sel)]
        return out

    handle = module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def capture_activation(module: torch.nn.Module, forward_fn: Callable[[], None]) -> torch.Tensor:
    """Run ``forward_fn`` once, capturing and returning ``module``'s output
    tensor (detached, cloned). ``forward_fn`` must trigger exactly one call
    to ``module`` -- typically a single batched forward pass."""

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        captured["value"] = output.detach().clone()

    handle = module.register_forward_hook(_hook)
    try:
        forward_fn()
    finally:
        handle.remove()

    if "value" not in captured:
        raise RuntimeError("forward_fn did not trigger the hooked module")
    return captured["value"]


@contextmanager
def inject_activation(module: torch.nn.Module, replacement: torch.Tensor):
    """While active, ``module``'s forward output is replaced wholesale with
    ``replacement`` (must match the shape the module would have produced).
    Reconstruct-inject: capture a donor run's activation with
    ``capture_activation``, then substitute it into a recipient run to test
    whether the donor's activation is *sufficient* to restore the donor's
    readout on the recipient's forward pass."""

    def _hook(_module, _inputs, output):
        if replacement.shape != output.shape:
            raise ValueError(
                f"injected replacement shape {tuple(replacement.shape)} != "
                f"module output shape {tuple(output.shape)}"
            )
        return replacement

    handle = module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def _content_span_at_full_resolution(batch, batch_index: int, seq_len: int) -> tuple[int, int]:
    """Map a sequence's real-nucleotide span from padded-input coordinates to
    a (B, C, L) module output's own resolution, for subspace interventions
    that need to touch only real content positions, not padding."""
    factor = batch.padded_length // seq_len
    start, end = batch.content_spans[batch_index]
    cs, ce = start // factor, -(-end // factor)
    return max(0, cs), min(seq_len, ce)


@contextmanager
def project_out_subspace(module: torch.nn.Module, batch, basis: torch.Tensor):
    """While active, ``module``'s (B, C, L) output has the row space of
    ``basis`` (shape ``[r, C]``, e.g. top-r right-singular vectors of a
    delta-vector SVD) projected out at every real-nucleotide position
    (``batch.content_spans``, mapped to this output's resolution -- padding
    positions are left untouched since they carry no sequence-specific
    signal to remove). Tests **necessity**: does removing this subspace's
    contribution reduce a downstream readout?"""

    projector = basis.T @ basis  # [C, C], projects onto the subspace's span

    def _hook(_module, _inputs, output):
        out = output.clone()
        seq_len = out.shape[2]
        proj = projector.to(out.dtype)
        for i in range(out.shape[0]):
            cs, ce = _content_span_at_full_resolution(batch, i, seq_len)
            if ce <= cs:
                continue
            seg = out[i, :, cs:ce]  # (C, seg_len)
            out[i, :, cs:ce] = seg - proj @ seg
        return out

    handle = module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def inject_subspace_direction(module: torch.nn.Module, batch, direction: torch.Tensor, scale: float):
    """While active, ``module``'s (B, C, L) output has ``scale * direction``
    (shape ``[C]``, e.g. a unit promoter direction) added at every
    real-nucleotide position. Tests **sufficiency**: does pushing activity
    along this direction increase a downstream readout?"""

    def _hook(_module, _inputs, output):
        out = output.clone()
        seq_len = out.shape[2]
        add = (scale * direction).to(out.dtype).view(-1, 1)  # (C, 1)
        for i in range(out.shape[0]):
            cs, ce = _content_span_at_full_resolution(batch, i, seq_len)
            if ce <= cs:
                continue
            out[i, :, cs:ce] = out[i, :, cs:ce] + add
        return out

    handle = module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def _assert_all_dtypes_float32(config) -> None:
    dtype_fields = [
        "embedding_compute_dtype", "stem_compute_dtype", "down_convolution_compute_dtype",
        "transformer_qkvo_compute_dtype", "transformer_ffn_compute_dtype",
        "up_convolution_compute_dtype", "lmhead_compute_dtype",
    ]
    bad = {f: getattr(config, f) for f in dtype_fields if getattr(config, f).lower() != "float32"}
    if bad:
        raise ValueError(
            f"forward_with_ablated_skip assumes every compute dtype is float32 "
            f"(so autocast is a no-op); found non-float32 fields: {bad}. "
            f"This checkpoint needs the autocast contexts reinstated."
        )


def forward_with_ablated_skip(
    core: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    ablate_conv_stage: int | None = None,
    replacement: torch.Tensor | None = None,
    output_hidden_states: bool = False,
) -> dict[str, torch.Tensor]:
    """Reimplementation of ``Core.forward`` (verified line-for-line against
    the real ``modeling_ntv3_pretrained.py``, autocast contexts dropped --
    see module docstring) that can replace one conv-tower stage's stored
    skip-connection residual with ``replacement`` before the deconv tower
    adds it back in. ``ablate_conv_stage`` is the 0-indexed conv-tower stage
    whose residual feeds the *matching* deconv stage (stage 0's residual is
    used by the *last* deconv block, mirroring ``Core.forward``'s
    ``residuals[-(i+1)]`` lookup). ``replacement`` must broadcast against
    that residual's shape (typically a per-channel mean from
    ``collect_channel_means`` on the source module, reshaped to
    ``(1, C, 1)``); ``ablate_conv_stage=None`` runs the unmodified forward.
    """

    config = core.config
    _assert_all_dtypes_float32(config)

    x = core.embed_layer(input_ids)
    x = core.stem(x.permute(0, 2, 1))  # (B, C, L)

    residuals: list[torch.Tensor] = []
    hidden_states: list[torch.Tensor] = []
    for i, block in enumerate(core.conv_tower_blocks):
        residuals.append(x)
        y = block.conv(x)
        y = block.res_conv(y)
        if output_hidden_states:
            hidden_states.append(y.permute(0, 2, 1))
        x = block.avg_pool(y)

    if ablate_conv_stage is not None:
        if replacement is None:
            raise ValueError("replacement is required when ablate_conv_stage is set")
        residuals[ablate_conv_stage] = replacement.expand_as(residuals[ablate_conv_stage])

    x = x.permute(0, 2, 1)  # (B, L, C)
    for layer in core.transformer_blocks:
        out = layer(x, attention_mask=None, attention_weight_bias=None)
        x = out["embeddings"]
        if output_hidden_states:
            hidden_states.append(x)

    x = x.permute(0, 2, 1)  # (B, C, L)
    for i, block in enumerate(core.deconv_tower_blocks):
        r = residuals[-(i + 1)]
        y = block(x)
        if config.use_skip_connection:
            y = y + r
        x = y
        if output_hidden_states:
            hidden_states.append(x.permute(0, 2, 1))

    y = x.permute(0, 2, 1)
    y = F.gelu(y, approximate="tanh")
    for hl in core.lm_head["hidden_layers"]:
        y = F.gelu(hl(y), approximate="tanh")
    y = y.to(core.lm_head["head"].weight.dtype)
    logits = core.lm_head["head"](y)

    result: dict[str, torch.Tensor] = {"logits": logits}
    if output_hidden_states:
        result["hidden_states"] = hidden_states
    return result
