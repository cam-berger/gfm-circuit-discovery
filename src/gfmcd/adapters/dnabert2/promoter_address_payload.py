"""CLS-query address/payload counterfactuals for MosaicBERT attention heads."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from .promoter_classifier import Dnabert2PromoterBundle
from .promoter_head_atlas import (
    Head,
    PairedPromoterBatch,
    promoter_margin_tensor,
    validate_promoter_heads,
)

QkvOperation = Literal[
    "identity",
    "key_permute",
    "value_permute",
    "key_value_joint_permute",
]


def counterfactual_cls_head(
    query: Any,
    keys: Any,
    values: Any,
    *,
    bias: Any,
    source_indices: Any,
    permuted_indices: Any,
    operation: QkvOperation,
) -> Any:
    """Recompute one CLS head after a shape- and multiset-preserving K/V surgery."""

    import torch

    if (
        query.ndim != 1
        or keys.ndim != 2
        or values.shape != keys.shape
        or keys.shape[1] != query.shape[0]
        or bias.shape != (keys.shape[0],)
    ):
        raise ValueError("malformed Q/K/V or CLS attention bias geometry")
    source = source_indices.to(keys.device)
    permuted = permuted_indices.to(keys.device)
    if (
        source.ndim != 1
        or source.numel() < 2
        or source.shape != permuted.shape
        or len({int(value) for value in source.cpu()}) != source.numel()
        or {int(value) for value in source.cpu()}
        != {int(value) for value in permuted.cpu()}
    ):
        raise ValueError(
            "source and permuted indices must declare one nontrivial permutation"
        )
    counterfactual_keys = keys
    counterfactual_values = values
    if operation in {"key_permute", "key_value_joint_permute"}:
        counterfactual_keys = keys.clone()
        counterfactual_keys[source] = keys[permuted]
    if operation in {"value_permute", "key_value_joint_permute"}:
        counterfactual_values = values.clone()
        counterfactual_values[source] = values[permuted]
    if operation not in {
        "identity",
        "key_permute",
        "value_permute",
        "key_value_joint_permute",
    }:
        raise ValueError(f"unknown QKV operation: {operation}")
    scores = (counterfactual_keys @ query) / math.sqrt(query.shape[0])
    probabilities = torch.softmax(scores + bias, dim=0)
    output = probabilities @ counterfactual_values
    if not torch.isfinite(output).all():
        raise RuntimeError("QKV counterfactual created a non-finite head context")
    return output


def paired_source_permutations(
    batch: PairedPromoterBatch,
    *,
    source_token_count: int,
    seed: int,
) -> tuple[tuple[Any, Any], ...]:
    """Choose the same fixed token dose and a derangement for every sequence."""

    import torch

    if source_token_count < 2:
        raise ValueError("source-token dose must be at least two")
    output = []
    for sequence_index, count in enumerate(batch.token_counts):
        offsets = batch.offset_mapping[sequence_index, :count].tolist()
        genomic = [
            index for index, (start, end) in enumerate(offsets) if int(end) > int(start)
        ]
        if len(genomic) < source_token_count:
            raise ValueError(
                f"sequence {sequence_index} has {len(genomic)} genomic tokens, "
                f"below fixed dose {source_token_count}"
            )
        pair_id = batch.pair_ids[sequence_index // 2]
        branch = "target" if sequence_index % 2 == 0 else "control"
        ranked = sorted(
            genomic,
            key=lambda index: hashlib.sha256(
                (
                    f"{seed}|source-subset|{pair_id}|{branch}|"
                    f"{offsets[index][0]}:{offsets[index][1]}|{index}"
                ).encode()
            ).hexdigest(),
        )[:source_token_count]
        source = sorted(ranked)
        shift_hash = hashlib.sha256(
            f"{seed}|source-shift|{pair_id}|{branch}".encode()
        ).digest()
        shift = 1 + int.from_bytes(shift_hash[:8], "big") % (source_token_count - 1)
        permuted = source[shift:] + source[:shift]
        if any(left == right for left, right in zip(source, permuted)):
            raise RuntimeError(
                "cyclic source permutation unexpectedly has a fixed point"
            )
        output.append(
            (
                torch.tensor(source, dtype=torch.long),
                torch.tensor(permuted, dtype=torch.long),
            )
        )
    return tuple(output)


def paired_cls_qkv_counterfactual_tensor(
    context: Any,
    qkv: Any,
    *,
    batch: PairedPromoterBatch,
    heads: Sequence[int],
    num_heads: int,
    alibi: Any,
    operation: QkvOperation,
    source_token_count: int,
    seed: int,
) -> Any:
    """Replace selected CLS head contexts with exact Q/K/V counterfactuals."""

    import torch

    if (
        context.ndim != 2
        or context.shape[1] % num_heads
        or qkv.ndim != 2
        or qkv.shape[0] != context.shape[0]
        or qkv.shape[1] != 3 * context.shape[1]
    ):
        raise ValueError("attention context and QKV tensors have incompatible geometry")
    head_size = context.shape[1] // num_heads
    split_context = context.view(context.shape[0], num_heads, head_size).clone()
    split_qkv = qkv.view(qkv.shape[0], 3, num_heads, head_size)
    permutations = paired_source_permutations(
        batch,
        source_token_count=source_token_count,
        seed=seed,
    )
    cursor = 0
    for sequence_index, count in enumerate(batch.token_counts):
        start, end = cursor, cursor + count
        source, permuted = permutations[sequence_index]
        for raw_head in heads:
            head = int(raw_head)
            if not 0 <= head < num_heads:
                raise ValueError(f"head index outside layer: {head}")
            sequence_qkv = split_qkv[start:end, :, head]
            bias = alibi[0, head, 0, :count].to(
                device=qkv.device,
                dtype=qkv.dtype,
            )
            split_context[start, head] = counterfactual_cls_head(
                sequence_qkv[0, 0],
                sequence_qkv[:, 1],
                sequence_qkv[:, 2],
                bias=bias,
                source_indices=source,
                permuted_indices=permuted,
                operation=operation,
            )
        cursor = end
    if cursor != context.shape[0] or not torch.isfinite(split_context).all():
        raise RuntimeError("QKV counterfactual failed the unpadded sequence map")
    return split_context.reshape_as(context)


@contextmanager
def patch_paired_cls_qkv(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    heads: Iterable[Head],
    *,
    operation: QkvOperation,
    source_token_count: int,
    seed: int,
):
    """Install paired CLS Q/K/V counterfactual hooks for selected heads."""

    selected = validate_promoter_heads(bundle, heads)
    by_layer: dict[int, list[int]] = {}
    for layer, head in selected:
        by_layer.setdefault(layer, []).append(head)
    qkv_by_layer: dict[int, Any] = {}
    handles = []
    num_heads = int(bundle.config.num_attention_heads)
    alibi = bundle.model.bert.encoder.alibi

    def make_capture_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            qkv_by_layer[layer] = output
            return output

        return hook

    def make_context_hook(layer: int, layer_heads: list[int]):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if layer not in qkv_by_layer:
                raise RuntimeError(f"QKV capture did not precede layer {layer} context")
            return paired_cls_qkv_counterfactual_tensor(
                output,
                qkv_by_layer[layer],
                batch=batch,
                heads=layer_heads,
                num_heads=num_heads,
                alibi=alibi,
                operation=operation,
                source_token_count=source_token_count,
                seed=seed,
            )

        return hook

    try:
        for layer, layer_heads in by_layer.items():
            block = bundle.model.bert.encoder.layer[layer].attention.self
            handles.append(block.Wqkv.register_forward_hook(make_capture_hook(layer)))
            handles.append(
                block.register_forward_hook(make_context_hook(layer, layer_heads))
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


def qkv_operation_effects_for_batch(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    heads: Iterable[Head],
    *,
    operation: QkvOperation,
    source_token_count: int,
    seed: int,
    native_margins: Any | None = None,
) -> dict[str, list[float]]:
    """Return target, control, and target-specific margin losses."""

    import torch

    with torch.no_grad():
        native = (
            promoter_margin_tensor(bundle, batch)
            if native_margins is None
            else native_margins.to(bundle.device)
        )
        with patch_paired_cls_qkv(
            bundle,
            batch,
            heads,
            operation=operation,
            source_token_count=source_token_count,
            seed=seed,
        ):
            patched = promoter_margin_tensor(bundle, batch)
    target_loss = native[0::2] - patched[0::2]
    control_loss = native[1::2] - patched[1::2]
    difference = target_loss - control_loss
    if not all(
        torch.isfinite(values).all()
        for values in (target_loss, control_loss, difference)
    ):
        raise RuntimeError("non-finite QKV operation effect")
    return {
        "target_loss": [float(value) for value in target_loss.cpu()],
        "control_loss": [float(value) for value in control_loss.cpu()],
        "target_specific_loss": [float(value) for value in difference.cpu()],
    }


def explicit_token_mask_effects_for_batch(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    token_indices_by_sequence: Sequence[Sequence[int]],
    *,
    native_margins: Any | None = None,
) -> dict[str, list[float]]:
    """Mask an explicit equal-dose BPE region without retokenizing the sequence."""

    import torch

    if len(token_indices_by_sequence) != 2 * batch.n_pairs:
        raise ValueError("one explicit token-index set is required per sequence")
    mask_id = bundle.tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("promoter tokenizer does not expose a mask token")
    encoded = {name: tensor.clone() for name, tensor in batch.encoded.items()}
    input_ids = encoded["input_ids"]
    doses = {len(tuple(indices)) for indices in token_indices_by_sequence}
    if len(doses) != 1 or next(iter(doses), 0) < 1:
        raise ValueError("explicit mask requires one nonzero equal token dose")
    for sequence_index, raw_indices in enumerate(token_indices_by_sequence):
        indices = tuple(int(value) for value in raw_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("explicit mask token indices must be unique")
        count = batch.token_counts[sequence_index]
        offsets = batch.offset_mapping[sequence_index, :count].tolist()
        for token_index in indices:
            if not 0 <= token_index < count or int(offsets[token_index][1]) <= int(
                offsets[token_index][0]
            ):
                raise ValueError("explicit mask index is not a genomic token")
            input_ids[sequence_index, token_index] = int(mask_id)
    masked_batch = PairedPromoterBatch(
        pair_ids=batch.pair_ids,
        encoded=encoded,
        offset_mapping=batch.offset_mapping,
        token_counts=batch.token_counts,
        cls_rows=batch.cls_rows,
    )
    with torch.no_grad():
        native = (
            promoter_margin_tensor(bundle, batch)
            if native_margins is None
            else native_margins.to(bundle.device)
        )
        masked = promoter_margin_tensor(bundle, masked_batch)
    return _paired_margin_losses(native, masked)


def explicit_qkv_operation_effects_for_batch(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    heads: Iterable[Head],
    token_indices_by_sequence: Sequence[Sequence[int]],
    *,
    operation: QkvOperation,
    native_margins: Any | None = None,
) -> dict[str, list[float]]:
    """Run a Q/K/V surgery on one explicit equal-dose region per sequence."""

    import torch

    with torch.no_grad():
        native = (
            promoter_margin_tensor(bundle, batch)
            if native_margins is None
            else native_margins.to(bundle.device)
        )
        with patch_paired_cls_qkv_explicit(
            bundle,
            batch,
            heads,
            token_indices_by_sequence=token_indices_by_sequence,
            operation=operation,
        ):
            patched = promoter_margin_tensor(bundle, batch)
    return _paired_margin_losses(native, patched)


@contextmanager
def patch_paired_cls_qkv_explicit(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    heads: Iterable[Head],
    *,
    token_indices_by_sequence: Sequence[Sequence[int]],
    operation: QkvOperation,
):
    """Install explicit-region CLS Q/K/V counterfactual hooks."""

    selected = validate_promoter_heads(bundle, heads)
    by_layer: dict[int, list[int]] = {}
    for layer, head in selected:
        by_layer.setdefault(layer, []).append(head)
    permutations = explicit_source_permutations(
        batch, token_indices_by_sequence=token_indices_by_sequence
    )
    qkv_by_layer: dict[int, Any] = {}
    handles = []
    num_heads = int(bundle.config.num_attention_heads)
    alibi = bundle.model.bert.encoder.alibi

    def make_capture_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            qkv_by_layer[layer] = output
            return output

        return hook

    def make_context_hook(layer: int, layer_heads: list[int]):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if layer not in qkv_by_layer:
                raise RuntimeError(f"QKV capture did not precede layer {layer} context")
            return paired_cls_qkv_explicit_counterfactual_tensor(
                output,
                qkv_by_layer[layer],
                batch=batch,
                heads=layer_heads,
                num_heads=num_heads,
                alibi=alibi,
                operation=operation,
                permutations=permutations,
            )

        return hook

    try:
        for layer, layer_heads in by_layer.items():
            block = bundle.model.bert.encoder.layer[layer].attention.self
            handles.append(block.Wqkv.register_forward_hook(make_capture_hook(layer)))
            handles.append(
                block.register_forward_hook(make_context_hook(layer, layer_heads))
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


def explicit_source_permutations(
    batch: PairedPromoterBatch,
    *,
    token_indices_by_sequence: Sequence[Sequence[int]],
) -> tuple[tuple[Any, Any], ...]:
    """Validate explicit genomic-token sets and return deterministic derangements."""

    import torch

    if len(token_indices_by_sequence) != len(batch.token_counts):
        raise ValueError("one explicit token-index set is required per sequence")
    doses = {len(tuple(indices)) for indices in token_indices_by_sequence}
    if len(doses) != 1 or next(iter(doses), 0) < 2:
        raise ValueError("explicit QKV surgery requires an equal dose of at least two")
    output = []
    for sequence_index, (count, raw_indices) in enumerate(
        zip(batch.token_counts, token_indices_by_sequence)
    ):
        source = tuple(int(value) for value in raw_indices)
        if len(set(source)) != len(source):
            raise ValueError("explicit QKV source indices must be unique")
        offsets = batch.offset_mapping[sequence_index, :count].tolist()
        if any(
            not 0 <= index < count or int(offsets[index][1]) <= int(offsets[index][0])
            for index in source
        ):
            raise ValueError("explicit QKV source index is not a genomic token")
        permuted = source[1:] + source[:1]
        if any(left == right for left, right in zip(source, permuted)):
            raise RuntimeError("explicit cyclic permutation has a fixed point")
        output.append(
            (
                torch.tensor(source, dtype=torch.long),
                torch.tensor(permuted, dtype=torch.long),
            )
        )
    return tuple(output)


def paired_cls_qkv_explicit_counterfactual_tensor(
    context: Any,
    qkv: Any,
    *,
    batch: PairedPromoterBatch,
    heads: Sequence[int],
    num_heads: int,
    alibi: Any,
    operation: QkvOperation,
    permutations: Sequence[tuple[Any, Any]],
) -> Any:
    """Replace selected CLS head contexts using explicit regional source tokens."""

    import torch

    if (
        context.ndim != 2
        or context.shape[1] % num_heads
        or qkv.ndim != 2
        or qkv.shape[0] != context.shape[0]
        or qkv.shape[1] != 3 * context.shape[1]
        or len(permutations) != len(batch.token_counts)
    ):
        raise ValueError("explicit attention context/QKV geometry is incompatible")
    head_size = context.shape[1] // num_heads
    split_context = context.view(context.shape[0], num_heads, head_size).clone()
    split_qkv = qkv.view(qkv.shape[0], 3, num_heads, head_size)
    cursor = 0
    for sequence_index, count in enumerate(batch.token_counts):
        start, end = cursor, cursor + count
        source, permuted = permutations[sequence_index]
        for raw_head in heads:
            head = int(raw_head)
            if not 0 <= head < num_heads:
                raise ValueError(f"head index outside layer: {head}")
            sequence_qkv = split_qkv[start:end, :, head]
            bias = alibi[0, head, 0, :count].to(
                device=qkv.device,
                dtype=qkv.dtype,
            )
            split_context[start, head] = counterfactual_cls_head(
                sequence_qkv[0, 0],
                sequence_qkv[:, 1],
                sequence_qkv[:, 2],
                bias=bias,
                source_indices=source,
                permuted_indices=permuted,
                operation=operation,
            )
        cursor = end
    if cursor != context.shape[0] or not torch.isfinite(split_context).all():
        raise RuntimeError("explicit QKV counterfactual failed the unpadded map")
    return split_context.reshape_as(context)


def _paired_margin_losses(native: Any, perturbed: Any) -> dict[str, list[float]]:
    import torch

    target_loss = native[0::2] - perturbed[0::2]
    control_loss = native[1::2] - perturbed[1::2]
    difference = target_loss - control_loss
    if not all(
        torch.isfinite(values).all()
        for values in (target_loss, control_loss, difference)
    ):
        raise RuntimeError("non-finite paired margin effect")
    return {
        "target_loss": [float(value) for value in target_loss.cpu()],
        "control_loss": [float(value) for value in control_loss.cpu()],
        "target_specific_loss": [float(value) for value in difference.cpu()],
    }


def head_residual_write_tensor(
    context: Any,
    output_weight: Any,
    *,
    head: int,
    num_heads: int,
) -> Any:
    """Map one context slice through its columns of the shared output projection."""

    import torch

    if (
        context.ndim != 2
        or context.shape[1] % num_heads
        or output_weight.ndim != 2
        or output_weight.shape[1] != context.shape[1]
    ):
        raise ValueError("head context and output weight geometry are incompatible")
    if not 0 <= int(head) < num_heads:
        raise ValueError(f"head index outside layer: {head}")
    head_size = context.shape[1] // num_heads
    start = int(head) * head_size
    stop = start + head_size
    write = torch.nn.functional.linear(
        context[:, start:stop],
        output_weight[:, start:stop],
        bias=None,
    )
    if not torch.isfinite(write).all():
        raise RuntimeError("head residual write is non-finite")
    return write


def capture_cls_residual_writes(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    heads: Iterable[Head],
) -> tuple[Any, dict[Head, Any]]:
    """Capture native head-specific residual writes at paired CLS rows."""

    import torch

    selected = validate_promoter_heads(bundle, heads)
    by_layer: dict[int, list[int]] = {}
    for layer, head in selected:
        by_layer.setdefault(layer, []).append(head)
    contexts: dict[int, Any] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            contexts[layer] = output.detach().clone()
            return output

        return hook

    try:
        for layer in by_layer:
            module = bundle.model.bert.encoder.layer[layer].attention.self
            handles.append(module.register_forward_hook(make_hook(layer)))
        with torch.no_grad():
            margins = promoter_margin_tensor(bundle, batch).detach()
    finally:
        for handle in handles:
            handle.remove()
    if set(contexts) != set(by_layer):
        raise RuntimeError("failed to capture requested attention contexts")
    rows = batch.cls_rows.to(bundle.device)
    num_heads = int(bundle.config.num_attention_heads)
    writes = {}
    for layer, layer_heads in by_layer.items():
        weight = bundle.model.bert.encoder.layer[
            layer
        ].attention.output.dense.weight.detach()
        cls_context = contexts[layer][rows]
        for head in layer_heads:
            writes[(layer, head)] = head_residual_write_tensor(
                cls_context,
                weight,
                head=head,
                num_heads=num_heads,
            ).detach()
    return margins, writes
