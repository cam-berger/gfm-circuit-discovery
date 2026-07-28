"""Head-level attribution and paired CLS activation patching for promoter DNABERT-2.

The fine-tuned classifier reads the pooled ``[CLS]`` state.  MosaicBERT emits each
attention head's context vector in an unpadded ``[n_valid_tokens, hidden]`` tensor
before the shared output projection.  This module keeps those architecture facts in
one place and exposes three deliberately separate measurements:

* first-order target↔control head attribution at ``[CLS]``;
* exact paired activation patches at the same head/position;
* native ``[CLS]`` attention source mass reconstructed from Q/K and ALiBi.

Attribution and attention are candidate generators.  Exact patches are interventions,
but a directed mechanism still requires downstream mediation and path tests.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from .promoter_classifier import Dnabert2PromoterBundle

Head = tuple[int, int]
PatchDirection = Literal[
    "target_from_control",
    "control_from_target",
    "target_from_self",
    "bidirectional_swap",
]
QueryPatchDirection = Literal["target_from_control", "target_from_self"]

PROMOTER_SOURCE_BANDS: Mapping[str, tuple[int, int]] = {
    "far_upstream": (-249, -50),
    "proximal_upstream": (-50, -2),
    "tss_anchor": (-2, 3),
    "proximal_downstream": (3, 13),
    "distal_downstream": (13, 51),
}


@dataclass
class PairedPromoterBatch:
    """Interleaved target/control classifier inputs and their unpadding map."""

    pair_ids: tuple[str, ...]
    encoded: dict[str, Any]
    offset_mapping: Any
    token_counts: tuple[int, ...]
    cls_rows: Any

    @property
    def n_pairs(self) -> int:
        return len(self.pair_ids)


def all_promoter_heads(bundle: Dnabert2PromoterBundle) -> tuple[Head, ...]:
    return tuple(
        (layer, head)
        for layer in range(int(bundle.config.num_hidden_layers))
        for head in range(int(bundle.config.num_attention_heads))
    )


def head_label(head: Head) -> str:
    return f"L{head[0]}H{head[1]}"


def tokenize_promoter_pairs(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[tuple[str, str, str]],
    *,
    maximum_tokens: int = 256,
) -> PairedPromoterBatch:
    """Tokenize ``(pair_id, target, control)`` rows in interleaved order."""

    import torch

    if not pairs:
        raise ValueError("promoter head atlas requires at least one pair")
    sequences: list[str] = []
    pair_ids: list[str] = []
    expected_nt = bundle.checkpoint.input_length_nt
    for pair_id, target, control in pairs:
        target = str(target).upper()
        control = str(control).upper()
        if (
            len(target) != expected_nt
            or len(control) != expected_nt
            or set(target + control) - set("ACGT")
        ):
            raise ValueError(f"invalid promoter pair sequence: {pair_id}")
        pair_ids.append(str(pair_id))
        sequences.extend((target, control))
    encoded = bundle.tokenizer(
        sequences,
        padding=True,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    token_counts = tuple(int(value) for value in encoded["attention_mask"].sum(dim=1))
    if max(token_counts) > maximum_tokens:
        raise ValueError(
            f"promoter pair requires {max(token_counts)} tokens, above {maximum_tokens}"
        )
    cls_rows = []
    cursor = 0
    for count in token_counts:
        cls_rows.append(cursor)
        cursor += count
    if cursor != int(encoded["attention_mask"].sum()):
        raise RuntimeError("failed to construct the MosaicBERT unpadding map")
    return PairedPromoterBatch(
        pair_ids=tuple(pair_ids),
        encoded=dict(encoded),
        offset_mapping=offsets,
        token_counts=token_counts,
        cls_rows=torch.tensor(cls_rows, dtype=torch.long),
    )


def promoter_margin_tensor(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
) -> Any:
    encoded = {name: tensor.to(bundle.device) for name, tensor in batch.encoded.items()}
    logits = bundle.model(**encoded).logits
    if logits.ndim != 2 or logits.shape != (2 * batch.n_pairs, 2):
        raise RuntimeError(f"malformed promoter logits: {tuple(logits.shape)}")
    return logits[:, 1] - logits[:, 0]


def validate_promoter_heads(
    bundle: Dnabert2PromoterBundle,
    heads: Iterable[Head],
) -> tuple[Head, ...]:
    selected = tuple(sorted({(int(layer), int(head)) for layer, head in heads}))
    layers = int(bundle.config.num_hidden_layers)
    n_heads = int(bundle.config.num_attention_heads)
    for layer, head in selected:
        if not 0 <= layer < layers or not 0 <= head < n_heads:
            raise ValueError(f"invalid promoter head: {(layer, head)}")
    return selected


def validate_promoter_layers(
    bundle: Dnabert2PromoterBundle,
    layers: Iterable[int],
) -> tuple[int, ...]:
    selected = tuple(sorted({int(layer) for layer in layers}))
    n_layers = int(bundle.config.num_hidden_layers)
    for layer in selected:
        if not 0 <= layer < n_layers:
            raise ValueError(f"invalid promoter layer: {layer}")
    return selected


def paired_cls_patch_tensor(
    context: Any,
    *,
    cls_rows: Any,
    heads: Sequence[int],
    num_heads: int,
    direction: PatchDirection,
) -> Any:
    """Return a context tensor with selected paired ``[CLS]`` head slices patched."""

    import torch

    if context.ndim != 2 or context.shape[1] % num_heads:
        raise ValueError("MosaicBERT attention context must be [valid_tokens, hidden]")
    rows = cls_rows.to(context.device)
    if rows.ndim != 1 or rows.numel() % 2:
        raise ValueError("CLS rows must contain interleaved target/control pairs")
    if int(rows.max()) >= context.shape[0]:
        raise ValueError("CLS row falls outside the unpadded attention context")
    split = context.view(context.shape[0], num_heads, -1).clone()
    target_rows, control_rows = rows[0::2], rows[1::2]
    original = split.clone()
    if direction == "target_from_control":
        destination, source = target_rows, control_rows
    elif direction == "control_from_target":
        destination, source = control_rows, target_rows
    elif direction == "target_from_self":
        destination = source = target_rows
    elif direction == "bidirectional_swap":
        for head in heads:
            if not 0 <= int(head) < num_heads:
                raise ValueError(f"head index outside layer: {head}")
            split[target_rows, int(head)] = original[control_rows, int(head)]
            split[control_rows, int(head)] = original[target_rows, int(head)]
        patched = split.reshape_as(context)
        if not torch.isfinite(patched).all():
            raise RuntimeError("head patch created a non-finite activation")
        return patched
    else:  # pragma: no cover - guarded by the Literal at typed call sites
        raise ValueError(f"unknown patch direction: {direction}")
    for head in heads:
        if not 0 <= int(head) < num_heads:
            raise ValueError(f"head index outside layer: {head}")
        split[destination, int(head)] = original[source, int(head)]
    patched = split.reshape_as(context)
    if not torch.isfinite(patched).all():
        raise RuntimeError("head patch created a non-finite activation")
    return patched


def paired_cls_write_patch_tensor(
    writes: Any,
    *,
    cls_rows: Any,
    direction: QueryPatchDirection,
) -> Any:
    """Patch paired ``[CLS]`` residual writes such as the MLP output."""

    import torch

    if writes.ndim != 2:
        raise ValueError("MosaicBERT residual write must be [valid_tokens, hidden]")
    rows = cls_rows.to(writes.device)
    if rows.ndim != 1 or rows.numel() % 2:
        raise ValueError("CLS rows must contain interleaved target/control pairs")
    if int(rows.max()) >= writes.shape[0]:
        raise ValueError("CLS row falls outside the unpadded residual write")
    target_rows, control_rows = rows[0::2], rows[1::2]
    patched = writes.clone()
    if direction == "target_from_control":
        patched[target_rows] = writes[control_rows]
    elif direction == "target_from_self":
        patched[target_rows] = writes[target_rows]
    else:  # pragma: no cover - guarded by the Literal at typed call sites
        raise ValueError(f"unknown write-patch direction: {direction}")
    if not torch.isfinite(patched).all():
        raise RuntimeError("residual-write patch created a non-finite activation")
    return patched


def paired_cls_query_patch_tensor(
    qkv: Any,
    *,
    cls_rows: Any,
    heads: Sequence[int],
    num_heads: int,
    direction: QueryPatchDirection,
) -> Any:
    """Patch only paired ``[CLS]`` query slices, leaving keys and values native."""

    import torch

    if qkv.ndim != 2 or qkv.shape[1] % (3 * num_heads):
        raise ValueError("MosaicBERT QKV must be [valid_tokens, 3 * hidden]")
    rows = cls_rows.to(qkv.device)
    if rows.ndim != 1 or rows.numel() % 2:
        raise ValueError("CLS rows must contain interleaved target/control pairs")
    if int(rows.max()) >= qkv.shape[0]:
        raise ValueError("CLS row falls outside the unpadded QKV tensor")
    split = qkv.view(qkv.shape[0], 3, num_heads, -1).clone()
    original = split.clone()
    target_rows, control_rows = rows[0::2], rows[1::2]
    if direction == "target_from_control":
        source_rows = control_rows
    elif direction == "target_from_self":
        source_rows = target_rows
    else:  # pragma: no cover - guarded by the Literal at typed call sites
        raise ValueError(f"unknown query-patch direction: {direction}")
    for head in heads:
        if not 0 <= int(head) < num_heads:
            raise ValueError(f"head index outside layer: {head}")
        split[target_rows, 0, int(head)] = original[source_rows, 0, int(head)]
    patched = split.reshape_as(qkv)
    if not torch.isfinite(patched).all():
        raise RuntimeError("query patch created a non-finite activation")
    return patched


def restore_cls_heads_from_reference(
    context: Any,
    *,
    reference: Any,
    cls_rows: Any,
    heads: Sequence[int],
    num_heads: int,
) -> Any:
    """Restore selected target ``[CLS]`` head contexts from a native cache."""

    import torch

    if context.shape != reference.shape or context.ndim != 2:
        raise ValueError("native reference must match the attention context tensor")
    rows = cls_rows.to(context.device)
    target_rows = rows[0::2]
    split = context.view(context.shape[0], num_heads, -1).clone()
    source = reference.to(context.device).view(context.shape[0], num_heads, -1)
    for head in heads:
        if not 0 <= int(head) < num_heads:
            raise ValueError(f"head index outside layer: {head}")
        split[target_rows, int(head)] = source[target_rows, int(head)]
    patched = split.reshape_as(context)
    if not torch.isfinite(patched).all():
        raise RuntimeError("cached-head restoration created a non-finite activation")
    return patched


@contextmanager
def patch_paired_cls_heads(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    heads: Iterable[Head],
    *,
    direction: PatchDirection,
):
    """Patch target/control ``[CLS]`` head contexts within each sequence pair."""

    selected = validate_promoter_heads(bundle, heads)
    by_layer: dict[int, list[int]] = {}
    for layer, head in selected:
        by_layer.setdefault(layer, []).append(head)
    handles = []
    num_heads = int(bundle.config.num_attention_heads)

    def make_hook(layer_heads: list[int]):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if not hasattr(output, "shape"):
                raise TypeError("MosaicBERT self-attention did not return a tensor")
            return paired_cls_patch_tensor(
                output,
                cls_rows=batch.cls_rows,
                heads=layer_heads,
                num_heads=num_heads,
                direction=direction,
            )

        return hook

    try:
        for layer, layer_heads in by_layer.items():
            module = bundle.model.bert.encoder.layer[layer].attention.self
            handles.append(module.register_forward_hook(make_hook(layer_heads)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_paired_cls_mlp_writes(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    layers: Iterable[int],
    *,
    direction: QueryPatchDirection,
):
    """Patch target ``[CLS]`` MLP writes from its paired control."""

    selected = validate_promoter_layers(bundle, layers)
    handles = []

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if not hasattr(output, "shape"):
            raise TypeError("MosaicBERT MLP output projection did not return a tensor")
        return paired_cls_write_patch_tensor(
            output,
            cls_rows=batch.cls_rows,
            direction=direction,
        )

    try:
        for layer in selected:
            module = bundle.model.bert.encoder.layer[layer].mlp.wo
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_paired_cls_queries(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    heads: Iterable[Head],
    *,
    direction: QueryPatchDirection,
):
    """Patch only the target ``[CLS]`` query projection for selected heads."""

    selected = validate_promoter_heads(bundle, heads)
    by_layer: dict[int, list[int]] = {}
    for layer, head in selected:
        by_layer.setdefault(layer, []).append(head)
    handles = []
    num_heads = int(bundle.config.num_attention_heads)

    def make_hook(layer_heads: list[int]):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if not hasattr(output, "shape"):
                raise TypeError("MosaicBERT QKV projection did not return a tensor")
            return paired_cls_query_patch_tensor(
                output,
                cls_rows=batch.cls_rows,
                heads=layer_heads,
                num_heads=num_heads,
                direction=direction,
            )

        return hook

    try:
        for layer, layer_heads in by_layer.items():
            module = bundle.model.bert.encoder.layer[layer].attention.self.Wqkv
            handles.append(module.register_forward_hook(make_hook(layer_heads)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def scrub_paired_cls_complement(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    *,
    retained_heads: Iterable[Head],
    retained_mlp_layers: Iterable[int],
):
    """Scrub every target CLS write except the explicitly retained node set."""

    all_heads = set(all_promoter_heads(bundle))
    retained_head_set = set(validate_promoter_heads(bundle, retained_heads))
    all_layers = set(range(int(bundle.config.num_hidden_layers)))
    retained_layer_set = set(validate_promoter_layers(bundle, retained_mlp_layers))
    with ExitStack() as stack:
        stack.enter_context(
            patch_paired_cls_heads(
                bundle,
                batch,
                sorted(all_heads - retained_head_set),
                direction="target_from_control",
            )
        )
        stack.enter_context(
            patch_paired_cls_mlp_writes(
                bundle,
                batch,
                sorted(all_layers - retained_layer_set),
                direction="target_from_control",
            )
        )
        yield


def capture_native_attention_contexts(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
) -> tuple[Any, dict[int, Any]]:
    """Run natively and cache every pre-output-projection attention context."""

    import torch

    captured: dict[int, Any] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            captured[layer] = output.detach().clone()
            return output

        return hook

    try:
        for layer, block in enumerate(bundle.model.bert.encoder.layer):
            handles.append(block.attention.self.register_forward_hook(make_hook(layer)))
        with torch.no_grad():
            margins = promoter_margin_tensor(bundle, batch).detach()
    finally:
        for handle in handles:
            handle.remove()
    expected = set(range(int(bundle.config.num_hidden_layers)))
    if set(captured) != expected:
        raise RuntimeError("failed to cache every native attention context")
    return margins, captured


@contextmanager
def corrupt_and_restore_paired_cls_heads(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    *,
    corrupt_heads: Iterable[Head],
    restore_heads: Iterable[Head],
    native_contexts: Mapping[int, Any],
):
    """Corrupt upstream heads and restore downstream heads from a native run."""

    corrupted = validate_promoter_heads(bundle, corrupt_heads)
    restored = validate_promoter_heads(bundle, restore_heads)
    if set(corrupted) & set(restored):
        raise ValueError("corrupt and restore head sets must be disjoint")
    missing = sorted({layer for layer, _ in restored} - set(native_contexts))
    if missing:
        raise ValueError(f"native context cache is missing layers: {missing}")
    by_layer: dict[int, list[int]] = {}
    for layer, head in restored:
        by_layer.setdefault(layer, []).append(head)
    handles = []
    num_heads = int(bundle.config.num_attention_heads)

    def make_restore_hook(layer: int, layer_heads: list[int]):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            return restore_cls_heads_from_reference(
                output,
                reference=native_contexts[layer],
                cls_rows=batch.cls_rows,
                heads=layer_heads,
                num_heads=num_heads,
            )

        return hook

    try:
        with patch_paired_cls_heads(
            bundle,
            batch,
            corrupted,
            direction="target_from_control",
        ):
            for layer, layer_heads in by_layer.items():
                module = bundle.model.bert.encoder.layer[layer].attention.self
                handles.append(
                    module.register_forward_hook(make_restore_hook(layer, layer_heads))
                )
            yield
    finally:
        for handle in handles:
            handle.remove()


def patch_effect_for_batch(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    heads: Iterable[Head],
    *,
    direction: PatchDirection,
    native_margins: Any | None = None,
) -> list[float]:
    """Measure exact per-pair margin loss/gain for one paired CLS patch."""

    import torch

    with torch.no_grad():
        native = (
            promoter_margin_tensor(bundle, batch)
            if native_margins is None
            else native_margins.to(bundle.device)
        )
        with patch_paired_cls_heads(bundle, batch, heads, direction=direction):
            patched = promoter_margin_tensor(bundle, batch)
    if direction in {"target_from_control", "target_from_self"}:
        effects = native[0::2] - patched[0::2]
    elif direction == "control_from_target":
        effects = patched[1::2] - native[1::2]
    else:
        raise ValueError(
            "use bidirectional_patch_effects_for_batch for a bidirectional swap"
        )
    if not torch.isfinite(effects).all():
        raise RuntimeError("non-finite paired head-patch effect")
    return [float(value) for value in effects.cpu()]


def bidirectional_patch_effects_for_batch(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    head: Head,
    *,
    native_margins: Any | None = None,
) -> dict[str, list[float]]:
    """Measure both single-head patch directions in one independent-sequence forward."""

    import torch

    validate_promoter_heads(bundle, [head])
    with torch.no_grad():
        native = (
            promoter_margin_tensor(bundle, batch)
            if native_margins is None
            else native_margins.to(bundle.device)
        )
        with patch_paired_cls_heads(
            bundle, batch, [head], direction="bidirectional_swap"
        ):
            patched = promoter_margin_tensor(bundle, batch)
    necessity = native[0::2] - patched[0::2]
    sufficiency = patched[1::2] - native[1::2]
    if not torch.isfinite(necessity).all() or not torch.isfinite(sufficiency).all():
        raise RuntimeError("non-finite bidirectional head-patch effect")
    return {
        "necessity": [float(value) for value in necessity.cpu()],
        "sufficiency": [float(value) for value in sufficiency.cpu()],
    }


def paired_head_attributions(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
) -> dict[Head, dict[str, list[float]]]:
    """First-order necessity and sufficiency attribution for every CLS head."""

    import torch

    activations: dict[int, Any] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            output.requires_grad_(True)
            output.retain_grad()
            activations[layer] = output
            return output

        return hook

    try:
        for layer, block in enumerate(bundle.model.bert.encoder.layer):
            handles.append(block.attention.self.register_forward_hook(make_hook(layer)))
        bundle.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            margins = promoter_margin_tensor(bundle, batch)
            margins.sum().backward()
        expected_layers = int(bundle.config.num_hidden_layers)
        if set(activations) != set(range(expected_layers)):
            raise RuntimeError("failed to capture every promoter attention layer")
        rows = batch.cls_rows.to(bundle.device)
        target_rows, control_rows = rows[0::2], rows[1::2]
        num_heads = int(bundle.config.num_attention_heads)
        output: dict[Head, dict[str, list[float]]] = {}
        for layer in range(expected_layers):
            activation = activations[layer]
            if activation.grad is None:
                raise RuntimeError(f"no classifier gradient reached layer {layer}")
            split = activation.view(activation.shape[0], num_heads, -1)
            gradient = activation.grad.view(activation.shape[0], num_heads, -1)
            difference = split[target_rows] - split[control_rows]
            necessity = (difference * gradient[target_rows]).sum(dim=2)
            sufficiency = (difference * gradient[control_rows]).sum(dim=2)
            if (
                not torch.isfinite(necessity).all()
                or not torch.isfinite(sufficiency).all()
            ):
                raise RuntimeError("non-finite promoter head attribution")
            for head in range(num_heads):
                output[(layer, head)] = {
                    "necessity": [
                        float(value) for value in necessity[:, head].detach().cpu()
                    ],
                    "sufficiency": [
                        float(value) for value in sufficiency[:, head].detach().cpu()
                    ],
                }
        return output
    finally:
        for handle in handles:
            handle.remove()
        bundle.model.zero_grad(set_to_none=True)


def capture_cls_attention_bands(
    bundle: Dnabert2PromoterBundle,
    batch: PairedPromoterBatch,
    *,
    tss_index: int = 249,
    bands: Mapping[str, tuple[int, int]] = PROMOTER_SOURCE_BANDS,
) -> dict[Head, dict[str, list[float]]]:
    """Reconstruct native CLS attention and aggregate it over nucleotide bands."""

    import torch

    qkv_by_layer: dict[int, Any] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            qkv_by_layer[layer] = output.detach()
            return output

        return hook

    try:
        for layer, block in enumerate(bundle.model.bert.encoder.layer):
            handles.append(
                block.attention.self.Wqkv.register_forward_hook(make_hook(layer))
            )
        with torch.no_grad():
            promoter_margin_tensor(bundle, batch)
    finally:
        for handle in handles:
            handle.remove()
    expected_layers = int(bundle.config.num_hidden_layers)
    if set(qkv_by_layer) != set(range(expected_layers)):
        raise RuntimeError("failed to capture every QKV projection")
    num_heads = int(bundle.config.num_attention_heads)
    head_size = int(bundle.config.hidden_size) // num_heads
    output: dict[Head, dict[str, list[float]]] = {
        (layer, head): {name: [] for name in ("special", *bands)}
        for layer in range(expected_layers)
        for head in range(num_heads)
    }
    offsets = batch.offset_mapping.cpu()
    cursor_by_sequence = []
    cursor = 0
    for count in batch.token_counts:
        cursor_by_sequence.append((cursor, cursor + count))
        cursor += count
    alibi = bundle.model.bert.encoder.alibi
    for layer in range(expected_layers):
        qkv_all = qkv_by_layer[layer]
        for sequence_index, ((start, end), count) in enumerate(
            zip(cursor_by_sequence, batch.token_counts)
        ):
            qkv = qkv_all[start:end].view(count, 3, num_heads, head_size)
            query = qkv[0, 0]
            keys = qkv[:, 1]
            scores = torch.einsum("hd,shd->hs", query, keys) / math.sqrt(head_size)
            scores = scores + alibi[0, :, 0, :count].to(
                device=scores.device, dtype=scores.dtype
            )
            probabilities = torch.softmax(scores, dim=1)
            allocations = _token_band_allocations(
                offsets[sequence_index, :count].tolist(),
                tss_index=tss_index,
                bands=bands,
            ).to(probabilities.device)
            masses = probabilities @ allocations
            if not torch.allclose(
                masses.sum(dim=1),
                torch.ones(num_heads, device=masses.device),
                atol=1e-5,
                rtol=0,
            ):
                raise RuntimeError("CLS attention band mass does not sum to one")
            names = ("special", *bands)
            for head in range(num_heads):
                for band_index, name in enumerate(names):
                    output[(layer, head)][name].append(
                        float(masses[head, band_index].cpu())
                    )
    return output


def _token_band_allocations(
    offsets: Sequence[Sequence[int]],
    *,
    tss_index: int,
    bands: Mapping[str, tuple[int, int]],
) -> Any:
    """Allocate each token across source bands by nucleotide overlap."""

    import torch

    names = tuple(bands)
    allocations = torch.zeros((len(offsets), 1 + len(names)), dtype=torch.float32)
    for token_index, raw_offset in enumerate(offsets):
        start, end = (int(raw_offset[0]), int(raw_offset[1]))
        if end <= start:
            allocations[token_index, 0] = 1.0
            continue
        relative_start, relative_end = start - tss_index, end - tss_index
        length = relative_end - relative_start
        assigned = 0.0
        for band_index, name in enumerate(names, start=1):
            lower, upper = bands[name]
            overlap = max(0, min(relative_end, upper) - max(relative_start, lower))
            allocations[token_index, band_index] = overlap / length
            assigned += overlap / length
        if not math.isclose(assigned, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"token offset {(start, end)} falls outside declared promoter bands"
            )
    return allocations
