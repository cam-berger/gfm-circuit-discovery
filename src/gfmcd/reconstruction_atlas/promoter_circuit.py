"""Gated causal interventions for the fine-tuned promoter CLS circuit."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from gfmcd.adapters.dnabert2 import (
    Dnabert2PromoterBundle,
    Head,
    capture_native_attention_contexts,
    corrupt_and_restore_paired_cls_heads,
    patch_paired_cls_heads,
    patch_paired_cls_queries,
    promoter_margin_tensor,
    scrub_paired_cls_complement,
    tokenize_promoter_pairs,
)

Pair = tuple[str, str, str]


def layer_pair_nulls(
    candidate_pair: Sequence[Head],
    *,
    num_heads: int,
) -> tuple[tuple[Head, Head], ...]:
    """Enumerate every candidate-excluding two-head pair in one layer."""

    candidate = tuple(sorted(set(candidate_pair)))
    if len(candidate) != 2 or candidate[0][0] != candidate[1][0]:
        raise ValueError("candidate pair must contain two distinct heads in one layer")
    layer = candidate[0][0]
    available = [
        (layer, head) for head in range(num_heads) if (layer, head) not in candidate
    ]
    return tuple(itertools.combinations(available, 2))


def strict_faithfulness_rows(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    conditions: Mapping[str, tuple[Sequence[Head], Sequence[int]]],
    *,
    batch_pairs: int,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, float | str]]:
    """Complement-scrub CLS head/MLP writes for each retained-node condition."""

    import torch

    _validate_batch_size(batch_pairs)
    output: list[dict[str, float | str]] = []
    for batch_index, chunk in enumerate(_chunks(pairs, batch_pairs), start=1):
        batch = tokenize_promoter_pairs(bundle, chunk)
        with torch.no_grad():
            native = promoter_margin_tensor(bundle, batch).detach()
        patched_by_condition: dict[str, Any] = {}
        for name, (heads, mlp_layers) in conditions.items():
            with (
                scrub_paired_cls_complement(
                    bundle,
                    batch,
                    retained_heads=heads,
                    retained_mlp_layers=mlp_layers,
                ),
                torch.no_grad(),
            ):
                patched_by_condition[name] = promoter_margin_tensor(
                    bundle, batch
                ).detach()
        for pair_index, pair_id in enumerate(batch.pair_ids):
            target = float(native[2 * pair_index])
            control = float(native[2 * pair_index + 1])
            for name, margins in patched_by_condition.items():
                patched = float(margins[2 * pair_index])
                output.append(
                    {
                        "pair_id": pair_id,
                        "condition": name,
                        "native_target_margin": target,
                        "native_control_margin": control,
                        "patched_target_margin": patched,
                        "clean_enrichment": target - control,
                        "retained_enrichment": patched - control,
                    }
                )
        if progress:
            progress(f"strict-faithfulness batch {batch_index}")
    return output


def exact_pair_patch_rows(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    head_pairs: Mapping[str, Sequence[Head]],
    *,
    batch_pairs: int,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, float | str]]:
    """Measure bidirectional exact patches for candidate and null head pairs."""

    import torch

    _validate_batch_size(batch_pairs)
    output: list[dict[str, float | str]] = []
    for batch_index, chunk in enumerate(_chunks(pairs, batch_pairs), start=1):
        batch = tokenize_promoter_pairs(bundle, chunk)
        with torch.no_grad():
            native = promoter_margin_tensor(bundle, batch).detach()
        for name, heads in head_pairs.items():
            with (
                patch_paired_cls_heads(
                    bundle,
                    batch,
                    heads,
                    direction="bidirectional_swap",
                ),
                torch.no_grad(),
            ):
                patched = promoter_margin_tensor(bundle, batch).detach()
            necessity = native[0::2] - patched[0::2]
            sufficiency = patched[1::2] - native[1::2]
            if (
                not torch.isfinite(necessity).all()
                or not torch.isfinite(sufficiency).all()
            ):
                raise RuntimeError("non-finite exact pair-patch effect")
            for pair_index, pair_id in enumerate(batch.pair_ids):
                output.append(
                    {
                        "pair_id": pair_id,
                        "head_pair": name,
                        "necessity": float(necessity[pair_index]),
                        "sufficiency": float(sufficiency[pair_index]),
                    }
                )
        if progress:
            progress(f"exact-pair batch {batch_index}")
    return output


def query_patch_rows(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    head_pairs: Mapping[str, Sequence[Head]],
    *,
    batch_pairs: int,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, float | str]]:
    """Measure target-margin loss after paired-control CLS-query transfer."""

    import torch

    _validate_batch_size(batch_pairs)
    output: list[dict[str, float | str]] = []
    for batch_index, chunk in enumerate(_chunks(pairs, batch_pairs), start=1):
        batch = tokenize_promoter_pairs(bundle, chunk)
        with torch.no_grad():
            native = promoter_margin_tensor(bundle, batch).detach()
        for name, heads in head_pairs.items():
            with (
                patch_paired_cls_queries(
                    bundle,
                    batch,
                    heads,
                    direction="target_from_control",
                ),
                torch.no_grad(),
            ):
                patched = promoter_margin_tensor(bundle, batch).detach()
            effects = native[0::2] - patched[0::2]
            if not torch.isfinite(effects).all():
                raise RuntimeError("non-finite CLS-query patch effect")
            for pair_index, pair_id in enumerate(batch.pair_ids):
                output.append(
                    {
                        "pair_id": pair_id,
                        "head_pair": name,
                        "query_necessity": float(effects[pair_index]),
                    }
                )
        if progress:
            progress(f"query-patch batch {batch_index}")
    return output


def path_rescue_rows(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    *,
    corrupt_heads: Sequence[Head],
    restore_pairs: Mapping[str, Sequence[Head]],
    batch_pairs: int,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, float | str]]:
    """Corrupt upstream candidate heads, then restore proposed downstream mediators."""

    import torch

    _validate_batch_size(batch_pairs)
    output: list[dict[str, float | str]] = []
    for batch_index, chunk in enumerate(_chunks(pairs, batch_pairs), start=1):
        batch = tokenize_promoter_pairs(bundle, chunk)
        native, contexts = capture_native_attention_contexts(bundle, batch)
        with (
            patch_paired_cls_heads(
                bundle,
                batch,
                corrupt_heads,
                direction="target_from_control",
            ),
            torch.no_grad(),
        ):
            corrupted = promoter_margin_tensor(bundle, batch).detach()
        for name, restore_heads in restore_pairs.items():
            with (
                corrupt_and_restore_paired_cls_heads(
                    bundle,
                    batch,
                    corrupt_heads=corrupt_heads,
                    restore_heads=restore_heads,
                    native_contexts=contexts,
                ),
                torch.no_grad(),
            ):
                restored = promoter_margin_tensor(bundle, batch).detach()
            source_loss = native[0::2] - corrupted[0::2]
            rescue = restored[0::2] - corrupted[0::2]
            if (
                not torch.isfinite(source_loss).all()
                or not torch.isfinite(rescue).all()
            ):
                raise RuntimeError("non-finite promoter path-rescue effect")
            for pair_index, pair_id in enumerate(batch.pair_ids):
                output.append(
                    {
                        "pair_id": pair_id,
                        "restore_pair": name,
                        "source_loss": float(source_loss[pair_index]),
                        "rescue": float(rescue[pair_index]),
                    }
                )
        if progress:
            progress(f"path-rescue batch {batch_index}")
    return output


def no_op_diagnostics(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
) -> dict[str, float | int]:
    """Exercise all hook families with exact native/self/no-op interventions."""

    import torch

    batch = tokenize_promoter_pairs(bundle, pairs)
    native, contexts = capture_native_attention_contexts(bundle, batch)
    all_heads = [
        (layer, head)
        for layer in range(int(bundle.config.num_hidden_layers))
        for head in range(int(bundle.config.num_attention_heads))
    ]
    all_layers = list(range(int(bundle.config.num_hidden_layers)))
    with (
        scrub_paired_cls_complement(
            bundle,
            batch,
            retained_heads=all_heads,
            retained_mlp_layers=all_layers,
        ),
        torch.no_grad(),
    ):
        empty_complement = promoter_margin_tensor(bundle, batch).detach()
    with (
        patch_paired_cls_heads(
            bundle,
            batch,
            all_heads,
            direction="target_from_self",
        ),
        torch.no_grad(),
    ):
        self_heads = promoter_margin_tensor(bundle, batch).detach()
    with (
        patch_paired_cls_queries(
            bundle,
            batch,
            all_heads,
            direction="target_from_self",
        ),
        torch.no_grad(),
    ):
        self_queries = promoter_margin_tensor(bundle, batch).detach()
    with (
        corrupt_and_restore_paired_cls_heads(
            bundle,
            batch,
            corrupt_heads=[],
            restore_heads=all_heads,
            native_contexts=contexts,
        ),
        torch.no_grad(),
    ):
        restored = promoter_margin_tensor(bundle, batch).detach()
    return {
        "captured_attention_layers": len(contexts),
        "empty_complement_max_abs_margin_difference": float(
            (native - empty_complement).abs().max()
        ),
        "head_self_patch_max_abs_margin_difference": float(
            (native - self_heads).abs().max()
        ),
        "query_self_patch_max_abs_margin_difference": float(
            (native - self_queries).abs().max()
        ),
        "native_restore_max_abs_margin_difference": float(
            (native - restored).abs().max()
        ),
    }


def _validate_batch_size(batch_pairs: int) -> None:
    if batch_pairs < 1:
        raise ValueError("batch_pairs must be positive")


def _chunks(values: Sequence[Pair], size: int) -> list[Sequence[Pair]]:
    _validate_batch_size(size)
    return [values[start : start + size] for start in range(0, len(values), size)]
