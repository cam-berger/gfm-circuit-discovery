"""Execution helpers for equal-dose promoter source-localization assays."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from gfmcd.adapters.dnabert2 import (
    Dnabert2PromoterBundle,
    Head,
    explicit_qkv_operation_effects_for_batch,
    explicit_token_mask_effects_for_batch,
    promoter_margin_tensor,
    tokenize_promoter_pairs,
)

from .promoter_source_localization import paired_fixed_dose_tile

Pair = tuple[str, str, str]


def scan_equal_dose_tiles(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    *,
    tile_width_nt: int,
    token_dose: int,
    maximum_cover_nt: int,
    maximum_cover_difference_nt: int,
    candidate_heads: Sequence[Head],
    include_all_layer_heads: bool,
    include_identity: bool,
    batch_pairs: int,
    tile_indices: Sequence[int] | None = None,
    operations_by_tile: Mapping[int, Sequence[str]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Measure mask and explicit regional V effects over all nominal tiles."""

    import torch

    if not pairs or 300 % tile_width_nt or batch_pairs < 1 or token_dose < 2:
        raise ValueError("invalid equal-dose tile scan request")
    initial = tokenize_promoter_pairs(bundle, pairs)
    output: list[dict[str, Any]] = []
    all_layer_heads = tuple(
        (candidate_heads[0][0], head)
        for head in range(int(bundle.config.num_attention_heads))
    )
    requested_tiles = (
        tuple(range(300 // tile_width_nt))
        if tile_indices is None
        else tuple(sorted({int(value) for value in tile_indices}))
    )
    if not requested_tiles or any(
        not 0 <= value < 300 // tile_width_nt for value in requested_tiles
    ):
        raise ValueError("requested tile index is outside the promoter input")
    for progress_index, tile_index in enumerate(requested_tiles, start=1):
        nominal_start = tile_index * tile_width_nt
        nominal_end = nominal_start + tile_width_nt
        eligible_indices = []
        for pair_index in range(initial.n_pairs):
            geometry = paired_fixed_dose_tile(
                initial.offset_mapping[
                    2 * pair_index, : initial.token_counts[2 * pair_index]
                ].tolist(),
                initial.offset_mapping[
                    2 * pair_index + 1, : initial.token_counts[2 * pair_index + 1]
                ].tolist(),
                nominal_start=nominal_start,
                nominal_end=nominal_end,
                token_dose=token_dose,
                maximum_cover_nt=maximum_cover_nt,
                maximum_cover_length_difference_nt=maximum_cover_difference_nt,
            )
            if geometry is not None:
                eligible_indices.append(pair_index)
        for chunk_start in range(0, len(eligible_indices), batch_pairs):
            chunk_indices = eligible_indices[chunk_start : chunk_start + batch_pairs]
            chunk = [pairs[index] for index in chunk_indices]
            batch = tokenize_promoter_pairs(bundle, chunk)
            token_indices = []
            geometries = []
            for pair_index in range(batch.n_pairs):
                geometry = paired_fixed_dose_tile(
                    batch.offset_mapping[
                        2 * pair_index, : batch.token_counts[2 * pair_index]
                    ].tolist(),
                    batch.offset_mapping[
                        2 * pair_index + 1, : batch.token_counts[2 * pair_index + 1]
                    ].tolist(),
                    nominal_start=nominal_start,
                    nominal_end=nominal_end,
                    token_dose=token_dose,
                    maximum_cover_nt=maximum_cover_nt,
                    maximum_cover_length_difference_nt=(maximum_cover_difference_nt),
                )
                if geometry is None:
                    raise RuntimeError(
                        "tile geometry changed after eligible subsetting"
                    )
                target_geometry, control_geometry = geometry
                token_indices.extend(
                    (
                        target_geometry.token_indices,
                        control_geometry.token_indices,
                    )
                )
                geometries.append(geometry)
            with torch.no_grad():
                native = promoter_margin_tensor(bundle, batch).detach()
            measurements = {
                **(
                    {
                        "mask": explicit_token_mask_effects_for_batch(
                            bundle,
                            batch,
                            token_indices,
                            native_margins=native,
                        )
                    }
                    if operations_by_tile is None
                    or "mask" in operations_by_tile.get(tile_index, ())
                    else {}
                ),
                **(
                    {
                        "candidate_value_swap": (
                            explicit_qkv_operation_effects_for_batch(
                                bundle,
                                batch,
                                candidate_heads,
                                token_indices,
                                operation="value_permute",
                                native_margins=native,
                            )
                        )
                    }
                    if operations_by_tile is None
                    or "candidate_value_swap" in operations_by_tile.get(tile_index, ())
                    else {}
                ),
            }
            if include_all_layer_heads:
                measurements["all_layer_value_swap"] = (
                    explicit_qkv_operation_effects_for_batch(
                        bundle,
                        batch,
                        all_layer_heads,
                        token_indices,
                        operation="value_permute",
                        native_margins=native,
                    )
                )
            if include_identity:
                measurements["candidate_identity"] = (
                    explicit_qkv_operation_effects_for_batch(
                        bundle,
                        batch,
                        candidate_heads,
                        token_indices,
                        operation="identity",
                        native_margins=native,
                    )
                )
            for local_pair_index, pair_id in enumerate(batch.pair_ids):
                target_geometry, control_geometry = geometries[local_pair_index]
                for operation, effects in measurements.items():
                    output.append(
                        {
                            "pair_id": pair_id,
                            "tile_index": tile_index,
                            "nominal_start": nominal_start,
                            "nominal_end": nominal_end,
                            "relative_start": nominal_start - 249,
                            "relative_end": nominal_end - 249,
                            "operation": operation,
                            "target_token_indices": list(target_geometry.token_indices),
                            "control_token_indices": list(
                                control_geometry.token_indices
                            ),
                            "target_cover_start": target_geometry.cover_start,
                            "target_cover_end": target_geometry.cover_end,
                            "control_cover_start": control_geometry.cover_start,
                            "control_cover_end": control_geometry.cover_end,
                            "target_loss": effects["target_loss"][local_pair_index],
                            "control_loss": effects["control_loss"][local_pair_index],
                            "target_specific_loss": effects["target_specific_loss"][
                                local_pair_index
                            ],
                        }
                    )
        if progress is not None:
            progress(
                f"tile {progress_index}/{len(requested_tiles)} "
                f"(index {tile_index}): "
                f"{len(eligible_indices)} eligible pairs"
            )
    return output


def scan_head_pair_regional_values(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    *,
    tile: Mapping[str, int],
    head_pairs: Mapping[str, Sequence[Head]],
    token_dose: int,
    maximum_cover_nt: int,
    maximum_cover_difference_nt: int,
    batch_pairs: int,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Compare one explicit regional V swap across matched head pairs."""

    import torch

    initial = tokenize_promoter_pairs(bundle, pairs)
    eligible_indices = []
    for pair_index in range(initial.n_pairs):
        geometry = paired_fixed_dose_tile(
            initial.offset_mapping[
                2 * pair_index, : initial.token_counts[2 * pair_index]
            ].tolist(),
            initial.offset_mapping[
                2 * pair_index + 1, : initial.token_counts[2 * pair_index + 1]
            ].tolist(),
            nominal_start=int(tile["nominal_start"]),
            nominal_end=int(tile["nominal_end"]),
            token_dose=token_dose,
            maximum_cover_nt=maximum_cover_nt,
            maximum_cover_length_difference_nt=maximum_cover_difference_nt,
        )
        if geometry is not None:
            eligible_indices.append(pair_index)
    output = []
    for chunk_start in range(0, len(eligible_indices), batch_pairs):
        chunk_indices = eligible_indices[chunk_start : chunk_start + batch_pairs]
        batch = tokenize_promoter_pairs(
            bundle, [pairs[index] for index in chunk_indices]
        )
        token_indices = []
        for pair_index in range(batch.n_pairs):
            geometry = paired_fixed_dose_tile(
                batch.offset_mapping[
                    2 * pair_index, : batch.token_counts[2 * pair_index]
                ].tolist(),
                batch.offset_mapping[
                    2 * pair_index + 1, : batch.token_counts[2 * pair_index + 1]
                ].tolist(),
                nominal_start=int(tile["nominal_start"]),
                nominal_end=int(tile["nominal_end"]),
                token_dose=token_dose,
                maximum_cover_nt=maximum_cover_nt,
                maximum_cover_length_difference_nt=(maximum_cover_difference_nt),
            )
            if geometry is None:
                raise RuntimeError("regional null geometry changed after subsetting")
            token_indices.extend((geometry[0].token_indices, geometry[1].token_indices))
        with torch.no_grad():
            native = promoter_margin_tensor(bundle, batch).detach()
        for pair_name, heads in head_pairs.items():
            effects = explicit_qkv_operation_effects_for_batch(
                bundle,
                batch,
                heads,
                token_indices,
                operation="value_permute",
                native_margins=native,
            )
            for pair_index, pair_id in enumerate(batch.pair_ids):
                output.append(
                    {
                        "pair_id": pair_id,
                        "tile_index": int(tile["tile_index"]),
                        "head_pair": pair_name,
                        "target_loss": effects["target_loss"][pair_index],
                        "control_loss": effects["control_loss"][pair_index],
                        "target_specific_loss": effects["target_specific_loss"][
                            pair_index
                        ],
                    }
                )
        if progress is not None:
            progress(
                f"tile {tile['tile_index']} regional-null batch "
                f"{chunk_start // batch_pairs + 1}"
            )
    return output


def score_motif_disruptions(
    bundle: Dnabert2PromoterBundle,
    rows: Sequence[Mapping[str, Any]],
    *,
    motifs: Sequence[str],
    nominated_tile_indices: Sequence[int],
    batch_pairs: int,
) -> list[dict[str, Any]]:
    """Score geometry-qualified motif disruptions overlapping nominated tiles."""

    import torch

    nominated = {int(value) for value in nominated_tile_indices}
    records = []
    for row in rows:
        for disruption in row["localization"]["motif_disruptions"]:
            if disruption["motif"] not in motifs:
                continue
            overlapping = [
                tile["tile_index"]
                for tile in row["localization"]["tiles"]
                if tile["tile_index"] in nominated
                and disruption["start"] < tile["nominal_end"]
                and tile["nominal_start"] < disruption["end"]
            ]
            if not overlapping:
                continue
            records.append((row, disruption, overlapping))
    output = []
    for start in range(0, len(records), batch_pairs):
        chunk = records[start : start + batch_pairs]
        native_pairs = [
            (
                f"{row['example_id']}|{disruption['motif']}",
                str(row["target"]["clean_sequence"]),
                str(row["control"]["clean_sequence"]),
            )
            for row, disruption, _ in chunk
        ]
        disrupted_pairs = [
            (
                f"{row['example_id']}|{disruption['motif']}",
                str(disruption["target_sequence"]),
                str(disruption["control_sequence"]),
            )
            for row, disruption, _ in chunk
        ]
        native_batch = tokenize_promoter_pairs(bundle, native_pairs)
        disrupted_batch = tokenize_promoter_pairs(bundle, disrupted_pairs)
        if not _same_token_geometry(native_batch, disrupted_batch):
            raise RuntimeError("motif disruption violated frozen BPE geometry")
        with torch.no_grad():
            native = promoter_margin_tensor(bundle, native_batch)
            disrupted = promoter_margin_tensor(bundle, disrupted_batch)
        target_loss = native[0::2] - disrupted[0::2]
        control_loss = native[1::2] - disrupted[1::2]
        difference = target_loss - control_loss
        for record_index, (row, disruption, overlapping) in enumerate(chunk):
            output.append(
                {
                    "pair_id": str(row["example_id"]),
                    "motif": str(disruption["motif"]),
                    "family": str(disruption["family"]),
                    "start": int(disruption["start"]),
                    "end": int(disruption["end"]),
                    "position0_relative_tss": int(disruption["position0_relative_tss"]),
                    "changed_nucleotides_each": int(
                        disruption["changed_nucleotides_each"]
                    ),
                    "overlapping_nominated_tiles": overlapping,
                    "target_loss": float(target_loss[record_index]),
                    "control_loss": float(control_loss[record_index]),
                    "target_specific_loss": float(difference[record_index]),
                }
            )
    return output


def _same_token_geometry(left: Any, right: Any) -> bool:
    if left.token_counts != right.token_counts:
        return False
    for sequence_index, count in enumerate(left.token_counts):
        if not (
            left.offset_mapping[sequence_index, :count]
            == right.offset_mapping[sequence_index, :count]
        ).all():
            return False
    return True
