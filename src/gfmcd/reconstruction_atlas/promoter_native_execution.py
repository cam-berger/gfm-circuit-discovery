"""Model-bearing execution helpers for promoter native-source edits."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from gfmcd.adapters.dnabert2 import (
    Dnabert2PromoterBundle,
    promoter_margin_tensor,
    tokenize_promoter_pairs,
)

from .promoter_native_source import geometry_qualified_native_edit_ensemble

Pair = tuple[str, str, str]


def build_geometry_qualified_gue_rows(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    *,
    config: Mapping[str, Any],
    maximum: int,
) -> list[dict[str, Any]]:
    """Select outcome-blind GUE pairs that support every native edit family."""

    if maximum < 1:
        raise ValueError("maximum qualified GUE pairs must be positive")
    start, end = (
        int(value)
        for value in config["scope"]["candidate_source_region_sequence_indices"]
    )
    policy = config["native_interventions"]
    required = set(config["measurement"]["primary_operations"])
    rows = []
    for pair_id, target, control in pairs:
        edit_ensemble = geometry_qualified_native_edit_ensemble(
            bundle.tokenizer,
            target,
            control,
            start=start,
            end=end,
            gc_depletion_dose=int(policy["gc_depletion_dose"]),
            policy=policy,
            seed=_row_seed(int(config["inference"]["seed"]), pair_id),
            replicates=int(policy.get("edit_replicates_per_locus_and_family", 1)),
            operations=tuple(config["measurement"]["primary_operations"]),
        )
        if set(edit_ensemble) != required:
            continue
        rows.append(
            {
                "example_id": pair_id,
                "homology_cluster": pair_id,
                "target": {"clean_sequence": target},
                "control": {"clean_sequence": control},
                "native_source": {
                    "region": {
                        "start": start,
                        "end": end,
                        "relative_start": start - 249,
                        "relative_end": end - 249,
                    },
                    "edit_ensemble": edit_ensemble,
                },
            }
        )
        if len(rows) == maximum:
            break
    return rows


def score_native_edit_rows(
    bundle: Dnabert2PromoterBundle,
    rows: Sequence[Mapping[str, Any]],
    *,
    operations: Sequence[str],
    batch_pairs: int,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Score clean and native-edited margins for each operation."""

    import torch

    if not rows or batch_pairs < 1 or not operations:
        raise ValueError("invalid native-edit scoring request")
    output = []
    for operation in operations:
        replicates = {
            len(row["native_source"]["edit_ensemble"][operation]) for row in rows
        }
        if len(replicates) != 1:
            raise ValueError("native edit replicate count differs across loci")
        for replicate in range(next(iter(replicates))):
            for chunk_start in range(0, len(rows), batch_pairs):
                chunk = rows[chunk_start : chunk_start + batch_pairs]
                clean_pairs = [
                    (
                        str(row["example_id"]),
                        str(row["target"]["clean_sequence"]),
                        str(row["control"]["clean_sequence"]),
                    )
                    for row in chunk
                ]
                edited_pairs = [
                    (
                        str(row["example_id"]),
                        str(
                            row["native_source"]["edit_ensemble"][operation][replicate][
                                "target"
                            ]["edited_sequence"]
                        ),
                        str(
                            row["native_source"]["edit_ensemble"][operation][replicate][
                                "control"
                            ]["edited_sequence"]
                        ),
                    )
                    for row in chunk
                ]
                clean_batch = tokenize_promoter_pairs(bundle, clean_pairs)
                edited_batch = tokenize_promoter_pairs(bundle, edited_pairs)
                with torch.no_grad():
                    clean = promoter_margin_tensor(bundle, clean_batch).detach()
                    edited = promoter_margin_tensor(bundle, edited_batch).detach()
                target_loss = clean[0::2] - edited[0::2]
                control_loss = clean[1::2] - edited[1::2]
                target_specific = target_loss - control_loss
                if not all(
                    torch.isfinite(values).all()
                    for values in (clean, edited, target_specific)
                ):
                    raise RuntimeError("native edit produced non-finite margins")
                for index, row in enumerate(chunk):
                    edit = row["native_source"]["edit_ensemble"][operation][replicate]
                    output.append(
                        {
                            "pair_id": str(row["example_id"]),
                            "homology_cluster": str(row["homology_cluster"]),
                            "operation": operation,
                            "edit_replicate": replicate,
                            "target_clean_margin": float(clean[2 * index]),
                            "target_edited_margin": float(edited[2 * index]),
                            "control_clean_margin": float(clean[2 * index + 1]),
                            "control_edited_margin": float(edited[2 * index + 1]),
                            "target_loss": float(target_loss[index]),
                            "control_loss": float(control_loss[index]),
                            "target_specific_loss": float(target_specific[index]),
                            "changed_nucleotides_each": int(
                                edit["changed_nucleotides_each"]
                            ),
                            "target_token_count_delta": int(
                                edit["target"]["token_geometry"]["token_count_delta"]
                            ),
                            "control_token_count_delta": int(
                                edit["control"]["token_geometry"]["token_count_delta"]
                            ),
                        }
                    )
                if progress is not None:
                    progress(
                        f"{operation} edit {replicate + 1}: "
                        f"{min(chunk_start + batch_pairs, len(rows))}/{len(rows)}"
                    )
    return output


def score_identity_replay(
    bundle: Dnabert2PromoterBundle,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_pairs: int,
) -> float:
    """Return maximum margin drift when clean inputs are scored twice."""

    import torch

    maximum = 0.0
    for start in range(0, len(rows), batch_pairs):
        chunk = rows[start : start + batch_pairs]
        pairs = [
            (
                str(row["example_id"]),
                str(row["target"]["clean_sequence"]),
                str(row["control"]["clean_sequence"]),
            )
            for row in chunk
        ]
        batch = tokenize_promoter_pairs(bundle, pairs)
        with torch.no_grad():
            first = promoter_margin_tensor(bundle, batch).detach()
            second = promoter_margin_tensor(bundle, batch).detach()
        maximum = max(maximum, float((first - second).abs().max()))
    return maximum


def _row_seed(seed: int, pair_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}|{pair_id}".encode()).digest()[:8],
        "big",
    )
