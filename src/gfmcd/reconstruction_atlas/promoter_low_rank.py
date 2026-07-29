"""Low-rank causal decomposition of promoter CLS MLP writes."""

from __future__ import annotations

import base64
import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from gfmcd.adapters.dnabert2 import (
    Dnabert2PromoterBundle,
    capture_paired_cls_mlp_writes,
    patch_paired_cls_mlp_projection,
    patch_paired_cls_mlp_writes,
    promoter_margin_tensor,
    tokenize_promoter_pairs,
)
from gfmcd.reconstruction_atlas.promoter_assay import binary_classification_metrics
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    cluster_bootstrap_interval,
    paired_cluster_sign_permutation,
)

Pair = tuple[str, str, str]


def fit_uncentered_difference_basis(
    paired_writes: Any,
    *,
    maximum_rank: int,
) -> tuple[Any, dict[str, Any]]:
    """Fit right singular vectors of target-minus-control CLS writes."""

    import torch

    if (
        paired_writes.ndim != 2
        or paired_writes.shape[0] < 2
        or paired_writes.shape[0] % 2
    ):
        raise ValueError("paired writes must be [2 * pairs, hidden]")
    if maximum_rank < 1:
        raise ValueError("maximum rank must be positive")
    differences = (
        paired_writes[0::2].detach().float().cpu()
        - paired_writes[1::2].detach().float().cpu()
    )
    available_rank = min(differences.shape)
    rank = min(int(maximum_rank), available_rank)
    _left, singular_values, right = torch.linalg.svd(differences, full_matrices=False)
    basis = canonicalize_basis_signs(right[:rank].T.contiguous())
    squared = singular_values.square()
    total = float(squared.sum())
    cumulative = (
        (squared.cumsum(0) / total).tolist() if total > 0 else [0.0] * len(squared)
    )
    mean_difference = differences.mean(dim=0)
    mean_norm = float(torch.linalg.vector_norm(mean_difference))
    mean_capture = []
    for candidate_rank in range(1, rank + 1):
        selected = basis[:, :candidate_rank]
        projected = (mean_difference @ selected) @ selected.T
        mean_capture.append(
            float(torch.linalg.vector_norm(projected) / mean_norm)
            if mean_norm > 0
            else 0.0
        )
    diagnostics = {
        "pairs": int(differences.shape[0]),
        "hidden_size": int(differences.shape[1]),
        "fitted_rank": rank,
        "singular_values": [float(value) for value in singular_values[:rank]],
        "cumulative_difference_energy": [float(value) for value in cumulative[:rank]],
        "cumulative_mean_difference_norm": mean_capture,
    }
    return basis, diagnostics


def canonicalize_basis_signs(basis: Any) -> Any:
    """Resolve the arbitrary sign of each orthonormal basis column."""

    import torch

    if basis.ndim != 2:
        raise ValueError("basis must be a matrix")
    output = basis.detach().float().cpu().clone()
    for column in range(output.shape[1]):
        vector = output[:, column]
        pivot = int(torch.argmax(torch.abs(vector)))
        if float(vector[pivot]) < 0:
            output[:, column].mul_(-1)
    return output


def random_orthonormal_basis(
    *,
    hidden_size: int,
    rank: int,
    seed: int,
) -> Any:
    """Draw a deterministic Haar-like dimension-matched random subspace."""

    import torch

    if hidden_size < 1 or not 0 <= rank <= hidden_size:
        raise ValueError("random-basis dimensions are invalid")
    if rank == 0:
        return torch.empty((hidden_size, 0), dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn((hidden_size, rank), generator=generator)
    basis, _upper = torch.linalg.qr(matrix, mode="reduced")
    return canonicalize_basis_signs(basis)


def encode_float32_basis(basis: Any) -> dict[str, Any]:
    """Encode a compact exact float32 basis for an immutable JSON result."""

    array = basis.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    payload = array.tobytes(order="C")
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": "float32_le",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "base64": base64.b64encode(payload).decode("ascii"),
    }


def decode_float32_basis(record: Mapping[str, Any]) -> Any:
    """Decode and verify an immutable basis record."""

    import numpy as np
    import torch

    shape = tuple(int(value) for value in record["shape"])
    if len(shape) != 2 or min(shape) < 0 or record.get("dtype") != "float32_le":
        raise ValueError("malformed basis record")
    payload = base64.b64decode(str(record["base64"]), validate=True)
    if hashlib.sha256(payload).hexdigest() != str(record["sha256"]):
        raise ValueError("basis payload hash mismatch")
    expected_bytes = math.prod(shape) * 4
    if len(payload) != expected_bytes:
        raise ValueError("basis payload length mismatch")
    array = np.frombuffer(payload, dtype="<f4").reshape(shape).copy()
    return torch.from_numpy(array)


def capture_mlp_writes_for_pairs(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    *,
    layer: int,
    batch_pairs: int,
) -> tuple[list[str], Any, Any]:
    """Capture ordered native margins and CLS MLP writes."""

    import torch

    if not pairs or batch_pairs < 1:
        raise ValueError("capture requires pairs and a positive batch size")
    pair_ids: list[str] = []
    margins = []
    writes = []
    for start in range(0, len(pairs), batch_pairs):
        chunk = pairs[start : start + batch_pairs]
        batch = tokenize_promoter_pairs(bundle, chunk)
        chunk_margins, chunk_writes = capture_paired_cls_mlp_writes(
            bundle, batch, layer=layer
        )
        pair_ids.extend(batch.pair_ids)
        margins.append(chunk_margins.detach().float().cpu())
        writes.append(chunk_writes.detach().float().cpu())
    return pair_ids, torch.cat(margins), torch.cat(writes)


def paired_mlp_projection_rows(
    bundle: Dnabert2PromoterBundle,
    pairs: Sequence[Pair],
    *,
    layer: int,
    batch_pairs: int,
    direction: str,
    basis: Any | None,
    component: str = "subspace",
) -> list[dict[str, Any]]:
    """Score a full or projected paired CLS MLP-write exchange."""

    import torch

    if not pairs or batch_pairs < 1:
        raise ValueError("projection assay requires pairs and a positive batch size")
    rows = []
    for start in range(0, len(pairs), batch_pairs):
        chunk = pairs[start : start + batch_pairs]
        batch = tokenize_promoter_pairs(bundle, chunk)
        with torch.no_grad():
            native = promoter_margin_tensor(bundle, batch).detach()
        context = (
            patch_paired_cls_mlp_writes(
                bundle,
                batch,
                (layer,),
                direction=direction,
            )
            if basis is None
            else patch_paired_cls_mlp_projection(
                bundle,
                batch,
                layer=layer,
                basis=basis,
                direction=direction,
                component=component,
            )
        )
        with context, torch.no_grad():
            patched = promoter_margin_tensor(bundle, batch).detach()
        if not torch.isfinite(patched).all():
            raise RuntimeError("low-rank projection produced non-finite margins")
        for pair_index, pair_id in enumerate(batch.pair_ids):
            target_native = float(native[2 * pair_index])
            control_native = float(native[2 * pair_index + 1])
            target_patched = float(patched[2 * pair_index])
            control_patched = float(patched[2 * pair_index + 1])
            target_correctness_loss = int(target_native > 0) - int(target_patched > 0)
            control_correctness_loss = int(control_native <= 0) - int(
                control_patched <= 0
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "native_target_margin": target_native,
                    "native_control_margin": control_native,
                    "patched_target_margin": target_patched,
                    "patched_control_margin": control_patched,
                    "target_margin_loss": target_native - target_patched,
                    "control_margin_loss": control_patched - control_native,
                    "class_margin_loss": (
                        target_native
                        - target_patched
                        + control_patched
                        - control_native
                    )
                    / 2,
                    "target_correctness_loss": target_correctness_loss,
                    "control_correctness_loss": control_correctness_loss,
                    "accuracy_loss": (
                        target_correctness_loss + control_correctness_loss
                    )
                    / 2,
                }
            )
    return rows


def summarize_projection_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_draws: int,
    sign_permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize one projected intervention at the paired-locus unit."""

    if not rows or bootstrap_draws < 1 or sign_permutations < 1:
        raise ValueError("projection rows and inference draws must be positive")
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(set(pair_ids)) != len(rows):
        raise ValueError("projection summary requires one row per pair")
    class_losses = [float(row["class_margin_loss"]) for row in rows]
    accuracy_losses = [float(row["accuracy_loss"]) for row in rows]
    observed, sign_null = paired_cluster_sign_permutation(
        class_losses,
        pair_ids,
        permutations=sign_permutations,
        seed=seed,
    )
    labels = [label for _row in rows for label in (1, 0)]
    native_margins = [
        margin
        for row in rows
        for margin in (
            float(row["native_target_margin"]),
            float(row["native_control_margin"]),
        )
    ]
    patched_margins = [
        margin
        for row in rows
        for margin in (
            float(row["patched_target_margin"]),
            float(row["patched_control_margin"]),
        )
    ]
    clean = binary_classification_metrics(labels, native_margins)
    patched = binary_classification_metrics(labels, patched_margins)
    return {
        "n_pairs": len(rows),
        "class_margin_loss": observed,
        "class_margin_loss_CI95": list(
            cluster_bootstrap_interval(
                class_losses,
                pair_ids,
                draws=bootstrap_draws,
                seed=seed ^ 0xA11CE,
            )
        ),
        "class_margin_loss_p_add_one": add_one_permutation_pvalue(
            observed, sign_null, alternative="greater"
        ),
        "target_margin_loss": statistics.mean(
            float(row["target_margin_loss"]) for row in rows
        ),
        "control_margin_loss": statistics.mean(
            float(row["control_margin_loss"]) for row in rows
        ),
        "accuracy_loss": statistics.mean(accuracy_losses),
        "accuracy_loss_CI95": list(
            cluster_bootstrap_interval(
                accuracy_losses,
                pair_ids,
                draws=bootstrap_draws,
                seed=seed ^ 0xACC,
            )
        ),
        "auroc_loss": float(clean["auroc"]) - float(patched["auroc"]),
        "clean": clean,
        "patched": patched,
    }


def random_subspace_add_one_pvalue(
    candidate: Mapping[str, Any],
    nulls: Sequence[Mapping[str, Any]],
) -> float:
    """Compare one causal subspace with dimension-matched random subspaces."""

    return add_one_permutation_pvalue(
        float(candidate["class_margin_loss"]),
        [float(null["class_margin_loss"]) for null in nulls],
        alternative="greater",
    )


def select_low_rank_candidate(
    rank_summaries: Mapping[int, Mapping[str, Any]],
    *,
    full_summary: Mapping[str, Any],
    maximum_rank: int,
    minimum_class_margin_loss: float,
    minimum_capture_fraction: float,
    minimum_accuracy_loss: float,
) -> tuple[int | None, dict[int, dict[str, Any]]]:
    """Choose the smallest preregistered rank clearing development thresholds."""

    full_effect = float(full_summary["class_margin_loss"])
    if full_effect <= 0:
        raise ValueError("full L11MLP effect must be positive before rank selection")
    checks: dict[int, dict[str, Any]] = {}
    selected = None
    for rank in sorted(rank_summaries):
        if rank > maximum_rank:
            continue
        summary = rank_summaries[rank]
        capture_fraction = float(summary["class_margin_loss"]) / full_effect
        rank_checks = {
            "class_margin_loss_at_least_SESOI": float(summary["class_margin_loss"])
            >= minimum_class_margin_loss,
            "capture_fraction_at_least_threshold": capture_fraction
            >= minimum_capture_fraction,
            "accuracy_loss_at_least_SESOI": float(summary["accuracy_loss"])
            >= minimum_accuracy_loss,
            "positive_class_margin_CI": float(summary["class_margin_loss_CI95"][0]) > 0,
        }
        checks[rank] = {
            "capture_fraction": capture_fraction,
            "checks": rank_checks,
            "qualified_before_random_null": all(rank_checks.values()),
        }
        if selected is None and all(rank_checks.values()):
            selected = rank
    return selected, checks
