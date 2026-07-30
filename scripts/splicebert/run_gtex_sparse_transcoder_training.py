#!/usr/bin/env python
"""Fit and preflight label-free six-view sparse final-MLP transcoders."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.circuits.multiview_transcoder import (  # noqa: E402
    StructuredMultiViewSparseTranscoder,
    StructuredMultiViewTranscoderConfig,
    assert_label_free_payload,
    feature_similarity_matrices,
    multiview_transcoder_loss,
    standardized_consistency_loss,
)
from gfmcd.provenance import (  # noqa: E402
    ExperimentManifest,
    file_record,
    sha256_file,
)


EXPECTED_SEEDS = tuple(range(2020, 2025))
VIEWS = ("base", *(str(seed) for seed in EXPECTED_SEEDS))
TUNED_VIEWS = tuple(str(seed) for seed in EXPECTED_SEEDS)
NAMESPACE = "gtex-native-biological-features-v1"
FEATURE_WIDTHS = (128, 256, 512, 1024)
ACTIVE_COUNTS = (8, 16, 32)
INITIALIZATIONS = (0, 1, 2)
TRAIN_NAMESPACE = "gtex-native-biological-features-transcoder-train-v1"
VALIDATION_PERCENT = 20
PAIR_PERMUTATIONS = 20
PERMUTATION_GREATER_FRACTION = 0.95
PERMUTATION_ABSOLUTE_GAP = 0.05
PERMUTATION_RATIO = 1.10
DEFAULT_EPOCHS = 60
DEFAULT_PATIENCE = 8
DEFAULT_BATCH = 256
DEFAULT_LR = 1e-3
LOSS_WEIGHTS = {
    "reconstruction": 1.0,
    "paired_difference": 1.0,
    "view_difference": 1.0,
    "sparsity": 1e-3,
    "consistency": 0.1,
}


def stable_bucket(value: str, *, namespace: str, modulo: int = 100) -> int:
    digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def stable_identifier_hash(values: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def gene_train_validation_split(
    genes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    validation_genes = {
        gene
        for gene in set(genes)
        if stable_bucket(gene, namespace=TRAIN_NAMESPACE) < VALIDATION_PERCENT
    }
    validation = np.asarray(
        [gene in validation_genes for gene in genes],
        dtype=bool,
    )
    train = ~validation
    if len(set(np.asarray(genes, dtype=object)[train])) < 2 or len(
        set(np.asarray(genes, dtype=object)[validation])
    ) < 2:
        raise RuntimeError("dictionary train/validation gene split is too small")
    return train, validation


def pair_indices(
    states: list[Mapping[str, Any]],
    selected: np.ndarray,
) -> np.ndarray:
    state_index = {str(row["state_id"]): index for index, row in enumerate(states)}
    pairs = []
    for changed_index, row in enumerate(states):
        reference_id = row.get("reference_state_id")
        if reference_id is None:
            continue
        reference_index = state_index[str(reference_id)]
        if selected[reference_index] and selected[changed_index]:
            pairs.append((reference_index, changed_index))
    if not pairs:
        raise RuntimeError("no reference/counterfactual pairs in dictionary split")
    return np.asarray(pairs, dtype=np.int64)


def local_pairs(
    indices: np.ndarray,
    global_pairs: np.ndarray,
) -> torch.Tensor:
    locations: dict[int, list[int]] = {}
    for local, global_index in enumerate(indices.tolist()):
        locations.setdefault(int(global_index), []).append(local)
    pairs = []
    for reference, changed in global_pairs:
        if int(reference) in locations and int(changed) in locations:
            pairs.append(
                (
                    locations[int(reference)][0],
                    locations[int(changed)][0],
                )
            )
    return torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2)


def build_epoch_batches(
    train_indices: np.ndarray,
    pairs: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> list[np.ndarray]:
    if batch_size < 8:
        raise ValueError("training batch must be at least eight")
    generator = np.random.default_rng(seed)
    regular = generator.permutation(train_indices)
    pair_order = pairs[generator.permutation(len(pairs))]
    regular_count = batch_size // 2
    pair_count = max(1, (batch_size - regular_count) // 2)
    batches = []
    pair_cursor = 0
    for start in range(0, len(regular), regular_count):
        base = regular[start : start + regular_count]
        if pair_cursor + pair_count > len(pair_order):
            pair_order = pairs[generator.permutation(len(pairs))]
            pair_cursor = 0
        chosen = pair_order[pair_cursor : pair_cursor + pair_count].reshape(-1)
        pair_cursor += pair_count
        batches.append(np.concatenate([base, chosen]))
    return batches


def load_activation_artifact(
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = json.loads(report_path.read_text())
    assert_label_free_payload(report)
    if (
        report.get("status") != "COMPLETE"
        or report.get("stage") != "label-free activation capture"
        or report.get("cohort_namespace") != NAMESPACE
        or report.get("label_free") is not True
        or report.get("training_authorized") is not True
        or tuple(report.get("views", [])) != VIEWS
        or not report.get("instrument_gate", {}).get("PASS")
    ):
        raise RuntimeError("activation artifact does not authorize dictionary training")
    arrays_path = Path(report["arrays"]["path"]).expanduser().resolve()
    if sha256_file(arrays_path) != report["arrays"]["sha256"]:
        raise RuntimeError("activation array hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    expected_keys = {
        f"{view}__{kind}" for view in VIEWS for kind in ("input", "write")
    }
    if set(arrays) != expected_keys:
        raise RuntimeError("six-view activation array keys mismatch")
    expected_shape = (int(report["n_states"]), 512)
    if any(
        value.shape != expected_shape or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise RuntimeError("six-view activation array shape/finiteness failed")
    return report, arrays


def normalize_inputs(
    arrays: Mapping[str, np.ndarray],
    train_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    normalized: dict[str, np.ndarray] = {}
    statistics: dict[str, dict[str, np.ndarray]] = {}
    for view in VIEWS:
        values = arrays[f"{view}__input"].astype(np.float32)
        mean = values[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values[train_mask].std(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(scale, np.float32(1e-4))
        normalized[view] = ((values - mean) / scale).astype(np.float32)
        statistics[view] = {"mean": mean, "scale": scale}
    return normalized, statistics


def tensors_at(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
    suffix: str,
) -> dict[str, torch.Tensor]:
    return {
        view: torch.as_tensor(
            arrays[f"{view}__{suffix}"][indices],
            dtype=torch.float32,
            device=device,
        )
        for view in VIEWS
    }


def model_config(width: int, active: int) -> StructuredMultiViewTranscoderConfig:
    return StructuredMultiViewTranscoderConfig(
        input_dim=512,
        output_dim=512,
        num_features=width,
        active_features=active,
        views=VIEWS,
        tuned_views=TUNED_VIEWS,
    )


def clone_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def fractional_energy_explained(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    center: bool,
) -> float:
    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if truth.shape != estimate.shape:
        raise ValueError("prediction and target shapes disagree")
    baseline = truth.mean(axis=0, keepdims=True) if center else 0.0
    denominator = float(np.square(truth - baseline).sum())
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return 1.0 - float(np.square(estimate - truth).sum()) / denominator


@torch.no_grad()
def evaluate_model(
    model: StructuredMultiViewSparseTranscoder,
    normalized_inputs: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    pairs: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    predictions: dict[str, list[np.ndarray]] = {view: [] for view in VIEWS}
    features: dict[str, list[np.ndarray]] = {view: [] for view in VIEWS}
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        for view in VIEWS:
            inputs = torch.as_tensor(
                normalized_inputs[view][selected],
                dtype=torch.float32,
                device=device,
            )
            activation, prediction = model(view, inputs)
            features[view].append(activation.cpu().numpy())
            predictions[view].append(prediction.cpu().numpy())
    feature_arrays = {
        view: np.concatenate(values) for view, values in features.items()
    }
    prediction_arrays = {
        view: np.concatenate(values) for view, values in predictions.items()
    }
    location = {int(global_index): local for local, global_index in enumerate(indices)}
    local_pair_array = np.asarray(
        [(location[int(left)], location[int(right)]) for left, right in pairs],
        dtype=np.int64,
    )
    write_fve = {}
    delta_fve = {}
    l0 = {}
    for view in VIEWS:
        target = arrays[f"{view}__write"][indices]
        write_fve[view] = fractional_energy_explained(
            prediction_arrays[view],
            target,
            center=True,
        )
        target_delta = (
            target[local_pair_array[:, 0]] - target[local_pair_array[:, 1]]
        )
        prediction_delta = (
            prediction_arrays[view][local_pair_array[:, 0]]
            - prediction_arrays[view][local_pair_array[:, 1]]
        )
        delta_fve[view] = fractional_energy_explained(
            prediction_delta,
            target_delta,
            center=False,
        )
        l0[view] = float(
            np.count_nonzero(feature_arrays[view], axis=1).mean()
        )

    tuned_features = {
        view: torch.as_tensor(
            feature_arrays[view][:, model.config.consistent_slice]
        )
        for view in TUNED_VIEWS
    }
    actual_consistency = float(
        standardized_consistency_loss(tuned_features, TUNED_VIEWS)
    )
    permuted_losses = []
    for draw in range(PAIR_PERMUTATIONS):
        permuted = {}
        for view_index, view in enumerate(TUNED_VIEWS):
            generator = np.random.default_rng(
                91_000 + 1_000 * draw + view_index
            )
            order = (
                np.arange(len(indices))
                if view_index == 0
                else generator.permutation(len(indices))
            )
            permuted[view] = torch.as_tensor(
                feature_arrays[view][order, model.config.consistent_slice]
            )
        permuted_losses.append(
            float(standardized_consistency_loss(permuted, TUNED_VIEWS))
        )
    mean_permuted_consistency = float(np.mean(permuted_losses))
    greater_fraction = float(
        np.mean(np.asarray(permuted_losses) > actual_consistency)
    )
    ratio = mean_permuted_consistency / max(actual_consistency, 1e-8)
    pair_permutation_gate = (
        greater_fraction >= PERMUTATION_GREATER_FRACTION
        and mean_permuted_consistency
        >= actual_consistency + PERMUTATION_ABSOLUTE_GAP
        and ratio >= PERMUTATION_RATIO
    )
    metrics = {
        "write_fve_by_view": write_fve,
        "write_delta_fve_by_view": delta_fve,
        "mean_write_fve": float(np.mean(list(write_fve.values()))),
        "mean_write_delta_fve": float(np.mean(list(delta_fve.values()))),
        "l0_by_view": l0,
        "actual_seed_consistency_loss": actual_consistency,
        "permuted_seed_consistency_losses": permuted_losses,
        "mean_permuted_seed_consistency_loss": mean_permuted_consistency,
        "permuted_greater_fraction": greater_fraction,
        "permuted_to_actual_consistency_ratio": ratio,
        "pair_permutation_thresholds": {
            "draws": PAIR_PERMUTATIONS,
            "minimum_greater_fraction": PERMUTATION_GREATER_FRACTION,
            "minimum_absolute_gap": PERMUTATION_ABSOLUTE_GAP,
            "minimum_ratio": PERMUTATION_RATIO,
        },
        "pair_permutation_reconstruction_unchanged_by_construction": True,
        "pair_permutation_gate": pair_permutation_gate,
    }
    return metrics, feature_arrays


def train_one(
    normalized_inputs: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    train_pairs: np.ndarray,
    validation_pairs: np.ndarray,
    *,
    width: int,
    active: int,
    initialization: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    view_difference_weight: float,
    consistency_weight: float,
) -> tuple[
    StructuredMultiViewSparseTranscoder,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    seed = 1_000_000 + 10_000 * width + 100 * active + initialization
    torch.manual_seed(seed)
    model = StructuredMultiViewSparseTranscoder(
        model_config(width, active)
    ).to(device)
    with torch.no_grad():
        for view in VIEWS:
            model.output_bias[view].copy_(
                torch.as_tensor(
                    arrays[f"{view}__write"][train_indices].mean(axis=0),
                    dtype=torch.float32,
                    device=device,
                )
            )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        batches = build_epoch_batches(
            train_indices,
            train_pairs,
            batch_size=batch_size,
            seed=seed + epoch,
        )
        for batch_indices in batches:
            inputs = {
                view: torch.as_tensor(
                    normalized_inputs[view][batch_indices],
                    dtype=torch.float32,
                    device=device,
                )
                for view in VIEWS
            }
            targets = tensors_at(
                arrays,
                batch_indices,
                device=device,
                suffix="write",
            )
            batch_pairs = local_pairs(batch_indices, train_pairs).to(device)
            optimizer.zero_grad(set_to_none=True)
            result = multiview_transcoder_loss(
                model,
                inputs,
                targets,
                pair_indices=batch_pairs,
                contrast_pairs=tuple(
                    ("base", view) for view in TUNED_VIEWS
                ),
                reconstruction_weight=LOSS_WEIGHTS["reconstruction"],
                difference_weight=LOSS_WEIGHTS["paired_difference"],
                view_difference_weight=view_difference_weight,
                sparsity_weight=LOSS_WEIGHTS["sparsity"],
                consistency_weight=consistency_weight,
                consistency_indices=model.config.consistent_slice,
            )
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(result["loss"].detach().cpu()))

        metrics, _ = evaluate_model(
            model,
            normalized_inputs,
            arrays,
            validation_indices,
            validation_pairs,
            device=device,
            batch_size=batch_size,
        )
        validation_loss = 1.0 - metrics["mean_write_delta_fve"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "validation_write_delta_error": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = clone_cpu_state(model)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("transcoder training never produced a finite candidate")
    model.load_state_dict(best_state)
    model.to(device)
    calibration_inputs = {
        view: torch.as_tensor(
            normalized_inputs[view][train_indices],
            dtype=torch.float32,
            device=device,
        )
        for view in VIEWS
    }
    model.calibrate_thresholds(calibration_inputs)
    metrics, features = evaluate_model(
        model,
        normalized_inputs,
        arrays,
        validation_indices,
        validation_pairs,
        device=device,
        batch_size=batch_size,
    )
    metrics.update(
        {
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "training_tail": history[-5:],
            "seed": seed,
        }
    )
    return model, metrics, features


def stability_score(
    models: list[StructuredMultiViewSparseTranscoder],
    feature_sets: list[dict[str, np.ndarray]],
    *,
    views: tuple[str, ...] = TUNED_VIEWS,
) -> dict[str, Any]:
    if len(models) != len(feature_sets) or len(models) != len(INITIALIZATIONS):
        raise ValueError("stability requires exactly one feature set per initialization")
    if not views or set(views) - set(TUNED_VIEWS):
        raise ValueError("stability views must be non-empty tuned views")
    scores = []
    consistent_stop = models[0].config.consistent_slice.stop
    for view in views:
        for left, right in combinations(range(len(models)), 2):
            matrices = feature_similarity_matrices(
                torch.as_tensor(feature_sets[left][view][:, :consistent_stop]),
                torch.as_tensor(feature_sets[right][view][:, :consistent_stop]),
                models[left].decoder(view).detach().cpu()[:, :consistent_stop],
                models[right].decoder(view).detach().cpu()[:, :consistent_stop],
            )["combined"].numpy()
            forward = matrices.argmax(axis=1)
            reverse = matrices.argmax(axis=0)
            reciprocal = np.asarray(
                [reverse[target] == source for source, target in enumerate(forward)]
            )
            reciprocal_scores = matrices[
                np.arange(len(forward)),
                forward,
            ][reciprocal]
            scores.append(
                {
                    "view": view,
                    "initialization_pair": [left, right],
                    "reciprocal_match_fraction": float(reciprocal.mean()),
                    "reciprocal_score_mean": (
                        float(reciprocal_scores.mean())
                        if reciprocal_scores.size
                        else 0.0
                    ),
                }
            )
    passed = all(
        row["reciprocal_match_fraction"] >= 0.25
        and row["reciprocal_score_mean"] >= 0.50
        for row in scores
    )
    return {
        "views": list(views),
        "all_initialization_pairs": scores,
        "thresholds": {
            "reciprocal_match_fraction": 0.25,
            "reciprocal_score_mean": 0.50,
        },
        "PASS": passed,
    }


def posthoc_cross_view_matching(
    model: StructuredMultiViewSparseTranscoder,
    feature_set: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Match independently fitted view features without assuming common indices."""

    stop = model.config.consistent_slice.stop
    rows = []
    for left, right in combinations(TUNED_VIEWS, 2):
        matrices = feature_similarity_matrices(
            torch.as_tensor(feature_set[left][:, :stop]),
            torch.as_tensor(feature_set[right][:, :stop]),
            model.decoder(left).detach().cpu()[:, :stop],
            model.decoder(right).detach().cpu()[:, :stop],
        )["combined"].numpy()
        forward = matrices.argmax(axis=1)
        reverse = matrices.argmax(axis=0)
        reciprocal = np.asarray(
            [reverse[target] == source for source, target in enumerate(forward)]
        )
        reciprocal_scores = matrices[
            np.arange(len(forward)),
            forward,
        ][reciprocal]
        rows.append(
            {
                "view_pair": [left, right],
                "reciprocal_match_fraction": float(reciprocal.mean()),
                "reciprocal_score_mean": (
                    float(reciprocal_scores.mean())
                    if reciprocal_scores.size
                    else 0.0
                ),
            }
        )
    return {
        "view_pairs": rows,
        "mean_reciprocal_match_fraction": float(
            np.mean([row["reciprocal_match_fraction"] for row in rows])
        ),
        "mean_reciprocal_score": float(
            np.mean([row["reciprocal_score_mean"] for row in rows])
        ),
    }


def subspace_baselines(
    arrays: Mapping[str, np.ndarray],
    train_pairs: np.ndarray,
    validation_pairs: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"pca_and_haar": {}, "random_dictionaries": {}}
    for rank in ACTIVE_COUNTS:
        pca_scores = []
        random_scores = []
        for view_index, view in enumerate(VIEWS):
            writes = arrays[f"{view}__write"].astype(np.float64)
            train_delta = (
                writes[train_pairs[:, 0]] - writes[train_pairs[:, 1]]
            )
            validation_delta = (
                writes[validation_pairs[:, 0]]
                - writes[validation_pairs[:, 1]]
            )
            _, _, vt = np.linalg.svd(train_delta, full_matrices=False)
            pca_basis = vt[:rank]
            pca_prediction = (
                validation_delta @ pca_basis.T
            ) @ pca_basis
            pca_scores.append(
                fractional_energy_explained(
                    pca_prediction,
                    validation_delta,
                    center=False,
                )
            )
            generator = np.random.default_rng(700_000 + 100 * rank + view_index)
            random_matrix = generator.standard_normal((512, rank))
            random_basis, _ = np.linalg.qr(random_matrix, mode="reduced")
            random_prediction = (
                validation_delta @ random_basis
            ) @ random_basis.T
            random_scores.append(
                fractional_energy_explained(
                    random_prediction,
                    validation_delta,
                    center=False,
                )
            )
        result["pca_and_haar"][str(rank)] = {
            "pca_svd_mean_write_delta_fve": float(np.mean(pca_scores)),
            "haar_random_mean_write_delta_fve": float(np.mean(random_scores)),
        }
    for width in FEATURE_WIDTHS:
        for active in ACTIVE_COUNTS:
            scores = []
            for view_index, view in enumerate(VIEWS):
                writes = arrays[f"{view}__write"].astype(np.float64)
                validation_delta = (
                    writes[validation_pairs[:, 0]]
                    - writes[validation_pairs[:, 1]]
                )
                generator = np.random.default_rng(
                    800_000 + 10_000 * width + 100 * active + view_index
                )
                decoder = generator.standard_normal((512, width))
                decoder /= np.maximum(
                    np.linalg.norm(decoder, axis=0, keepdims=True),
                    np.finfo(np.float64).eps,
                )
                coefficients = validation_delta @ decoder
                chosen = np.argpartition(
                    np.abs(coefficients),
                    kth=width - active,
                    axis=1,
                )[:, -active:]
                sparse = np.zeros_like(coefficients)
                rows = np.arange(len(coefficients))[:, None]
                sparse[rows, chosen] = coefficients[rows, chosen]
                prediction = sparse @ decoder.T
                scores.append(
                    fractional_energy_explained(
                        prediction,
                        validation_delta,
                        center=False,
                    )
                )
            result["random_dictionaries"][f"w{width}_k{active}"] = {
                "mean_write_delta_fve": float(np.mean(scores)),
                "writer_column_norm": 1.0,
                "active_coefficients_per_delta": active,
            }
    return result


def planted_feature_preflight(device: torch.device) -> dict[str, Any]:
    """Train a tiny label-free organism and verify recovery of a known feature."""

    generator = np.random.default_rng(913_771)
    n = 256
    inputs_array = generator.standard_normal((n, 8)).astype(np.float32)
    planted_activation = np.maximum(inputs_array[:, 0] - 0.5, 0.0)
    writer = generator.standard_normal(8)
    writer /= np.linalg.norm(writer)
    config = StructuredMultiViewTranscoderConfig(
        input_dim=8,
        output_dim=8,
        num_features=16,
        active_features=4,
        views=VIEWS,
        tuned_views=TUNED_VIEWS,
        threshold_momentum=0.0,
    )
    torch.manual_seed(913_771)
    model = StructuredMultiViewSparseTranscoder(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x = {
        view: torch.as_tensor(inputs_array, device=device) for view in VIEWS
    }
    target = {
        "base": torch.zeros((n, 8), dtype=torch.float32, device=device),
        **{
            view: torch.as_tensor(
                planted_activation[:, None] * writer[None, :],
                dtype=torch.float32,
                device=device,
            )
            for view in TUNED_VIEWS
        },
    }
    for _ in range(700):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = multiview_transcoder_loss(
            model,
            x,
            target,
            contrast_pairs=tuple(("base", view) for view in TUNED_VIEWS),
            reconstruction_weight=1.0,
            difference_weight=0.0,
            view_difference_weight=1.0,
            sparsity_weight=1e-3,
            consistency_weight=0.1,
            consistency_indices=config.consistent_slice,
        )
        result["loss"].backward()
        optimizer.step()
    model.calibrate_thresholds(x)
    model.eval()
    correlations = []
    writer_cosines = []
    planted = torch.as_tensor(planted_activation)
    planted_centered = planted - planted.mean()
    for view in TUNED_VIEWS:
        features = model.encode(view, x[view]).detach().cpu()
        centered = features - features.mean(dim=0, keepdim=True)
        denominator = (
            torch.linalg.vector_norm(centered, dim=0)
            * torch.linalg.vector_norm(planted_centered)
        ).clamp_min(1e-8)
        correlation = (centered.T @ planted_centered) / denominator
        feature = int(torch.argmax(correlation))
        correlations.append(float(correlation[feature]))
        decoder = model.decoder(view).detach().cpu()[:, feature].numpy()
        writer_cosines.append(float(decoder @ writer))
    minimum_correlation = min(correlations)
    minimum_writer_cosine = min(writer_cosines)
    return {
        "minimum_activation_correlation": minimum_correlation,
        "minimum_writer_cosine": minimum_writer_cosine,
        "activation_correlation_floor": 0.80,
        "writer_cosine_floor": 0.70,
        "PASS": minimum_correlation >= 0.80 and minimum_writer_cosine >= 0.70,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-artifact", required=True)
    parser.add_argument("--native-unit-artifact", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (
        args.batch != DEFAULT_BATCH
        or args.epochs != DEFAULT_EPOCHS
        or args.patience != DEFAULT_PATIENCE
        or args.learning_rate != DEFAULT_LR
    ):
        raise ValueError("training hyperparameters differ from frozen runner defaults")
    device = torch.device(
        args.device
        or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_BIOLOGICAL_FEATURES_V1.md"
    )
    addendum = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "IMPLEMENTATION_ADDENDUM_SPARSE_TRANSCODER_V1.md"
    )
    activation_path = Path(args.activation_artifact).expanduser().resolve()
    report, arrays = load_activation_artifact(activation_path)
    native_unit_path = Path(args.native_unit_artifact).expanduser().resolve()
    native_unit = json.loads(native_unit_path.read_text())
    if (
        native_unit.get("status") != "COMPLETE"
        or native_unit.get("study")
        != "native GTEx exact final-MLP unit census"
    ):
        raise RuntimeError("GFS20 native-unit ceiling artifact is unavailable")
    output = guard_output_path(Path(args.out).resolve())
    model_output = guard_output_path(Path(args.model_out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    if len({output, model_output, manifest_path}) != 3:
        raise ValueError("result, model, and manifest outputs must be distinct")
    if output.suffix != ".json" or model_output.suffix not in {".pt", ".pth"}:
        raise ValueError("training requires .json result and .pt/.pth model artifact")

    states = report["states"]
    genes = [str(row["gene_id"]) for row in states]
    train_mask, validation_mask = gene_train_validation_split(genes)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    train_pairs = pair_indices(states, train_mask)
    validation_pairs = pair_indices(states, validation_mask)
    validation_gene_ids = sorted(set(np.asarray(genes)[validation_mask]))
    validation_state_ids = [
        str(state["state_id"])
        for state, selected in zip(states, validation_mask)
        if selected
    ]
    normalized_inputs, normalization = normalize_inputs(arrays, train_mask)
    planted = planted_feature_preflight(device)
    if not planted["PASS"]:
        raise RuntimeError(f"planted sparse-feature preflight failed: {planted}")

    candidates: dict[str, Any] = {}
    model_payload: dict[str, Any] = {
        "schema_version": 1,
        "views": list(VIEWS),
        "tuned_views": list(TUNED_VIEWS),
        "normalization": {
            view: {
                "mean": torch.as_tensor(values["mean"]).clone(),
                "scale": torch.as_tensor(values["scale"]).clone(),
            }
            for view, values in normalization.items()
        },
        "models": {},
    }
    baseline_results: dict[str, Any] = {
        "native_units": {
            "artifact": str(native_unit_path),
            "sha256": sha256_file(native_unit_path),
            "status": native_unit.get("status"),
            "scope": (
                "preregistered historical GFS20 gene-disjoint evaluation cohort; "
                "not re-estimated on the sparse dictionary split"
            ),
            "context_unit_curve": native_unit.get("context_unit_curve"),
            "context_unit_verdict": native_unit.get("context_unit_verdict"),
            "complete_native_mlp_ceiling_write_delta_fve": 1.0,
        },
        "linear_and_random": subspace_baselines(
            arrays,
            train_pairs,
            validation_pairs,
        ),
        "ablations": {},
    }

    for width in FEATURE_WIDTHS:
        for active in ACTIVE_COUNTS:
            setting = f"w{width}_k{active}"
            models = []
            features = []
            replicate_rows = []
            for initialization in INITIALIZATIONS:
                model, metrics, feature_values = train_one(
                    normalized_inputs,
                    arrays,
                    train_indices,
                    validation_indices,
                    train_pairs,
                    validation_pairs,
                    width=width,
                    active=active,
                    initialization=initialization,
                    device=device,
                    batch_size=args.batch,
                    epochs=args.epochs,
                    patience=args.patience,
                    learning_rate=args.learning_rate,
                    view_difference_weight=LOSS_WEIGHTS["view_difference"],
                    consistency_weight=LOSS_WEIGHTS["consistency"],
                )
                model_id = f"{setting}_r{initialization}"
                model_payload["models"][model_id] = {
                    "config": asdict(model.config),
                    "state_dict": clone_cpu_state(model),
                }
                models.append(deepcopy(model).cpu())
                features.append(feature_values)
                replicate_rows.append(metrics)
                del model
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
            stability = stability_score(models, features)
            representation_pass = (
                min(
                    min(row["write_delta_fve_by_view"].values())
                    for row in replicate_rows
                )
                >= 0.95
                and all(row["pair_permutation_gate"] for row in replicate_rows)
                and stability["PASS"]
            )
            candidates[setting] = {
                "feature_width": width,
                "active_features": active,
                "replicates": replicate_rows,
                "stability": stability,
                "representation_thresholds": {
                    "minimum_write_delta_fve": 0.95,
                    "pair_permutation": "permuted consistency loss must increase",
                },
                "representation_gate": representation_pass,
                "behavioral_preflight_pending": True,
            }

            ablations = {}
            for name, view_weight, consistency_weight in (
                ("without_view_difference", 0.0, LOSS_WEIGHTS["consistency"]),
                ("without_seed_consistency", LOSS_WEIGHTS["view_difference"], 0.0),
                ("independent_per_view", 0.0, 0.0),
            ):
                baseline_model, baseline_metrics, baseline_features = train_one(
                    normalized_inputs,
                    arrays,
                    train_indices,
                    validation_indices,
                    train_pairs,
                    validation_pairs,
                    width=width,
                    active=active,
                    initialization=0,
                    device=device,
                    batch_size=args.batch,
                    epochs=args.epochs,
                    patience=args.patience,
                    learning_rate=args.learning_rate,
                    view_difference_weight=view_weight,
                    consistency_weight=consistency_weight,
                )
                if name == "independent_per_view":
                    baseline_metrics["posthoc_cross_view_matching"] = (
                        posthoc_cross_view_matching(
                            baseline_model,
                            baseline_features,
                        )
                    )
                ablations[name] = baseline_metrics
                del baseline_model
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
            baseline_results["ablations"][setting] = ablations
            print(f"complete {setting}", flush=True)

    representation_candidates = [
        row for row in candidates.values() if row["representation_gate"]
    ]
    representation_gate_passed = bool(representation_candidates)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_payload, model_output)
    result = {
        "study": "native GTEx six-view sparse final-MLP feature model",
        "stage": "label-free sparse training and representation preflight",
        "status": (
            "COMPLETE"
            if representation_gate_passed
            else "INSTRUMENT_UNRESOLVED"
        ),
        "evidence_tier": "candidate-generation instrument",
        "protocol": str(protocol.relative_to(REPO)),
        "implementation_addendum": str(addendum.relative_to(REPO)),
        "cohort_namespace": NAMESPACE,
        "label_free": True,
        "activation_artifact": str(activation_path),
        "activation_arrays_sha256": report["arrays"]["sha256"],
        "dictionary_split": {
            "namespace": TRAIN_NAMESPACE,
            "validation_percent": VALIDATION_PERCENT,
            "n_train_states": int(train_mask.sum()),
            "n_validation_states": int(validation_mask.sum()),
            "n_train_genes": len(set(np.asarray(genes)[train_mask])),
            "n_validation_genes": len(set(np.asarray(genes)[validation_mask])),
            "n_train_pairs": len(train_pairs),
            "n_validation_pairs": len(validation_pairs),
            "validation_gene_ids": validation_gene_ids,
            "validation_gene_ids_sha256": stable_identifier_hash(
                validation_gene_ids
            ),
            "validation_state_ids_sha256": stable_identifier_hash(
                validation_state_ids
            ),
        },
        "grid": {
            "feature_widths": list(FEATURE_WIDTHS),
            "active_features": list(ACTIVE_COUNTS),
            "initializations": list(INITIALIZATIONS),
            "capacity": "one-half shared, one-quarter tuned-only, one-quarter private",
            "sparsity": (
                "blockwise BatchTopK training with fixed-L0 per-example TopK "
                "evaluation; full-training thresholds retained as diagnostics"
            ),
            "loss_weights": LOSS_WEIGHTS,
        },
        "planted_feature_preflight": planted,
        "candidates": candidates,
        "baselines": baseline_results,
        "model_artifact": {
            "path": str(model_output),
            "sha256": sha256_file(model_output),
            "safe_load": "torch.load(weights_only=True)",
        },
        "instrument_gate": {
            "label_free_training": True,
            "planted_feature_recovered": planted["PASS"],
            "at_least_one_representation_candidate": (
                representation_gate_passed
            ),
            "behavioral_preflight_required": True,
            "PASS": representation_gate_passed,
        },
        "behavioral_preflight_authorized": representation_gate_passed,
    }
    assert_label_free_payload(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:sparse_transcoder_training_v1",
        sys.argv,
        repo=REPO,
        status=(
            "EXPLORATORY"
            if representation_gate_passed
            else "INSTRUMENT_UNRESOLVED"
        ),
        config={
            "grid": result["grid"],
            "dictionary_split": result["dictionary_split"],
            "device": str(device),
            "batch": args.batch,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
        },
        seeds={
            "dictionary_initializations": list(INITIALIZATIONS),
            "checkpoint_family": list(EXPECTED_SEEDS),
        },
        notes=[
            "The process loaded only recorded label-free MLP inputs and writes.",
            "Classifier parameters, native margins, sequence labels, and external outcomes were unavailable.",
            "Configuration selection is deferred to the immutable behavioral preflight.",
        ],
    )
    for path in (
        protocol,
        addendum,
        activation_path,
        Path(report["arrays"]["path"]),
        native_unit_path,
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.extend([file_record(output), file_record(model_output)])
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "instrument_gate": result["instrument_gate"],
                "representation_candidates": [
                    key
                    for key, value in candidates.items()
                    if value["representation_gate"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
