#!/usr/bin/env python
"""Fit a signed J-SAE to only the base-to-GTEx-fine-tuned L5 change."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gc
import hashlib
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.circuits.jacobian_sae import edge_mass_count  # noqa: E402
from gfmcd.circuits.model_difference_jsae import (  # noqa: E402
    ModelDifferenceJSAEConfig,
    ModelDifferenceJacobianSAE,
)
from gfmcd.circuits.multiview_transcoder import (  # noqa: E402
    assert_label_free_payload,
)
from gfmcd.provenance import (  # noqa: E402
    ExperimentManifest,
    file_record,
    sha256_file,
)


def load_script(name: str) -> Any:
    path = REPO / "scripts/splicebert" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    spec.loader.exec_module(module)
    return module


INSTRUMENT = load_script("run_gtex_jacobian_sae_instrument.py")
LARGE = load_script("run_gtex_jacobian_sae_large.py")

TUNED_SEEDS = tuple(range(2020, 2025))
INITIALIZATIONS = (0, 1, 2)
CONDITIONS = {
    "shapley_deltaJ": 0.25,
    "shapley_no_deltaJ": 0.0,
}
INPUT_FEATURES = 512
OUTPUT_FEATURES = 128
ACTIVE_FEATURES = 16
STATE_BATCH = 64
PAIR_BATCH = 32
EPOCHS = 40
LEARNING_RATE = 5e-4
GRADIENT_CLIP = 1.0
FINITE_DIFFERENCE_DOSE = 0.01
GRAPH_RESOLUTION_FLOOR = 1e-5
AUTOGRAD_RELATIVE_TOLERANCE = 1e-5
FROZEN_COUNTS = LARGE.FROZEN_COUNTS
STATE_FAMILIES = (
    "clean",
    "target_shuffle",
    "sham_shuffle",
    "max_recode",
    "synonymous_single",
)
PAIR_FAMILIES = LARGE.PAIR_FAMILIES
ENDPOINTS = LARGE.ENDPOINTS
EPS = 1e-12


def stable_seed(seed: int, *parts: object) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    )


def state_family(variant_kind: str) -> str:
    if variant_kind.startswith("max_recode_draw"):
        return "max_recode"
    if variant_kind in STATE_FAMILIES:
        return variant_kind
    raise ValueError(f"unsupported state family {variant_kind}")


def initialization_seed(checkpoint_seed: int, initialization: int) -> int:
    if (
        checkpoint_seed not in TUNED_SEEDS
        or initialization not in INITIALIZATIONS
    ):
        raise ValueError("unknown checkpoint or initialization")
    return 5_000_000 + checkpoint_seed * 10_000 + initialization


def hierarchical_state_schedule(
    states: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Balance endpoint/family, then gene, then state."""

    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in indices:
        row = states[int(index)]
        grouped[
            (str(row["endpoint"]), state_family(str(row["variant_kind"])))
        ][str(row["gene_id"])].append(int(index))
    strata = sorted(grouped)
    expected = sorted(
        (endpoint, family)
        for endpoint in ENDPOINTS
        for family in STATE_FAMILIES
    )
    if strata != expected:
        raise RuntimeError("ordinary states lack a frozen endpoint/family cell")
    chosen: list[int] = []
    stratum_counts: dict[tuple[str, str], int] = defaultdict(int)
    gene_counts: dict[tuple[tuple[str, str], str], int] = defaultdict(int)
    for draw in range(draws):
        stratum_cycle, stratum_position = divmod(draw, len(strata))
        stratum_order = np.random.default_rng(
            stable_seed(seed, "state-stratum", stratum_cycle)
        ).permutation(len(strata))
        stratum = strata[int(stratum_order[stratum_position])]
        genes = sorted(grouped[stratum])
        occurrence = stratum_counts[stratum]
        gene_cycle, gene_position = divmod(occurrence, len(genes))
        gene_order = np.random.default_rng(
            stable_seed(seed, "state-gene", *stratum, gene_cycle)
        ).permutation(len(genes))
        gene = genes[int(gene_order[gene_position])]
        candidates = grouped[stratum][gene]
        gene_occurrence = gene_counts[(stratum, gene)]
        state_cycle, state_position = divmod(
            gene_occurrence, len(candidates)
        )
        state_order = np.random.default_rng(
            stable_seed(
                seed, "state-within-gene", *stratum, gene, state_cycle
            )
        ).permutation(len(candidates))
        chosen.append(candidates[int(state_order[state_position])])
        stratum_counts[stratum] += 1
        gene_counts[(stratum, gene)] += 1
    counts = {
        f"{endpoint}:{family}": int(stratum_counts[(endpoint, family)])
        for endpoint, family in strata
    }
    return np.asarray(chosen, dtype=np.int64), {
        "draws": draws,
        "counts_by_stratum": counts,
        "maximum_stratum_imbalance": max(counts.values()) - min(counts.values()),
        "PASS": max(counts.values()) - min(counts.values()) <= 1,
    }


def hierarchy_groups(
    states: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> dict[tuple[str, str], dict[str, list[int]]]:
    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for local_index, global_index in enumerate(indices):
        row = states[int(global_index)]
        grouped[
            (str(row["endpoint"]), state_family(str(row["variant_kind"])))
        ][str(row["gene_id"])].append(local_index)
    return grouped


def hierarchical_row_weights(
    states: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> np.ndarray:
    """Return endpoint/family/gene/state weights summing to one."""

    grouped = hierarchy_groups(states, indices)
    weights = np.zeros(len(indices), dtype=np.float64)
    n_strata = len(grouped)
    if n_strata == 0:
        raise ValueError("cannot weight an empty state set")
    for genes in grouped.values():
        n_genes = len(genes)
        for rows in genes.values():
            value = 1.0 / (n_strata * n_genes * len(rows))
            weights[np.asarray(rows, dtype=np.int64)] = value
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise RuntimeError("hierarchical row weights do not sum to one")
    return weights


def hierarchical_mean_square(
    values: torch.Tensor,
    states: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    allow_zero: bool = False,
) -> float:
    if values.ndim != 2 or len(values) != len(indices):
        raise ValueError("hierarchical power inputs do not align")
    grouped = hierarchy_groups(states, indices)
    stratum_values = []
    values64 = values.double()
    for genes in grouped.values():
        gene_values = [
            float(values64[rows].square().mean())
            for rows in genes.values()
        ]
        stratum_values.append(float(np.mean(gene_values)))
    power = float(np.mean(stratum_values))
    if not np.isfinite(power) or (power <= EPS and not allow_zero):
        raise RuntimeError("hierarchical shift power is non-finite or zero")
    return power


def hierarchical_fsee(
    target: torch.Tensor,
    predicted: torch.Tensor,
    states: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> float:
    if target.shape != predicted.shape or len(target) != len(indices):
        raise ValueError("FSEE arrays do not align")
    numerator = hierarchical_mean_square(
        target - predicted, states, indices, allow_zero=True
    )
    denominator = hierarchical_mean_square(target, states, indices)
    return 1.0 - numerator / denominator


def centered_fve(target: torch.Tensor, predicted: torch.Tensor) -> float:
    target64 = target.double()
    predicted64 = predicted.double()
    denominator = torch.sum(
        (target64 - target64.mean(dim=0, keepdim=True)).square()
    )
    if float(denominator) <= EPS:
        return float("-inf")
    return float(
        1.0 - torch.sum((target64 - predicted64).square()) / denominator
    )


def flattened_cosine(target: torch.Tensor, predicted: torch.Tensor) -> float:
    target64 = target.double().reshape(-1)
    predicted64 = predicted.double().reshape(-1)
    denominator = torch.linalg.vector_norm(
        target64
    ) * torch.linalg.vector_norm(predicted64)
    if float(denominator) <= EPS:
        return 0.0
    return float(torch.dot(target64, predicted64) / denominator)


def pair_hierarchical_power(
    values: torch.Tensor,
    pairs: Sequence[Mapping[str, Any]],
) -> float:
    by_stratum: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    values64 = values.double()
    for pair in pairs:
        difference = (
            values64[int(pair["changed"])]
            - values64[int(pair["reference"])]
        )
        by_stratum[(str(pair["endpoint"]), str(pair["family"]))][
            str(pair["gene_id"])
        ].append(float(difference.square().mean()))
    stratum_values = []
    for genes in by_stratum.values():
        stratum_values.append(
            float(
                np.mean(
                    [
                        np.mean(gene_values)
                        for gene_values in genes.values()
                    ]
                )
            )
        )
    power = float(np.mean(stratum_values))
    if not np.isfinite(power) or power <= EPS:
        raise RuntimeError("pair contrast power is non-finite or zero")
    return power


def run_mlp(
    raw_input: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    outputs = []
    parameters = {
        key: value.to(device) for key, value in weights.items()
    }
    with torch.no_grad():
        for start in range(0, len(raw_input), batch_size):
            values = raw_input[start : start + batch_size].to(device)
            hidden = F.gelu(
                F.linear(
                    values,
                    parameters["input_weight"],
                    parameters["input_bias"],
                ),
                approximate="none",
            )
            outputs.append(
                F.linear(
                    hidden,
                    parameters["output_weight"],
                    parameters["output_bias"],
                ).cpu()
            )
    return torch.cat(outputs)


def build_difference_targets(
    base_inputs: torch.Tensor,
    tuned_inputs: torch.Tensor,
    base_writes: torch.Tensor,
    tuned_writes: torch.Tensor,
    base_weights: Mapping[str, torch.Tensor],
    tuned_weights: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build exact four-cell targets while checking captured native writes."""

    y00_replay = run_mlp(base_inputs, base_weights, device=device)
    y11_replay = run_mlp(tuned_inputs, tuned_weights, device=device)
    y10 = run_mlp(tuned_inputs, base_weights, device=device)
    y01 = run_mlp(base_inputs, tuned_weights, device=device)
    replay = {
        "base_maximum_absolute_error": float(
            (y00_replay - base_writes).abs().max()
        ),
        "tuned_maximum_absolute_error": float(
            (y11_replay - tuned_writes).abs().max()
        ),
        "tolerance": INSTRUMENT.MLP_REPLAY_TOLERANCE,
    }
    replay["PASS"] = (
        max(
            replay["base_maximum_absolute_error"],
            replay["tuned_maximum_absolute_error"],
        )
        <= replay["tolerance"]
    )
    if not replay["PASS"]:
        raise RuntimeError("native MLP replay failed")

    # Captured diagonal writes are the native endpoints.  Cross cells come from
    # the exact frozen MLPs.  Perform the factorial arithmetic in float64.
    cells64 = {
        "00": base_writes.double(),
        "10": y10.double(),
        "01": y01.double(),
        "11": tuned_writes.double(),
    }
    state64 = 0.5 * (
        (cells64["10"] - cells64["00"])
        + (cells64["11"] - cells64["01"])
    )
    parameter64 = 0.5 * (
        (cells64["01"] - cells64["00"])
        + (cells64["11"] - cells64["10"])
    )
    total64 = cells64["11"] - cells64["00"]
    closure = float((state64 + parameter64 - total64).abs().max())
    if closure > 1e-10:
        raise RuntimeError("float64 write Shapley closure failed")
    targets = {
        "input": (tuned_inputs - base_inputs).float(),
        "state": state64.float(),
        "parameter": parameter64.float(),
        "total": total64.float(),
    }
    interaction = (
        cells64["11"] - cells64["10"] - cells64["01"] + cells64["00"]
    )

    def energy_ratio(values: torch.Tensor) -> float:
        denominator = torch.sum(total64.square()).clamp_min(EPS)
        return float(torch.sum(values.square()) / denominator)

    def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        denominator = (
            torch.linalg.vector_norm(left)
            * torch.linalg.vector_norm(right)
        ).clamp_min(EPS)
        return float(torch.sum(left * right) / denominator)

    report = {
        "native_replay": replay,
        "write_shapley_closure_maximum_absolute_error_float64": closure,
        "state_energy_over_total_energy": energy_ratio(state64),
        "parameter_energy_over_total_energy": energy_ratio(parameter64),
        "interaction_energy_over_total_energy": energy_ratio(interaction),
        "state_parameter_cosine": cosine(state64, parameter64),
        "tuned_path": {
            "state_energy_over_total_energy": energy_ratio(
                cells64["11"] - cells64["01"]
            ),
            "parameter_energy_over_total_energy": energy_ratio(
                cells64["01"] - cells64["00"]
            ),
        },
        "base_path": {
            "state_energy_over_total_energy": energy_ratio(
                cells64["10"] - cells64["00"]
            ),
            "parameter_energy_over_total_energy": energy_ratio(
                cells64["11"] - cells64["10"]
            ),
        },
        "PASS": replay["PASS"] and closure <= 1e-10,
    }
    return targets, report


def build_model(
    base_weights: Mapping[str, torch.Tensor],
    tuned_weights: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    states: Sequence[Mapping[str, Any]],
    train_indices: np.ndarray,
    *,
    device: torch.device,
) -> ModelDifferenceJacobianSAE:
    input_power = hierarchical_mean_square(
        targets["input"][train_indices], states, train_indices
    )
    output_power = hierarchical_mean_square(
        targets["total"][train_indices], states, train_indices
    )
    config = ModelDifferenceJSAEConfig(
        input_dim=512,
        output_dim=512,
        mlp_dim=2048,
        input_features=INPUT_FEATURES,
        output_features=OUTPUT_FEATURES,
        active_features=ACTIVE_FEATURES,
    )
    return ModelDifferenceJacobianSAE(
        config,
        base_weights=dict(base_weights),
        tuned_weights=dict(tuned_weights),
        input_scale=torch.tensor(math.sqrt(input_power)),
        output_scale=torch.tensor(math.sqrt(output_power)),
    ).to(device)


def normalized_targets(
    model: ModelDifferenceJacobianSAE,
    targets: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "input": targets["input"] / model.input_scale.detach().cpu(),
        "state": targets["state"] / model.output_scale.detach().cpu(),
        "parameter": targets["parameter"] / model.output_scale.detach().cpu(),
        "total": targets["total"] / model.output_scale.detach().cpu(),
    }


def reconstruction_powers(
    model: ModelDifferenceJacobianSAE,
    targets: Mapping[str, torch.Tensor],
    states: Sequence[Mapping[str, Any]],
    train_indices: np.ndarray,
    train_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    normalized = normalized_targets(model, targets)
    return {
        "level": {
            key: hierarchical_mean_square(
                values[train_indices], states, train_indices
            )
            for key, values in normalized.items()
        },
        "contrast": {
            key: pair_hierarchical_power(values, train_pairs)
            for key, values in normalized.items()
        },
    }


def reconstruct_batch(
    model: ModelDifferenceJacobianSAE,
    targets: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = model.input_scale.device
    input_values = targets["input"][indices].to(device)
    state_values = targets["state"][indices].to(device)
    parameter_values = targets["parameter"][indices].to(device)
    return model.encode_difference(
        input_values, state_values, parameter_values
    )


def normalized_reconstruction_loss(
    run: Mapping[str, torch.Tensor],
    powers: Mapping[str, float],
) -> torch.Tensor:
    errors = {
        "input": run["reconstructed_input"] - run["normalized_input"],
        "state": run["reconstructed_state"] - run["normalized_state"],
        "parameter": (
            run["reconstructed_parameter"] - run["normalized_parameter"]
        ),
        "total": run["reconstructed_total"] - run["normalized_total"],
    }
    return torch.stack(
        [
            error.square().mean() / float(powers[key])
            for key, error in errors.items()
        ]
    ).mean()


def contrast_loss(
    model: ModelDifferenceJacobianSAE,
    targets: Mapping[str, torch.Tensor],
    pair_indices: np.ndarray,
    powers: Mapping[str, float],
) -> torch.Tensor:
    reference = torch.as_tensor(pair_indices[:, 0], dtype=torch.long)
    changed = torch.as_tensor(pair_indices[:, 1], dtype=torch.long)
    reference_run = reconstruct_batch(model, targets, reference)
    changed_run = reconstruct_batch(model, targets, changed)
    keys = {
        "input": ("normalized_input", "reconstructed_input"),
        "state": ("normalized_state", "reconstructed_state"),
        "parameter": ("normalized_parameter", "reconstructed_parameter"),
        "total": ("normalized_total", "reconstructed_total"),
    }
    losses = []
    for name, (target_key, prediction_key) in keys.items():
        target_delta = (
            changed_run[target_key] - reference_run[target_key]
        )
        predicted_delta = (
            changed_run[prediction_key] - reference_run[prediction_key]
        )
        losses.append(
            (predicted_delta - target_delta).square().mean()
            / float(powers[name])
        )
    return torch.stack(losses).mean()


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compact_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(("input_sae.", "output_sae."))
        or name in {"input_scale", "output_scale"}
    }


def train_condition(
    base_weights: Mapping[str, torch.Tensor],
    tuned_weights: Mapping[str, torch.Tensor],
    base_inputs: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    states: Sequence[Mapping[str, Any]],
    train_indices: np.ndarray,
    train_pairs: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    device: torch.device,
    seed: int,
) -> tuple[ModelDifferenceJacobianSAE, dict[str, Any]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    torch.manual_seed(seed)
    model = build_model(
        base_weights,
        tuned_weights,
        targets,
        states,
        train_indices,
        device=device,
    )
    initial_hash = state_hash(model)
    powers = reconstruction_powers(
        model, targets, states, train_indices, train_pairs
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999)
    )
    steps_per_epoch = math.ceil(len(train_indices) / STATE_BATCH)
    total_steps = EPOCHS * steps_per_epoch
    pair_schedule, pair_report = LARGE.balanced_pair_schedule(
        train_pairs,
        draws=EPOCHS * steps_per_epoch * PAIR_BATCH,
        seed=seed,
    )
    if not pair_report["PASS"]:
        raise RuntimeError("paired schedule failed")
    history = []
    global_step = 0
    for epoch in range(EPOCHS):
        ordinary_schedule, state_report = hierarchical_state_schedule(
            states,
            train_indices,
            draws=steps_per_epoch * STATE_BATCH,
            seed=seed + 10_000 * epoch,
        )
        if not state_report["PASS"]:
            raise RuntimeError("ordinary schedule failed")
        epoch_values: dict[str, list[float]] = defaultdict(list)
        model.train()
        for step in range(steps_per_epoch):
            selected = ordinary_schedule[
                step * STATE_BATCH : (step + 1) * STATE_BATCH
            ]
            pair_start = (epoch * steps_per_epoch + step) * PAIR_BATCH
            selected_pairs = pair_schedule[
                pair_start : pair_start + PAIR_BATCH
            ]
            selected_tensor = torch.as_tensor(selected, dtype=torch.long)
            run = reconstruct_batch(model, targets, selected_tensor)
            level = normalized_reconstruction_loss(run, powers["level"])
            contrast = contrast_loss(
                model, targets, selected_pairs, powers["contrast"]
            )
            graph_coefficient = INSTRUMENT.scheduled_jacobian_coefficient(
                CONDITIONS[condition], global_step, total_steps
            )
            base_values = base_inputs[selected].to(device)
            input_values = targets["input"][selected].to(device)
            graph = model.active_jacobian_shapley(
                base_values,
                input_values,
                run["input_indices"],
                run["output_indices"],
            )
            native_operator = model.active_operator_shapley(
                base_values, input_values, run["input_indices"]
            )
            decoded_operator = model.decoded_operator_shapley(
                graph, run["output_indices"]
            )
            total_operator_power = (
                native_operator["total"].square().mean().detach()
            )
            operator_terms = []
            for name in ("state", "parameter", "total"):
                native_power = (
                    native_operator[name].square().mean().detach()
                )
                denominator = torch.maximum(
                    native_power,
                    0.05 * total_operator_power,
                ).clamp_min(EPS)
                operator_terms.append(
                    (decoded_operator[name] - native_operator[name])
                    .square()
                    .mean()
                    / denominator
                )
            operator = torch.stack(operator_terms).mean()
            graph_penalty = (
                torch.linalg.vector_norm(
                    torch.stack(
                        [graph["state"], graph["parameter"]], dim=0
                    ),
                    dim=0,
                )
                / math.sqrt(2.0)
            ).mean()
            loss = (
                level
                + contrast
                + operator
                + graph_coefficient * graph_penalty
            )
            optimizer.param_groups[0]["lr"] = (
                INSTRUMENT.scheduled_learning_rate(global_step, total_steps)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            for key, value in (
                ("loss", loss),
                ("level", level),
                ("contrast", contrast),
                ("operator", operator),
                ("graph", graph_penalty),
            ):
                epoch_values[key].append(float(value.detach().cpu()))
            global_step += 1
        history.append(
            {
                "epoch": epoch,
                **{
                    key: float(np.mean(values))
                    for key, values in epoch_values.items()
                },
                "ordinary_schedule": state_report,
            }
        )
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == EPOCHS:
            print(
                f"condition={condition} epoch={epoch + 1}/{EPOCHS} "
                f"loss={history[-1]['loss']:.6f}",
                flush=True,
            )
    return model, {
        "condition": condition,
        "graph_coefficient": CONDITIONS[condition],
        "initialization_seed": seed,
        "initial_state_sha256": initial_hash,
        "fixed_epochs": EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "state_batch": STATE_BATCH,
        "pair_batch": PAIR_BATCH,
        "powers": powers,
        "pair_schedule": pair_report,
        "history": history,
        "auxiliary_loss_used": False,
    }


def distribution(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    if values.shape != weights.shape or values.ndim != 1:
        raise ValueError("weighted quantile arrays must be aligned vectors")
    if not 0 <= quantile <= 1 or np.any(weights < 0):
        raise ValueError("invalid weighted quantile inputs")
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    if cumulative[-1] <= 0:
        raise ValueError("weighted quantile has zero total weight")
    cumulative /= cumulative[-1]
    return float(ordered_values[np.searchsorted(cumulative, quantile)])


def weighted_distribution(
    values: Sequence[float] | np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if array.shape != weights.shape:
        raise ValueError("weighted distribution arrays do not align")
    normalized = weights / weights.sum()
    return {
        "mean": float(np.sum(array * normalized)),
        "median": weighted_quantile(array, normalized, 0.50),
        "p90": weighted_quantile(array, normalized, 0.90),
        "p99": weighted_quantile(array, normalized, 0.99),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "hierarchical_weighting": True,
    }


def effective_edge_count(graph: torch.Tensor) -> torch.Tensor:
    """Return inverse participation, with exact zero for zero graph mass."""

    if graph.ndim != 3:
        raise ValueError("graph must be [batch, output_features, input_features]")
    flat = graph.abs().flatten(1)
    mass = flat.sum(dim=1, keepdim=True)
    probabilities = flat / mass.clamp_min(EPS)
    effective = (
        1.0 / probabilities.square().sum(dim=1).clamp_min(EPS)
    )
    return torch.where(
        mass[:, 0] > EPS,
        effective,
        torch.zeros_like(effective),
    )


@torch.no_grad()
def evaluate_model(
    model: ModelDifferenceJacobianSAE,
    base_inputs: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    states: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    device = model.input_scale.device
    row_weights = hierarchical_row_weights(states, indices)
    target_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    predicted_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    cross_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    input_features = []
    output_state_features = []
    output_parameter_features = []
    input_frequency = torch.zeros(INPUT_FEATURES, dtype=torch.int64)
    output_frequency = torch.zeros(OUTPUT_FEATURES, dtype=torch.int64)
    input_l0 = []
    output_l0 = []
    e50 = []
    e90 = []
    e99 = []
    effective = []
    graph_rms = []
    graph_total_rms = []
    total_e50 = []
    total_e90 = []
    total_e99 = []
    total_effective = []
    positive = []
    operator_power: dict[str, list[float]] = defaultdict(list)
    operator_error: dict[str, list[float]] = defaultdict(list)
    graph_matrix = torch.zeros(
        OUTPUT_FEATURES, INPUT_FEATURES, device=device
    )
    closure_errors = []
    for start in range(0, len(indices), 128):
        selected = indices[start : start + 128]
        selected_tensor = torch.as_tensor(selected, dtype=torch.long)
        run = reconstruct_batch(model, targets, selected_tensor)
        base_values = base_inputs[selected].to(device)
        input_values = targets["input"][selected].to(device)
        graphs = model.active_jacobian_shapley(
            base_values,
            input_values,
            run["input_indices"],
            run["output_indices"],
        )
        group_graph = torch.sqrt(
            0.5
            * (
                graphs["state"].square()
                + graphs["parameter"].square()
            )
        )
        native_operator = model.active_operator_shapley(
            base_values, input_values, run["input_indices"]
        )
        decoded_operator = model.decoded_operator_shapley(
            graphs, run["output_indices"]
        )
        input_hat_raw = model.denormalize_input_delta(
            run["reconstructed_input"]
        )
        cross = model.write_shapley(base_values, input_hat_raw)
        mapping = {
            "input": ("normalized_input", "reconstructed_input"),
            "state": ("normalized_state", "reconstructed_state"),
            "parameter": (
                "normalized_parameter",
                "reconstructed_parameter",
            ),
            "total": ("normalized_total", "reconstructed_total"),
        }
        for name, (target_key, predicted_key) in mapping.items():
            target_values[name].append(run[target_key].cpu())
            predicted_values[name].append(run[predicted_key].cpu())
        for name in ("state", "parameter", "total"):
            cross_values[name].append(
                model.normalize_output_delta(cross[name]).cpu()
            )
        input_group = run["input_activations"].abs()
        output_group = torch.sqrt(
            run["state_activations"].square()
            + run["parameter_activations"].square()
        )
        input_frequency += (input_group > 0).sum(dim=0).cpu()
        output_frequency += (output_group > 0).sum(dim=0).cpu()
        input_nonzero = run["normalized_input"].square().sum(dim=1) > EPS
        output_nonzero = (
            run["normalized_state"].square().sum(dim=1)
            + run["normalized_parameter"].square().sum(dim=1)
        ) > EPS
        input_l0.extend(
            (input_group[input_nonzero] > 0)
            .sum(dim=1)
            .cpu()
            .numpy()
            .tolist()
        )
        output_l0.extend(
            (output_group[output_nonzero] > 0)
            .sum(dim=1)
            .cpu()
            .numpy()
            .tolist()
        )
        input_features.append(run["input_activations"].cpu())
        output_state_features.append(run["state_activations"].cpu())
        output_parameter_features.append(
            run["parameter_activations"].cpu()
        )
        e50.extend(edge_mass_count(group_graph, fraction=0.50).cpu().tolist())
        e90.extend(edge_mass_count(group_graph, fraction=0.90).cpu().tolist())
        e99.extend(edge_mass_count(group_graph, fraction=0.99).cpu().tolist())
        effective_batch = effective_edge_count(group_graph)
        effective.extend(effective_batch.cpu().tolist())
        total_e50.extend(
            edge_mass_count(graphs["total"], fraction=0.50).cpu().tolist()
        )
        total_e90.extend(
            edge_mass_count(graphs["total"], fraction=0.90).cpu().tolist()
        )
        total_e99.extend(
            edge_mass_count(graphs["total"], fraction=0.99).cpu().tolist()
        )
        total_effective_batch = effective_edge_count(graphs["total"])
        total_effective.extend(total_effective_batch.cpu().tolist())
        graph_rms.extend(
            group_graph.square()
            .mean(dim=(-2, -1))
            .sqrt()
            .cpu()
            .tolist()
        )
        graph_total_rms.extend(
            graphs["total"]
            .square()
            .mean(dim=(-2, -1))
            .sqrt()
            .cpu()
            .tolist()
        )
        positive.extend(
            (graphs["total"] > 0)
            .float()
            .mean(dim=(-2, -1))
            .cpu()
            .tolist()
        )
        output_grid = run["output_indices"][:, :, None].expand_as(group_graph)
        input_grid = run["input_indices"][:, None, :].expand_as(group_graph)
        batch_weights = torch.as_tensor(
            row_weights[start : start + len(selected)],
            dtype=group_graph.dtype,
            device=device,
        )
        graph_matrix.index_put_(
            (output_grid.reshape(-1), input_grid.reshape(-1)),
            (
                group_graph
                * batch_weights[:, None, None]
            ).reshape(-1),
            accumulate=True,
        )
        for name in ("state", "parameter", "total"):
            operator_power[name].extend(
                native_operator[name]
                .square()
                .mean(dim=(-2, -1))
                .cpu()
                .tolist()
            )
            operator_error[name].extend(
                (
                    decoded_operator[name] - native_operator[name]
                )
                .square()
                .mean(dim=(-2, -1))
                .cpu()
                .tolist()
            )
        closure_errors.append(
            float(
                (
                    graphs["state"]
                    + graphs["parameter"]
                    - graphs["total"]
                )
                .abs()
                .max()
                .cpu()
            )
        )
    all_target = {
        key: torch.cat(values) for key, values in target_values.items()
    }
    all_predicted = {
        key: torch.cat(values) for key, values in predicted_values.items()
    }
    all_cross = {
        key: torch.cat(values) for key, values in cross_values.items()
    }
    level = {
        key: {
            "fsee": hierarchical_fsee(
                all_target[key], all_predicted[key], states, indices
            ),
            "centered_fve": centered_fve(
                all_target[key], all_predicted[key]
            ),
            "cosine": flattened_cosine(
                all_target[key], all_predicted[key]
            ),
        }
        for key in all_target
    }
    cross_replay = {
        key: {
            "fsee": hierarchical_fsee(
                all_target[key], all_cross[key], states, indices
            ),
            "cosine": flattened_cosine(all_target[key], all_cross[key]),
        }
        for key in all_cross
    }
    input_frequency_array = input_frequency.numpy()
    output_frequency_array = output_frequency.numpy()
    operator_power_weighted = {
        key: float(np.sum(np.asarray(values) * row_weights))
        for key, values in operator_power.items()
    }
    operator_error_weighted = {
        key: float(np.sum(np.asarray(values) * row_weights))
        for key, values in operator_error.items()
    }
    total_operator_power = operator_power_weighted["total"]
    operator_report = {
        key: {
            "fsee": (
                1.0 - operator_error_weighted[key] / power
                if power > EPS
                else float("-inf")
            ),
            "native_power": power,
            "error_power": operator_error_weighted[key],
            "native_power_fraction_of_total": (
                power / total_operator_power
                if total_operator_power > EPS
                else 0.0
            ),
            "error_power_fraction_of_total": (
                operator_error_weighted[key] / total_operator_power
                if total_operator_power > EPS
                else float("inf")
            ),
            "hierarchical_weighting": True,
        }
        for key, power in operator_power_weighted.items()
    }
    report = {
        "n_examples": len(indices),
        "level": level,
        "cross_replay": cross_replay,
        "operator_action": operator_report,
        "input_l0": distribution(input_l0),
        "output_l0": distribution(output_l0),
        "input_dead_feature_fraction": float(
            np.mean(input_frequency_array == 0)
        ),
        "output_dead_feature_fraction": float(
            np.mean(output_frequency_array == 0)
        ),
        "input_feature_frequency": input_frequency_array.tolist(),
        "output_feature_frequency": output_frequency_array.tolist(),
        "component_union_graph": {
            "definition": "sqrt((Phi_J_state^2 + Phi_J_parameter^2) / 2)",
            "E50": weighted_distribution(e50, row_weights),
            "E90": weighted_distribution(e90, row_weights),
            "E99": weighted_distribution(e99, row_weights),
            "effective_active_edge_count": weighted_distribution(
                effective, row_weights
            ),
            "normalized_rms": weighted_distribution(graph_rms, row_weights),
            "positive_total_edge_fraction": weighted_distribution(
                positive, row_weights
            ),
            "active_edge_count": ACTIVE_FEATURES**2,
        },
        "signed_total_deltaJ": {
            "definition": "Phi_J_state + Phi_J_parameter = J_11 - J_00",
            "E50": weighted_distribution(total_e50, row_weights),
            "E90": weighted_distribution(total_e90, row_weights),
            "E99": weighted_distribution(total_e99, row_weights),
            "effective_active_edge_count": weighted_distribution(
                total_effective, row_weights
            ),
            "normalized_rms": weighted_distribution(
                graph_total_rms, row_weights
            ),
            "active_edge_count": ACTIVE_FEATURES**2,
        },
        "float32_jacobian_shapley_closure_maximum_absolute_error": max(
            closure_errors
        ),
    }
    fingerprints = {
        "input_features": torch.cat(input_features).numpy(),
        "output_features": torch.cat(
            [
                torch.cat(output_state_features),
                torch.cat(output_parameter_features),
            ]
        ).numpy(),
        "input_sample_weights": row_weights,
        "output_sample_weights": np.concatenate(
            [0.5 * row_weights, 0.5 * row_weights]
        ),
        "input_decoder": model.input_sae.decoder().cpu().numpy().T,
        "output_decoder": model.output_sae.decoder().cpu().numpy().T,
        "graph": graph_matrix.cpu().numpy(),
    }
    return report, fingerprints


@torch.no_grad()
def contrast_fsee(
    model: ModelDifferenceJacobianSAE,
    targets: Mapping[str, torch.Tensor],
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model.eval()
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[(str(pair["endpoint"]), str(pair["family"]))].append(pair)
    result = {}
    keys = {
        "input": ("normalized_input", "reconstructed_input"),
        "state": ("normalized_state", "reconstructed_state"),
        "parameter": ("normalized_parameter", "reconstructed_parameter"),
        "total": ("normalized_total", "reconstructed_total"),
    }
    for stratum, rows in sorted(grouped.items()):
        by_gene: dict[
            str, dict[str, list[torch.Tensor]]
        ] = defaultdict(lambda: defaultdict(list))
        for start in range(0, len(rows), PAIR_BATCH):
            batch = rows[start : start + PAIR_BATCH]
            reference = torch.as_tensor(
                [int(row["reference"]) for row in batch], dtype=torch.long
            )
            changed = torch.as_tensor(
                [int(row["changed"]) for row in batch], dtype=torch.long
            )
            reference_run = reconstruct_batch(model, targets, reference)
            changed_run = reconstruct_batch(model, targets, changed)
            for row_index, row in enumerate(batch):
                gene = str(row["gene_id"])
                for name, (target_key, prediction_key) in keys.items():
                    target_delta = (
                        changed_run[target_key][row_index]
                        - reference_run[target_key][row_index]
                    ).cpu()
                    predicted_delta = (
                        changed_run[prediction_key][row_index]
                        - reference_run[prediction_key][row_index]
                    ).cpu()
                    by_gene[gene][f"{name}:target"].append(target_delta)
                    by_gene[gene][f"{name}:predicted"].append(
                        predicted_delta
                    )
        cell = {}
        for name in keys:
            numerators = []
            denominators = []
            for gene_values in by_gene.values():
                target = torch.stack(gene_values[f"{name}:target"]).double()
                predicted = torch.stack(
                    gene_values[f"{name}:predicted"]
                ).double()
                numerators.append(float((target - predicted).square().mean()))
                denominators.append(float(target.square().mean()))
            denominator = float(np.mean(denominators))
            cell[name] = {
                "fsee": (
                    1.0 - float(np.mean(numerators)) / denominator
                    if denominator > EPS
                    else float("-inf")
                ),
                "power": denominator,
            }
        result[f"{stratum[0]}:{stratum[1]}"] = {
            "n_pairs": len(rows),
            "n_genes": len(by_gene),
            **cell,
        }
    return {"by_stratum": result}


def decoder_invariants(
    model: ModelDifferenceJacobianSAE,
) -> dict[str, Any]:
    with torch.no_grad():
        input_norm_error = float(
            (model.input_sae.decoder().norm(dim=1) - 1)
            .abs()
            .max()
            .cpu()
        )
        output_norm_error = float(
            (model.output_sae.decoder().norm(dim=1) - 1)
            .abs()
            .max()
            .cpu()
        )
        zero_input = torch.zeros(
            2, model.config.input_dim, device=model.input_scale.device
        )
        zero_output = torch.zeros(
            2, model.config.output_dim, device=model.input_scale.device
        )
        zero_run = model.encode_difference(
            zero_input, zero_output, zero_output
        )
        zero_error = max(
            float(zero_run["reconstructed_input"].abs().max().cpu()),
            float(zero_run["reconstructed_total"].abs().max().cpu()),
        )
    return {
        "maximum_input_decoder_norm_error": input_norm_error,
        "maximum_output_decoder_norm_error": output_norm_error,
        "maximum_zero_shift_reconstruction_error": zero_error,
        "decoder_norm_tolerance": 1e-5,
        "zero_shift_tolerance": 1e-12,
        "no_encoder_bias": not hasattr(model.input_sae, "encoder_bias")
        and not hasattr(model.output_sae, "encoder_bias"),
        "no_decoder_bias": not hasattr(model.input_sae, "decoder_bias")
        and not hasattr(model.output_sae, "decoder_bias"),
        "PASS": (
            max(input_norm_error, output_norm_error) <= 1e-5
            and zero_error <= 1e-12
            and not hasattr(model.input_sae, "encoder_bias")
            and not hasattr(model.output_sae, "encoder_bias")
            and not hasattr(model.input_sae, "decoder_bias")
            and not hasattr(model.output_sae, "decoder_bias")
        ),
    }


def autograd_control(
    model: ModelDifferenceJacobianSAE,
    base_inputs: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    indices: np.ndarray,
    *,
    examples: int = 4,
) -> dict[str, Any]:
    # Mechanical agreement is evaluated in float64 on CPU so accelerator
    # float32 rounding is not mistaken for an algebra error.
    work_model = copy.deepcopy(model).cpu().double()
    work_model.eval()
    device = torch.device("cpu")
    errors: dict[str, list[float]] = defaultdict(list)
    relative: dict[str, list[float]] = defaultdict(list)
    for global_index in indices[:examples]:
        index = int(global_index)
        base_input = base_inputs[index : index + 1].to(
            device=device, dtype=torch.float64
        )
        input_delta = targets["input"][index : index + 1].to(
            device=device, dtype=torch.float64
        )
        writes = work_model.write_shapley(base_input, input_delta)
        encoded = work_model.encode_difference(
            input_delta, writes["state"], writes["parameter"]
        )
        input_indices = encoded["input_indices"]
        output_indices = encoded["output_indices"]
        analytic = work_model.active_jacobian_shapley(
            base_input, input_delta, input_indices, output_indices
        )
        directions = (
            work_model.input_sae.decoder()[input_indices[0]]
            * work_model.input_scale
        ).detach()
        observed_cells = {}
        for state, raw in (
            ("0", base_input),
            ("1", base_input + input_delta),
        ):
            for view, model_name in (("0", "base"), ("1", "tuned")):
                key = state + view

                def output_features(
                    coefficients: torch.Tensor,
                    *,
                    raw_input: torch.Tensor = raw,
                    selected_view: str = model_name,
                ) -> torch.Tensor:
                    perturbed = raw_input + coefficients @ directions
                    output = work_model.mlp(perturbed, view=selected_view)
                    assert isinstance(output, torch.Tensor)
                    features = (
                        output
                        / work_model.output_scale
                        @ work_model.output_sae.encoder_weight
                    )
                    return features[0, output_indices[0]]

                coefficients = torch.zeros(
                    model.config.active_features,
                    dtype=base_input.dtype,
                    device=device,
                    requires_grad=True,
                )
                observed_cells[key] = torch.autograd.functional.jacobian(
                    output_features, coefficients
                ).detach()
        observed = work_model.shapley_from_cells(observed_cells)
        for name in ("state", "parameter", "total"):
            difference = observed[name] - analytic[name][0]
            errors[name].append(
                float(difference.abs().max().detach().cpu())
            )
            relative[name].append(
                float(
                    (
                        torch.linalg.vector_norm(difference)
                        / torch.linalg.vector_norm(observed[name]).clamp_min(
                            1e-8
                        )
                    )
                    .detach()
                    .cpu()
                )
            )
    del work_model
    maximum_relative = max(max(values) for values in relative.values())
    return {
        "examples": min(examples, len(indices)),
        "maximum_absolute_error_by_block": {
            key: max(values) for key, values in errors.items()
        },
        "maximum_relative_frobenius_error_by_block": {
            key: max(values) for key, values in relative.items()
        },
        "maximum_relative_frobenius_error": maximum_relative,
        "relative_tolerance": AUTOGRAD_RELATIVE_TOLERANCE,
        "PASS": maximum_relative <= AUTOGRAD_RELATIVE_TOLERANCE,
    }


@torch.no_grad()
def finite_difference_control(
    model: ModelDifferenceJacobianSAE,
    base_inputs: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    indices: np.ndarray,
    *,
    examples: int = 8,
) -> dict[str, Any]:
    model.eval()
    device = model.input_scale.device
    analytic_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    observed_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    for global_index in indices[:examples]:
        index = int(global_index)
        base_input = base_inputs[index : index + 1].to(device)
        input_delta = targets["input"][index : index + 1].to(device)
        writes = model.write_shapley(base_input, input_delta)
        encoded = model.encode_difference(
            input_delta, writes["state"], writes["parameter"]
        )
        input_indices = encoded["input_indices"]
        output_indices = encoded["output_indices"]
        analytic = model.active_jacobian_shapley(
            base_input, input_delta, input_indices, output_indices
        )
        directions = (
            model.input_sae.decoder()[input_indices[0]] * model.input_scale
        )
        observed_cells = {}
        for state, raw in (
            ("0", base_input),
            ("1", base_input + input_delta),
        ):
            for view, model_name in (("0", "base"), ("1", "tuned")):
                key = state + view
                baseline_output = model.mlp(raw, view=model_name)
                assert isinstance(baseline_output, torch.Tensor)
                perturbed_output = model.mlp(
                    raw + FINITE_DIFFERENCE_DOSE * directions,
                    view=model_name,
                )
                assert isinstance(perturbed_output, torch.Tensor)
                output_difference = (
                    (perturbed_output - baseline_output)
                    / FINITE_DIFFERENCE_DOSE
                    / model.output_scale
                )
                # [k_in, k_out] -> [k_out, k_in]
                observed_cells[key] = (
                    output_difference
                    @ model.output_sae.encoder_weight[:, output_indices[0]]
                ).T
        observed = model.shapley_from_cells(observed_cells)
        for name in ("state", "parameter", "total"):
            analytic_values[name].append(analytic[name][0].cpu())
            observed_values[name].append(observed[name].cpu())
    blocks = {}
    for name in ("state", "parameter", "total"):
        analytic = torch.stack(analytic_values[name]).double().reshape(-1)
        observed = torch.stack(observed_values[name]).double().reshape(-1)
        difference = observed - analytic
        analytic_rms = float(analytic.square().mean().sqrt())
        observed_rms = float(observed.square().mean().sqrt())
        resolved = analytic_rms >= GRAPH_RESOLUTION_FLOOR
        correlation = float(
            torch.dot(
                observed - observed.mean(), analytic - analytic.mean()
            )
            / (
                torch.linalg.vector_norm(observed - observed.mean())
                * torch.linalg.vector_norm(analytic - analytic.mean())
            ).clamp_min(EPS)
        )
        eligible = analytic.abs() >= GRAPH_RESOLUTION_FLOOR
        sign = (
            float(
                (torch.sign(analytic[eligible]) == torch.sign(observed[eligible]))
                .double()
                .mean()
            )
            if bool(eligible.any())
            else 0.0
        )
        relative_error = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(observed).clamp_min(EPS)
        )
        absolute_rms_error = float(difference.square().mean().sqrt())
        passed = (
            correlation >= 0.95
            and sign >= 0.95
            and relative_error <= 0.10
            and int(eligible.sum()) > 0
            if resolved
            else (
                absolute_rms_error <= GRAPH_RESOLUTION_FLOOR
                and observed_rms <= 2 * GRAPH_RESOLUTION_FLOOR
            )
        )
        blocks[name] = {
            "correlation": correlation,
            "eligible_edge_sign_agreement": sign,
            "relative_frobenius_error": relative_error,
            "absolute_rms_error": absolute_rms_error,
            "analytic_rms": analytic_rms,
            "observed_rms": observed_rms,
            "resolved": resolved,
            "eligible_edges": int(eligible.sum()),
            "PASS": passed,
        }
    return {
        "examples": min(examples, len(indices)),
        "dose": FINITE_DIFFERENCE_DOSE,
        "resolution_floor": GRAPH_RESOLUTION_FLOOR,
        "blocks": blocks,
        "PASS": all(block["PASS"] for block in blocks.values()),
    }


def stratified_control_indices(
    states: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
) -> np.ndarray:
    """Choose one deterministic gene/state from every endpoint/family cell."""

    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in indices:
        row = states[int(index)]
        grouped[
            (str(row["endpoint"]), state_family(str(row["variant_kind"])))
        ][str(row["gene_id"])].append(int(index))
    expected = {
        (endpoint, family)
        for endpoint in ENDPOINTS
        for family in STATE_FAMILIES
    }
    if set(grouped) != expected:
        raise RuntimeError("control rows lack an endpoint/family stratum")
    selected = []
    for stratum in sorted(grouped):
        gene = sorted(grouped[stratum])[0]
        selected.append(sorted(grouped[stratum][gene])[0])
    return np.asarray(selected, dtype=np.int64)


def feature_match(
    left_features: np.ndarray,
    right_features: np.ndarray,
    left_decoder: np.ndarray,
    right_decoder: np.ndarray,
    sample_weights: np.ndarray,
) -> dict[str, Any]:
    """Sign-invariant reciprocal matching for signed sparse features."""

    if left_features.shape != right_features.shape:
        raise ValueError("paired feature fingerprints must have matching shapes")
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if sample_weights.shape != (left_features.shape[0],):
        raise ValueError("feature-match weights do not align")
    sample_weights = sample_weights / sample_weights.sum()
    left_density = np.sum(
        (np.abs(left_features) > 0) * sample_weights[:, None], axis=0
    )
    right_density = np.sum(
        (np.abs(right_features) > 0) * sample_weights[:, None], axis=0
    )
    left_live = np.flatnonzero(left_density >= 0.001)
    right_live = np.flatnonzero(right_density >= 0.001)
    if not len(left_live) or not len(right_live):
        return {
            "n_left_eligible": len(left_live),
            "n_right_eligible": len(right_live),
            "reciprocal_match_fraction": 0.0,
            "reciprocal_score_mean": 0.0,
            "PASS": False,
        }
    left = left_features[:, left_live].astype(np.float64)
    right = right_features[:, right_live].astype(np.float64)
    left = left - np.sum(
        left * sample_weights[:, None], axis=0, keepdims=True
    )
    right = right - np.sum(
        right * sample_weights[:, None], axis=0, keepdims=True
    )
    left /= np.sqrt(
        np.sum(sample_weights[:, None] * left**2, axis=0, keepdims=True)
    ).clip(min=EPS)
    right /= np.sqrt(
        np.sum(sample_weights[:, None] * right**2, axis=0, keepdims=True)
    ).clip(min=EPS)
    activation = (left * sample_weights[:, None]).T @ right
    left_d = left_decoder[:, left_live].astype(np.float64)
    right_d = right_decoder[:, right_live].astype(np.float64)
    left_d /= np.linalg.norm(left_d, axis=0, keepdims=True).clip(min=EPS)
    right_d /= np.linalg.norm(right_d, axis=0, keepdims=True).clip(min=EPS)
    writer = left_d.T @ right_d
    # One common feature sign must align both detector and writer.  Independent
    # absolute values would falsely accept inconsistent sign gauges.
    combined = 0.5 * np.abs(activation + writer)
    forward = combined.argmax(axis=1)
    reverse = combined.argmax(axis=0)
    reciprocal = np.asarray(
        [reverse[target] == source for source, target in enumerate(forward)]
    )
    scores = combined[np.arange(len(forward)), forward][reciprocal]
    fraction = float(np.mean(reciprocal))
    mean_score = float(np.mean(scores)) if len(scores) else 0.0
    return {
        "n_left_eligible": len(left_live),
        "n_right_eligible": len(right_live),
        "reciprocal_matches": int(reciprocal.sum()),
        "reciprocal_match_fraction": fraction,
        "reciprocal_score_mean": mean_score,
        "activation_similarity_mean": (
            float(
                np.mean(
                    activation[
                        np.arange(len(forward))[reciprocal],
                        forward[reciprocal],
                    ]
                )
            )
            if bool(reciprocal.any())
            else 0.0
        ),
        "writer_cosine_mean": (
            float(
                np.mean(
                    writer[
                        np.arange(len(forward))[reciprocal],
                        forward[reciprocal],
                    ]
                )
            )
            if bool(reciprocal.any())
            else 0.0
        ),
        "density_floor": 0.001,
        "sign_invariant": True,
        "common_sign_orientation": True,
        "hierarchical_sample_weighting": True,
        "PASS": fraction >= 0.25 and mean_score >= 0.50,
    }


def _toy_weights(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "input_weight": torch.randn(
            2048, 512, generator=generator
        )
        / math.sqrt(512),
        "input_bias": torch.randn(2048, generator=generator) / 20,
        "output_weight": torch.randn(
            512, 2048, generator=generator
        )
        / math.sqrt(2048),
        "output_bias": torch.randn(512, generator=generator) / 20,
    }


def aligned_planted_controls() -> dict[str, Any]:
    """Calibrate a known sparse edge, signed support, dense null, and floor."""

    config = ModelDifferenceJSAEConfig(
        input_dim=4,
        output_dim=4,
        mlp_dim=4,
        input_features=4,
        output_features=4,
        active_features=1,
    )
    base = {
        "input_weight": torch.eye(4),
        "input_bias": torch.zeros(4),
        "output_weight": torch.zeros(4, 4),
        "output_bias": torch.zeros(4),
    }

    def planted_model(output_amplitude: float) -> ModelDifferenceJacobianSAE:
        tuned = {key: value.clone() for key, value in base.items()}
        tuned["output_weight"][0, 0] = output_amplitude
        model = ModelDifferenceJacobianSAE(
            config,
            base_weights=base,
            tuned_weights=tuned,
            input_scale=torch.tensor(1.0),
            output_scale=torch.tensor(1.0),
        )
        with torch.no_grad():
            model.input_sae.encoder_weight.copy_(torch.eye(4))
            model.input_sae.decoder_raw.copy_(torch.eye(4))
            model.output_sae.encoder_weight.copy_(torch.eye(4))
            model.output_sae.decoder_raw.copy_(torch.eye(4))
        return model

    model = planted_model(2.0)
    base_input = torch.zeros(1, 4)
    input_delta = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        writes = model.write_shapley(base_input, input_delta)
        encoded = model.encode_difference(
            input_delta, writes["state"], writes["parameter"]
        )
        graphs = model.active_jacobian_shapley(
            base_input,
            input_delta,
            encoded["input_indices"],
            encoded["output_indices"],
        )
        native_operator = model.active_operator_shapley(
            base_input, input_delta, encoded["input_indices"]
        )
        decoded_operator = model.decoded_operator_shapley(
            graphs, encoded["output_indices"]
        )
        known_location = (
            int(encoded["input_indices"][0, 0]) == 0
            and int(encoded["output_indices"][0, 0]) == 0
        )
        known_sign = float(graphs["total"][0, 0, 0]) > 0
        operator_error = float(
            (
                decoded_operator["total"] - native_operator["total"]
            )
            .abs()
            .max()
        )

        negative_delta = -input_delta
        negative_writes = model.write_shapley(base_input, negative_delta)
        negative_encoded = model.encode_difference(
            negative_delta,
            negative_writes["state"],
            negative_writes["parameter"],
        )
        signed_input = float(
            negative_encoded["input_activations"][0, 0]
        ) < 0
        common_output_support = torch.equal(
            torch.nonzero(
                negative_encoded["state_activations"][0], as_tuple=False
            ),
            torch.nonzero(
                negative_encoded["parameter_activations"][0],
                as_tuple=False,
            ),
        )

    derivative = float(
        ModelDifferenceJacobianSAE.gelu_derivative(torch.tensor(0.1))
    )
    amplitude_results = {}
    for label, target_value in (
        ("below_floor", 0.5 * GRAPH_RESOLUTION_FLOOR),
        ("above_floor", 2.0 * GRAPH_RESOLUTION_FLOOR),
    ):
        amplitude_model = planted_model(target_value / derivative)
        with torch.no_grad():
            amplitude_writes = amplitude_model.write_shapley(
                base_input, input_delta
            )
            amplitude_encoded = amplitude_model.encode_difference(
                input_delta,
                amplitude_writes["state"],
                amplitude_writes["parameter"],
            )
            amplitude_graph = amplitude_model.active_jacobian_shapley(
                base_input,
                input_delta,
                amplitude_encoded["input_indices"],
                amplitude_encoded["output_indices"],
            )["total"]
            observed = float(amplitude_graph.abs().mean())
        amplitude_results[label] = {
            "target": target_value,
            "observed": observed,
            "relative_error": abs(observed - target_value) / target_value,
        }

    dense_config = ModelDifferenceJSAEConfig(
        input_dim=4,
        output_dim=4,
        mlp_dim=4,
        input_features=4,
        output_features=4,
        active_features=4,
    )
    dense_tuned = {key: value.clone() for key, value in base.items()}
    dense_tuned["output_weight"] = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    )
    dense = ModelDifferenceJacobianSAE(
        dense_config,
        base_weights=base,
        tuned_weights=dense_tuned,
        input_scale=torch.tensor(1.0),
        output_scale=torch.tensor(1.0),
    )
    with torch.no_grad():
        dense.input_sae.encoder_weight.copy_(torch.eye(4))
        dense.input_sae.decoder_raw.copy_(torch.eye(4))
        dense.output_sae.encoder_weight.copy_(torch.eye(4))
        dense.output_sae.decoder_raw.copy_(torch.eye(4))
        dense_input_delta = torch.full((1, 4), 0.1)
        dense_writes = dense.write_shapley(base_input, dense_input_delta)
        dense_encoded = dense.encode_difference(
            dense_input_delta,
            dense_writes["state"],
            dense_writes["parameter"],
        )
        dense_graph = dense.active_jacobian_shapley(
            base_input,
            dense_input_delta,
            dense_encoded["input_indices"],
            dense_encoded["output_indices"],
        )["total"]
        dense_e90 = int(edge_mass_count(dense_graph, fraction=0.90)[0])

    checks = {
        "known_edge_location": known_location,
        "known_edge_sign": known_sign,
        "known_operator_action_replay": operator_error <= 1e-7,
        "signed_input_activation": signed_input,
        "common_output_group_support": common_output_support,
        "below_floor_is_below": (
            amplitude_results["below_floor"]["observed"]
            < GRAPH_RESOLUTION_FLOOR
        ),
        "above_floor_is_above": (
            amplitude_results["above_floor"]["observed"]
            > GRAPH_RESOLUTION_FLOOR
        ),
        "amplitude_recovery": all(
            row["relative_error"] <= 1e-4
            for row in amplitude_results.values()
        ),
        "dense_control_not_one_edge": dense_e90 >= 12,
    }
    return {
        "checks": checks,
        "known_total_edge": float(graphs["total"][0, 0, 0]),
        "known_operator_maximum_absolute_error": operator_error,
        "amplitude_sensitivity": amplitude_results,
        "dense_control_E90": dense_e90,
        "PASS": all(checks.values()),
    }


def synthetic_preflight(device: torch.device) -> dict[str, Any]:
    """Run positive, negative, sensitivity, and invariant controls."""

    aligned = aligned_planted_controls()
    torch.manual_seed(771_204)
    base_weights = _toy_weights(771_204)
    tuned_weights = {
        key: value.clone() for key, value in base_weights.items()
    }
    tuned_weights["output_weight"][0, :32] += 0.10
    config = ModelDifferenceJSAEConfig(
        input_dim=512,
        output_dim=512,
        mlp_dim=2048,
        input_features=INPUT_FEATURES,
        output_features=OUTPUT_FEATURES,
        active_features=ACTIVE_FEATURES,
    )
    model = ModelDifferenceJacobianSAE(
        config,
        base_weights=base_weights,
        tuned_weights=tuned_weights,
        input_scale=torch.tensor(1.0),
        output_scale=torch.tensor(1.0),
    ).to(device)
    base_input = torch.randn(2, 512)
    input_delta = torch.randn(2, 512) * 0.05
    with torch.no_grad():
        writes = model.write_shapley(
            base_input.to(device), input_delta.to(device)
        )
        write_closure = float(
            (
                writes["state"]
                + writes["parameter"]
                - writes["total"]
            )
            .abs()
            .max()
            .cpu()
        )
        encoded = model.encode_difference(
            input_delta.to(device),
            writes["state"],
            writes["parameter"],
        )
        graphs = model.active_jacobian_shapley(
            base_input.to(device),
            input_delta.to(device),
            encoded["input_indices"],
            encoded["output_indices"],
        )
        graph_closure = float(
            (
                graphs["state"]
                + graphs["parameter"]
                - graphs["total"]
            )
            .abs()
            .max()
            .cpu()
        )
        planted_rms = float(
            graphs["total"].square().mean().sqrt().cpu()
        )
    target_map = {
        "input": input_delta,
        "state": writes["state"].cpu(),
        "parameter": writes["parameter"].cpu(),
        "total": writes["total"].cpu(),
    }
    rows = np.arange(2, dtype=np.int64)
    autograd_report = autograd_control(
        model, base_input, target_map, rows, examples=2
    )
    finite_report = finite_difference_control(
        model, base_input, target_map, rows, examples=2
    )
    invariants = decoder_invariants(model)

    torch.manual_seed(771_205)
    negative = ModelDifferenceJacobianSAE(
        config,
        base_weights=base_weights,
        tuned_weights={
            key: value.clone() for key, value in base_weights.items()
        },
        input_scale=torch.tensor(1.0),
        output_scale=torch.tensor(1.0),
    ).to(device)
    zero_delta = torch.zeros_like(input_delta).to(device)
    with torch.no_grad():
        negative_writes = negative.write_shapley(
            base_input.to(device), zero_delta
        )
        negative_encoded = negative.encode_difference(
            zero_delta,
            negative_writes["state"],
            negative_writes["parameter"],
        )
        negative_graph = negative.active_jacobian_shapley(
            base_input.to(device),
            zero_delta,
            negative_encoded["input_indices"],
            negative_encoded["output_indices"],
        )
        negative_maximum = max(
            float(negative_writes["total"].abs().max().cpu()),
            float(negative_graph["total"].abs().max().cpu()),
            float(
                negative_encoded["reconstructed_total"].abs().max().cpu()
            ),
        )
    checks = {
        "aligned_planted_controls": aligned["PASS"],
        "write_shapley_closure": write_closure <= 2e-6,
        "jacobian_shapley_closure": graph_closure <= 2e-6,
        "accelerator_model_difference_signal": (
            planted_rms >= GRAPH_RESOLUTION_FLOOR
        ),
        "analytic_autograd": autograd_report["PASS"],
        "finite_difference": finite_report["PASS"],
        "identical_model_zero_difference": negative_maximum <= 1e-12,
        "dictionary_invariants": invariants["PASS"],
    }
    return {
        "kind": "deterministic planted model-difference instrument",
        "seed": 771_204,
        "checks": checks,
        "aligned_planted_controls": aligned,
        "write_shapley_closure_maximum_absolute_error": write_closure,
        "jacobian_shapley_closure_maximum_absolute_error": graph_closure,
        "planted_total_graph_rms": planted_rms,
        "identical_model_negative_maximum_absolute_value": negative_maximum,
        "autograd": autograd_report,
        "finite_difference": finite_report,
        "invariants": invariants,
        "PASS": all(checks.values()),
    }


def float64_real_preflight(
    base_weights: Mapping[str, torch.Tensor],
    tuned_weights: Mapping[str, torch.Tensor],
    base_inputs: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    states: Sequence[Mapping[str, Any]],
    train_indices: np.ndarray,
) -> dict[str, Any]:
    torch.manual_seed(99_701)
    model = build_model(
        base_weights,
        tuned_weights,
        targets,
        states,
        train_indices,
        device=torch.device("cpu"),
    ).double()
    selected = train_indices[:2]
    base_input = base_inputs[selected].double()
    input_delta = targets["input"][selected].double()
    with torch.no_grad():
        writes = model.write_shapley(base_input, input_delta)
        encoded = model.encode_difference(
            input_delta, writes["state"], writes["parameter"]
        )
        graphs = model.active_jacobian_shapley(
            base_input,
            input_delta,
            encoded["input_indices"],
            encoded["output_indices"],
        )
        write_error = float(
            (
                writes["state"]
                + writes["parameter"]
                - writes["total"]
            )
            .abs()
            .max()
        )
        graph_error = float(
            (
                graphs["state"]
                + graphs["parameter"]
                - graphs["total"]
            )
            .abs()
            .max()
        )
    return {
        "rows": selected.tolist(),
        "write_shapley_closure_maximum_absolute_error": write_error,
        "jacobian_shapley_closure_maximum_absolute_error": graph_error,
        "tolerance": 1e-10,
        "PASS": max(write_error, graph_error) <= 1e-10,
    }


def condition_record(
    model: ModelDifferenceJacobianSAE,
    training: dict[str, Any],
    base_inputs: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
    states: Sequence[Mapping[str, Any]],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    validation_pairs: Sequence[Mapping[str, Any]],
    *,
    primary: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_report, _ = evaluate_model(
        model, base_inputs, targets, states, train_indices
    )
    validation_report, fingerprints = evaluate_model(
        model, base_inputs, targets, states, validation_indices
    )
    record = {
        "training": training,
        "train": train_report,
        "validation": validation_report,
        "contrast_validation": contrast_fsee(
            model, targets, validation_pairs
        ),
    }
    if primary:
        control_indices = stratified_control_indices(
            states, validation_indices
        )
        record.update(
            {
                "mechanical_control_indices": control_indices.tolist(),
                "mechanical_control_stratification": (
                    "one deterministic row per endpoint/state-family cell"
                ),
                "decoder_invariants": decoder_invariants(model),
                "autograd_control": autograd_control(
                    model,
                    base_inputs,
                    targets,
                    control_indices,
                    examples=len(control_indices),
                ),
                "finite_difference_control": finite_difference_control(
                    model,
                    base_inputs,
                    targets,
                    control_indices,
                    examples=len(control_indices),
                ),
            }
        )
    return record, fingerprints


def representation_gate(
    primary: Mapping[str, Any],
    matched: Mapping[str, Any],
    decomposition_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    train = primary["train"]
    validation = primary["validation"]
    matched_validation = matched["validation"]
    primary_graph = validation["component_union_graph"]
    matched_graph = matched_validation["component_union_graph"]
    total_graph = validation["signed_total_deltaJ"]
    e90 = primary_graph["E90"]["median"]
    matched_e90 = matched_graph["E90"]["median"]
    effective = primary_graph["effective_active_edge_count"]["median"]
    matched_effective = matched_graph["effective_active_edge_count"]["median"]
    e90_reduction = (
        (matched_e90 - e90) / matched_e90 if matched_e90 > 0 else 0.0
    )
    effective_reduction = (
        (matched_effective - effective) / matched_effective
        if matched_effective > 0
        else 0.0
    )
    gates = {
        "decomposition_preflight": decomposition_preflight["PASS"],
        "input_level_fsee": validation["level"]["input"]["fsee"] >= 0.90,
        "state_level_fsee": validation["level"]["state"]["fsee"] >= 0.95,
        "parameter_level_fsee": (
            validation["level"]["parameter"]["fsee"] >= 0.95
        ),
        "total_level_fsee": validation["level"]["total"]["fsee"] >= 0.95,
        "state_cross_replay_fsee": (
            validation["cross_replay"]["state"]["fsee"] >= 0.90
        ),
        "parameter_cross_replay_fsee": (
            validation["cross_replay"]["parameter"]["fsee"] >= 0.90
        ),
        "total_cross_replay_fsee": (
            validation["cross_replay"]["total"]["fsee"] >= 0.90
        ),
        "input_dead_fraction": train["input_dead_feature_fraction"] <= 0.50,
        "output_dead_fraction": train["output_dead_feature_fraction"] <= 0.50,
        "input_exact_signed_l0": (
            validation["input_l0"]["minimum"] == ACTIVE_FEATURES
            and validation["input_l0"]["maximum"] == ACTIVE_FEATURES
        ),
        "output_exact_signed_l0": (
            validation["output_l0"]["minimum"] == ACTIVE_FEATURES
            and validation["output_l0"]["maximum"] == ACTIVE_FEATURES
        ),
        "decoder_invariants": primary["decoder_invariants"]["PASS"],
        "autograd": primary["autograd_control"]["PASS"],
        "finite_difference": primary["finite_difference_control"]["PASS"],
        "component_union_graph_resolution": (
            primary_graph["normalized_rms"]["median"]
            >= GRAPH_RESOLUTION_FLOOR
        ),
        "signed_total_deltaJ_resolution": (
            total_graph["normalized_rms"]["median"]
            >= GRAPH_RESOLUTION_FLOOR
        ),
        "E90_reduction": e90_reduction >= 0.15,
        "effective_edge_reduction": effective_reduction >= 0.10,
        "total_operator_action_fsee": (
            validation["operator_action"]["total"]["fsee"] >= 0.90
        ),
    }
    for key in ("input", "state", "parameter", "total"):
        gates[f"{key}_level_matched"] = (
            validation["level"][key]["fsee"]
            >= matched_validation["level"][key]["fsee"] - 0.03
        )
    for key in ("state", "parameter", "total"):
        gates[f"{key}_cross_replay_matched"] = (
            validation["cross_replay"][key]["fsee"]
            >= matched_validation["cross_replay"][key]["fsee"] - 0.03
        )
    for key in ("state", "parameter"):
        operator = validation["operator_action"][key]
        matched_operator = matched_validation["operator_action"][key]
        if operator["native_power_fraction_of_total"] >= 0.05:
            gates[f"{key}_operator_action_fsee"] = (
                operator["fsee"] >= 0.85
            )
            gates[f"{key}_operator_action_matched"] = (
                operator["fsee"] >= matched_operator["fsee"] - 0.03
            )
        else:
            gates[f"{key}_operator_action_below_resolution"] = (
                operator["error_power_fraction_of_total"] <= 0.05
            )
    gates["total_operator_action_matched"] = (
        validation["operator_action"]["total"]["fsee"]
        >= matched_validation["operator_action"]["total"]["fsee"] - 0.03
    )
    for stratum, values in primary["contrast_validation"][
        "by_stratum"
    ].items():
        gates[f"{stratum}:input_contrast_fsee"] = (
            values["input"]["fsee"] >= 0.85
        )
        for key in ("state", "parameter", "total"):
            gates[f"{stratum}:{key}_contrast_fsee"] = (
                values[key]["fsee"] >= 0.90
            )
    return {
        "gates": gates,
        "median_E90_reduction_fraction": e90_reduction,
        "median_effective_edge_reduction_fraction": effective_reduction,
        "PASS": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-artifact", required=True)
    parser.add_argument("--model-dir", action="append", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--resume-pilot-result")
    parser.add_argument("--resume-pilot-model")
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if bool(args.resume_pilot_result) != bool(args.resume_pilot_model):
        raise ValueError("pilot result and pilot model must be resumed together")
    if args.pilot_only and args.resume_pilot_result:
        raise ValueError("pilot-only and pilot resume are mutually exclusive")

    activation_path = Path(args.activation_artifact).expanduser().resolve()
    activation_report = json.loads(activation_path.read_text())
    assert_label_free_payload(activation_report)
    if (
        activation_report.get("status") != "COMPLETE"
        or activation_report.get("stage") != "label-free activation capture"
        or activation_report.get("label_free") is not True
    ):
        raise RuntimeError("model-difference J-SAE requires label-free capture")
    model_dirs = LARGE.parse_model_dirs(args.model_dir)
    checkpoint_hashes = LARGE.validate_frozen_sources(
        activation_path, activation_report, model_dirs
    )
    device = torch.device(
        args.device
        or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    output = guard_output_path(Path(args.out).resolve())
    model_output = guard_output_path(Path(args.model_out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    if len({output, model_output, manifest_path}) != 3:
        raise ValueError("result, model, and manifest outputs must be distinct")

    states = activation_report["states"]
    genes = [str(row["gene_id"]) for row in states]
    train_mask, validation_mask = INSTRUMENT.gene_train_validation_split(genes)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    train_pairs = LARGE.build_reference_pairs(states, train_mask)
    validation_pairs = LARGE.build_reference_pairs(states, validation_mask)
    observed_counts = {
        "states": len(states),
        "training_states": len(train_indices),
        "validation_states": len(validation_indices),
        "training_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "training_genes": len(set(np.asarray(genes)[train_mask])),
        "validation_genes": len(set(np.asarray(genes)[validation_mask])),
    }
    if observed_counts != FROZEN_COUNTS:
        raise RuntimeError(f"frozen data counts changed: {observed_counts}")

    print("running deterministic synthetic instrument preflight", flush=True)
    synthetic = synthetic_preflight(device)
    if not synthetic["PASS"]:
        raise RuntimeError("synthetic model-difference preflight failed")

    _, base_input_array, base_write_array = INSTRUMENT.load_activation_artifact(
        activation_path, view="base"
    )
    base_inputs = torch.from_numpy(base_input_array)
    base_writes = torch.from_numpy(base_write_array)
    base_weights, base_hash = INSTRUMENT.extract_mlp_weights(model_dirs["base"])
    if base_hash != checkpoint_hashes["base"]:
        raise RuntimeError("base checkpoint changed during extraction")

    addendum = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "IMPLEMENTATION_ADDENDUM_MODEL_DIFFERENCE_JSAE_V1.md"
    )
    implementation_path = (
        REPO / "src/gfmcd/circuits/model_difference_jsae.py"
    )
    runner_sha256 = sha256_file(Path(__file__).resolve())
    protocol_sha256 = sha256_file(addendum)
    implementation_sha256 = sha256_file(implementation_path)
    resume_result_path = (
        Path(args.resume_pilot_result).expanduser().resolve()
        if args.resume_pilot_result
        else None
    )
    resume_model_path = (
        Path(args.resume_pilot_model).expanduser().resolve()
        if args.resume_pilot_model
        else None
    )
    if resume_result_path is not None and resume_model_path is not None:
        resume_result = json.loads(resume_result_path.read_text())
        if (
            resume_result.get("execution_mode") != "pilot_only"
            or resume_result.get("status") != "PILOT_PASS"
            or resume_result.get("pilot", {}).get("PASS") is not True
            or resume_result.get("activation_artifact_sha256")
            != sha256_file(activation_path)
            or resume_result.get("runner_sha256") != runner_sha256
            or resume_result.get("protocol_sha256") != protocol_sha256
            or resume_result.get("implementation_sha256")
            != implementation_sha256
            or resume_result.get("model_artifact", {}).get("sha256")
            != sha256_file(resume_model_path)
        ):
            raise RuntimeError("pilot resume artifact failed identity checks")
        resume_model_payload = torch.load(
            resume_model_path, map_location="cpu", weights_only=False
        )
        if (
            resume_model_payload.get("schema")
            != "gfmcd.splicebert.model_difference_jsae.v1"
        ):
            raise RuntimeError("pilot model payload schema mismatch")
        result_seeds = copy.deepcopy(resume_result["seeds"])
        model_payload = resume_model_payload
        stored_fingerprints = model_payload["checkpoint_models"]["2020"][
            "replicates"
        ]["0"]["fingerprints"]
        fitted_primary = {
            "2020": [
                {
                    key: (
                        value.numpy()
                        if isinstance(value, torch.Tensor)
                        else np.asarray(value)
                    )
                    for key, value in stored_fingerprints.items()
                }
            ]
        }
        pilot_pass = True
        execution_mode = "resume_full_family"
    else:
        result_seeds: dict[str, Any] = {}
        fitted_primary: dict[str, list[dict[str, np.ndarray]]] = {}
        model_payload: dict[str, Any] = {
            "schema": "gfmcd.splicebert.model_difference_jsae.v1",
            "architecture": {
                "input_features": INPUT_FEATURES,
                "output_features": OUTPUT_FEATURES,
                "active_features": ACTIVE_FEATURES,
                "signed": True,
                "zero_bias": True,
            },
            "checkpoint_models": {},
        }
        pilot_pass = False
        execution_mode = "pilot_only" if args.pilot_only else "full_family"
    stop_family = False

    for checkpoint_seed in TUNED_SEEDS:
        view = str(checkpoint_seed)
        print(f"building exact model-difference targets for {view}", flush=True)
        _, tuned_input_array, tuned_write_array = (
            INSTRUMENT.load_activation_artifact(
                activation_path, view=view
            )
        )
        tuned_inputs = torch.from_numpy(tuned_input_array)
        tuned_writes = torch.from_numpy(tuned_write_array)
        tuned_weights, tuned_hash = INSTRUMENT.extract_mlp_weights(
            model_dirs[view]
        )
        if tuned_hash != checkpoint_hashes[view]:
            raise RuntimeError(f"checkpoint {view} changed during extraction")
        targets, decomposition = build_difference_targets(
            base_inputs,
            tuned_inputs,
            base_writes,
            tuned_writes,
            base_weights,
            tuned_weights,
            device=device,
        )
        float64_preflight = float64_real_preflight(
            base_weights,
            tuned_weights,
            base_inputs,
            targets,
            states,
            train_indices,
        )
        decomposition["float64_local_operator_preflight"] = float64_preflight
        decomposition["PASS"] = (
            decomposition["PASS"] and float64_preflight["PASS"]
        )
        if not decomposition["PASS"]:
            raise RuntimeError(f"checkpoint {view} decomposition failed")
        if view in result_seeds:
            if (
                view != "2020"
                or not args.resume_pilot_result
                or result_seeds[view]["checkpoint_hash"] != tuned_hash
            ):
                raise RuntimeError("unexpected pre-existing checkpoint result")
            result_seeds[view]["decomposition_resume_replay"] = decomposition
            initializations = INITIALIZATIONS[1:]
        else:
            result_seeds[view] = {
                "checkpoint_hash": tuned_hash,
                "decomposition": decomposition,
                "replicates": [],
            }
            fitted_primary[view] = []
            model_payload["checkpoint_models"][view] = {
                "checkpoint_hash": tuned_hash,
                "replicates": {},
            }
            initializations = INITIALIZATIONS
        for initialization in initializations:
            seed = initialization_seed(checkpoint_seed, initialization)
            conditions: dict[str, Any] = {}
            initial_hashes = set()
            primary_state = None
            primary_fingerprints = None
            for condition in CONDITIONS:
                print(
                    f"seed={view} init={initialization} starting {condition}",
                    flush=True,
                )
                model, training = train_condition(
                    base_weights,
                    tuned_weights,
                    base_inputs,
                    targets,
                    states,
                    train_indices,
                    train_pairs,
                    condition=condition,
                    device=device,
                    seed=seed,
                )
                record, fingerprints = condition_record(
                    model,
                    training,
                    base_inputs,
                    targets,
                    states,
                    train_indices,
                    validation_indices,
                    validation_pairs,
                    primary=condition == "shapley_deltaJ",
                )
                conditions[condition] = record
                initial_hashes.add(training["initial_state_sha256"])
                if condition == "shapley_deltaJ":
                    primary_state = compact_state(model)
                    primary_fingerprints = fingerprints
                print(
                    f"seed={view} init={initialization} {condition} "
                    f"dx_FSEE={record['validation']['level']['input']['fsee']:.4f} "
                    f"dy_FSEE={record['validation']['level']['total']['fsee']:.4f} "
                    "union_E90="
                    f"{record['validation']['component_union_graph']['E90']['median']:.1f}",
                    flush=True,
                )
                del model
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
            if len(initial_hashes) != 1:
                raise RuntimeError("matched conditions did not share initialization")
            if primary_state is None or primary_fingerprints is None:
                raise RuntimeError("primary model was not preserved")
            gate = representation_gate(
                conditions["shapley_deltaJ"],
                conditions["shapley_no_deltaJ"],
                decomposition,
            )
            result_seeds[view]["replicates"].append(
                {
                    "initialization": initialization,
                    "initialization_seed": seed,
                    "identical_initial_state": len(initial_hashes) == 1,
                    "conditions": conditions,
                    "representation_gate": gate,
                }
            )
            fitted_primary[view].append(primary_fingerprints)
            model_payload["checkpoint_models"][view]["replicates"][
                str(initialization)
            ] = {
                "primary_state": primary_state,
                "mean_absolute_group_graph": torch.from_numpy(
                    primary_fingerprints["graph"]
                ),
                "fingerprints": {
                    key: torch.from_numpy(value)
                    for key, value in primary_fingerprints.items()
                },
            }
            print(
                f"seed={view} init={initialization} gate={gate['PASS']}",
                flush=True,
            )
            if checkpoint_seed == 2020 and initialization == 0:
                pilot_pass = gate["PASS"]
                if not pilot_pass or args.pilot_only:
                    stop_family = True
                    break
        del tuned_inputs, tuned_writes, targets
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        if stop_family:
            break

    complete_family = (
        pilot_pass
        and set(result_seeds) == {str(seed) for seed in TUNED_SEEDS}
        and all(
            len(result_seeds[str(seed)]["replicates"])
            == len(INITIALIZATIONS)
            for seed in TUNED_SEEDS
        )
    )
    stability_rows = []
    if complete_family:
        for checkpoint_seed in TUNED_SEEDS:
            view = str(checkpoint_seed)
            for left, right in combinations(INITIALIZATIONS, 2):
                sites = {
                    site: feature_match(
                        fitted_primary[view][left][f"{site}_features"],
                        fitted_primary[view][right][f"{site}_features"],
                        fitted_primary[view][left][f"{site}_decoder"],
                        fitted_primary[view][right][f"{site}_decoder"],
                        fitted_primary[view][left][
                            f"{site}_sample_weights"
                        ],
                    )
                    for site in ("input", "output")
                }
                stability_rows.append(
                    {
                        "checkpoint_seed": checkpoint_seed,
                        "initialization_pair": [left, right],
                        "sites": sites,
                        "PASS": all(row["PASS"] for row in sites.values()),
                    }
                )
    fit_pass = complete_family and all(
        replicate["representation_gate"]["PASS"]
        for seed in result_seeds.values()
        for replicate in seed["replicates"]
    )
    stability_pass = complete_family and all(
        row["PASS"] for row in stability_rows
    )
    overall_pass = fit_pass and stability_pass
    pilot_only_complete = (
        execution_mode == "pilot_only"
        and set(result_seeds) == {"2020"}
        and len(result_seeds["2020"]["replicates"]) == 1
    )
    if overall_pass:
        status = "INSTRUMENT_PASS"
    elif pilot_only_complete and pilot_pass:
        status = "PILOT_PASS"
    else:
        status = "INSTRUMENT_UNRESOLVED"
    result = {
        "study": "SpliceBERT GTEx base-to-tuned model-difference J-SAE",
        "stage": "label-free comparative representation instrument",
        "status": status,
        "execution_mode": execution_mode,
        "evidence_tier": "instrument",
        "implementation_addendum": str(addendum.relative_to(REPO)),
        "runner_sha256": runner_sha256,
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "activation_artifact": str(activation_path),
        "activation_artifact_sha256": sha256_file(activation_path),
        "synthetic_preflight": synthetic,
        "architecture": {
            "input_features": INPUT_FEATURES,
            "output_features": OUTPUT_FEATURES,
            "active_features": ACTIVE_FEATURES,
            "native_width": 512,
            "mlp_width": 2048,
            "signed_topk": True,
            "zero_encoder_and_decoder_bias": True,
            "shared_output_support_for_shapley_blocks": True,
        },
        "objective": {
            "conditions": CONDITIONS,
            "fixed_epochs": EPOCHS,
            "state_batch": STATE_BATCH,
            "pair_batch": PAIR_BATCH,
            "auxiliary_loss": None,
            "level_targets": [
                "input",
                "state",
                "parameter",
                "total",
            ],
            "contrast_orientation": "changed-reference",
        },
        "data": observed_counts,
        "checkpoint_hashes": checkpoint_hashes,
        "seeds": result_seeds,
        "pilot": {
            "checkpoint_seed": 2020,
            "initialization": 0,
            "PASS": pilot_pass,
            "pilot_only_complete": pilot_only_complete,
            "stopped_family_for_failure": not pilot_pass,
        },
        "within_checkpoint_stability": {
            "rows": stability_rows,
            "thresholds": {
                "density_floor": 0.001,
                "reciprocal_match_fraction": 0.25,
                "reciprocal_score_mean": 0.50,
                "sign_invariant": True,
            },
            "PASS": stability_pass,
        },
        "instrument_gate": {
            "pilot_only": pilot_only_complete,
            "complete_family": complete_family,
            "all_primary_representation_fits_PASS": fit_pass,
            "all_site_stability_comparisons_PASS": stability_pass,
            "classifier_parameter_or_margin_opened": False,
            "feature_identity_opened": False,
            "PASS": overall_pass,
        },
        "full_family_authorized": pilot_pass,
        "fresh_causal_confirmation_authorized": overall_pass,
        "feature_semantics_established": False,
        "licensed_claim": (
            "Across five independently fine-tuned checkpoints, a "
            "zero-anchored signed sparse coordinate system preserves the "
            "held-out base-to-tuned layer-5 representation shift and the "
            "exact state-versus-final-MLP-parameter contributions to the "
            "write and local Jacobian shift, while concentrating the "
            "projected derivative-change graph relative to matched controls."
            if overall_pass
            else (
                "The seed-2020 initialization-0 model-difference pilot "
                "passes its frozen representation, operator-faithfulness, "
                "and graph-control gates and authorizes the remaining "
                "checkpoint/initialization family. Feature identity and "
                "classifier outcomes remain closed."
                if status == "PILOT_PASS"
                else "The base-to-tuned model-difference J-SAE remains an "
                "unresolved representation instrument; classifier behavior, "
                "feature identity, and biological semantics stay closed."
            )
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_payload, model_output)
    result["model_artifact"] = {
        "path": str(model_output),
        "sha256": sha256_file(model_output),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:model_difference_jsae_v1",
        sys.argv,
        repo=REPO,
        status=status,
        config={
            "architecture": result["architecture"],
            "objective": result["objective"],
            "device": str(device),
        },
        seeds={
            "executed_dictionary_initializations": [
                replicate["initialization_seed"]
                for seed in result_seeds.values()
                for replicate in seed["replicates"]
            ],
            "planned_dictionary_initializations": [
                initialization_seed(seed, initialization)
                for seed in TUNED_SEEDS
                for initialization in INITIALIZATIONS
            ],
            "synthetic_preflight": [synthetic["seed"]],
        },
        notes=[
            "Only base-to-tuned differences were reconstructed.",
            "State and final-MLP-parameter write/Jacobian shifts use exact Shapley accounting.",
            "All fits were sequential and used no AuxK.",
            "No classifier parameter, task margin, feature ranking, or biological outcome was loaded.",
        ],
    )
    arrays_path = Path(activation_report["arrays"]["path"]).resolve()
    model_files = []
    for model_dir in model_dirs.values():
        model_files.extend(
            (
                model_dir / "config.json",
                LARGE.checkpoint_weight_path(model_dir),
            )
        )
    resume_inputs = (
        [resume_result_path, resume_model_path]
        if resume_result_path is not None and resume_model_path is not None
        else []
    )
    for path in (
        addendum,
        implementation_path,
        activation_path,
        arrays_path,
        *model_files,
        *resume_inputs,
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.extend([file_record(output), file_record(model_output)])
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": status,
                "pilot": result["pilot"],
                "instrument_gate": result["instrument_gate"],
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
