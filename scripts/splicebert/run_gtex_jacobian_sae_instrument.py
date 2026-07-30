#!/usr/bin/env python
"""Develop and preflight a checkpoint-specific layer-5 Jacobian SAE."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import hashlib
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
from gfmcd.circuits.jacobian_sae import (  # noqa: E402
    JacobianSAEConfig,
    JacobianSAEPair,
    auxiliary_dead_feature_loss,
    edge_mass_count,
    explained_variance,
    scalar_normalization,
)
from gfmcd.circuits.multiview_transcoder import (  # noqa: E402
    assert_label_free_payload,
)
from gfmcd.provenance import (  # noqa: E402
    ExperimentManifest,
    file_record,
    sha256_file,
)


NAMESPACE = "gtex-native-biological-features-v1"
TRAIN_NAMESPACE = "gtex-native-biological-features-transcoder-train-v1"
VALIDATION_PERCENT = 20
FROZEN_VIEW = "2020"
FEATURE_WIDTH = 512
CAPACITY_REPAIR_INPUT_WIDTH = 256
CAPACITY_REPAIR_OUTPUT_WIDTH = 64
ACTIVE_FEATURES = 8
JACOBIAN_COEFFICIENTS = (0.0, 0.015625, 0.0625, 0.25)
INITIALIZATION_SEED = 1_512_008
DEFAULT_BATCH = 128
DEFAULT_EPOCHS = 40
DEFAULT_PATIENCE = 6
DEFAULT_LR = 5e-4
MINIMUM_IMPROVEMENT = 1e-5
GRADIENT_CLIP = 1.0
AUTOAUX_FEATURES = 256
AUTOAUX_COEFFICIENT = 1 / 32
MLP_REPLAY_TOLERANCE = 5e-4
AUTOGRAD_ABSOLUTE_TOLERANCE = 2e-5
AUTOGRAD_RELATIVE_TOLERANCE = 2e-4


def stable_bucket(value: str, *, namespace: str, modulo: int = 100) -> int:
    digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


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
        raise RuntimeError("J-SAE train/validation gene split is too small")
    return train, validation


def load_activation_artifact(
    report_path: Path,
    *,
    view: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    report = json.loads(report_path.read_text())
    assert_label_free_payload(report)
    if (
        report.get("status") != "COMPLETE"
        or report.get("stage") != "label-free activation capture"
        or report.get("cohort_namespace") != NAMESPACE
        or report.get("label_free") is not True
        or report.get("training_authorized") is not True
        or not report.get("instrument_gate", {}).get("PASS")
        or view not in report.get("views", [])
    ):
        raise RuntimeError("activation artifact does not authorize J-SAE training")
    arrays_path = Path(report["arrays"]["path"]).expanduser().resolve()
    if sha256_file(arrays_path) != report["arrays"]["sha256"]:
        raise RuntimeError("activation array hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        expected = {f"{view}__input", f"{view}__write"}
        if not expected.issubset(archive.files):
            raise RuntimeError("requested J-SAE view is absent")
        inputs = archive[f"{view}__input"].astype(np.float32)
        writes = archive[f"{view}__write"].astype(np.float32)
    expected_shape = (int(report["n_states"]), 512)
    if (
        inputs.shape != expected_shape
        or writes.shape != expected_shape
        or not np.isfinite(inputs).all()
        or not np.isfinite(writes).all()
    ):
        raise RuntimeError("J-SAE activation arrays failed shape/finiteness checks")
    return report, inputs, writes


def extract_mlp_weights(
    model_dir: Path,
) -> tuple[dict[str, torch.Tensor], str]:
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(
        str(model_dir),
        attn_implementation="eager",
    )
    if (
        getattr(model.config, "hidden_act", None) != "gelu"
        or model.config.num_hidden_layers != 6
        or model.config.hidden_size != 512
        or model.config.intermediate_size != 2048
    ):
        raise RuntimeError("checkpoint is not the frozen SpliceBERT architecture")
    layer = model.bert.encoder.layer[5]
    weights = {
        "input_weight": layer.intermediate.dense.weight.detach().cpu().float().clone(),
        "input_bias": layer.intermediate.dense.bias.detach().cpu().float().clone(),
        "output_weight": layer.output.dense.weight.detach().cpu().float().clone(),
        "output_bias": layer.output.dense.bias.detach().cpu().float().clone(),
    }
    del model
    gc.collect()
    weight_path = model_dir / "model.safetensors"
    if not weight_path.is_file():
        weight_path = model_dir / "pytorch_model.bin"
    if not weight_path.is_file():
        raise FileNotFoundError("checkpoint weight file is missing")
    return weights, sha256_file(weight_path)


def build_pair(
    weights: Mapping[str, torch.Tensor],
    train_inputs: torch.Tensor,
    train_writes: torch.Tensor,
    *,
    device: torch.device,
    capacity_repair: bool = False,
) -> JacobianSAEPair:
    input_center, input_scale = scalar_normalization(train_inputs)
    output_center, output_scale = scalar_normalization(train_writes)
    config = JacobianSAEConfig(
        input_dim=512,
        output_dim=512,
        mlp_dim=2048,
        num_features=(
            CAPACITY_REPAIR_INPUT_WIDTH
            if capacity_repair
            else FEATURE_WIDTH
        ),
        output_num_features=(
            CAPACITY_REPAIR_OUTPUT_WIDTH
            if capacity_repair
            else None
        ),
        active_features=ACTIVE_FEATURES,
    )
    return JacobianSAEPair(
        config,
        input_weight=weights["input_weight"],
        input_bias=weights["input_bias"],
        output_weight=weights["output_weight"],
        output_bias=weights["output_bias"],
        input_center=input_center,
        input_scale=input_scale,
        output_center=output_center,
        output_scale=output_scale,
    ).to(device)


def gene_balanced_epoch(
    train_indices: np.ndarray,
    genes: list[str],
    *,
    seed: int,
) -> np.ndarray:
    """Sample one state-balanced epoch with equal expected gene weight."""

    generator = np.random.default_rng(seed)
    by_gene: dict[str, list[int]] = {}
    for index in train_indices:
        by_gene.setdefault(genes[int(index)], []).append(int(index))
    per_gene = math.ceil(len(train_indices) / len(by_gene))
    selected = []
    for gene in sorted(by_gene):
        candidates = np.asarray(by_gene[gene], dtype=np.int64)
        selected.extend(
            generator.choice(
                candidates,
                size=per_gene,
                replace=len(candidates) < per_gene,
            ).tolist()
        )
    selected_array = np.asarray(selected, dtype=np.int64)
    generator.shuffle(selected_array)
    return selected_array[: len(train_indices)]


def scheduled_learning_rate(step: int, total_steps: int) -> float:
    warmup = max(1, math.ceil(0.01 * total_steps))
    decay_start = math.floor(0.80 * total_steps)
    if step < warmup:
        factor = (step + 1) / warmup
    elif step >= decay_start:
        remaining = max(0, total_steps - step - 1)
        factor = remaining / max(1, total_steps - decay_start)
    else:
        factor = 1.0
    return DEFAULT_LR * factor


def scheduled_jacobian_coefficient(
    target: float,
    step: int,
    total_steps: int,
) -> float:
    warmup = max(1, math.ceil(0.05 * total_steps))
    return target * min(1.0, (step + 1) / warmup)


def clone_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


@torch.no_grad()
def reconstruction_loss(
    model: JacobianSAEPair,
    inputs: torch.Tensor,
    writes: torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
) -> float:
    model.eval()
    losses = []
    weights = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        x = inputs[selected].to(model.input_center.device)
        y = writes[selected].to(model.input_center.device)
        x_norm = model.normalize_input(x)
        y_norm = model.normalize_output(y)
        x_hat, _, _ = model.input_sae(x_norm)
        y_hat, _, _ = model.output_sae(y_norm)
        losses.append(
            float(
                (
                    torch.mean((x_hat - x_norm) ** 2)
                    + torch.mean((y_hat - y_norm) ** 2)
                ).cpu()
            )
        )
        weights.append(len(selected))
    return float(np.average(losses, weights=weights))


def train_one(
    weights: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    writes: torch.Tensor,
    genes: list[str],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    coefficient: float,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    auxk_repair: bool = False,
    capacity_repair: bool = False,
) -> tuple[JacobianSAEPair, dict[str, Any]]:
    torch.manual_seed(INITIALIZATION_SEED)
    model = build_pair(
        weights,
        inputs[train_indices],
        writes[train_indices],
        device=device,
        capacity_repair=capacity_repair,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=DEFAULT_LR,
        betas=(0.9, 0.999),
    )
    steps_per_epoch = math.ceil(len(train_indices) / batch_size)
    total_steps = epochs * steps_per_epoch
    global_step = 0
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    history = []
    input_steps_since_fired = torch.zeros(
        model.input_sae.num_features,
        dtype=torch.long,
        device=device,
    )
    output_steps_since_fired = torch.zeros(
        model.output_sae.num_features,
        dtype=torch.long,
        device=device,
    )
    dead_steps_threshold = steps_per_epoch

    for epoch in range(epochs):
        model.train()
        order = gene_balanced_epoch(
            train_indices,
            genes,
            seed=INITIALIZATION_SEED + 10_000 * epoch,
        )
        epoch_losses = []
        epoch_reconstruction = []
        epoch_jacobian = []
        epoch_auxiliary = []
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            x = inputs[selected].to(device)
            y = writes[selected].to(device)
            optimizer.param_groups[0]["lr"] = scheduled_learning_rate(
                global_step,
                total_steps,
            )
            current_coefficient = scheduled_jacobian_coefficient(
                coefficient,
                global_step,
                total_steps,
            )
            optimizer.zero_grad(set_to_none=True)

            x_norm = model.normalize_input(x)
            y_norm = model.normalize_output(y)
            x_hat, input_acts, input_indices = model.input_sae(x_norm)
            y_hat, output_acts, output_indices = model.output_sae(y_norm)
            input_mse = torch.mean((x_hat - x_norm) ** 2)
            output_mse = torch.mean((y_hat - y_norm) ** 2)
            reconstruction = input_mse + output_mse
            if current_coefficient > 0:
                jacobian = model.active_jacobian(
                    x,
                    input_indices,
                    output_indices,
                )
                jacobian_penalty = jacobian.abs().mean()
            else:
                jacobian_penalty = reconstruction.new_zeros(())
            if auxk_repair:
                with torch.no_grad():
                    input_fired = (input_acts > 1e-3).any(dim=0)
                    output_fired = (output_acts > 1e-3).any(dim=0)
                    input_steps_since_fired.mul_(~input_fired).add_(1)
                    output_steps_since_fired.mul_(~output_fired).add_(1)
                    input_dead = (
                        input_steps_since_fired > dead_steps_threshold
                    )
                    output_dead = (
                        output_steps_since_fired > dead_steps_threshold
                    )
                input_auxiliary = auxiliary_dead_feature_loss(
                    model.input_sae,
                    x_norm,
                    x_hat,
                    input_dead,
                    auxiliary_features=AUTOAUX_FEATURES,
                )
                output_auxiliary = auxiliary_dead_feature_loss(
                    model.output_sae,
                    y_norm,
                    y_hat,
                    output_dead,
                    auxiliary_features=AUTOAUX_FEATURES,
                )
                auxiliary = AUTOAUX_COEFFICIENT * (
                    input_auxiliary + output_auxiliary
                )
            else:
                auxiliary = reconstruction.new_zeros(())
            loss = (
                reconstruction
                + current_coefficient * jacobian_penalty
                + auxiliary
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))
            epoch_reconstruction.append(float(reconstruction.detach().cpu()))
            epoch_jacobian.append(float(jacobian_penalty.detach().cpu()))
            epoch_auxiliary.append(float(auxiliary.detach().cpu()))
            global_step += 1

        validation_loss = reconstruction_loss(
            model,
            inputs,
            writes,
            validation_indices,
            batch_size=batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(epoch_losses)),
                "training_reconstruction_loss": float(
                    np.mean(epoch_reconstruction)
                ),
                "training_jacobian_l1": float(np.mean(epoch_jacobian)),
                "training_auxiliary_loss": float(np.mean(epoch_auxiliary)),
                "validation_reconstruction_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - MINIMUM_IMPROVEMENT:
            best_loss = validation_loss
            best_state = clone_cpu_state(model)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        print(
            f"lambda={coefficient:g} epoch={epoch + 1}/{epochs} "
            f"val={validation_loss:.6f}",
            flush=True,
        )
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("J-SAE training produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_reconstruction_loss": best_loss,
        "history": history,
        "auxk_repair": auxk_repair,
        "auxiliary_features": AUTOAUX_FEATURES if auxk_repair else 0,
        "auxiliary_coefficient": (
            AUTOAUX_COEFFICIENT if auxk_repair else 0.0
        ),
        "dead_steps_threshold": (
            dead_steps_threshold if auxk_repair else None
        ),
    }


@torch.no_grad()
def evaluate_model(
    model: JacobianSAEPair,
    inputs: torch.Tensor,
    writes: torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    device = model.input_center.device
    raw_inputs = []
    reconstructed_inputs = []
    raw_writes = []
    reconstructed_writes = []
    sandwich_writes = []
    e50 = []
    e90 = []
    e99 = []
    raw_l1 = []
    l1_over_rms = []
    threshold_counts = {str(value): [] for value in (0.005, 0.01, 0.02)}
    input_frequency = torch.zeros(
        model.input_sae.num_features,
        dtype=torch.int64,
    )
    output_frequency = torch.zeros(
        model.output_sae.num_features,
        dtype=torch.int64,
    )
    input_l0 = []
    output_l0 = []

    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        x = inputs[selected].to(device)
        y = writes[selected].to(device)
        x_norm = model.normalize_input(x)
        y_norm = model.normalize_output(y)
        x_hat_norm, input_acts, input_indices = model.input_sae(x_norm)
        y_hat_norm, output_acts, output_indices = model.output_sae(y_norm)
        x_hat = model.denormalize_input(x_hat_norm)
        y_hat = model.denormalize_output(y_hat_norm)
        sandwich = model.mlp(x_hat)
        assert isinstance(sandwich, torch.Tensor)
        jacobian = model.active_jacobian(
            x,
            input_indices,
            output_indices,
        )

        raw_inputs.append(x.cpu())
        reconstructed_inputs.append(x_hat.cpu())
        raw_writes.append(y.cpu())
        reconstructed_writes.append(y_hat.cpu())
        sandwich_writes.append(sandwich.cpu())
        e50.append(edge_mass_count(jacobian, fraction=0.50).cpu())
        e90.append(edge_mass_count(jacobian, fraction=0.90).cpu())
        e99.append(edge_mass_count(jacobian, fraction=0.99).cpu())
        raw_l1.append(jacobian.abs().mean(dim=(-2, -1)).cpu())
        rms = jacobian.square().mean(dim=(-2, -1)).sqrt()
        l1_over_rms.append(
            (
                jacobian.abs().mean(dim=(-2, -1))
                / rms.clamp_min(model.config.eps)
            ).cpu()
        )
        for value in threshold_counts:
            threshold_counts[value].append(
                (jacobian.abs() >= float(value))
                .sum(dim=(-2, -1))
                .cpu()
            )
        input_frequency += (input_acts > 0).sum(dim=0).cpu()
        output_frequency += (output_acts > 0).sum(dim=0).cpu()
        input_l0.append((input_acts > 0).sum(dim=1).cpu())
        output_l0.append((output_acts > 0).sum(dim=1).cpu())

    x_all = torch.cat(raw_inputs)
    x_hat_all = torch.cat(reconstructed_inputs)
    y_all = torch.cat(raw_writes)
    y_hat_all = torch.cat(reconstructed_writes)
    sandwich_all = torch.cat(sandwich_writes)
    input_frequency_array = input_frequency.numpy()
    output_frequency_array = output_frequency.numpy()

    def distribution(values: list[torch.Tensor]) -> dict[str, float]:
        array = torch.cat(values).numpy().astype(np.float64)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.90)),
            "p99": float(np.quantile(array, 0.99)),
        }

    return {
        "n_examples": len(indices),
        "input_fve": explained_variance(x_all, x_hat_all),
        "output_write_fve": explained_variance(y_all, y_hat_all),
        "sandwich_output_fve": explained_variance(y_all, sandwich_all),
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
        "active_edge_mass_count": {
            "E50": distribution(e50),
            "E90": distribution(e90),
            "E99": distribution(e99),
        },
        "active_jacobian_l1": distribution(raw_l1),
        "active_jacobian_l1_over_rms": distribution(l1_over_rms),
        "active_edge_threshold_counts": {
            threshold: distribution(values)
            for threshold, values in threshold_counts.items()
        },
    }


def autograd_control(
    model: JacobianSAEPair,
    inputs: torch.Tensor,
    indices: np.ndarray,
    *,
    examples: int = 16,
) -> dict[str, Any]:
    model.eval()
    device = model.input_center.device
    selected_rows = indices[:examples]
    absolute_errors = []
    relative_errors = []
    for global_index in selected_rows:
        raw = inputs[int(global_index) : int(global_index) + 1].to(device)
        with torch.no_grad():
            x_norm = model.normalize_input(raw)
            _, input_acts, input_indices = model.input_sae(x_norm)
            native_output = model.mlp(raw)
            assert isinstance(native_output, torch.Tensor)
            y_norm = model.normalize_output(native_output)
            _, _, output_indices = model.output_sae(y_norm)
            analytic = model.active_jacobian(
                raw,
                input_indices,
                output_indices,
            )[0]
            directions = (
                model.input_sae.decoder()[input_indices[0]]
                * model.input_scale
            )
            baseline_features = input_acts[0].detach()
            del baseline_features

        def output_preactive(delta: torch.Tensor) -> torch.Tensor:
            perturbed = raw + delta @ directions
            output = model.mlp(perturbed)
            assert isinstance(output, torch.Tensor)
            normalized = model.normalize_output(output)
            preactive = (
                normalized @ model.output_sae.encoder_weight
                + model.output_sae.encoder_bias
            )
            return preactive[0, output_indices[0]]

        delta = torch.zeros(
            model.config.active_features,
            dtype=raw.dtype,
            device=device,
            requires_grad=True,
        )
        observed = torch.autograd.functional.jacobian(output_preactive, delta)
        difference = observed - analytic
        absolute_errors.append(float(difference.abs().max().detach().cpu()))
        relative_errors.append(
            float(
                (
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(observed).clamp_min(1e-8)
                )
                .detach()
                .cpu()
            )
        )
    maximum_absolute = max(absolute_errors)
    maximum_relative = max(relative_errors)
    return {
        "examples": len(selected_rows),
        "maximum_absolute_error": maximum_absolute,
        "maximum_relative_frobenius_error": maximum_relative,
        "absolute_tolerance": AUTOGRAD_ABSOLUTE_TOLERANCE,
        "relative_tolerance": AUTOGRAD_RELATIVE_TOLERANCE,
        "PASS": (
            maximum_absolute <= AUTOGRAD_ABSOLUTE_TOLERANCE
            and maximum_relative <= AUTOGRAD_RELATIVE_TOLERANCE
        ),
    }


def decoder_invariants(model: JacobianSAEPair) -> dict[str, Any]:
    with torch.no_grad():
        input_error = float(
            (model.input_sae.decoder().norm(dim=1) - 1.0).abs().max().cpu()
        )
        output_error = float(
            (model.output_sae.decoder().norm(dim=1) - 1.0).abs().max().cpu()
        )
    return {
        "maximum_input_decoder_norm_error": input_error,
        "maximum_output_decoder_norm_error": output_error,
        "tolerance": 1e-5,
        "PASS": max(input_error, output_error) <= 1e-5,
    }


def select_candidate(
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline = candidates["0"]
    baseline_metrics = baseline["validation"]
    baseline_e90 = baseline_metrics["active_edge_mass_count"]["E90"]["median"]
    ordered_keys = [f"{value:g}" for value in JACOBIAN_COEFFICIENTS]
    previous_e90 = baseline_e90
    eligible = []
    for key in ordered_keys[1:]:
        row = candidates[key]
        metrics = row["validation"]
        e90 = metrics["active_edge_mass_count"]["E90"]["median"]
        monotonic = e90 <= previous_e90
        previous_e90 = e90
        reconstruction_gates = {
            "input_fve_floor": metrics["input_fve"] >= 0.85,
            "output_fve_floor": metrics["output_write_fve"] >= 0.95,
            "sandwich_fve_floor": metrics["sandwich_output_fve"] >= 0.90,
            "input_fve_matched": (
                metrics["input_fve"] >= baseline_metrics["input_fve"] - 0.03
            ),
            "output_fve_matched": (
                metrics["output_write_fve"]
                >= baseline_metrics["output_write_fve"] - 0.03
            ),
            "sandwich_fve_matched": (
                metrics["sandwich_output_fve"]
                >= baseline_metrics["sandwich_output_fve"] - 0.03
            ),
            "input_dead_fraction": (
                row["train"]["input_dead_feature_fraction"] <= 0.50
            ),
            "output_dead_fraction": (
                row["train"]["output_dead_feature_fraction"] <= 0.50
            ),
        }
        edge_reduction = (
            (baseline_e90 - e90) / baseline_e90 if baseline_e90 > 0 else 0.0
        )
        graph_gates = {
            "E90_reduction_at_least_15_percent": edge_reduction >= 0.15,
            "E90_nonincreasing_through_candidate": monotonic,
        }
        row["comparison_to_zero_penalty"] = {
            "median_E90_reduction_fraction": edge_reduction,
            "input_fve_change": (
                metrics["input_fve"] - baseline_metrics["input_fve"]
            ),
            "output_fve_change": (
                metrics["output_write_fve"]
                - baseline_metrics["output_write_fve"]
            ),
            "sandwich_fve_change": (
                metrics["sandwich_output_fve"]
                - baseline_metrics["sandwich_output_fve"]
            ),
        }
        row["representation_gates"] = {
            **reconstruction_gates,
            **graph_gates,
        }
        row["representation_PASS"] = all(
            reconstruction_gates.values()
        ) and all(graph_gates.values())
        if row["representation_PASS"]:
            eligible.append(key)
    eligible.sort(
        key=lambda key: (
            -candidates[key]["comparison_to_zero_penalty"][
                "median_E90_reduction_fraction"
            ],
            -candidates[key]["validation"]["output_write_fve"],
            float(key),
        )
    )
    selected = eligible[0] if eligible else None
    return {
        "eligible_coefficients": eligible,
        "selected_coefficient": (
            None if selected is None else float(selected)
        ),
        "selection_rule": (
            "maximum median E90 reduction among all-gate passing nonzero "
            "coefficients; then output FVE; then smaller coefficient"
        ),
        "PASS": selected is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-artifact", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--view", default=FROZEN_VIEW)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--auxk-repair", action="store_true")
    parser.add_argument("--capacity-repair", action="store_true")
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (
        args.view != FROZEN_VIEW
        or args.batch != DEFAULT_BATCH
        or args.epochs != DEFAULT_EPOCHS
        or args.patience != DEFAULT_PATIENCE
        or args.learning_rate != DEFAULT_LR
    ):
        raise ValueError("instrument arguments differ from frozen protocol")
    device = torch.device(
        args.device
        or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    if args.capacity_repair and args.auxk_repair:
        raise ValueError("--capacity-repair already includes the v2 AuxK repair")
    version = 3 if args.capacity_repair else 2 if args.auxk_repair else 1
    effective_auxk_repair = args.auxk_repair or args.capacity_repair
    protocol_name = f"PREREGISTRATION_JACOBIAN_SAE_V{version}.md"
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        / protocol_name
    )
    parent_failure = (
        REPO
        / "results/splicebert_gtex_finetune_shift/"
        f"jacobian_sae_instrument_v{version - 1}.json"
    )
    if version > 1 and not parent_failure.is_file():
        raise RuntimeError(
            f"v{version} requires the preserved v{version - 1} failure artifact"
        )
    activation_path = Path(args.activation_artifact).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    model_output = guard_output_path(Path(args.model_out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    if len({output, model_output, manifest_path}) != 3:
        raise ValueError("result, model, and manifest outputs must be distinct")
    if output.suffix != ".json" or model_output.suffix not in {".pt", ".pth"}:
        raise ValueError("J-SAE run requires .json result and .pt/.pth model")

    report, input_array, write_array = load_activation_artifact(
        activation_path,
        view=args.view,
    )
    genes = [str(row["gene_id"]) for row in report["states"]]
    train_mask, validation_mask = gene_train_validation_split(genes)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    inputs = torch.from_numpy(input_array)
    writes = torch.from_numpy(write_array)
    weights, checkpoint_sha256 = extract_mlp_weights(model_dir)

    replay_pair = build_pair(
        weights,
        inputs[train_indices],
        writes[train_indices],
        device=device,
        capacity_repair=args.capacity_repair,
    )
    maximum_replay_error = 0.0
    with torch.no_grad():
        for start in range(0, len(inputs), args.batch):
            x = inputs[start : start + args.batch].to(device)
            y = writes[start : start + args.batch].to(device)
            replay = replay_pair.mlp(x)
            assert isinstance(replay, torch.Tensor)
            maximum_replay_error = max(
                maximum_replay_error,
                float((replay - y).abs().max().cpu()),
            )
    if maximum_replay_error > MLP_REPLAY_TOLERANCE:
        raise RuntimeError(
            f"frozen MLP replay failed: {maximum_replay_error:.8g}"
        )
    del replay_pair
    if device.type == "mps":
        torch.mps.empty_cache()

    candidates: dict[str, dict[str, Any]] = {}
    model_payload = {
        "schema": (
            f"gfmcd.splicebert.jacobian_sae.instrument.v{version}"
        ),
        "view": args.view,
        "config": asdict(
            JacobianSAEConfig(
                512,
                512,
                2048,
                (
                    CAPACITY_REPAIR_INPUT_WIDTH
                    if args.capacity_repair
                    else FEATURE_WIDTH
                ),
                ACTIVE_FEATURES,
                output_num_features=(
                    CAPACITY_REPAIR_OUTPUT_WIDTH
                    if args.capacity_repair
                    else None
                ),
            )
        ),
        "initialization_seed": INITIALIZATION_SEED,
        "checkpoint_sha256": checkpoint_sha256,
        "candidates": {},
    }
    for coefficient in JACOBIAN_COEFFICIENTS:
        key = f"{coefficient:g}"
        model, training = train_one(
            weights,
            inputs,
            writes,
            genes,
            train_indices,
            validation_indices,
            coefficient=coefficient,
            device=device,
            batch_size=args.batch,
            epochs=args.epochs,
            patience=args.patience,
            auxk_repair=effective_auxk_repair,
            capacity_repair=args.capacity_repair,
        )
        validation = evaluate_model(
            model,
            inputs,
            writes,
            validation_indices,
            batch_size=args.batch,
        )
        train_metrics = evaluate_model(
            model,
            inputs,
            writes,
            train_indices,
            batch_size=args.batch,
        )
        autograd = autograd_control(
            model,
            inputs,
            validation_indices,
        )
        decoder = decoder_invariants(model)
        candidates[key] = {
            "jacobian_coefficient": coefficient,
            "training": training,
            "train": train_metrics,
            "validation": validation,
            "autograd_control": autograd,
            "decoder_invariants": decoder,
            "mechanical_PASS": autograd["PASS"] and decoder["PASS"],
        }
        model_payload["candidates"][key] = clone_cpu_state(model)
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    selection = select_candidate(candidates)
    all_mechanical = all(row["mechanical_PASS"] for row in candidates.values())
    status = (
        "INSTRUMENT_PASS"
        if all_mechanical and selection["PASS"]
        else "INSTRUMENT_UNRESOLVED"
    )
    result = {
        "study": "SpliceBERT GTEx layer-5 Jacobian SAE",
        "stage": "seed-2020 outcome-blind instrument development",
        "status": status,
        "evidence_tier": "instrument",
        "protocol": str(protocol.relative_to(REPO)),
        "parent_failure": (
            {
                "path": str(parent_failure),
                "sha256": sha256_file(parent_failure),
            }
            if version > 1
            else None
        ),
        "view": args.view,
        "checkpoint": {
            "model_dir": str(model_dir),
            "sha256": checkpoint_sha256,
        },
        "data": {
            "activation_artifact": str(activation_path),
            "activation_artifact_sha256": sha256_file(activation_path),
            "arrays_path": report["arrays"]["path"],
            "arrays_sha256": report["arrays"]["sha256"],
            "n_states": len(inputs),
            "n_train_states": len(train_indices),
            "n_validation_states": len(validation_indices),
            "n_train_genes": len(
                set(np.asarray(genes, dtype=object)[train_mask])
            ),
            "n_validation_genes": len(
                set(np.asarray(genes, dtype=object)[validation_mask])
            ),
            "validation_gene_ids": sorted(
                set(np.asarray(genes, dtype=object)[validation_mask])
            ),
            "validation_state_id_sha256": hashlib.sha256(
                "\n".join(
                    sorted(
                        str(report["states"][index]["state_id"])
                        for index in validation_indices
                    )
                ).encode("utf-8")
            ).hexdigest(),
            "label_free": True,
        },
        "architecture": {
            "input_feature_width": (
                CAPACITY_REPAIR_INPUT_WIDTH
                if args.capacity_repair
                else FEATURE_WIDTH
            ),
            "output_feature_width": (
                CAPACITY_REPAIR_OUTPUT_WIDTH
                if args.capacity_repair
                else FEATURE_WIDTH
            ),
            "active_features": ACTIVE_FEATURES,
            "derivative_evaluation": "native_mlp_input",
            "active_jacobian_shape": [
                ACTIVE_FEATURES,
                ACTIVE_FEATURES,
            ],
            "jacobian_metric_trained": "raw_l1",
            "posthoc_scale_sensitivity": "l1_over_rms_and_edge_mass_counts",
            "auxk_dead_feature_repair": effective_auxk_repair,
            "asymmetric_capacity_repair": args.capacity_repair,
            "auxiliary_features": (
                AUTOAUX_FEATURES if effective_auxk_repair else 0
            ),
            "auxiliary_coefficient": (
                AUTOAUX_COEFFICIENT if effective_auxk_repair else 0.0
            ),
        },
        "mechanical_preflight": {
            "maximum_mlp_replay_error": maximum_replay_error,
            "mlp_replay_tolerance": MLP_REPLAY_TOLERANCE,
            "all_autograd_controls_PASS": all(
                row["autograd_control"]["PASS"]
                for row in candidates.values()
            ),
            "all_decoder_invariants_PASS": all(
                row["decoder_invariants"]["PASS"]
                for row in candidates.values()
            ),
            "PASS": all_mechanical,
        },
        "candidates": candidates,
        "selection": selection,
        "downstream_behavior_opened": False,
        "task_feature_association_opened": False,
        "five_seed_replication_authorized": (
            all_mechanical and selection["PASS"]
        ),
        "licensed_claim": (
            "A seed-2020 checkpoint-specific J-SAE instrument passed its "
            "outcome-blind representation gates."
            if status == "INSTRUMENT_PASS"
            else "No J-SAE feature semantics or steering are authorized."
        ),
    }
    assert_label_free_payload(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_payload, model_output)
    result["model_artifact"] = {
        "path": str(model_output),
        "sha256": sha256_file(model_output),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        (
            f"splicebert_gtex_finetune_shift:jacobian_sae_instrument_v{version}"
        ),
        sys.argv,
        repo=REPO,
        status=status,
        config={
            "view": args.view,
            "input_feature_width": (
                CAPACITY_REPAIR_INPUT_WIDTH
                if args.capacity_repair
                else FEATURE_WIDTH
            ),
            "output_feature_width": (
                CAPACITY_REPAIR_OUTPUT_WIDTH
                if args.capacity_repair
                else FEATURE_WIDTH
            ),
            "active_features": ACTIVE_FEATURES,
            "jacobian_coefficients": list(JACOBIAN_COEFFICIENTS),
            "batch": args.batch,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "auxk_repair": effective_auxk_repair,
            "capacity_repair": args.capacity_repair,
        },
        seeds={"dictionary_initialization": INITIALIZATION_SEED},
        notes=[
            "No classifier output, margin, class label, or external outcome was read.",
            "The zero-Jacobian-penalty candidate is the matched TopK SAE-pair control.",
            "The exact active Jacobian is evaluated at the native MLP input.",
        ],
    )
    parent_protocols = [
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        f"PREREGISTRATION_JACOBIAN_SAE_V{parent_version}.md"
        for parent_version in range(1, version)
    ]
    for path in (
        protocol,
        *parent_protocols,
        *([parent_failure] if version > 1 else []),
        activation_path,
        Path(report["arrays"]["path"]),
        model_dir / "config.json",
        model_dir / "model.safetensors",
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.extend([file_record(output), file_record(model_output)])
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": status,
                "selection": selection,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
