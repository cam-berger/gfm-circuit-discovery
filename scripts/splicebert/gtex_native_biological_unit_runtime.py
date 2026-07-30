"""Model runtime shared by native-unit fallback discovery and confirmation."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.circuits.hf_bert import patch_heads
from gfmcd.provenance import sha256_file
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier
from gfmcd.splicing.native_classifier import NativeClassifierTask
from gfmcd.splicing.native_mechanism import (
    cache_final_layer_inputs,
    capture_final_layer_run,
    donor_token_positions,
    position_mask,
)
from gfmcd.splicing.mlp_units import (
    composed_classifier_margin,
    unit_group_write_delta,
)

from gtex_native_biological_unit_fallback import (
    ENDPOINT_LABELS,
    EXPECTED_SEEDS,
    INTERMEDIATE_SIZE,
    LOCK_SCHEMA,
)


FINAL_LAYER = 5
HIDDEN_SIZE = 512
NUM_HEADS = 16
WRITE_TOLERANCE = 5e-4
COMPOSITION_TOLERANCE = 1e-5


def parse_checkpoints(values: Sequence[str]) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in checkpoints:
            raise ValueError(f"duplicate checkpoint seed {seed}")
        checkpoints[seed] = Path(path_text).expanduser().resolve()
    if tuple(sorted(checkpoints)) != EXPECTED_SEEDS:
        raise ValueError("complete checkpoint family 2020--2024 required")
    return checkpoints


def read_lock(path: str | Path) -> dict[str, Any]:
    lock = json.loads(Path(path).read_text())
    if (
        lock.get("schema") != LOCK_SCHEMA
        or lock.get("status") != "FROZEN_LOCK_C"
        or lock.get("model_loaded") is not False
        or lock.get("classifier_outcomes_read") is not False
        or lock.get("sparse_weights_used") is not False
    ):
        raise RuntimeError("native-unit fallback Lock C is invalid")
    return lock


def verify_file(path: str | Path, record: Mapping[str, Any]) -> Path:
    resolved = Path(path).expanduser().resolve()
    if str(resolved) != record["path"] or sha256_file(resolved) != record["sha256"]:
        raise RuntimeError(f"frozen input identity mismatch: {resolved}")
    return resolved


def verify_checkpoints(
    checkpoints: Mapping[int, Path],
    lock: Mapping[str, Any],
) -> None:
    for seed, path in checkpoints.items():
        verify_file(path, lock["checkpoints"][str(seed)])


def parse_heads(values: Sequence[str]) -> tuple[int, ...]:
    heads = tuple(int(str(value).split("H", 1)[1]) for value in values)
    if len(heads) != 3 or len(set(heads)) != 3 or any(
        value < 0 or value >= NUM_HEADS for value in heads
    ):
        raise ValueError("head set must be three unique L5 heads")
    return heads


def target_rows(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[rows, positions.to(device=tensor.device)]


def native_margin(
    model: Any,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    *,
    endpoint: str,
) -> torch.Tensor:
    target = target_rows(hidden, positions)
    logits = model.classifier(model.dropout(target)).float()
    label = ENDPOINT_LABELS[endpoint]
    margin = logits[:, label] - logits[:, 0]
    if not torch.isfinite(margin).all():
        raise RuntimeError("non-finite native endpoint margin")
    return margin


def classifier_parameters(model: Any, layer: Any, endpoint: str) -> dict[str, Any]:
    label = ENDPOINT_LABELS[endpoint]
    return {
        "layer_norm_weight": (
            layer.output.LayerNorm.weight.detach().cpu().double().numpy()
        ),
        "layer_norm_bias": (
            layer.output.LayerNorm.bias.detach().cpu().double().numpy()
        ),
        "layer_norm_eps": float(layer.output.LayerNorm.eps),
        "classifier_direction": (
            model.classifier.weight[label] - model.classifier.weight[0]
        )
        .detach()
        .cpu()
        .double()
        .numpy(),
        "classifier_bias": float(
            (model.classifier.bias[label] - model.classifier.bias[0])
            .detach()
            .cpu()
        ),
    }


def _state_arrays(run: Any, positions: torch.Tensor, margin: torch.Tensor) -> dict[str, np.ndarray]:
    if run.mlp_residual is None or run.mlp_intermediate is None:
        raise RuntimeError("final MLP input/post-GELU capture is incomplete")
    return {
        "mlp_input": target_rows(run.mlp_residual, positions)
        .detach()
        .cpu()
        .float()
        .numpy(),
        "activation": target_rows(run.mlp_intermediate, positions)
        .detach()
        .cpu()
        .float()
        .numpy(),
        "mlp_write": target_rows(run.mlp_write, positions)
        .detach()
        .cpu()
        .float()
        .numpy(),
        "margin": margin.detach().cpu().double().numpy(),
    }


def capture_family(
    model: Any,
    adapter: SpliceBertAdapter,
    task: NativeClassifierTask,
    pairs: Sequence[Mapping[str, Any]],
    *,
    endpoint: str,
    candidate_heads: Sequence[int],
    control_head_draws: Sequence[Sequence[int]],
    batch_size: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Capture low/high states and cross-state head-induced activations."""

    buffers: dict[str, dict[str, list[np.ndarray]]] = {
        state: {
            field: []
            for field in (
                "mlp_input",
                "activation",
                "mlp_write",
                "margin",
                "candidate_head_patched_activation",
                *(
                    f"control_head_patched_activation_{draw}"
                    for draw in range(len(control_head_draws))
                ),
            )
        }
        for state in ("low", "high")
    }
    max_write_error = 0.0
    output_dense = adapter.encoder_layers[FINAL_LAYER].output.dense
    output_weight = output_dense.weight.detach().cpu().double().numpy()
    output_bias = output_dense.bias.detach().cpu().double().numpy()
    candidate_nodes = [(FINAL_LAYER, int(head)) for head in candidate_heads]
    control_nodes = [
        [(FINAL_LAYER, int(head)) for head in draw] for draw in control_head_draws
    ]
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        state_objects: dict[str, tuple[Any, Any, torch.Tensor, Any]] = {}
        for state in ("low", "high"):
            encoded = task.encode([str(pair[f"{state}_sequence"]) for pair in chunk])
            positions = donor_token_positions(
                [int(pair["target_position"]) for pair in chunk],
                prefix_tokens=task.prefix_tokens,
                device=device,
            )
            inputs = cache_final_layer_inputs(model, encoded, layer=FINAL_LAYER)
            run = capture_final_layer_run(adapter, inputs, layer=FINAL_LAYER)
            margin = native_margin(model, run.hidden, positions, endpoint=endpoint)
            values = _state_arrays(run, positions, margin)
            for field, value in values.items():
                buffers[state][field].append(value)
            reconstruction = values["activation"].astype(np.float64) @ output_weight.T
            reconstruction += output_bias
            max_write_error = max(
                max_write_error,
                float(
                    np.max(
                        np.abs(
                            reconstruction - values["mlp_write"].astype(np.float64)
                        )
                    )
                ),
            )
            state_objects[state] = (encoded, inputs, positions, run)
        for state, source in (("low", "high"), ("high", "low")):
            encoded, inputs, positions, _run = state_objects[state]
            source_heads = state_objects[source][3].heads
            mask = position_mask(
                positions,
                int(encoded["input_ids"].shape[1]),
            )
            with patch_heads(
                adapter,
                {FINAL_LAYER: source_heads},
                candidate_nodes,
                token_mask=mask,
            ):
                patched = capture_final_layer_run(adapter, inputs, layer=FINAL_LAYER)
            if patched.mlp_intermediate is None:
                raise RuntimeError("candidate-head patched activation was not captured")
            buffers[state]["candidate_head_patched_activation"].append(
                target_rows(patched.mlp_intermediate, positions)
                .detach()
                .cpu()
                .float()
                .numpy()
            )
            for draw, nodes in enumerate(control_nodes):
                with patch_heads(
                    adapter,
                    {FINAL_LAYER: source_heads},
                    nodes,
                    token_mask=mask,
                ):
                    control = capture_final_layer_run(
                        adapter, inputs, layer=FINAL_LAYER
                    )
                if control.mlp_intermediate is None:
                    raise RuntimeError("control-head patched activation was not captured")
                buffers[state][f"control_head_patched_activation_{draw}"].append(
                    target_rows(control.mlp_intermediate, positions)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
    arrays = {
        state: {
            field: np.concatenate(values, axis=0)
            for field, values in fields.items()
        }
        for state, fields in buffers.items()
    }
    return arrays, {"write_reconstruction_max_abs_error": max_write_error}


def preflight_family(
    states: Mapping[str, Mapping[str, np.ndarray]],
    output_weight: np.ndarray,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    composed_errors = []
    for state in ("low", "high"):
        composed = composed_classifier_margin(
            states[state]["mlp_input"],
            states[state]["mlp_write"],
            **dict(parameters),
        )
        composed_errors.append(
            float(np.max(np.abs(composed - states[state]["margin"])))
        )
    activation_delta = states["high"]["activation"] - states["low"]["activation"]
    all_units = unit_group_write_delta(
        activation_delta, output_weight, range(INTERMEDIATE_SIZE)
    )
    native_delta = states["high"]["mlp_write"] - states["low"]["mlp_write"]
    all_unit_error = float(np.max(np.abs(all_units - native_delta)))
    zero_error = float(
        np.max(
            np.abs(unit_group_write_delta(activation_delta, output_weight, []))
        )
    )
    composition_error = max(composed_errors)
    passed = (
        composition_error <= COMPOSITION_TOLERANCE
        and all_unit_error <= WRITE_TOLERANCE
        and zero_error == 0.0
    )
    if not passed:
        raise RuntimeError(
            "native-unit family preflight failed: "
            f"composition={composition_error:.9g}, "
            f"all_units={all_unit_error:.9g}, zero={zero_error:.9g}"
        )
    return {
        "composition_max_abs_error": composition_error,
        "all_unit_delta_max_abs_error": all_unit_error,
        "zero_population_max_abs_error": zero_error,
        "PASS": True,
    }


def load_checkpoint_runtime(
    base_model: Path,
    checkpoint: Path,
    *,
    seed: int,
    device: str,
    batch_size: int,
) -> tuple[Any, Any, NativeClassifierTask, SpliceBertAdapter, Any]:
    model, tokenizer, report = load_gtex_token_classifier(
        base_model,
        checkpoint,
        device=device,
        attn_implementation="eager",
    )
    task = NativeClassifierTask(
        model, tokenizer, device=device, batch_size=batch_size
    )
    adapter = SpliceBertAdapter(
        model, tokenizer, source=str(checkpoint), device=device
    )
    dense = adapter.encoder_layers[FINAL_LAYER].output.dense
    if (
        report.seed != seed
        or adapter.num_layers != 6
        or adapter.num_heads != NUM_HEADS
        or adapter.hidden_size != HIDDEN_SIZE
        or dense.in_features != INTERMEDIATE_SIZE
        or dense.out_features != HIDDEN_SIZE
    ):
        raise RuntimeError("checkpoint/architecture invariant failed")
    return model, tokenizer, task, adapter, report


def release_checkpoint(
    objects: Sequence[Any],
    *,
    device: str,
) -> None:
    del objects
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


__all__ = [
    "FINAL_LAYER",
    "capture_family",
    "classifier_parameters",
    "load_checkpoint_runtime",
    "parse_checkpoints",
    "parse_heads",
    "preflight_family",
    "read_lock",
    "release_checkpoint",
    "verify_checkpoints",
    "verify_file",
]
