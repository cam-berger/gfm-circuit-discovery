"""Exact final-layer mechanics for the native GTEx splice classifier.

The final GTEx mechanism study repeatedly intervenes inside encoder layer 5.  Running
layers 0--4 again for every candidate would waste most of the compute and make a
component scan needlessly contentious on shared hardware.  These helpers cache the
input to the final layer, replay that layer directly, and project its donor state
through the released classifier.

Every direct replay is checked against the ordinary full-model forward in preflight.
The helpers do not change the scientific intervention: hooks still act on the real
Hugging Face BERT modules and tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any

import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter


@dataclass(frozen=True)
class FinalLayerInputs:
    """Frozen input state and additive attention mask for one encoder layer."""

    hidden: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class FinalLayerRun:
    """Captured tensors from one direct final-layer replay."""

    hidden: torch.Tensor
    heads: torch.Tensor
    mlp_write: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention: torch.Tensor | None
    mlp_residual: torch.Tensor | None = None


def donor_token_positions(
    nucleotide_positions: list[int] | tuple[int, ...],
    *,
    prefix_tokens: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert zero-based nucleotide positions to one-token-prefix coordinates."""

    positions = torch.as_tensor(
        nucleotide_positions,
        dtype=torch.long,
        device=device,
    )
    if positions.ndim != 1 or torch.any(positions < 0):
        raise ValueError("nucleotide positions must be a non-negative 1D sequence")
    return positions + prefix_tokens


def position_mask(
    token_positions: torch.Tensor,
    sequence_length: int,
    *,
    selected_examples: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boolean ``[B,S]`` mask selecting one token per chosen example."""

    positions = token_positions.to(dtype=torch.long)
    if positions.ndim != 1:
        raise ValueError("token_positions must be 1D")
    if torch.any(positions < 0) or torch.any(positions >= sequence_length):
        raise ValueError("token position outside sequence")
    mask = torch.zeros(
        (positions.shape[0], sequence_length),
        dtype=torch.bool,
        device=positions.device,
    )
    rows = torch.arange(positions.shape[0], device=positions.device)
    if selected_examples is None:
        mask[rows, positions] = True
    else:
        selected = selected_examples.to(device=positions.device, dtype=torch.bool)
        if selected.shape != positions.shape:
            raise ValueError("selected_examples must match token_positions")
        mask[rows[selected], positions[selected]] = True
    return mask


def cache_final_layer_inputs(
    model: Any,
    encoded: Any,
    *,
    layer: int,
) -> FinalLayerInputs:
    """Run the ordinary encoder once and retain the input to ``layer``."""

    with torch.no_grad():
        output = model.bert(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = output.hidden_states
    if hidden_states is None or len(hidden_states) <= layer:
        raise RuntimeError("encoder did not return the requested layer input")
    hidden = hidden_states[layer].detach().clone()
    input_shape = tuple(encoded["input_ids"].shape)
    attention_mask = model.bert.get_extended_attention_mask(
        encoded["attention_mask"],
        input_shape,
    ).detach()
    return FinalLayerInputs(hidden=hidden, attention_mask=attention_mask)


def _first_tensor(output: Any) -> torch.Tensor:
    tensor = output if torch.is_tensor(output) else output[0]
    if not torch.is_tensor(tensor):
        raise TypeError("module output does not begin with a tensor")
    return tensor


def _split_heads(context: torch.Tensor, num_heads: int) -> torch.Tensor:
    if context.ndim != 3 or context.shape[-1] % num_heads:
        raise ValueError("attention context is not [B,S,H*Dh]")
    batch, sequence, width = context.shape
    head_dim = width // num_heads
    return (
        context.view(batch, sequence, num_heads, head_dim)
        .permute(0, 2, 1, 3)
        .detach()
        .clone()
    )


def replay_final_layer(
    adapter: SpliceBertAdapter,
    inputs: FinalLayerInputs,
    *,
    layer: int,
    output_attentions: bool = False,
) -> torch.Tensor:
    """Replay one real BERT layer and return its final hidden state."""

    with torch.no_grad():
        output = adapter.encoder_layers[layer](
            inputs.hidden,
            attention_mask=inputs.attention_mask,
            output_attentions=output_attentions,
        )
    return _first_tensor(output)


def capture_final_layer_run(
    adapter: SpliceBertAdapter,
    inputs: FinalLayerInputs,
    *,
    layer: int,
    output_attentions: bool = False,
) -> FinalLayerRun:
    """Replay one layer while capturing exact head, Q/K/V, and MLP tensors."""

    attention_module = adapter.attention_self_modules[layer]
    mlp_output = adapter.encoder_layers[layer].output
    mlp_dense = mlp_output.dense
    store: dict[str, torch.Tensor] = {}
    handles = []

    def capture_context(_module: Any, _inputs: Any, output: Any) -> None:
        store["heads"] = _split_heads(_first_tensor(output), adapter.num_heads)
        if (
            output_attentions
            and isinstance(output, (tuple, list))
            and len(output) > 1
            and torch.is_tensor(output[1])
        ):
            store["attention"] = output[1].detach().clone()

    def capture(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            store[name] = _first_tensor(output).detach().clone()

        return hook

    def capture_mlp_residual(_module: Any, inputs: Any) -> None:
        if len(inputs) < 2 or not torch.is_tensor(inputs[1]):
            raise RuntimeError("BERT MLP output did not receive its residual input")
        store["mlp_residual"] = inputs[1].detach().clone()

    handles.append(attention_module.register_forward_hook(capture_context))
    for name in ("query", "key", "value"):
        handles.append(getattr(attention_module, name).register_forward_hook(capture(name)))
    handles.append(mlp_dense.register_forward_hook(capture("mlp_write")))
    handles.append(mlp_output.register_forward_pre_hook(capture_mlp_residual))
    try:
        with torch.no_grad():
            output = adapter.encoder_layers[layer](
                inputs.hidden,
                attention_mask=inputs.attention_mask,
                output_attentions=output_attentions,
            )
    finally:
        for handle in handles:
            handle.remove()

    required = {
        "heads",
        "query",
        "key",
        "value",
        "mlp_write",
        "mlp_residual",
    }
    if output_attentions:
        required.add("attention")
    if set(store) != required:
        raise RuntimeError(f"missing final-layer captures: {sorted(required - set(store))}")
    attention = store.get("attention")
    hidden = _first_tensor(output).detach().clone()
    return FinalLayerRun(
        hidden=hidden,
        heads=store["heads"],
        mlp_write=store["mlp_write"],
        query=store["query"],
        key=store["key"],
        value=store["value"],
        attention=attention,
        mlp_residual=store["mlp_residual"],
    )


def compose_target_mlp_hidden(
    run: FinalLayerRun,
    token_positions: torch.Tensor,
    write_vectors: torch.Tensor,
    layer_norm: Any,
) -> torch.Tensor:
    """Compose target states from a residual and substituted final-MLP writes.

    There is no token mixing after the final BERT MLP. At one target token,
    ``LayerNorm(mlp_residual + mlp_write)`` is therefore algebraically
    equivalent to clamping the final MLP dense output at that position.
    """

    if run.mlp_residual is None:
        raise ValueError("run did not capture the MLP residual input")
    if token_positions.ndim != 1 or write_vectors.ndim != 2:
        raise ValueError("expected positions [B] and write vectors [B,D]")
    if (
        run.mlp_residual.shape[0] != token_positions.shape[0]
        or write_vectors.shape
        != (run.mlp_residual.shape[0], run.mlp_residual.shape[2])
    ):
        raise ValueError("target-vector batch or width mismatch")
    positions = token_positions.to(
        device=run.mlp_residual.device,
        dtype=torch.long,
    )
    if torch.any(positions < 0) or torch.any(
        positions >= run.mlp_residual.shape[1]
    ):
        raise ValueError("target position outside residual tensor")
    rows = torch.arange(run.mlp_residual.shape[0], device=positions.device)
    residual = run.mlp_residual[rows, positions]
    write = write_vectors.to(device=residual.device, dtype=residual.dtype)
    return layer_norm(residual + write)


def donor_margin(
    model: Any,
    hidden: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    """Project donor states through the released classifier, donor minus no-site."""

    if hidden.ndim != 3 or token_positions.ndim != 1:
        raise ValueError("hidden must be [B,S,D] and token_positions must be [B]")
    if hidden.shape[0] != token_positions.shape[0]:
        raise ValueError("batch mismatch between hidden states and token positions")
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    positions = token_positions.to(device=hidden.device)
    logits = model.classifier(model.dropout(hidden[rows, positions])).float()
    margin = logits[:, 1] - logits[:, 0]
    if not torch.isfinite(margin).all():
        raise RuntimeError("non-finite native donor margin")
    return margin


def attention_context_reconstruction_error(
    run: FinalLayerRun,
    *,
    num_heads: int,
) -> float:
    """Maximum error between eager attention ``A @ V`` and returned head context."""

    if run.attention is None:
        raise ValueError("attention probabilities were not captured")
    batch, sequence, width = run.value.shape
    if width % num_heads:
        raise ValueError("value width is not divisible by num_heads")
    head_dim = width // num_heads
    values = (
        run.value.view(batch, sequence, num_heads, head_dim)
        .permute(0, 2, 1, 3)
    )
    reconstructed = torch.matmul(run.attention.float(), values.float())
    return float((reconstructed - run.heads.float()).abs().max().detach().cpu())


def select_head_candidates(
    summaries: dict[int, dict[str, dict[str, float]]],
    *,
    min_effect: float = 0.05,
    max_q: float = 0.10,
    max_candidates: int = 3,
    q_field: str = "q_bh_32_tests",
) -> list[int]:
    """Apply the frozen bidirectional candidate rule and deterministic ranking."""

    eligible: list[tuple[float, int]] = []
    for head, result in summaries.items():
        damage = result["damage"]
        recovery = result["recovery"]
        passes = all(
            (
                float(damage["estimate"]) >= min_effect,
                float(recovery["estimate"]) >= min_effect,
                float(damage["ci_low"]) > 0.0,
                float(recovery["ci_low"]) > 0.0,
                float(damage[q_field]) <= max_q,
                float(recovery[q_field]) <= max_q,
            )
        )
        if passes:
            rank_score = min(
                float(damage["estimate"]),
                float(recovery["estimate"]),
            )
            eligible.append((rank_score, int(head)))
    eligible.sort(key=lambda item: (-item[0], item[1]))
    return [head for _, head in eligible[:max_candidates]]


def deterministic_control_heads(
    candidates: list[int] | tuple[int, ...],
    *,
    num_heads: int,
    controls_per_candidate: int = 3,
    tag: str = "gtex-native-mechanism-controls-v1",
) -> dict[int, list[int]]:
    """Draw non-candidate equal-size controls without looking at their outcomes."""

    selected = [int(head) for head in candidates]
    if len(set(selected)) != len(selected):
        raise ValueError("candidate heads must be unique")
    if any(head < 0 or head >= num_heads for head in selected):
        raise ValueError("candidate head outside architecture")
    required = len(selected) * controls_per_candidate
    pool = [head for head in range(num_heads) if head not in selected]
    if required > len(pool):
        raise ValueError("not enough unique non-candidate control heads")
    seed = int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(pool)
    return {
        candidate: pool[
            index * controls_per_candidate : (index + 1) * controls_per_candidate
        ]
        for index, candidate in enumerate(selected)
    }


def orthogonal_procrustes_cosine(x: Any, y: Any) -> float:
    """Cosine after the optimal orthogonal alignment of centered row representations."""

    import numpy as np

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if (
        x_array.ndim != 2
        or y_array.ndim != 2
        or x_array.shape != y_array.shape
    ):
        raise ValueError("Procrustes cosine requires matched 2D matrices")
    x_centered = x_array - x_array.mean(axis=0, keepdims=True)
    y_centered = y_array - y_array.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(x_centered.T @ y_centered, full_matrices=False)
    aligned = x_centered @ (u @ vt)
    denominator = np.linalg.norm(aligned) * np.linalg.norm(y_centered)
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("Procrustes cosine is undefined for zero-variance inputs")
    return float(np.sum(aligned * y_centered) / denominator)


__all__ = [
    "FinalLayerInputs",
    "FinalLayerRun",
    "attention_context_reconstruction_error",
    "cache_final_layer_inputs",
    "capture_final_layer_run",
    "compose_target_mlp_hidden",
    "deterministic_control_heads",
    "donor_margin",
    "donor_token_positions",
    "orthogonal_procrustes_cosine",
    "position_mask",
    "replay_final_layer",
    "select_head_candidates",
]
