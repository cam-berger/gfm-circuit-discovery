#!/usr/bin/env python
"""Train the frozen larger paired-delta J-SAE factorial."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.circuits.jacobian_sae import (  # noqa: E402
    JacobianSAEConfig,
    JacobianSAEPair,
    auxiliary_dead_feature_loss,
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


def load_script(name: str) -> Any:
    path = REPO / "scripts/splicebert" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    spec.loader.exec_module(module)
    return module


INSTRUMENT = load_script("run_gtex_jacobian_sae_instrument.py")
REPLICATION = load_script("run_gtex_jacobian_sae_replication.py")

TUNED_SEEDS = tuple(range(2020, 2025))
VIEWS = (*map(str, TUNED_SEEDS), "base")
FROZEN_ACTIVATION_SHA256 = (
    "142ce4ee5976970b9b296685ef403cce30e56f440a87d7222a1933fe09d96b5b"
)
FROZEN_ARRAY_SHA256 = (
    "fa6c972f4765d5c57354abeb262b3e62d6ac3ff8c69e5850a1511c7cf054d559"
)
FROZEN_CHECKPOINT_HASHES = {
    "2020": "0bf47fd1f656f753a487fe2556ba063d6acdb1439c8d4f0c9857ea1e41438811",
    "2021": "a111a959c1a00cb2f794d3dd92d8468310ae2c6ad47ab887cd2bce2b5c3928fe",
    "2022": "a28b286fd199747d44d604d95f3f7219067b5b9678396903559d8f0a15edb598",
    "2023": "d24180427adaaf2eac0b78ee2bb2d7c47370fdf998ac4378823537e815b53187",
    "2024": "0c7ea06d3d16285670c21d38d76bf9cdce5962d7463bf30c03ab278c4db214e2",
    "base": "2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d",
}
INITIALIZATIONS = (0, 1, 2)
INPUT_FEATURES = 512
OUTPUT_FEATURES = 128
ACTIVE_FEATURES = 16
JACOBIAN_COEFFICIENT = 0.25
DELTA_COEFFICIENT = 1.0
STATE_BATCH = 64
PAIR_BATCH = 32
EPOCHS = 40
LEARNING_RATE = 5e-4
GRADIENT_CLIP = 1.0
AUXILIARY_FEATURES = 64
AUXILIARY_COEFFICIENT = 1 / 32
LATE_AUXILIARY_FRACTION_MAXIMUM = 0.25
DELTA_POWER_EPS = 1e-8
FINITE_DIFFERENCE_DOSE = 0.01
FINITE_DIFFERENCE_CORRELATION_MINIMUM = 0.95
FINITE_DIFFERENCE_SIGN_MINIMUM = 0.95
FINITE_DIFFERENCE_RELATIVE_ERROR_MAXIMUM = 0.10
EFFECTIVE_EDGE_REDUCTION_MINIMUM = 0.10
E90_REDUCTION_MINIMUM = 0.15
CONDITIONS = {
    "delta_jacobian": {
        "delta_coefficient": DELTA_COEFFICIENT,
        "jacobian_coefficient": JACOBIAN_COEFFICIENT,
    },
    "delta_no_jacobian": {
        "delta_coefficient": DELTA_COEFFICIENT,
        "jacobian_coefficient": 0.0,
    },
    "no_delta_jacobian": {
        "delta_coefficient": 0.0,
        "jacobian_coefficient": JACOBIAN_COEFFICIENT,
    },
    "plain_topk": {
        "delta_coefficient": 0.0,
        "jacobian_coefficient": 0.0,
    },
}
BASE_CONDITIONS = ("delta_jacobian", "delta_no_jacobian")
PAIR_FAMILIES = (
    "target_shuffle",
    "sham_shuffle",
    "max_recode",
    "synonymous_single",
)
ENDPOINTS = ("donor", "acceptor")
FROZEN_COUNTS = {
    "states": 7_991,
    "training_states": 6_530,
    "validation_states": 1_461,
    "training_pairs": 6_320,
    "validation_pairs": 1_415,
    "training_genes": 105,
    "validation_genes": 23,
}


def parse_model_dirs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        view, path_text = value.split("=", 1)
        if view in result:
            raise ValueError(f"duplicate model view {view}")
        result[view] = Path(path_text).expanduser().resolve()
    if set(result) != set(VIEWS):
        raise ValueError("model directories require base and seeds 2020--2024")
    return result


def checkpoint_weight_path(model_dir: Path) -> Path:
    path = model_dir / "model.safetensors"
    if not path.is_file():
        path = model_dir / "pytorch_model.bin"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint weights missing under {model_dir}")
    return path


def validate_frozen_sources(
    activation_path: Path,
    activation_report: Mapping[str, Any],
    model_dirs: Mapping[str, Path],
) -> dict[str, str]:
    if sha256_file(activation_path) != FROZEN_ACTIVATION_SHA256:
        raise RuntimeError("activation report is not the frozen large-J-SAE source")
    arrays_path = Path(activation_report["arrays"]["path"]).resolve()
    if (
        activation_report["arrays"].get("sha256") != FROZEN_ARRAY_SHA256
        or sha256_file(arrays_path) != FROZEN_ARRAY_SHA256
    ):
        raise RuntimeError("activation arrays are not the frozen large-J-SAE source")
    if int(activation_report.get("n_states", -1)) != FROZEN_COUNTS["states"]:
        raise RuntimeError("activation state count does not match the frozen source")
    if tuple(activation_report.get("views", ())) != (
        "base",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
    ):
        raise RuntimeError("activation views do not match the frozen family")
    observed = {
        view: sha256_file(checkpoint_weight_path(model_dirs[view]))
        for view in VIEWS
    }
    if observed != FROZEN_CHECKPOINT_HASHES:
        raise RuntimeError("checkpoint family does not match the frozen hashes")
    return observed


def pair_family(variant_kind: str) -> str:
    if variant_kind.startswith("max_recode_draw"):
        return "max_recode"
    if variant_kind in PAIR_FAMILIES:
        return variant_kind
    raise ValueError(f"unsupported paired variant family {variant_kind}")


def build_reference_pairs(
    states: Sequence[Mapping[str, Any]],
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    if mask.shape != (len(states),) or mask.dtype != bool:
        raise ValueError("pair mask must be boolean and match states")
    state_index = {
        str(row["state_id"]): index for index, row in enumerate(states)
    }
    pairs: list[dict[str, Any]] = []
    for changed, row in enumerate(states):
        reference_id = row.get("reference_state_id")
        if reference_id is None or not mask[changed]:
            continue
        if str(reference_id) not in state_index:
            raise ValueError("paired state has a missing reference")
        reference = state_index[str(reference_id)]
        if not mask[reference]:
            raise ValueError("reference/changed pair crosses the gene split")
        clean = states[reference]
        if (
            str(clean["variant_kind"]) != "clean"
            or str(clean["gene_id"]) != str(row["gene_id"])
            or str(clean["endpoint"]) != str(row["endpoint"])
        ):
            raise ValueError("reference/changed identity mismatch")
        pairs.append(
            {
                "reference": reference,
                "changed": changed,
                "gene_id": str(row["gene_id"]),
                "endpoint": str(row["endpoint"]),
                "family": pair_family(str(row["variant_kind"])),
                "state_id": str(row["state_id"]),
            }
        )
    observed = {
        (str(pair["endpoint"]), str(pair["family"])) for pair in pairs
    }
    expected = {
        (endpoint, family)
        for endpoint in ENDPOINTS
        for family in PAIR_FAMILIES
    }
    if observed != expected:
        raise RuntimeError(f"paired-state strata mismatch: {observed ^ expected}")
    return pairs


def stable_seed(seed: int, *parts: object) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    )


def fixed_state_epoch(
    train_indices: np.ndarray,
    genes: list[str],
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Return complete fixed-size batches from the frozen gene-balanced sampler."""

    if draws < len(train_indices):
        raise ValueError("state schedule cannot drop the frozen epoch")
    order = INSTRUMENT.gene_balanced_epoch(train_indices, genes, seed=seed)
    if len(order) != len(train_indices):
        raise RuntimeError("gene-balanced epoch returned an unexpected size")
    chunks = [order]
    remaining = draws - len(order)
    cycle = 0
    while remaining:
        extra = INSTRUMENT.gene_balanced_epoch(
            train_indices,
            genes,
            seed=stable_seed(seed, "state-padding", cycle),
        )
        chunks.append(extra[:remaining])
        remaining -= min(remaining, len(extra))
        cycle += 1
    order = np.concatenate(chunks)
    if len(order) != draws:
        raise RuntimeError("fixed state schedule has the wrong size")
    return order


def balanced_pair_schedule(
    pairs: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Balance endpoint/family, then genes, while cycling pairs."""

    if draws < len(pairs):
        raise ValueError("schedule must be long enough to cover every pair")
    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pair_index, pair in enumerate(pairs):
        grouped[(str(pair["endpoint"]), str(pair["family"]))][
            str(pair["gene_id"])
        ].append(pair_index)
    strata = sorted(grouped)
    expected = sorted(
        (endpoint, family)
        for endpoint in ENDPOINTS
        for family in PAIR_FAMILIES
    )
    if strata != expected:
        raise RuntimeError("pair schedule lacks a frozen endpoint/family stratum")

    stratum_counts: dict[tuple[str, str], int] = defaultdict(int)
    gene_counts: dict[tuple[tuple[str, str], str], int] = defaultdict(int)
    chosen: list[int] = []
    for draw in range(draws):
        stratum_cycle, stratum_position = divmod(draw, len(strata))
        stratum_order = np.random.default_rng(
            stable_seed(seed, "stratum", stratum_cycle)
        ).permutation(len(strata))
        stratum = strata[int(stratum_order[stratum_position])]
        genes = sorted(grouped[stratum])
        occurrence = stratum_counts[stratum]
        gene_cycle, gene_position = divmod(occurrence, len(genes))
        gene_order = np.random.default_rng(
            stable_seed(seed, "gene", *stratum, gene_cycle)
        ).permutation(len(genes))
        gene = genes[int(gene_order[gene_position])]
        candidates = grouped[stratum][gene]
        gene_occurrence = gene_counts[(stratum, gene)]
        pair_cycle, pair_position = divmod(gene_occurrence, len(candidates))
        pair_order = np.random.default_rng(
            stable_seed(seed, "pair", *stratum, gene, pair_cycle)
        ).permutation(len(candidates))
        pair_index = candidates[int(pair_order[pair_position])]
        chosen.append(pair_index)
        stratum_counts[stratum] += 1
        gene_counts[(stratum, gene)] += 1

    selected = np.asarray(
        [
            [int(pairs[index]["reference"]), int(pairs[index]["changed"])]
            for index in chosen
        ],
        dtype=np.int64,
    )
    counts = {
        f"{endpoint}:{family}": int(
            sum(
                1
                for index in chosen
                if pairs[index]["endpoint"] == endpoint
                and pairs[index]["family"] == family
            )
        )
        for endpoint, family in strata
    }
    coverage = len(set(chosen)) / len(pairs)
    return selected, {
        "draws": draws,
        "unique_pairs_seen": len(set(chosen)),
        "total_pairs": len(pairs),
        "coverage_fraction": coverage,
        "counts_by_stratum": counts,
        "PASS": coverage == 1.0 and max(counts.values()) - min(counts.values()) <= 1,
    }


def build_large_pair(
    weights: Mapping[str, torch.Tensor],
    training_inputs: torch.Tensor,
    training_writes: torch.Tensor,
    *,
    device: torch.device,
) -> JacobianSAEPair:
    input_center, input_scale = scalar_normalization(training_inputs)
    output_center, output_scale = scalar_normalization(training_writes)
    config = JacobianSAEConfig(
        input_dim=training_inputs.shape[1],
        output_dim=training_writes.shape[1],
        mlp_dim=weights["input_weight"].shape[0],
        num_features=INPUT_FEATURES,
        output_num_features=OUTPUT_FEATURES,
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
        if not name.startswith("mlp_")
    }


def initialization_seed(view: str, initialization: int) -> int:
    if view not in VIEWS or initialization not in INITIALIZATIONS:
        raise ValueError("unknown view or initialization")
    view_number = 2019 if view == "base" else int(view)
    return 4_000_000 + 10_000 * view_number + initialization


def delta_powers(
    model: JacobianSAEPair,
    inputs: torch.Tensor,
    writes: torch.Tensor,
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    by_gene: dict[
        tuple[str, str], dict[str, list[tuple[float, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    input_scale = float(model.input_scale.detach().cpu())
    output_scale = float(model.output_scale.detach().cpu())
    for pair in pairs:
        reference = int(pair["reference"])
        changed = int(pair["changed"])
        dx = (
            inputs[changed].double() - inputs[reference].double()
        ) / input_scale
        dy = (
            writes[changed].double() - writes[reference].double()
        ) / output_scale
        by_gene[(str(pair["endpoint"]), str(pair["family"]))][
            str(pair["gene_id"])
        ].append(
            (float(torch.mean(dx.square())), float(torch.mean(dy.square())))
        )
    stratum_powers = [
        (
            float(
                np.mean(
                    [
                        np.mean([value[0] for value in values])
                        for values in genes.values()
                    ]
                )
            ),
            float(
                np.mean(
                    [
                        np.mean([value[1] for value in values])
                        for values in genes.values()
                    ]
                )
            ),
        )
        for genes in by_gene.values()
    ]
    input_power = float(
        np.mean([value[0] for value in stratum_powers])
    )
    output_power = float(
        np.mean([value[1] for value in stratum_powers])
    )
    if (
        not np.isfinite(input_power)
        or not np.isfinite(output_power)
        or min(input_power, output_power) <= DELTA_POWER_EPS
    ):
        raise RuntimeError("training-pair delta power is non-finite or too small")
    return {
        "input": input_power,
        "output": output_power,
        "formula": (
            "equal endpoint/family mean of equal-gene means of normalized "
            "squared changed-reference delta"
        ),
    }


def paired_delta_loss(
    model: JacobianSAEPair,
    inputs: torch.Tensor,
    writes: torch.Tensor,
    pair_indices: np.ndarray,
    powers: Mapping[str, float],
) -> torch.Tensor:
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("pair indices must be [pairs, reference/changed]")
    device = model.input_center.device
    reference = torch.as_tensor(pair_indices[:, 0], dtype=torch.long)
    changed = torch.as_tensor(pair_indices[:, 1], dtype=torch.long)
    x_reference = inputs[reference].to(device)
    x_changed = inputs[changed].to(device)
    y_reference = writes[reference].to(device)
    y_changed = writes[changed].to(device)
    x_reference_norm = model.normalize_input(x_reference)
    x_changed_norm = model.normalize_input(x_changed)
    y_reference_norm = model.normalize_output(y_reference)
    y_changed_norm = model.normalize_output(y_changed)
    x_reference_hat, _, _ = model.input_sae(x_reference_norm)
    x_changed_hat, _, _ = model.input_sae(x_changed_norm)
    y_reference_hat, _, _ = model.output_sae(y_reference_norm)
    y_changed_hat, _, _ = model.output_sae(y_changed_norm)
    input_error = (
        (x_changed_hat - x_reference_hat)
        - (x_changed_norm - x_reference_norm)
    )
    output_error = (
        (y_changed_hat - y_reference_hat)
        - (y_changed_norm - y_reference_norm)
    )
    return 0.5 * (
        torch.mean(input_error.square()) / float(powers["input"])
        + torch.mean(output_error.square()) / float(powers["output"])
    )


def validation_objective(
    model: JacobianSAEPair,
    inputs: torch.Tensor,
    writes: torch.Tensor,
    validation_indices: np.ndarray,
    validation_pairs: Sequence[Mapping[str, Any]],
    powers: Mapping[str, float],
    *,
    delta_coefficient: float,
) -> dict[str, float]:
    reconstruction = INSTRUMENT.reconstruction_loss(
        model,
        inputs,
        writes,
        validation_indices,
        batch_size=128,
    ) / 2.0
    grouped: dict[tuple[str, str], list[list[int]]] = defaultdict(list)
    for pair in validation_pairs:
        grouped[(str(pair["endpoint"]), str(pair["family"]))].append(
            [int(pair["reference"]), int(pair["changed"])]
        )
    losses = []
    with torch.no_grad():
        for values in grouped.values():
            indices = np.asarray(values, dtype=np.int64)
            losses.append(
                float(
                    paired_delta_loss(
                        model,
                        inputs,
                        writes,
                        indices,
                        powers,
                    )
                    .detach()
                    .cpu()
                )
            )
    delta = float(np.mean(losses))
    return {
        "state_reconstruction": reconstruction,
        "paired_delta": delta,
        "selection_objective": reconstruction + delta_coefficient * delta,
    }


def train_condition(
    weights: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    writes: torch.Tensor,
    genes: list[str],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    train_pairs: Sequence[Mapping[str, Any]],
    validation_pairs: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    device: torch.device,
    seed: int,
) -> tuple[JacobianSAEPair, dict[str, Any]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    settings = CONDITIONS[condition]
    torch.manual_seed(seed)
    model = build_large_pair(
        weights,
        inputs[train_indices],
        writes[train_indices],
        device=device,
    )
    initial_hash = state_hash(model)
    powers = delta_powers(model, inputs, writes, train_pairs)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
    )
    steps_per_epoch = math.ceil(len(train_indices) / STATE_BATCH)
    total_steps = EPOCHS * steps_per_epoch
    pair_schedule, schedule_report = balanced_pair_schedule(
        train_pairs,
        draws=EPOCHS * steps_per_epoch * PAIR_BATCH,
        seed=seed,
    )
    if not schedule_report["PASS"]:
        raise RuntimeError("pair schedule failed balance or coverage")
    input_since_fired = torch.zeros(
        INPUT_FEATURES, dtype=torch.long, device=device
    )
    output_since_fired = torch.zeros(
        OUTPUT_FEATURES, dtype=torch.long, device=device
    )
    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        ordinary_order = fixed_state_epoch(
            train_indices,
            genes,
            draws=steps_per_epoch * STATE_BATCH,
            seed=seed + 10_000 * epoch,
        )
        epoch_values: dict[str, list[float]] = defaultdict(list)
        for step in range(steps_per_epoch):
            start = step * STATE_BATCH
            selected = ordinary_order[start : start + STATE_BATCH]
            pair_start = (
                (epoch * steps_per_epoch + step) * PAIR_BATCH
            )
            selected_pairs = pair_schedule[
                pair_start : pair_start + PAIR_BATCH
            ]
            x = inputs[selected].to(device)
            y = writes[selected].to(device)
            optimizer.param_groups[0]["lr"] = INSTRUMENT.scheduled_learning_rate(
                global_step, total_steps
            )
            current_jacobian_coefficient = (
                INSTRUMENT.scheduled_jacobian_coefficient(
                    float(settings["jacobian_coefficient"]),
                    global_step,
                    total_steps,
                )
            )
            optimizer.zero_grad(set_to_none=True)
            x_norm = model.normalize_input(x)
            y_norm = model.normalize_output(y)
            x_hat, input_acts, input_indices = model.input_sae(x_norm)
            y_hat, output_acts, output_indices = model.output_sae(y_norm)
            reconstruction = 0.5 * (
                torch.mean((x_hat - x_norm).square())
                + torch.mean((y_hat - y_norm).square())
            )
            if float(settings["delta_coefficient"]) > 0:
                delta = paired_delta_loss(
                    model,
                    inputs,
                    writes,
                    selected_pairs,
                    powers,
                )
            else:
                delta = reconstruction.new_zeros(())
            if current_jacobian_coefficient > 0:
                jacobian = model.active_jacobian(
                    x, input_indices, output_indices
                )
                jacobian_penalty = jacobian.abs().mean()
            else:
                jacobian_penalty = reconstruction.new_zeros(())
            with torch.no_grad():
                input_fired = (input_acts > 1e-3).any(dim=0)
                output_fired = (output_acts > 1e-3).any(dim=0)
                input_since_fired.copy_(
                    torch.where(
                        input_fired,
                        torch.zeros_like(input_since_fired),
                        input_since_fired + 1,
                    )
                )
                output_since_fired.copy_(
                    torch.where(
                        output_fired,
                        torch.zeros_like(output_since_fired),
                        output_since_fired + 1,
                    )
                )
                input_dead = input_since_fired > steps_per_epoch
                output_dead = output_since_fired > steps_per_epoch
            input_auxiliary = auxiliary_dead_feature_loss(
                model.input_sae,
                x_norm,
                x_hat,
                input_dead,
                auxiliary_features=AUXILIARY_FEATURES,
            )
            output_auxiliary = auxiliary_dead_feature_loss(
                model.output_sae,
                y_norm,
                y_hat,
                output_dead,
                auxiliary_features=AUXILIARY_FEATURES,
            )
            auxiliary = AUXILIARY_COEFFICIENT * (
                input_auxiliary + output_auxiliary
            )
            loss = (
                reconstruction
                + float(settings["delta_coefficient"]) * delta
                + current_jacobian_coefficient * jacobian_penalty
                + auxiliary
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            for key, value in (
                ("loss", loss),
                ("reconstruction", reconstruction),
                ("delta", delta),
                ("jacobian", jacobian_penalty),
                ("auxiliary", auxiliary),
            ):
                epoch_values[key].append(float(value.detach().cpu()))
            global_step += 1
        validation = validation_objective(
            model,
            inputs,
            writes,
            validation_indices,
            validation_pairs,
            powers,
            delta_coefficient=float(settings["delta_coefficient"]),
        )
        history.append(
            {
                "epoch": epoch,
                **{
                    f"training_{key}": float(np.mean(values))
                    for key, values in epoch_values.items()
                },
                "validation": validation,
            }
        )
    late = history[-5:]
    late_auxiliary = float(
        np.mean([row["training_auxiliary"] for row in late])
    )
    late_total = float(np.mean([row["training_loss"] for row in late]))
    return model, {
        "condition": condition,
        "settings": settings,
        "initialization_seed": seed,
        "initial_state_sha256": initial_hash,
        "fixed_epochs": EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "state_batch": STATE_BATCH,
        "pair_batch": PAIR_BATCH,
        "delta_powers": powers,
        "pair_schedule": schedule_report,
        "auxiliary": {
            "coefficient": AUXILIARY_COEFFICIENT,
            "input_features": AUXILIARY_FEATURES,
            "output_features": AUXILIARY_FEATURES,
            "late_fraction": late_auxiliary / max(late_total, 1e-12),
            "maximum_late_fraction": LATE_AUXILIARY_FRACTION_MAXIMUM,
        },
        "history": history,
    }


@torch.no_grad()
def delta_fve(
    model: JacobianSAEPair,
    inputs: torch.Tensor,
    writes: torch.Tensor,
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model.eval()
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[(str(pair["endpoint"]), str(pair["family"]))].append(pair)
    result: dict[str, Any] = {}
    device = model.input_center.device
    for (endpoint, family), rows in sorted(grouped.items()):
        target_input = []
        predicted_input = []
        target_output = []
        predicted_output = []
        for start in range(0, len(rows), PAIR_BATCH):
            batch = rows[start : start + PAIR_BATCH]
            reference = torch.as_tensor(
                [int(pair["reference"]) for pair in batch],
                dtype=torch.long,
            )
            changed = torch.as_tensor(
                [int(pair["changed"]) for pair in batch],
                dtype=torch.long,
            )
            xr = model.normalize_input(inputs[reference].to(device))
            xc = model.normalize_input(inputs[changed].to(device))
            yr = model.normalize_output(writes[reference].to(device))
            yc = model.normalize_output(writes[changed].to(device))
            xr_hat, _, _ = model.input_sae(xr)
            xc_hat, _, _ = model.input_sae(xc)
            yr_hat, _, _ = model.output_sae(yr)
            yc_hat, _, _ = model.output_sae(yc)
            target_input.append((xc - xr).cpu())
            predicted_input.append((xc_hat - xr_hat).cpu())
            target_output.append((yc - yr).cpu())
            predicted_output.append((yc_hat - yr_hat).cpu())

        def energy_fve(
            target_values: list[torch.Tensor],
            predicted_values: list[torch.Tensor],
        ) -> float:
            target = torch.cat(target_values).double()
            predicted = torch.cat(predicted_values).double()
            denominator = torch.sum(target.square())
            if denominator <= DELTA_POWER_EPS:
                return float("-inf")
            return float(
                1.0 - torch.sum((target - predicted).square()) / denominator
            )

        result[f"{endpoint}:{family}"] = {
            "n_pairs": len(rows),
            "input_delta_energy_fve": energy_fve(
                target_input, predicted_input
            ),
            "output_delta_energy_fve": energy_fve(
                target_output, predicted_output
            ),
        }
    return {"by_stratum": result}


@torch.no_grad()
def active_graph_metrics(
    model: JacobianSAEPair,
    inputs: torch.Tensor,
    writes: torch.Tensor,
    indices: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    device = model.input_center.device
    effective = []
    top_mass = []
    positive = []
    for start in range(0, len(indices), 128):
        selected = indices[start : start + 128]
        x = inputs[selected].to(device)
        y = writes[selected].to(device)
        _, input_indices = model.input_sae.encode(
            model.normalize_input(x), return_indices=True
        )
        _, output_indices = model.output_sae.encode(
            model.normalize_output(y), return_indices=True
        )
        jacobian = model.active_jacobian(
            x, input_indices, output_indices
        )
        absolute = jacobian.abs().flatten(1)
        mass = absolute.sum(dim=1, keepdim=True).clamp_min(1e-12)
        probabilities = absolute / mass
        effective.append(
            (1.0 / probabilities.square().sum(dim=1).clamp_min(1e-12)).cpu()
        )
        top_count = math.ceil(0.10 * absolute.shape[1])
        top_mass.append(
            (torch.topk(absolute, top_count, dim=1).values.sum(dim=1) / mass[:, 0])
            .cpu()
        )
        positive.append((jacobian > 0).float().mean(dim=(-2, -1)).cpu())

    def distribution(values: list[torch.Tensor]) -> dict[str, float]:
        array = torch.cat(values).double().numpy()
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.90)),
            "p99": float(np.quantile(array, 0.99)),
        }

    return {
        "effective_active_edge_count": distribution(effective),
        "top_10_percent_edge_mass": distribution(top_mass),
        "positive_edge_fraction": distribution(positive),
        "active_edge_count": ACTIVE_FEATURES**2,
    }


def finite_difference_control(
    model: JacobianSAEPair,
    inputs: torch.Tensor,
    indices: np.ndarray,
    *,
    examples: int = 16,
) -> dict[str, Any]:
    model.eval()
    device = model.input_center.device
    analytic_values = []
    observed_values = []
    with torch.no_grad():
        for global_index in indices[:examples]:
            raw = inputs[int(global_index) : int(global_index) + 1].to(device)
            input_acts, input_indices = model.input_sae.encode(
                model.normalize_input(raw), return_indices=True
            )
            del input_acts
            native_output = model.mlp(raw)
            assert isinstance(native_output, torch.Tensor)
            output_indices = model.output_sae.encode(
                model.normalize_output(native_output),
                return_indices=True,
            )[1]
            analytic = model.active_jacobian(
                raw, input_indices, output_indices
            )[0]
            directions = (
                model.input_sae.decoder()[input_indices[0]]
                * model.input_scale
            )
            perturbed = raw + FINITE_DIFFERENCE_DOSE * directions
            perturbed_output = model.mlp(perturbed)
            assert isinstance(perturbed_output, torch.Tensor)
            baseline_preactive = model.output_sae.preactivations(
                model.normalize_output(native_output)
            )[0, output_indices[0]]
            perturbed_preactive = model.output_sae.preactivations(
                model.normalize_output(perturbed_output)
            )[:, output_indices[0]]
            observed = (
                (perturbed_preactive - baseline_preactive)
                / FINITE_DIFFERENCE_DOSE
            ).T
            analytic_values.append(analytic.cpu().double())
            observed_values.append(observed.cpu().double())
    analytic_all = torch.cat([value.flatten() for value in analytic_values])
    observed_all = torch.cat([value.flatten() for value in observed_values])
    difference = observed_all - analytic_all
    centered_a = analytic_all - analytic_all.mean()
    centered_o = observed_all - observed_all.mean()
    correlation = float(
        torch.dot(centered_a, centered_o)
        / (
            torch.linalg.vector_norm(centered_a)
            * torch.linalg.vector_norm(centered_o)
        ).clamp_min(1e-12)
    )
    relative = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(observed_all).clamp_min(1e-12)
    )
    eligible = analytic_all.abs() >= 1e-4
    sign = float(
        torch.mean(
            (torch.sign(analytic_all[eligible]) == torch.sign(observed_all[eligible]))
            .double()
        )
    )
    gates = {
        "correlation": correlation >= FINITE_DIFFERENCE_CORRELATION_MINIMUM,
        "sign_agreement": sign >= FINITE_DIFFERENCE_SIGN_MINIMUM,
        "relative_error": relative <= FINITE_DIFFERENCE_RELATIVE_ERROR_MAXIMUM,
    }
    return {
        "dose": FINITE_DIFFERENCE_DOSE,
        "examples": len(analytic_values),
        "correlation": correlation,
        "sign_agreement": sign,
        "relative_frobenius_error": relative,
        "gates": gates,
        "PASS": all(gates.values()),
    }


def delta_threshold(family: str, site: str) -> float:
    primary = family in {"target_shuffle", "sham_shuffle"}
    if site == "input":
        return 0.90 if primary else 0.85
    if site == "output":
        return 0.95 if primary else 0.90
    raise ValueError(site)


def representation_gate(
    primary: Mapping[str, Any],
    delta_no_jacobian: Mapping[str, Any],
    no_delta_jacobian: Mapping[str, Any] | None,
) -> dict[str, Any]:
    train = primary["train"]
    validation = primary["validation"]
    matched = delta_no_jacobian["validation"]
    e90 = validation["active_edge_mass_count"]["E90"]["median"]
    matched_e90 = matched["active_edge_mass_count"]["E90"]["median"]
    e90_reduction = (
        (matched_e90 - e90) / matched_e90 if matched_e90 > 0 else 0.0
    )
    effective = primary["graph_metrics"]["effective_active_edge_count"]["median"]
    matched_effective = delta_no_jacobian["graph_metrics"][
        "effective_active_edge_count"
    ]["median"]
    effective_reduction = (
        (matched_effective - effective) / matched_effective
        if matched_effective > 0
        else 0.0
    )
    gates = {
        "autograd": primary["autograd_control"]["PASS"],
        "finite_difference": primary["finite_difference_control"]["PASS"],
        "decoder_norm": primary["decoder_invariants"]["PASS"],
        "input_dead_fraction": train["input_dead_feature_fraction"] <= 0.50,
        "output_dead_fraction": train["output_dead_feature_fraction"] <= 0.50,
        "input_fve": validation["input_fve"] >= 0.95,
        "output_fve": validation["output_write_fve"] >= 0.95,
        "sandwich_fve": validation["sandwich_output_fve"] >= 0.90,
        "input_fve_matched": (
            validation["input_fve"] >= matched["input_fve"] - 0.03
        ),
        "output_fve_matched": (
            validation["output_write_fve"]
            >= matched["output_write_fve"] - 0.03
        ),
        "sandwich_fve_matched": (
            validation["sandwich_output_fve"]
            >= matched["sandwich_output_fve"] - 0.03
        ),
        "E90_reduction": e90_reduction >= E90_REDUCTION_MINIMUM,
        "effective_edge_reduction": (
            effective_reduction >= EFFECTIVE_EDGE_REDUCTION_MINIMUM
        ),
        "late_auxiliary_fraction": (
            primary["training"]["auxiliary"]["late_fraction"]
            <= LATE_AUXILIARY_FRACTION_MAXIMUM
        ),
    }
    for key, values in primary["delta_validation"]["by_stratum"].items():
        _, family = key.split(":", 1)
        gates[f"{key}:input_delta_fve"] = (
            values["input_delta_energy_fve"]
            >= delta_threshold(family, "input")
        )
        gates[f"{key}:output_delta_fve"] = (
            values["output_delta_energy_fve"]
            >= delta_threshold(family, "output")
        )
        if (
            family in {"target_shuffle", "sham_shuffle"}
            and no_delta_jacobian is not None
        ):
            comparator = no_delta_jacobian["delta_validation"][
                "by_stratum"
            ][key]["output_delta_energy_fve"]
            gates[f"{key}:delta_objective_matched"] = (
                values["output_delta_energy_fve"] >= comparator - 0.02
            )
    return {
        "gates": gates,
        "median_E90_reduction_fraction": e90_reduction,
        "median_effective_edge_reduction_fraction": effective_reduction,
        "PASS": all(gates.values()),
    }


def condition_record(
    model: JacobianSAEPair,
    training: dict[str, Any],
    inputs: torch.Tensor,
    writes: torch.Tensor,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    validation_pairs: Sequence[Mapping[str, Any]],
    *,
    primary: bool,
) -> dict[str, Any]:
    record = {
        "training": training,
        "train": INSTRUMENT.evaluate_model(
            model, inputs, writes, train_indices, batch_size=128
        ),
        "validation": INSTRUMENT.evaluate_model(
            model, inputs, writes, validation_indices, batch_size=128
        ),
        "delta_validation": delta_fve(
            model, inputs, writes, validation_pairs
        ),
        "graph_metrics": active_graph_metrics(
            model, inputs, writes, validation_indices
        ),
    }
    if primary:
        record["autograd_control"] = INSTRUMENT.autograd_control(
            model, inputs, validation_indices
        )
        record["finite_difference_control"] = finite_difference_control(
            model, inputs, validation_indices
        )
        record["decoder_invariants"] = INSTRUMENT.decoder_invariants(model)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-artifact", required=True)
    parser.add_argument("--model-dir", action="append", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    activation_path = Path(args.activation_artifact).expanduser().resolve()
    activation_report = json.loads(activation_path.read_text())
    assert_label_free_payload(activation_report)
    if (
        activation_report.get("status") != "COMPLETE"
        or activation_report.get("stage") != "label-free activation capture"
        or activation_report.get("label_free") is not True
    ):
        raise RuntimeError("large J-SAE requires the frozen label-free capture")
    model_dirs = parse_model_dirs(args.model_dir)
    frozen_checkpoint_hashes = validate_frozen_sources(
        activation_path,
        activation_report,
        model_dirs,
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
    addendum = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "IMPLEMENTATION_ADDENDUM_JACOBIAN_SAE_LARGE_V1.md"
    )
    genes = [str(row["gene_id"]) for row in activation_report["states"]]
    train_mask, validation_mask = INSTRUMENT.gene_train_validation_split(genes)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    train_pairs = build_reference_pairs(
        activation_report["states"], train_mask
    )
    validation_pairs = build_reference_pairs(
        activation_report["states"], validation_mask
    )
    observed_counts = {
        "states": len(genes),
        "training_states": len(train_indices),
        "validation_states": len(validation_indices),
        "training_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "training_genes": len(set(np.asarray(genes)[train_mask])),
        "validation_genes": len(set(np.asarray(genes)[validation_mask])),
    }
    if observed_counts != FROZEN_COUNTS:
        raise RuntimeError(
            f"frozen state/pair/gene counts changed: {observed_counts}"
        )
    result_views: dict[str, Any] = {}
    fitted_primary: dict[str, list[dict[str, Any]]] = {}
    model_payload: dict[str, Any] = {
        "schema": "gfmcd.splicebert.jacobian_sae.large.v1",
        "architecture": {
            "input_features": INPUT_FEATURES,
            "output_features": OUTPUT_FEATURES,
            "active_features": ACTIVE_FEATURES,
        },
        "checkpoint_models": {},
    }
    pilot_pass = False

    for view in VIEWS:
        _, input_array, write_array = INSTRUMENT.load_activation_artifact(
            activation_path, view=view
        )
        inputs = torch.from_numpy(input_array)
        writes = torch.from_numpy(write_array)
        weights, checkpoint_hash = INSTRUMENT.extract_mlp_weights(
            model_dirs[view]
        )
        if checkpoint_hash != frozen_checkpoint_hashes[view]:
            raise RuntimeError(f"view {view} changed during MLP extraction")
        replay_model = build_large_pair(
            weights,
            inputs[train_indices],
            writes[train_indices],
            device=device,
        )
        maximum_replay_error = 0.0
        with torch.no_grad():
            for start in range(0, len(inputs), 128):
                x = inputs[start : start + 128].to(device)
                y = writes[start : start + 128].to(device)
                replay = replay_model.mlp(x)
                assert isinstance(replay, torch.Tensor)
                maximum_replay_error = max(
                    maximum_replay_error,
                    float((replay - y).abs().max().cpu()),
                )
        del replay_model
        if maximum_replay_error > INSTRUMENT.MLP_REPLAY_TOLERANCE:
            raise RuntimeError(f"view {view} MLP replay failed")
        view_conditions = (
            BASE_CONDITIONS if view == "base" else tuple(CONDITIONS)
        )
        result_views[view] = {
            "checkpoint_hash": checkpoint_hash,
            "maximum_mlp_replay_error": maximum_replay_error,
            "replicates": [],
        }
        fitted_primary[view] = []
        model_payload["checkpoint_models"][view] = {
            "model_dir": str(model_dirs[view]),
            "checkpoint_hash": checkpoint_hash,
            "replicates": {},
        }
        for initialization in INITIALIZATIONS:
            seed = initialization_seed(view, initialization)
            conditions: dict[str, Any] = {}
            initial_hashes = set()
            primary_state: dict[str, torch.Tensor] | None = None
            primary_features: dict[str, Any] | None = None
            for condition in view_conditions:
                model, training = train_condition(
                    weights,
                    inputs,
                    writes,
                    genes,
                    train_indices,
                    validation_indices,
                    train_pairs,
                    validation_pairs,
                    condition=condition,
                    device=device,
                    seed=seed,
                )
                record = condition_record(
                    model,
                    training,
                    inputs,
                    writes,
                    train_indices,
                    validation_indices,
                    validation_pairs,
                    primary=condition == "delta_jacobian",
                )
                conditions[condition] = record
                initial_hashes.add(training["initial_state_sha256"])
                if condition == "delta_jacobian":
                    input_features, output_features, graph = (
                        REPLICATION.fingerprints_and_graph(
                            model,
                            inputs,
                            writes,
                            validation_indices,
                            batch_size=128,
                        )
                    )
                    primary_features = {
                        "input_features": input_features,
                        "output_features": output_features,
                        "input_decoder": (
                            model.input_sae.decoder()
                            .detach()
                            .cpu()
                            .numpy()
                            .T
                        ),
                        "output_decoder": (
                            model.output_sae.decoder()
                            .detach()
                            .cpu()
                            .numpy()
                            .T
                        ),
                        "graph": graph,
                    }
                    primary_state = compact_state(model)
                print(
                    f"view={view} init={initialization} condition={condition} "
                    f"input_fve={record['validation']['input_fve']:.4f} "
                    f"output_fve={record['validation']['output_write_fve']:.4f}",
                    flush=True,
                )
                del model
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
            if len(initial_hashes) != 1:
                raise RuntimeError("factorial conditions did not share initialization")
            if primary_state is None or primary_features is None:
                raise RuntimeError("primary J-SAE fit was not preserved")
            gate = representation_gate(
                conditions["delta_jacobian"],
                conditions["delta_no_jacobian"],
                conditions.get("no_delta_jacobian"),
            )
            fitted_primary[view].append(primary_features)
            result_views[view]["replicates"].append(
                {
                    "initialization": initialization,
                    "initialization_seed": seed,
                    "identical_initial_state": len(initial_hashes) == 1,
                    "conditions": conditions,
                    "representation_gate": gate,
                }
            )
            model_payload["checkpoint_models"][view]["replicates"][
                str(initialization)
            ] = {
                "primary_state": primary_state,
                "mean_absolute_validation_graph": torch.as_tensor(
                    primary_features["graph"]
                ),
            }
            if view == "2020" and initialization == 0:
                pilot_pass = gate["PASS"]
                if not pilot_pass:
                    break
        if view == "2020" and not pilot_pass:
            break

    complete_family = (
        pilot_pass
        and set(result_views) == set(VIEWS)
        and all(
            len(result_views[view]["replicates"]) == len(INITIALIZATIONS)
            for view in VIEWS
        )
    )
    stability_rows = []
    if complete_family:
        for view in VIEWS:
            for left, right in combinations(INITIALIZATIONS, 2):
                site_matches = {}
                for site in ("input", "output"):
                    site_matches[site] = REPLICATION.feature_match(
                        fitted_primary[view][left][f"{site}_features"],
                        fitted_primary[view][right][f"{site}_features"],
                        fitted_primary[view][left][f"{site}_decoder"],
                        fitted_primary[view][right][f"{site}_decoder"],
                    )
                stability_rows.append(
                    {
                        "view": view,
                        "initialization_pair": [left, right],
                        "sites": site_matches,
                        "graph_diagnostic": REPLICATION.graph_alignment(
                            fitted_primary[view][left]["graph"],
                            fitted_primary[view][right]["graph"],
                            site_matches["input"]["forward_map"],
                            site_matches["output"]["forward_map"],
                        ),
                        "PASS": all(
                            site_matches[site]["PASS"]
                            for site in ("input", "output")
                        ),
                    }
                )
    fit_pass = complete_family and all(
        replicate["representation_gate"]["PASS"]
        for view in result_views.values()
        for replicate in view["replicates"]
    )
    stability_pass = complete_family and all(
        row["PASS"] for row in stability_rows
    )
    overall_pass = fit_pass and stability_pass
    status = "INSTRUMENT_PASS" if overall_pass else "INSTRUMENT_UNRESOLVED"
    result = {
        "study": "SpliceBERT GTEx layer-5 larger paired-delta Jacobian SAE",
        "stage": "base plus five-tuned-checkpoint representation factorial",
        "status": status,
        "evidence_tier": "instrument",
        "implementation_addendum": str(addendum.relative_to(REPO)),
        "activation_artifact": str(activation_path),
        "activation_artifact_sha256": sha256_file(activation_path),
        "architecture": {
            "input_features": INPUT_FEATURES,
            "output_features": OUTPUT_FEATURES,
            "active_features": ACTIVE_FEATURES,
            "native_width": 512,
            "mlp_width": 2048,
        },
        "objective": {
            "jacobian_coefficient": JACOBIAN_COEFFICIENT,
            "delta_coefficient": DELTA_COEFFICIENT,
            "conditions": CONDITIONS,
            "fixed_epochs": EPOCHS,
            "state_batch": STATE_BATCH,
            "pair_batch": PAIR_BATCH,
            "auxiliary_features": AUXILIARY_FEATURES,
            "auxiliary_coefficient": AUXILIARY_COEFFICIENT,
        },
        "data": {
            **observed_counts,
            "ordinary_draws_per_epoch": (
                math.ceil(len(train_indices) / STATE_BATCH) * STATE_BATCH
            ),
        },
        "views": result_views,
        "pilot": {
            "view": "2020",
            "initialization": 0,
            "PASS": pilot_pass,
            "stopped_family": not pilot_pass,
        },
        "within_checkpoint_stability": {
            "rows": stability_rows,
            "thresholds": {
                "density_floor": REPLICATION.DENSITY_FLOOR,
                "reciprocal_match_fraction": (
                    REPLICATION.RECIPROCAL_FRACTION_FLOOR
                ),
                "reciprocal_score_mean": REPLICATION.RECIPROCAL_SCORE_FLOOR,
            },
            "PASS": stability_pass,
        },
        "instrument_gate": {
            "complete_family": complete_family,
            "all_primary_representation_fits_PASS": fit_pass,
            "all_site_stability_comparisons_PASS": stability_pass,
            "classifier_parameter_or_margin_opened": False,
            "PASS": overall_pass,
        },
        "fresh_v2_confirmation_authorized": overall_pass,
        "feature_identity_opened": False,
        "feature_semantics_established": False,
        "licensed_claim": (
            "The larger paired-delta J-SAE representation instrument passes "
            "across base and all five tuned checkpoints and authorizes a new "
            "gene-fresh behavioral confirmation."
            if overall_pass
            else "The larger paired-delta J-SAE remains an unresolved "
            "instrument; classifier behavior and feature identity stay closed."
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
        "splicebert_gtex_finetune_shift:jacobian_sae_large_v1",
        sys.argv,
        repo=REPO,
        status=status,
        config={
            "views": list(VIEWS),
            "initializations": list(INITIALIZATIONS),
            "architecture": result["architecture"],
            "objective": result["objective"],
            "device": str(device),
        },
        seeds={
            "dictionary_initializations": [
                initialization_seed(view, initialization)
                for view in VIEWS
                for initialization in INITIALIZATIONS
            ]
        },
        notes=[
            "All fits were run sequentially with fixed 40-epoch budgets.",
            "Factorial conditions share byte-identical initialization and schedules.",
            "No classifier parameter, task margin, or feature example was opened.",
            "The base checkpoint is a representation comparator only.",
        ],
    )
    arrays_path = Path(activation_report["arrays"]["path"]).resolve()
    model_files = []
    for model_dir in model_dirs.values():
        weight_path = checkpoint_weight_path(model_dir)
        model_files.extend((model_dir / "config.json", weight_path))
    for path in (
        addendum,
        activation_path,
        arrays_path,
        *model_files,
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.extend([file_record(output), file_record(model_output)])
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": status,
                "instrument_gate": result["instrument_gate"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
