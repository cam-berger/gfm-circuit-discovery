#!/usr/bin/env python
"""Apply frozen native heads to replicated J-SAE output reconstructions."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import (  # noqa: E402
    ExperimentManifest,
    file_record,
    sha256_file,
)
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.mlp_units import composed_classifier_margin  # noqa: E402


def load_script(name: str) -> Any:
    path = REPO / "scripts/splicebert_gtex_finetune_shift" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    spec.loader.exec_module(module)
    return module


INSTRUMENT = load_script("run_gtex_jacobian_sae_instrument.py")
REPLICATION = load_script("run_gtex_jacobian_sae_replication.py")
EXPECTED_SEEDS = tuple(range(2020, 2025))
INITIALIZATIONS = (0, 1, 2)
ENDPOINT_LABELS = {"donor": 1, "acceptor": 2}
MARGIN_MAE_THRESHOLD = 0.05
NATIVE_EFFECT_THRESHOLD = 0.05
RECOVERY_RATIO_THRESHOLD = 0.90
SHAM_ERROR_ALLOWANCE = 0.02


def parse_checkpoints(values: list[str]) -> dict[int, Path]:
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


def counterfactual_pairs(
    states: list[Mapping[str, Any]],
    *,
    variant_kind: str,
) -> dict[str, np.ndarray]:
    index = {str(row["state_id"]): position for position, row in enumerate(states)}
    pairs: dict[str, list[tuple[int, int]]] = {
        endpoint: [] for endpoint in ENDPOINT_LABELS
    }
    for changed, row in enumerate(states):
        endpoint = str(row["endpoint"])
        if endpoint not in pairs or str(row["variant_kind"]) != variant_kind:
            continue
        reference_id = row.get("reference_state_id")
        if reference_id is None or str(reference_id) not in index:
            raise ValueError(f"{variant_kind} state lacks a valid clean reference")
        reference = index[str(reference_id)]
        reference_row = states[reference]
        if (
            str(reference_row["variant_kind"]) != "clean"
            or str(reference_row["endpoint"]) != endpoint
            or str(reference_row["gene_id"]) != str(row["gene_id"])
        ):
            raise ValueError(f"{variant_kind}/clean identity mismatch")
        pairs[endpoint].append((reference, changed))
    result = {
        endpoint: np.asarray(values, dtype=np.int64).reshape(-1, 2)
        for endpoint, values in pairs.items()
    }
    if any(len(values) < 2 for values in result.values()):
        raise RuntimeError(f"{variant_kind} requires both endpoint pair families")
    return result


@torch.no_grad()
def reconstruct_writes(
    model: Any,
    writes: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    device = model.input_center.device
    reconstructed = []
    for start in range(0, len(writes), batch_size):
        native = writes[start : start + batch_size].to(device)
        normalized = model.normalize_output(native)
        decoded, _, _ = model.output_sae(normalized)
        reconstructed.append(model.denormalize_output(decoded).cpu().numpy())
    return np.concatenate(reconstructed)


def endpoint_metrics(
    residual: np.ndarray,
    native_write: np.ndarray,
    reconstructed_write: np.ndarray,
    target_pairs: np.ndarray,
    sham_pairs: np.ndarray,
    parameters: Mapping[str, Any],
) -> dict[str, float | bool]:
    native_margin = composed_classifier_margin(
        residual,
        native_write,
        **parameters,
    )
    reconstructed_margin = composed_classifier_margin(
        residual,
        reconstructed_write,
        **parameters,
    )
    target_rows = np.unique(target_pairs.reshape(-1))
    sham_rows = np.unique(sham_pairs.reshape(-1))
    union_rows = np.unique(np.concatenate([target_rows, sham_rows]))
    absolute_error = np.abs(reconstructed_margin - native_margin)
    native_effect = float(
        np.mean(
            native_margin[target_pairs[:, 0]]
            - native_margin[target_pairs[:, 1]]
        )
    )
    reconstructed_effect = float(
        np.mean(
            reconstructed_margin[target_pairs[:, 0]]
            - reconstructed_margin[target_pairs[:, 1]]
        )
    )
    ratio = (
        reconstructed_effect / native_effect
        if abs(native_effect) > np.finfo(np.float64).eps
        else float("nan")
    )
    margin_mae = float(np.mean(absolute_error[union_rows]))
    target_mae = float(np.mean(absolute_error[target_rows]))
    sham_mae = float(np.mean(absolute_error[sham_rows]))
    gates = {
        "margin_mae": margin_mae <= MARGIN_MAE_THRESHOLD,
        "native_target_effect": native_effect >= NATIVE_EFFECT_THRESHOLD,
        "reconstructed_target_effect": reconstructed_effect > 0,
        "effect_recovery_ratio": ratio >= RECOVERY_RATIO_THRESHOLD,
        "sham_error": sham_mae <= target_mae + SHAM_ERROR_ALLOWANCE,
    }
    return {
        "margin_mae": margin_mae,
        "target_margin_mae": target_mae,
        "sham_margin_mae": sham_mae,
        "native_target_shuffle_effect": native_effect,
        "reconstructed_target_shuffle_effect": reconstructed_effect,
        "effect_recovery_ratio": ratio,
        "recovery_ratio_above_1_10": ratio > 1.10,
        "gates": gates,
        "PASS": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replication-artifact", required=True)
    parser.add_argument("--activation-artifact", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_JACOBIAN_SAE_V1.md"
    )
    addendum = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "IMPLEMENTATION_ADDENDUM_JACOBIAN_SAE_BEHAVIOR_V1.md"
    )
    replication_path = Path(args.replication_artifact).expanduser().resolve()
    replication = json.loads(replication_path.read_text())
    if (
        replication.get("status") != "INSTRUMENT_PASS"
        or replication.get("behavioral_preflight_authorized") is not True
        or not replication.get("instrument_gate", {}).get("PASS")
        or replication.get("feature_association_opened") is not False
    ):
        raise RuntimeError("replication artifact does not authorize behavior preflight")
    activation_path = Path(args.activation_artifact).expanduser().resolve()
    if (
        str(activation_path)
        != str(Path(replication["activation_artifact"]).resolve())
        or sha256_file(activation_path)
        != replication["activation_artifact_sha256"]
    ):
        raise RuntimeError("activation artifact differs from replication")
    activation = json.loads(activation_path.read_text())
    arrays_path = Path(activation["arrays"]["path"]).expanduser().resolve()
    if sha256_file(arrays_path) != activation["arrays"]["sha256"]:
        raise RuntimeError("activation arrays changed")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    genes = [str(row["gene_id"]) for row in activation["states"]]
    _, validation_mask = INSTRUMENT.gene_train_validation_split(genes)
    validation_indices = np.flatnonzero(validation_mask)
    states = [activation["states"][int(index)] for index in validation_indices]
    target_pairs = counterfactual_pairs(states, variant_kind="target_shuffle")
    sham_pairs = counterfactual_pairs(states, variant_kind="sham_shuffle")
    arrays = {
        key: value[validation_indices]
        for key, value in arrays.items()
        if key.split("__", 1)[0] in {str(seed) for seed in EXPECTED_SEEDS}
    }

    model_path = Path(replication["model_artifact"]["path"]).expanduser().resolve()
    if sha256_file(model_path) != replication["model_artifact"]["sha256"]:
        raise RuntimeError("replicated J-SAE model artifact changed")
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "gfmcd.splicebert.jacobian_sae.replication.v1":
        raise RuntimeError("replicated J-SAE model schema mismatch")
    checkpoints = parse_checkpoints(args.checkpoint)
    base_model = Path(args.base_model).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    if output == manifest_path or output.suffix != ".json":
        raise ValueError("behavior preflight needs a distinct .json output")
    device = torch.device(args.device)

    # Reconstruct every write before the first classifier is loaded.
    frozen_reconstructions: dict[str, dict[str, np.ndarray]] = {}
    for seed in EXPECTED_SEEDS:
        view = str(seed)
        inputs = torch.from_numpy(arrays[f"{view}__input"])
        writes = torch.from_numpy(arrays[f"{view}__write"])
        weights, checkpoint_hash = INSTRUMENT.extract_mlp_weights(
            Path(payload["checkpoint_models"][view]["model_dir"])
        )
        if checkpoint_hash != payload["checkpoint_models"][view]["checkpoint_hash"]:
            raise RuntimeError(f"J-SAE source checkpoint {view} changed")
        frozen_reconstructions[view] = {}
        for initialization in INITIALIZATIONS:
            model = INSTRUMENT.build_pair(
                weights,
                inputs,
                writes,
                device=device,
                capacity_repair=True,
            )
            state = payload["checkpoint_models"][view]["replicates"][
                str(initialization)
            ]["candidate_state"]
            load = model.load_state_dict(state, strict=False)
            expected_missing = {
                "mlp_input_weight",
                "mlp_input_bias",
                "mlp_output_weight",
                "mlp_output_bias",
            }
            if set(load.missing_keys) != expected_missing or load.unexpected_keys:
                raise RuntimeError("compact J-SAE state contract mismatch")
            frozen_reconstructions[view][str(initialization)] = (
                reconstruct_writes(model, writes, batch_size=args.batch)
            )
            del model
        if device.type == "mps":
            torch.mps.empty_cache()
    reconstructions_frozen_before_classifier_load = True

    classifier_parameters: dict[str, dict[str, Any]] = {}
    checkpoint_records: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        view = str(seed)
        model, _, report = load_gtex_token_classifier(
            base_model,
            checkpoints[seed],
            device=str(device),
            attn_implementation="eager",
        )
        expected_hash = activation["view_records"][view]["checkpoint_sha256"]
        if report.checkpoint_sha256 != expected_hash:
            raise RuntimeError(f"classifier checkpoint {view} changed")
        layer = model.bert.encoder.layer[5]
        classifier_parameters[view] = {}
        for endpoint, label in ENDPOINT_LABELS.items():
            classifier_parameters[view][endpoint] = {
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
                    (
                        model.classifier.bias[label]
                        - model.classifier.bias[0]
                    )
                    .detach()
                    .cpu()
                ),
            }
        checkpoint_records[view] = {
            "path": str(checkpoints[seed]),
            "sha256": report.checkpoint_sha256,
        }
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    cells = []
    for seed in EXPECTED_SEEDS:
        view = str(seed)
        residual = arrays[f"{view}__input"].astype(np.float64)
        native_write = arrays[f"{view}__write"].astype(np.float64)
        for initialization in INITIALIZATIONS:
            reconstructed = frozen_reconstructions[view][
                str(initialization)
            ].astype(np.float64)
            for endpoint in ENDPOINT_LABELS:
                metrics = endpoint_metrics(
                    residual,
                    native_write,
                    reconstructed,
                    target_pairs[endpoint],
                    sham_pairs[endpoint],
                    classifier_parameters[view][endpoint],
                )
                cells.append(
                    {
                        "checkpoint_seed": seed,
                        "initialization": initialization,
                        "endpoint": endpoint,
                        **metrics,
                    }
                )
    overall_pass = all(cell["PASS"] for cell in cells)
    status = "INSTRUMENT_PASS" if overall_pass else "INSTRUMENT_UNRESOLVED"
    result = {
        "study": "SpliceBERT GTEx layer-5 Jacobian SAE",
        "stage": "frozen native-head behavioral fidelity preflight",
        "status": status,
        "evidence_tier": "instrument",
        "protocol": str(protocol.relative_to(REPO)),
        "implementation_addendum": str(addendum.relative_to(REPO)),
        "replication_artifact": str(replication_path),
        "replication_artifact_sha256": sha256_file(replication_path),
        "activation_artifact": str(activation_path),
        "activation_artifact_sha256": sha256_file(activation_path),
        "checkpoint_records": checkpoint_records,
        "held_out_validation": {
            "n_states": len(states),
            "n_genes": len(set(np.asarray(genes)[validation_mask])),
            "target_pairs_by_endpoint": {
                endpoint: len(values)
                for endpoint, values in target_pairs.items()
            },
            "sham_pairs_by_endpoint": {
                endpoint: len(values)
                for endpoint, values in sham_pairs.items()
            },
        },
        "thresholds": {
            "margin_mae_maximum": MARGIN_MAE_THRESHOLD,
            "native_target_effect_minimum": NATIVE_EFFECT_THRESHOLD,
            "effect_recovery_ratio_minimum": RECOVERY_RATIO_THRESHOLD,
            "sham_minus_target_error_maximum": SHAM_ERROR_ALLOWANCE,
            "required_cells": 30,
        },
        "cells": cells,
        "instrument_gate": {
            "dictionary_weights_frozen_before_classifier_load": (
                reconstructions_frozen_before_classifier_load
            ),
            "all_30_cells_PASS": overall_pass,
            "PASS": overall_pass,
        },
        "feature_edge_discovery_authorized": overall_pass,
        "feature_semantics_established": False,
        "licensed_claim": (
            "Replicated J-SAE output reconstruction preserves the frozen "
            "native-head target-shuffle behavior."
            if overall_pass
            else "J-SAE feature/edge discovery and steering remain closed."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:jacobian_sae_behavior_preflight_v1",
        sys.argv,
        repo=REPO,
        status=status,
        config={
            "batch": args.batch,
            "device": str(device),
            "margin_mae_threshold": MARGIN_MAE_THRESHOLD,
            "native_effect_threshold": NATIVE_EFFECT_THRESHOLD,
            "recovery_ratio_threshold": RECOVERY_RATIO_THRESHOLD,
            "sham_error_allowance": SHAM_ERROR_ALLOWANCE,
        },
        seeds={
            "checkpoint_family": list(EXPECTED_SEEDS),
            "dictionary_initializations": list(INITIALIZATIONS),
        },
        notes=[
            "All reconstructed writes were computed before classifier loading.",
            "No optimizer, feature selection, or threshold adaptation was run.",
            "Passing this preflight does not make an individual feature causal.",
        ],
    )
    for path in (
        protocol,
        addendum,
        replication_path,
        model_path,
        activation_path,
        arrays_path,
        *checkpoints.values(),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": status,
                "instrument_gate": result["instrument_gate"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
