#!/usr/bin/env python
"""Preflight residual-preserving J-SAE feature-delta interventions."""

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
    path = REPO / "scripts/splicebert" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    spec.loader.exec_module(module)
    return module


INSTRUMENT = load_script("run_gtex_jacobian_sae_instrument.py")
BEHAVIOR = load_script("run_gtex_jacobian_sae_behavior_preflight.py")
EXPECTED_SEEDS = tuple(range(2020, 2025))
INITIALIZATIONS = (0, 1, 2)
ENDPOINT_LABELS = {"donor": 1, "acceptor": 2}
DENSE_EFFECT_THRESHOLD = 0.05
RELATIVE_ERROR_THRESHOLD = 0.10
SHAM_ERROR_ALLOWANCE = 0.02
DECOMPOSITION_ATOL = 2e-5
IDENTITY_ATOL = 1e-12


@torch.no_grad()
def reconstruct_writes_and_codes(
    model: Any,
    writes: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return decoded writes, output codes, and the raw-unit decoder matrix."""

    model.eval()
    device = model.input_center.device
    reconstructed: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    for start in range(0, len(writes), batch_size):
        native = writes[start : start + batch_size].to(device)
        normalized = model.normalize_output(native)
        decoded, activations, _ = model.output_sae(normalized)
        reconstructed.append(model.denormalize_output(decoded).cpu().numpy())
        codes.append(activations.cpu().numpy())
    decoder = (
        model.output_sae.decoder() * model.output_scale
    ).detach().cpu().numpy()
    return np.concatenate(reconstructed), np.concatenate(codes), decoder


def decoder_difference_error(
    reconstructed: np.ndarray,
    codes: np.ndarray,
    decoder: np.ndarray,
    pairs: np.ndarray,
) -> float:
    """Verify that decoded paired differences equal the all-feature sum."""

    clean = pairs[:, 0]
    changed = pairs[:, 1]
    decoded_delta = reconstructed[changed] - reconstructed[clean]
    feature_delta = (codes[changed] - codes[clean]) @ decoder
    return float(np.max(np.abs(decoded_delta - feature_delta)))


def paired_intervention_effects(
    residual: np.ndarray,
    native_write: np.ndarray,
    reconstructed_write: np.ndarray,
    pairs: np.ndarray,
    parameters: Mapping[str, Any],
) -> dict[str, float]:
    """Compare dense native-write swaps with residual-preserving SAE deltas."""

    clean = pairs[:, 0]
    changed = pairs[:, 1]
    clean_margin = composed_classifier_margin(
        residual[clean],
        native_write[clean],
        **parameters,
    )
    changed_margin = composed_classifier_margin(
        residual[changed],
        native_write[changed],
        **parameters,
    )
    dense_noised = composed_classifier_margin(
        residual[clean],
        native_write[changed],
        **parameters,
    )
    sparse_noised = composed_classifier_margin(
        residual[clean],
        native_write[clean]
        + reconstructed_write[changed]
        - reconstructed_write[clean],
        **parameters,
    )
    dense_recovered = composed_classifier_margin(
        residual[changed],
        native_write[clean],
        **parameters,
    )
    sparse_recovered = composed_classifier_margin(
        residual[changed],
        native_write[changed]
        + reconstructed_write[clean]
        - reconstructed_write[changed],
        **parameters,
    )
    dense_damage = float(np.mean(clean_margin - dense_noised))
    sparse_damage = float(np.mean(clean_margin - sparse_noised))
    dense_recovery = float(np.mean(dense_recovered - changed_margin))
    sparse_recovery = float(np.mean(sparse_recovered - changed_margin))
    damage_absolute_error = abs(sparse_damage - dense_damage)
    recovery_absolute_error = abs(sparse_recovery - dense_recovery)
    tiny = np.finfo(np.float64).eps
    damage_relative_error = (
        damage_absolute_error / abs(dense_damage)
        if abs(dense_damage) > tiny
        else float("inf")
    )
    recovery_relative_error = (
        recovery_absolute_error / abs(dense_recovery)
        if abs(dense_recovery) > tiny
        else float("inf")
    )
    return {
        "sequence_effect": float(np.mean(clean_margin - changed_margin)),
        "dense_damage": dense_damage,
        "sparse_damage": sparse_damage,
        "damage_absolute_error": damage_absolute_error,
        "damage_relative_error": damage_relative_error,
        "damage_ratio": (
            sparse_damage / dense_damage
            if abs(dense_damage) > tiny
            else float("nan")
        ),
        "dense_recovery": dense_recovery,
        "sparse_recovery": sparse_recovery,
        "recovery_absolute_error": recovery_absolute_error,
        "recovery_relative_error": recovery_relative_error,
        "recovery_ratio": (
            sparse_recovery / dense_recovery
            if abs(dense_recovery) > tiny
            else float("nan")
        ),
    }


def endpoint_metrics(
    residual: np.ndarray,
    native_write: np.ndarray,
    reconstructed_write: np.ndarray,
    target_pairs: np.ndarray,
    sham_pairs: np.ndarray,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    target = paired_intervention_effects(
        residual,
        native_write,
        reconstructed_write,
        target_pairs,
        parameters,
    )
    sham = paired_intervention_effects(
        residual,
        native_write,
        reconstructed_write,
        sham_pairs,
        parameters,
    )
    gates = {
        "dense_damage": target["dense_damage"] >= DENSE_EFFECT_THRESHOLD,
        "dense_recovery": target["dense_recovery"] >= DENSE_EFFECT_THRESHOLD,
        "sparse_damage_sign": target["sparse_damage"] > 0,
        "sparse_recovery_sign": target["sparse_recovery"] > 0,
        "damage_relative_error": (
            target["damage_relative_error"] <= RELATIVE_ERROR_THRESHOLD
        ),
        "recovery_relative_error": (
            target["recovery_relative_error"] <= RELATIVE_ERROR_THRESHOLD
        ),
        "sham_damage_error": (
            sham["damage_absolute_error"]
            <= target["damage_absolute_error"] + SHAM_ERROR_ALLOWANCE
        ),
        "sham_recovery_error": (
            sham["recovery_absolute_error"]
            <= target["recovery_absolute_error"] + SHAM_ERROR_ALLOWANCE
        ),
    }
    return {
        "target_shuffle": target,
        "sham_shuffle": sham,
        "gates": gates,
        "PASS": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-artifact", required=True)
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
    biological_protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_BIOLOGICAL_FEATURES_V1.md"
    )
    addendum = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "IMPLEMENTATION_ADDENDUM_JACOBIAN_SAE_DELTA_BEHAVIOR_V2.md"
    )
    behavior_path = Path(args.behavior_artifact).expanduser().resolve()
    behavior = json.loads(behavior_path.read_text())
    if (
        behavior.get("status") != "INSTRUMENT_UNRESOLVED"
        or behavior.get("instrument_gate", {}).get("PASS") is not False
        or behavior.get("feature_edge_discovery_authorized") is not False
    ):
        raise RuntimeError("v1 artifact is not the preserved reconstruction failure")
    replication_path = Path(args.replication_artifact).expanduser().resolve()
    replication = json.loads(replication_path.read_text())
    if (
        replication.get("status") != "INSTRUMENT_PASS"
        or not replication.get("instrument_gate", {}).get("PASS")
        or replication.get("feature_association_opened") is not False
        or str(Path(behavior["replication_artifact"]).resolve())
        != str(replication_path)
        or behavior["replication_artifact_sha256"] != sha256_file(replication_path)
    ):
        raise RuntimeError("replication artifact contract mismatch")
    activation_path = Path(args.activation_artifact).expanduser().resolve()
    if (
        str(Path(replication["activation_artifact"]).resolve())
        != str(activation_path)
        or replication["activation_artifact_sha256"] != sha256_file(activation_path)
    ):
        raise RuntimeError("activation artifact differs from replication")
    activation = json.loads(activation_path.read_text())
    arrays_path = Path(activation["arrays"]["path"]).expanduser().resolve()
    if activation["arrays"]["sha256"] != sha256_file(arrays_path):
        raise RuntimeError("activation arrays changed")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}

    genes = [str(row["gene_id"]) for row in activation["states"]]
    _, validation_mask = INSTRUMENT.gene_train_validation_split(genes)
    validation_indices = np.flatnonzero(validation_mask)
    states = [activation["states"][int(index)] for index in validation_indices]
    target_pairs = BEHAVIOR.counterfactual_pairs(
        states, variant_kind="target_shuffle"
    )
    sham_pairs = BEHAVIOR.counterfactual_pairs(
        states, variant_kind="sham_shuffle"
    )
    arrays = {
        key: value[validation_indices]
        for key, value in arrays.items()
        if key.split("__", 1)[0] in {str(seed) for seed in EXPECTED_SEEDS}
    }

    model_path = Path(replication["model_artifact"]["path"]).expanduser().resolve()
    if replication["model_artifact"]["sha256"] != sha256_file(model_path):
        raise RuntimeError("replicated J-SAE model artifact changed")
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "gfmcd.splicebert.jacobian_sae.replication.v1":
        raise RuntimeError("replicated J-SAE model schema mismatch")
    checkpoints = BEHAVIOR.parse_checkpoints(args.checkpoint)
    base_model = Path(args.base_model).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    if output == manifest_path or output.suffix != ".json":
        raise ValueError("delta preflight needs a distinct .json output")
    device = torch.device(args.device)

    frozen: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    mechanical_cells: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        view = str(seed)
        inputs = torch.from_numpy(arrays[f"{view}__input"])
        writes = torch.from_numpy(arrays[f"{view}__write"])
        weights, checkpoint_hash = INSTRUMENT.extract_mlp_weights(
            Path(payload["checkpoint_models"][view]["model_dir"])
        )
        if checkpoint_hash != payload["checkpoint_models"][view]["checkpoint_hash"]:
            raise RuntimeError(f"J-SAE source checkpoint {view} changed")
        frozen[view] = {}
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
            reconstructed, codes, decoder = reconstruct_writes_and_codes(
                model, writes, batch_size=args.batch
            )
            frozen[view][str(initialization)] = {
                "reconstructed": reconstructed,
            }
            errors = []
            for family in (target_pairs, sham_pairs):
                for endpoint in ENDPOINT_LABELS:
                    errors.append(
                        decoder_difference_error(
                            reconstructed,
                            codes,
                            decoder,
                            family[endpoint],
                        )
                    )
            decomposition_error = max(errors)
            identity_error = float(
                np.max(
                    np.abs(
                        writes.cpu().numpy()
                        + reconstructed
                        - reconstructed
                        - writes.cpu().numpy()
                    )
                )
            )
            gates = {
                "all_feature_decomposition": (
                    decomposition_error <= DECOMPOSITION_ATOL
                ),
                "zero_delta_identity": identity_error <= IDENTITY_ATOL,
            }
            mechanical_cells.append(
                {
                    "checkpoint_seed": seed,
                    "initialization": initialization,
                    "decoder_difference_max_abs_error": decomposition_error,
                    "zero_delta_identity_max_abs_error": identity_error,
                    "gates": gates,
                    "PASS": all(gates.values()),
                }
            )
            del model, codes, decoder
        if device.type == "mps":
            torch.mps.empty_cache()
    arrays_frozen_before_classifier_load = True

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
                    model.classifier.bias[label] - model.classifier.bias[0]
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

    cells: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        view = str(seed)
        residual = arrays[f"{view}__input"].astype(np.float64)
        native_write = arrays[f"{view}__write"].astype(np.float64)
        for initialization in INITIALIZATIONS:
            reconstructed = frozen[view][str(initialization)][
                "reconstructed"
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

    mechanical_pass = all(cell["PASS"] for cell in mechanical_cells)
    behavioral_pass = all(cell["PASS"] for cell in cells)
    overall_pass = mechanical_pass and behavioral_pass
    status = "INSTRUMENT_PASS" if overall_pass else "INSTRUMENT_UNRESOLVED"
    result = {
        "study": "SpliceBERT GTEx layer-5 Jacobian SAE",
        "stage": "residual-preserving all-feature delta preflight",
        "status": status,
        "evidence_tier": "instrument",
        "protocol": str(protocol.relative_to(REPO)),
        "biological_protocol": str(biological_protocol.relative_to(REPO)),
        "implementation_addendum": str(addendum.relative_to(REPO)),
        "behavior_artifact": str(behavior_path),
        "behavior_artifact_sha256": sha256_file(behavior_path),
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
            "dense_effect_minimum": DENSE_EFFECT_THRESHOLD,
            "relative_error_maximum": RELATIVE_ERROR_THRESHOLD,
            "sham_minus_target_absolute_error_maximum": SHAM_ERROR_ALLOWANCE,
            "decoder_decomposition_max_abs_error": DECOMPOSITION_ATOL,
            "zero_delta_identity_max_abs_error": IDENTITY_ATOL,
            "required_behavioral_cells": 30,
            "required_mechanical_cells": 15,
        },
        "mechanical_cells": mechanical_cells,
        "cells": cells,
        "instrument_gate": {
            "decoded_arrays_frozen_before_classifier_load": (
                arrays_frozen_before_classifier_load
            ),
            "mechanical_PASS": mechanical_pass,
            "all_30_behavioral_cells_PASS": behavioral_pass,
            "PASS": overall_pass,
        },
        "complete_write_replacement_authorized": False,
        "residual_preserving_feature_delta_authorized": overall_pass,
        "feature_identity_opened": False,
        "feature_semantics_established": False,
        "licensed_claim": (
            "The complete output-feature delta approximates the native final-MLP "
            "write delta at the classifier while native reconstruction error is "
            "retained; frozen feature/edge candidate generation is authorized."
            if overall_pass
            else "Residual-preserving J-SAE feature/edge discovery and steering "
            "remain closed."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:jacobian_sae_delta_preflight_v1",
        sys.argv,
        repo=REPO,
        status=status,
        config={
            "batch": args.batch,
            "device": str(device),
            "dense_effect_threshold": DENSE_EFFECT_THRESHOLD,
            "relative_error_threshold": RELATIVE_ERROR_THRESHOLD,
            "sham_error_allowance": SHAM_ERROR_ALLOWANCE,
            "decomposition_atol": DECOMPOSITION_ATOL,
            "identity_atol": IDENTITY_ATOL,
        },
        seeds={
            "checkpoint_family": list(EXPECTED_SEEDS),
            "dictionary_initializations": list(INITIALIZATIONS),
        },
        notes=[
            "This is a post-v1-failure instrument repair, not outcome-fresh.",
            "All decoded arrays were computed before classifier loading.",
            "No individual feature identity, edge, or association was inspected.",
            "Complete-write replacement remains unauthorized regardless of result.",
        ],
    )
    for path in (
        protocol,
        biological_protocol,
        addendum,
        behavior_path,
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
        )
    )


if __name__ == "__main__":
    main()
