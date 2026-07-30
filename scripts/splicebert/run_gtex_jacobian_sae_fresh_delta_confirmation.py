#!/usr/bin/env python
"""Confirm the original J-SAE delta interface on gene-fresh states."""

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
DELTA = load_script("run_gtex_jacobian_sae_delta_preflight.py")
COHORT = load_script("build_gtex_jacobian_sae_fresh_cohort.py")
EXPECTED_SEEDS = tuple(range(2020, 2025))
INITIALIZATIONS = (0, 1, 2)
ENDPOINT_LABELS = {"donor": 1, "acceptor": 2}
ERROR_SHAPLEY_FRACTION_MAXIMUM = 0.10
JOINT_MARGIN_ATOL = 1e-10


def two_block_shapley(
    residual: np.ndarray,
    native_write: np.ndarray,
    reconstructed_write: np.ndarray,
    pairs: np.ndarray,
    parameters: Mapping[str, Any],
) -> dict[str, float]:
    """Allocate paired dense-write behavior between feature and error blocks."""

    clean = pairs[:, 0]
    changed = pairs[:, 1]

    def direction(
        baseline_residual: np.ndarray,
        baseline_write: np.ndarray,
        dense_delta: np.ndarray,
        feature_delta: np.ndarray,
        *,
        sign: float,
    ) -> dict[str, float]:
        error_delta = dense_delta - feature_delta
        baseline = composed_classifier_margin(
            baseline_residual, baseline_write, **parameters
        )
        feature_margin = composed_classifier_margin(
            baseline_residual,
            baseline_write + feature_delta,
            **parameters,
        )
        error_margin = composed_classifier_margin(
            baseline_residual,
            baseline_write + error_delta,
            **parameters,
        )
        joint_margin = composed_classifier_margin(
            baseline_residual,
            baseline_write + feature_delta + error_delta,
            **parameters,
        )
        dense_margin = composed_classifier_margin(
            baseline_residual,
            baseline_write + dense_delta,
            **parameters,
        )
        feature_value = sign * (feature_margin - baseline)
        error_value = sign * (error_margin - baseline)
        joint_value = sign * (joint_margin - baseline)
        feature_shapley = 0.5 * (
            feature_value + joint_value - error_value
        )
        error_shapley = 0.5 * (
            error_value + joint_value - feature_value
        )
        dense_mean = float(np.mean(joint_value))
        error_mean = float(np.mean(error_shapley))
        return {
            "dense_effect": dense_mean,
            "feature_only_effect": float(np.mean(feature_value)),
            "error_only_effect": float(np.mean(error_value)),
            "feature_shapley_effect": float(np.mean(feature_shapley)),
            "error_shapley_effect": error_mean,
            "error_shapley_fraction": (
                abs(error_mean) / abs(dense_mean)
                if abs(dense_mean) > np.finfo(np.float64).eps
                else float("inf")
            ),
            "mean_absolute_error_shapley": float(
                np.mean(np.abs(error_shapley))
            ),
            "shapley_sum_max_abs_error": float(
                np.max(
                    np.abs(feature_shapley + error_shapley - joint_value)
                )
            ),
            "joint_dense_margin_max_abs_error": float(
                np.max(np.abs(joint_margin - dense_margin))
            ),
        }

    clean_write = native_write[clean]
    changed_write = native_write[changed]
    feature_changed = (
        reconstructed_write[changed] - reconstructed_write[clean]
    )
    damage = direction(
        residual[clean],
        clean_write,
        changed_write - clean_write,
        feature_changed,
        sign=-1.0,
    )
    feature_clean = -feature_changed
    recovery = direction(
        residual[changed],
        changed_write,
        clean_write - changed_write,
        feature_clean,
        sign=1.0,
    )
    return {"damage": damage, "recovery": recovery}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-artifact", required=True)
    parser.add_argument("--development-artifact", required=True)
    parser.add_argument("--replication-artifact", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("--batch must be positive")

    addendum = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "IMPLEMENTATION_ADDENDUM_JACOBIAN_SAE_FRESH_DELTA_CONFIRMATION_V1.md"
    )
    capture_path = Path(args.capture_artifact).expanduser().resolve()
    capture = json.loads(capture_path.read_text())
    if (
        capture.get("status") != "COMPLETE"
        or capture.get("namespace") != COHORT.NAMESPACE
        or capture.get("confirmation_authorized") is not True
        or capture.get("classifier_output_observed") is not False
        or not capture.get("instrument_gate", {}).get("PASS")
    ):
        raise RuntimeError("fresh activation capture contract mismatch")
    development_path = Path(args.development_artifact).expanduser().resolve()
    development = json.loads(development_path.read_text())
    if (
        development.get("status") != "INSTRUMENT_UNRESOLVED"
        or development.get("instrument_gate", {}).get("PASS") is not False
        or development.get("residual_preserving_feature_delta_authorized")
        is not False
    ):
        raise RuntimeError("development failure is not preserved")
    replication_path = Path(args.replication_artifact).expanduser().resolve()
    replication = json.loads(replication_path.read_text())
    if (
        replication.get("status") != "INSTRUMENT_PASS"
        or not replication.get("instrument_gate", {}).get("PASS")
        or replication.get("feature_association_opened") is not False
    ):
        raise RuntimeError("replication artifact contract mismatch")
    arrays_path = Path(capture["arrays"]["path"]).expanduser().resolve()
    if capture["arrays"]["sha256"] != sha256_file(arrays_path):
        raise RuntimeError("fresh activation arrays changed")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    states = capture["states"]
    target_pairs = BEHAVIOR.counterfactual_pairs(
        states, variant_kind="target_shuffle"
    )
    sham_pairs = BEHAVIOR.counterfactual_pairs(
        states, variant_kind="sham_shuffle"
    )

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
        raise ValueError("confirmation needs a distinct .json output")
    device = torch.device(args.device)

    frozen: dict[str, dict[str, np.ndarray]] = {}
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
            reconstructed, codes, decoder = DELTA.reconstruct_writes_and_codes(
                model, writes, batch_size=args.batch
            )
            frozen[view][str(initialization)] = reconstructed
            errors = [
                DELTA.decoder_difference_error(
                    reconstructed,
                    codes,
                    decoder,
                    family[endpoint],
                )
                for family in (target_pairs, sham_pairs)
                for endpoint in ENDPOINT_LABELS
            ]
            decomposition_error = max(errors)
            native64 = writes.cpu().numpy().astype(np.float64)
            reconstructed64 = reconstructed.astype(np.float64)
            zero_delta = reconstructed64 - reconstructed64
            identity_error = float(
                np.max(np.abs(native64 + zero_delta - native64))
            )
            gates = {
                "all_feature_decomposition": (
                    decomposition_error <= DELTA.DECOMPOSITION_ATOL
                ),
                "zero_delta_identity": (
                    identity_error <= DELTA.IDENTITY_ATOL
                ),
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
        expected_hash = capture["view_records"][view]["checkpoint_sha256"]
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
            reconstructed = frozen[view][str(initialization)].astype(np.float64)
            for endpoint in ENDPOINT_LABELS:
                metrics = DELTA.endpoint_metrics(
                    residual,
                    native_write,
                    reconstructed,
                    target_pairs[endpoint],
                    sham_pairs[endpoint],
                    classifier_parameters[view][endpoint],
                )
                target_accounting = two_block_shapley(
                    residual,
                    native_write,
                    reconstructed,
                    target_pairs[endpoint],
                    classifier_parameters[view][endpoint],
                )
                sham_accounting = two_block_shapley(
                    residual,
                    native_write,
                    reconstructed,
                    sham_pairs[endpoint],
                    classifier_parameters[view][endpoint],
                )
                accounting_gates = {
                    "damage_joint_replay": (
                        target_accounting["damage"][
                            "joint_dense_margin_max_abs_error"
                        ]
                        <= JOINT_MARGIN_ATOL
                    ),
                    "recovery_joint_replay": (
                        target_accounting["recovery"][
                            "joint_dense_margin_max_abs_error"
                        ]
                        <= JOINT_MARGIN_ATOL
                    ),
                    "damage_error_fraction": (
                        target_accounting["damage"][
                            "error_shapley_fraction"
                        ]
                        <= ERROR_SHAPLEY_FRACTION_MAXIMUM
                    ),
                    "recovery_error_fraction": (
                        target_accounting["recovery"][
                            "error_shapley_fraction"
                        ]
                        <= ERROR_SHAPLEY_FRACTION_MAXIMUM
                    ),
                }
                combined_gates = {**metrics["gates"], **accounting_gates}
                cells.append(
                    {
                        "checkpoint_seed": seed,
                        "initialization": initialization,
                        "endpoint": endpoint,
                        "target_shuffle": metrics["target_shuffle"],
                        "sham_shuffle": metrics["sham_shuffle"],
                        "target_error_accounting": target_accounting,
                        "sham_error_accounting": sham_accounting,
                        "gates": combined_gates,
                        "PASS": all(combined_gates.values()),
                    }
                )

    mechanical_pass = all(cell["PASS"] for cell in mechanical_cells)
    behavioral_pass = all(cell["PASS"] for cell in cells)
    overall_pass = mechanical_pass and behavioral_pass
    status = "INSTRUMENT_PASS" if overall_pass else "INSTRUMENT_UNRESOLVED"
    result = {
        "study": "SpliceBERT GTEx layer-5 Jacobian SAE",
        "stage": "gene-fresh residual-preserving delta confirmation",
        "status": status,
        "evidence_tier": "instrument",
        "implementation_addendum": str(addendum.relative_to(REPO)),
        "capture_artifact": str(capture_path),
        "capture_artifact_sha256": sha256_file(capture_path),
        "development_artifact": str(development_path),
        "development_artifact_sha256": sha256_file(development_path),
        "replication_artifact": str(replication_path),
        "replication_artifact_sha256": sha256_file(replication_path),
        "checkpoint_records": checkpoint_records,
        "fresh_cohort": {
            "namespace": COHORT.NAMESPACE,
            "n_states": len(states),
            "n_genes": len({str(row["gene_id"]) for row in states}),
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
            "dense_effect_minimum": DELTA.DENSE_EFFECT_THRESHOLD,
            "relative_error_maximum": DELTA.RELATIVE_ERROR_THRESHOLD,
            "sham_minus_target_absolute_error_maximum": (
                DELTA.SHAM_ERROR_ALLOWANCE
            ),
            "decoder_decomposition_max_abs_error": (
                DELTA.DECOMPOSITION_ATOL
            ),
            "zero_delta_identity_max_abs_error": DELTA.IDENTITY_ATOL,
            "error_shapley_fraction_maximum": (
                ERROR_SHAPLEY_FRACTION_MAXIMUM
            ),
            "joint_margin_max_abs_error": JOINT_MARGIN_ATOL,
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
        "individual_feature_identity_opened": False,
        "feature_semantics_established": False,
        "licensed_claim": (
            "The original J-SAE feature block generalizes to gene-fresh states "
            "as a faithful residual-preserving coordinate system for the "
            "target-shuffle final-MLP path; frozen feature/edge candidate "
            "generation is authorized."
            if overall_pass
            else "Residual-preserving J-SAE feature/edge discovery and steering "
            "remain closed."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:jacobian_sae_fresh_delta_confirmation_v1",
        sys.argv,
        repo=REPO,
        status=status,
        config={
            "batch": args.batch,
            "device": str(device),
            "relative_error_threshold": DELTA.RELATIVE_ERROR_THRESHOLD,
            "sham_error_allowance": DELTA.SHAM_ERROR_ALLOWANCE,
            "error_shapley_fraction_maximum": (
                ERROR_SHAPLEY_FRACTION_MAXIMUM
            ),
        },
        seeds={
            "checkpoint_family": list(EXPECTED_SEEDS),
            "dictionary_initializations": list(INITIALIZATIONS),
        },
        notes=[
            "No decoder refit or feature selection was performed.",
            "All decoded arrays were frozen before classifier directions loaded.",
            "The error block is a dense uninterpreted remainder, not a feature.",
            "The exposed development failure remains preserved.",
        ],
    )
    for path in (
        addendum,
        capture_path,
        arrays_path,
        development_path,
        replication_path,
        model_path,
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
