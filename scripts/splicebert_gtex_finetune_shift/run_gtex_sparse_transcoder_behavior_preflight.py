#!/usr/bin/env python
"""Apply frozen native heads to preflight label-free sparse reconstructions."""

from __future__ import annotations

import argparse
from dataclasses import fields
import gc
import hashlib
import json
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
)
from gfmcd.provenance import (  # noqa: E402
    ExperimentManifest,
    file_record,
    sha256_file,
)
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.mlp_units import composed_classifier_margin  # noqa: E402


EXPECTED_SEEDS = tuple(range(2020, 2025))
VIEWS = ("base", *(str(seed) for seed in EXPECTED_SEEDS))
TUNED_VIEWS = tuple(str(seed) for seed in EXPECTED_SEEDS)
TARGET_SHUFFLE_KIND = "target_shuffle"
ENDPOINT_LABELS = {"donor": 1, "acceptor": 2}
MARGIN_MAE_THRESHOLD = 0.05
EFFECT_RECOVERY_THRESHOLD = 0.90
NATIVE_EFFECT_THRESHOLD = 0.05


def stable_identifier_hash(values: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


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


def config_from_dict(values: Mapping[str, Any]) -> StructuredMultiViewTranscoderConfig:
    allowed = {field.name for field in fields(StructuredMultiViewTranscoderConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown structured-transcoder config fields: {unknown}")
    payload = dict(values)
    payload["views"] = tuple(payload["views"])
    payload["tuned_views"] = tuple(payload["tuned_views"])
    return StructuredMultiViewTranscoderConfig(**payload)


def target_shuffle_pairs(
    states: list[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    index = {str(row["state_id"]): position for position, row in enumerate(states)}
    pairs: dict[str, list[tuple[int, int]]] = {
        endpoint: [] for endpoint in ENDPOINT_LABELS
    }
    for changed, row in enumerate(states):
        endpoint = str(row["endpoint"])
        if (
            endpoint not in pairs
            or str(row["variant_kind"]) != TARGET_SHUFFLE_KIND
        ):
            continue
        reference_id = row.get("reference_state_id")
        if reference_id is None:
            raise ValueError("target shuffle lacks a reference_state_id")
        reference = index[str(reference_id)]
        reference_row = states[reference]
        if (
            str(reference_row["endpoint"]) != endpoint
            or str(reference_row["gene_id"]) != str(row["gene_id"])
        ):
            raise ValueError("target shuffle/reference identity mismatch")
        pairs[endpoint].append((reference, changed))
    result = {
        endpoint: np.asarray(values, dtype=np.int64).reshape(-1, 2)
        for endpoint, values in pairs.items()
    }
    if any(len(values) < 2 for values in result.values()):
        raise RuntimeError("both endpoints require target-shuffle pairs")
    return result


@torch.no_grad()
def predict_write(
    model: StructuredMultiViewSparseTranscoder,
    view: str,
    inputs: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    for start in range(0, len(inputs), batch_size):
        normalized = (inputs[start : start + batch_size] - mean) / scale
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device)
        _, prediction = model(view, tensor)
        predictions.append(prediction.detach().cpu().numpy())
    return np.concatenate(predictions)


def endpoint_metrics(
    residual: np.ndarray,
    native_write: np.ndarray,
    reconstructed_write: np.ndarray,
    pairs: np.ndarray,
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
    selected = np.unique(pairs.reshape(-1))
    margin_mae = float(
        np.mean(
            np.abs(
                reconstructed_margin[selected] - native_margin[selected]
            )
        )
    )
    native_effects = (
        native_margin[pairs[:, 0]] - native_margin[pairs[:, 1]]
    )
    reconstructed_effects = (
        reconstructed_margin[pairs[:, 0]]
        - reconstructed_margin[pairs[:, 1]]
    )
    native_effect = float(native_effects.mean())
    reconstructed_effect = float(reconstructed_effects.mean())
    ratio = (
        reconstructed_effect / native_effect
        if abs(native_effect) > np.finfo(np.float64).eps
        else float("nan")
    )
    passed = (
        margin_mae <= MARGIN_MAE_THRESHOLD
        and native_effect >= NATIVE_EFFECT_THRESHOLD
        and reconstructed_effect > 0.0
        and ratio >= EFFECT_RECOVERY_THRESHOLD
    )
    return {
        "margin_mae": margin_mae,
        "native_target_shuffle_effect": native_effect,
        "reconstructed_target_shuffle_effect": reconstructed_effect,
        "effect_recovery_ratio": ratio,
        "PASS": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-artifact", required=True)
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
        "PREREGISTRATION_NATIVE_BIOLOGICAL_FEATURES_V1.md"
    )
    addendum = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "IMPLEMENTATION_ADDENDUM_SPARSE_TRANSCODER_V1.md"
    )
    training_path = Path(args.training_artifact).expanduser().resolve()
    training = json.loads(training_path.read_text())
    if (
        training.get("status") != "COMPLETE"
        or training.get("stage")
        != "label-free sparse training and representation preflight"
        or training.get("label_free") is not True
        or training.get("behavioral_preflight_authorized") is not True
        or not training.get("instrument_gate", {}).get("PASS")
    ):
        raise RuntimeError("training artifact does not authorize behavioral preflight")
    activation_path = Path(args.activation_artifact).expanduser().resolve()
    activation = json.loads(activation_path.read_text())
    if (
        str(activation_path) != str(Path(training["activation_artifact"]))
        or activation["arrays"]["sha256"]
        != training["activation_arrays_sha256"]
    ):
        raise RuntimeError("training/activation artifact identity mismatch")
    arrays_path = Path(activation["arrays"]["path"]).expanduser().resolve()
    if sha256_file(arrays_path) != activation["arrays"]["sha256"]:
        raise RuntimeError("activation array hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    states = activation["states"]
    split = training.get("dictionary_split", {})
    validation_gene_ids = split.get("validation_gene_ids")
    if not isinstance(validation_gene_ids, list) or not validation_gene_ids:
        raise RuntimeError("training artifact lacks frozen validation gene identities")
    validation_gene_ids = [str(value) for value in validation_gene_ids]
    if stable_identifier_hash(validation_gene_ids) != split.get(
        "validation_gene_ids_sha256"
    ):
        raise RuntimeError("validation gene identity hash mismatch")
    validation_gene_set = set(validation_gene_ids)
    validation_indices = np.asarray(
        [
            index
            for index, row in enumerate(states)
            if str(row["gene_id"]) in validation_gene_set
        ],
        dtype=np.int64,
    )
    validation_states = [states[int(index)] for index in validation_indices]
    validation_state_ids = [str(row["state_id"]) for row in validation_states]
    if (
        len(validation_states) != int(split.get("n_validation_states", -1))
        or stable_identifier_hash(validation_state_ids)
        != split.get("validation_state_ids_sha256")
        or {str(row["gene_id"]) for row in validation_states}
        != validation_gene_set
    ):
        raise RuntimeError("held-out validation state identity mismatch")
    arrays = {
        key: value[validation_indices]
        for key, value in arrays.items()
    }
    states = validation_states
    shuffle_pairs = target_shuffle_pairs(states)

    model_path = Path(training["model_artifact"]["path"]).expanduser().resolve()
    if sha256_file(model_path) != training["model_artifact"]["sha256"]:
        raise RuntimeError("sparse model artifact hash mismatch")
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != 1
        or tuple(payload.get("views", [])) != VIEWS
        or tuple(payload.get("tuned_views", [])) != TUNED_VIEWS
    ):
        raise RuntimeError("sparse model payload schema mismatch")
    checkpoints = parse_checkpoints(args.checkpoint)
    base_model = Path(args.base_model).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    if output == manifest_path or output.suffix != ".json":
        raise ValueError("behavioral preflight requires a distinct .json result")
    device = torch.device(args.device)

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
            raise RuntimeError(f"checkpoint {seed} changed since activation capture")
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
                    model.classifier.weight[label]
                    - model.classifier.weight[0]
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

    candidate_results: dict[str, Any] = {}
    passing_settings = []
    for setting, candidate in training["candidates"].items():
        if not candidate["representation_gate"]:
            candidate_results[setting] = {
                "representation_gate": False,
                "behavioral_gate": False,
                "reason": "representation preflight failed",
            }
            continue
        replicate_results = []
        for initialization in range(3):
            model_id = f"{setting}_r{initialization}"
            saved = payload["models"][model_id]
            config = config_from_dict(saved["config"])
            model = StructuredMultiViewSparseTranscoder(config)
            model.load_state_dict(saved["state_dict"], strict=True)
            model.to(device)
            per_view: dict[str, Any] = {}
            for view in TUNED_VIEWS:
                normalization = payload["normalization"][view]
                reconstructed = predict_write(
                    model,
                    view,
                    arrays[f"{view}__input"],
                    np.asarray(normalization["mean"]),
                    np.asarray(normalization["scale"]),
                    batch_size=args.batch,
                )
                per_view[view] = {}
                for endpoint in ENDPOINT_LABELS:
                    per_view[view][endpoint] = endpoint_metrics(
                        arrays[f"{view}__input"].astype(np.float64),
                        arrays[f"{view}__write"].astype(np.float64),
                        reconstructed.astype(np.float64),
                        shuffle_pairs[endpoint],
                        classifier_parameters[view][endpoint],
                    )
            replicate_pass = all(
                per_view[view][endpoint]["PASS"]
                for view in TUNED_VIEWS
                for endpoint in ENDPOINT_LABELS
            )
            replicate_results.append(
                {
                    "initialization": initialization,
                    "views": per_view,
                    "PASS": replicate_pass,
                }
            )
            del model
        behavioral_pass = all(row["PASS"] for row in replicate_results)
        candidate_results[setting] = {
            "representation_gate": True,
            "replicates": replicate_results,
            "thresholds": {
                "margin_mae_maximum": MARGIN_MAE_THRESHOLD,
                "native_target_shuffle_effect_minimum": NATIVE_EFFECT_THRESHOLD,
                "effect_recovery_ratio_minimum": EFFECT_RECOVERY_THRESHOLD,
                "required_replicates": 3,
                "required_checkpoint_signs": 5,
            },
            "behavioral_gate": behavioral_pass,
        }
        if behavioral_pass:
            passing_settings.append(setting)

    if not passing_settings:
        selected = None
        status = "INSTRUMENT_UNRESOLVED"
        overall_pass = False
    else:
        def selection_key(setting: str) -> tuple[float, float, float, str]:
            candidate = training["candidates"][setting]
            validation_error = np.mean(
                [
                    1.0 - row["mean_write_delta_fve"]
                    for row in candidate["replicates"]
                ]
            )
            return (
                float(validation_error),
                float(candidate["feature_width"]),
                float(candidate["active_features"]),
                setting,
            )

        selected = min(passing_settings, key=selection_key)
        candidate = training["candidates"][selected]
        primary_initialization = min(
            range(3),
            key=lambda initialization: (
                1.0
                - candidate["replicates"][initialization][
                    "mean_write_delta_fve"
                ],
                initialization,
            ),
        )
        status = "COMPLETE"
        overall_pass = True

    result = {
        "study": "native GTEx six-view sparse final-MLP feature model",
        "stage": "frozen native-head behavioral preflight and model selection",
        "status": status,
        "evidence_tier": "instrument",
        "protocol": str(protocol.relative_to(REPO)),
        "implementation_addendum": str(addendum.relative_to(REPO)),
        "training_artifact": str(training_path),
        "activation_artifact": str(activation_path),
        "checkpoint_records": checkpoint_records,
        "held_out_dictionary_validation": {
            "n_states": len(states),
            "n_genes": len(validation_gene_set),
            "validation_gene_ids_sha256": split[
                "validation_gene_ids_sha256"
            ],
            "validation_state_ids_sha256": split[
                "validation_state_ids_sha256"
            ],
        },
        "target_shuffle_variant_kind": TARGET_SHUFFLE_KIND,
        "candidate_results": candidate_results,
        "selection_rule": (
            "minimum passing validation write-delta error, then feature width, then "
            "active count; primary initialization has minimum validation error"
        ),
        "selected_setting": selected,
        "selected_initialization": (
            primary_initialization if selected is not None else None
        ),
        "instrument_gate": {
            "dictionary_weights_frozen_before_native_head_access": True,
            "all_three_initializations_required": True,
            "at_least_one_complete_setting": overall_pass,
            "PASS": overall_pass,
        },
        "feature_interpretation_authorized": overall_pass,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:sparse_transcoder_behavior_preflight_v1",
        sys.argv,
        repo=REPO,
        status="INSTRUMENT" if overall_pass else "INSTRUMENT_UNRESOLVED",
        config={
            "batch": args.batch,
            "device": str(device),
            "margin_mae_threshold": MARGIN_MAE_THRESHOLD,
            "effect_recovery_threshold": EFFECT_RECOVERY_THRESHOLD,
            "native_effect_threshold": NATIVE_EFFECT_THRESHOLD,
        },
        seeds={"checkpoint_family": list(EXPECTED_SEEDS)},
        notes=[
            "Dictionary weights were loaded read-only after label-free training completed.",
            "No optimizer or feature-selection step exists in this runner.",
            "Failure leaves GFS20 native-unit evidence unchanged.",
        ],
    )
    for path in (
        protocol,
        addendum,
        training_path,
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
                "instrument_gate": result["instrument_gate"],
                "selected_setting": selected,
                "selected_initialization": result["selected_initialization"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
