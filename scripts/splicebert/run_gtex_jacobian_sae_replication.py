#!/usr/bin/env python
"""Replicate the frozen capacity-matched J-SAE across checkpoints and fits."""

from __future__ import annotations

import argparse
from itertools import combinations
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.circuits.multiview_transcoder import (  # noqa: E402
    assert_label_free_payload,
    feature_similarity_matrices,
)
from gfmcd.provenance import (  # noqa: E402
    ExperimentManifest,
    file_record,
    sha256_file,
)


def load_instrument_module() -> Any:
    path = REPO / "scripts/splicebert/run_gtex_jacobian_sae_instrument.py"
    spec = importlib.util.spec_from_file_location("gtex_jsae_instrument", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("J-SAE instrument runner loader is unavailable")
    spec.loader.exec_module(module)
    return module


INSTRUMENT = load_instrument_module()
EXPECTED_SEEDS = tuple(range(2020, 2025))
INITIALIZATIONS = (0, 1, 2)
SELECTED_COEFFICIENT = 0.25
DENSITY_FLOOR = 0.001
RECIPROCAL_FRACTION_FLOOR = 0.25
RECIPROCAL_SCORE_FLOOR = 0.50
TOP_GRAPH_EDGES = 64


def parse_model_dirs(values: list[str]) -> dict[int, Path]:
    model_dirs: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in model_dirs:
            raise ValueError(f"duplicate model seed {seed}")
        model_dirs[seed] = Path(path_text).expanduser().resolve()
    if tuple(sorted(model_dirs)) != EXPECTED_SEEDS:
        raise ValueError("complete model family 2020--2024 required")
    return model_dirs


def initialization_seed(checkpoint_seed: int, initialization: int) -> int:
    if checkpoint_seed not in EXPECTED_SEEDS or initialization not in INITIALIZATIONS:
        raise ValueError("unknown replication seed or initialization")
    return 2_000_000 + 10_000 * checkpoint_seed + initialization


def compact_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("mlp_")
    }


@torch.no_grad()
def fingerprints_and_graph(
    model: Any,
    inputs: torch.Tensor,
    writes: torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    device = model.input_center.device
    input_features = []
    output_features = []
    graph = torch.zeros(
        (
            model.output_sae.num_features,
            model.input_sae.num_features,
        ),
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        x = inputs[selected].to(device)
        y = writes[selected].to(device)
        input_acts, input_indices = model.input_sae.encode(
            model.normalize_input(x),
            return_indices=True,
        )
        output_acts, output_indices = model.output_sae.encode(
            model.normalize_output(y),
            return_indices=True,
        )
        jacobian = model.active_jacobian(
            x,
            input_indices,
            output_indices,
        )
        output_grid = output_indices[:, :, None].expand_as(jacobian)
        input_grid = input_indices[:, None, :].expand_as(jacobian)
        graph.index_put_(
            (output_grid.reshape(-1), input_grid.reshape(-1)),
            jacobian.abs().reshape(-1),
            accumulate=True,
        )
        input_features.append(input_acts.cpu().numpy())
        output_features.append(output_acts.cpu().numpy())
    graph /= len(indices)
    return (
        np.concatenate(input_features),
        np.concatenate(output_features),
        graph.cpu().numpy(),
    )


def feature_match(
    left_features: np.ndarray,
    right_features: np.ndarray,
    left_decoder: np.ndarray,
    right_decoder: np.ndarray,
    *,
    activation_weight: float = 0.5,
) -> dict[str, Any]:
    if left_features.shape[0] != right_features.shape[0]:
        raise ValueError("feature fingerprints require paired examples")
    left_density = np.mean(left_features > 0, axis=0)
    right_density = np.mean(right_features > 0, axis=0)
    left_live = np.flatnonzero(left_density >= DENSITY_FLOOR)
    right_live = np.flatnonzero(right_density >= DENSITY_FLOOR)
    if not len(left_live) or not len(right_live):
        return {
            "n_left_eligible": len(left_live),
            "n_right_eligible": len(right_live),
            "reciprocal_match_fraction": 0.0,
            "reciprocal_score_mean": 0.0,
            "PASS": False,
            "forward_map": [-1] * left_features.shape[1],
        }
    matrices = feature_similarity_matrices(
        torch.as_tensor(left_features[:, left_live]),
        torch.as_tensor(right_features[:, right_live]),
        torch.as_tensor(left_decoder[:, left_live]),
        torch.as_tensor(right_decoder[:, right_live]),
        activation_weight=activation_weight,
    )
    combined = matrices["combined"].numpy()
    forward_local = combined.argmax(axis=1)
    reverse_local = combined.argmax(axis=0)
    reciprocal = np.asarray(
        [
            reverse_local[target] == source
            for source, target in enumerate(forward_local)
        ]
    )
    scores = combined[np.arange(len(forward_local)), forward_local]
    reciprocal_scores = scores[reciprocal]
    fraction = float(np.mean(reciprocal))
    mean_score = (
        float(np.mean(reciprocal_scores))
        if len(reciprocal_scores)
        else 0.0
    )
    forward_map = np.full(left_features.shape[1], -1, dtype=np.int64)
    forward_map[left_live] = right_live[forward_local]
    return {
        "n_left_eligible": len(left_live),
        "n_right_eligible": len(right_live),
        "reciprocal_matches": int(np.sum(reciprocal)),
        "reciprocal_match_fraction": fraction,
        "reciprocal_score_mean": mean_score,
        "activation_correlation_mean_for_reciprocal": (
            float(
                np.mean(
                    matrices["activation_correlation"].numpy()[
                        np.arange(len(forward_local))[reciprocal],
                        forward_local[reciprocal],
                    ]
                )
            )
            if np.any(reciprocal)
            else 0.0
        ),
        "decoder_cosine_mean_for_reciprocal": (
            float(
                np.mean(
                    matrices["decoder_cosine"].numpy()[
                        np.arange(len(forward_local))[reciprocal],
                        forward_local[reciprocal],
                    ]
                )
            )
            if np.any(reciprocal)
            else 0.0
        ),
        "density_floor": DENSITY_FLOOR,
        "activation_weight": activation_weight,
        "thresholds": {
            "reciprocal_match_fraction": RECIPROCAL_FRACTION_FLOOR,
            "reciprocal_score_mean": RECIPROCAL_SCORE_FLOOR,
        },
        "PASS": (
            fraction >= RECIPROCAL_FRACTION_FLOOR
            and mean_score >= RECIPROCAL_SCORE_FLOOR
        ),
        "forward_map": forward_map.tolist(),
    }


def graph_alignment(
    left_graph: np.ndarray,
    right_graph: np.ndarray,
    input_map: list[int],
    output_map: list[int],
) -> dict[str, float | int]:
    input_map_array = np.asarray(input_map, dtype=np.int64)
    output_map_array = np.asarray(output_map, dtype=np.int64)
    valid_input = np.flatnonzero(input_map_array >= 0)
    valid_output = np.flatnonzero(output_map_array >= 0)
    if not len(valid_input) or not len(valid_output):
        return {
            "valid_aligned_edges": 0,
            "aligned_matrix_cosine": 0.0,
            "top64_mapped_edge_overlap": 0.0,
        }
    left_values = left_graph[np.ix_(valid_output, valid_input)].reshape(-1)
    right_values = right_graph[
        np.ix_(
            output_map_array[valid_output],
            input_map_array[valid_input],
        )
    ].reshape(-1)
    denominator = np.linalg.norm(left_values) * np.linalg.norm(right_values)
    cosine = (
        float(left_values @ right_values / denominator)
        if denominator > 0
        else 0.0
    )
    left_top_flat = np.argpartition(
        left_graph.reshape(-1),
        -min(TOP_GRAPH_EDGES, left_graph.size),
    )[-TOP_GRAPH_EDGES:]
    left_rows, left_columns = np.unravel_index(left_top_flat, left_graph.shape)
    mapped_edges = {
        (int(output_map_array[row]), int(input_map_array[column]))
        for row, column in zip(left_rows, left_columns)
        if output_map_array[row] >= 0 and input_map_array[column] >= 0
    }
    right_top_flat = np.argpartition(
        right_graph.reshape(-1),
        -min(TOP_GRAPH_EDGES, right_graph.size),
    )[-TOP_GRAPH_EDGES:]
    right_rows, right_columns = np.unravel_index(
        right_top_flat,
        right_graph.shape,
    )
    right_edges = set(zip(map(int, right_rows), map(int, right_columns)))
    overlap = len(mapped_edges & right_edges) / TOP_GRAPH_EDGES
    return {
        "valid_aligned_edges": int(len(left_values)),
        "aligned_matrix_cosine": cosine,
        "top64_mapped_edge_overlap": float(overlap),
    }


def representation_gate(
    candidate_train: dict[str, Any],
    candidate_validation: dict[str, Any],
    baseline_validation: dict[str, Any],
    *,
    mechanical_pass: bool,
) -> dict[str, Any]:
    baseline_e90 = baseline_validation["active_edge_mass_count"]["E90"]["median"]
    candidate_e90 = candidate_validation["active_edge_mass_count"]["E90"]["median"]
    reduction = (
        (baseline_e90 - candidate_e90) / baseline_e90
        if baseline_e90 > 0
        else 0.0
    )
    gates = {
        "mechanical": mechanical_pass,
        "input_dead_fraction": (
            candidate_train["input_dead_feature_fraction"] <= 0.50
        ),
        "output_dead_fraction": (
            candidate_train["output_dead_feature_fraction"] <= 0.50
        ),
        "input_fve_floor": candidate_validation["input_fve"] >= 0.85,
        "output_fve_floor": (
            candidate_validation["output_write_fve"] >= 0.95
        ),
        "sandwich_fve_floor": (
            candidate_validation["sandwich_output_fve"] >= 0.90
        ),
        "input_fve_matched": (
            candidate_validation["input_fve"]
            >= baseline_validation["input_fve"] - 0.03
        ),
        "output_fve_matched": (
            candidate_validation["output_write_fve"]
            >= baseline_validation["output_write_fve"] - 0.03
        ),
        "sandwich_fve_matched": (
            candidate_validation["sandwich_output_fve"]
            >= baseline_validation["sandwich_output_fve"] - 0.03
        ),
        "E90_reduction_at_least_15_percent": reduction >= 0.15,
    }
    return {
        "gates": gates,
        "median_E90_reduction_fraction": reduction,
        "PASS": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument-artifact", required=True)
    parser.add_argument("--activation-artifact", required=True)
    parser.add_argument("--model-dir", action="append", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=INSTRUMENT.DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=INSTRUMENT.DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=INSTRUMENT.DEFAULT_PATIENCE)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (
        args.batch != INSTRUMENT.DEFAULT_BATCH
        or args.epochs != INSTRUMENT.DEFAULT_EPOCHS
        or args.patience != INSTRUMENT.DEFAULT_PATIENCE
    ):
        raise ValueError("replication arguments differ from the frozen addendum")
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
        "PREREGISTRATION_JACOBIAN_SAE_V3.md"
    )
    addendum = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "IMPLEMENTATION_ADDENDUM_JACOBIAN_SAE_REPLICATION_V1.md"
    )
    instrument_path = Path(args.instrument_artifact).expanduser().resolve()
    instrument = json.loads(instrument_path.read_text())
    if (
        instrument.get("status") != "INSTRUMENT_PASS"
        or instrument.get("selection", {}).get("selected_coefficient")
        != SELECTED_COEFFICIENT
        or instrument.get("five_seed_replication_authorized") is not True
        or instrument.get("downstream_behavior_opened") is not False
    ):
        raise RuntimeError("v3 artifact does not authorize representation replication")
    activation_path = Path(args.activation_artifact).expanduser().resolve()
    if (
        str(activation_path) != str(
            Path(instrument["data"]["activation_artifact"]).resolve()
        )
        or sha256_file(activation_path)
        != instrument["data"]["activation_artifact_sha256"]
    ):
        raise RuntimeError("replication activation artifact differs from v3")
    model_dirs = parse_model_dirs(args.model_dir)
    output = guard_output_path(Path(args.out).resolve())
    model_output = guard_output_path(Path(args.model_out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    if len({output, model_output, manifest_path}) != 3:
        raise ValueError("result, model, and manifest outputs must be distinct")

    activation_report = json.loads(activation_path.read_text())
    genes = [str(row["gene_id"]) for row in activation_report["states"]]
    train_mask, validation_mask = INSTRUMENT.gene_train_validation_split(genes)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    replicate_records: dict[str, Any] = {}
    fitted: dict[str, list[dict[str, Any]]] = {}
    model_payload: dict[str, Any] = {
        "schema": "gfmcd.splicebert.jacobian_sae.replication.v1",
        "selected_coefficient": SELECTED_COEFFICIENT,
        "checkpoint_models": {},
    }
    checkpoint_hashes = {}

    for checkpoint_seed in EXPECTED_SEEDS:
        view = str(checkpoint_seed)
        _, input_array, write_array = INSTRUMENT.load_activation_artifact(
            activation_path,
            view=view,
        )
        inputs = torch.from_numpy(input_array)
        writes = torch.from_numpy(write_array)
        weights, checkpoint_hash = INSTRUMENT.extract_mlp_weights(
            model_dirs[checkpoint_seed]
        )
        checkpoint_hashes[view] = checkpoint_hash
        replay_model = INSTRUMENT.build_pair(
            weights,
            inputs[train_indices],
            writes[train_indices],
            device=device,
            capacity_repair=True,
        )
        maximum_replay_error = 0.0
        with torch.no_grad():
            for start in range(0, len(inputs), args.batch):
                x = inputs[start : start + args.batch].to(device)
                y = writes[start : start + args.batch].to(device)
                replay = replay_model.mlp(x)
                maximum_replay_error = max(
                    maximum_replay_error,
                    float((replay - y).abs().max().cpu()),
                )
        if maximum_replay_error > INSTRUMENT.MLP_REPLAY_TOLERANCE:
            raise RuntimeError(f"checkpoint {view} MLP replay failed")
        del replay_model

        fitted[view] = []
        replicate_records[view] = {
            "checkpoint_hash": checkpoint_hash,
            "maximum_mlp_replay_error": maximum_replay_error,
            "replicates": [],
        }
        model_payload["checkpoint_models"][view] = {
            "model_dir": str(model_dirs[checkpoint_seed]),
            "checkpoint_hash": checkpoint_hash,
            "replicates": {},
        }
        for initialization in INITIALIZATIONS:
            fit_seed = initialization_seed(checkpoint_seed, initialization)
            baseline, baseline_training = INSTRUMENT.train_one(
                weights,
                inputs,
                writes,
                genes,
                train_indices,
                validation_indices,
                coefficient=0.0,
                device=device,
                batch_size=args.batch,
                epochs=args.epochs,
                patience=args.patience,
                auxk_repair=True,
                capacity_repair=True,
                initialization_seed=fit_seed,
            )
            candidate, candidate_training = INSTRUMENT.train_one(
                weights,
                inputs,
                writes,
                genes,
                train_indices,
                validation_indices,
                coefficient=SELECTED_COEFFICIENT,
                device=device,
                batch_size=args.batch,
                epochs=args.epochs,
                patience=args.patience,
                auxk_repair=True,
                capacity_repair=True,
                initialization_seed=fit_seed,
            )
            baseline_validation = INSTRUMENT.evaluate_model(
                baseline,
                inputs,
                writes,
                validation_indices,
                batch_size=args.batch,
            )
            candidate_validation = INSTRUMENT.evaluate_model(
                candidate,
                inputs,
                writes,
                validation_indices,
                batch_size=args.batch,
            )
            candidate_train = INSTRUMENT.evaluate_model(
                candidate,
                inputs,
                writes,
                train_indices,
                batch_size=args.batch,
            )
            autograd = INSTRUMENT.autograd_control(
                candidate,
                inputs,
                validation_indices,
            )
            decoder = INSTRUMENT.decoder_invariants(candidate)
            gate = representation_gate(
                candidate_train,
                candidate_validation,
                baseline_validation,
                mechanical_pass=autograd["PASS"] and decoder["PASS"],
            )
            input_features, output_features, graph = fingerprints_and_graph(
                candidate,
                inputs,
                writes,
                validation_indices,
                batch_size=args.batch,
            )
            fitted[view].append(
                {
                    "input_features": input_features,
                    "output_features": output_features,
                    "input_decoder": (
                        candidate.input_sae.decoder().detach().cpu().numpy().T
                    ),
                    "output_decoder": (
                        candidate.output_sae.decoder().detach().cpu().numpy().T
                    ),
                    "graph": graph,
                }
            )
            replicate_records[view]["replicates"].append(
                {
                    "initialization": initialization,
                    "initialization_seed": fit_seed,
                    "baseline_training": baseline_training,
                    "candidate_training": candidate_training,
                    "baseline_validation": baseline_validation,
                    "candidate_train": candidate_train,
                    "candidate_validation": candidate_validation,
                    "autograd_control": autograd,
                    "decoder_invariants": decoder,
                    "representation_gate": gate,
                }
            )
            model_payload["checkpoint_models"][view]["replicates"][
                str(initialization)
            ] = {
                "baseline_state": compact_state(baseline),
                "candidate_state": compact_state(candidate),
                "mean_absolute_validation_graph": torch.as_tensor(graph),
            }
            del baseline, candidate
            if device.type == "mps":
                torch.mps.empty_cache()

    stability_rows = []
    for view in map(str, EXPECTED_SEEDS):
        for left, right in combinations(INITIALIZATIONS, 2):
            site_matches = {}
            for site in ("input", "output"):
                match = feature_match(
                    fitted[view][left][f"{site}_features"],
                    fitted[view][right][f"{site}_features"],
                    fitted[view][left][f"{site}_decoder"],
                    fitted[view][right][f"{site}_decoder"],
                )
                site_matches[site] = match
            graph = graph_alignment(
                fitted[view][left]["graph"],
                fitted[view][right]["graph"],
                site_matches["input"]["forward_map"],
                site_matches["output"]["forward_map"],
            )
            stability_rows.append(
                {
                    "view": view,
                    "initialization_pair": [left, right],
                    "sites": site_matches,
                    "graph_diagnostic": graph,
                    "PASS": all(
                        site_matches[site]["PASS"]
                        for site in ("input", "output")
                    ),
                }
            )

    cross_seed_rows = []
    for left_seed, right_seed in combinations(map(str, EXPECTED_SEEDS), 2):
        for site in ("input", "output"):
            activation_only = feature_match(
                fitted[left_seed][0][f"{site}_features"],
                fitted[right_seed][0][f"{site}_features"],
                fitted[left_seed][0][f"{site}_decoder"],
                fitted[right_seed][0][f"{site}_decoder"],
                activation_weight=1.0,
            )
            raw_combined = feature_match(
                fitted[left_seed][0][f"{site}_features"],
                fitted[right_seed][0][f"{site}_features"],
                fitted[left_seed][0][f"{site}_decoder"],
                fitted[right_seed][0][f"{site}_decoder"],
                activation_weight=0.5,
            )
            cross_seed_rows.append(
                {
                    "checkpoint_pair": [left_seed, right_seed],
                    "site": site,
                    "activation_only": activation_only,
                    "raw_coordinate_sensitivity": raw_combined,
                    "gating": False,
                }
            )

    all_fit_gates = all(
        replicate["representation_gate"]["PASS"]
        for view in replicate_records.values()
        for replicate in view["replicates"]
    )
    stability_pass = all(row["PASS"] for row in stability_rows)
    overall_pass = all_fit_gates and stability_pass
    status = "INSTRUMENT_PASS" if overall_pass else "INSTRUMENT_UNRESOLVED"
    result = {
        "study": "SpliceBERT GTEx layer-5 Jacobian SAE",
        "stage": "five-seed three-initialization representation replication",
        "status": status,
        "evidence_tier": "instrument",
        "protocol": str(protocol.relative_to(REPO)),
        "implementation_addendum": str(addendum.relative_to(REPO)),
        "instrument_artifact": str(instrument_path),
        "instrument_artifact_sha256": sha256_file(instrument_path),
        "activation_artifact": str(activation_path),
        "activation_artifact_sha256": sha256_file(activation_path),
        "checkpoint_hashes": checkpoint_hashes,
        "selected_architecture": {
            "input_features": 256,
            "output_features": 64,
            "active_features": 8,
            "jacobian_coefficient": SELECTED_COEFFICIENT,
            "auxk": True,
        },
        "replicates": replicate_records,
        "within_checkpoint_stability": {
            "rows": stability_rows,
            "thresholds": {
                "density_floor": DENSITY_FLOOR,
                "reciprocal_match_fraction": RECIPROCAL_FRACTION_FLOOR,
                "reciprocal_score_mean": RECIPROCAL_SCORE_FLOOR,
            },
            "PASS": stability_pass,
        },
        "cross_checkpoint_matching_diagnostic": cross_seed_rows,
        "instrument_gate": {
            "all_15_representation_fits_PASS": all_fit_gates,
            "all_30_site_stability_comparisons_PASS": stability_pass,
            "classifier_or_task_outcome_opened": False,
            "PASS": overall_pass,
        },
        "behavioral_preflight_authorized": overall_pass,
        "feature_association_opened": False,
        "licensed_claim": (
            "Five-seed checkpoint-specific J-SAE representation instruments "
            "replicate with within-checkpoint optimizer stability."
            if overall_pass
            else "No J-SAE feature semantics or downstream patching are authorized."
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
        "splicebert_gtex_finetune_shift:jacobian_sae_replication_v1",
        sys.argv,
        repo=REPO,
        status=status,
        config={
            "checkpoint_seeds": list(EXPECTED_SEEDS),
            "initializations": list(INITIALIZATIONS),
            "selected_coefficient": SELECTED_COEFFICIENT,
            "batch": args.batch,
            "epochs": args.epochs,
            "patience": args.patience,
            "device": str(device),
        },
        seeds={
            "dictionary_initializations": [
                initialization_seed(seed, initialization)
                for seed in EXPECTED_SEEDS
                for initialization in INITIALIZATIONS
            ]
        },
        notes=[
            "All J-SAE and matched zero-penalty fits were run sequentially.",
            "No classifier parameter, task margin, or feature example was opened.",
            "Cross-checkpoint raw-coordinate matching is diagnostic only.",
        ],
    )
    for path in (
        protocol,
        addendum,
        instrument_path,
        activation_path,
        Path(activation_report["arrays"]["path"]),
        *(
            item
            for model_dir in model_dirs.values()
            for item in (
                model_dir / "config.json",
                model_dir / "model.safetensors",
            )
        ),
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
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
