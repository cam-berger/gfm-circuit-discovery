#!/usr/bin/env python
"""Causally evaluate frozen per-seed and aligned final-MLP codes."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.adapters.splicebert import SpliceBertAdapter  # noqa: E402
from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.rigor import benjamini_hochberg  # noqa: E402
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gfmcd.splicing.context_joint_lowrank import (  # noqa: E402
    deterministic_random_basis,
    deterministic_signed_permutation,
    project_rows,
    projection_matrix_error,
    transform_row_basis,
)
from gfmcd.splicing.context_localization import (  # noqa: E402
    RelativeWindow,
    shuffle_relative_window,
)
from gfmcd.splicing.grouped_stats import (  # noqa: E402
    grouped_bootstrap,
    grouped_bootstrap_ratio,
    grouped_signflip_pvalue,
)
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.native_classifier import (  # noqa: E402
    ACCEPTOR_LABEL,
    DONOR_LABEL,
    NativeClassifierTask,
    load_annotation_gene_hash_cohort,
)
from gfmcd.splicing.native_mechanism import (  # noqa: E402
    cache_final_layer_inputs,
    capture_final_layer_run,
    donor_token_positions,
)


EXPECTED_SEEDS = tuple(range(2020, 2025))
EXPECTED_EXCLUSIONS = frozenset(
    {
        "native_mechanism_preflight_v1.json",
        "native_mechanism_discovery_v1.json",
        "native_context_preflight_v1.json",
        "native_context_discovery_v1.json",
        "native_context_head_census_discovery_v1.json",
        "native_context_head_census_confirmation_v1.json",
    }
)
N_EVALUATION = 200
N_BOOT = 5000
N_PERM = 9999
FINAL_LAYER = 5
NUM_HEADS = 16
HIDDEN_SIZE = 512
MIN_EFFECT = 0.05
MAX_Q = 0.05
RANKS = (1, 2, 4, 8, 16, 32, 64)
PRIMARY_RANKS = (4, 64)
NAMESPACE = "gtex-native-context-joint-lowrank-v1"
COHORT_TAG = "gtex-native-context-joint-lowrank-confirmation-v1"
FORBIDDEN_CHROMS = frozenset({"chr8", "chr9", "chr11", "chr16", "chr19"})


def parse_checkpoints(values: list[str]) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in checkpoints:
            raise ValueError(f"duplicate checkpoint seed {seed}")
        checkpoints[seed] = Path(path_text).expanduser().resolve()
    if tuple(sorted(checkpoints)) != EXPECTED_SEEDS:
        raise ValueError("the complete checkpoint family 2020--2024 is required")
    return checkpoints


def model_weight(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = model_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"model weights not found in {model_dir}")


def parse_window(name: str) -> RelativeWindow:
    start, end = name.split(":")
    return RelativeWindow(int(start), int(end))


def gene_from_example_id(example_id: str) -> str:
    return example_id.rsplit(":", 1)[1]


def raw_gene_ids(artifact: dict[str, Any]) -> set[str]:
    raw = artifact.get("raw", [])
    rows = (
        [row for values in raw.values() for row in values]
        if isinstance(raw, dict)
        else raw
    )
    genes = {
        str(row["gene_id"])
        for row in rows
        if isinstance(row, dict) and "gene_id" in row
    }
    genes.update(
        gene_from_example_id(str(value))
        for value in artifact.get("example_ids", [])
    )
    return genes


def target_rows(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[rows, positions.to(device=tensor.device)]


def native_margin(
    model: Any,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    *,
    label: int,
) -> torch.Tensor:
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    target = hidden[rows, positions.to(device=hidden.device)]
    logits = model.classifier(model.dropout(target)).float()
    return logits[:, label] - logits[:, 0]


def composed_margin(
    residual: np.ndarray,
    write: np.ndarray,
    parameters: dict[str, Any],
) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float64) + np.asarray(
        write, dtype=np.float64
    )
    mean = values.mean(axis=1, keepdims=True)
    variance = np.square(values - mean).mean(axis=1, keepdims=True)
    normalized = (values - mean) / np.sqrt(variance + parameters["eps"])
    hidden = normalized * parameters["ln_weight"] + parameters["ln_bias"]
    return hidden @ parameters["classifier_direction"] + parameters[
        "classifier_bias"
    ]


def intervention_effects(
    arrays: dict[str, np.ndarray],
    vector: np.ndarray,
    parameters: dict[str, Any],
) -> dict[str, np.ndarray]:
    noised = composed_margin(
        arrays["clean_residual"],
        arrays["clean_write"] - vector,
        parameters,
    )
    recovered = composed_margin(
        arrays["corrupt_residual"],
        arrays["corrupt_write"] + vector,
        parameters,
    )
    return {
        "damage": arrays["clean_margin"] - noised,
        "recovery": recovered - arrays["corrupt_margin"],
    }


def summarize(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    infer: bool = True,
) -> dict[str, Any]:
    result = grouped_bootstrap(values, groups, n_boot=N_BOOT, seed=seed)
    if infer:
        result["p_signflip_add_one"] = grouped_signflip_pvalue(
            values,
            groups,
            n_perm=N_PERM,
            seed=seed + 100_000,
        )
    return result


def apply_q(entries: list[dict[str, Any]], field: str) -> None:
    qvalues = benjamini_hochberg(
        [entry["summary"]["p_signflip_add_one"] for entry in entries]
    )
    for entry, qvalue in zip(entries, qvalues):
        entry["summary"][field] = float(qvalue)


def average_arrays(values: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(values, axis=0), axis=0)


def add_ratio(
    summary: dict[str, Any],
    values: np.ndarray,
    ceiling: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> None:
    summary["ratio_to_full_mlp"] = grouped_bootstrap_ratio(
        values,
        ceiling,
        groups,
        n_boot=N_BOOT,
        seed=seed,
    )


def transport_matrix(
    values: dict[tuple[int, int], dict[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        str(target): {
            str(source): {
                direction: float(np.mean(direction_values))
                for direction, direction_values in values[(source, target)].items()
            }
            for source in EXPECTED_SEEDS
            if source != target
        }
        for target in EXPECTED_SEEDS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--discovery-arrays", required=True)
    parser.add_argument("--prior-confirmation", required=True)
    parser.add_argument("--exclude-artifact", action="append", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=N_EVALUATION)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--device", default=None)
    parser.add_argument("--arrays-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != N_EVALUATION:
        raise ValueError(f"evaluation is frozen at n={N_EVALUATION}")
    if args.n_boot != N_BOOT or args.n_perm != N_PERM:
        raise ValueError("inference settings differ from the frozen protocol")
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CONTEXT_PER_SEED_ALIGNMENT_V1.md"
    )
    discovery_path = Path(args.discovery).expanduser().resolve()
    discovery_arrays_path = Path(args.discovery_arrays).expanduser().resolve()
    discovery = json.loads(discovery_path.read_text())
    if (
        discovery.get("status") != "COMPLETE"
        or not discovery.get("freeze", {}).get("evaluation_authorized")
        or discovery.get("freeze", {}).get("ranks") != list(RANKS)
        or discovery.get("cohort_namespace") != NAMESPACE
    ):
        raise RuntimeError("discovery artifact does not authorize evaluation")
    frozen = np.load(discovery_arrays_path, allow_pickle=False)
    prior_confirmation_path = (
        Path(args.prior_confirmation).expanduser().resolve()
    )
    prior_confirmation = json.loads(prior_confirmation_path.read_text())
    if (
        prior_confirmation.get("status") != "COMPLETE"
        or prior_confirmation.get("cohort_namespace") != NAMESPACE
    ):
        raise RuntimeError("GFS18 confirmation artifact is not the cohort source")
    exclusion_paths = [
        Path(value).expanduser().resolve() for value in args.exclude_artifact
    ]
    if {path.name for path in exclusion_paths} != EXPECTED_EXCLUSIONS:
        raise ValueError("the frozen six-artifact exclusion family is required")
    artifacts = {
        path.name: json.loads(path.read_text()) for path in exclusion_paths
    }
    prior_genes: set[str] = set()
    for artifact in artifacts.values():
        prior_genes.update(raw_gene_ids(artifact))
    discovery_genes = set(map(str, frozen["gene_ids"].tolist()))
    excluded_genes = prior_genes | discovery_genes
    checkpoints = parse_checkpoints(args.checkpoint)
    base_model = Path(args.base_model).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    arrays_output = guard_output_path(Path(args.arrays_out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)
    examples = load_annotation_gene_hash_cohort(
        args.gtf,
        args.fasta,
        base_split="train",
        analysis_partition="confirmation",
        namespace=NAMESPACE,
        discovery_percent=50,
        n=args.n,
        tag=COHORT_TAG,
        excluded_gene_ids=sorted(excluded_genes),
    )
    example_ids = [example.example_id for example in examples]
    prior_ids = [str(row["example_id"]) for row in prior_confirmation["raw"]]
    if example_ids != prior_ids:
        raise RuntimeError("reconstructed evaluation examples differ from GFS18")
    groups = np.asarray([example.gene_id for example in examples], dtype=object)
    if set(groups) & excluded_genes:
        raise RuntimeError("evaluation overlaps discovery or prior-exposed genes")
    normalized_chroms = {
        chrom if chrom.startswith("chr") else f"chr{chrom}"
        for chrom in (example.chrom for example in examples)
    }
    if normalized_chroms & FORBIDDEN_CHROMS:
        raise RuntimeError("evaluation opened a forbidden chromosome")

    census = artifacts["native_context_head_census_discovery_v1.json"]
    specs = {
        "donor": {
            "anchors": [int(example.donor_pos) for example in examples],
            "targets": [int(example.donor_pos) - 1 for example in examples],
            "label": DONOR_LABEL,
            "window": parse_window(census["regions"]["donor"]),
        },
        "acceptor": {
            "anchors": [int(example.next_acceptor_pos) for example in examples],
            "targets": [int(example.next_acceptor_pos) for example in examples],
            "label": ACCEPTOR_LABEL,
            "window": parse_window(census["regions"]["acceptor"]),
        },
    }
    clean_sequences = [example.sequence for example in examples]
    corrupt_sequences: dict[str, list[str]] = {}
    shuffle_audits: dict[str, list[dict[str, Any]]] = {}
    for endpoint, spec in specs.items():
        corrupt_sequences[endpoint] = []
        shuffle_audits[endpoint] = []
        for example, anchor in zip(examples, spec["anchors"]):
            shuffled, audit = shuffle_relative_window(
                example.sequence,
                anchor=anchor,
                window=spec["window"],
                key=(
                    example.example_id,
                    endpoint,
                    spec["window"].name,
                    "joint-lowrank-v1",
                ),
                draw=0,
            )
            if audit.hamming_fraction < 0.25:
                raise RuntimeError("evaluation shuffle failed dose floor")
            corrupt_sequences[endpoint].append(shuffled)
            shuffle_audits[endpoint].append(audit.to_dict())

    captured: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    parameters: dict[int, dict[str, dict[str, Any]]] = {}
    checkpoint_reports: dict[str, Any] = {}
    composition_errors: list[dict[str, Any]] = []
    npz: dict[str, np.ndarray] = {
        "example_ids": np.asarray(example_ids, dtype=str),
        "gene_ids": groups.astype(str),
    }
    for seed in EXPECTED_SEEDS:
        model, tokenizer, report = load_gtex_token_classifier(
            base_model,
            checkpoints[seed],
            device=device,
            attn_implementation="eager",
        )
        task = NativeClassifierTask(
            model,
            tokenizer,
            device=device,
            batch_size=args.batch,
        )
        adapter = SpliceBertAdapter(
            model,
            tokenizer,
            source=str(checkpoints[seed]),
            device=device,
        )
        if (
            report.seed != seed
            or adapter.num_layers != 6
            or adapter.num_heads != NUM_HEADS
            or adapter.hidden_size != HIDDEN_SIZE
        ):
            raise RuntimeError("checkpoint or architecture invariant failed")
        layer_norm = adapter.encoder_layers[FINAL_LAYER].output.LayerNorm
        captured[seed] = {}
        parameters[seed] = {}
        for endpoint_index, (endpoint, spec) in enumerate(specs.items(), start=1):
            buffers = {
                name: []
                for name in (
                    "clean_margin",
                    "corrupt_margin",
                    "clean_write",
                    "corrupt_write",
                    "clean_residual",
                    "corrupt_residual",
                )
            }
            for start in range(0, len(examples), args.batch):
                stop = min(start + args.batch, len(examples))
                positions = donor_token_positions(
                    spec["targets"][start:stop],
                    prefix_tokens=task.prefix_tokens,
                    device=device,
                )
                clean_run = capture_final_layer_run(
                    adapter,
                    cache_final_layer_inputs(
                        model,
                        task.encode(clean_sequences[start:stop]),
                        layer=FINAL_LAYER,
                    ),
                    layer=FINAL_LAYER,
                )
                corrupt_run = capture_final_layer_run(
                    adapter,
                    cache_final_layer_inputs(
                        model,
                        task.encode(corrupt_sequences[endpoint][start:stop]),
                        layer=FINAL_LAYER,
                    ),
                    layer=FINAL_LAYER,
                )
                clean_margin = native_margin(
                    model,
                    clean_run.hidden,
                    positions,
                    label=int(spec["label"]),
                )
                corrupt_margin = native_margin(
                    model,
                    corrupt_run.hidden,
                    positions,
                    label=int(spec["label"]),
                )
                buffers["clean_margin"].extend(clean_margin.cpu().tolist())
                buffers["corrupt_margin"].extend(corrupt_margin.cpu().tolist())
                for prefix, run in (
                    ("clean", clean_run),
                    ("corrupt", corrupt_run),
                ):
                    if run.mlp_residual is None:
                        raise RuntimeError("MLP residual capture is missing")
                    buffers[f"{prefix}_write"].extend(
                        target_rows(run.mlp_write, positions).cpu().float().tolist()
                    )
                    buffers[f"{prefix}_residual"].extend(
                        target_rows(run.mlp_residual, positions)
                        .cpu()
                        .float()
                        .tolist()
                    )
                print(
                    f"seed {seed} endpoint {endpoint_index}/2 {endpoint}: "
                    f"{stop}/{len(examples)}",
                    end="\r",
                    flush=True,
                )
            print()
            arrays = {
                name: np.asarray(
                    values,
                    dtype=np.float32
                    if name.endswith(("write", "residual"))
                    else np.float64,
                )
                for name, values in buffers.items()
            }
            arrays["mlp_delta"] = arrays["clean_write"] - arrays["corrupt_write"]
            arrays["behavior"] = arrays["clean_margin"] - arrays["corrupt_margin"]
            label = int(spec["label"])
            classifier_direction = (
                model.classifier.weight[label] - model.classifier.weight[0]
            ).detach().cpu().double().numpy()
            classifier_bias = float(
                (
                    model.classifier.bias[label] - model.classifier.bias[0]
                )
                .detach()
                .cpu()
            )
            params = {
                "eps": float(layer_norm.eps),
                "ln_weight": layer_norm.weight.detach().cpu().double().numpy(),
                "ln_bias": layer_norm.bias.detach().cpu().double().numpy(),
                "classifier_direction": classifier_direction,
                "classifier_bias": classifier_bias,
            }
            clean_composed = composed_margin(
                arrays["clean_residual"], arrays["clean_write"], params
            )
            corrupt_composed = composed_margin(
                arrays["corrupt_residual"], arrays["corrupt_write"], params
            )
            max_error = float(
                max(
                    np.max(np.abs(clean_composed - arrays["clean_margin"])),
                    np.max(np.abs(corrupt_composed - arrays["corrupt_margin"])),
                )
            )
            composition_errors.append(
                {"seed": seed, "endpoint": endpoint, "max_abs_error": max_error}
            )
            if max_error > 1e-5:
                raise RuntimeError("NumPy final-layer composition preflight failed")
            captured[seed][endpoint] = arrays
            parameters[seed][endpoint] = params
            for name, values in arrays.items():
                npz[f"{name}_{seed}_{endpoint}"] = values
        checkpoint_reports[str(seed)] = report.to_dict()
        del task, adapter, model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # Positive behavior and complete-MLP ceilings.
    behavior: dict[str, np.ndarray] = {}
    full_mlp: dict[str, dict[str, np.ndarray]] = {}
    behavior_summaries: dict[str, Any] = {}
    behavior_entries: list[dict[str, Any]] = []
    for endpoint_index, endpoint in enumerate(specs):
        behavior[endpoint] = average_arrays(
            [captured[seed][endpoint]["behavior"] for seed in EXPECTED_SEEDS]
        )
        summary = summarize(
            behavior[endpoint], groups, seed=820_000 + endpoint_index
        )
        summary["seed_estimates"] = {
            str(seed): float(captured[seed][endpoint]["behavior"].mean())
            for seed in EXPECTED_SEEDS
        }
        behavior_summaries[endpoint] = summary
        behavior_entries.append({"endpoint": endpoint, "summary": summary})
        full_mlp[endpoint] = {"damage": [], "recovery": []}
        for seed in EXPECTED_SEEDS:
            effects = intervention_effects(
                captured[seed][endpoint],
                captured[seed][endpoint]["mlp_delta"],
                parameters[seed][endpoint],
            )
            for direction in ("damage", "recovery"):
                full_mlp[endpoint][direction].append(effects[direction])
        full_mlp[endpoint] = {
            direction: average_arrays(values)
            for direction, values in full_mlp[endpoint].items()
        }
    apply_q(behavior_entries, "q_bh_2_evaluation_behavior")
    behavior_gate = all(
        entry["summary"]["estimate"] >= MIN_EFFECT
        and entry["summary"]["ci_low"] > 0
        and entry["summary"]["q_bh_2_evaluation_behavior"] <= MAX_Q
        and all(
            value > 0 for value in entry["summary"]["seed_estimates"].values()
        )
        for entry in behavior_entries
    )
    if not behavior_gate:
        raise RuntimeError("evaluation behavior gate failed")

    # Causal projections and vector transports.
    own_values: dict[str, dict[int, dict[str, np.ndarray]]] = {
        endpoint: {} for endpoint in specs
    }
    own_seed_estimates: dict[str, Any] = {endpoint: {} for endpoint in specs}
    cross_values: dict[
        str, dict[str, dict[int, dict[tuple[int, int], dict[str, np.ndarray]]]]
    ] = {
        method: {
            endpoint: {rank: {} for rank in RANKS} for endpoint in specs
        }
        for method in ("raw", "aligned")
    }
    random_values: dict[
        str, dict[int, dict[int, list[dict[str, np.ndarray]]]]
    ] = {endpoint: {rank: {} for rank in RANKS} for endpoint in specs}
    vector_values: dict[
        str, dict[str, dict[tuple[int, int], dict[str, np.ndarray]]]
    ] = {
        method: {endpoint: {} for endpoint in specs}
        for method in ("raw", "aligned")
    }
    vector_controls: dict[
        str, dict[tuple[int, int], list[dict[str, np.ndarray]]]
    ] = {endpoint: {} for endpoint in specs}

    for endpoint in specs:
        for target in EXPECTED_SEEDS:
            target_arrays = captured[target][endpoint]
            params = parameters[target][endpoint]
            target_basis = np.asarray(
                frozen[f"basis_{target}_{endpoint}"], dtype=np.float64
            )
            own_seed_estimates[endpoint][str(target)] = {}
            for rank in RANKS:
                vector = project_rows(
                    target_arrays["mlp_delta"], target_basis[:rank]
                )
                effects = intervention_effects(target_arrays, vector, params)
                own_seed_estimates[endpoint][str(target)][str(rank)] = {
                    direction: float(values.mean())
                    for direction, values in effects.items()
                }
                own_values[endpoint].setdefault(
                    rank, {"damage": [], "recovery": []}
                )
                for direction in ("damage", "recovery"):
                    own_values[endpoint][rank][direction].append(
                        effects[direction]
                    )
                random_values[endpoint][rank][target] = []
                for draw in range(3):
                    random_basis = deterministic_random_basis(
                        HIDDEN_SIZE,
                        rank,
                        tag=f"per-seed-alignment:{endpoint}:{target}:{rank}",
                        draw=draw,
                    )
                    random_vector = project_rows(
                        target_arrays["mlp_delta"], random_basis
                    )
                    random_values[endpoint][rank][target].append(
                        intervention_effects(
                            target_arrays, random_vector, params
                        )
                    )
            for source in EXPECTED_SEEDS:
                if source == target:
                    continue
                source_basis = np.asarray(
                    frozen[f"basis_{source}_{endpoint}"], dtype=np.float64
                )
                rotation = np.asarray(
                    frozen[f"rotation_{source}_{target}_{endpoint}"],
                    dtype=np.float64,
                )
                aligned_basis = transform_row_basis(source_basis, rotation)
                if projection_matrix_error(aligned_basis) > 2e-6:
                    raise RuntimeError("serialized aligned basis is not orthonormal")
                for rank in RANKS:
                    for method, basis in (
                        ("raw", source_basis[:rank]),
                        ("aligned", aligned_basis[:rank]),
                    ):
                        vector = project_rows(target_arrays["mlp_delta"], basis)
                        cross_values[method][endpoint][rank][(source, target)] = (
                            intervention_effects(target_arrays, vector, params)
                        )
                source_delta = captured[source][endpoint]["mlp_delta"]
                for method, vector in (
                    ("raw", source_delta),
                    ("aligned", source_delta @ rotation),
                ):
                    vector_values[method][endpoint][(source, target)] = (
                        intervention_effects(target_arrays, vector, params)
                    )
                vector_controls[endpoint][(source, target)] = [
                    intervention_effects(
                        target_arrays,
                        deterministic_signed_permutation(
                            source_delta,
                            tag=(
                                f"per-seed-alignment-vector:{endpoint}:"
                                f"{source}:{target}"
                            ),
                            draw=draw,
                        ),
                        params,
                    )
                    for draw in range(3)
                ]

    own_entries: list[dict[str, Any]] = []
    own_results: dict[str, Any] = {endpoint: {} for endpoint in specs}
    compactness: dict[str, Any] = {}
    for endpoint_index, endpoint in enumerate(specs):
        for rank_index, rank in enumerate(RANKS):
            own_results[endpoint][str(rank)] = {}
            for direction_index, direction in enumerate(("damage", "recovery")):
                values = average_arrays(own_values[endpoint][rank][direction])
                own_values[endpoint][rank][direction] = values
                summary = summarize(
                    values,
                    groups,
                    seed=830_000
                    + 10_000 * endpoint_index
                    + 100 * rank_index
                    + direction_index,
                )
                add_ratio(
                    summary,
                    values,
                    full_mlp[endpoint][direction],
                    groups,
                    seed=831_000
                    + 10_000 * endpoint_index
                    + 100 * rank_index
                    + direction_index,
                )
                own_results[endpoint][str(rank)][direction] = summary
                own_entries.append(
                    {
                        "endpoint": endpoint,
                        "rank": rank,
                        "direction": direction,
                        "summary": summary,
                    }
                )
    apply_q(own_entries, "q_bh_28_own_basis_effects")
    for endpoint in specs:
        passing = [
            rank
            for rank in RANKS
            if rank <= 16
            and all(
                own_results[endpoint][str(rank)][direction][
                    "ratio_to_full_mlp"
                ]["ci_low"]
                >= 0.50
                for direction in ("damage", "recovery")
            )
        ]
        through64 = [
            rank
            for rank in RANKS
            if all(
                own_results[endpoint][str(rank)][direction][
                    "ratio_to_full_mlp"
                ]["ci_low"]
                >= 0.50
                for direction in ("damage", "recovery")
            )
        ]
        compactness[endpoint] = {
            "minimal_rank_at_most_16_reaching_50pct": (
                min(passing) if passing else None
            ),
            "minimal_tested_rank_through_64_reaching_50pct": (
                min(through64) if through64 else None
            ),
            "compact_within_checkpoint": bool(passing),
        }

    cross_results: dict[str, Any] = {
        method: {endpoint: {} for endpoint in specs}
        for method in ("raw", "aligned")
    }
    cross_aggregate: dict[
        str, dict[str, dict[int, dict[str, np.ndarray]]]
    ] = {
        method: {endpoint: {} for endpoint in specs}
        for method in ("raw", "aligned")
    }
    random_aggregate: dict[str, dict[int, dict[str, np.ndarray]]] = {
        endpoint: {} for endpoint in specs
    }
    basis_effect_entries: list[dict[str, Any]] = []
    basis_control_entries: list[dict[str, Any]] = []
    aligned_raw_entries: list[dict[str, Any]] = []
    for endpoint_index, endpoint in enumerate(specs):
        for rank_index, rank in enumerate(RANKS):
            random_aggregate[endpoint][rank] = {}
            for direction in ("damage", "recovery"):
                random_aggregate[endpoint][rank][direction] = average_arrays(
                    [
                        random_values[endpoint][rank][target][draw][direction]
                        for target in EXPECTED_SEEDS
                        for draw in range(3)
                    ]
                )
            for method_index, method in enumerate(("raw", "aligned")):
                cross_results[method][endpoint][str(rank)] = {
                    "matrix": transport_matrix(
                        cross_values[method][endpoint][rank]
                    )
                }
                cross_aggregate[method][endpoint][rank] = {}
                for direction_index, direction in enumerate(
                    ("damage", "recovery")
                ):
                    values = average_arrays(
                        [
                            cross_values[method][endpoint][rank][
                                (source, target)
                            ][direction]
                            for target in EXPECTED_SEEDS
                            for source in EXPECTED_SEEDS
                            if source != target
                        ]
                    )
                    cross_aggregate[method][endpoint][rank][direction] = values
                    summary = summarize(
                        values,
                        groups,
                        seed=840_000
                        + 10_000 * endpoint_index
                        + 1_000 * rank_index
                        + 100 * method_index
                        + direction_index,
                        infer=rank in PRIMARY_RANKS,
                    )
                    add_ratio(
                        summary,
                        values,
                        full_mlp[endpoint][direction],
                        groups,
                        seed=841_000
                        + 10_000 * endpoint_index
                        + 1_000 * rank_index
                        + 100 * method_index
                        + direction_index,
                    )
                    control_values = random_aggregate[endpoint][rank][direction]
                    contrast_values = values - control_values
                    contrast = summarize(
                        contrast_values,
                        groups,
                        seed=842_000
                        + 10_000 * endpoint_index
                        + 1_000 * rank_index
                        + 100 * method_index
                        + direction_index,
                        infer=rank in PRIMARY_RANKS,
                    )
                    contrast["candidate_estimate"] = float(values.mean())
                    contrast["random_estimate"] = float(control_values.mean())
                    summary["vs_random"] = contrast
                    cross_results[method][endpoint][str(rank)][direction] = (
                        summary
                    )
                    if rank in PRIMARY_RANKS:
                        basis_effect_entries.append(
                            {
                                "method": method,
                                "endpoint": endpoint,
                                "rank": rank,
                                "direction": direction,
                                "summary": summary,
                            }
                        )
                        basis_control_entries.append(
                            {
                                "method": method,
                                "endpoint": endpoint,
                                "rank": rank,
                                "direction": direction,
                                "summary": contrast,
                            }
                        )
            if rank in PRIMARY_RANKS:
                for direction_index, direction in enumerate(
                    ("damage", "recovery")
                ):
                    contrast_values = (
                        cross_aggregate["aligned"][endpoint][rank][direction]
                        - cross_aggregate["raw"][endpoint][rank][direction]
                    )
                    contrast = summarize(
                        contrast_values,
                        groups,
                        seed=843_000
                        + 10_000 * endpoint_index
                        + 1_000 * rank_index
                        + direction_index,
                    )
                    aligned_raw_entries.append(
                        {
                            "endpoint": endpoint,
                            "rank": rank,
                            "direction": direction,
                            "summary": contrast,
                        }
                    )
    apply_q(basis_effect_entries, "q_bh_16_cross_basis_effects")
    apply_q(
        basis_control_entries, "q_bh_16_cross_basis_random_contrasts"
    )
    apply_q(aligned_raw_entries, "q_bh_8_aligned_minus_raw_basis")

    vector_results: dict[str, Any] = {
        method: {endpoint: {} for endpoint in specs}
        for method in ("raw", "aligned")
    }
    vector_aggregate: dict[str, dict[str, dict[str, np.ndarray]]] = {
        method: {endpoint: {} for endpoint in specs}
        for method in ("raw", "aligned")
    }
    vector_control_aggregate: dict[str, dict[str, np.ndarray]] = {
        endpoint: {} for endpoint in specs
    }
    vector_effect_entries: list[dict[str, Any]] = []
    vector_control_entries: list[dict[str, Any]] = []
    vector_aligned_raw_entries: list[dict[str, Any]] = []
    for endpoint_index, endpoint in enumerate(specs):
        for direction in ("damage", "recovery"):
            vector_control_aggregate[endpoint][direction] = average_arrays(
                [
                    vector_controls[endpoint][(source, target)][draw][direction]
                    for target in EXPECTED_SEEDS
                    for source in EXPECTED_SEEDS
                    if source != target
                    for draw in range(3)
                ]
            )
        for method_index, method in enumerate(("raw", "aligned")):
            vector_results[method][endpoint]["matrix"] = transport_matrix(
                vector_values[method][endpoint]
            )
            for direction_index, direction in enumerate(
                ("damage", "recovery")
            ):
                values = average_arrays(
                    [
                        vector_values[method][endpoint][(source, target)][
                            direction
                        ]
                        for target in EXPECTED_SEEDS
                        for source in EXPECTED_SEEDS
                        if source != target
                    ]
                )
                vector_aggregate[method][endpoint][direction] = values
                summary = summarize(
                    values,
                    groups,
                    seed=850_000
                    + 10_000 * endpoint_index
                    + 100 * method_index
                    + direction_index,
                )
                add_ratio(
                    summary,
                    values,
                    full_mlp[endpoint][direction],
                    groups,
                    seed=851_000
                    + 10_000 * endpoint_index
                    + 100 * method_index
                    + direction_index,
                )
                control_values = vector_control_aggregate[endpoint][direction]
                contrast = summarize(
                    values - control_values,
                    groups,
                    seed=852_000
                    + 10_000 * endpoint_index
                    + 100 * method_index
                    + direction_index,
                )
                contrast["candidate_estimate"] = float(values.mean())
                contrast["signed_permutation_estimate"] = float(
                    control_values.mean()
                )
                summary["vs_signed_permutation"] = contrast
                vector_results[method][endpoint][direction] = summary
                vector_effect_entries.append(
                    {
                        "method": method,
                        "endpoint": endpoint,
                        "direction": direction,
                        "summary": summary,
                    }
                )
                vector_control_entries.append(
                    {
                        "method": method,
                        "endpoint": endpoint,
                        "direction": direction,
                        "summary": contrast,
                    }
                )
        for direction_index, direction in enumerate(("damage", "recovery")):
            contrast = summarize(
                vector_aggregate["aligned"][endpoint][direction]
                - vector_aggregate["raw"][endpoint][direction],
                groups,
                seed=853_000 + 100 * endpoint_index + direction_index,
            )
            vector_aligned_raw_entries.append(
                {
                    "endpoint": endpoint,
                    "direction": direction,
                    "summary": contrast,
                }
            )
    apply_q(vector_effect_entries, "q_bh_8_vector_effects")
    apply_q(
        vector_control_entries, "q_bh_8_vector_signed_permutation_contrasts"
    )
    apply_q(vector_aligned_raw_entries, "q_bh_4_vector_aligned_minus_raw")

    decision: dict[str, Any] = {}
    for endpoint in specs:
        compact_rank = compactness[endpoint][
            "minimal_rank_at_most_16_reaching_50pct"
        ]
        decision_rank = next(
            (
                rank
                for rank in PRIMARY_RANKS
                if rank <= 16
                and all(
                    own_results[endpoint][str(rank)][direction][
                        "ratio_to_full_mlp"
                    ]["ci_low"]
                    >= 0.50
                    for direction in ("damage", "recovery")
                )
            ),
            None,
        )
        basis_pass = False
        if decision_rank is not None:
            basis_pass = all(
                cross_results["aligned"][endpoint][str(decision_rank)][direction][
                    "ratio_to_full_mlp"
                ]["ci_low"]
                >= 0.50
                and cross_results["aligned"][endpoint][str(decision_rank)][
                    direction
                ]["q_bh_16_cross_basis_effects"]
                <= MAX_Q
                and cross_results["aligned"][endpoint][str(decision_rank)][
                    direction
                ]["vs_random"]["ci_low"]
                > 0
                and cross_results["aligned"][endpoint][str(decision_rank)][
                    direction
                ]["vs_random"][
                    "q_bh_16_cross_basis_random_contrasts"
                ]
                <= MAX_Q
                for direction in ("damage", "recovery")
            ) and all(
                entry["summary"]["ci_low"] > 0
                and entry["summary"]["q_bh_8_aligned_minus_raw_basis"] <= MAX_Q
                for entry in aligned_raw_entries
                if entry["endpoint"] == endpoint
                and entry["rank"] == decision_rank
            )
        vector_pass = all(
            vector_results["aligned"][endpoint][direction][
                "ratio_to_full_mlp"
            ]["ci_low"]
            >= 0.50
            and vector_results["aligned"][endpoint][direction][
                "q_bh_8_vector_effects"
            ]
            <= MAX_Q
            and vector_results["aligned"][endpoint][direction][
                "vs_signed_permutation"
            ]["ci_low"]
            > 0
            and vector_results["aligned"][endpoint][direction][
                "vs_signed_permutation"
            ]["q_bh_8_vector_signed_permutation_contrasts"]
            <= MAX_Q
            for direction in ("damage", "recovery")
        ) and all(
            entry["summary"]["ci_low"] > 0
            and entry["summary"]["q_bh_4_vector_aligned_minus_raw"] <= MAX_Q
            for entry in vector_aligned_raw_entries
            if entry["endpoint"] == endpoint
        )
        decision[endpoint] = {
            **compactness[endpoint],
            "cross_seed_decision_rank": decision_rank,
            "rotated_shared_lowrank_code": basis_pass,
            "aligned_instance_vector_transport": vector_pass,
        }

    for endpoint in specs:
        for direction in ("damage", "recovery"):
            npz[f"full_mlp_{endpoint}_{direction}"] = full_mlp[endpoint][
                direction
            ]
            for rank in RANKS:
                npz[f"own_{endpoint}_{rank}_{direction}"] = own_values[endpoint][
                    rank
                ][direction]
                for method in ("raw", "aligned"):
                    npz[
                        f"cross_{method}_{endpoint}_{rank}_{direction}"
                    ] = cross_aggregate[method][endpoint][rank][direction]
            for method in ("raw", "aligned"):
                npz[f"vector_{method}_{endpoint}_{direction}"] = vector_aggregate[
                    method
                ][endpoint][direction]
    arrays_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_output, **npz)
    full_mlp_results = {
        endpoint: {
            direction: {
                **summarize(
                    full_mlp[endpoint][direction],
                    groups,
                    seed=860_000
                    + 100 * endpoint_index
                    + direction_index,
                ),
                "ratio_to_behavior": grouped_bootstrap_ratio(
                    full_mlp[endpoint][direction],
                    behavior[endpoint],
                    groups,
                    n_boot=N_BOOT,
                    seed=861_000
                    + 100 * endpoint_index
                    + direction_index,
                ),
            }
            for direction_index, direction in enumerate(("damage", "recovery"))
        }
        for endpoint_index, endpoint in enumerate(specs)
    }
    result = {
        "study": "native GTEx per-seed final-MLP alignment",
        "stage": "gene-disjoint causal evaluation on reused GFS18 partition",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "evidence_tier": "exploratory causal alignment",
        "device": device,
        "split": "reused GFS18 confirmation genes",
        "cohort_namespace": NAMESPACE,
        "cohort_tag": COHORT_TAG,
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "example_ids": example_ids,
        "discovery_gene_overlap": len(set(groups) & discovery_genes),
        "forbidden_chromosomes": sorted(FORBIDDEN_CHROMS),
        "checkpoint_reports": checkpoint_reports,
        "regions": {
            endpoint: spec["window"].name for endpoint, spec in specs.items()
        },
        "instrument_gate": {
            "cohort_identity": True,
            "discovery_gene_disjoint": True,
            "behavior_gate": behavior_gate,
            "numpy_composition_max_abs_error": max(
                row["max_abs_error"] for row in composition_errors
            ),
            "numpy_composition_at_most_1e_5": True,
        },
        "composition_checks": composition_errors,
        "behavior": behavior_summaries,
        "full_mlp_ceiling": full_mlp_results,
        "own_checkpoint_subspaces": {
            "summaries": own_results,
            "seed_estimates": own_seed_estimates,
        },
        "cross_seed_basis_transport": cross_results,
        "aligned_minus_raw_basis": aligned_raw_entries,
        "instance_vector_transport": vector_results,
        "aligned_minus_raw_vectors": vector_aligned_raw_entries,
        "decision": decision,
        "multiplicity_families": {
            "own_basis_effects": len(own_entries),
            "primary_cross_basis_effects": len(basis_effect_entries),
            "primary_cross_basis_random_contrasts": len(
                basis_control_entries
            ),
            "aligned_minus_raw_basis": len(aligned_raw_entries),
            "vector_effects": len(vector_effect_entries),
            "vector_signed_permutation_contrasts": len(vector_control_entries),
            "aligned_minus_raw_vectors": len(vector_aligned_raw_entries),
        },
        "arrays": str(arrays_output.relative_to(REPO)),
        "raw": [
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "chrom": example.chrom,
                "shuffle_audits": {
                    endpoint: shuffle_audits[endpoint][index]
                    for endpoint in specs
                },
            }
            for index, example in enumerate(examples)
        ],
        "strongest_claim": (
            "Exploratory only: decision fields state whether seed-specific compact "
            "codes and outcome-blind aligned transport meet the frozen causal gates."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_per_seed_alignment_evaluation_v1",
        sys.argv,
        repo=REPO,
        status="EXPLORATORY",
        config={
            "split": result["split"],
            "n": args.n,
            "batch": args.batch,
            "device": device,
            "cohort_namespace": NAMESPACE,
            "ranks": list(RANKS),
            "primary_ranks": list(PRIMARY_RANKS),
            "n_boot": N_BOOT,
            "n_perm": N_PERM,
        },
        seeds={"checkpoint_family": list(EXPECTED_SEEDS)},
        notes=[
            "All bases and maps were frozen in the immutable discovery artifact.",
            "Evaluation reuses known GFS18 genes, so the causal result remains exploratory.",
            "Raw, aligned, random-subspace, and norm-preserving signed-permutation controls are separate.",
        ],
    )
    for path in (
        protocol,
        discovery_path,
        discovery_arrays_path,
        prior_confirmation_path,
        *exclusion_paths,
        model_weight(base_model),
        base_model / "config.json",
        base_model / "tokenizer.json",
        *checkpoints.values(),
        Path(args.gtf),
        Path(args.fasta),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.extend([file_record(output), file_record(arrays_output)])
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "instrument_gate": result["instrument_gate"],
                "decision": decision,
                "vector_estimates": {
                    method: {
                        endpoint: {
                            direction: vector_results[method][endpoint][direction][
                                "estimate"
                            ]
                            for direction in ("damage", "recovery")
                        }
                        for endpoint in specs
                    }
                    for method in ("raw", "aligned")
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"COMPLETE: saved -> {output}")


if __name__ == "__main__":
    main()
