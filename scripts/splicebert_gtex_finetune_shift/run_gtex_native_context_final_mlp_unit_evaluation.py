#!/usr/bin/env python
"""Evaluate frozen native-unit groups and exact L5-head→unit paths."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.adapters.splicebert import SpliceBertAdapter  # noqa: E402
from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.circuits.hf_bert import patch_heads  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.rigor import benjamini_hochberg  # noqa: E402
from gfmcd.splicing.circuit import pick_device  # noqa: E402
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
from gfmcd.splicing.mlp_units import (  # noqa: E402
    composed_classifier_margin,
    unit_group_write_delta,
)
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
    position_mask,
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
EXPECTED_HEADS = {
    "donor": (14, 8, 12),
    "acceptor": (11, 14, 8),
}
N_EVALUATION = 200
N_BOOT = 5000
N_PERM = 9999
FINAL_LAYER = 5
NUM_HEADS = 16
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 2048
CONTROLLED_SIZES = (1, 4, 16, 64, 256, 1024)
ALL_SIZES = (*CONTROLLED_SIZES, INTERMEDIATE_SIZE)
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
        raise ValueError("complete checkpoint family 2020--2024 required")
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


def intervention_effects(
    arrays: dict[str, np.ndarray],
    noising_vector: np.ndarray,
    recovery_vector: np.ndarray,
    parameters: dict[str, Any],
) -> dict[str, np.ndarray]:
    noised = composed_classifier_margin(
        arrays["clean_residual"],
        arrays["clean_write"] - noising_vector,
        **parameters,
    )
    recovered = composed_classifier_margin(
        arrays["corrupt_residual"],
        arrays["corrupt_write"] + recovery_vector,
        **parameters,
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


def head_name(heads: tuple[int, ...]) -> str:
    return "+".join(f"L5H{head}" for head in heads)


def prior_indirect_values(
    artifact: dict[str, Any],
    *,
    endpoint: str,
    seed: int,
    direction: str,
) -> np.ndarray:
    return np.asarray(
        [
            row["tasks"][endpoint]["seeds"][str(seed)]["path_candidate"][
                "indirect"
            ][direction]
            for row in artifact["raw"]
        ],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", required=True)
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
        raise ValueError("inference settings differ from frozen protocol")
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CONTEXT_FINAL_MLP_UNIT_CENSUS_V1.md"
    )
    discovery_path = Path(args.discovery).expanduser().resolve()
    discovery = json.loads(discovery_path.read_text())
    if (
        discovery.get("status") != "COMPLETE"
        or not discovery.get("freeze", {}).get("evaluation_authorized")
        or tuple(discovery.get("freeze", {}).get("group_sizes", []))
        != CONTROLLED_SIZES
        or discovery.get("freeze", {}).get("ceiling_size") != INTERMEDIATE_SIZE
    ):
        raise RuntimeError("unit discovery does not authorize evaluation")
    prior_path = Path(args.prior_confirmation).expanduser().resolve()
    prior = json.loads(prior_path.read_text())
    if (
        prior.get("status") != "COMPLETE"
        or prior.get("cohort_namespace") != NAMESPACE
    ):
        raise RuntimeError("GFS18 confirmation artifact is not complete")
    exclusion_paths = [
        Path(value).expanduser().resolve() for value in args.exclude_artifact
    ]
    if {path.name for path in exclusion_paths} != EXPECTED_EXCLUSIONS:
        raise ValueError("frozen six-artifact exclusion family required")
    artifacts = {
        path.name: json.loads(path.read_text()) for path in exclusion_paths
    }
    prior_genes: set[str] = set()
    for artifact in artifacts.values():
        prior_genes.update(raw_gene_ids(artifact))
    discovery_genes = raw_gene_ids(discovery)
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
    if example_ids != [str(row["example_id"]) for row in prior["raw"]]:
        raise RuntimeError("evaluation cohort does not exactly replay GFS18")
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

    # Per-seed arrays. Group interventions are inexpensive after these captures.
    seed_values: dict[int, dict[str, Any]] = {}
    checkpoint_reports: dict[str, Any] = {}
    instrument_rows: list[dict[str, Any]] = []
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
        layer = adapter.encoder_layers[FINAL_LAYER]
        output_dense = layer.output.dense
        if (
            report.seed != seed
            or adapter.num_layers != 6
            or adapter.num_heads != NUM_HEADS
            or adapter.hidden_size != HIDDEN_SIZE
            or output_dense.in_features != INTERMEDIATE_SIZE
        ):
            raise RuntimeError("checkpoint/architecture invariant failed")
        output_weight = output_dense.weight.detach().cpu().double().numpy()
        seed_values[seed] = {}
        for endpoint_index, (endpoint, spec) in enumerate(specs.items(), start=1):
            selected_sets = {
                head_name((head,)): (head,) for head in EXPECTED_HEADS[endpoint]
            }
            selected_sets[head_name(EXPECTED_HEADS[endpoint])] = EXPECTED_HEADS[
                endpoint
            ]
            buffers: dict[str, Any] = {
                name: []
                for name in (
                    "clean_margin",
                    "corrupt_margin",
                    "clean_write",
                    "corrupt_write",
                    "clean_residual",
                    "corrupt_residual",
                    "clean_intermediate",
                    "corrupt_intermediate",
                )
            }
            patched_buffers = {
                name: {
                    "clean_margin": [],
                    "corrupt_margin": [],
                    "clean_intermediate": [],
                    "corrupt_intermediate": [],
                }
                for name in selected_sets
            }
            for start in range(0, len(examples), args.batch):
                stop = min(start + args.batch, len(examples))
                clean_encoded = task.encode(clean_sequences[start:stop])
                corrupt_encoded = task.encode(
                    corrupt_sequences[endpoint][start:stop]
                )
                positions = donor_token_positions(
                    spec["targets"][start:stop],
                    prefix_tokens=task.prefix_tokens,
                    device=device,
                )
                token_mask = position_mask(
                    positions, int(clean_encoded["input_ids"].shape[1])
                )
                clean_inputs = cache_final_layer_inputs(
                    model, clean_encoded, layer=FINAL_LAYER
                )
                corrupt_inputs = cache_final_layer_inputs(
                    model, corrupt_encoded, layer=FINAL_LAYER
                )
                clean_run = capture_final_layer_run(
                    adapter, clean_inputs, layer=FINAL_LAYER
                )
                corrupt_run = capture_final_layer_run(
                    adapter, corrupt_inputs, layer=FINAL_LAYER
                )
                if (
                    clean_run.mlp_intermediate is None
                    or corrupt_run.mlp_intermediate is None
                    or clean_run.mlp_residual is None
                    or corrupt_run.mlp_residual is None
                ):
                    raise RuntimeError("baseline final MLP capture missing")
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
                for prefix, run in (("clean", clean_run), ("corrupt", corrupt_run)):
                    buffers[f"{prefix}_write"].extend(
                        target_rows(run.mlp_write, positions).cpu().float().tolist()
                    )
                    buffers[f"{prefix}_residual"].extend(
                        target_rows(run.mlp_residual, positions)
                        .cpu()
                        .float()
                        .tolist()
                    )
                    buffers[f"{prefix}_intermediate"].extend(
                        target_rows(run.mlp_intermediate, positions)
                        .cpu()
                        .float()
                        .tolist()
                    )
                for name, heads in selected_sets.items():
                    with patch_heads(
                        adapter,
                        {FINAL_LAYER: corrupt_run.heads},
                        [(FINAL_LAYER, head) for head in heads],
                        token_mask=token_mask,
                    ):
                        patched_clean = capture_final_layer_run(
                            adapter, clean_inputs, layer=FINAL_LAYER
                        )
                    with patch_heads(
                        adapter,
                        {FINAL_LAYER: clean_run.heads},
                        [(FINAL_LAYER, head) for head in heads],
                        token_mask=token_mask,
                    ):
                        patched_corrupt = capture_final_layer_run(
                            adapter, corrupt_inputs, layer=FINAL_LAYER
                        )
                    if (
                        patched_clean.mlp_intermediate is None
                        or patched_corrupt.mlp_intermediate is None
                    ):
                        raise RuntimeError("patched intermediate capture missing")
                    patched_buffers[name]["clean_margin"].extend(
                        native_margin(
                            model,
                            patched_clean.hidden,
                            positions,
                            label=int(spec["label"]),
                        )
                        .cpu()
                        .tolist()
                    )
                    patched_buffers[name]["corrupt_margin"].extend(
                        native_margin(
                            model,
                            patched_corrupt.hidden,
                            positions,
                            label=int(spec["label"]),
                        )
                        .cpu()
                        .tolist()
                    )
                    patched_buffers[name]["clean_intermediate"].extend(
                        target_rows(patched_clean.mlp_intermediate, positions)
                        .cpu()
                        .float()
                        .tolist()
                    )
                    patched_buffers[name]["corrupt_intermediate"].extend(
                        target_rows(patched_corrupt.mlp_intermediate, positions)
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
                    dtype=np.float64
                    if name.endswith("margin")
                    else np.float32,
                )
                for name, values in buffers.items()
            }
            arrays["behavior"] = arrays["clean_margin"] - arrays["corrupt_margin"]
            patched_arrays = {
                name: {
                    field: np.asarray(
                        values,
                        dtype=np.float64
                        if field.endswith("margin")
                        else np.float32,
                    )
                    for field, values in fields.items()
                }
                for name, fields in patched_buffers.items()
            }
            label = int(spec["label"])
            parameters = {
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
            composition_error = float(
                max(
                    np.max(
                        np.abs(
                            composed_classifier_margin(
                                arrays["clean_residual"],
                                arrays["clean_write"],
                                **parameters,
                            )
                            - arrays["clean_margin"]
                        )
                    ),
                    np.max(
                        np.abs(
                            composed_classifier_margin(
                                arrays["corrupt_residual"],
                                arrays["corrupt_write"],
                                **parameters,
                            )
                            - arrays["corrupt_margin"]
                        )
                    ),
                )
            )
            if composition_error > 1e-5:
                raise RuntimeError("evaluation composition preflight failed")
            context_delta = (
                arrays["clean_intermediate"]
                - arrays["corrupt_intermediate"]
            )
            context_results: dict[int, Any] = {}
            context_controls: dict[int, Any] = {}
            frozen_endpoint = discovery["rankings"][str(seed)][endpoint]
            for size in ALL_SIZES:
                units = (
                    np.arange(INTERMEDIATE_SIZE)
                    if size == INTERMEDIATE_SIZE
                    else np.asarray(
                        frozen_endpoint["context"]["groups"][str(size)][
                            "candidates"
                        ],
                        dtype=np.int64,
                    )
                )
                vector = unit_group_write_delta(
                    context_delta, output_weight, units
                )
                context_results[size] = intervention_effects(
                    arrays, vector, vector, parameters
                )
                if size in CONTROLLED_SIZES:
                    context_controls[size] = [
                        intervention_effects(
                            arrays,
                            unit_group_write_delta(
                                context_delta, output_weight, control
                            ),
                            unit_group_write_delta(
                                context_delta, output_weight, control
                            ),
                            parameters,
                        )
                        for control in frozen_endpoint["context"]["groups"][
                            str(size)
                        ]["controls"]
                    ]

            head_results: dict[str, Any] = {}
            full_name = head_name(EXPECTED_HEADS[endpoint])
            for name, values in patched_arrays.items():
                noising_delta = (
                    arrays["clean_intermediate"]
                    - values["clean_intermediate"]
                )
                recovery_delta = (
                    values["corrupt_intermediate"]
                    - arrays["corrupt_intermediate"]
                )
                result_entry: dict[str, Any] = {
                    "total": {
                        "damage": (
                            arrays["clean_margin"] - values["clean_margin"]
                        ),
                        "recovery": (
                            values["corrupt_margin"] - arrays["corrupt_margin"]
                        ),
                    },
                    "unit_paths": {},
                }
                for size in ALL_SIZES:
                    units = (
                        np.arange(INTERMEDIATE_SIZE)
                        if size == INTERMEDIATE_SIZE
                        else np.asarray(
                            frozen_endpoint["head_induced"]["groups"][
                                str(size)
                            ]["candidates"],
                            dtype=np.int64,
                        )
                    )
                    result_entry["unit_paths"][size] = intervention_effects(
                        arrays,
                        unit_group_write_delta(
                            noising_delta, output_weight, units
                        ),
                        unit_group_write_delta(
                            recovery_delta, output_weight, units
                        ),
                        parameters,
                    )
                head_results[name] = result_entry
                if name == full_name:
                    result_entry["controls"] = {}
                    for size in CONTROLLED_SIZES:
                        result_entry["controls"][size] = []
                        for control in frozen_endpoint["head_induced"]["groups"][
                            str(size)
                        ]["controls"]:
                            result_entry["controls"][size].append(
                                intervention_effects(
                                    arrays,
                                    unit_group_write_delta(
                                        noising_delta, output_weight, control
                                    ),
                                    unit_group_write_delta(
                                        recovery_delta, output_weight, control
                                    ),
                                    parameters,
                                )
                            )
            full_path = head_results[full_name]["unit_paths"][
                INTERMEDIATE_SIZE
            ]
            prior_error = max(
                float(
                    np.max(
                        np.abs(
                            full_path[direction]
                            - prior_indirect_values(
                                prior,
                                endpoint=endpoint,
                                seed=seed,
                                direction=direction,
                            )
                        )
                    )
                )
                for direction in ("damage", "recovery")
            )
            if prior_error > 1e-4:
                raise RuntimeError(
                    f"head→MLP path replay differs from GFS18 by {prior_error}"
                )
            instrument_rows.append(
                {
                    "seed": seed,
                    "endpoint": endpoint,
                    "composition_max_abs_error": composition_error,
                    "prior_path_max_abs_error": prior_error,
                    "PASS": True,
                }
            )
            seed_values[seed][endpoint] = {
                "behavior": arrays["behavior"],
                "context": context_results,
                "context_controls": context_controls,
                "heads": head_results,
            }
        checkpoint_reports[str(seed)] = report.to_dict()
        del task, adapter, model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    behavior: dict[str, np.ndarray] = {}
    behavior_entries: list[dict[str, Any]] = []
    for endpoint_index, endpoint in enumerate(specs):
        behavior[endpoint] = average_arrays(
            [seed_values[seed][endpoint]["behavior"] for seed in EXPECTED_SEEDS]
        )
        summary = summarize(
            behavior[endpoint], groups, seed=880_000 + endpoint_index
        )
        summary["seed_estimates"] = {
            str(seed): float(seed_values[seed][endpoint]["behavior"].mean())
            for seed in EXPECTED_SEEDS
        }
        behavior_entries.append({"endpoint": endpoint, "summary": summary})
    apply_q(behavior_entries, "q_bh_2_behavior")
    behavior_gate = all(
        entry["summary"]["estimate"] >= 0.05
        and entry["summary"]["ci_low"] > 0
        and entry["summary"]["q_bh_2_behavior"] <= 0.05
        and all(
            value > 0 for value in entry["summary"]["seed_estimates"].values()
        )
        for entry in behavior_entries
    )
    if not behavior_gate:
        raise RuntimeError("evaluation behavior gate failed")

    npz: dict[str, np.ndarray] = {
        "example_ids": np.asarray(example_ids, dtype=str),
        "gene_ids": groups.astype(str),
    }

    def analyze_curve(
        family: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        results: dict[str, Any] = {}
        effect_entries: list[dict[str, Any]] = []
        contrast_entries: list[dict[str, Any]] = []
        for endpoint_index, endpoint in enumerate(specs):
            results[endpoint] = {}
            for size_index, size in enumerate(ALL_SIZES):
                results[endpoint][str(size)] = {}
                for direction_index, direction in enumerate(
                    ("damage", "recovery")
                ):
                    if family == "context":
                        values = average_arrays(
                            [
                                seed_values[seed][endpoint]["context"][size][
                                    direction
                                ]
                                for seed in EXPECTED_SEEDS
                            ]
                        )
                        ceiling = average_arrays(
                            [
                                seed_values[seed][endpoint]["context"][
                                    INTERMEDIATE_SIZE
                                ][direction]
                                for seed in EXPECTED_SEEDS
                            ]
                        )
                    else:
                        full_name = head_name(EXPECTED_HEADS[endpoint])
                        values = average_arrays(
                            [
                                seed_values[seed][endpoint]["heads"][full_name][
                                    "unit_paths"
                                ][size][direction]
                                for seed in EXPECTED_SEEDS
                            ]
                        )
                        ceiling = average_arrays(
                            [
                                seed_values[seed][endpoint]["heads"][full_name][
                                    "unit_paths"
                                ][INTERMEDIATE_SIZE][direction]
                                for seed in EXPECTED_SEEDS
                            ]
                        )
                    summary = summarize(
                        values,
                        groups,
                        seed=890_000
                        + (0 if family == "context" else 100_000)
                        + 10_000 * endpoint_index
                        + 100 * size_index
                        + direction_index,
                    )
                    summary["ratio_to_complete_path"] = grouped_bootstrap_ratio(
                        values,
                        ceiling,
                        groups,
                        n_boot=N_BOOT,
                        seed=891_000
                        + (0 if family == "context" else 100_000)
                        + 10_000 * endpoint_index
                        + 100 * size_index
                        + direction_index,
                    )
                    summary["seed_estimates"] = {
                        str(seed): float(
                            (
                                seed_values[seed][endpoint]["context"][size][
                                    direction
                                ]
                                if family == "context"
                                else seed_values[seed][endpoint]["heads"][
                                    head_name(EXPECTED_HEADS[endpoint])
                                ]["unit_paths"][size][direction]
                            ).mean()
                        )
                        for seed in EXPECTED_SEEDS
                    }
                    if size in CONTROLLED_SIZES:
                        if family == "context":
                            control_values = average_arrays(
                                [
                                    seed_values[seed][endpoint][
                                        "context_controls"
                                    ][size][draw][direction]
                                    for seed in EXPECTED_SEEDS
                                    for draw in range(3)
                                ]
                            )
                        else:
                            control_values = average_arrays(
                                [
                                    seed_values[seed][endpoint]["heads"][
                                        head_name(EXPECTED_HEADS[endpoint])
                                    ]["controls"][size][draw][direction]
                                    for seed in EXPECTED_SEEDS
                                    for draw in range(3)
                                ]
                            )
                        contrast = summarize(
                            values - control_values,
                            groups,
                            seed=892_000
                            + (0 if family == "context" else 100_000)
                            + 10_000 * endpoint_index
                            + 100 * size_index
                            + direction_index,
                        )
                        contrast["candidate_estimate"] = float(values.mean())
                        contrast["control_estimate"] = float(
                            control_values.mean()
                        )
                        summary["vs_output_norm_controls"] = contrast
                        contrast_entries.append(
                            {
                                "endpoint": endpoint,
                                "size": size,
                                "direction": direction,
                                "summary": contrast,
                            }
                        )
                    results[endpoint][str(size)][direction] = summary
                    effect_entries.append(
                        {
                            "endpoint": endpoint,
                            "size": size,
                            "direction": direction,
                            "summary": summary,
                        }
                    )
                    npz[
                        f"{family}_{endpoint}_{size}_{direction}"
                    ] = values
        apply_q(effect_entries, f"q_bh_28_{family}_unit_effects")
        apply_q(
            contrast_entries,
            f"q_bh_24_{family}_unit_control_contrasts",
        )
        return results, effect_entries, contrast_entries

    context_curve, context_entries, context_contrasts = analyze_curve("context")
    head_curve, head_entries, head_contrasts = analyze_curve("head")

    def curve_verdict(curve: dict[str, Any]) -> dict[str, Any]:
        verdict: dict[str, Any] = {}
        for endpoint in specs:
            crossing: dict[str, int | None] = {}
            for threshold_name, threshold in (("50pct", 0.50), ("80pct", 0.80)):
                passing = [
                    size
                    for size in ALL_SIZES
                    if all(
                        curve[endpoint][str(size)][direction][
                            "ratio_to_complete_path"
                        ]["ci_low"]
                        >= threshold
                        for direction in ("damage", "recovery")
                    )
                ]
                crossing[threshold_name] = min(passing) if passing else None
            verdict[endpoint] = {
                "minimum_50pct_size": crossing["50pct"],
                "minimum_80pct_size": crossing["80pct"],
                "sparse_at_80pct": bool(
                    crossing["80pct"] is not None
                    and crossing["80pct"] <= 64
                ),
                "distributed_by_frozen_rule": bool(
                    crossing["50pct"] is None or crossing["50pct"] > 256
                ),
            }
        return verdict

    context_verdict = curve_verdict(context_curve)
    head_verdict = curve_verdict(head_curve)

    singleton_paths: dict[str, Any] = {}
    triple_totals: dict[str, Any] = {}
    for endpoint_index, endpoint in enumerate(specs):
        full_name = head_name(EXPECTED_HEADS[endpoint])
        triple_totals[endpoint] = {}
        for direction_index, direction in enumerate(("damage", "recovery")):
            total_values = average_arrays(
                [
                    seed_values[seed][endpoint]["heads"][full_name]["total"][
                        direction
                    ]
                    for seed in EXPECTED_SEEDS
                ]
            )
            path_values = average_arrays(
                [
                    seed_values[seed][endpoint]["heads"][full_name][
                        "unit_paths"
                    ][INTERMEDIATE_SIZE][direction]
                    for seed in EXPECTED_SEEDS
                ]
            )
            triple_totals[endpoint][direction] = {
                "total": summarize(
                    total_values,
                    groups,
                    seed=1_100_000 + 100 * endpoint_index + direction_index,
                    infer=False,
                ),
                "complete_unit_path": summarize(
                    path_values,
                    groups,
                    seed=1_101_000 + 100 * endpoint_index + direction_index,
                    infer=False,
                ),
                "path_to_total_ratio": grouped_bootstrap_ratio(
                    path_values,
                    total_values,
                    groups,
                    n_boot=N_BOOT,
                    seed=1_102_000 + 100 * endpoint_index + direction_index,
                ),
            }
        singleton_paths[endpoint] = {}
        for head in EXPECTED_HEADS[endpoint]:
            name = head_name((head,))
            singleton_paths[endpoint][name] = {}
            for size in ALL_SIZES:
                singleton_paths[endpoint][name][str(size)] = {}
                for direction_index, direction in enumerate(
                    ("damage", "recovery")
                ):
                    values = average_arrays(
                        [
                            seed_values[seed][endpoint]["heads"][name][
                                "unit_paths"
                            ][size][direction]
                            for seed in EXPECTED_SEEDS
                        ]
                    )
                    total = average_arrays(
                        [
                            seed_values[seed][endpoint]["heads"][name]["total"][
                                direction
                            ]
                            for seed in EXPECTED_SEEDS
                        ]
                    )
                    singleton_paths[endpoint][name][str(size)][direction] = {
                        **summarize(
                            values,
                            groups,
                            seed=1_110_000
                            + 10_000 * endpoint_index
                            + 1_000 * head
                            + 10 * ALL_SIZES.index(size)
                            + direction_index,
                            infer=False,
                        ),
                        "ratio_to_singleton_total": grouped_bootstrap_ratio(
                            values,
                            total,
                            groups,
                            n_boot=N_BOOT,
                            seed=1_120_000
                            + 10_000 * endpoint_index
                            + 1_000 * head
                            + 10 * ALL_SIZES.index(size)
                            + direction_index,
                        ),
                    }

    overlaps: dict[str, Any] = {}
    for endpoint in specs:
        overlaps[endpoint] = {}
        for size in CONTROLLED_SIZES:
            per_seed = {}
            for seed in EXPECTED_SEEDS:
                endpoint_freeze = discovery["rankings"][str(seed)][endpoint]
                context_units = set(
                    endpoint_freeze["context"]["groups"][str(size)][
                        "candidates"
                    ]
                )
                head_units = set(
                    endpoint_freeze["head_induced"]["groups"][str(size)][
                        "candidates"
                    ]
                )
                intersection = len(context_units & head_units)
                per_seed[str(seed)] = {
                    "intersection": intersection,
                    "fraction_of_group": intersection / size,
                    "jaccard": intersection / len(context_units | head_units),
                }
            overlaps[endpoint][str(size)] = {
                "per_seed": per_seed,
                "mean_fraction_of_group": float(
                    np.mean(
                        [
                            row["fraction_of_group"]
                            for row in per_seed.values()
                        ]
                    )
                ),
            }

    arrays_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_output, **npz)
    result = {
        "study": "native GTEx exact final-MLP unit census",
        "stage": "gene-disjoint causal group and head→unit evaluation",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "evidence_tier": "exploratory causal native-unit mechanism",
        "device": device,
        "split": "reused GFS18 confirmation genes",
        "cohort_namespace": NAMESPACE,
        "cohort_tag": COHORT_TAG,
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "example_ids": example_ids,
        "discovery_gene_overlap": len(set(groups) & discovery_genes),
        "forbidden_chromosomes": sorted(FORBIDDEN_CHROMS),
        "regions": {
            endpoint: spec["window"].name for endpoint, spec in specs.items()
        },
        "frozen_heads": discovery["frozen_heads"],
        "checkpoint_reports": checkpoint_reports,
        "instrument_checks": instrument_rows,
        "instrument_gate": {
            "cohort_identity": True,
            "discovery_gene_disjoint": True,
            "behavior_gate": behavior_gate,
            "max_composition_error": max(
                row["composition_max_abs_error"] for row in instrument_rows
            ),
            "max_prior_path_error": max(
                row["prior_path_max_abs_error"] for row in instrument_rows
            ),
            "PASS": True,
        },
        "behavior": behavior_entries,
        "context_unit_curve": context_curve,
        "context_unit_verdict": context_verdict,
        "head_to_unit_curve": head_curve,
        "head_to_unit_verdict": head_verdict,
        "triple_total_and_complete_unit_path": triple_totals,
        "singleton_head_paths": singleton_paths,
        "context_head_unit_overlap": overlaps,
        "multiplicity_families": {
            "context_effects": len(context_entries),
            "context_control_contrasts": len(context_contrasts),
            "head_effects": len(head_entries),
            "head_control_contrasts": len(head_contrasts),
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
            "Exploratory native-unit completeness and exact head→unit mediation "
            "under the frozen curves; biological feature identity remains open."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_final_mlp_unit_evaluation_v1",
        sys.argv,
        repo=REPO,
        status="EXPLORATORY",
        config={
            "split": result["split"],
            "n": args.n,
            "batch": args.batch,
            "device": device,
            "intermediate_size": INTERMEDIATE_SIZE,
            "group_sizes": list(ALL_SIZES),
            "n_boot": N_BOOT,
            "n_perm": N_PERM,
        },
        seeds={"checkpoint_family": list(EXPECTED_SEEDS)},
        notes=[
            "All native-unit groups and controls were frozen in the discovery artifact.",
            "The complete head-induced native-unit path replays the prior GFS18 MLP path.",
            "Singleton-head paths are descriptive and do not receive component labels.",
            "The reused GFS18 partition makes the result exploratory.",
        ],
    )
    for path in (
        protocol,
        discovery_path,
        prior_path,
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
                "context_verdict": context_verdict,
                "head_to_unit_verdict": head_verdict,
                "top64_overlap": {
                    endpoint: overlaps[endpoint]["64"][
                        "mean_fraction_of_group"
                    ]
                    for endpoint in specs
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"COMPLETE: saved -> {output}")


if __name__ == "__main__":
    main()
