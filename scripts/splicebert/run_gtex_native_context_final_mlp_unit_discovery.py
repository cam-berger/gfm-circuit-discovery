#!/usr/bin/env python
"""Run the complete 2,048-unit final-MLP discovery census."""

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
    grouped_signflip_pvalue,
)
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.mlp_units import (  # noqa: E402
    composed_classifier_margin,
    exact_single_unit_effects,
    output_norm_matched_controls,
    rank_units_bidirectional,
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
N_DISCOVERY = 128
N_BOOT = 5000
N_PERM = 9999
FINAL_LAYER = 5
NUM_HEADS = 16
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 2048
GROUP_SIZES = (1, 4, 16, 64, 256, 1024)
NAMESPACE = "gtex-native-context-joint-lowrank-v1"
COHORT_TAG = "gtex-native-context-lowrank-discovery-v1"
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


def summarize(values: np.ndarray, groups: np.ndarray, *, seed: int) -> dict[str, Any]:
    result = grouped_bootstrap(values, groups, n_boot=N_BOOT, seed=seed)
    result["p_signflip_add_one"] = grouped_signflip_pvalue(
        values,
        groups,
        n_perm=N_PERM,
        seed=seed + 100_000,
    )
    return result


def exact_effects_for_two_deltas(
    noising_delta: np.ndarray,
    recovery_delta: np.ndarray,
    output_weight: np.ndarray,
    arrays: dict[str, np.ndarray],
    parameters: dict[str, Any],
) -> dict[str, np.ndarray]:
    noising = exact_single_unit_effects(
        noising_delta,
        output_weight,
        clean_residual=arrays["clean_residual"],
        clean_write=arrays["clean_write"],
        corrupt_residual=arrays["corrupt_residual"],
        corrupt_write=arrays["corrupt_write"],
        clean_margin=arrays["clean_margin"],
        corrupt_margin=arrays["corrupt_margin"],
        parameters=parameters,
    )
    if np.array_equal(noising_delta, recovery_delta):
        return noising
    recovery = exact_single_unit_effects(
        recovery_delta,
        output_weight,
        clean_residual=arrays["clean_residual"],
        clean_write=arrays["clean_write"],
        corrupt_residual=arrays["corrupt_residual"],
        corrupt_write=arrays["corrupt_write"],
        clean_margin=arrays["clean_margin"],
        corrupt_margin=arrays["corrupt_margin"],
        parameters=parameters,
    )
    return {"damage": noising["damage"], "recovery": recovery["recovery"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-discovery", required=True)
    parser.add_argument("--alignment-audit", required=True)
    parser.add_argument("--exclude-artifact", action="append", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=N_DISCOVERY)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.n != N_DISCOVERY:
        raise ValueError(f"discovery is frozen at n={N_DISCOVERY}")
    if args.n_boot != N_BOOT or args.n_perm != N_PERM:
        raise ValueError("inference settings differ from frozen protocol")
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CONTEXT_FINAL_MLP_UNIT_CENSUS_V1.md"
    )
    prior_path = Path(args.prior_discovery).expanduser().resolve()
    prior = json.loads(prior_path.read_text())
    alignment_audit_path = Path(args.alignment_audit).expanduser().resolve()
    alignment_audit = json.loads(alignment_audit_path.read_text())
    if (
        prior.get("status") != "COMPLETE"
        or prior.get("cohort_namespace") != NAMESPACE
        or alignment_audit.get("status") != "PASS"
        or not alignment_audit.get("all_corrected_gates_pass")
    ):
        raise RuntimeError("GFS18/GFS19 prerequisites are not complete")
    exclusion_paths = [
        Path(value).expanduser().resolve() for value in args.exclude_artifact
    ]
    if {path.name for path in exclusion_paths} != EXPECTED_EXCLUSIONS:
        raise ValueError("frozen six-artifact exclusion family required")
    artifacts = {
        path.name: json.loads(path.read_text()) for path in exclusion_paths
    }
    if artifacts["native_mechanism_preflight_v1.json"].get("status") != "PASS":
        raise RuntimeError("native mechanism preflight is not PASS")
    prior_genes: set[str] = set()
    for artifact in artifacts.values():
        prior_genes.update(raw_gene_ids(artifact))
    checkpoints = parse_checkpoints(args.checkpoint)
    base_model = Path(args.base_model).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)
    examples = load_annotation_gene_hash_cohort(
        args.gtf,
        args.fasta,
        base_split="train",
        analysis_partition="discovery",
        namespace=NAMESPACE,
        discovery_percent=50,
        n=args.n,
        tag=COHORT_TAG,
        excluded_gene_ids=sorted(prior_genes),
    )
    example_ids = [example.example_id for example in examples]
    if example_ids != [str(row["example_id"]) for row in prior["raw"]]:
        raise RuntimeError("discovery cohort does not exactly replay GFS18")
    groups = np.asarray([example.gene_id for example in examples], dtype=object)
    if set(groups) & prior_genes:
        raise RuntimeError("discovery contains prior-exposed genes")
    normalized_chroms = {
        chrom if chrom.startswith("chr") else f"chr{chrom}"
        for chrom in (example.chrom for example in examples)
    }
    if normalized_chroms & FORBIDDEN_CHROMS:
        raise RuntimeError("discovery opened a forbidden chromosome")
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
                raise RuntimeError("discovery shuffle failed dose floor")
            corrupt_sequences[endpoint].append(shuffled)
            shuffle_audits[endpoint].append(audit.to_dict())

    rankings: dict[str, Any] = {}
    checkpoint_reports: dict[str, Any] = {}
    preflight_rows: list[dict[str, Any]] = []
    behavior_entries: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
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
        output_weight = (
            output_dense.weight.detach().cpu().double().numpy()
        )
        output_bias = output_dense.bias.detach().cpu().double().numpy()
        output_norms = np.linalg.norm(output_weight, axis=0)
        rankings[str(seed)] = {}
        for endpoint_index, (endpoint, spec) in enumerate(specs.items()):
            buffers: dict[str, list[Any]] = {
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
                    "patched_clean_intermediate",
                    "patched_corrupt_intermediate",
                )
            }
            max_reconstruction_error = 0.0
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
                    raise RuntimeError("final MLP capture is incomplete")
                with patch_heads(
                    adapter,
                    {FINAL_LAYER: corrupt_run.heads},
                    [(FINAL_LAYER, head) for head in EXPECTED_HEADS[endpoint]],
                    token_mask=token_mask,
                ):
                    patched_clean = capture_final_layer_run(
                        adapter, clean_inputs, layer=FINAL_LAYER
                    )
                with patch_heads(
                    adapter,
                    {FINAL_LAYER: clean_run.heads},
                    [(FINAL_LAYER, head) for head in EXPECTED_HEADS[endpoint]],
                    token_mask=token_mask,
                ):
                    patched_corrupt = capture_final_layer_run(
                        adapter, corrupt_inputs, layer=FINAL_LAYER
                    )
                if (
                    patched_clean.mlp_intermediate is None
                    or patched_corrupt.mlp_intermediate is None
                ):
                    raise RuntimeError("patched MLP intermediate capture missing")
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
                run_pairs = (
                    ("clean", clean_run),
                    ("corrupt", corrupt_run),
                    ("patched_clean", patched_clean),
                    ("patched_corrupt", patched_corrupt),
                )
                for prefix, run in run_pairs:
                    intermediate = target_rows(
                        run.mlp_intermediate, positions
                    )
                    buffers[f"{prefix}_intermediate"].extend(
                        intermediate.cpu().float().tolist()
                    )
                    if prefix in ("clean", "corrupt"):
                        write = target_rows(run.mlp_write, positions)
                        residual = target_rows(run.mlp_residual, positions)
                        buffers[f"{prefix}_write"].extend(
                            write.cpu().float().tolist()
                        )
                        buffers[f"{prefix}_residual"].extend(
                            residual.cpu().float().tolist()
                        )
                        reconstruction = (
                            intermediate.cpu().double().numpy()
                            @ output_weight.T
                            + output_bias
                        )
                        max_reconstruction_error = max(
                            max_reconstruction_error,
                            float(
                                np.max(
                                    np.abs(
                                        reconstruction
                                        - write.cpu().double().numpy()
                                    )
                                )
                            ),
                        )
                print(
                    f"seed {seed} endpoint {endpoint_index + 1}/2 {endpoint}: "
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
            context_delta = (
                arrays["clean_intermediate"]
                - arrays["corrupt_intermediate"]
            )
            head_noising_delta = (
                arrays["clean_intermediate"]
                - arrays["patched_clean_intermediate"]
            )
            head_recovery_delta = (
                arrays["patched_corrupt_intermediate"]
                - arrays["corrupt_intermediate"]
            )
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
            clean_composed = composed_classifier_margin(
                arrays["clean_residual"],
                arrays["clean_write"],
                **parameters,
            )
            corrupt_composed = composed_classifier_margin(
                arrays["corrupt_residual"],
                arrays["corrupt_write"],
                **parameters,
            )
            composition_error = float(
                max(
                    np.max(np.abs(clean_composed - arrays["clean_margin"])),
                    np.max(np.abs(corrupt_composed - arrays["corrupt_margin"])),
                )
            )
            complete_delta = unit_group_write_delta(
                context_delta, output_weight, range(INTERMEDIATE_SIZE)
            )
            target_write_delta = (
                arrays["clean_write"] - arrays["corrupt_write"]
            )
            complete_delta_error = float(
                np.max(np.abs(complete_delta - target_write_delta))
            )
            zero_error = float(
                np.max(
                    np.abs(
                        unit_group_write_delta(
                            context_delta, output_weight, []
                        )
                    )
                )
            )
            if (
                max_reconstruction_error > 1e-5
                or composition_error > 1e-5
                or complete_delta_error > 1e-5
                or zero_error != 0.0
            ):
                raise RuntimeError("native-unit algebra preflight failed")
            preflight_rows.append(
                {
                    "seed": seed,
                    "endpoint": endpoint,
                    "write_reconstruction_max_abs_error": max_reconstruction_error,
                    "composition_max_abs_error": composition_error,
                    "complete_delta_max_abs_error": complete_delta_error,
                    "zero_group_max_abs_error": zero_error,
                    "PASS": True,
                }
            )
            summary = summarize(
                arrays["behavior"],
                groups,
                seed=870_000 + 100 * seed_index + endpoint_index,
            )
            behavior_entries.append(
                {"seed": seed, "endpoint": endpoint, "summary": summary}
            )
            families = {
                "context": exact_effects_for_two_deltas(
                    context_delta,
                    context_delta,
                    output_weight,
                    arrays,
                    parameters,
                ),
                "head_induced": exact_effects_for_two_deltas(
                    head_noising_delta,
                    head_recovery_delta,
                    output_weight,
                    arrays,
                    parameters,
                ),
            }
            rankings[str(seed)][endpoint] = {}
            for family, effects in families.items():
                ranked = rank_units_bidirectional(
                    effects["damage"], effects["recovery"], groups
                )
                groups_and_controls: dict[str, Any] = {}
                for size in GROUP_SIZES:
                    candidates = ranked["order"][:size]
                    controls = output_norm_matched_controls(
                        candidates,
                        output_norms,
                        tag=f"final-mlp-unit:{family}:{seed}:{endpoint}:{size}",
                    )
                    groups_and_controls[str(size)] = {
                        "candidates": candidates.tolist(),
                        "controls": [control.tolist() for control in controls],
                        "candidate_output_norm_mean": float(
                            output_norms[candidates].mean()
                        ),
                        "control_output_norm_means": [
                            float(output_norms[control].mean())
                            for control in controls
                        ],
                    }
                rankings[str(seed)][endpoint][family] = {
                    "ranking_rule": "minimum of gene-mean damage and recovery",
                    "unit_order": ranked["order"].tolist(),
                    "unit_score": ranked["score"].tolist(),
                    "unit_mean_damage": ranked["mean_damage"].tolist(),
                    "unit_mean_recovery": ranked["mean_recovery"].tolist(),
                    "groups": groups_and_controls,
                }
        checkpoint_reports[str(seed)] = report.to_dict()
        del task, adapter, model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    qvalues = benjamini_hochberg(
        [entry["summary"]["p_signflip_add_one"] for entry in behavior_entries]
    )
    for entry, qvalue in zip(behavior_entries, qvalues):
        entry["summary"]["q_bh_10_discovery_behavior"] = float(qvalue)
    behavior_gate = all(
        entry["summary"]["estimate"] >= 0.05
        and entry["summary"]["ci_low"] > 0
        and entry["summary"]["q_bh_10_discovery_behavior"] <= 0.05
        for entry in behavior_entries
    )
    if not behavior_gate:
        raise RuntimeError("unit discovery behavior gate failed")
    result = {
        "study": "native GTEx exact final-MLP unit census",
        "stage": "complete discovery census and group freeze",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "evidence_tier": "exploratory causal unit discovery",
        "device": device,
        "split": "reused GFS18 discovery genes",
        "cohort_namespace": NAMESPACE,
        "cohort_tag": COHORT_TAG,
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "example_ids": example_ids,
        "forbidden_chromosomes": sorted(FORBIDDEN_CHROMS),
        "regions": {
            endpoint: spec["window"].name for endpoint, spec in specs.items()
        },
        "frozen_heads": {
            endpoint: [f"L5H{head}" for head in heads]
            for endpoint, heads in EXPECTED_HEADS.items()
        },
        "checkpoint_reports": checkpoint_reports,
        "instrument_checks": preflight_rows,
        "instrument_gate": {
            "architecture_6x16x512x2048": True,
            "cohort_identity": True,
            "behavior_gate": behavior_gate,
            "max_write_reconstruction_error": max(
                row["write_reconstruction_max_abs_error"]
                for row in preflight_rows
            ),
            "max_composition_error": max(
                row["composition_max_abs_error"] for row in preflight_rows
            ),
            "max_complete_delta_error": max(
                row["complete_delta_max_abs_error"] for row in preflight_rows
            ),
            "PASS": True,
        },
        "behavior": behavior_entries,
        "rankings": rankings,
        "freeze": {
            "group_sizes": list(GROUP_SIZES),
            "ceiling_size": INTERMEDIATE_SIZE,
            "control_draws": 3,
            "control_match": "nearest non-candidate output-column norm",
            "evaluation_authorized": True,
        },
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
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_final_mlp_unit_discovery_v1",
        sys.argv,
        repo=REPO,
        status="EXPLORATORY",
        config={
            "split": result["split"],
            "n": args.n,
            "batch": args.batch,
            "device": device,
            "intermediate_size": INTERMEDIATE_SIZE,
            "group_sizes": list(GROUP_SIZES),
            "n_boot": N_BOOT,
            "n_perm": N_PERM,
        },
        seeds={"checkpoint_family": list(EXPECTED_SEEDS)},
        notes=[
            "Every one of 2,048 post-GELU native units is evaluated by exact interchange.",
            "Context and head-induced rankings are frozen separately per checkpoint and endpoint.",
            "The reused GFS18 partition makes this exploratory rather than independent confirmation.",
        ],
    )
    for path in (
        protocol,
        prior_path,
        alignment_audit_path,
        *exclusion_paths,
        model_weight(base_model),
        base_model / "config.json",
        base_model / "tokenizer.json",
        *checkpoints.values(),
        Path(args.gtf),
        Path(args.fasta),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "instrument_gate": result["instrument_gate"],
                "top_units": {
                    seed: {
                        endpoint: {
                            family: rankings[seed][endpoint][family][
                                "unit_order"
                            ][:8]
                            for family in ("context", "head_induced")
                        }
                        for endpoint in specs
                    }
                    for seed in map(str, EXPECTED_SEEDS)
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"COMPLETE: saved -> {output}")


if __name__ == "__main__":
    main()
