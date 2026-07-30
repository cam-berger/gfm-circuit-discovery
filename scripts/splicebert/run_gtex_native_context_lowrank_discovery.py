#!/usr/bin/env python
"""Fit frozen low-rank final-MLP context directions on fresh train genes."""

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
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.rigor import benjamini_hochberg  # noqa: E402
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gfmcd.splicing.context_joint_lowrank import (  # noqa: E402
    fit_lowrank_structure,
    project_rows,
    projection_matrix_error,
)
from gfmcd.splicing.context_localization import (  # noqa: E402
    RelativeWindow,
    shuffle_relative_window,
)
from gfmcd.splicing.grouped_stats import (  # noqa: E402
    grouped_bootstrap,
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


N_DISCOVERY = 128
N_BOOT = 5000
N_PERM = 9999
FINAL_LAYER = 5
NUM_HEADS = 16
HIDDEN_SIZE = 512
MAX_RANK = 64
MIN_BEHAVIOR = 0.05
MAX_Q = 0.05
NAMESPACE = "gtex-native-context-joint-lowrank-v1"
COHORT_TAG = "gtex-native-context-lowrank-discovery-v1"
FORBIDDEN_CHROMS = frozenset({"chr8", "chr9", "chr11", "chr16", "chr19"})
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
    "donor": ["L5H14", "L5H8", "L5H12"],
    "acceptor": ["L5H11", "L5H14", "L5H8"],
}


def model_weight(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model weights not found in {model_dir}")


def parse_window(name: str) -> RelativeWindow:
    start_text, end_text = name.split(":")
    return RelativeWindow(int(start_text), int(end_text))


def classifier_margin(
    model: Any,
    hidden: torch.Tensor,
    token_positions: torch.Tensor,
    *,
    label: int,
) -> torch.Tensor:
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    positions = token_positions.to(device=hidden.device)
    logits = model.classifier(model.dropout(hidden[rows, positions])).float()
    margin = logits[:, label] - logits[:, 0]
    if not torch.isfinite(margin).all():
        raise RuntimeError("non-finite classifier margin")
    return margin


def summarize(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    result = grouped_bootstrap(values, groups, n_boot=N_BOOT, seed=seed)
    result["p_signflip_add_one"] = grouped_signflip_pvalue(
        values,
        groups,
        n_perm=N_PERM,
        seed=seed + 100_000,
    )
    return result


def gene_from_example_id(example_id: str) -> str:
    try:
        return example_id.rsplit(":", 1)[1]
    except IndexError as exc:
        raise ValueError(f"cannot recover gene from {example_id!r}") from exc


def raw_gene_ids(artifact: dict[str, Any]) -> set[str]:
    raw = artifact.get("raw", [])
    if isinstance(raw, dict):
        rows = [row for values in raw.values() for row in values]
    else:
        rows = raw
    genes = {
        str(row["gene_id"])
        for row in rows
        if isinstance(row, dict) and "gene_id" in row
    }
    genes.update(
        gene_from_example_id(str(example_id))
        for example_id in artifact.get("example_ids", [])
    )
    return genes


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-artifact", action="append", required=True)
    parser.add_argument("--context-audit", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
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
        raise ValueError(f"low-rank discovery is frozen at n={N_DISCOVERY}")
    if args.n_boot != N_BOOT or args.n_perm != N_PERM:
        raise ValueError(
            f"inference is frozen at {N_BOOT} bootstraps/{N_PERM} sign flips"
        )
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CONTEXT_JOINT_LOWRANK_V1.md"
    )
    exclusion_paths = [
        Path(value).expanduser().resolve() for value in args.exclude_artifact
    ]
    if {path.name for path in exclusion_paths} != EXPECTED_EXCLUSIONS:
        raise ValueError(
            "exclude-artifact family does not match the frozen six artifacts"
        )
    artifacts = {
        path.name: json.loads(path.read_text()) for path in exclusion_paths
    }
    mechanism_preflight = artifacts["native_mechanism_preflight_v1.json"]
    census_discovery = artifacts[
        "native_context_head_census_discovery_v1.json"
    ]
    census_confirmation = artifacts[
        "native_context_head_census_confirmation_v1.json"
    ]
    context_audit_path = Path(args.context_audit).expanduser().resolve()
    context_audit = json.loads(context_audit_path.read_text())
    prerequisite_gate = {
        "mechanism_preflight_pass": mechanism_preflight.get("status") == "PASS",
        "census_discovery_complete": census_discovery.get("status")
        == "COMPLETE",
        "census_confirmation_complete": census_confirmation.get("status")
        == "COMPLETE",
        "census_heads_replay": all(
            census_discovery.get("freeze", {})
            .get(task_name, {})
            .get("candidate_heads")
            == expected
            and census_confirmation.get("confirmed_heads", {}).get(task_name)
            == expected
            for task_name, expected in EXPECTED_HEADS.items()
        ),
        "donor_region_licensed": bool(
            context_audit.get("licenses", {})
            .get("donor", {})
            .get("confirmed_32nt_region_licensed")
        ),
        "acceptor_region_licensed": bool(
            context_audit.get("licenses", {})
            .get("acceptor", {})
            .get("confirmed_32nt_region_licensed")
        ),
    }
    if not all(prerequisite_gate.values()):
        raise RuntimeError(
            f"joint/low-rank prerequisite gate failed: {prerequisite_gate}"
        )
    prior_gene_ids = set()
    for artifact in artifacts.values():
        prior_gene_ids.update(raw_gene_ids(artifact))
    if len(prior_gene_ids) < 400:
        raise RuntimeError("prior-exposed gene exclusion set is unexpectedly small")

    base_model = Path(args.base_model).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
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
        excluded_gene_ids=sorted(prior_gene_ids),
    )
    example_genes = {example.gene_id for example in examples}
    if example_genes & prior_gene_ids:
        raise RuntimeError("low-rank discovery contains a prior-exposed gene")
    normalized_chroms = {
        chrom if chrom.startswith("chr") else f"chr{chrom}"
        for chrom in (example.chrom for example in examples)
    }
    if normalized_chroms & FORBIDDEN_CHROMS:
        raise RuntimeError("low-rank discovery opened a forbidden chromosome")
    groups = np.asarray([example.gene_id for example in examples], dtype=object)

    model, tokenizer, report = load_gtex_token_classifier(
        base_model,
        checkpoint,
        device=device,
        attn_implementation="eager",
    )
    if report.seed != 2020:
        raise ValueError(f"discovery requires seed 2020, found {report.seed}")
    task = NativeClassifierTask(
        model,
        tokenizer,
        device=device,
        batch_size=args.batch,
    )
    adapter = SpliceBertAdapter(
        model,
        tokenizer,
        source=str(checkpoint),
        device=device,
    )
    if (
        adapter.num_layers != 6
        or adapter.num_heads != NUM_HEADS
        or adapter.hidden_size != HIDDEN_SIZE
    ):
        raise RuntimeError("frozen architecture is 6 x 16 x 512")

    windows = {
        task_name: parse_window(census_discovery["regions"][task_name])
        for task_name in ("donor", "acceptor")
    }
    clean_sequences = [example.sequence for example in examples]
    task_specs: dict[str, dict[str, Any]] = {
        "donor": {
            "anchors": [int(example.donor_pos) for example in examples],
            "targets": [int(example.donor_pos) - 1 for example in examples],
            "label": DONOR_LABEL,
            "window": windows["donor"],
        },
        "acceptor": {
            "anchors": [
                int(example.next_acceptor_pos) for example in examples
            ],
            "targets": [
                int(example.next_acceptor_pos) for example in examples
            ],
            "label": ACCEPTOR_LABEL,
            "window": windows["acceptor"],
        },
    }
    corrupt_sequences: dict[str, list[str]] = {}
    shuffle_audits: dict[str, list[dict[str, Any]]] = {}
    for task_name, spec in task_specs.items():
        corrupt_sequences[task_name] = []
        shuffle_audits[task_name] = []
        for example, anchor in zip(examples, spec["anchors"]):
            shuffled, audit = shuffle_relative_window(
                example.sequence,
                anchor=anchor,
                window=spec["window"],
                key=(
                    example.example_id,
                    task_name,
                    spec["window"].name,
                    "joint-lowrank-v1",
                ),
                draw=0,
            )
            if audit.hamming_fraction < 0.25:
                raise RuntimeError("low-rank discovery shuffle failed dose floor")
            corrupt_sequences[task_name].append(shuffled)
            shuffle_audits[task_name].append(audit.to_dict())

    arrays: dict[str, dict[str, Any]] = {}
    for task_index, (task_name, spec) in enumerate(task_specs.items(), start=1):
        behavior: list[float] = []
        mlp_delta: list[list[float]] = []
        for start in range(0, len(examples), args.batch):
            stop = min(start + args.batch, len(examples))
            clean_encoded = task.encode(clean_sequences[start:stop])
            corrupt_encoded = task.encode(
                corrupt_sequences[task_name][start:stop]
            )
            target_tokens = donor_token_positions(
                spec["targets"][start:stop],
                prefix_tokens=task.prefix_tokens,
                device=device,
            )
            clean_inputs = cache_final_layer_inputs(
                model,
                clean_encoded,
                layer=FINAL_LAYER,
            )
            corrupt_inputs = cache_final_layer_inputs(
                model,
                corrupt_encoded,
                layer=FINAL_LAYER,
            )
            clean_run = capture_final_layer_run(
                adapter,
                clean_inputs,
                layer=FINAL_LAYER,
            )
            corrupt_run = capture_final_layer_run(
                adapter,
                corrupt_inputs,
                layer=FINAL_LAYER,
            )
            clean_margin = classifier_margin(
                model,
                clean_run.hidden,
                target_tokens,
                label=int(spec["label"]),
            )
            corrupt_margin = classifier_margin(
                model,
                corrupt_run.hidden,
                target_tokens,
                label=int(spec["label"]),
            )
            behavior.extend(
                (clean_margin - corrupt_margin).detach().cpu().tolist()
            )
            rows = torch.arange(
                target_tokens.shape[0],
                device=target_tokens.device,
            )
            delta = (
                clean_run.mlp_write[rows, target_tokens]
                - corrupt_run.mlp_write[rows, target_tokens]
            )
            mlp_delta.extend(delta.detach().cpu().float().tolist())
            print(
                f"endpoint {task_index}/2 {task_name}: {stop}/{len(examples)}",
                end="\r",
                flush=True,
            )
        print()
        arrays[task_name] = {
            "behavior": np.asarray(behavior, dtype=np.float64),
            "mlp_delta": np.asarray(mlp_delta, dtype=np.float64),
        }

    behavior_summaries = {
        task_name: summarize(
            arrays[task_name]["behavior"],
            groups,
            seed=600_000 + task_index,
        )
        for task_index, task_name in enumerate(task_specs)
    }
    behavior_qvalues = benjamini_hochberg(
        [
            float(behavior_summaries[task_name]["p_signflip_add_one"])
            for task_name in task_specs
        ]
    )
    for task_name, qvalue in zip(task_specs, behavior_qvalues):
        behavior_summaries[task_name]["q_bh_2_behavior_tests"] = float(qvalue)
    behavior_gates = {
        task_name: {
            "mean_at_least_0_05": bool(
                summary["estimate"] >= MIN_BEHAVIOR
            ),
            "ci_above_zero": bool(summary["ci_low"] > 0),
            "q_at_most_0_05": bool(
                summary["q_bh_2_behavior_tests"] <= MAX_Q
            ),
        }
        for task_name, summary in behavior_summaries.items()
    }
    if not all(
        value for gates in behavior_gates.values() for value in gates.values()
    ):
        raise RuntimeError(f"low-rank discovery behavior gate failed: {behavior_gates}")

    structures: dict[str, dict[str, Any]] = {}
    geometry_checks: dict[str, dict[str, Any]] = {}
    for task_name, spec in task_specs.items():
        fitted = fit_lowrank_structure(
            arrays[task_name]["mlp_delta"],
            arrays[task_name]["behavior"],
            groups,
            max_rank=MAX_RANK,
        )
        basis = fitted["basis"]
        direction = fitted["task_direction"]
        identity = np.eye(HIDDEN_SIZE, dtype=np.float64)
        complete_projection_error = float(
            np.max(
                np.abs(
                    project_rows(arrays[task_name]["mlp_delta"], identity)
                    - arrays[task_name]["mlp_delta"]
                )
            )
        )
        zero_projection_error = float(
            np.max(
                np.abs(
                    project_rows(
                        arrays[task_name]["mlp_delta"],
                        np.zeros((0, HIDDEN_SIZE), dtype=np.float64),
                    )
                )
            )
        )
        basis_error = projection_matrix_error(basis)
        direction_norm_error = abs(float(np.linalg.norm(direction)) - 1.0)
        geometry_checks[task_name] = {
            "basis_orthonormal_error": basis_error,
            "direction_norm_error": direction_norm_error,
            "complete_projection_error": complete_projection_error,
            "zero_projection_error": zero_projection_error,
            "PASS": (
                basis_error <= 1e-8
                and direction_norm_error <= 1e-8
                and complete_projection_error <= 1e-6
                and zero_projection_error == 0.0
            ),
        }
        classifier_direction = (
            model.classifier.weight[int(spec["label"])]
            - model.classifier.weight[0]
        ).detach().cpu().double().numpy()
        classifier_direction /= np.linalg.norm(classifier_direction)
        structures[task_name] = {
            "gene_ids": fitted["gene_ids"],
            "gene_matrix": fitted["gene_matrix"],
            "gene_behavior": fitted["gene_behavior"],
            "svd_basis_64": basis,
            "singular_values": fitted["singular_values"],
            "cumulative_energy": fitted["cumulative_energy"],
            "task_direction": direction,
            "seed2020_classifier_direction_cosine": float(
                direction @ classifier_direction
            ),
            "rank_energy": {
                str(rank): float(fitted["cumulative_energy"][rank - 1])
                for rank in (1, 4, 16, 64)
            },
        }
    if not all(check["PASS"] for check in geometry_checks.values()):
        raise RuntimeError(f"low-rank geometry preflight failed: {geometry_checks}")

    donor_direction = np.asarray(
        structures["donor"]["task_direction"],
        dtype=np.float64,
    )
    acceptor_direction = np.asarray(
        structures["acceptor"]["task_direction"],
        dtype=np.float64,
    )
    donor_basis = np.asarray(
        structures["donor"]["svd_basis_64"],
        dtype=np.float64,
    )
    acceptor_basis = np.asarray(
        structures["acceptor"]["svd_basis_64"],
        dtype=np.float64,
    )
    principal_cosines = np.linalg.svd(
        donor_basis @ acceptor_basis.T,
        compute_uv=False,
    )

    raw = [
        {
            "example_id": example.example_id,
            "gene_id": example.gene_id,
            "chrom": example.chrom,
            "tasks": {
                task_name: {
                    "behavior": float(arrays[task_name]["behavior"][index]),
                    "mlp_delta": arrays[task_name]["mlp_delta"][index],
                    "shuffle_audit": shuffle_audits[task_name][index],
                }
                for task_name in task_specs
            },
        }
        for index, example in enumerate(examples)
    ]
    result = {
        "study": "native GTEx joint context and low-rank mechanism",
        "stage": "train-only MLP-write direction freeze",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "evidence_tier": "exploratory direction discovery",
        "device": device,
        "split": "train gene-hash low-rank discovery",
        "cohort_tag": COHORT_TAG,
        "cohort_namespace": NAMESPACE,
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "prior_exposed_genes_excluded": len(prior_gene_ids),
        "forbidden_chromosomes": sorted(FORBIDDEN_CHROMS),
        "checkpoint": report.to_dict(),
        "regions": {
            task_name: spec["window"].name
            for task_name, spec in task_specs.items()
        },
        "frozen_heads": EXPECTED_HEADS,
        "prerequisite_gate": prerequisite_gate,
        "behavior_summaries": behavior_summaries,
        "behavior_gates": behavior_gates,
        "geometry_checks": geometry_checks,
        "structures": structures,
        "cross_endpoint_geometry": {
            "task_direction_cosine": float(
                donor_direction @ acceptor_direction
            ),
            "rank64_principal_angle_cosines": principal_cosines,
        },
        "freeze": {
            "ranks": [1, 4, 16, 64],
            "task_direction": True,
            "random_draws_per_structure": 3,
            "random_basis_tag": "gtex-native-context-lowrank-random-v1",
            "confirmation_authorized": True,
        },
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default)
        + "\n"
    )
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_lowrank_discovery_v1",
        sys.argv,
        repo=REPO,
        status="EXPLORATORY",
        config={
            "split": result["split"],
            "n": args.n,
            "batch": args.batch,
            "device": device,
            "cohort_tag": COHORT_TAG,
            "cohort_namespace": NAMESPACE,
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "max_rank": MAX_RANK,
            "ranks": result["freeze"]["ranks"],
            "frozen_heads": EXPECTED_HEADS,
        },
        seeds={
            "checkpoint": 2020,
            "behavior_summary_base": 600_000,
        },
        notes=[
            "No causal projection outcome is used to choose a direction or rank.",
            "Every fit uses gene-mean MLP-write deltas from a fresh discovery partition.",
            "The task direction and SVD bases are frozen before confirmation.",
            "The fresh split is analysis-sealed, not claimed absent from checkpoint training.",
        ],
    )
    for path in (
        protocol,
        *exclusion_paths,
        context_audit_path,
        model_weight(base_model),
        base_model / "config.json",
        base_model / "tokenizer.json",
        checkpoint,
        Path(args.gtf),
        Path(args.fasta),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "behavior_gates": behavior_gates,
                "geometry_checks": geometry_checks,
                "rank_energy": {
                    task_name: values["rank_energy"]
                    for task_name, values in structures.items()
                },
                "task_direction_cosine": result["cross_endpoint_geometry"][
                    "task_direction_cosine"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"COMPLETE: saved -> {output}")
    del task, adapter, model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
