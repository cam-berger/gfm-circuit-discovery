#!/usr/bin/env python
"""Fit per-seed final-MLP subspaces and outcome-blind alignment maps."""

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
    fit_orthogonal_alignment,
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
N_DISCOVERY = 128
N_BOOT = 5000
N_PERM = 9999
FINAL_LAYER = 5
NUM_HEADS = 16
HIDDEN_SIZE = 512
MAX_RANK = 64
MIN_EFFECT = 0.05
MAX_Q = 0.05
NAMESPACE = "gtex-native-context-joint-lowrank-v1"
COHORT_TAG = "gtex-native-context-lowrank-discovery-v1"
FORBIDDEN_CHROMS = frozenset({"chr8", "chr9", "chr11", "chr16", "chr19"})
RANKS = (1, 2, 4, 8, 16, 32, 64)


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


def classifier_margin(
    model: Any,
    hidden: torch.Tensor,
    token_positions: torch.Tensor,
    *,
    label: int,
) -> torch.Tensor:
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    target = hidden[rows, token_positions.to(device=hidden.device)]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-discovery", required=True)
    parser.add_argument("--context-audit", required=True)
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
    parser.add_argument("--arrays-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != N_DISCOVERY:
        raise ValueError(f"discovery is frozen at n={N_DISCOVERY}")
    if args.n_boot != N_BOOT or args.n_perm != N_PERM:
        raise ValueError("inference settings differ from the frozen protocol")
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CONTEXT_PER_SEED_ALIGNMENT_V1.md"
    )
    prior_path = Path(args.prior_discovery).expanduser().resolve()
    prior = json.loads(prior_path.read_text())
    if (
        prior.get("status") != "COMPLETE"
        or prior.get("cohort_namespace") != NAMESPACE
    ):
        raise RuntimeError("GFS18 discovery artifact is not the frozen cohort source")
    exclusion_paths = [
        Path(value).expanduser().resolve() for value in args.exclude_artifact
    ]
    if {path.name for path in exclusion_paths} != EXPECTED_EXCLUSIONS:
        raise ValueError("the frozen six-artifact exclusion family is required")
    artifacts = {
        path.name: json.loads(path.read_text()) for path in exclusion_paths
    }
    audit_path = Path(args.context_audit).expanduser().resolve()
    audit = json.loads(audit_path.read_text())
    prerequisite_gate = {
        "mechanism_preflight_pass": artifacts[
            "native_mechanism_preflight_v1.json"
        ].get("status")
        == "PASS",
        "prior_discovery_complete": prior.get("status") == "COMPLETE",
        "donor_region_licensed": bool(
            audit.get("licenses", {})
            .get("donor", {})
            .get("confirmed_32nt_region_licensed")
        ),
        "acceptor_region_licensed": bool(
            audit.get("licenses", {})
            .get("acceptor", {})
            .get("confirmed_32nt_region_licensed")
        ),
    }
    if not all(prerequisite_gate.values()):
        raise RuntimeError(f"alignment prerequisite failed: {prerequisite_gate}")
    prior_genes: set[str] = set()
    for artifact in artifacts.values():
        prior_genes.update(raw_gene_ids(artifact))
    if len(prior_genes) < 400:
        raise RuntimeError("prior gene exclusion set is unexpectedly small")

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
        analysis_partition="discovery",
        namespace=NAMESPACE,
        discovery_percent=50,
        n=args.n,
        tag=COHORT_TAG,
        excluded_gene_ids=sorted(prior_genes),
    )
    example_ids = [example.example_id for example in examples]
    prior_ids = [str(row["example_id"]) for row in prior["raw"]]
    if example_ids != prior_ids:
        raise RuntimeError("reconstructed discovery examples differ from GFS18")
    example_genes = {example.gene_id for example in examples}
    if example_genes & prior_genes:
        raise RuntimeError("discovery contains a prior-exposed gene")
    normalized_chroms = {
        chrom if chrom.startswith("chr") else f"chr{chrom}"
        for chrom in (example.chrom for example in examples)
    }
    if normalized_chroms & FORBIDDEN_CHROMS:
        raise RuntimeError("discovery opened a forbidden chromosome")
    groups = np.asarray([example.gene_id for example in examples], dtype=object)

    census = artifacts["native_context_head_census_discovery_v1.json"]
    windows = {
        endpoint: parse_window(census["regions"][endpoint])
        for endpoint in ("donor", "acceptor")
    }
    specs = {
        "donor": {
            "anchors": [int(example.donor_pos) for example in examples],
            "targets": [int(example.donor_pos) - 1 for example in examples],
            "label": DONOR_LABEL,
            "window": windows["donor"],
        },
        "acceptor": {
            "anchors": [int(example.next_acceptor_pos) for example in examples],
            "targets": [int(example.next_acceptor_pos) for example in examples],
            "label": ACCEPTOR_LABEL,
            "window": windows["acceptor"],
        },
    }
    clean_sequences = [example.sequence for example in examples]
    corrupt_sequences: dict[str, list[str]] = {}
    shuffle_audits: dict[str, list[dict[str, Any]]] = {}
    for endpoint, spec in specs.items():
        corrupt_sequences[endpoint] = []
        shuffle_audits[endpoint] = []
        for example, anchor in zip(examples, spec["anchors"]):
            shuffled, shuffle_audit = shuffle_relative_window(
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
            if shuffle_audit.hamming_fraction < 0.25:
                raise RuntimeError("discovery shuffle failed the frozen dose floor")
            corrupt_sequences[endpoint].append(shuffled)
            shuffle_audits[endpoint].append(shuffle_audit.to_dict())

    captured: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    checkpoint_reports: dict[str, Any] = {}
    layer_norms: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        model, tokenizer, report = load_gtex_token_classifier(
            base_model,
            checkpoints[seed],
            device=device,
            attn_implementation="eager",
        )
        if report.seed != seed:
            raise RuntimeError(f"checkpoint seed mismatch: requested {seed}")
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
            adapter.num_layers != 6
            or adapter.num_heads != NUM_HEADS
            or adapter.hidden_size != HIDDEN_SIZE
        ):
            raise RuntimeError("frozen architecture is 6 x 16 x 512")
        captured[seed] = {}
        for endpoint_index, (endpoint, spec) in enumerate(specs.items(), start=1):
            buffers = {
                name: []
                for name in (
                    "behavior",
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
                clean_encoded = task.encode(clean_sequences[start:stop])
                corrupt_encoded = task.encode(
                    corrupt_sequences[endpoint][start:stop]
                )
                positions = donor_token_positions(
                    spec["targets"][start:stop],
                    prefix_tokens=task.prefix_tokens,
                    device=device,
                )
                clean_run = capture_final_layer_run(
                    adapter,
                    cache_final_layer_inputs(
                        model, clean_encoded, layer=FINAL_LAYER
                    ),
                    layer=FINAL_LAYER,
                )
                corrupt_run = capture_final_layer_run(
                    adapter,
                    cache_final_layer_inputs(
                        model, corrupt_encoded, layer=FINAL_LAYER
                    ),
                    layer=FINAL_LAYER,
                )
                clean_margin = classifier_margin(
                    model,
                    clean_run.hidden,
                    positions,
                    label=int(spec["label"]),
                )
                corrupt_margin = classifier_margin(
                    model,
                    corrupt_run.hidden,
                    positions,
                    label=int(spec["label"]),
                )
                buffers["clean_margin"].extend(clean_margin.cpu().tolist())
                buffers["corrupt_margin"].extend(corrupt_margin.cpu().tolist())
                buffers["behavior"].extend(
                    (clean_margin - corrupt_margin).cpu().tolist()
                )
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
            converted = {
                name: np.asarray(
                    values,
                    dtype=np.float32
                    if name.endswith(("write", "residual"))
                    else np.float64,
                )
                for name, values in buffers.items()
            }
            converted["mlp_delta"] = (
                converted["clean_write"] - converted["corrupt_write"]
            )
            captured[seed][endpoint] = converted
        checkpoint_reports[str(seed)] = report.to_dict()
        layer = adapter.encoder_layers[FINAL_LAYER]
        layer_norms[str(seed)] = {
            "eps": float(layer.output.LayerNorm.eps),
            "weight_shape": list(layer.output.LayerNorm.weight.shape),
        }
        del task, adapter, model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    behavior_entries: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        for endpoint_index, endpoint in enumerate(specs):
            summary = summarize(
                captured[seed][endpoint]["behavior"],
                groups,
                seed=810_000 + 100 * seed_index + endpoint_index,
            )
            behavior_entries.append(
                {"seed": seed, "endpoint": endpoint, "summary": summary}
            )
    qvalues = benjamini_hochberg(
        [row["summary"]["p_signflip_add_one"] for row in behavior_entries]
    )
    for row, qvalue in zip(behavior_entries, qvalues):
        row["summary"]["q_bh_10_discovery_behavior"] = float(qvalue)
    behavior_gate = all(
        row["summary"]["estimate"] >= MIN_EFFECT
        and row["summary"]["ci_low"] > 0
        and row["summary"]["q_bh_10_discovery_behavior"] <= MAX_Q
        for row in behavior_entries
    )
    if not behavior_gate:
        raise RuntimeError("per-seed discovery behavior gate failed")

    structures: dict[str, Any] = {}
    npz: dict[str, np.ndarray] = {}
    geometry_checks: dict[str, Any] = {}
    gene_matrices: dict[tuple[int, str], np.ndarray] = {}
    bases: dict[tuple[int, str], np.ndarray] = {}
    for seed in EXPECTED_SEEDS:
        structures[str(seed)] = {}
        geometry_checks[str(seed)] = {}
        for endpoint in specs:
            fitted = fit_lowrank_structure(
                captured[seed][endpoint]["mlp_delta"],
                captured[seed][endpoint]["behavior"],
                groups,
                max_rank=MAX_RANK,
            )
            basis = np.asarray(fitted["basis"], dtype=np.float64)
            basis_error = projection_matrix_error(basis)
            identity = np.eye(HIDDEN_SIZE)
            complete_error = float(
                np.max(
                    np.abs(
                        project_rows(
                            captured[seed][endpoint]["mlp_delta"], identity
                        )
                        - captured[seed][endpoint]["mlp_delta"]
                    )
                )
            )
            zero_error = float(
                np.max(
                    np.abs(
                        project_rows(
                            captured[seed][endpoint]["mlp_delta"],
                            np.zeros((0, HIDDEN_SIZE)),
                        )
                    )
                )
            )
            if basis_error > 1e-8 or complete_error > 1e-6 or zero_error != 0:
                raise RuntimeError("per-seed subspace geometry preflight failed")
            gene_matrices[(seed, endpoint)] = fitted["gene_matrix"]
            bases[(seed, endpoint)] = basis
            npz[f"basis_{seed}_{endpoint}"] = basis.astype(np.float32)
            npz[f"gene_matrix_{seed}_{endpoint}"] = fitted[
                "gene_matrix"
            ].astype(np.float32)
            structures[str(seed)][endpoint] = {
                "gene_ids": fitted["gene_ids"],
                "singular_values": fitted["singular_values"].tolist(),
                "rank_energy": {
                    str(rank): float(fitted["cumulative_energy"][rank - 1])
                    for rank in RANKS
                },
            }
            geometry_checks[str(seed)][endpoint] = {
                "basis_orthonormal_error": basis_error,
                "complete_projection_error": complete_error,
                "zero_projection_error": zero_error,
                "PASS": True,
            }
            for name, values in captured[seed][endpoint].items():
                npz[f"{name}_{seed}_{endpoint}"] = values

    alignments: dict[str, Any] = {}
    max_alignment_error = 0.0
    for source in EXPECTED_SEEDS:
        for target in EXPECTED_SEEDS:
            if source == target:
                continue
            pair = f"{source}_to_{target}"
            alignments[pair] = {}
            for endpoint in specs:
                fitted = fit_orthogonal_alignment(
                    gene_matrices[(source, endpoint)],
                    gene_matrices[(target, endpoint)],
                )
                rotation = fitted["rotation"]
                transported = transform_row_basis(
                    bases[(source, endpoint)], rotation
                )
                transported_error = projection_matrix_error(transported)
                max_alignment_error = max(
                    max_alignment_error,
                    fitted["orthogonality_error"],
                    transported_error,
                )
                if (
                    fitted["orthogonality_error"] > 1e-8
                    or transported_error > 1e-8
                ):
                    raise RuntimeError("orthogonal alignment preflight failed")
                npz[f"rotation_{source}_{target}_{endpoint}"] = rotation.astype(
                    np.float32
                )
                rank_geometry = {}
                for rank in (4, 64):
                    target_basis = bases[(target, endpoint)][:rank]
                    raw = np.linalg.svd(
                        bases[(source, endpoint)][:rank] @ target_basis.T,
                        compute_uv=False,
                    )
                    aligned = np.linalg.svd(
                        transported[:rank] @ target_basis.T,
                        compute_uv=False,
                    )
                    rank_geometry[str(rank)] = {
                        "raw_mean_principal_cosine": float(np.mean(raw)),
                        "aligned_mean_principal_cosine": float(np.mean(aligned)),
                        "raw_min_principal_cosine": float(np.min(raw)),
                        "aligned_min_principal_cosine": float(np.min(aligned)),
                    }
                alignments[pair][endpoint] = {
                    "relative_error": fitted["relative_error"],
                    "mean_row_cosine": fitted["mean_row_cosine"],
                    "orthogonality_error": fitted["orthogonality_error"],
                    "transported_basis_orthonormal_error": transported_error,
                    "rank_geometry": rank_geometry,
                }

    npz["example_ids"] = np.asarray(example_ids, dtype=str)
    npz["gene_ids"] = groups.astype(str)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_output, **npz)
    result = {
        "study": "native GTEx per-seed final-MLP alignment",
        "stage": "outcome-blind discovery freeze",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "evidence_tier": "exploratory representation discovery",
        "device": device,
        "split": "reused GFS18 discovery genes",
        "cohort_namespace": NAMESPACE,
        "cohort_tag": COHORT_TAG,
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "example_ids": example_ids,
        "prior_exposed_genes_excluded": len(prior_genes),
        "forbidden_chromosomes": sorted(FORBIDDEN_CHROMS),
        "checkpoint_reports": checkpoint_reports,
        "regions": {
            endpoint: spec["window"].name for endpoint, spec in specs.items()
        },
        "prerequisite_gate": prerequisite_gate,
        "cohort_identity_gate": True,
        "behavior": behavior_entries,
        "behavior_gate": behavior_gate,
        "structures": structures,
        "alignments": alignments,
        "geometry_checks": geometry_checks,
        "max_alignment_or_transport_orthogonality_error": max_alignment_error,
        "arrays": str(arrays_output.relative_to(REPO)),
        "freeze": {
            "ranks": list(RANKS),
            "max_rank": MAX_RANK,
            "alignment": "full-width orthogonal Procrustes on gene-mean MLP deltas",
            "alignment_uses_outcomes": False,
            "random_subspace_draws": 3,
            "signed_permutation_draws": 3,
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
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_per_seed_alignment_discovery_v1",
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
            "max_rank": MAX_RANK,
            "n_boot": N_BOOT,
            "n_perm": N_PERM,
        },
        seeds={"checkpoint_family": list(EXPECTED_SEEDS)},
        notes=[
            "The reused GFS18 partitions make this developmental, not independent confirmation.",
            "Orthogonal maps use discovery MLP deltas only; no labels or margins enter fitting.",
            "The discovery JSON and compressed arrays freeze all evaluation structures.",
        ],
    )
    for path in (
        protocol,
        prior_path,
        audit_path,
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
                "behavior_gate": behavior_gate,
                "max_alignment_error": max_alignment_error,
                "rank4_energy": {
                    str(seed): {
                        endpoint: structures[str(seed)][endpoint]["rank_energy"]["4"]
                        for endpoint in specs
                    }
                    for seed in EXPECTED_SEEDS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"COMPLETE: saved -> {output}")


if __name__ == "__main__":
    main()
