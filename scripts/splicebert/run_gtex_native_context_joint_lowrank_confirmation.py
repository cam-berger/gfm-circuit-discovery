#!/usr/bin/env python
"""Confirm joint L5 heads, head-to-MLP mediation, and low-rank MLP structure."""

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
from gfmcd.splicing.circuit import clamp_write, pick_device  # noqa: E402
from gfmcd.splicing.context_joint_lowrank import (  # noqa: E402
    deterministic_random_basis,
    matched_control_sets,
    ordered_nonempty_subsets,
    projection_matrix_error,
    replace_target_vectors,
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
    compose_target_mlp_hidden,
    donor_token_positions,
    position_mask,
    replay_final_layer,
)


EXPECTED_SEEDS = frozenset(range(2020, 2025))
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
N_CONFIRMATION = 200
N_BOOT = 5000
N_PERM = 9999
FINAL_LAYER = 5
NUM_HEADS = 16
HIDDEN_SIZE = 768
RANKS = (1, 4, 16, 64)
MIN_EFFECT = 0.05
MAX_Q = 0.05
COMPLETENESS_FLOOR = 0.80
LOWRANK_FLOOR = 0.50
NAMESPACE = "gtex-native-context-joint-lowrank-v1"
COHORT_TAG = "gtex-native-context-joint-lowrank-confirmation-v1"
FORBIDDEN_CHROMS = frozenset({"chr8", "chr9", "chr11", "chr16", "chr19"})


def parse_checkpoints(values: list[str]) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for value in values:
        try:
            seed_text, path_text = value.split("=", 1)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError("--checkpoint must be SEED=PATH") from exc
        if seed in checkpoints:
            raise ValueError(f"duplicate seed {seed}")
        checkpoints[seed] = Path(path_text).expanduser().resolve()
    if set(checkpoints) != set(EXPECTED_SEEDS):
        raise ValueError(
            f"complete seed family required; found {sorted(checkpoints)}"
        )
    return checkpoints


def model_weight(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model weights not found in {model_dir}")


def parse_window(name: str) -> RelativeWindow:
    start_text, end_text = name.split(":")
    return RelativeWindow(int(start_text), int(end_text))


def head_number(name: str) -> int:
    prefix = f"L{FINAL_LAYER}H"
    if not name.startswith(prefix):
        raise ValueError(f"expected layer-5 head, found {name!r}")
    head = int(name.removeprefix(prefix))
    if not 0 <= head < NUM_HEADS:
        raise ValueError(f"head outside architecture: {name!r}")
    return head


def head_set_name(heads: tuple[int, ...]) -> str:
    return "+".join(f"L{FINAL_LAYER}H{head}" for head in heads)


def classifier_margin(
    model: Any,
    hidden: torch.Tensor,
    token_positions: torch.Tensor,
    *,
    label: int,
) -> torch.Tensor:
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    positions = token_positions.to(device=hidden.device)
    target_hidden = hidden[rows, positions]
    return classifier_margin_from_target(model, target_hidden, label=label)


def classifier_margin_from_target(
    model: Any,
    target_hidden: torch.Tensor,
    *,
    label: int,
) -> torch.Tensor:
    logits = model.classifier(model.dropout(target_hidden)).float()
    margin = logits[:, label] - logits[:, 0]
    if not torch.isfinite(margin).all():
        raise RuntimeError("non-finite classifier margin")
    return margin


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


def apply_qvalues(
    entries: list[tuple[dict[str, Any], str]],
    *,
    field: str,
) -> None:
    qvalues = benjamini_hochberg(
        [float(result["p_signflip_add_one"]) for result, _ in entries]
    )
    for (result, _), qvalue in zip(entries, qvalues):
        result[field] = float(qvalue)


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


def average_seed_values(
    seed_arrays: dict[int, dict[str, Any]],
    task_name: str,
    getter: Any,
) -> np.ndarray:
    return np.mean(
        np.stack(
            [
                np.asarray(getter(seed_arrays[seed][task_name]), dtype=np.float64)
                for seed in sorted(EXPECTED_SEEDS)
            ],
            axis=0,
        ),
        axis=0,
    )


def seed_estimates(
    seed_arrays: dict[int, dict[str, Any]],
    task_name: str,
    getter: Any,
) -> dict[str, float]:
    return {
        str(seed): float(
            np.mean(np.asarray(getter(seed_arrays[seed][task_name])))
        )
        for seed in sorted(EXPECTED_SEEDS)
    }


def target_vectors(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[rows, positions.to(device=tensor.device)]


def project_torch_rows(
    matrix: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    return (matrix.float() @ basis.float().T) @ basis.float()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--census-discovery", required=True)
    parser.add_argument("--exclude-artifact", action="append", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=N_CONFIRMATION)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != N_CONFIRMATION:
        raise ValueError(f"confirmation is frozen at n={N_CONFIRMATION}")
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
    discovery_path = Path(args.discovery).expanduser().resolve()
    census_path = Path(args.census_discovery).expanduser().resolve()
    discovery = json.loads(discovery_path.read_text())
    census = json.loads(census_path.read_text())
    if (
        discovery.get("status") != "COMPLETE"
        or not discovery.get("freeze", {}).get("confirmation_authorized")
        or discovery.get("cohort_namespace") != NAMESPACE
    ):
        raise RuntimeError("low-rank discovery does not authorize confirmation")
    if discovery.get("frozen_heads") != EXPECTED_HEADS:
        raise RuntimeError("frozen head sets changed after preregistration")
    if discovery.get("freeze", {}).get("ranks") != list(RANKS):
        raise RuntimeError("frozen low-rank family changed")

    exclusion_paths = [
        Path(value).expanduser().resolve() for value in args.exclude_artifact
    ]
    if {path.name for path in exclusion_paths} != EXPECTED_EXCLUSIONS:
        raise ValueError("exclude-artifact family does not match the frozen set")
    prior_artifacts = {
        path.name: json.loads(path.read_text()) for path in exclusion_paths
    }
    prior_gene_ids: set[str] = set()
    for artifact in prior_artifacts.values():
        prior_gene_ids.update(raw_gene_ids(artifact))
    discovery_gene_ids = raw_gene_ids(discovery)
    excluded_gene_ids = prior_gene_ids | discovery_gene_ids
    if len(discovery_gene_ids) != int(discovery["n_groups"]):
        raise RuntimeError("discovery raw genes do not replay its group count")

    task_components: dict[str, dict[str, Any]] = {}
    freeze = census.get("freeze", {})
    for task_name in ("donor", "acceptor"):
        candidates = [
            head_number(name) for name in EXPECTED_HEADS[task_name]
        ]
        census_freeze = freeze.get(task_name, {})
        if census_freeze.get("candidate_heads") != EXPECTED_HEADS[task_name]:
            raise RuntimeError(f"{task_name} census candidate freeze changed")
        controls = {
            head_number(candidate): [head_number(name) for name in names]
            for candidate, names in census_freeze.get(
                "control_heads", {}
            ).items()
        }
        if set(controls) != set(candidates):
            raise RuntimeError(f"{task_name} control freeze is incomplete")
        if any(len(values) != 3 for values in controls.values()):
            raise RuntimeError(f"{task_name} requires three controls/candidate")
        subsets = ordered_nonempty_subsets(candidates)
        matched = {
            size: matched_control_sets(candidates, controls, size=size)
            for size in range(1, len(candidates) + 1)
        }
        task_components[task_name] = {
            "candidates": tuple(candidates),
            "controls": controls,
            "subsets": subsets,
            "matched_controls": matched,
        }

    structures: dict[str, dict[str, np.ndarray]] = {}
    random_structures: dict[str, dict[str, list[np.ndarray]]] = {}
    random_tag = str(discovery["freeze"]["random_basis_tag"])
    for task_name in ("donor", "acceptor"):
        fitted = discovery["structures"][task_name]
        svd_basis = np.asarray(fitted["svd_basis_64"], dtype=np.float64)
        direction = np.asarray(fitted["task_direction"], dtype=np.float64)
        if svd_basis.shape != (64, HIDDEN_SIZE) or direction.shape != (
            HIDDEN_SIZE,
        ):
            raise RuntimeError(f"{task_name} low-rank structure shape changed")
        if projection_matrix_error(svd_basis) > 1e-8:
            raise RuntimeError(f"{task_name} SVD basis is not orthonormal")
        if abs(float(np.linalg.norm(direction)) - 1.0) > 1e-8:
            raise RuntimeError(f"{task_name} task direction is not unit length")
        structures[task_name] = {
            "task_direction": direction[None, :],
            **{
                f"svd_rank_{rank}": svd_basis[:rank]
                for rank in RANKS
            },
        }
        random_structures[task_name] = {
            name: [
                deterministic_random_basis(
                    HIDDEN_SIZE,
                    basis.shape[0],
                    tag=f"{random_tag}:{task_name}:{name}",
                    draw=draw,
                )
                for draw in range(3)
            ]
            for name, basis in structures[task_name].items()
        }

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
        analysis_partition="confirmation",
        namespace=NAMESPACE,
        discovery_percent=50,
        n=args.n,
        tag=COHORT_TAG,
        excluded_gene_ids=sorted(excluded_gene_ids),
    )
    example_genes = {example.gene_id for example in examples}
    if example_genes & excluded_gene_ids:
        raise RuntimeError("confirmation contains a prior-exposed gene")
    normalized_chroms = {
        chrom if chrom.startswith("chr") else f"chr{chrom}"
        for chrom in (example.chrom for example in examples)
    }
    if normalized_chroms & FORBIDDEN_CHROMS:
        raise RuntimeError("confirmation opened a forbidden chromosome")
    groups = np.asarray([example.gene_id for example in examples], dtype=object)

    windows = {
        task_name: parse_window(discovery["regions"][task_name])
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
                raise RuntimeError("confirmation shuffle failed dose floor")
            corrupt_sequences[task_name].append(shuffled)
            shuffle_audits[task_name].append(audit.to_dict())

    seed_arrays: dict[int, dict[str, Any]] = {}
    checkpoint_reports: dict[int, dict[str, Any]] = {}
    instrument_checks: dict[str, list[dict[str, float]]] = {
        "composition_identity": [],
        "literal_clamp_equivalence": [],
    }
    for seed in sorted(EXPECTED_SEEDS):
        model, tokenizer, report = load_gtex_token_classifier(
            base_model,
            checkpoints[seed],
            device=device,
            attn_implementation="eager",
        )
        if report.seed != seed:
            raise RuntimeError(f"checkpoint seed mismatch: {seed} vs {report.seed}")
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
            raise RuntimeError("frozen architecture is 6 x 16 x 768")
        layer_norm = adapter.encoder_layers[FINAL_LAYER].output.LayerNorm
        seed_arrays[seed] = {}

        for task_index, (task_name, spec) in enumerate(
            task_specs.items(),
            start=1,
        ):
            components = task_components[task_name]
            candidate_names = {
                head_set_name(heads): heads for heads in components["subsets"]
            }
            control_names = {
                head_set_name(heads): heads
                for size in (1, 2, 3)
                for heads in components["matched_controls"][size]
            }
            full_candidate_name = head_set_name(components["candidates"])
            full_control_names = [
                head_set_name(heads)
                for heads in components["matched_controls"][3]
            ]
            arrays: dict[str, Any] = {
                "behavior": [],
                "full_mlp": {"damage": [], "recovery": []},
                "heads": {
                    name: {"damage": [], "recovery": []}
                    for name in candidate_names
                },
                "controls": {
                    name: {"damage": [], "recovery": []}
                    for name in control_names
                },
                "path_candidate": {
                    "indirect": {"damage": [], "recovery": []},
                    "direct": {"damage": [], "recovery": []},
                },
                "path_controls": {
                    name: {
                        "indirect": {"damage": [], "recovery": []},
                        "direct": {"damage": [], "recovery": []},
                    }
                    for name in full_control_names
                },
                "lowrank": {
                    name: {"damage": [], "recovery": []}
                    for name in structures[task_name]
                },
                "lowrank_controls": {
                    name: [
                        {"damage": [], "recovery": []} for _ in range(3)
                    ]
                    for name in structures[task_name]
                },
                "classifier_direction_cosine": float("nan"),
            }
            classifier_direction = (
                model.classifier.weight[int(spec["label"])]
                - model.classifier.weight[0]
            ).detach().cpu().double().numpy()
            classifier_direction /= np.linalg.norm(classifier_direction)
            arrays["classifier_direction_cosine"] = float(
                structures[task_name]["task_direction"][0]
                @ classifier_direction
            )

            torch_structures = {
                name: torch.as_tensor(
                    basis,
                    dtype=torch.float32,
                    device=device,
                )
                for name, basis in structures[task_name].items()
            }
            torch_random = {
                name: [
                    torch.as_tensor(
                        basis,
                        dtype=torch.float32,
                        device=device,
                    )
                    for basis in bases
                ]
                for name, bases in random_structures[task_name].items()
            }

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
                target_mask = position_mask(
                    target_tokens,
                    int(clean_encoded["input_ids"].shape[1]),
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
                arrays["behavior"].extend(
                    (clean_margin - corrupt_margin).detach().cpu().tolist()
                )
                clean_write = target_vectors(
                    clean_run.mlp_write, target_tokens
                )
                corrupt_write = target_vectors(
                    corrupt_run.mlp_write, target_tokens
                )
                delta = clean_write - corrupt_write

                noised_target = compose_target_mlp_hidden(
                    clean_run,
                    target_tokens,
                    corrupt_write,
                    layer_norm,
                )
                denoised_target = compose_target_mlp_hidden(
                    corrupt_run,
                    target_tokens,
                    clean_write,
                    layer_norm,
                )
                noised_margin = classifier_margin_from_target(
                    model, noised_target, label=int(spec["label"])
                )
                denoised_margin = classifier_margin_from_target(
                    model, denoised_target, label=int(spec["label"])
                )
                arrays["full_mlp"]["damage"].extend(
                    (clean_margin - noised_margin).detach().cpu().tolist()
                )
                arrays["full_mlp"]["recovery"].extend(
                    (denoised_margin - corrupt_margin).detach().cpu().tolist()
                )

                identity_target = compose_target_mlp_hidden(
                    clean_run,
                    target_tokens,
                    clean_write,
                    layer_norm,
                )
                clean_target = target_vectors(clean_run.hidden, target_tokens)
                identity_error = float(
                    (identity_target - clean_target)
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                )
                instrument_checks["composition_identity"].append(
                    {
                        "seed": float(seed),
                        "endpoint": float(task_index),
                        "batch": float(start // args.batch),
                        "max_abs_error": identity_error,
                    }
                )

                if start == 0:
                    source_write = replace_target_vectors(
                        clean_run.mlp_write,
                        target_tokens,
                        corrupt_write,
                    )
                    with clamp_write(
                        adapter,
                        [(FINAL_LAYER, "mlp")],
                        {(FINAL_LAYER, "mlp"): source_write},
                        positions=target_tokens,
                    ):
                        literal_hidden = replay_final_layer(
                            adapter,
                            clean_inputs,
                            layer=FINAL_LAYER,
                        )
                    literal_margin = classifier_margin(
                        model,
                        literal_hidden,
                        target_tokens,
                        label=int(spec["label"]),
                    )
                    literal_error = float(
                        (literal_margin - noised_margin)
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    )
                    instrument_checks["literal_clamp_equivalence"].append(
                        {
                            "seed": float(seed),
                            "endpoint": float(task_index),
                            "max_abs_margin_error": literal_error,
                        }
                    )

                def evaluate_head_set(
                    heads: tuple[int, ...],
                    *,
                    path: bool,
                ) -> dict[str, dict[str, torch.Tensor]]:
                    clean_source = {FINAL_LAYER: clean_run.heads}
                    corrupt_source = {FINAL_LAYER: corrupt_run.heads}
                    selected = [(FINAL_LAYER, head) for head in heads]
                    with patch_heads(
                        adapter,
                        corrupt_source,
                        selected,
                        token_mask=target_mask,
                    ):
                        patched_clean = capture_final_layer_run(
                            adapter,
                            clean_inputs,
                            layer=FINAL_LAYER,
                        )
                    with patch_heads(
                        adapter,
                        clean_source,
                        selected,
                        token_mask=target_mask,
                    ):
                        patched_corrupt = capture_final_layer_run(
                            adapter,
                            corrupt_inputs,
                            layer=FINAL_LAYER,
                        )
                    patched_clean_margin = classifier_margin(
                        model,
                        patched_clean.hidden,
                        target_tokens,
                        label=int(spec["label"]),
                    )
                    patched_corrupt_margin = classifier_margin(
                        model,
                        patched_corrupt.hidden,
                        target_tokens,
                        label=int(spec["label"]),
                    )
                    result = {
                        "total": {
                            "damage": clean_margin - patched_clean_margin,
                            "recovery": patched_corrupt_margin - corrupt_margin,
                        }
                    }
                    if path:
                        patched_clean_write = target_vectors(
                            patched_clean.mlp_write, target_tokens
                        )
                        patched_corrupt_write = target_vectors(
                            patched_corrupt.mlp_write, target_tokens
                        )
                        indirect_noised_target = compose_target_mlp_hidden(
                            clean_run,
                            target_tokens,
                            patched_clean_write,
                            layer_norm,
                        )
                        indirect_denoised_target = compose_target_mlp_hidden(
                            corrupt_run,
                            target_tokens,
                            patched_corrupt_write,
                            layer_norm,
                        )
                        direct_noised_target = compose_target_mlp_hidden(
                            patched_clean,
                            target_tokens,
                            clean_write,
                            layer_norm,
                        )
                        direct_denoised_target = compose_target_mlp_hidden(
                            patched_corrupt,
                            target_tokens,
                            corrupt_write,
                            layer_norm,
                        )
                        result["indirect"] = {
                            "damage": clean_margin
                            - classifier_margin_from_target(
                                model,
                                indirect_noised_target,
                                label=int(spec["label"]),
                            ),
                            "recovery": classifier_margin_from_target(
                                model,
                                indirect_denoised_target,
                                label=int(spec["label"]),
                            )
                            - corrupt_margin,
                        }
                        result["direct"] = {
                            "damage": clean_margin
                            - classifier_margin_from_target(
                                model,
                                direct_noised_target,
                                label=int(spec["label"]),
                            ),
                            "recovery": classifier_margin_from_target(
                                model,
                                direct_denoised_target,
                                label=int(spec["label"]),
                            )
                            - corrupt_margin,
                        }
                    return result

                for name, heads in candidate_names.items():
                    path = name == full_candidate_name
                    effects = evaluate_head_set(heads, path=path)
                    for direction in ("damage", "recovery"):
                        arrays["heads"][name][direction].extend(
                            effects["total"][direction]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        if path:
                            for route in ("indirect", "direct"):
                                arrays["path_candidate"][route][direction].extend(
                                    effects[route][direction]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                )

                for name, heads in control_names.items():
                    path = name in full_control_names
                    effects = evaluate_head_set(heads, path=path)
                    for direction in ("damage", "recovery"):
                        arrays["controls"][name][direction].extend(
                            effects["total"][direction]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        if path:
                            for route in ("indirect", "direct"):
                                arrays["path_controls"][name][route][
                                    direction
                                ].extend(
                                    effects[route][direction]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                )

                for structure_name, basis in torch_structures.items():
                    projected = project_torch_rows(delta, basis).to(
                        dtype=clean_write.dtype
                    )
                    projected_noised_target = compose_target_mlp_hidden(
                        clean_run,
                        target_tokens,
                        clean_write - projected,
                        layer_norm,
                    )
                    projected_denoised_target = compose_target_mlp_hidden(
                        corrupt_run,
                        target_tokens,
                        corrupt_write + projected,
                        layer_norm,
                    )
                    projected_noised_margin = classifier_margin_from_target(
                        model,
                        projected_noised_target,
                        label=int(spec["label"]),
                    )
                    projected_denoised_margin = classifier_margin_from_target(
                        model,
                        projected_denoised_target,
                        label=int(spec["label"]),
                    )
                    arrays["lowrank"][structure_name]["damage"].extend(
                        (clean_margin - projected_noised_margin)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    arrays["lowrank"][structure_name]["recovery"].extend(
                        (projected_denoised_margin - corrupt_margin)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    for draw, random_basis in enumerate(
                        torch_random[structure_name]
                    ):
                        random_projected = project_torch_rows(
                            delta, random_basis
                        ).to(dtype=clean_write.dtype)
                        random_noised_target = compose_target_mlp_hidden(
                            clean_run,
                            target_tokens,
                            clean_write - random_projected,
                            layer_norm,
                        )
                        random_denoised_target = compose_target_mlp_hidden(
                            corrupt_run,
                            target_tokens,
                            corrupt_write + random_projected,
                            layer_norm,
                        )
                        random_noised_margin = classifier_margin_from_target(
                            model,
                            random_noised_target,
                            label=int(spec["label"]),
                        )
                        random_denoised_margin = classifier_margin_from_target(
                            model,
                            random_denoised_target,
                            label=int(spec["label"]),
                        )
                        arrays["lowrank_controls"][structure_name][draw][
                            "damage"
                        ].extend(
                            (clean_margin - random_noised_margin)
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        arrays["lowrank_controls"][structure_name][draw][
                            "recovery"
                        ].extend(
                            (random_denoised_margin - corrupt_margin)
                            .detach()
                            .cpu()
                            .tolist()
                        )

                print(
                    f"seed {seed} endpoint {task_index}/2 {task_name}: "
                    f"{stop}/{len(examples)}",
                    end="\r",
                    flush=True,
                )
                del (
                    clean_encoded,
                    corrupt_encoded,
                    clean_inputs,
                    corrupt_inputs,
                    clean_run,
                    corrupt_run,
                )
            print()

            arrays["behavior"] = np.asarray(arrays["behavior"], dtype=np.float64)
            for direction in ("damage", "recovery"):
                arrays["full_mlp"][direction] = np.asarray(
                    arrays["full_mlp"][direction], dtype=np.float64
                )
            for family_name in ("heads", "controls"):
                for values in arrays[family_name].values():
                    for direction in ("damage", "recovery"):
                        values[direction] = np.asarray(
                            values[direction], dtype=np.float64
                        )
            for route in ("indirect", "direct"):
                for direction in ("damage", "recovery"):
                    arrays["path_candidate"][route][direction] = np.asarray(
                        arrays["path_candidate"][route][direction],
                        dtype=np.float64,
                    )
            for values in arrays["path_controls"].values():
                for route in ("indirect", "direct"):
                    for direction in ("damage", "recovery"):
                        values[route][direction] = np.asarray(
                            values[route][direction], dtype=np.float64
                        )
            for values in arrays["lowrank"].values():
                for direction in ("damage", "recovery"):
                    values[direction] = np.asarray(
                        values[direction], dtype=np.float64
                    )
            for draws in arrays["lowrank_controls"].values():
                for values in draws:
                    for direction in ("damage", "recovery"):
                        values[direction] = np.asarray(
                            values[direction], dtype=np.float64
                        )
            seed_arrays[seed][task_name] = arrays
        checkpoint_reports[seed] = report.to_dict()
        del task, adapter, model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    max_composition_error = max(
        row["max_abs_error"]
        for row in instrument_checks["composition_identity"]
    )
    max_clamp_error = max(
        row["max_abs_margin_error"]
        for row in instrument_checks["literal_clamp_equivalence"]
    )
    instrument_gate = {
        "composition_identity_max_abs_error": max_composition_error,
        "literal_clamp_margin_max_abs_error": max_clamp_error,
        "composition_identity_at_most_1e_5": max_composition_error <= 1e-5,
        "literal_clamp_equivalence_at_most_1e_5": max_clamp_error <= 1e-5,
    }
    if not all(
        value for key, value in instrument_gate.items() if key.endswith("1e_5")
    ):
        raise RuntimeError(f"path-composition instrument failed: {instrument_gate}")

    family_behavior: dict[str, np.ndarray] = {}
    behavior_summaries: dict[str, dict[str, Any]] = {}
    behavior_entries: list[tuple[dict[str, Any], str]] = []
    for task_index, task_name in enumerate(task_specs):
        values = average_seed_values(
            seed_arrays,
            task_name,
            lambda arrays: arrays["behavior"],
        )
        family_behavior[task_name] = values
        summary = summarize(values, groups, seed=700_000 + task_index)
        summary["seed_estimates"] = seed_estimates(
            seed_arrays,
            task_name,
            lambda arrays: arrays["behavior"],
        )
        behavior_summaries[task_name] = summary
        behavior_entries.append((summary, task_name))
    apply_qvalues(behavior_entries, field="q_bh_2_behavior_tests")
    behavior_gates = {
        task_name: {
            "mean_at_least_0_05": bool(summary["estimate"] >= MIN_EFFECT),
            "ci_above_zero": bool(summary["ci_low"] > 0),
            "q_at_most_0_05": bool(
                summary["q_bh_2_behavior_tests"] <= MAX_Q
            ),
            "positive_5_of_5": all(
                float(value) > 0
                for value in summary["seed_estimates"].values()
            ),
        }
        for task_name, summary in behavior_summaries.items()
    }

    full_mlp: dict[str, dict[str, Any]] = {}
    full_mlp_entries: list[tuple[dict[str, Any], str]] = []
    full_mlp_values: dict[str, dict[str, np.ndarray]] = {}
    for task_index, task_name in enumerate(task_specs):
        full_mlp[task_name] = {}
        full_mlp_values[task_name] = {}
        for direction_index, direction in enumerate(("damage", "recovery")):
            values = average_seed_values(
                seed_arrays,
                task_name,
                lambda arrays, d=direction: arrays["full_mlp"][d],
            )
            full_mlp_values[task_name][direction] = values
            summary = summarize(
                values,
                groups,
                seed=710_000 + 100 * task_index + direction_index,
            )
            summary["ratio_to_behavior"] = grouped_bootstrap_ratio(
                values,
                family_behavior[task_name],
                groups,
                n_boot=N_BOOT,
                seed=711_000 + 100 * task_index + direction_index,
            )
            summary["seed_estimates"] = seed_estimates(
                seed_arrays,
                task_name,
                lambda arrays, d=direction: arrays["full_mlp"][d],
            )
            full_mlp[task_name][direction] = summary
            full_mlp_entries.append(
                (summary, f"{task_name}:{direction}")
            )
    apply_qvalues(full_mlp_entries, field="q_bh_4_full_mlp_tests")

    joint: dict[str, Any] = {
        "subsets": {},
        "candidate_vs_matched_controls": {},
        "interactions": {},
        "full_set_verdicts": {},
    }
    joint_values: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    joint_entries: list[tuple[dict[str, Any], str]] = []
    joint_contrast_entries: list[tuple[dict[str, Any], str]] = []
    for task_index, task_name in enumerate(task_specs):
        components = task_components[task_name]
        joint["subsets"][task_name] = {}
        joint["candidate_vs_matched_controls"][task_name] = {}
        joint["interactions"][task_name] = {}
        joint_values[task_name] = {}
        for subset_index, heads in enumerate(components["subsets"]):
            name = head_set_name(heads)
            size = len(heads)
            control_names = [
                head_set_name(control_heads)
                for control_heads in components["matched_controls"][size]
            ]
            joint["subsets"][task_name][name] = {}
            joint["candidate_vs_matched_controls"][task_name][name] = {}
            joint_values[task_name][name] = {}
            for direction_index, direction in enumerate(
                ("damage", "recovery")
            ):
                candidate_values = average_seed_values(
                    seed_arrays,
                    task_name,
                    lambda arrays, n=name, d=direction: arrays["heads"][n][d],
                )
                joint_values[task_name][name][direction] = candidate_values
                summary = summarize(
                    candidate_values,
                    groups,
                    seed=720_000
                    + 10_000 * task_index
                    + 100 * subset_index
                    + direction_index,
                )
                summary["ratio_to_behavior"] = grouped_bootstrap_ratio(
                    candidate_values,
                    family_behavior[task_name],
                    groups,
                    n_boot=N_BOOT,
                    seed=721_000
                    + 10_000 * task_index
                    + 100 * subset_index
                    + direction_index,
                )
                summary["seed_estimates"] = seed_estimates(
                    seed_arrays,
                    task_name,
                    lambda arrays, n=name, d=direction: arrays["heads"][n][d],
                )
                joint["subsets"][task_name][name][direction] = summary
                joint_entries.append(
                    (summary, f"{task_name}:{name}:{direction}")
                )

                control_values = average_seed_values(
                    seed_arrays,
                    task_name,
                    lambda arrays, names=control_names, d=direction: np.mean(
                        np.stack(
                            [arrays["controls"][control][d] for control in names],
                            axis=0,
                        ),
                        axis=0,
                    ),
                )
                contrast_values = candidate_values - control_values
                contrast = summarize(
                    contrast_values,
                    groups,
                    seed=730_000
                    + 10_000 * task_index
                    + 100 * subset_index
                    + direction_index,
                )
                contrast["candidate_estimate"] = float(
                    candidate_values.mean()
                )
                contrast["matched_control_estimate"] = float(
                    control_values.mean()
                )
                contrast["matched_control_sets"] = control_names
                joint["candidate_vs_matched_controls"][task_name][name][
                    direction
                ] = contrast
                joint_contrast_entries.append(
                    (contrast, f"{task_name}:{name}:{direction}")
                )
    apply_qvalues(joint_entries, field="q_bh_28_joint_subset_tests")
    apply_qvalues(
        joint_contrast_entries,
        field="q_bh_28_joint_control_contrasts",
    )

    for task_index, task_name in enumerate(task_specs):
        components = task_components[task_name]
        full_name = head_set_name(components["candidates"])
        singles = [
            head_set_name((head,)) for head in components["candidates"]
        ]
        for direction_index, direction in enumerate(("damage", "recovery")):
            full_values = joint_values[task_name][full_name][direction]
            singleton_sum = np.sum(
                np.stack(
                    [
                        joint_values[task_name][name][direction]
                        for name in singles
                    ],
                    axis=0,
                ),
                axis=0,
            )
            interaction = full_values - singleton_sum
            interaction_summary = summarize(
                interaction,
                groups,
                seed=740_000 + 100 * task_index + direction_index,
                infer=False,
            )
            interaction_summary["joint_to_singleton_sum_ratio"] = (
                grouped_bootstrap_ratio(
                    full_values,
                    singleton_sum,
                    groups,
                    n_boot=N_BOOT,
                    seed=741_000 + 100 * task_index + direction_index,
                )
            )
            leave_one_out: dict[str, Any] = {}
            for head_index, head in enumerate(components["candidates"]):
                remaining = tuple(
                    value
                    for value in components["candidates"]
                    if value != head
                )
                remaining_name = head_set_name(remaining)
                marginal = (
                    full_values
                    - joint_values[task_name][remaining_name][direction]
                )
                leave_one_out[f"L{FINAL_LAYER}H{head}"] = summarize(
                    marginal,
                    groups,
                    seed=742_000
                    + 1_000 * task_index
                    + 10 * direction_index
                    + head_index,
                    infer=False,
                )
            joint["interactions"][task_name][direction] = {
                "joint_minus_sum_singletons": interaction_summary,
                "leave_one_out_marginals": leave_one_out,
            }

        checks = {"behavior_gate": all(behavior_gates[task_name].values())}
        for direction in ("damage", "recovery"):
            effect = joint["subsets"][task_name][full_name][direction]
            contrast = joint["candidate_vs_matched_controls"][task_name][
                full_name
            ][direction]
            checks[f"{direction}_joint_effect"] = bool(
                effect["estimate"] >= MIN_EFFECT
                and effect["ci_low"] > 0
                and effect["q_bh_28_joint_subset_tests"] <= MAX_Q
                and all(
                    float(value) > 0
                    for value in effect["seed_estimates"].values()
                )
            )
            checks[f"{direction}_vs_matched_controls"] = bool(
                contrast["ci_low"] > 0
                and contrast["q_bh_28_joint_control_contrasts"] <= MAX_Q
            )
            checks[f"{direction}_completeness_at_least_0_80"] = bool(
                effect["ratio_to_behavior"]["ci_low"]
                >= COMPLETENESS_FLOOR
            )
        joint["full_set_verdicts"][task_name] = {
            "checks": checks,
            "joint_effect_confirmed": all(
                value
                for key, value in checks.items()
                if "completeness" not in key
            ),
            "complete_at_80_percent": all(checks.values()),
        }

    path: dict[str, Any] = {
        "candidate": {},
        "candidate_vs_full_matched_controls": {},
        "verdicts": {},
        "nonadditivity_note": (
            "total, indirect, and direct are separate interventions and are "
            "not assumed to add under LayerNorm"
        ),
    }
    path_entries: list[tuple[dict[str, Any], str]] = []
    path_contrast_entries: list[tuple[dict[str, Any], str]] = []
    for task_index, task_name in enumerate(task_specs):
        full_control_names = [
            head_set_name(heads)
            for heads in task_components[task_name]["matched_controls"][3]
        ]
        full_candidate_name = head_set_name(
            task_components[task_name]["candidates"]
        )
        path["candidate"][task_name] = {}
        path["candidate_vs_full_matched_controls"][task_name] = {}
        for direction_index, direction in enumerate(("damage", "recovery")):
            path["candidate"][task_name][direction] = {}
            path["candidate_vs_full_matched_controls"][task_name][direction] = {}
            total_values = joint_values[task_name][full_candidate_name][direction]
            for route_index, route in enumerate(("indirect", "direct")):
                values = average_seed_values(
                    seed_arrays,
                    task_name,
                    lambda arrays, r=route, d=direction: arrays[
                        "path_candidate"
                    ][r][d],
                )
                summary = summarize(
                    values,
                    groups,
                    seed=750_000
                    + 10_000 * task_index
                    + 100 * direction_index
                    + route_index,
                    infer=route == "indirect",
                )
                summary["ratio_to_total"] = grouped_bootstrap_ratio(
                    values,
                    total_values,
                    groups,
                    n_boot=N_BOOT,
                    seed=751_000
                    + 10_000 * task_index
                    + 100 * direction_index
                    + route_index,
                )
                summary["ratio_to_behavior"] = grouped_bootstrap_ratio(
                    values,
                    family_behavior[task_name],
                    groups,
                    n_boot=N_BOOT,
                    seed=752_000
                    + 10_000 * task_index
                    + 100 * direction_index
                    + route_index,
                )
                summary["seed_estimates"] = seed_estimates(
                    seed_arrays,
                    task_name,
                    lambda arrays, r=route, d=direction: arrays[
                        "path_candidate"
                    ][r][d],
                )
                path["candidate"][task_name][direction][route] = summary
                if route == "indirect":
                    path_entries.append(
                        (summary, f"{task_name}:{direction}:{route}")
                    )
                    control_values = average_seed_values(
                        seed_arrays,
                        task_name,
                        lambda arrays, names=full_control_names, d=direction: (
                            np.mean(
                                np.stack(
                                    [
                                        arrays["path_controls"][name][
                                            "indirect"
                                        ][d]
                                        for name in names
                                    ],
                                    axis=0,
                                ),
                                axis=0,
                            )
                        ),
                    )
                    contrast = summarize(
                        values - control_values,
                        groups,
                        seed=760_000
                        + 1_000 * task_index
                        + 10 * direction_index,
                    )
                    contrast["candidate_estimate"] = float(values.mean())
                    contrast["matched_control_estimate"] = float(
                        control_values.mean()
                    )
                    contrast["matched_control_sets"] = full_control_names
                    path["candidate_vs_full_matched_controls"][task_name][
                        direction
                    ] = contrast
                    path_contrast_entries.append(
                        (contrast, f"{task_name}:{direction}")
                    )
    apply_qvalues(path_entries, field="q_bh_4_indirect_path_tests")
    apply_qvalues(
        path_contrast_entries,
        field="q_bh_4_indirect_control_contrasts",
    )
    for task_name in task_specs:
        checks = {"behavior_gate": all(behavior_gates[task_name].values())}
        for direction in ("damage", "recovery"):
            indirect = path["candidate"][task_name][direction]["indirect"]
            contrast = path["candidate_vs_full_matched_controls"][task_name][
                direction
            ]
            checks[f"{direction}_indirect_effect"] = bool(
                indirect["estimate"] >= MIN_EFFECT
                and indirect["ci_low"] > 0
                and indirect["q_bh_4_indirect_path_tests"] <= MAX_Q
                and all(
                    float(value) > 0
                    for value in indirect["seed_estimates"].values()
                )
            )
            checks[f"{direction}_vs_matched_controls"] = bool(
                contrast["ci_low"] > 0
                and contrast["q_bh_4_indirect_control_contrasts"] <= MAX_Q
            )
        path["verdicts"][task_name] = {
            "checks": checks,
            "directed_head_to_mlp_mediation_confirmed": all(checks.values()),
        }

    lowrank: dict[str, Any] = {
        "structures": {},
        "signal_vs_dimension_matched_random": {},
        "verdicts": {},
        "minimum_rank_reaching_50_percent_full_mlp": {},
        "classifier_direction_cosines": {},
    }
    lowrank_entries: list[tuple[dict[str, Any], str]] = []
    lowrank_contrast_entries: list[tuple[dict[str, Any], str]] = []
    for task_index, task_name in enumerate(task_specs):
        lowrank["structures"][task_name] = {}
        lowrank["signal_vs_dimension_matched_random"][task_name] = {}
        lowrank["verdicts"][task_name] = {}
        lowrank["classifier_direction_cosines"][task_name] = {
            str(seed): float(
                seed_arrays[seed][task_name][
                    "classifier_direction_cosine"
                ]
            )
            for seed in sorted(EXPECTED_SEEDS)
        }
        for structure_index, structure_name in enumerate(
            structures[task_name]
        ):
            lowrank["structures"][task_name][structure_name] = {}
            lowrank["signal_vs_dimension_matched_random"][task_name][
                structure_name
            ] = {}
            for direction_index, direction in enumerate(
                ("damage", "recovery")
            ):
                values = average_seed_values(
                    seed_arrays,
                    task_name,
                    lambda arrays, name=structure_name, d=direction: arrays[
                        "lowrank"
                    ][name][d],
                )
                summary = summarize(
                    values,
                    groups,
                    seed=770_000
                    + 10_000 * task_index
                    + 100 * structure_index
                    + direction_index,
                )
                summary["ratio_to_full_mlp"] = grouped_bootstrap_ratio(
                    values,
                    full_mlp_values[task_name][direction],
                    groups,
                    n_boot=N_BOOT,
                    seed=771_000
                    + 10_000 * task_index
                    + 100 * structure_index
                    + direction_index,
                )
                summary["ratio_to_behavior"] = grouped_bootstrap_ratio(
                    values,
                    family_behavior[task_name],
                    groups,
                    n_boot=N_BOOT,
                    seed=772_000
                    + 10_000 * task_index
                    + 100 * structure_index
                    + direction_index,
                )
                summary["seed_estimates"] = seed_estimates(
                    seed_arrays,
                    task_name,
                    lambda arrays, name=structure_name, d=direction: arrays[
                        "lowrank"
                    ][name][d],
                )
                lowrank["structures"][task_name][structure_name][
                    direction
                ] = summary
                lowrank_entries.append(
                    (
                        summary,
                        f"{task_name}:{structure_name}:{direction}",
                    )
                )

                random_values = average_seed_values(
                    seed_arrays,
                    task_name,
                    lambda arrays, name=structure_name, d=direction: np.mean(
                        np.stack(
                            [
                                draw[d]
                                for draw in arrays["lowrank_controls"][name]
                            ],
                            axis=0,
                        ),
                        axis=0,
                    ),
                )
                contrast = summarize(
                    values - random_values,
                    groups,
                    seed=780_000
                    + 10_000 * task_index
                    + 100 * structure_index
                    + direction_index,
                )
                contrast["signal_estimate"] = float(values.mean())
                contrast["random_estimate"] = float(random_values.mean())
                contrast["random_draws"] = 3
                lowrank["signal_vs_dimension_matched_random"][task_name][
                    structure_name
                ][direction] = contrast
                lowrank_contrast_entries.append(
                    (
                        contrast,
                        f"{task_name}:{structure_name}:{direction}",
                    )
                )
    apply_qvalues(lowrank_entries, field="q_bh_20_lowrank_tests")
    apply_qvalues(
        lowrank_contrast_entries,
        field="q_bh_20_lowrank_random_contrasts",
    )

    for task_name in task_specs:
        for structure_name in structures[task_name]:
            checks = {"behavior_gate": all(behavior_gates[task_name].values())}
            for direction in ("damage", "recovery"):
                effect = lowrank["structures"][task_name][structure_name][
                    direction
                ]
                contrast = lowrank[
                    "signal_vs_dimension_matched_random"
                ][task_name][structure_name][direction]
                checks[f"{direction}_effect"] = bool(
                    effect["estimate"] >= MIN_EFFECT
                    and effect["ci_low"] > 0
                    and effect["q_bh_20_lowrank_tests"] <= MAX_Q
                    and all(
                        float(value) > 0
                        for value in effect["seed_estimates"].values()
                    )
                )
                checks[f"{direction}_vs_random"] = bool(
                    contrast["ci_low"] > 0
                    and contrast["q_bh_20_lowrank_random_contrasts"] <= MAX_Q
                )
            lowrank["verdicts"][task_name][structure_name] = {
                "checks": checks,
                "causal_structure_confirmed": all(checks.values()),
            }
        minimum_rank = None
        for rank in RANKS:
            name = f"svd_rank_{rank}"
            if (
                lowrank["verdicts"][task_name][name][
                    "causal_structure_confirmed"
                ]
                and all(
                    lowrank["structures"][task_name][name][direction][
                        "ratio_to_full_mlp"
                    ]["ci_low"]
                    >= LOWRANK_FLOOR
                    for direction in ("damage", "recovery")
                )
            ):
                minimum_rank = rank
                break
        lowrank["minimum_rank_reaching_50_percent_full_mlp"][
            task_name
        ] = minimum_rank

    raw: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        task_rows: dict[str, Any] = {}
        for task_name in task_specs:
            task_rows[task_name] = {
                "family_behavior": float(family_behavior[task_name][index]),
                "shuffle_audit": shuffle_audits[task_name][index],
                "seeds": {},
            }
            for seed in sorted(EXPECTED_SEEDS):
                arrays = seed_arrays[seed][task_name]
                task_rows[task_name]["seeds"][str(seed)] = {
                    "behavior": float(arrays["behavior"][index]),
                    "full_mlp": {
                        direction: float(
                            arrays["full_mlp"][direction][index]
                        )
                        for direction in ("damage", "recovery")
                    },
                    "heads": {
                        name: {
                            direction: float(values[direction][index])
                            for direction in ("damage", "recovery")
                        }
                        for name, values in arrays["heads"].items()
                    },
                    "matched_controls": {
                        name: {
                            direction: float(values[direction][index])
                            for direction in ("damage", "recovery")
                        }
                        for name, values in arrays["controls"].items()
                    },
                    "path_candidate": {
                        route: {
                            direction: float(
                                arrays["path_candidate"][route][direction][
                                    index
                                ]
                            )
                            for direction in ("damage", "recovery")
                        }
                        for route in ("indirect", "direct")
                    },
                    "path_controls": {
                        name: {
                            route: {
                                direction: float(
                                    values[route][direction][index]
                                )
                                for direction in ("damage", "recovery")
                            }
                            for route in ("indirect", "direct")
                        }
                        for name, values in arrays["path_controls"].items()
                    },
                    "lowrank": {
                        name: {
                            direction: float(values[direction][index])
                            for direction in ("damage", "recovery")
                        }
                        for name, values in arrays["lowrank"].items()
                    },
                    "lowrank_random": {
                        name: [
                            {
                                direction: float(values[direction][index])
                                for direction in ("damage", "recovery")
                            }
                            for values in draws
                        ]
                        for name, draws in arrays[
                            "lowrank_controls"
                        ].items()
                    },
                }
        raw.append(
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "chrom": example.chrom,
                "tasks": task_rows,
            }
        )

    strongest_claims = {
        "joint_head_set_confirmed": {
            task_name: bool(
                joint["full_set_verdicts"][task_name][
                    "joint_effect_confirmed"
                ]
            )
            for task_name in task_specs
        },
        "head_set_complete_at_80_percent": {
            task_name: bool(
                joint["full_set_verdicts"][task_name][
                    "complete_at_80_percent"
                ]
            )
            for task_name in task_specs
        },
        "directed_head_to_mlp_mediation_confirmed": {
            task_name: bool(
                path["verdicts"][task_name][
                    "directed_head_to_mlp_mediation_confirmed"
                ]
            )
            for task_name in task_specs
        },
        "task_direction_confirmed": {
            task_name: bool(
                lowrank["verdicts"][task_name]["task_direction"][
                    "causal_structure_confirmed"
                ]
            )
            for task_name in task_specs
        },
        "minimum_svd_rank_reaching_50_percent_full_mlp": lowrank[
            "minimum_rank_reaching_50_percent_full_mlp"
        ],
    }
    result = {
        "study": "native GTEx joint context and low-rank mechanism",
        "stage": "five-seed joint/path/low-rank confirmation",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "evidence_tier": "confirmatory causal intervention",
        "device": device,
        "split": "train gene-hash confirmation",
        "cohort_tag": COHORT_TAG,
        "cohort_namespace": NAMESPACE,
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "prior_exposed_genes_excluded": len(excluded_gene_ids),
        "discovery_gene_overlap": len(example_genes & discovery_gene_ids),
        "forbidden_chromosomes": sorted(FORBIDDEN_CHROMS),
        "checkpoints": checkpoint_reports,
        "regions": {
            task_name: spec["window"].name
            for task_name, spec in task_specs.items()
        },
        "frozen_heads": EXPECTED_HEADS,
        "matched_control_sets": {
            task_name: {
                str(size): [
                    head_set_name(heads)
                    for heads in values["matched_controls"][size]
                ]
                for size in (1, 2, 3)
            }
            for task_name, values in task_components.items()
        },
        "instrument_gate": instrument_gate,
        "instrument_checks": instrument_checks,
        "behavior_summaries": behavior_summaries,
        "behavior_gates": behavior_gates,
        "full_mlp_ceiling": full_mlp,
        "joint_head_patching": joint,
        "head_to_mlp_path_patching": path,
        "lowrank_mlp_write": lowrank,
        "cross_endpoint_discovery_geometry": discovery[
            "cross_endpoint_geometry"
        ],
        "strongest_claims": strongest_claims,
        "multiplicity_families": {
            "behavior": 2,
            "full_mlp": 4,
            "joint_subset_effects": 28,
            "joint_candidate_control_contrasts": 28,
            "indirect_path_effects": 4,
            "indirect_path_control_contrasts": 4,
            "lowrank_effects": 20,
            "lowrank_random_contrasts": 20,
        },
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default)
        + "\n"
    )
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:"
        "native_context_joint_lowrank_confirmation_v1",
        sys.argv,
        repo=REPO,
        status="CONFIRMATORY",
        config={
            "split": result["split"],
            "n": args.n,
            "batch": args.batch,
            "device": device,
            "cohort_tag": COHORT_TAG,
            "cohort_namespace": NAMESPACE,
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "ranks": list(RANKS),
            "random_draws_per_structure": 3,
            "frozen_heads": EXPECTED_HEADS,
            "multiplicity_families": result["multiplicity_families"],
        },
        seeds={
            "checkpoints": sorted(EXPECTED_SEEDS),
            "behavior_summary_base": 700_000,
            "random_basis_tag": random_tag,
        },
        notes=[
            "Candidate head sets and all low-rank bases were frozen before this run.",
            "Head controls are matched within intervention size and never pooled.",
            "Low-rank controls are three deterministic Haar-random subspaces per dimension.",
            "The final-layer composition identity was checked against literal hook clamping.",
            "Total, indirect, and direct path effects are not assumed additive under LayerNorm.",
            "The confirmation genes are disjoint from every earlier analysis artifact.",
            "The split is analysis-sealed, not claimed absent from checkpoint training.",
        ],
    )
    for path_value in (
        protocol,
        discovery_path,
        census_path,
        *exclusion_paths,
        model_weight(base_model),
        base_model / "config.json",
        base_model / "tokenizer.json",
        *checkpoints.values(),
        Path(args.gtf),
        Path(args.fasta),
    ):
        manifest.inputs.append(file_record(path_value))
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(json.dumps(strongest_claims, indent=2, sort_keys=True))
    print(f"COMPLETE: saved -> {output}")


if __name__ == "__main__":
    main()
