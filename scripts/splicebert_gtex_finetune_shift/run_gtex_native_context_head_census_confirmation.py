#!/usr/bin/env python
"""Five-seed confirmation of train-frozen L5 context-head candidates."""

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
    replay_final_layer,
)


EXPECTED_SEEDS = frozenset(range(2020, 2025))
N_CONFIRMATION = 200
N_BOOT = 5000
N_PERM = 9999
FINAL_LAYER = 5
NUM_HEADS = 16
MIN_EFFECT = 0.05
MAX_Q = 0.05
NAMESPACE = "gtex-native-context-head-census-v1"
COHORT_TAG = "gtex-native-context-head-census-confirmation-v1"
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
    if not entries:
        return
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
    genes = {
        str(row["gene_id"])
        for row in artifact.get("raw", [])
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
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--mechanism-preflight", required=True)
    parser.add_argument("--mechanism-discovery", required=True)
    parser.add_argument("--context-preflight", required=True)
    parser.add_argument("--context-discovery", required=True)
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
        raise ValueError(
            f"head-census confirmation is frozen at n={N_CONFIRMATION}"
        )
    if args.n_boot != N_BOOT or args.n_perm != N_PERM:
        raise ValueError(
            f"inference is frozen at {N_BOOT} bootstraps/{N_PERM} sign flips"
        )
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CONTEXT_HEAD_CENSUS_V1.md"
    )
    discovery_path = Path(args.discovery).resolve()
    discovery = json.loads(discovery_path.read_text())
    if discovery.get("status") != "COMPLETE":
        raise RuntimeError("head-census discovery is incomplete")
    if discovery.get("multiplicity_family") != 64:
        raise RuntimeError("discovery did not scan the complete 64-test family")
    if discovery.get("cohort_namespace") != NAMESPACE:
        raise RuntimeError("discovery cohort namespace does not match protocol")

    prior_paths = {
        "mechanism_preflight": Path(args.mechanism_preflight).resolve(),
        "mechanism_discovery": Path(args.mechanism_discovery).resolve(),
        "context_preflight": Path(args.context_preflight).resolve(),
        "context_discovery": Path(args.context_discovery).resolve(),
    }
    prior_artifacts = {
        name: json.loads(path.read_text())
        for name, path in prior_paths.items()
    }
    prior_gene_ids = set()
    for artifact in prior_artifacts.values():
        prior_gene_ids.update(raw_gene_ids(artifact))
    if len(prior_gene_ids) != int(
        discovery["prior_exposed_genes_excluded"]
    ):
        raise RuntimeError("prior-exposed gene freeze does not replay discovery")

    freeze = discovery.get("freeze", {})
    task_components: dict[str, dict[str, Any]] = {}
    for task_name in ("donor", "acceptor"):
        task_freeze = freeze.get(task_name, {})
        candidates = [
            head_number(name)
            for name in task_freeze.get("candidate_heads", [])
        ]
        controls = {
            head_number(candidate): [
                head_number(name) for name in names
            ]
            for candidate, names in task_freeze.get(
                "control_heads",
                {},
            ).items()
        }
        if bool(candidates) != bool(
            task_freeze.get("confirmation_authorized")
        ):
            raise RuntimeError(f"{task_name} confirmation authorization mismatch")
        if set(candidates) != set(controls):
            raise RuntimeError(f"{task_name} candidate/control freeze incomplete")
        flat_controls = [
            head for values in controls.values() for head in values
        ]
        if len(flat_controls) != len(set(flat_controls)):
            raise RuntimeError(f"{task_name} controls are not unique")
        if set(candidates) & set(flat_controls):
            raise RuntimeError(f"{task_name} candidate appears as a control")
        if any(len(values) != 3 for values in controls.values()):
            raise RuntimeError(f"{task_name} does not have three controls/candidate")
        task_components[task_name] = {
            "candidates": candidates,
            "controls": controls,
            "tested_heads": sorted(set(candidates) | set(flat_controls)),
        }
    if not any(
        values["candidates"] for values in task_components.values()
    ):
        raise RuntimeError("discovery authorized no confirmation endpoint")

    base_model = Path(args.base_model).expanduser().resolve()
    checkpoints = parse_checkpoints(args.checkpoint)
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
        n=args.n,
        tag=COHORT_TAG,
        excluded_gene_ids=sorted(prior_gene_ids),
    )
    discovery_genes = {
        str(row["gene_id"]) for row in discovery.get("raw", [])
    }
    confirmation_genes = {example.gene_id for example in examples}
    if confirmation_genes & discovery_genes:
        raise RuntimeError("confirmation and discovery genes overlap")
    if confirmation_genes & prior_gene_ids:
        raise RuntimeError("confirmation contains a prior-exposed gene")
    normalized_chroms = {
        chrom if chrom.startswith("chr") else f"chr{chrom}"
        for chrom in (example.chrom for example in examples)
    }
    if normalized_chroms & FORBIDDEN_CHROMS:
        raise RuntimeError("confirmation opened a forbidden chromosome")
    groups = np.asarray([example.gene_id for example in examples], dtype=object)
    clean_sequences = [example.sequence for example in examples]

    task_specs: dict[str, dict[str, Any]] = {
        "donor": {
            "anchors": [int(example.donor_pos) for example in examples],
            "targets": [int(example.donor_pos) - 1 for example in examples],
            "label": DONOR_LABEL,
            "window": parse_window(discovery["regions"]["donor"]),
        },
        "acceptor": {
            "anchors": [
                int(example.next_acceptor_pos) for example in examples
            ],
            "targets": [
                int(example.next_acceptor_pos) for example in examples
            ],
            "label": ACCEPTOR_LABEL,
            "window": parse_window(discovery["regions"]["acceptor"]),
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
                    "head-census-v1",
                ),
                draw=0,
            )
            if audit.hamming_fraction < 0.25:
                raise RuntimeError("head-census shuffle failed dose floor")
            corrupt_sequences[task_name].append(shuffled)
            shuffle_audits[task_name].append(audit.to_dict())

    seed_arrays: dict[int, dict[str, Any]] = {}
    checkpoint_reports: dict[int, dict[str, Any]] = {}
    for seed, checkpoint in sorted(checkpoints.items()):
        model, tokenizer, report = load_gtex_token_classifier(
            base_model,
            checkpoint,
            device=device,
            attn_implementation="eager",
        )
        if report.seed != seed:
            raise RuntimeError(
                f"checkpoint key {seed} loaded reported seed {report.seed}"
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
            source=str(checkpoint),
            device=device,
        )
        if adapter.num_layers != 6 or adapter.num_heads != NUM_HEADS:
            raise RuntimeError("frozen architecture is 6 layers x 16 heads")
        seed_arrays[seed] = {}
        for task_index, (task_name, spec) in enumerate(
            task_specs.items(),
            start=1,
        ):
            tested_heads = task_components[task_name]["tested_heads"]
            arrays: dict[str, Any] = {
                "behavior": [],
                "heads": {
                    head: {"damage": [], "recovery": []}
                    for head in tested_heads
                },
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
                clean_source = {FINAL_LAYER: clean_run.heads}
                corrupt_source = {FINAL_LAYER: corrupt_run.heads}
                for head in tested_heads:
                    with patch_heads(
                        adapter,
                        corrupt_source,
                        [(FINAL_LAYER, head)],
                        token_mask=target_mask,
                    ):
                        noised_hidden = replay_final_layer(
                            adapter,
                            clean_inputs,
                            layer=FINAL_LAYER,
                        )
                    noised = classifier_margin(
                        model,
                        noised_hidden,
                        target_tokens,
                        label=int(spec["label"]),
                    )
                    arrays["heads"][head]["damage"].extend(
                        (clean_margin - noised).detach().cpu().tolist()
                    )
                    with patch_heads(
                        adapter,
                        clean_source,
                        [(FINAL_LAYER, head)],
                        token_mask=target_mask,
                    ):
                        denoised_hidden = replay_final_layer(
                            adapter,
                            corrupt_inputs,
                            layer=FINAL_LAYER,
                        )
                    denoised = classifier_margin(
                        model,
                        denoised_hidden,
                        target_tokens,
                        label=int(spec["label"]),
                    )
                    arrays["heads"][head]["recovery"].extend(
                        (denoised - corrupt_margin).detach().cpu().tolist()
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
            arrays["behavior"] = np.asarray(
                arrays["behavior"],
                dtype=np.float64,
            )
            for head in tested_heads:
                for direction in ("damage", "recovery"):
                    arrays["heads"][head][direction] = np.asarray(
                        arrays["heads"][head][direction],
                        dtype=np.float64,
                    )
            seed_arrays[seed][task_name] = arrays
        checkpoint_reports[seed] = report.to_dict()
        del task, adapter, model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    family_behavior: dict[str, np.ndarray] = {}
    behavior_summaries: dict[str, dict[str, Any]] = {}
    behavior_entries: list[tuple[dict[str, Any], str]] = []
    for task_index, task_name in enumerate(task_specs):
        values = np.mean(
            np.stack(
                [
                    seed_arrays[seed][task_name]["behavior"]
                    for seed in sorted(EXPECTED_SEEDS)
                ],
                axis=0,
            ),
            axis=0,
        )
        family_behavior[task_name] = values
        result = summarize(values, groups, seed=400_000 + task_index)
        result["seed_estimates"] = {
            str(seed): float(
                np.mean(seed_arrays[seed][task_name]["behavior"])
            )
            for seed in sorted(EXPECTED_SEEDS)
        }
        behavior_summaries[task_name] = result
        behavior_entries.append((result, task_name))
    apply_qvalues(behavior_entries, field="q_bh_2_behavior_tests")

    family: dict[str, Any] = {
        "heads": {},
        "candidate_vs_controls": {},
        "tested_head_descriptives": {},
    }
    primary_entries: list[tuple[dict[str, Any], str]] = []
    control_entries: list[tuple[dict[str, Any], str]] = []
    for task_index, task_name in enumerate(task_specs):
        family["heads"][task_name] = {}
        family["candidate_vs_controls"][task_name] = {}
        family["tested_head_descriptives"][task_name] = {}
        components = task_components[task_name]
        for head in components["tested_heads"]:
            head_name = f"L{FINAL_LAYER}H{head}"
            family["tested_head_descriptives"][task_name][head_name] = {}
            for direction_index, direction in enumerate(
                ("damage", "recovery")
            ):
                values = np.mean(
                    np.stack(
                        [
                            seed_arrays[seed][task_name]["heads"][head][
                                direction
                            ]
                            for seed in sorted(EXPECTED_SEEDS)
                        ],
                        axis=0,
                    ),
                    axis=0,
                )
                descriptive = summarize(
                    values,
                    groups,
                    seed=410_000
                    + 10_000 * task_index
                    + 100 * head
                    + direction_index,
                    infer=False,
                )
                descriptive["ratio_to_behavior"] = grouped_bootstrap_ratio(
                    values,
                    family_behavior[task_name],
                    groups,
                    n_boot=N_BOOT,
                    seed=430_000
                    + 10_000 * task_index
                    + 100 * head
                    + direction_index,
                )
                descriptive["seed_estimates"] = {
                    str(seed): float(
                        np.mean(
                            seed_arrays[seed][task_name]["heads"][head][
                                direction
                            ]
                        )
                    )
                    for seed in sorted(EXPECTED_SEEDS)
                }
                family["tested_head_descriptives"][task_name][head_name][
                    direction
                ] = descriptive

        for candidate in components["candidates"]:
            candidate_name = f"L{FINAL_LAYER}H{candidate}"
            family["heads"][task_name][candidate_name] = {}
            family["candidate_vs_controls"][task_name][candidate_name] = {}
            for direction_index, direction in enumerate(
                ("damage", "recovery")
            ):
                candidate_values = np.mean(
                    np.stack(
                        [
                            seed_arrays[seed][task_name]["heads"][candidate][
                                direction
                            ]
                            for seed in sorted(EXPECTED_SEEDS)
                        ],
                        axis=0,
                    ),
                    axis=0,
                )
                candidate_summary = summarize(
                    candidate_values,
                    groups,
                    seed=500_000
                    + 10_000 * task_index
                    + 100 * candidate
                    + direction_index,
                )
                candidate_summary["ratio_to_behavior"] = (
                    grouped_bootstrap_ratio(
                        candidate_values,
                        family_behavior[task_name],
                        groups,
                        n_boot=N_BOOT,
                        seed=510_000
                        + 10_000 * task_index
                        + 100 * candidate
                        + direction_index,
                    )
                )
                candidate_summary["seed_estimates"] = family[
                    "tested_head_descriptives"
                ][task_name][candidate_name][direction]["seed_estimates"]
                family["heads"][task_name][candidate_name][
                    direction
                ] = candidate_summary
                primary_entries.append(
                    (
                        candidate_summary,
                        f"{task_name}:{candidate_name}:{direction}",
                    )
                )

                control_values = np.mean(
                    np.stack(
                        [
                            seed_arrays[seed][task_name]["heads"][head][
                                direction
                            ]
                            for seed in sorted(EXPECTED_SEEDS)
                            for head in components["controls"][candidate]
                        ],
                        axis=0,
                    ),
                    axis=0,
                )
                contrast = candidate_values - control_values
                contrast_summary = summarize(
                    contrast,
                    groups,
                    seed=520_000
                    + 10_000 * task_index
                    + 100 * candidate
                    + direction_index,
                )
                family["candidate_vs_controls"][task_name][candidate_name][
                    direction
                ] = contrast_summary
                control_entries.append(
                    (
                        contrast_summary,
                        f"{task_name}:{candidate_name}:{direction}",
                    )
                )

    primary_q_field = f"q_bh_{len(primary_entries)}_candidate_tests"
    control_q_field = f"q_bh_{len(control_entries)}_control_contrasts"
    apply_qvalues(primary_entries, field=primary_q_field)
    apply_qvalues(control_entries, field=control_q_field)

    behavior_gates = {
        task_name: {
            "mean_at_least_0_05": bool(
                behavior_summaries[task_name]["estimate"] >= MIN_EFFECT
            ),
            "ci_above_zero": bool(
                behavior_summaries[task_name]["ci_low"] > 0
            ),
            "q_at_most_0_05": bool(
                behavior_summaries[task_name]["q_bh_2_behavior_tests"] <= MAX_Q
            ),
            "positive_5_of_5": all(
                float(value) > 0
                for value in behavior_summaries[task_name][
                    "seed_estimates"
                ].values()
            ),
        }
        for task_name in task_specs
    }
    verdicts: dict[str, dict[str, Any]] = {}
    confirmed_by_endpoint: dict[str, list[str]] = {}
    for task_name in task_specs:
        verdicts[task_name] = {}
        confirmed_by_endpoint[task_name] = []
        for candidate in task_components[task_name]["candidates"]:
            candidate_name = f"L{FINAL_LAYER}H{candidate}"
            checks = {"behavior_gate": all(behavior_gates[task_name].values())}
            for direction in ("damage", "recovery"):
                effect = family["heads"][task_name][candidate_name][direction]
                contrast = family["candidate_vs_controls"][task_name][
                    candidate_name
                ][direction]
                checks[f"{direction}_effect"] = bool(
                    effect["estimate"] >= MIN_EFFECT
                    and effect["ci_low"] > 0
                    and effect[primary_q_field] <= MAX_Q
                    and all(
                        float(value) > 0
                        for value in effect["seed_estimates"].values()
                    )
                )
                checks[f"{direction}_vs_controls"] = bool(
                    contrast["ci_low"] > 0
                    and contrast[control_q_field] <= MAX_Q
                )
            passed = all(checks.values())
            verdicts[task_name][candidate_name] = {
                "checks": checks,
                "PASS": passed,
            }
            if passed:
                confirmed_by_endpoint[task_name].append(candidate_name)

    raw = []
    for index, example in enumerate(examples):
        task_rows = {}
        for task_name in task_specs:
            tested_heads = task_components[task_name]["tested_heads"]
            task_rows[task_name] = {
                "family_behavior": float(family_behavior[task_name][index]),
                "shuffle_audit": shuffle_audits[task_name][index],
                "seeds": {
                    str(seed): {
                        "behavior": float(
                            seed_arrays[seed][task_name]["behavior"][index]
                        ),
                        "heads": {
                            f"L{FINAL_LAYER}H{head}": {
                                direction: float(
                                    seed_arrays[seed][task_name]["heads"][head][
                                        direction
                                    ][index]
                                )
                                for direction in ("damage", "recovery")
                            }
                            for head in tested_heads
                        },
                    }
                    for seed in sorted(EXPECTED_SEEDS)
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

    result = {
        "study": "native GTEx context head census",
        "stage": "fresh five-seed candidate confirmation",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "evidence_tier": "confirmatory on fresh internal gene split",
        "device": device,
        "split": "train gene-hash confirmation",
        "cohort_tag": COHORT_TAG,
        "cohort_namespace": NAMESPACE,
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "prior_exposed_genes_excluded": len(prior_gene_ids),
        "discovery_confirmation_gene_overlap": 0,
        "forbidden_chromosomes": sorted(FORBIDDEN_CHROMS),
        "family_seeds": sorted(EXPECTED_SEEDS),
        "checkpoints": checkpoint_reports,
        "regions": {
            task_name: spec["window"].name
            for task_name, spec in task_specs.items()
        },
        "freeze": freeze,
        "behavior_summaries": behavior_summaries,
        "behavior_gates": behavior_gates,
        "family": family,
        "multiplicity": {
            "behavior_tests": len(behavior_entries),
            "candidate_tests": len(primary_entries),
            "control_contrasts": len(control_entries),
        },
        "verdicts": verdicts,
        "confirmed_heads": confirmed_by_endpoint,
        "thresholds": {
            "min_effect": MIN_EFFECT,
            "max_q": MAX_Q,
            "positive_seed_requirement": "5_of_5",
            "candidate_control_min_effect": None,
        },
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default)
        + "\n"
    )
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_head_census_confirmation_v1",
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
            "checkpoint_seeds": sorted(EXPECTED_SEEDS),
            "candidate_heads": {
                task_name: [
                    f"L{FINAL_LAYER}H{head}"
                    for head in components["candidates"]
                ]
                for task_name, components in task_components.items()
            },
            "control_heads": {
                task_name: {
                    f"L{FINAL_LAYER}H{candidate}": [
                        f"L{FINAL_LAYER}H{head}" for head in controls
                    ]
                    for candidate, controls in components["controls"].items()
                }
                for task_name, components in task_components.items()
            },
            "multiplicity": result["multiplicity"],
            "thresholds": result["thresholds"],
        },
        notes=[
            "Candidates and controls replay the immutable train discovery freeze.",
            "All five checkpoints use one fresh, gene-disjoint internal confirmation cohort.",
            "Prior train-exposed genes and all validation/test chromosomes are excluded.",
            "The fresh split is analysis-sealed, not claimed absent from checkpoint training.",
        ],
    )
    for path in (
        protocol,
        discovery_path,
        *prior_paths.values(),
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
                "behavior_gates": behavior_gates,
                "confirmed_heads": confirmed_by_endpoint,
                "verdicts": verdicts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"COMPLETE: saved -> {output}")


if __name__ == "__main__":
    main()
