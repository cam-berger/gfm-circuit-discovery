#!/usr/bin/env python
"""Test the frozen L5H6/final-MLP route on confirmed context-region effects."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
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
    load_annotation_cohort,
)
from gfmcd.splicing.native_mechanism import (  # noqa: E402
    cache_final_layer_inputs,
    capture_final_layer_run,
    donor_token_positions,
    position_mask,
    replay_final_layer,
)


EXPECTED_SEEDS = frozenset(range(2020, 2025))
N = 200
N_BOOT = 5000
N_PERM = 9999
FINAL_LAYER = 5
CANDIDATE_HEAD = 6
CONTROL_HEADS = (7, 11, 9)
MIN_HEAD_EFFECT = 0.05
MIN_CONTROL_CONTRAST = 0.05
MIN_MLP_FRACTION = 0.50
MAX_Q = 0.05
NUMERIC_ATOL = 2e-5


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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-preflight", required=True)
    parser.add_argument("--mechanism-confirmation", required=True)
    parser.add_argument("--context-discovery", required=True)
    parser.add_argument("--context-confirmation", required=True)
    parser.add_argument("--dose-audit", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != N:
        raise ValueError(f"context component bridge is frozen at n={N}")
    if args.n_boot != N_BOOT or args.n_perm != N_PERM:
        raise ValueError(
            f"inference is frozen at {N_BOOT} bootstraps/{N_PERM} sign flips"
        )
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CONTEXT_COMPONENT_BRIDGE_V1.md"
    )
    mechanism_preflight_path = Path(
        args.mechanism_preflight
    ).expanduser().resolve()
    mechanism_confirmation_path = Path(
        args.mechanism_confirmation
    ).expanduser().resolve()
    discovery_path = Path(args.context_discovery).expanduser().resolve()
    confirmation_path = Path(
        args.context_confirmation
    ).expanduser().resolve()
    audit_path = Path(args.dose_audit).expanduser().resolve()
    mechanism_preflight = json.loads(mechanism_preflight_path.read_text())
    mechanism_confirmation = json.loads(
        mechanism_confirmation_path.read_text()
    )
    discovery = json.loads(discovery_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    dose_audit = json.loads(audit_path.read_text())
    if mechanism_preflight.get("status") != "PASS":
        raise RuntimeError("native mechanism preflight did not pass")
    frozen_controls = mechanism_confirmation["freeze"]["control_heads"][
        "L5H6"
    ]
    if frozen_controls != ["L5H7", "L5H11", "L5H9"]:
        raise RuntimeError("frozen L5H6 controls do not match bridge protocol")
    for task_name in ("donor", "acceptor"):
        if not dose_audit["licenses"][task_name][
            "confirmed_32nt_region_licensed"
        ]:
            raise RuntimeError(f"{task_name} context region is unlicensed")
    checkpoints = parse_checkpoints(args.checkpoint)
    base_model = Path(args.base_model).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)

    examples = load_annotation_cohort(
        args.gtf,
        args.fasta,
        split="val",
        n=args.n,
        tag="gtex-native-context-localization-confirmation-v1",
    )
    if [example.example_id for example in examples] != [
        row["example_id"] for row in confirmation["raw"]
    ]:
        raise RuntimeError("component bridge cohort does not replay confirmation")
    groups = np.asarray([example.gene_id for example in examples], dtype=object)
    clean_sequences = [example.sequence for example in examples]
    task_specs: dict[str, dict[str, Any]] = {
        "donor": {
            "anchors": [int(example.donor_pos) for example in examples],
            "targets": [int(example.donor_pos) - 1 for example in examples],
            "label": DONOR_LABEL,
            "window": parse_window(
                discovery["freeze"]["donor"]["distal_candidate"]
            ),
        },
        "acceptor": {
            "anchors": [
                int(example.next_acceptor_pos) for example in examples
            ],
            "targets": [
                int(example.next_acceptor_pos) for example in examples
            ],
            "label": ACCEPTOR_LABEL,
            "window": parse_window(
                discovery["freeze"]["acceptor"]["distal_candidate"]
            ),
        },
    }
    corrupt_sequences: dict[str, list[str]] = {}
    for task_name, spec in task_specs.items():
        corrupt_sequences[task_name] = []
        for example, anchor in zip(examples, spec["anchors"]):
            shuffled, audit = shuffle_relative_window(
                example.sequence,
                anchor=int(anchor),
                window=spec["window"],
                key=(
                    example.example_id,
                    task_name,
                    spec["window"].name,
                    "confirmation",
                ),
                draw=0,
            )
            if audit.hamming_fraction < 0.25:
                raise RuntimeError("candidate intervention failed dose replay")
            corrupt_sequences[task_name].append(shuffled)

    heads = (CANDIDATE_HEAD, *CONTROL_HEADS)
    seed_values: dict[int, dict[str, Any]] = {}
    checkpoint_reports: dict[int, dict[str, Any]] = {}
    replay_errors: dict[int, dict[str, float]] = {}
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
        seed_values[seed] = {}
        replay_errors[seed] = {}
        for task_name, spec in task_specs.items():
            arrays: dict[str, Any] = {
                "behavior": [],
                "heads": {
                    head: {"damage": [], "recovery": []}
                    for head in heads
                },
                "mlp": {"damage": [], "recovery": []},
                "controlled_direct_l5h6_damage": [],
            }
            for start in range(0, len(examples), args.batch):
                stop = min(start + args.batch, len(examples))
                clean_encoded = task.encode(clean_sequences[start:stop])
                corrupt_encoded = task.encode(
                    corrupt_sequences[task_name][start:stop]
                )
                chunk_targets = spec["targets"][start:stop]
                tokens = donor_token_positions(
                    chunk_targets,
                    prefix_tokens=task.prefix_tokens,
                    device=device,
                )
                token_mask = position_mask(
                    tokens,
                    int(clean_encoded["input_ids"].shape[1]),
                )
                token_positions_2d = tokens[:, None]
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
                    tokens,
                    label=int(spec["label"]),
                )
                corrupt_margin = classifier_margin(
                    model,
                    corrupt_run.hidden,
                    tokens,
                    label=int(spec["label"]),
                )
                arrays["behavior"].extend(
                    (clean_margin - corrupt_margin).detach().cpu().tolist()
                )
                clean_head_source = {FINAL_LAYER: clean_run.heads}
                corrupt_head_source = {FINAL_LAYER: corrupt_run.heads}
                for head in heads:
                    with patch_heads(
                        adapter,
                        corrupt_head_source,
                        [(FINAL_LAYER, head)],
                        token_mask=token_mask,
                    ):
                        noised_hidden = replay_final_layer(
                            adapter,
                            clean_inputs,
                            layer=FINAL_LAYER,
                        )
                    noised = classifier_margin(
                        model,
                        noised_hidden,
                        tokens,
                        label=int(spec["label"]),
                    )
                    arrays["heads"][head]["damage"].extend(
                        (clean_margin - noised).detach().cpu().tolist()
                    )
                    with patch_heads(
                        adapter,
                        clean_head_source,
                        [(FINAL_LAYER, head)],
                        token_mask=token_mask,
                    ):
                        denoised_hidden = replay_final_layer(
                            adapter,
                            corrupt_inputs,
                            layer=FINAL_LAYER,
                        )
                    denoised = classifier_margin(
                        model,
                        denoised_hidden,
                        tokens,
                        label=int(spec["label"]),
                    )
                    arrays["heads"][head]["recovery"].extend(
                        (denoised - corrupt_margin).detach().cpu().tolist()
                    )

                with clamp_write(
                    adapter,
                    [(FINAL_LAYER, "mlp")],
                    {(FINAL_LAYER, "mlp"): corrupt_run.mlp_write},
                    positions=token_positions_2d,
                ):
                    noised_mlp_hidden = replay_final_layer(
                        adapter,
                        clean_inputs,
                        layer=FINAL_LAYER,
                    )
                noised_mlp = classifier_margin(
                    model,
                    noised_mlp_hidden,
                    tokens,
                    label=int(spec["label"]),
                )
                arrays["mlp"]["damage"].extend(
                    (clean_margin - noised_mlp).detach().cpu().tolist()
                )
                with clamp_write(
                    adapter,
                    [(FINAL_LAYER, "mlp")],
                    {(FINAL_LAYER, "mlp"): clean_run.mlp_write},
                    positions=token_positions_2d,
                ):
                    denoised_mlp_hidden = replay_final_layer(
                        adapter,
                        corrupt_inputs,
                        layer=FINAL_LAYER,
                    )
                denoised_mlp = classifier_margin(
                    model,
                    denoised_mlp_hidden,
                    tokens,
                    label=int(spec["label"]),
                )
                arrays["mlp"]["recovery"].extend(
                    (denoised_mlp - corrupt_margin).detach().cpu().tolist()
                )

                with ExitStack() as stack:
                    stack.enter_context(
                        patch_heads(
                            adapter,
                            corrupt_head_source,
                            [(FINAL_LAYER, CANDIDATE_HEAD)],
                            token_mask=token_mask,
                        )
                    )
                    stack.enter_context(
                        clamp_write(
                            adapter,
                            [(FINAL_LAYER, "mlp")],
                            {(FINAL_LAYER, "mlp"): clean_run.mlp_write},
                            positions=token_positions_2d,
                        )
                    )
                    direct_hidden = replay_final_layer(
                        adapter,
                        clean_inputs,
                        layer=FINAL_LAYER,
                    )
                direct_margin = classifier_margin(
                    model,
                    direct_hidden,
                    tokens,
                    label=int(spec["label"]),
                )
                arrays["controlled_direct_l5h6_damage"].extend(
                    (clean_margin - direct_margin).detach().cpu().tolist()
                )

            arrays["behavior"] = np.asarray(
                arrays["behavior"],
                dtype=np.float64,
            )
            for head in heads:
                for direction in ("damage", "recovery"):
                    arrays["heads"][head][direction] = np.asarray(
                        arrays["heads"][head][direction],
                        dtype=np.float64,
                    )
            for direction in ("damage", "recovery"):
                arrays["mlp"][direction] = np.asarray(
                    arrays["mlp"][direction],
                    dtype=np.float64,
                )
            arrays["controlled_direct_l5h6_damage"] = np.asarray(
                arrays["controlled_direct_l5h6_damage"],
                dtype=np.float64,
            )
            arrays["l5h6_minus_control"] = {
                direction: arrays["heads"][CANDIDATE_HEAD][direction]
                - np.mean(
                    np.stack(
                        [
                            arrays["heads"][head][direction]
                            for head in CONTROL_HEADS
                        ],
                        axis=0,
                    ),
                    axis=0,
                )
                for direction in ("damage", "recovery")
            }
            arrays["mlp_dependent_l5h6_damage"] = (
                arrays["heads"][CANDIDATE_HEAD]["damage"]
                - arrays["controlled_direct_l5h6_damage"]
            )
            expected = np.asarray(
                [
                    confirmation["raw"][index]["seeds"][str(seed)][
                        task_name
                    ]["candidate_draw0"]
                    for index in range(args.n)
                ],
                dtype=np.float64,
            )
            replay_errors[seed][task_name] = float(
                np.max(np.abs(arrays["behavior"] - expected))
            )
            seed_values[seed][task_name] = arrays
        checkpoint_reports[seed] = report.to_dict()
        del task, adapter, model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        print(f"checkpoint {seed} complete", flush=True)

    if any(
        error > NUMERIC_ATOL
        for task_errors in replay_errors.values()
        for error in task_errors.values()
    ):
        raise RuntimeError("component bridge did not replay context confirmation")

    family_behavior: dict[str, np.ndarray] = {}
    family_metrics: dict[str, dict[str, np.ndarray]] = {}
    metric_names = (
        *(f"L5H{head}_{direction}" for head in heads for direction in ("damage", "recovery")),
        "l5h6_minus_control_damage",
        "l5h6_minus_control_recovery",
        "mlp_damage",
        "mlp_recovery",
        "controlled_direct_l5h6_damage",
        "mlp_dependent_l5h6_damage",
    )
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    pvalue_entries: list[tuple[str, str, float]] = []
    summary_seed = 220_000
    for task_name in task_specs:
        family_behavior[task_name] = np.mean(
            np.stack(
                [
                    seed_values[seed][task_name]["behavior"]
                    for seed in sorted(EXPECTED_SEEDS)
                ],
                axis=0,
            ),
            axis=0,
        )
        per_seed_metrics: dict[int, dict[str, np.ndarray]] = {}
        for seed in sorted(EXPECTED_SEEDS):
            arrays = seed_values[seed][task_name]
            metrics = {
                f"L5H{head}_{direction}": arrays["heads"][head][direction]
                for head in heads
                for direction in ("damage", "recovery")
            }
            metrics.update(
                {
                    f"l5h6_minus_control_{direction}": arrays[
                        "l5h6_minus_control"
                    ][direction]
                    for direction in ("damage", "recovery")
                }
            )
            metrics.update(
                {
                    f"mlp_{direction}": arrays["mlp"][direction]
                    for direction in ("damage", "recovery")
                }
            )
            metrics["controlled_direct_l5h6_damage"] = arrays[
                "controlled_direct_l5h6_damage"
            ]
            metrics["mlp_dependent_l5h6_damage"] = arrays[
                "mlp_dependent_l5h6_damage"
            ]
            per_seed_metrics[seed] = metrics
        family_metrics[task_name] = {}
        summaries[task_name] = {}
        for metric_name in metric_names:
            values = np.mean(
                np.stack(
                    [
                        per_seed_metrics[seed][metric_name]
                        for seed in sorted(EXPECTED_SEEDS)
                    ],
                    axis=0,
                ),
                axis=0,
            )
            family_metrics[task_name][metric_name] = values
            summary = grouped_bootstrap(
                values,
                groups,
                n_boot=N_BOOT,
                seed=summary_seed,
            )
            summary["p_signflip_add_one"] = grouped_signflip_pvalue(
                values,
                groups,
                n_perm=N_PERM,
                seed=summary_seed + 100_000,
            )
            summary["ratio_to_behavior"] = grouped_bootstrap_ratio(
                values,
                family_behavior[task_name],
                groups,
                n_boot=N_BOOT,
                seed=summary_seed + 10_000,
            )
            summary["seed_estimates"] = {
                str(seed): float(
                    np.mean(per_seed_metrics[seed][metric_name])
                )
                for seed in sorted(EXPECTED_SEEDS)
            }
            summaries[task_name][metric_name] = summary
            pvalue_entries.append(
                (
                    task_name,
                    metric_name,
                    float(summary["p_signflip_add_one"]),
                )
            )
            summary_seed += 1

    if len(pvalue_entries) != 28:
        raise RuntimeError(
            f"frozen multiplicity family is 28, found {len(pvalue_entries)}"
        )
    qvalues = benjamini_hochberg([entry[2] for entry in pvalue_entries])
    for (task_name, metric_name, _), qvalue in zip(
        pvalue_entries,
        qvalues,
    ):
        summaries[task_name][metric_name]["q_bh_28_tests"] = float(qvalue)

    decisions: dict[str, Any] = {}
    for task_name in task_specs:
        damage = summaries[task_name]["L5H6_damage"]
        recovery = summaries[task_name]["L5H6_recovery"]
        damage_contrast = summaries[task_name][
            "l5h6_minus_control_damage"
        ]
        recovery_contrast = summaries[task_name][
            "l5h6_minus_control_recovery"
        ]
        head_gates = {
            "damage_effect": float(damage["estimate"]) >= MIN_HEAD_EFFECT,
            "recovery_effect": float(recovery["estimate"]) >= MIN_HEAD_EFFECT,
            "damage_ci_q": (
                float(damage["ci_low"]) > 0
                and float(damage["q_bh_28_tests"]) <= MAX_Q
            ),
            "recovery_ci_q": (
                float(recovery["ci_low"]) > 0
                and float(recovery["q_bh_28_tests"]) <= MAX_Q
            ),
            "damage_positive_5_of_5": all(
                float(value) > 0
                for value in damage["seed_estimates"].values()
            ),
            "recovery_positive_5_of_5": all(
                float(value) > 0
                for value in recovery["seed_estimates"].values()
            ),
            "damage_control_contrast": (
                float(damage_contrast["estimate"])
                >= MIN_CONTROL_CONTRAST
                and float(damage_contrast["ci_low"]) > 0
                and float(damage_contrast["q_bh_28_tests"]) <= MAX_Q
            ),
            "recovery_control_contrast": (
                float(recovery_contrast["estimate"])
                >= MIN_CONTROL_CONTRAST
                and float(recovery_contrast["ci_low"]) > 0
                and float(recovery_contrast["q_bh_28_tests"]) <= MAX_Q
            ),
        }
        mlp_gates = {}
        for direction in ("damage", "recovery"):
            metric = summaries[task_name][f"mlp_{direction}"]
            mlp_gates[direction] = (
                float(metric["ci_low"]) > 0
                and float(metric["q_bh_28_tests"]) <= MAX_Q
                and float(metric["ratio_to_behavior"]["estimate"])
                >= MIN_MLP_FRACTION
                and all(
                    float(value) > 0
                    for value in metric["seed_estimates"].values()
                )
            )
        interaction = summaries[task_name]["mlp_dependent_l5h6_damage"]
        decisions[task_name] = {
            "l5h6_transfers": all(head_gates.values()),
            "l5h6_gates": head_gates,
            "final_mlp_transfers": all(mlp_gates.values()),
            "final_mlp_gates": mlp_gates,
            "l5h6_mlp_dependent": (
                all(head_gates.values())
                and float(interaction["estimate"]) >= MIN_HEAD_EFFECT
                and float(interaction["ci_low"]) > 0
                and float(interaction["q_bh_28_tests"]) <= MAX_Q
            ),
        }

    behavior_summaries = {
        task_name: grouped_bootstrap(
            values,
            groups,
            n_boot=N_BOOT,
            seed=290_000 + index,
        )
        for index, (task_name, values) in enumerate(family_behavior.items())
    }
    raw = []
    for index, example in enumerate(examples):
        task_rows = {}
        for task_name in task_specs:
            task_rows[task_name] = {
                "family_behavior": float(family_behavior[task_name][index]),
                "family_metrics": {
                    name: float(values[index])
                    for name, values in family_metrics[task_name].items()
                },
                "seed_behavior": {
                    str(seed): float(
                        seed_values[seed][task_name]["behavior"][index]
                    )
                    for seed in sorted(EXPECTED_SEEDS)
                },
            }
        raw.append(
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "tasks": task_rows,
            }
        )

    artifact = {
        "study": "native GTEx classifier context localization",
        "stage": "frozen final-block component bridge",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "evidence_tier": "mechanistic verification on reused validation cohort",
        "split": "val",
        "cohort_tag": "gtex-native-context-localization-confirmation-v1",
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "device": device,
        "family": sorted(EXPECTED_SEEDS),
        "checkpoints": checkpoint_reports,
        "regions": {
            task_name: spec["window"].name
            for task_name, spec in task_specs.items()
        },
        "components": {
            "candidate": "L5H6",
            "frozen_controls": ["L5H7", "L5H11", "L5H9"],
            "final_mlp": "layer5 output.dense write at target",
        },
        "confirmation_replay_max_abs_error": replay_errors,
        "numeric_atol": NUMERIC_ATOL,
        "multiplicity_family": len(pvalue_entries),
        "behavior_summaries": behavior_summaries,
        "summaries": summaries,
        "decisions": decisions,
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=_json_default)
        + "\n"
    )
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_component_bridge_v1",
        sys.argv,
        repo=REPO,
        status="VERIFICATION",
        config={
            "split": "val",
            "n": args.n,
            "batch": args.batch,
            "device": device,
            "cohort_tag": artifact["cohort_tag"],
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "multiplicity_family": len(pvalue_entries),
            "checkpoint_seeds": sorted(EXPECTED_SEEDS),
            "candidate_head": "L5H6",
            "control_heads": ["L5H7", "L5H11", "L5H9"],
        },
        notes=[
            "No component is selected from a context-region outcome.",
            "Component identities and controls replay the prior mechanism freeze.",
            "The validation cohort is reused, so this is mechanistic verification.",
        ],
    )
    for path in (
        protocol,
        mechanism_preflight_path,
        mechanism_confirmation_path,
        discovery_path,
        confirmation_path,
        audit_path,
        model_weight(base_model),
        base_model / "config.json",
        base_model / "tokenizer.json",
        Path(args.gtf),
        Path(args.fasta),
        *checkpoints.values(),
    ):
        manifest.inputs.append(file_record(path))
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(f"COMPLETE: saved -> {output}")
    print(json.dumps(decisions, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
