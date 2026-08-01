#!/usr/bin/env python
"""Five-seed confirmation of train-frozen native context regions and shams."""

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

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
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
from gfmcd.splicing.native_classifier import (  # noqa: E402
    ACCEPTOR_LABEL,
    DONOR_LABEL,
    NativeClassifierTask,
    load_annotation_cohort,
)


EXPECTED_SEEDS = frozenset(range(2020, 2025))
N_CONFIRMATION = 200
N_BOOT = 5000
N_PERM = 9999
MIN_EFFECT = 0.05
MAX_Q = 0.05


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


def family_summary(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    permutation: bool,
) -> dict[str, Any]:
    summary = grouped_bootstrap(values, groups, n_boot=N_BOOT, seed=seed)
    if permutation:
        summary["p_signflip_add_one"] = grouped_signflip_pvalue(
            values,
            groups,
            n_perm=N_PERM,
            seed=seed + 100_000,
        )
    return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--discovery", required=True)
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
        raise ValueError(f"context confirmation is frozen at n={N_CONFIRMATION}")
    if args.n_boot != N_BOOT or args.n_perm != N_PERM:
        raise ValueError(
            f"inference is frozen at {N_BOOT} bootstraps/{N_PERM} sign flips"
        )
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CONTEXT_LOCALIZATION_V1.md"
    )
    preflight_path = Path(args.preflight).expanduser().resolve()
    discovery_path = Path(args.discovery).expanduser().resolve()
    preflight = json.loads(preflight_path.read_text())
    discovery = json.loads(discovery_path.read_text())
    if preflight.get("status") != "PASS":
        raise RuntimeError("native context preflight did not pass")
    if discovery.get("status") != "COMPLETE":
        raise RuntimeError("native context discovery is incomplete")
    if not discovery.get("validation_authorized"):
        raise RuntimeError("train discovery authorized no distal validation")
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
    groups = np.asarray([example.gene_id for example in examples], dtype=object)
    clean_sequences = [example.sequence for example in examples]
    task_specs: dict[str, dict[str, Any]] = {
        "donor": {
            "anchors": [int(example.donor_pos) for example in examples],
            "targets": [int(example.donor_pos) - 1 for example in examples],
            "label": DONOR_LABEL,
        },
        "acceptor": {
            "anchors": [
                int(example.next_acceptor_pos) for example in examples
            ],
            "targets": [
                int(example.next_acceptor_pos) for example in examples
            ],
            "label": ACCEPTOR_LABEL,
        },
    }

    condition_sequences: dict[str, dict[str, list[str]]] = {}
    condition_audits: dict[str, dict[str, list[dict[str, Any]]]] = {}
    condition_windows: dict[str, dict[str, str]] = {}
    for task_name, spec in task_specs.items():
        frozen = discovery["freeze"][task_name]
        if not frozen["validation_authorized"]:
            continue
        primary_windows = {
            "central_draw0": frozen["central_window"],
            "candidate_draw0": frozen["distal_candidate"],
            "sham_draw0": frozen["matched_sham"],
            "half1_draw0": frozen["candidate_halves"][0],
            "half2_draw0": frozen["candidate_halves"][1],
            "candidate_draw1": frozen["distal_candidate"],
        }
        condition_sequences[task_name] = {}
        condition_audits[task_name] = {}
        condition_windows[task_name] = primary_windows
        for condition, window_name in primary_windows.items():
            window = parse_window(window_name)
            draw = 1 if condition.endswith("draw1") else 0
            sequences: list[str] = []
            audits: list[dict[str, Any]] = []
            for example, anchor in zip(examples, spec["anchors"]):
                shuffled, audit = shuffle_relative_window(
                    example.sequence,
                    anchor=int(anchor),
                    window=window,
                    key=(
                        example.example_id,
                        task_name,
                        window.name,
                        "confirmation",
                    ),
                    draw=draw,
                )
                sequences.append(shuffled)
                audits.append(audit.to_dict())
            condition_sequences[task_name][condition] = sequences
            condition_audits[task_name][condition] = audits

    seed_effects: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    seed_summaries: dict[int, dict[str, Any]] = {}
    checkpoint_reports: dict[int, dict[str, Any]] = {}
    for seed, checkpoint in sorted(checkpoints.items()):
        model, tokenizer, report = load_gtex_token_classifier(
            base_model,
            checkpoint,
            device=device,
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
        clean_logits = task.logits(clean_sequences)
        seed_effects[seed] = {}
        seed_summaries[seed] = {}
        for task_index, (task_name, spec) in enumerate(task_specs.items()):
            if task_name not in condition_sequences:
                continue
            clean_margin = task.score(
                clean_logits,
                spec["targets"],
                label=int(spec["label"]),
            ).margin
            seed_effects[seed][task_name] = {}
            seed_summaries[seed][task_name] = {
                "clean_margin": grouped_bootstrap(
                    clean_margin,
                    groups,
                    n_boot=N_BOOT,
                    seed=seed * 100 + task_index,
                )
            }
            for condition_index, (condition, sequences) in enumerate(
                condition_sequences[task_name].items()
            ):
                logits = task.logits(sequences)
                margin = task.score(
                    logits,
                    spec["targets"],
                    label=int(spec["label"]),
                ).margin
                damage = clean_margin - margin
                seed_effects[seed][task_name][condition] = damage
                seed_summaries[seed][task_name][condition] = grouped_bootstrap(
                    damage,
                    groups,
                    n_boot=N_BOOT,
                    seed=seed * 1000 + task_index * 100 + condition_index,
                )
            candidate = seed_effects[seed][task_name]["candidate_draw0"]
            sham = seed_effects[seed][task_name]["sham_draw0"]
            half1 = seed_effects[seed][task_name]["half1_draw0"]
            half2 = seed_effects[seed][task_name]["half2_draw0"]
            seed_effects[seed][task_name][
                "candidate_minus_sham"
            ] = candidate - sham
            seed_effects[seed][task_name][
                "half1_minus_half2"
            ] = half1 - half2
            seed_summaries[seed][task_name][
                "candidate_minus_sham"
            ] = grouped_bootstrap(
                candidate - sham,
                groups,
                n_boot=N_BOOT,
                seed=seed * 1000 + task_index * 100 + 90,
            )
            seed_summaries[seed][task_name][
                "half1_minus_half2"
            ] = grouped_bootstrap(
                half1 - half2,
                groups,
                n_boot=N_BOOT,
                seed=seed * 1000 + task_index * 100 + 91,
            )
        checkpoint_reports[seed] = report.to_dict()
        del task, model, tokenizer, clean_logits
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        print(f"checkpoint {seed} complete", flush=True)

    family_arrays: dict[str, dict[str, np.ndarray]] = {}
    family_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    primary_names = (
        "central_draw0",
        "candidate_draw0",
        "sham_draw0",
        "half1_draw0",
        "half2_draw0",
        "candidate_minus_sham",
        "half1_minus_half2",
    )
    pvalue_entries: list[tuple[str, str, float]] = []
    family_seed = 80_000
    for task_name in condition_sequences:
        family_arrays[task_name] = {}
        family_summaries[task_name] = {}
        available_names = tuple(
            seed_effects[min(EXPECTED_SEEDS)][task_name].keys()
        )
        for condition in available_names:
            values = np.mean(
                np.stack(
                    [
                        seed_effects[seed][task_name][condition]
                        for seed in sorted(EXPECTED_SEEDS)
                    ],
                    axis=0,
                ),
                axis=0,
            )
            family_arrays[task_name][condition] = values
            is_primary = condition in primary_names
            summary = family_summary(
                values,
                groups,
                seed=family_seed,
                permutation=is_primary,
            )
            family_seed += 1
            summary["seed_estimates"] = {
                str(seed): float(
                    np.mean(seed_effects[seed][task_name][condition])
                )
                for seed in sorted(EXPECTED_SEEDS)
            }
            family_summaries[task_name][condition] = summary
            if is_primary:
                pvalue_entries.append(
                    (
                        task_name,
                        condition,
                        float(summary["p_signflip_add_one"]),
                    )
                )

    qvalues = benjamini_hochberg([entry[2] for entry in pvalue_entries])
    for (task_name, condition, _), qvalue in zip(pvalue_entries, qvalues):
        family_summaries[task_name][condition][
            "q_bh_primary_family"
        ] = float(qvalue)

    confirmation: dict[str, dict[str, Any]] = {}
    for task_name in condition_sequences:
        candidate = family_summaries[task_name]["candidate_draw0"]
        contrast = family_summaries[task_name]["candidate_minus_sham"]
        draw1 = family_summaries[task_name]["candidate_draw1"]
        gates = {
            "candidate_ci_above_zero": float(candidate["ci_low"]) > 0,
            "candidate_effect_at_least_0.05": float(candidate["estimate"])
            >= MIN_EFFECT,
            "candidate_positive_5_of_5": all(
                float(value) > 0
                for value in candidate["seed_estimates"].values()
            ),
            "candidate_minus_sham_ci_above_zero": float(contrast["ci_low"])
            > 0,
            "candidate_q_at_most_0.05": float(
                candidate["q_bh_primary_family"]
            )
            <= MAX_Q,
            "candidate_minus_sham_q_at_most_0.05": float(
                contrast["q_bh_primary_family"]
            )
            <= MAX_Q,
            "draw1_ci_above_zero": float(draw1["ci_low"]) > 0,
            "draw1_positive_5_of_5": all(
                float(value) > 0
                for value in draw1["seed_estimates"].values()
            ),
        }
        confirmation[task_name] = {
            "confirmed": all(gates.values()),
            "gates": gates,
            "candidate_window": discovery["freeze"][task_name][
                "distal_candidate"
            ],
            "matched_sham_window": discovery["freeze"][task_name][
                "matched_sham"
            ],
            "candidate_halves": discovery["freeze"][task_name][
                "candidate_halves"
            ],
        }

    raw = []
    for index, example in enumerate(examples):
        seed_rows: dict[str, Any] = {}
        for seed in sorted(EXPECTED_SEEDS):
            task_rows = {}
            for task_name, task_conditions in seed_effects[seed].items():
                task_rows[task_name] = {
                    condition: float(values[index])
                    for condition, values in task_conditions.items()
                }
            seed_rows[str(seed)] = task_rows
        raw.append(
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "seeds": seed_rows,
            }
        )

    audit_summary = {}
    for task_name, task_conditions in condition_audits.items():
        audit_summary[task_name] = {}
        for condition, audits in task_conditions.items():
            audit_summary[task_name][condition] = {
                "window": condition_windows[task_name][condition],
                "draw": int(audits[0]["draw"]),
                "minimum_hamming_fraction": float(
                    min(audit["hamming_fraction"] for audit in audits)
                ),
                "mean_hamming_fraction": float(
                    np.mean(
                        [audit["hamming_fraction"] for audit in audits]
                    )
                ),
                "all_invariants_pass": all(
                    bool(
                        audit["length_preserved"]
                        and audit["outside_preserved"]
                        and audit["endpoints_preserved"]
                        and audit["mononucleotides_preserved"]
                        and audit["dinucleotides_preserved"]
                    )
                    for audit in audits
                ),
            }

    artifact = {
        "study": "native GTEx classifier context localization",
        "stage": "five-seed region confirmation",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "split": "val",
        "cohort_tag": "gtex-native-context-localization-confirmation-v1",
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "device": device,
        "family": sorted(EXPECTED_SEEDS),
        "checkpoints": checkpoint_reports,
        "freeze_replay": discovery["freeze"],
        "primary_multiplicity_family": len(pvalue_entries),
        "family_summaries": family_summaries,
        "seed_summaries": seed_summaries,
        "confirmation": confirmation,
        "audit_summary": audit_summary,
        "depth_trace_authorized": {
            task_name: bool(result["confirmed"])
            for task_name, result in confirmation.items()
        },
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=_json_default)
        + "\n"
    )
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_confirmation_v1",
        sys.argv,
        repo=REPO,
        status="CONFIRMATORY",
        config={
            "split": "val",
            "n": args.n,
            "batch": args.batch,
            "device": device,
            "cohort_tag": artifact["cohort_tag"],
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "primary_multiplicity_family": len(pvalue_entries),
            "checkpoint_seeds": sorted(EXPECTED_SEEDS),
        },
        notes=[
            "Regions, matched shams, and halves are replayed unchanged from train.",
            "Every published checkpoint is evaluated sequentially on the same cohort.",
            "Draw 1 is an independently keyed ordering-robustness intervention.",
        ],
    )
    for path in (
        protocol,
        preflight_path,
        discovery_path,
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
    print(json.dumps(confirmation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
