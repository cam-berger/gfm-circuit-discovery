#!/usr/bin/env python
"""Trace when confirmed sequence-region information reaches a classifier target."""

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
from gfmcd.circuits.hf_bert_residual import (  # noqa: E402
    capture_embedding_output,
    capture_layer_outputs,
    patch_embedding_output,
    patch_layer_output,
)
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.rigor import benjamini_hochberg  # noqa: E402
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gfmcd.splicing.context_localization import (  # noqa: E402
    RelativeWindow,
    nucleotide_token_mask,
    shuffle_relative_window,
    target_token_mask,
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


EXPECTED_SEEDS = frozenset(range(2020, 2025))
N_TRACE = 64
N_CONFIRMATION = 200
N_BOOT = 5000
N_PERM = 9999
MIN_ARRIVAL = 0.05
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


def native_margin(
    logits: torch.Tensor,
    nucleotide_positions: list[int],
    *,
    label: int,
    prefix_tokens: int,
) -> torch.Tensor:
    columns = torch.as_tensor(
        nucleotide_positions,
        dtype=torch.long,
        device=logits.device,
    ) + prefix_tokens
    rows = torch.arange(len(nucleotide_positions), device=logits.device)
    selected = logits[rows, columns].float()
    return selected[:, label] - selected[:, 0]


def max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (left.float() - right.float()).abs().max().detach().cpu()
    )


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
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--dose-audit", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=N_TRACE)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != N_TRACE:
        raise ValueError(f"context depth trace is frozen at n={N_TRACE}")
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
    confirmation_path = Path(args.confirmation).expanduser().resolve()
    audit_path = Path(args.dose_audit).expanduser().resolve()
    preflight = json.loads(preflight_path.read_text())
    discovery = json.loads(discovery_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    dose_audit = json.loads(audit_path.read_text())
    if preflight.get("status") != "PASS":
        raise RuntimeError("native context preflight did not pass")
    for task_name in ("donor", "acceptor"):
        if not dose_audit["licenses"][task_name][
            "confirmed_32nt_region_licensed"
        ]:
            raise RuntimeError(
                f"{task_name} 32-nt region is not licensed by dose audit"
            )
    checkpoints = parse_checkpoints(args.checkpoint)
    base_model = Path(args.base_model).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)

    confirmation_examples = load_annotation_cohort(
        args.gtf,
        args.fasta,
        split="val",
        n=N_CONFIRMATION,
        tag="gtex-native-context-localization-confirmation-v1",
    )
    examples = confirmation_examples[: args.n]
    expected_ids = [
        row["example_id"] for row in confirmation["raw"][: args.n]
    ]
    if [example.example_id for example in examples] != expected_ids:
        raise RuntimeError("depth-trace cohort is not the confirmation prefix")
    groups = np.asarray([example.gene_id for example in examples], dtype=object)
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
    clean_sequences = [example.sequence for example in examples]
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
                raise RuntimeError(
                    f"{task_name} candidate dose fell below frozen threshold"
                )
            corrupt_sequences[task_name].append(shuffled)

    depths = ("embedding", *(f"layer{layer}" for layer in range(6)))
    patch_sites = ("source_region", "target_token")
    seed_values: dict[
        int,
        dict[str, dict[str, dict[str, np.ndarray] | np.ndarray]],
    ] = {}
    checkpoint_reports: dict[int, dict[str, Any]] = {}
    replay_errors: dict[int, dict[str, float]] = {}
    instrument_errors: dict[int, dict[str, float]] = {}
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
        adapter = SpliceBertAdapter(
            model,
            tokenizer,
            source=str(checkpoint),
            device=device,
        )
        if adapter.num_layers != 6 or adapter.prefix_tokens != 1:
            raise RuntimeError("frozen architecture is six encoder layers")
        seed_values[seed] = {}
        replay_errors[seed] = {}
        instrument_errors[seed] = {}
        for task_name, spec in task_specs.items():
            behavior_values: list[float] = []
            recovery_values = {
                site: {depth: [] for depth in depths}
                for site in patch_sites
            }
            source_ceiling_error = 0.0
            target_ceiling_error = 0.0
            for start in range(0, len(examples), args.batch):
                stop = min(start + args.batch, len(examples))
                chunk_clean = clean_sequences[start:stop]
                chunk_corrupt = corrupt_sequences[task_name][start:stop]
                chunk_anchors = spec["anchors"][start:stop]
                chunk_targets = spec["targets"][start:stop]
                clean_encoded = task.encode(chunk_clean)
                corrupt_encoded = task.encode(chunk_corrupt)
                clean_logits = model(**clean_encoded).logits.float()
                corrupt_logits = model(**corrupt_encoded).logits.float()
                clean_margin = native_margin(
                    clean_logits,
                    chunk_targets,
                    label=int(spec["label"]),
                    prefix_tokens=task.prefix_tokens,
                )
                corrupt_margin = native_margin(
                    corrupt_logits,
                    chunk_targets,
                    label=int(spec["label"]),
                    prefix_tokens=task.prefix_tokens,
                )
                behavior_values.extend(
                    (clean_margin - corrupt_margin).detach().cpu().tolist()
                )
                source_mask = nucleotide_token_mask(
                    anchors=chunk_anchors,
                    window=spec["window"],
                    sequence_length=len(chunk_clean[0]),
                    prefix_tokens=task.prefix_tokens,
                    device=device,
                )
                target_mask = target_token_mask(
                    nucleotide_positions=chunk_targets,
                    sequence_length=len(chunk_clean[0]),
                    prefix_tokens=task.prefix_tokens,
                    device=device,
                )
                clean_embeddings = capture_embedding_output(
                    adapter,
                    lambda: model(**clean_encoded),
                )
                clean_layers = capture_layer_outputs(
                    adapter,
                    lambda: model(**clean_encoded),
                )

                for site, mask in (
                    ("source_region", source_mask),
                    ("target_token", target_mask),
                ):
                    with patch_embedding_output(
                        adapter,
                        clean_embeddings,
                        token_mask=mask,
                    ):
                        patched_logits = model(**corrupt_encoded).logits.float()
                    patched_margin = native_margin(
                        patched_logits,
                        chunk_targets,
                        label=int(spec["label"]),
                        prefix_tokens=task.prefix_tokens,
                    )
                    recovery_values[site]["embedding"].extend(
                        (patched_margin - corrupt_margin)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    if site == "source_region":
                        source_ceiling_error = max(
                            source_ceiling_error,
                            max_error(patched_margin, clean_margin),
                        )

                    for layer in range(adapter.num_layers):
                        with patch_layer_output(
                            adapter,
                            clean_layers,
                            layer,
                            token_mask=mask,
                        ):
                            patched_logits = model(
                                **corrupt_encoded
                            ).logits.float()
                        patched_margin = native_margin(
                            patched_logits,
                            chunk_targets,
                            label=int(spec["label"]),
                            prefix_tokens=task.prefix_tokens,
                        )
                        recovery_values[site][f"layer{layer}"].extend(
                            (patched_margin - corrupt_margin)
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        if site == "target_token" and layer == 5:
                            target_ceiling_error = max(
                                target_ceiling_error,
                                max_error(patched_margin, clean_margin),
                            )
            behavior_array = np.asarray(
                behavior_values,
                dtype=np.float64,
            )
            seed_values[seed][task_name] = {
                "behavior": behavior_array,
                "recoveries": {
                    site: {
                        depth: np.asarray(values, dtype=np.float64)
                        for depth, values in site_values.items()
                    }
                    for site, site_values in recovery_values.items()
                },
            }
            recorded = np.asarray(
                [
                    confirmation["raw"][index]["seeds"][str(seed)][
                        task_name
                    ]["candidate_draw0"]
                    for index in range(args.n)
                ],
                dtype=np.float64,
            )
            replay_errors[seed][task_name] = float(
                np.max(np.abs(behavior_array - recorded))
            )
            instrument_errors[seed][task_name] = {
                "source_embedding_ceiling_error": source_ceiling_error,
                "final_target_ceiling_error": target_ceiling_error,
            }
        checkpoint_reports[seed] = report.to_dict()
        del task, adapter, model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        print(f"checkpoint {seed} complete", flush=True)

    replay_gate = all(
        error <= NUMERIC_ATOL
        for task_errors in replay_errors.values()
        for error in task_errors.values()
    )
    ceiling_gate = all(
        float(error) <= NUMERIC_ATOL
        for task_errors in instrument_errors.values()
        for errors in task_errors.values()
        for error in errors.values()
    )
    if not replay_gate or not ceiling_gate:
        raise RuntimeError(
            "depth trace failed confirmation replay or patch ceiling"
        )

    family_behavior: dict[str, np.ndarray] = {}
    family_recoveries: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    summaries: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    pvalue_entries: list[tuple[str, str, str, float]] = []
    summary_seed = 120_000
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
        family_recoveries[task_name] = {}
        summaries[task_name] = {}
        for site in patch_sites:
            family_recoveries[task_name][site] = {}
            summaries[task_name][site] = {}
            for depth in depths:
                values = np.mean(
                    np.stack(
                        [
                            seed_values[seed][task_name]["recoveries"][
                                site
                            ][depth]
                            for seed in sorted(EXPECTED_SEEDS)
                        ],
                        axis=0,
                    ),
                    axis=0,
                )
                family_recoveries[task_name][site][depth] = values
                summary = grouped_bootstrap(
                    values,
                    groups,
                    n_boot=N_BOOT,
                    seed=summary_seed,
                )
                summary["recovery_fraction_ratio_of_means"] = (
                    grouped_bootstrap_ratio(
                        values,
                        family_behavior[task_name],
                        groups,
                        n_boot=N_BOOT,
                        seed=summary_seed + 10_000,
                    )
                )
                summary["p_signflip_add_one"] = grouped_signflip_pvalue(
                    values,
                    groups,
                    n_perm=N_PERM,
                    seed=summary_seed + 100_000,
                )
                summary["seed_estimates"] = {
                    str(seed): float(
                        np.mean(
                            seed_values[seed][task_name]["recoveries"][
                                site
                            ][depth]
                        )
                    )
                    for seed in sorted(EXPECTED_SEEDS)
                }
                summaries[task_name][site][depth] = summary
                pvalue_entries.append(
                    (
                        task_name,
                        site,
                        depth,
                        float(summary["p_signflip_add_one"]),
                    )
                )
                summary_seed += 1

    qvalues = benjamini_hochberg([entry[3] for entry in pvalue_entries])
    for (task_name, site, depth, _), qvalue in zip(
        pvalue_entries,
        qvalues,
    ):
        summaries[task_name][site][depth][
            "q_bh_28_tests"
        ] = float(qvalue)

    arrival: dict[str, Any] = {}
    for task_name in task_specs:
        earliest = None
        layer_names = [f"layer{layer}" for layer in range(6)]
        for index, depth in enumerate(layer_names):
            summary = summaries[task_name]["target_token"][depth]
            later = [
                summaries[task_name]["target_token"][later_depth]
                for later_depth in layer_names[index:]
            ]
            if (
                float(summary["ci_low"]) > 0
                and float(summary["estimate"]) >= MIN_ARRIVAL
                and float(summary["q_bh_28_tests"]) <= MAX_Q
                and all(float(row["estimate"]) > 0 for row in later)
            ):
                earliest = depth
                break
        arrival[task_name] = {
            "earliest_arrival_layer": earliest,
            "rule": (
                "first target-patch layer with CI>0, recovery>=0.05, "
                "BH q<=0.05, and positive mean through layer5"
            ),
        }

    raw = []
    for index, example in enumerate(examples):
        task_rows = {}
        for task_name in task_specs:
            task_rows[task_name] = {
                "family_behavior_damage": float(
                    family_behavior[task_name][index]
                ),
                "family_recovery": {
                    site: {
                        depth: float(
                            family_recoveries[task_name][site][depth][index]
                        )
                        for depth in depths
                    }
                    for site in patch_sites
                },
                "seeds": {
                    str(seed): {
                        "behavior_damage": float(
                            seed_values[seed][task_name]["behavior"][index]
                        ),
                        "recovery": {
                            site: {
                                depth: float(
                                    seed_values[seed][task_name][
                                        "recoveries"
                                    ][site][depth][index]
                                )
                                for depth in depths
                            }
                            for site in patch_sites
                        },
                    }
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

    behavior_summaries = {
        task_name: grouped_bootstrap(
            values,
            groups,
            n_boot=N_BOOT,
            seed=190_000 + index,
        )
        for index, (task_name, values) in enumerate(family_behavior.items())
    }
    artifact = {
        "study": "native GTEx classifier context localization",
        "stage": "causal source-to-target depth trace",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "split": "val",
        "cohort_tag": "gtex-native-context-localization-confirmation-v1:first64",
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "device": device,
        "family": sorted(EXPECTED_SEEDS),
        "checkpoints": checkpoint_reports,
        "regions": {
            task_name: spec["window"].name
            for task_name, spec in task_specs.items()
        },
        "confirmation_replay_max_abs_error": replay_errors,
        "instrument_errors": instrument_errors,
        "instrument_gates": {
            "confirmation_replay": replay_gate,
            "source_embedding_and_final_target_ceilings": ceiling_gate,
            "numeric_atol": NUMERIC_ATOL,
        },
        "behavior_summaries": behavior_summaries,
        "summaries": summaries,
        "multiplicity_family": len(pvalue_entries),
        "arrival": arrival,
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=_json_default)
        + "\n"
    )
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_depth_trace_v1",
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
            "patch_sites": list(patch_sites),
            "depths": list(depths),
            "multiplicity_family": len(pvalue_entries),
            "checkpoint_seeds": sorted(EXPECTED_SEEDS),
        },
        notes=[
            "Only dose-licensed confirmed 32-nt regions are traced.",
            "Clean activations are patched into the draw-0 corrupt run.",
            "Recovery fractions are ratios of gene-mean recovery and damage.",
        ],
    )
    for path in (
        protocol,
        preflight_path,
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
    print(json.dumps(arrival, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
