#!/usr/bin/env python
"""Train-only equal-width context localization for native splice classification."""

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
    ACCEPTOR_SCAN_WINDOWS,
    CENTRAL_SCAN_WINDOW,
    DONOR_SCAN_WINDOWS,
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


N_DISCOVERY = 128
N_BOOT = 5000
N_PERM = 9999
MIN_EFFECT = 0.05
MAX_Q = 0.05


def model_weight(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model weights not found in {model_dir}")


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


def select_candidate(
    summaries: dict[str, dict[str, Any]],
) -> tuple[str | None, str | None]:
    eligible = [
        name
        for name, summary in summaries.items()
        if name != CENTRAL_SCAN_WINDOW.name
        and float(summary["estimate"]) >= MIN_EFFECT
        and float(summary["ci_low"]) > 0
        and float(summary["q_bh_16_tests"]) <= MAX_Q
    ]
    if not eligible:
        return None, None
    candidate = min(
        eligible,
        key=lambda name: (
            -float(summaries[name]["estimate"]),
            int(name.split(":", 1)[0]),
        ),
    )
    sham_pool = [
        name
        for name in summaries
        if name not in {CENTRAL_SCAN_WINDOW.name, candidate}
    ]
    sham = min(
        sham_pool,
        key=lambda name: (
            abs(float(summaries[name]["estimate"])),
            int(name.split(":", 1)[0]),
        ),
    )
    return candidate, sham


def parse_window(name: str) -> RelativeWindow:
    start_text, end_text = name.split(":")
    return RelativeWindow(int(start_text), int(end_text))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
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
        raise ValueError(f"context discovery is frozen at n={N_DISCOVERY}")
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
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "PASS":
        raise RuntimeError("native context preflight did not pass")
    preflight_ids = {
        row["example_id"] for row in preflight.get("raw", [])
    }
    if len(preflight_ids) != 16:
        raise RuntimeError("preflight artifact does not contain 16 unique examples")
    base_model = Path(args.base_model).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)

    pool = load_annotation_cohort(
        args.gtf,
        args.fasta,
        split="train",
        n=args.n + len(preflight_ids),
        tag="gtex-native-context-localization-discovery-v1",
    )
    examples = [
        example for example in pool if example.example_id not in preflight_ids
    ][: args.n]
    if len(examples) != args.n:
        raise RuntimeError("could not form preflight-disjoint discovery cohort")
    groups = np.asarray([example.gene_id for example in examples], dtype=object)

    model, tokenizer, report = load_gtex_token_classifier(
        base_model,
        checkpoint,
        device=device,
    )
    if report.seed != 2020:
        raise ValueError(f"discovery requires seed 2020, found {report.seed}")
    task = NativeClassifierTask(
        model,
        tokenizer,
        device=device,
        batch_size=args.batch,
    )
    clean_sequences = [example.sequence for example in examples]
    clean_logits = task.logits(clean_sequences)

    task_specs = {
        "donor": {
            "windows": DONOR_SCAN_WINDOWS,
            "anchors": [int(example.donor_pos) for example in examples],
            "targets": [int(example.donor_pos) - 1 for example in examples],
            "label": DONOR_LABEL,
        },
        "acceptor": {
            "windows": ACCEPTOR_SCAN_WINDOWS,
            "anchors": [
                int(example.next_acceptor_pos) for example in examples
            ],
            "targets": [
                int(example.next_acceptor_pos) for example in examples
            ],
            "label": ACCEPTOR_LABEL,
        },
    }
    effects: dict[str, dict[str, np.ndarray]] = {}
    audit_lookup: dict[str, dict[str, list[dict[str, Any]]]] = {}
    clean_margins: dict[str, np.ndarray] = {}
    scan_index = 0
    for task_name, spec in task_specs.items():
        clean_margin = task.score(
            clean_logits,
            spec["targets"],
            label=int(spec["label"]),
        ).margin
        clean_margins[task_name] = clean_margin
        effects[task_name] = {}
        audit_lookup[task_name] = {}
        for window in spec["windows"]:
            shuffled_sequences: list[str] = []
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
                        "discovery",
                    ),
                    draw=0,
                )
                shuffled_sequences.append(shuffled)
                audits.append(audit.to_dict())
            shuffled_logits = task.logits(shuffled_sequences)
            shuffled_margin = task.score(
                shuffled_logits,
                spec["targets"],
                label=int(spec["label"]),
            ).margin
            effects[task_name][window.name] = clean_margin - shuffled_margin
            audit_lookup[task_name][window.name] = audits
            scan_index += 1
            print(
                f"context scan {scan_index}/16 "
                f"{task_name} {window.name}",
                flush=True,
            )

    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    pvalue_entries: list[tuple[str, str, float]] = []
    summary_seed = 20_000
    for task_name, task_effects in effects.items():
        summaries[task_name] = {}
        for window_name, values in task_effects.items():
            summary = summarize(values, groups, seed=summary_seed)
            summary_seed += 1
            summaries[task_name][window_name] = summary
            pvalue_entries.append(
                (
                    task_name,
                    window_name,
                    float(summary["p_signflip_add_one"]),
                )
            )
    qvalues = benjamini_hochberg([entry[2] for entry in pvalue_entries])
    for (task_name, window_name, _), qvalue in zip(pvalue_entries, qvalues):
        summaries[task_name][window_name]["q_bh_16_tests"] = float(qvalue)

    freeze: dict[str, Any] = {}
    for task_name in ("donor", "acceptor"):
        candidate, sham = select_candidate(summaries[task_name])
        if candidate is None:
            freeze[task_name] = {
                "central_window": CENTRAL_SCAN_WINDOW.name,
                "distal_candidate": None,
                "matched_sham": None,
                "candidate_halves": [],
                "validation_authorized": False,
            }
            continue
        candidate_window = parse_window(candidate)
        freeze[task_name] = {
            "central_window": CENTRAL_SCAN_WINDOW.name,
            "distal_candidate": candidate,
            "matched_sham": sham,
            "candidate_halves": [
                half.name for half in candidate_window.halves()
            ],
            "validation_authorized": True,
        }

    raw = []
    for index, example in enumerate(examples):
        task_rows = {}
        for task_name, task_effects in effects.items():
            task_rows[task_name] = {
                "clean_margin": float(clean_margins[task_name][index]),
                "windows": {
                    window_name: {
                        "damage": float(values[index]),
                        "shuffle": audit_lookup[task_name][window_name][index],
                    }
                    for window_name, values in task_effects.items()
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
        "stage": "train coarse discovery",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "split": "train",
        "cohort_tag": "gtex-native-context-localization-discovery-v1",
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "preflight_disjoint": True,
        "device": device,
        "checkpoint": report.to_dict(),
        "scan": {
            "window_width": 32,
            "draw": 0,
            "dinucleotide_shuffle_candidates": 64,
            "multiplicity_family": 16,
        },
        "summaries": summaries,
        "selection_rule": {
            "central_carried_as_calibration": CENTRAL_SCAN_WINDOW.name,
            "distal_min_effect": MIN_EFFECT,
            "distal_ci_low_above_zero": True,
            "distal_q_bh_16_tests_at_most": MAX_Q,
            "at_most_one_per_task": True,
            "rank": "descending estimate, relative-start tie-break",
            "matched_sham": (
                "same-task noncentral noncandidate window with smallest "
                "absolute train estimate"
            ),
        },
        "freeze": freeze,
        "validation_authorized": any(
            bool(task_freeze["validation_authorized"])
            for task_freeze in freeze.values()
        ),
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=_json_default)
        + "\n"
    )
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_discovery_v1",
        sys.argv,
        repo=REPO,
        status="EXPLORATORY",
        config={
            "split": "train",
            "n": args.n,
            "batch": args.batch,
            "device": device,
            "cohort_tag": artifact["cohort_tag"],
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "multiplicity_family": 16,
            "validation_outcomes_opened": False,
        },
        notes=[
            "Seed 2020 and train only select at most one distal region per endpoint.",
            "The central window is an assay calibration, not evidence-selected.",
            "Validation region identities, shams, and halves are frozen in this artifact.",
        ],
    )
    for path in (
        protocol,
        preflight_path,
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
    del task, model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    print(f"COMPLETE: saved -> {output}")
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
