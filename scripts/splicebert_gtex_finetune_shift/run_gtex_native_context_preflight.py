#!/usr/bin/env python
"""Preflight causal sequence-context localization through the native GTEx head."""

from __future__ import annotations

import argparse
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
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gfmcd.splicing.context_localization import (  # noqa: E402
    ACCEPTOR_PREFLIGHT_SHAM,
    DONOR_PREFLIGHT_SHAM,
    PREFLIGHT_CORE_WINDOW,
    nucleotide_token_mask,
    shuffle_relative_window,
    target_token_mask,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.native_classifier import (  # noqa: E402
    ACCEPTOR_LABEL,
    DONOR_LABEL,
    NativeClassifierTask,
    load_annotation_cohort,
)


N_PREFLIGHT = 16
FINAL_LAYER = 5
THRESHOLDS = {
    "positive_mean_damage": 5.0,
    "positive_fraction": 0.75,
    "core_mean_damage": 1.0,
    "core_minus_adjacent": 0.5,
    "minimum_hamming_fraction": 0.25,
    "numeric_atol": 2e-5,
}


def model_weight(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model weights not found in {model_dir}")


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
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=N_PREFLIGHT)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != N_PREFLIGHT:
        raise ValueError(f"context preflight is frozen at n={N_PREFLIGHT}")
    if args.batch < 2:
        raise ValueError("batch must be at least 2 for subset invariance")
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CONTEXT_LOCALIZATION_V1.md"
    )
    base_model = Path(args.base_model).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)

    examples = load_annotation_cohort(
        args.gtf,
        args.fasta,
        split="train",
        n=args.n,
        tag="gtex-native-context-localization-preflight-v1",
    )
    model, tokenizer, report = load_gtex_token_classifier(
        base_model,
        checkpoint,
        device=device,
        attn_implementation="eager",
    )
    if report.seed != 2020:
        raise ValueError(f"preflight requires seed 2020, found {report.seed}")
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
    architecture_ok = (
        adapter.num_layers == 6
        and adapter.prefix_tokens == 1
        and getattr(model.config, "_attn_implementation", None) == "eager"
    )

    clean_sequences = [example.sequence for example in examples]
    conditions: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for task_name in ("donor", "acceptor"):
        if task_name == "donor":
            anchors = [int(example.donor_pos) for example in examples]
            targets = [int(example.donor_pos) - 1 for example in examples]
            label = DONOR_LABEL
            sham_window = DONOR_PREFLIGHT_SHAM
            exact_sequences = [
                substitute(example.sequence, int(example.donor_pos), "CC")
                for example in examples
            ]
        else:
            anchors = [int(example.next_acceptor_pos) for example in examples]
            targets = [int(example.next_acceptor_pos) for example in examples]
            label = ACCEPTOR_LABEL
            sham_window = ACCEPTOR_PREFLIGHT_SHAM
            exact_sequences = [
                substitute(
                    example.sequence,
                    int(example.next_acceptor_pos) - 2,
                    "CC",
                )
                for example in examples
            ]

        core_sequences: list[str] = []
        sham_sequences: list[str] = []
        for example, anchor in zip(examples, anchors):
            core, core_audit = shuffle_relative_window(
                example.sequence,
                anchor=anchor,
                window=PREFLIGHT_CORE_WINDOW,
                key=(example.example_id, task_name, "preflight-core"),
                draw=0,
            )
            sham, sham_audit = shuffle_relative_window(
                example.sequence,
                anchor=anchor,
                window=sham_window,
                key=(example.example_id, task_name, "preflight-sham"),
                draw=0,
            )
            core_sequences.append(core)
            sham_sequences.append(sham)
            audit_rows.extend(
                [
                    {
                        "example_id": example.example_id,
                        "task": task_name,
                        "condition": "core",
                        **core_audit.to_dict(),
                    },
                    {
                        "example_id": example.example_id,
                        "task": task_name,
                        "condition": "adjacent",
                        **sham_audit.to_dict(),
                    },
                ]
            )

        clean_logits = task.logits(clean_sequences)
        identity_logits = task.logits(list(clean_sequences))
        exact_logits = task.logits(exact_sequences)
        core_logits = task.logits(core_sequences)
        sham_logits = task.logits(sham_sequences)
        clean_margin = task.score(clean_logits, targets, label=label).margin
        identity_margin = task.score(
            identity_logits,
            targets,
            label=label,
        ).margin
        exact_margin = task.score(exact_logits, targets, label=label).margin
        core_margin = task.score(core_logits, targets, label=label).margin
        sham_margin = task.score(sham_logits, targets, label=label).margin

        first_logits = task.logits([clean_sequences[0]])
        first_margin = task.score(
            first_logits,
            [targets[0]],
            label=label,
        ).margin[0]
        batch_invariance_error = float(abs(first_margin - clean_margin[0]))

        source_patch_error = 0.0
        target_patch_error = 0.0
        false_patch_error = 0.0
        for start in range(0, len(examples), args.batch):
            stop = min(start + args.batch, len(examples))
            chunk_clean = clean_sequences[start:stop]
            chunk_core = core_sequences[start:stop]
            chunk_anchors = anchors[start:stop]
            chunk_targets = targets[start:stop]
            clean_encoded = task.encode(chunk_clean)
            corrupt_encoded = task.encode(chunk_core)
            clean_native = model(**clean_encoded).logits.float()
            corrupt_native = model(**corrupt_encoded).logits.float()
            clean_torch_margin = native_margin(
                clean_native,
                chunk_targets,
                label=label,
                prefix_tokens=task.prefix_tokens,
            )
            corrupt_torch_margin = native_margin(
                corrupt_native,
                chunk_targets,
                label=label,
                prefix_tokens=task.prefix_tokens,
            )
            region_mask = nucleotide_token_mask(
                anchors=chunk_anchors,
                window=PREFLIGHT_CORE_WINDOW,
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
            false_mask = torch.zeros_like(target_mask)
            clean_embeddings = capture_embedding_output(
                adapter,
                lambda: model(**clean_encoded),
            )
            with patch_embedding_output(
                adapter,
                clean_embeddings,
                token_mask=region_mask,
            ):
                source_patched = model(**corrupt_encoded).logits.float()
            source_patch_error = max(
                source_patch_error,
                max_error(
                    native_margin(
                        source_patched,
                        chunk_targets,
                        label=label,
                        prefix_tokens=task.prefix_tokens,
                    ),
                    clean_torch_margin,
                ),
            )

            clean_layers = capture_layer_outputs(
                adapter,
                lambda: model(**clean_encoded),
            )
            with patch_layer_output(
                adapter,
                clean_layers,
                FINAL_LAYER,
                token_mask=target_mask,
            ):
                target_patched = model(**corrupt_encoded).logits.float()
            target_patch_error = max(
                target_patch_error,
                max_error(
                    native_margin(
                        target_patched,
                        chunk_targets,
                        label=label,
                        prefix_tokens=task.prefix_tokens,
                    ),
                    clean_torch_margin,
                ),
            )
            with patch_layer_output(
                adapter,
                clean_layers,
                FINAL_LAYER,
                token_mask=false_mask,
            ):
                false_patched = model(**corrupt_encoded).logits.float()
            false_patch_error = max(
                false_patch_error,
                max_error(
                    native_margin(
                        false_patched,
                        chunk_targets,
                        label=label,
                        prefix_tokens=task.prefix_tokens,
                    ),
                    corrupt_torch_margin,
                ),
            )

        exact_damage = clean_margin - exact_margin
        core_damage = clean_margin - core_margin
        sham_damage = clean_margin - sham_margin
        conditions[task_name] = {
            "label": label,
            "target_coordinate": (
                "donor_pos_minus_1"
                if task_name == "donor"
                else "next_acceptor_pos"
            ),
            "exact_damage": exact_damage,
            "core_damage": core_damage,
            "adjacent_damage": sham_damage,
            "core_minus_adjacent": core_damage - sham_damage,
            "identity_max_abs_error": float(
                np.max(np.abs(clean_margin - identity_margin))
            ),
            "batch_invariance_error": batch_invariance_error,
            "source_embedding_patch_error": source_patch_error,
            "final_target_patch_error": target_patch_error,
            "false_patch_error": false_patch_error,
        }

    checks: dict[str, bool] = {
        "architecture": architecture_ok,
        "shuffle_invariants": all(
            bool(
                row["length_preserved"]
                and row["outside_preserved"]
                and row["endpoints_preserved"]
                and row["mononucleotides_preserved"]
                and row["dinucleotides_preserved"]
            )
            for row in audit_rows
        ),
        "shuffle_dose": min(
            float(row["hamming_fraction"]) for row in audit_rows
        )
        >= THRESHOLDS["minimum_hamming_fraction"],
    }
    for task_name, result in conditions.items():
        checks.update(
            {
                f"{task_name}:positive_mean": float(
                    np.mean(result["exact_damage"])
                )
                >= THRESHOLDS["positive_mean_damage"],
                f"{task_name}:positive_fraction": float(
                    np.mean(result["exact_damage"] > 0)
                )
                >= THRESHOLDS["positive_fraction"],
                f"{task_name}:sensitivity": float(
                    np.mean(result["core_damage"])
                )
                >= THRESHOLDS["core_mean_damage"],
                f"{task_name}:resolution": float(
                    np.mean(result["core_minus_adjacent"])
                )
                >= THRESHOLDS["core_minus_adjacent"],
                f"{task_name}:identity": float(
                    result["identity_max_abs_error"]
                )
                <= THRESHOLDS["numeric_atol"],
                f"{task_name}:batch_invariance": float(
                    result["batch_invariance_error"]
                )
                <= THRESHOLDS["numeric_atol"],
                f"{task_name}:source_embedding_patch": float(
                    result["source_embedding_patch_error"]
                )
                <= THRESHOLDS["numeric_atol"],
                f"{task_name}:final_target_patch": float(
                    result["final_target_patch_error"]
                )
                <= THRESHOLDS["numeric_atol"],
                f"{task_name}:false_patch": float(
                    result["false_patch_error"]
                )
                <= THRESHOLDS["numeric_atol"],
            }
        )

    raw = []
    for index, example in enumerate(examples):
        raw.append(
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "donor": {
                    key: float(value[index])
                    for key, value in conditions["donor"].items()
                    if isinstance(value, np.ndarray)
                },
                "acceptor": {
                    key: float(value[index])
                    for key, value in conditions["acceptor"].items()
                    if isinstance(value, np.ndarray)
                },
            }
        )
    summaries = {}
    for task_name, result in conditions.items():
        summaries[task_name] = {
            key: (
                {
                    "mean": float(np.mean(value)),
                    "positive_fraction": float(np.mean(value > 0)),
                    "min": float(np.min(value)),
                    "max": float(np.max(value)),
                }
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in result.items()
        }

    artifact = {
        "study": "native GTEx classifier context localization",
        "stage": "preflight",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "split": "train",
        "cohort_tag": "gtex-native-context-localization-preflight-v1",
        "n": len(examples),
        "device": device,
        "checkpoint": report.to_dict(),
        "thresholds": THRESHOLDS,
        "checks": checks,
        "n_passed": int(sum(checks.values())),
        "n_checks": len(checks),
        "summaries": summaries,
        "shuffle_audits": audit_rows,
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=_json_default)
        + "\n"
    )
    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_context_preflight_v1",
        sys.argv,
        repo=REPO,
        status=artifact["status"],
        config={
            "split": "train",
            "n": args.n,
            "batch": args.batch,
            "device": device,
            "cohort_tag": artifact["cohort_tag"],
            "thresholds": THRESHOLDS,
            "validation_outcomes_opened": False,
        },
        notes=[
            "No context-shuffle model result was inspected before the protocol freeze.",
            "All preflight model outcomes use published seed 2020 on train.",
        ],
    )
    for path in (
        protocol,
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
    print(f"{artifact['status']}: saved -> {output}")
    if artifact["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"context preflight failed: {failed}")


if __name__ == "__main__":
    main()
