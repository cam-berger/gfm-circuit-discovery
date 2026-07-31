#!/usr/bin/env python
"""Preflight exact final-layer interventions for the native GTEx classifier."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
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
from gfmcd.circuits.hf_bert_residual import patch_layer_output  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.circuit import clamp_write, patch_proj, pick_device  # noqa: E402
from gfmcd.splicing.coupling import substitute  # noqa: E402
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.native_classifier import (  # noqa: E402
    NativeClassifierTask,
    load_annotation_cohort,
)
from gfmcd.splicing.native_mechanism import (  # noqa: E402
    attention_context_reconstruction_error,
    cache_final_layer_inputs,
    capture_final_layer_run,
    donor_margin,
    donor_token_positions,
    position_mask,
    replay_final_layer,
)

FINAL_LAYER = 5
PRESELECTED_HEAD = 0
ATOL_PROJECTION = 1e-5
ATOL_INTERVENTION = 2e-5


def model_weight(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model weights not found in {model_dir}")


def native_margin(
    logits: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    selected = logits[rows, token_positions]
    return selected[:, 1].float() - selected[:, 0].float()


def max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != 16:
        raise ValueError("native-mechanism preflight is frozen at n=16")
    if args.batch < 2:
        raise ValueError("preflight subset-invariance check requires batch >=2")
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_MECHANISM_V1.md"
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
        tag="gtex-native-mechanism-discovery-v1",
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
        and adapter.num_heads == 16
        and adapter.prefix_tokens == 1
        and getattr(model.config, "_attn_implementation", None) == "eager"
    )

    batch_rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), args.batch):
        chunk = examples[start : start + args.batch]
        clean_sequences = [example.sequence for example in chunk]
        corrupt_sequences = [
            substitute(
                example.sequence,
                int(example.next_acceptor_pos) - 2,
                "CC",
            )
            for example in chunk
        ]
        donor_nucleotides = [example.donor_pos - 1 for example in chunk]
        clean_encoded = task.encode(clean_sequences)
        corrupt_encoded = task.encode(corrupt_sequences)
        donor_tokens = donor_token_positions(
            donor_nucleotides,
            prefix_tokens=task.prefix_tokens,
            device=device,
        )
        sequence_length = int(clean_encoded["input_ids"].shape[1])
        donor_mask = position_mask(donor_tokens, sequence_length)
        false_mask = torch.zeros_like(donor_mask)
        selected_examples = torch.zeros(
            len(chunk),
            dtype=torch.bool,
            device=donor_tokens.device,
        )
        selected_examples[: max(1, len(chunk) // 2)] = True
        subset_mask = position_mask(
            donor_tokens,
            sequence_length,
            selected_examples=selected_examples,
        )
        donor_positions_2d = donor_tokens[:, None]
        empty_positions = torch.empty(
            (len(chunk), 0),
            dtype=torch.long,
            device=donor_tokens.device,
        )
        all_positions = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=donor_tokens.device,
        )[None, :].expand(len(chunk), sequence_length)

        native_clean_logits = model(**clean_encoded).logits.float()
        native_corrupt_logits = model(**corrupt_encoded).logits.float()
        native_clean = native_margin(native_clean_logits, donor_tokens)
        native_corrupt = native_margin(native_corrupt_logits, donor_tokens)

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
            output_attentions=True,
        )
        corrupt_run = capture_final_layer_run(
            adapter,
            corrupt_inputs,
            layer=FINAL_LAYER,
            output_attentions=False,
        )
        direct_clean = donor_margin(model, clean_run.hidden, donor_tokens)
        direct_corrupt = donor_margin(model, corrupt_run.hidden, donor_tokens)
        projection_error = max(
            task.final_projection_error(clean_sequences),
            task.final_projection_error(corrupt_sequences),
        )

        clean_head_source = {FINAL_LAYER: clean_run.heads}
        corrupt_head_source = {FINAL_LAYER: corrupt_run.heads}
        with patch_heads(
            adapter,
            clean_head_source,
            [(FINAL_LAYER, PRESELECTED_HEAD)],
            token_mask=donor_mask,
        ):
            identity_hidden = replay_final_layer(
                adapter,
                clean_inputs,
                layer=FINAL_LAYER,
            )
        identity_margin = donor_margin(model, identity_hidden, donor_tokens)

        with patch_heads(
            adapter,
            corrupt_head_source,
            [(FINAL_LAYER, PRESELECTED_HEAD)],
            token_mask=false_mask,
        ):
            false_hidden = replay_final_layer(
                adapter,
                clean_inputs,
                layer=FINAL_LAYER,
            )
        false_margin = donor_margin(model, false_hidden, donor_tokens)

        clean_mlp_source = {(FINAL_LAYER, "mlp"): clean_run.mlp_write}
        with clamp_write(
            adapter,
            [(FINAL_LAYER, "mlp")],
            clean_mlp_source,
            positions=donor_positions_2d,
        ):
            mlp_identity_hidden = replay_final_layer(
                adapter,
                clean_inputs,
                layer=FINAL_LAYER,
            )
        mlp_identity = donor_margin(model, mlp_identity_hidden, donor_tokens)
        with clamp_write(
            adapter,
            [(FINAL_LAYER, "mlp")],
            clean_mlp_source,
            positions=empty_positions,
        ):
            mlp_empty_hidden = replay_final_layer(
                adapter,
                clean_inputs,
                layer=FINAL_LAYER,
            )
        mlp_empty = donor_margin(model, mlp_empty_hidden, donor_tokens)

        with patch_layer_output(
            adapter,
            {FINAL_LAYER: corrupt_run.hidden},
            FINAL_LAYER,
            token_mask=donor_mask,
        ):
            positive_hidden = replay_final_layer(
                adapter,
                clean_inputs,
                layer=FINAL_LAYER,
            )
        positive_margin = donor_margin(model, positive_hidden, donor_tokens)

        with patch_heads(
            adapter,
            corrupt_head_source,
            [(FINAL_LAYER, PRESELECTED_HEAD)],
            token_mask=donor_mask,
        ):
            head_patched_hidden = replay_final_layer(
                adapter,
                clean_inputs,
                layer=FINAL_LAYER,
            )
        head_patched_margin = donor_margin(
            model,
            head_patched_hidden,
            donor_tokens,
        )

        with ExitStack() as stack:
            stack.enter_context(
                patch_proj(
                    adapter,
                    FINAL_LAYER,
                    PRESELECTED_HEAD,
                    "query",
                    corrupt_run.query,
                    donor_positions_2d,
                )
            )
            for which, source in (
                ("key", corrupt_run.key),
                ("value", corrupt_run.value),
            ):
                stack.enter_context(
                    patch_proj(
                        adapter,
                        FINAL_LAYER,
                        PRESELECTED_HEAD,
                        which,
                        source,
                        all_positions,
                    )
                )
            qkv_hidden = replay_final_layer(
                adapter,
                clean_inputs,
                layer=FINAL_LAYER,
            )
        qkv_margin = donor_margin(model, qkv_hidden, donor_tokens)

        with patch_heads(
            adapter,
            corrupt_head_source,
            [(FINAL_LAYER, PRESELECTED_HEAD)],
            token_mask=subset_mask,
        ):
            subset_hidden = replay_final_layer(
                adapter,
                clean_inputs,
                layer=FINAL_LAYER,
            )
        subset_margin = donor_margin(model, subset_hidden, donor_tokens)
        unselected = ~selected_examples
        subset_invariance_error = (
            max_error(subset_margin[unselected], direct_clean[unselected])
            if torch.any(unselected)
            else 0.0
        )

        batch_rows.append(
            {
                "start": start,
                "n": len(chunk),
                "sequence_length": sequence_length,
                "projection_error": projection_error,
                "direct_clean_replay_error": max_error(
                    direct_clean,
                    native_clean,
                ),
                "direct_corrupt_replay_error": max_error(
                    direct_corrupt,
                    native_corrupt,
                ),
                "attention_context_reconstruction_error": (
                    attention_context_reconstruction_error(
                        clean_run,
                        num_heads=adapter.num_heads,
                    )
                ),
                "head_clean_identity_error": max_error(
                    identity_margin,
                    direct_clean,
                ),
                "head_false_mask_noop_error": max_error(
                    false_margin,
                    direct_clean,
                ),
                "mlp_clean_identity_error": max_error(
                    mlp_identity,
                    direct_clean,
                ),
                "mlp_empty_positions_noop_error": max_error(
                    mlp_empty,
                    direct_clean,
                ),
                "final_state_positive_control_error": max_error(
                    positive_margin,
                    direct_corrupt,
                ),
                "qkv_vs_head_patch_error": max_error(
                    qkv_margin,
                    head_patched_margin,
                ),
                "unselected_example_invariance_error": subset_invariance_error,
                "coordinate_count_ok": bool(
                    donor_mask.sum().item() == len(chunk)
                    and torch.equal(
                        clean_encoded["attention_mask"],
                        corrupt_encoded["attention_mask"],
                    )
                    and clean_encoded["input_ids"].shape
                    == corrupt_encoded["input_ids"].shape
                ),
            }
        )

    maxima = {
        key: max(float(row[key]) for row in batch_rows)
        for key in (
            "projection_error",
            "direct_clean_replay_error",
            "direct_corrupt_replay_error",
            "attention_context_reconstruction_error",
            "head_clean_identity_error",
            "head_false_mask_noop_error",
            "mlp_clean_identity_error",
            "mlp_empty_positions_noop_error",
            "final_state_positive_control_error",
            "qkv_vs_head_patch_error",
            "unselected_example_invariance_error",
        )
    }
    checks = {
        "architecture_and_eager_attention": architecture_ok,
        "projection_reconstruction": maxima["projection_error"] <= ATOL_PROJECTION,
        "direct_final_layer_replay": (
            maxima["direct_clean_replay_error"] <= ATOL_INTERVENTION
            and maxima["direct_corrupt_replay_error"] <= ATOL_INTERVENTION
        ),
        "attention_context_reconstruction": (
            maxima["attention_context_reconstruction_error"]
            <= ATOL_INTERVENTION
        ),
        "head_identity_and_noop": (
            maxima["head_clean_identity_error"] <= ATOL_INTERVENTION
            and maxima["head_false_mask_noop_error"] <= ATOL_INTERVENTION
        ),
        "mlp_identity_and_noop": (
            maxima["mlp_clean_identity_error"] <= ATOL_INTERVENTION
            and maxima["mlp_empty_positions_noop_error"] <= ATOL_INTERVENTION
        ),
        "final_state_positive_control": (
            maxima["final_state_positive_control_error"] <= ATOL_INTERVENTION
        ),
        "qkv_replays_whole_head": (
            maxima["qkv_vs_head_patch_error"] <= ATOL_INTERVENTION
        ),
        "coordinates_shapes_and_unselected_invariance": (
            all(row["coordinate_count_ok"] for row in batch_rows)
            and maxima["unselected_example_invariance_error"]
            <= ATOL_INTERVENTION
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "study": "native GTEx final-layer mechanism preflight",
        "protocol": str(protocol.relative_to(REPO)),
        "status": status,
        "device": device,
        "n": len(examples),
        "batch_size": args.batch,
        "checkpoint": report.to_dict(),
        "architecture": {
            "num_layers": adapter.num_layers,
            "num_heads": adapter.num_heads,
            "prefix_tokens": adapter.prefix_tokens,
            "attention_implementation": getattr(
                model.config,
                "_attn_implementation",
                None,
            ),
        },
        "thresholds": {
            "projection_atol": ATOL_PROJECTION,
            "intervention_atol": ATOL_INTERVENTION,
        },
        "checks": checks,
        "maxima": maxima,
        "batches": batch_rows,
        "example_ids": [example.example_id for example in examples],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_mechanism_preflight_v1",
        sys.argv,
        repo=REPO,
        status=status,
        config={
            "split": "train",
            "n": args.n,
            "batch_size": args.batch,
            "cohort_tag": "gtex-native-mechanism-discovery-v1",
            "final_layer": FINAL_LAYER,
            "preselected_head": f"L{FINAL_LAYER}H{PRESELECTED_HEAD}",
            "projection_atol": ATOL_PROJECTION,
            "intervention_atol": ATOL_INTERVENTION,
        },
        seeds={"checkpoint": 2020},
        inputs=[
            file_record(protocol),
            file_record(base_model / "config.json"),
            file_record(model_weight(base_model)),
            file_record(checkpoint),
            file_record(args.gtf),
            file_record(args.fasta),
        ],
        notes=[
            "No final-layer head was ranked in preflight.",
            "QKV-to-whole-head equality is a load-bearing intervention memorial.",
            "Direct final-layer replay avoids redundant layers 0-4 compute after exact replay passes.",
        ],
    )
    manifest.add_output(output)
    manifest.write(manifest_path)

    print(json.dumps({"status": status, "checks": checks, "maxima": maxima}, indent=2))
    print(f"saved -> {output}")
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
