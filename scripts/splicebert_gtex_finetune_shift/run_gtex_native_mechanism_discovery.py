#!/usr/bin/env python
"""Train-split discovery of causal final-layer GTEx classifier components."""

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
from gfmcd.splicing.coupling import substitute  # noqa: E402
from gfmcd.splicing.grouped_stats import (  # noqa: E402
    grouped_bootstrap,
    grouped_signflip_pvalue,
)
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.native_classifier import (  # noqa: E402
    NativeClassifierTask,
    load_annotation_cohort,
)
from gfmcd.splicing.native_mechanism import (  # noqa: E402
    cache_final_layer_inputs,
    capture_final_layer_run,
    deterministic_control_heads,
    donor_margin,
    donor_token_positions,
    position_mask,
    replay_final_layer,
    select_head_candidates,
)

FINAL_LAYER = 5
MIN_EFFECT = 0.05
MAX_Q = 0.10
MAX_CANDIDATES = 3
CONTROLS_PER_CANDIDATE = 3


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
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    result = grouped_bootstrap(
        values,
        groups,
        n_boot=n_boot,
        seed=seed,
    )
    result["p_signflip_add_one"] = grouped_signflip_pvalue(
        values,
        groups,
        n_perm=n_perm,
        seed=seed + 100_000,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--n-perm", type=int, default=9999)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != 128:
        raise ValueError("native-mechanism discovery is frozen at n=128")
    if args.n_boot != 5000 or args.n_perm != 9999:
        raise ValueError("discovery inference is frozen at 5000 bootstraps/9999 sign flips")
    torch.set_grad_enabled(False)
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_MECHANISM_V1.md"
    )
    preflight_path = Path(args.preflight).expanduser().resolve()
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "PASS":
        raise RuntimeError("native-mechanism preflight did not pass")
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
    example_ids = [example.example_id for example in examples]
    if example_ids[: len(preflight["example_ids"])] != preflight["example_ids"]:
        raise RuntimeError("preflight and discovery cohort prefixes disagree")
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
    if adapter.num_layers != 6 or adapter.num_heads != 16:
        raise RuntimeError("frozen architecture is 6 layers x 16 heads")

    clean_values: list[float] = []
    corrupt_values: list[float] = []
    damage: dict[int, list[float]] = {
        head: [] for head in range(adapter.num_heads)
    }
    recovery: dict[int, list[float]] = {
        head: [] for head in range(adapter.num_heads)
    }
    mlp_damage: list[float] = []
    mlp_recovery: list[float] = []

    for start in range(0, len(examples), args.batch):
        chunk = examples[start : start + args.batch]
        clean_encoded = task.encode([example.sequence for example in chunk])
        corrupt_encoded = task.encode(
            [
                substitute(
                    example.sequence,
                    int(example.next_acceptor_pos) - 2,
                    "CC",
                )
                for example in chunk
            ]
        )
        donor_tokens = donor_token_positions(
            [example.donor_pos - 1 for example in chunk],
            prefix_tokens=task.prefix_tokens,
            device=device,
        )
        donor_mask = position_mask(
            donor_tokens,
            int(clean_encoded["input_ids"].shape[1]),
        )
        donor_positions_2d = donor_tokens[:, None]
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
        clean_margin = donor_margin(model, clean_run.hidden, donor_tokens)
        corrupt_margin = donor_margin(model, corrupt_run.hidden, donor_tokens)
        clean_values.extend(clean_margin.detach().cpu().tolist())
        corrupt_values.extend(corrupt_margin.detach().cpu().tolist())

        clean_source = {FINAL_LAYER: clean_run.heads}
        corrupt_source = {FINAL_LAYER: corrupt_run.heads}
        for head in range(adapter.num_heads):
            with patch_heads(
                adapter,
                corrupt_source,
                [(FINAL_LAYER, head)],
                token_mask=donor_mask,
            ):
                noised_hidden = replay_final_layer(
                    adapter,
                    clean_inputs,
                    layer=FINAL_LAYER,
                )
            noised = donor_margin(model, noised_hidden, donor_tokens)
            damage[head].extend(
                (clean_margin - noised).detach().cpu().tolist()
            )

            with patch_heads(
                adapter,
                clean_source,
                [(FINAL_LAYER, head)],
                token_mask=donor_mask,
            ):
                denoised_hidden = replay_final_layer(
                    adapter,
                    corrupt_inputs,
                    layer=FINAL_LAYER,
                )
            denoised = donor_margin(model, denoised_hidden, donor_tokens)
            recovery[head].extend(
                (denoised - corrupt_margin).detach().cpu().tolist()
            )

        with clamp_write(
            adapter,
            [(FINAL_LAYER, "mlp")],
            {(FINAL_LAYER, "mlp"): corrupt_run.mlp_write},
            positions=donor_positions_2d,
        ):
            noised_mlp_hidden = replay_final_layer(
                adapter,
                clean_inputs,
                layer=FINAL_LAYER,
            )
        noised_mlp = donor_margin(model, noised_mlp_hidden, donor_tokens)
        mlp_damage.extend(
            (clean_margin - noised_mlp).detach().cpu().tolist()
        )

        with clamp_write(
            adapter,
            [(FINAL_LAYER, "mlp")],
            {(FINAL_LAYER, "mlp"): clean_run.mlp_write},
            positions=donor_positions_2d,
        ):
            denoised_mlp_hidden = replay_final_layer(
                adapter,
                corrupt_inputs,
                layer=FINAL_LAYER,
            )
        denoised_mlp = donor_margin(model, denoised_mlp_hidden, donor_tokens)
        mlp_recovery.extend(
            (denoised_mlp - corrupt_margin).detach().cpu().tolist()
        )

        print(
            f"component scan {min(start + len(chunk), len(examples))}/{len(examples)}",
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

    head_arrays = {
        head: {
            "damage": np.asarray(damage[head], dtype=np.float64),
            "recovery": np.asarray(recovery[head], dtype=np.float64),
        }
        for head in range(adapter.num_heads)
    }
    summaries: dict[int, dict[str, dict[str, Any]]] = {}
    pvalue_entries: list[tuple[int, str, float]] = []
    for head in range(adapter.num_heads):
        summaries[head] = {}
        for direction_index, direction in enumerate(("damage", "recovery")):
            summary = summarize(
                head_arrays[head][direction],
                groups,
                n_boot=args.n_boot,
                n_perm=args.n_perm,
                seed=2020 + 100 * head + direction_index,
            )
            summaries[head][direction] = summary
            pvalue_entries.append(
                (head, direction, float(summary["p_signflip_add_one"]))
            )
    qvalues = benjamini_hochberg([row[2] for row in pvalue_entries])
    for (head, direction, _), qvalue in zip(pvalue_entries, qvalues):
        summaries[head][direction]["q_bh_32_tests"] = float(qvalue)

    mlp_arrays = {
        "damage": np.asarray(mlp_damage, dtype=np.float64),
        "recovery": np.asarray(mlp_recovery, dtype=np.float64),
    }
    mlp_summary = {
        direction: summarize(
            values,
            groups,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            seed=40_000 + index,
        )
        for index, (direction, values) in enumerate(mlp_arrays.items())
    }
    mlp_qvalues = benjamini_hochberg(
        [
            float(mlp_summary[direction]["p_signflip_add_one"])
            for direction in ("damage", "recovery")
        ]
    )
    for direction, qvalue in zip(("damage", "recovery"), mlp_qvalues):
        mlp_summary[direction]["q_bh_2_tests"] = float(qvalue)

    candidates = select_head_candidates(
        summaries,
        min_effect=MIN_EFFECT,
        max_q=MAX_Q,
        max_candidates=MAX_CANDIDATES,
    )
    controls = deterministic_control_heads(
        candidates,
        num_heads=adapter.num_heads,
        controls_per_candidate=CONTROLS_PER_CANDIDATE,
    )
    clean_array = np.asarray(clean_values, dtype=np.float64)
    corrupt_array = np.asarray(corrupt_values, dtype=np.float64)
    raw = []
    for index, example in enumerate(examples):
        raw.append(
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "clean_margin": clean_array[index],
                "corrupt_margin": corrupt_array[index],
                "behavior_gap": clean_array[index] - corrupt_array[index],
                "heads": {
                    f"L{FINAL_LAYER}H{head}": {
                        "damage": head_arrays[head]["damage"][index],
                        "recovery": head_arrays[head]["recovery"][index],
                    }
                    for head in range(adapter.num_heads)
                },
                "mlp": {
                    "damage": mlp_arrays["damage"][index],
                    "recovery": mlp_arrays["recovery"][index],
                },
            }
        )

    result = {
        "study": "native GTEx final-layer mechanism discovery",
        "protocol": str(protocol.relative_to(REPO)),
        "status": "COMPLETE",
        "device": device,
        "split": "train",
        "cohort_tag": "gtex-native-mechanism-discovery-v1",
        "n": len(examples),
        "n_groups": int(len(np.unique(groups))),
        "checkpoint": report.to_dict(),
        "behavior": grouped_bootstrap(
            clean_array - corrupt_array,
            groups,
            n_boot=args.n_boot,
            seed=90_000,
        ),
        "head_summaries": {
            f"L{FINAL_LAYER}H{head}": summaries[head]
            for head in range(adapter.num_heads)
        },
        "mlp_summary": mlp_summary,
        "candidate_rule": {
            "min_damage_and_recovery": MIN_EFFECT,
            "both_ci_low_above_zero": True,
            "both_q_bh_32_tests_at_most": MAX_Q,
            "max_candidates": MAX_CANDIDATES,
            "rank": "descending min(damage,recovery), head index tie-break",
        },
        "candidate_gate": {
            "PASS": bool(candidates),
            "interpretation_if_fail": (
                "no head-sparse candidate; final-layer change may be MLP-dominant "
                "or distributed below single-head resolution"
            ),
        },
        "freeze": {
            "candidate_heads": [
                f"L{FINAL_LAYER}H{head}" for head in candidates
            ],
            "control_heads": {
                f"L{FINAL_LAYER}H{candidate}": [
                    f"L{FINAL_LAYER}H{head}" for head in heads
                ]
                for candidate, heads in controls.items()
            },
            "controls_per_candidate": CONTROLS_PER_CANDIDATE,
            "control_tag": "gtex-native-mechanism-controls-v1",
            "validation_authorized": bool(candidates),
        },
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else value
            ),
        )
        + "\n"
    )

    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_mechanism_discovery_v1",
        sys.argv,
        repo=REPO,
        status="EXPLORATORY",
        config={
            "split": "train",
            "n": args.n,
            "batch_size": args.batch,
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "cohort_tag": "gtex-native-mechanism-discovery-v1",
            "final_layer": FINAL_LAYER,
            "heads_scanned": adapter.num_heads,
            "candidate_rule": result["candidate_rule"],
        },
        seeds={
            "checkpoint": 2020,
            "bootstrap_base": 2020,
            "mlp_inference_base": 40_000,
        },
        inputs=[
            file_record(protocol),
            file_record(preflight_path),
            file_record(base_model / "config.json"),
            file_record(model_weight(base_model)),
            file_record(checkpoint),
            file_record(args.gtf),
            file_record(args.fasta),
        ],
        notes=[
            "All 16 final-layer heads were scanned without outcome filtering.",
            "Noising and denoising patch only one head context at the donor token.",
            "The whole MLP is a separate intervention-size family.",
            "Candidate and equal-size control identities are frozen before validation.",
        ],
    )
    manifest.add_output(output)
    manifest.write(manifest_path)

    del task, adapter, model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    concise = {
        "behavior": result["behavior"],
        "candidate_gate": result["candidate_gate"],
        "candidate_heads": result["freeze"]["candidate_heads"],
        "control_heads": result["freeze"]["control_heads"],
        "mlp_summary": mlp_summary,
    }
    print(json.dumps(concise, indent=2))
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
