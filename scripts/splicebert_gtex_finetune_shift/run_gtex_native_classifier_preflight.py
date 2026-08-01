#!/usr/bin/env python
"""Preflight the published native GTEx splice-site classifiers without opening coupling."""

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
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import pick_device  # noqa: E402
from gfmcd.splicing.coupling import perturb_random, substitute  # noqa: E402
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.native_classifier import (  # noqa: E402
    ACCEPTOR_LABEL,
    DONOR_LABEL,
    NativeClassifierTask,
    load_annotation_cohort,
    neutral_dinucleotide_position,
)

EXPECTED_SEEDS = frozenset(range(2020, 2025))
THRESHOLDS = {
    "min_clean_donor_positive_fraction": 0.70,
    "min_clean_acceptor_positive_fraction": 0.70,
    "min_local_drop_mean": 1.0,
    "min_local_drop_positive_fraction": 0.75,
    "max_neutral_fraction_of_local": 0.25,
    "max_projection_error": 1e-5,
}


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
    raise FileNotFoundError(f"base model weight not found in {model_dir}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != 64:
        raise ValueError("native-classifier preflight is frozen at n=64")
    config_path = Path(args.config).resolve()
    protocol_path = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_NATIVE_CLASSIFIER_V2.md"
    )
    frozen = json.loads(config_path.read_text())
    base_model = Path(args.base_model).expanduser().resolve()
    checkpoints = parse_checkpoints(args.checkpoint)
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)

    expected_base_hash = frozen["base_checkpoint"]["sha256"]
    expected_checkpoint_hashes = {
        int(seed): digest
        for seed, digest in frozen["published_finetune"]["checkpoints"].items()
    }
    observed_hashes = {
        seed: sha256_file(path) for seed, path in checkpoints.items()
    }
    provenance_checks = {
        "base_hash": sha256_file(model_weight(base_model)) == expected_base_hash,
        "checkpoint_hashes": observed_hashes == expected_checkpoint_hashes,
    }

    examples = load_annotation_cohort(
        args.gtf,
        args.fasta,
        split="val",
        n=args.n,
        tag="sealed-unc-val",
    )
    neutral_positions = [neutral_dinucleotide_position(example) for example in examples]
    annotation_checks = {
        "cohort_size": len(examples) == args.n,
        "donor_motifs": all(
            example.sequence[example.donor_pos : example.donor_pos + 2] == "GT"
            for example in examples
        ),
        "acceptor_motifs": all(
            example.sequence[
                int(example.next_acceptor_pos) - 2 : int(example.next_acceptor_pos)
            ]
            == "AG"
            for example in examples
        ),
        "neutral_controls_available": all(
            position is not None for position in neutral_positions
        ),
    }
    if not annotation_checks["neutral_controls_available"]:
        raise RuntimeError("a frozen preflight example has no neutral control position")

    clean_sequences = [example.sequence for example in examples]
    local_cc_sequences = [
        substitute(example.sequence, example.donor_pos, "CC")
        for example in examples
    ]
    local_aa_sequences = [
        substitute(example.sequence, example.donor_pos, "AA")
        for example in examples
    ]
    neutral_sequences = [
        perturb_random(example.sequence, int(position), 2)
        for example, position in zip(examples, neutral_positions)
    ]
    donor_label_positions = [example.donor_pos - 1 for example in examples]
    acceptor_positions = [int(example.next_acceptor_pos) for example in examples]

    seed_results: dict[str, Any] = {}
    all_seed_checks: dict[str, bool] = {}
    for seed, checkpoint in sorted(checkpoints.items()):
        print(f"seed {seed}: native-classifier preflight on {len(examples)} examples")
        model, tokenizer, report = load_gtex_token_classifier(
            base_model,
            checkpoint,
            device=device,
        )
        task = NativeClassifierTask(
            model,
            tokenizer,
            device=device,
            batch_size=args.batch,
        )
        encoded = task.encode(clean_sequences[:1])
        tokenizer_coordinates = (
            int(encoded["input_ids"].shape[1]) == len(clean_sequences[0]) + 2
            and int(encoded["input_ids"][0, donor_label_positions[0] + 2])
            == tokenizer.convert_tokens_to_ids("G")
            and int(encoded["input_ids"][0, acceptor_positions[0] + 1])
            == tokenizer.convert_tokens_to_ids(
                clean_sequences[0][acceptor_positions[0]]
            )
        )
        projection_error = task.final_projection_error(clean_sequences[:2])

        clean_logits = task.logits(clean_sequences)
        cc_logits = task.logits(local_cc_sequences)
        aa_logits = task.logits(local_aa_sequences)
        neutral_logits = task.logits(neutral_sequences)
        clean_donor = task.score(
            clean_logits,
            donor_label_positions,
            label=DONOR_LABEL,
        )
        clean_acceptor = task.score(
            clean_logits,
            acceptor_positions,
            label=ACCEPTOR_LABEL,
        )
        cc_margin = task.score(
            cc_logits,
            donor_label_positions,
            label=DONOR_LABEL,
        ).margin
        aa_margin = task.score(
            aa_logits,
            donor_label_positions,
            label=DONOR_LABEL,
        ).margin
        neutral_margin = task.score(
            neutral_logits,
            donor_label_positions,
            label=DONOR_LABEL,
        ).margin
        cc_drop = clean_donor.margin - cc_margin
        aa_drop = clean_donor.margin - aa_margin
        neutral_drop = clean_donor.margin - neutral_margin
        local_floor = min(float(cc_drop.mean()), float(aa_drop.mean()))

        observed = {
            "clean_donor_margin_mean": float(clean_donor.margin.mean()),
            "clean_donor_probability_mean": float(clean_donor.probability.mean()),
            "clean_donor_positive_fraction": float(
                np.mean(clean_donor.margin > 0)
            ),
            "clean_acceptor_margin_mean": float(clean_acceptor.margin.mean()),
            "clean_acceptor_probability_mean": float(
                clean_acceptor.probability.mean()
            ),
            "clean_acceptor_positive_fraction": float(
                np.mean(clean_acceptor.margin > 0)
            ),
            "local_cc_drop_mean": float(cc_drop.mean()),
            "local_cc_drop_positive_fraction": float(np.mean(cc_drop > 0)),
            "local_aa_drop_mean": float(aa_drop.mean()),
            "local_aa_drop_positive_fraction": float(np.mean(aa_drop > 0)),
            "neutral_drop_mean": float(neutral_drop.mean()),
            "neutral_abs_mean_to_local_ratio": (
                abs(float(neutral_drop.mean())) / max(local_floor, 1e-12)
            ),
            "final_projection_max_abs_error": projection_error,
        }
        checks = {
            "strict_checkpoint": not report.missing_keys and not report.extra_keys,
            "tokenizer_coordinates": tokenizer_coordinates,
            "finite_forward": all(
                bool(np.isfinite(value).all())
                for value in (
                    clean_logits,
                    cc_logits,
                    aa_logits,
                    neutral_logits,
                )
            ),
            "final_projection": (
                projection_error <= THRESHOLDS["max_projection_error"]
            ),
            "clean_donor_recognition": (
                observed["clean_donor_positive_fraction"]
                >= THRESHOLDS["min_clean_donor_positive_fraction"]
            ),
            "clean_acceptor_recognition": (
                observed["clean_acceptor_positive_fraction"]
                >= THRESHOLDS["min_clean_acceptor_positive_fraction"]
            ),
            "local_cc_sensitivity": (
                observed["local_cc_drop_mean"] >= THRESHOLDS["min_local_drop_mean"]
                and observed["local_cc_drop_positive_fraction"]
                >= THRESHOLDS["min_local_drop_positive_fraction"]
            ),
            "local_aa_sensitivity": (
                observed["local_aa_drop_mean"] >= THRESHOLDS["min_local_drop_mean"]
                and observed["local_aa_drop_positive_fraction"]
                >= THRESHOLDS["min_local_drop_positive_fraction"]
            ),
            "neutral_specificity": (
                observed["neutral_abs_mean_to_local_ratio"]
                <= THRESHOLDS["max_neutral_fraction_of_local"]
            ),
        }
        seed_results[str(seed)] = {
            "report": report.to_dict(),
            "observed": observed,
            "checks": checks,
            "PASS": all(checks.values()),
            "raw": [
                {
                    "example_id": example.example_id,
                    "gene_id": example.gene_id,
                    "neutral_position": int(neutral_positions[index]),
                    "clean_donor_margin": float(clean_donor.margin[index]),
                    "clean_acceptor_margin": float(clean_acceptor.margin[index]),
                    "local_cc_drop": float(cc_drop[index]),
                    "local_aa_drop": float(aa_drop[index]),
                    "neutral_drop": float(neutral_drop[index]),
                }
                for index, example in enumerate(examples)
            ],
        }
        for name, passed in checks.items():
            all_seed_checks[f"seed_{seed}:{name}"] = bool(passed)
        del task, model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    checks = {
        **provenance_checks,
        **annotation_checks,
        **all_seed_checks,
    }
    artifact = {
        "study": "splicebert_gtex_finetune_shift",
        "stage": "native classifier preflight",
        "protocol": str(protocol_path.relative_to(REPO)),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "thresholds": THRESHOLDS,
        "checks": checks,
        "n_passed": int(sum(checks.values())),
        "n_checks": len(checks),
        "device": device,
        "hashes": {
            "base": sha256_file(model_weight(base_model)),
            "checkpoints": observed_hashes,
        },
        "seeds": seed_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, default=_json_default) + "\n")

    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:native_classifier_preflight",
        sys.argv,
        repo=REPO,
        status=artifact["status"],
        config={
            "split": "val",
            "cohort_tag": "sealed-unc-val",
            "n": args.n,
            "batch": args.batch,
            "device": device,
            "thresholds": THRESHOLDS,
            "paired_acceptor_outcomes_opened": False,
            "donor_label_coordinate": "donor_motif_start_minus_1",
        },
        notes=[
            "Frozen preflight inspects only clean splice sites, local donor destruction, "
            "and a deterministic distant non-motif mutation.",
            "All five published seeds are required and run sequentially.",
            "v2 corrects only the donor output coordinate caught by immutable v1.",
        ],
    )
    for path in (
        config_path,
        protocol_path,
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
    print(f"{artifact['status']}: saved -> {output}")
    if artifact["status"] != "PASS":
        raise RuntimeError("native-classifier preflight failed; target remains frozen")


if __name__ == "__main__":
    main()
