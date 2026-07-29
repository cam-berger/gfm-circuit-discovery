#!/usr/bin/env python
"""Materialize auditable MLM checkpoints with published GTEx-tuned encoders.

The output directories are analysis instruments, not published checkpoints and
not result artifacts.  Each contains a standard Hugging Face masked-LM model so
the existing SpliceBERT coupling runners can be reused unchanged via
``--model <output-directory>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.gtex_finetune import (  # noqa: E402
    checkpoint_seed,
    transplant_gtex_encoder,
)


def _weight_files(model_dir: Path) -> list[Path]:
    names = ("model.safetensors", "pytorch_model.bin")
    files = [model_dir / name for name in names if (model_dir / name).is_file()]
    if not files:
        files = sorted(model_dir.glob("model-*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model weight file found in {model_dir}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    base = Path(args.base_model).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    out_root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)

    seen_seeds: set[int] = set()
    for checkpoint_arg in args.checkpoint:
        checkpoint = Path(checkpoint_arg).expanduser().resolve()
        seed = checkpoint_seed(checkpoint)
        if seed is None:
            raise ValueError(
                f"cannot identify a 2020--2024 seed from {checkpoint.name}"
            )
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed {seed}")
        seen_seeds.add(seed)

        destination = out_root / f"seed-{seed}"
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing derived checkpoint: {destination}"
            )

        model = AutoModelForMaskedLM.from_pretrained(base, local_files_only=True)
        report = transplant_gtex_encoder(model, checkpoint)
        destination.mkdir(parents=True)
        model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)

        report_path = destination / "transplant.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

        manifest = ExperimentManifest.create(
            "splicebert_gtex_finetune_shift:prepare_mlm_transplant",
            sys.argv,
            repo=REPO,
            status="INSTRUMENT",
            config={
                "seed": seed,
                "base_model": str(base),
                "derived_model": str(destination),
                "readout_policy": report.mlm_head_policy,
            },
            seeds={"fine_tuning": seed},
            notes=[
                "Derived checkpoint: the authors published a token-classification "
                "encoder plus classifier, not a masked-LM checkpoint.",
                "All 101 bert.* tensors are transplanted. The base MLM transform and "
                "output bias are retained; the decoder stays tied to the transplanted "
                "nucleotide embeddings.",
            ],
        )
        for path in _weight_files(base):
            manifest.inputs.append(file_record(path))
        manifest.inputs.append(file_record(checkpoint))
        for path in _weight_files(destination):
            manifest.outputs.append(file_record(path))
        manifest.outputs.append(file_record(report_path))
        manifest.write(destination / "transplant.manifest.json")
        print(f"seed {seed}: {destination}")


if __name__ == "__main__":
    main()
