"""Self-fine-tune DNABERT-2 for promoter classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from gfmcd.dnabert.promoter_classifier import (
    DEFAULT_MODEL_NAME,
    DnaBertPromoterClassifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)

    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("results/dnabert_promoter/shared/train.csv"),
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=Path("results/dnabert_promoter/shared/validation.csv"),
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("results/dnabert_promoter/shared/test.csv"),
    )

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=24)

    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
    )

    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--max-test-examples", type=int)
    parser.add_argument("--overwrite-record", action="store_true")

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_split(
    path: Path,
    maximum: int | None,
    seed: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "split",
        "source_index",
        "sequence",
        "label",
        "sequence_sha256",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(
            f"Unexpected labels in {path}: "
            f"{sorted(frame['label'].unique())}"
        )

    if maximum is not None and maximum < len(frame):
        frame = (
            frame.sample(
                n=maximum,
                random_state=seed,
            )
            .sort_index()
            .reset_index(drop=True)
        )

    return frame.reset_index(drop=True)


def to_dataset(frame: pd.DataFrame) -> Dataset:
    return Dataset.from_dict(
        {
            "sequence": frame["sequence"].astype(str).tolist(),
            "labels": frame["label"].astype(int).tolist(),
            "sequence_sha256": frame[
                "sequence_sha256"
            ].astype(str).tolist(),
        }
    )


def metric_dict(logits, labels) -> dict[str, float]:
    logits = np.asarray(logits)
    labels = np.asarray(labels).astype(int)

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=-1,
    ).numpy()[:, 1]

    predictions = logits.argmax(axis=-1)

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(labels, probabilities)
        ),
        "f1": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
    }


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    return metric_dict(logits, labels)


def prediction_frame(
    source: pd.DataFrame,
    logits,
    labels,
) -> pd.DataFrame:
    logits = np.asarray(logits)
    labels = np.asarray(labels).astype(int)

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=-1,
    ).numpy()[:, 1]

    predictions = logits.argmax(axis=-1)

    output = source[
        [
            "split",
            "source_index",
            "sequence_sha256",
            "label",
        ]
    ].copy()

    output["logit_0"] = logits[:, 0]
    output["logit_1"] = logits[:, 1]
    output["promoter_probability"] = probabilities
    output["prediction"] = predictions
    output["correct"] = predictions == labels

    return output


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.output_dir.exists():
        has_files = any(args.output_dir.iterdir())

        if has_files and not args.overwrite_record:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_frame = load_split(
        args.train_path,
        args.max_train_examples,
        args.seed,
    )
    validation_frame = load_split(
        args.validation_path,
        args.max_validation_examples,
        args.seed,
    )
    test_frame = load_split(
        args.test_path,
        args.max_test_examples,
        args.seed,
    )

    print("Train examples:", len(train_frame))
    print("Validation examples:", len(validation_frame))
    print("Test examples:", len(test_frame))

    model = DnaBertPromoterClassifier(
        model_name=args.model_name,
        num_labels=2,
        dropout=0.1,
    )

    tokenizer = model.tokenizer

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Total parameters:", f"{total_parameters:,}")
    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )
    print(
        "Trainable percentage:",
        f"{100 * trainable_parameters / total_parameters:.2f}%",
    )

    if trainable_parameters < 0.95 * total_parameters:
        raise RuntimeError(
            "Fewer than 95% of model parameters are trainable."
        )

    train_dataset = to_dataset(train_frame)
    validation_dataset = to_dataset(validation_frame)
    test_dataset = to_dataset(test_frame)

    def tokenize_batch(batch):
        return tokenizer(
            batch["sequence"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    columns_to_remove = [
        "sequence",
        "sequence_sha256",
    ]

    train_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=columns_to_remove,
    )
    validation_dataset = validation_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=columns_to_remove,
    )
    test_dataset = test_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=columns_to_remove,
    )

    trainer_dir = args.output_dir / "trainer"

    training_args_kwargs = {
        "output_dir": str(trainer_dir),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": (
            args.train_batch_size
        ),
        "per_device_eval_batch_size": (
            args.eval_batch_size
        ),
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "balanced_accuracy",
        "greater_is_better": True,
        "save_total_limit": 2,
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "dataloader_num_workers": (
            args.dataloader_num_workers
        ),
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "remove_unused_columns": True,
    }

    signature = __import__("inspect").signature(
        TrainingArguments.__init__
    )

    if "eval_strategy" in signature.parameters:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs[
            "evaluation_strategy"
        ] = "epoch"

    training_args = TrainingArguments(
        **training_args_kwargs
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "compute_metrics": compute_metrics,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=(
                    args.early_stopping_patience
                )
            )
        ],
    }

    trainer_signature = __import__("inspect").signature(
        Trainer.__init__
    )

    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("\nStarting training...")
    train_result = trainer.train()

    print("\nEvaluating validation split...")
    validation_output = trainer.predict(
        validation_dataset
    )

    print("\nEvaluating sealed test split...")
    test_output = trainer.predict(test_dataset)

    validation_metrics = metric_dict(
        validation_output.predictions,
        validation_output.label_ids,
    )
    test_metrics = metric_dict(
        test_output.predictions,
        test_output.label_ids,
    )

    validation_predictions = prediction_frame(
        validation_frame,
        validation_output.predictions,
        validation_output.label_ids,
    )
    test_predictions = prediction_frame(
        test_frame,
        test_output.predictions,
        test_output.label_ids,
    )

    validation_predictions.to_csv(
        args.output_dir / "validation_predictions.csv",
        index=False,
    )
    test_predictions.to_csv(
        args.output_dir / "test_predictions.csv",
        index=False,
    )

    validation_confusion = confusion_matrix(
        validation_output.label_ids,
        validation_output.predictions.argmax(axis=-1),
    )
    test_confusion = confusion_matrix(
        test_output.label_ids,
        test_output.predictions.argmax(axis=-1),
    )

    best_checkpoint_dir = (
        args.output_dir / "best_checkpoint"
    )
    best_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        trainer.model.state_dict(),
        best_checkpoint_dir / "model_state_dict.pt",
    )

    tokenizer.save_pretrained(
        best_checkpoint_dir / "tokenizer"
    )

    trainer.model.encoder.config.save_pretrained(
        best_checkpoint_dir / "encoder_config"
    )

    model_manifest = {
        "architecture": "DnaBertPromoterClassifier",
        "base_model": args.model_name,
        "pooling": "attention-mask-aware mean",
        "hidden_size": int(
            trainer.model.encoder.config.hidden_size
        ),
        "num_labels": 2,
        "dropout": 0.1,
        "seed": args.seed,
        "max_length": args.max_length,
    }

    with (
        best_checkpoint_dir / "model_manifest.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(model_manifest, handle, indent=2)

    summary = {
        "experiment": "dnabert_promoter_self_finetuning",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git_commit": git_commit(),
        "seed": args.seed,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "train_examples": len(train_frame),
        "validation_examples": len(validation_frame),
        "test_examples": len(test_frame),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "best_model_checkpoint": (
            trainer.state.best_model_checkpoint
        ),
        "best_metric": trainer.state.best_metric,
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_confusion_matrix": (
            validation_confusion.tolist()
        ),
        "test_confusion_matrix": (
            test_confusion.tolist()
        ),
        "training_arguments": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),
            "warmup_ratio": args.warmup_ratio,
            "early_stopping_patience": (
                args.early_stopping_patience
            ),
        },
        "input_hashes": {
            "train": sha256_file(args.train_path),
            "validation": sha256_file(
                args.validation_path
            ),
            "test": sha256_file(args.test_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sklearn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
        },
    }

    with (
        args.output_dir / "summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    trainer.state.save_to_json(
        str(args.output_dir / "trainer_state.json")
    )

    print("\n===== VALIDATION METRICS =====")
    print(json.dumps(validation_metrics, indent=2))

    print("\n===== TEST METRICS =====")
    print(json.dumps(test_metrics, indent=2))

    print("\nCheckpoint saved to:")
    print(best_checkpoint_dir)


if __name__ == "__main__":
    main()
