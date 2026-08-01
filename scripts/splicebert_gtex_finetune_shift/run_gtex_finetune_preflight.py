#!/usr/bin/env python
"""Preflight the published GTEx encoder family and derived MLM instruments.

This runner inspects checkpoint structure, transplant mechanics, tokenizer and
readout invariants, then runs a small base-only behavioral positive/negative
control.  It does not inspect fine-tuned coupling outcomes.
"""

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
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CORRUPT_ALT,
    CouplingTask,
    load_examples,
    pick_device,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402
from gfmcd.splicing.gtex_finetune import (  # noqa: E402
    GtexCheckpointError,
    checkpoint_seed,
    load_gtex_state_dict,
    state_dict_sha256,
    validate_gtex_encoder,
)

EXPECTED_ARCHITECTURE = {
    "hidden_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 16,
    "max_position_embeddings": 1026,
    "vocab_size": 10,
}
MIN_BASE_COUPLING = 0.30
MAX_NONPARTNER_ABS = 0.25
MIN_PARTNER_MINUS_NONPARTNER = 0.30
MIN_DETECTABLE_RELATIVE_SHIFT = 1e-8
MIN_BASE_MLM_HEALTH = 0.20


def _weight_file(model_dir: Path) -> Path:
    for name in ("pytorch_model.bin", "model.safetensors"):
        path = model_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"no single model weight file found in {model_dir}")


def _head_state(model: Any) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {
        name: tensor
        for name, tensor in state.items()
        if name.startswith("cls.predictions.transform.")
        or name in {"cls.predictions.bias", "cls.predictions.decoder.bias"}
    }


def _total_relative_shift(report: Any) -> float:
    delta2 = sum(shift.delta_l2**2 for shift in report.tensor_shifts)
    base2 = sum(shift.base_l2**2 for shift in report.tensor_shifts)
    return float(np.sqrt(delta2 / max(base2, np.finfo(float).eps)))


def _short_forward(adapter: SpliceBertAdapter) -> dict[str, Any]:
    sequence = "ACGTN" * 12
    batch = adapter.tokenize(sequence)
    output = adapter.forward(
        batch,
        output_hidden_states=True,
        output_attentions=False,
    )
    logits = output.logits
    return {
        "shape": list(logits.shape),
        "finite": bool(torch.isfinite(logits).all()),
        "hidden_shape": list(output.last_hidden_state.shape),
    }


def _base_behavior_controls(
    adapter: SpliceBertAdapter,
    *,
    gtf: str,
    fasta: str,
    n: int,
    batch: int,
) -> dict[str, Any]:
    task = CouplingTask(adapter, batch=batch)
    examples, coupling = load_examples(
        gtf,
        fasta,
        "val",
        n,
        task=task,
        tag="gtex-finetune-preflight-base-v1",
        min_effect=None,
    )
    clean_sequences: list[str] = []
    control_sequences: list[str] = []
    donors: list[int] = []
    kept_ids: list[str] = []
    for example in examples:
        candidates = [pos for pos in example.other_ag if pos >= 3]
        if not candidates:
            continue
        decoy = min(candidates, key=lambda pos: abs(pos - example.next_acceptor_pos))
        clean_sequences.append(example.sequence)
        control_sequences.append(
            substitute(example.sequence, decoy - 2, CORRUPT_ALT)
        )
        donors.append(example.donor_pos)
        kept_ids.append(example.example_id)

    control: list[float] = []
    with torch.no_grad():
        for start in range(0, len(donors), batch):
            stop = start + batch
            pos = donors[start:stop]
            clean = task.encode(clean_sequences[start:stop], pos)
            corrupt = task.encode(control_sequences[start:stop], pos)
            clean_margin = task.margin(adapter.model(**clean).logits.float(), pos)
            corrupt_margin = task.margin(adapter.model(**corrupt).logits.float(), pos)
            control.extend((clean_margin - corrupt_margin).tolist())
    nonpartner = float(np.mean(control))
    health = task.mlm_health(examples, n_probe=20, seed=17)
    return {
        "n": len(examples),
        "example_ids": [example.example_id for example in examples],
        "coupling_mean": float(np.mean(coupling)),
        "coupling_median": float(np.median(coupling)),
        "nonpartner_n": len(control),
        "nonpartner_example_ids": kept_ids,
        "nonpartner_mean": nonpartner,
        "partner_minus_nonpartner": float(np.mean(coupling) - nonpartner),
        "mlm_health": health,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO / "configs/splicebert_gtex_finetune_shift/base.json"),
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--derived-root", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    base_dir = Path(args.base_model).resolve()
    archive = Path(args.archive).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    derived_root = Path(args.derived_root).resolve()
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)

    expected = config["published_finetune"]
    expected_hashes = expected["checkpoints"]
    checkpoints = {
        checkpoint_seed(path): path
        for path in sorted(checkpoint_dir.glob("*.pt"))
        if checkpoint_seed(path) is not None
    }
    expected_seeds = set(expected["training"]["seeds"])

    checks: dict[str, bool] = {}
    observed: dict[str, Any] = {
        "thresholds": {
            "min_base_coupling": MIN_BASE_COUPLING,
            "max_nonpartner_abs": MAX_NONPARTNER_ABS,
            "min_partner_minus_nonpartner": MIN_PARTNER_MINUS_NONPARTNER,
            "min_detectable_relative_shift": MIN_DETECTABLE_RELATIVE_SHIFT,
            "min_base_mlm_health": MIN_BASE_MLM_HEALTH,
        },
        "device": device,
    }

    base_weight = _weight_file(base_dir)
    checks["P1_base_hash"] = (
        sha256_file(base_weight) == config["base_checkpoint"]["sha256"]
    )
    checks["P1_archive_hash"] = sha256_file(archive) == expected["archive_sha256"]
    checks["P1_seed_family"] = set(checkpoints) == expected_seeds
    observed["hashes"] = {
        "base": sha256_file(base_weight),
        "archive": sha256_file(archive),
        "checkpoints": {
            str(seed): sha256_file(path) for seed, path in sorted(checkpoints.items())
        },
    }
    checks["P1_checkpoint_hashes"] = all(
        observed["hashes"]["checkpoints"].get(str(seed)) == expected_hashes[str(seed)]
        for seed in expected_seeds
    )

    base_model = AutoModelForMaskedLM.from_pretrained(
        base_dir,
        local_files_only=True,
        attn_implementation="eager",
    )
    architecture = {
        key: int(getattr(base_model.config, key)) for key in EXPECTED_ARCHITECTURE
    }
    observed["architecture"] = architecture
    checks["P2_architecture"] = architecture == EXPECTED_ARCHITECTURE

    reports: dict[str, Any] = {}
    total_shifts: dict[str, float] = {}
    for seed, checkpoint in sorted(checkpoints.items()):
        state = load_gtex_state_dict(checkpoint)
        report = validate_gtex_encoder(
            base_model,
            state,
            checkpoint_path=checkpoint,
        )
        total_shifts[str(seed)] = _total_relative_shift(report)
        reports[str(seed)] = report.to_dict()
    observed["transplants"] = reports
    observed["relative_encoder_shifts"] = total_shifts
    checks["P3_exact_transplants"] = all(
        report["encoder_tensor_count"] == 101
        and not report["missing_encoder_keys"]
        and not report["extra_encoder_keys"]
        and not report["extra_nonencoder_keys"]
        for report in reports.values()
    )
    checks["P4_raw_tensors_finite"] = all(
        all(np.isfinite(shift["max_abs_delta"]) for shift in report["tensor_shifts"])
        for report in reports.values()
    )
    checks["P7_published_shift_detected"] = all(
        value > MIN_DETECTABLE_RELATIVE_SHIFT for value in total_shifts.values()
    )

    identity_state = {
        f"bert.{name}": tensor.detach().clone()
        for name, tensor in base_model.bert.state_dict().items()
    }
    identity_state["classifier.weight"] = torch.zeros(
        3, EXPECTED_ARCHITECTURE["hidden_size"]
    )
    identity_state["classifier.bias"] = torch.zeros(3)
    identity_report = validate_gtex_encoder(
        base_model,
        identity_state,
        checkpoint_path=base_weight,
    )
    identity_shift = _total_relative_shift(identity_report)
    rejected_missing = False
    broken_state = dict(identity_state)
    broken_state.pop(next(key for key in broken_state if key.startswith("bert.")))
    try:
        validate_gtex_encoder(
            base_model,
            broken_state,
            checkpoint_path=base_weight,
        )
    except GtexCheckpointError:
        rejected_missing = True
    observed["negative_controls"] = {
        "identity_relative_shift": identity_shift,
        "missing_encoder_key_rejected": rejected_missing,
    }
    checks["P8_identity_zero"] = identity_shift == 0.0
    checks["P8_missing_key_rejected"] = rejected_missing

    base_head_digest = state_dict_sha256(_head_state(base_model))
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    base_adapter = SpliceBertAdapter(
        base_model,
        tokenizer,
        source=str(base_dir),
        device=device,
    )
    base_meta = base_adapter.metadata()
    observed["base_adapter"] = base_meta.__dict__
    checks["P5_tokenizer_coordinates"] = (
        base_meta.tokenizer_mode == "spaced"
        and base_meta.prefix_tokens == 1
        and base_meta.max_nucleotides == 1024
    )
    base_forward = _short_forward(base_adapter)
    observed["base_forward"] = base_forward
    checks["P4_base_forward_finite"] = base_forward["finite"]

    behavior = _base_behavior_controls(
        base_adapter,
        gtf=args.gtf,
        fasta=args.fasta,
        n=args.n,
        batch=args.batch,
    )
    observed["base_behavior_controls"] = behavior
    checks["P9_base_coupling_positive"] = (
        behavior["coupling_mean"] >= MIN_BASE_COUPLING
    )
    checks["P9_nonpartner_negative"] = (
        abs(behavior["nonpartner_mean"]) <= MAX_NONPARTNER_ABS
        and behavior["partner_minus_nonpartner"] >= MIN_PARTNER_MINUS_NONPARTNER
    )
    checks["P9_base_mlm_health"] = behavior["mlm_health"] >= MIN_BASE_MLM_HEALTH
    del base_adapter, base_model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    derived: dict[str, Any] = {}
    for seed in sorted(expected_seeds):
        model_dir = derived_root / f"seed-{seed}"
        model = AutoModelForMaskedLM.from_pretrained(
            model_dir,
            local_files_only=True,
            attn_implementation="eager",
        )
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        adapter = SpliceBertAdapter(
            model,
            tok,
            source=str(model_dir),
            device=device,
        )
        transplant_record = json.loads((model_dir / "transplant.json").read_text())
        decoder_tied = (
            model.cls.predictions.decoder.weight.data_ptr()
            == model.bert.embeddings.word_embeddings.weight.data_ptr()
        )
        head_digest = state_dict_sha256(_head_state(model))
        forward = _short_forward(adapter)
        derived[str(seed)] = {
            "model_dir": str(model_dir),
            "transplant_checkpoint_sha256": transplant_record[
                "encoder_checkpoint_sha256"
            ],
            "head_digest": head_digest,
            "decoder_tied": decoder_tied,
            "forward": forward,
            "adapter": adapter.metadata().__dict__,
        }
        del adapter, model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    observed["derived_models"] = derived
    checks["P4_derived_forward_finite"] = all(
        record["forward"]["finite"] for record in derived.values()
    )
    checks["P5_derived_tokenizers"] = all(
        record["adapter"]["tokenizer_mode"] == "spaced"
        and record["adapter"]["prefix_tokens"] == 1
        for record in derived.values()
    )
    checks["P6_base_head_preserved"] = all(
        record["head_digest"] == base_head_digest for record in derived.values()
    )
    checks["P6_decoder_tied"] = all(
        record["decoder_tied"] for record in derived.values()
    )
    checks["P6_checkpoint_provenance"] = all(
        record["transplant_checkpoint_sha256"]
        == expected_hashes[str(seed)]
        for seed, record in ((int(seed), record) for seed, record in derived.items())
    )

    passed = all(checks.values())
    result = {
        "study": "splicebert_gtex_finetune_shift",
        "stage": "preflight",
        "status": "PASS" if passed else "FAIL_INSTRUMENT_UNRESOLVED",
        "checks": checks,
        "n_passed": sum(checks.values()),
        "n_checks": len(checks),
        "observed": observed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:preflight",
        sys.argv,
        repo=REPO,
        status=result["status"],
        config={
            "architecture": EXPECTED_ARCHITECTURE,
            "thresholds": observed["thresholds"],
            "n": args.n,
            "batch": args.batch,
            "device": device,
        },
        seeds={"base_behavior_controls": 17},
        notes=[
            "Fine-tuned coupling outcomes are not evaluated in preflight.",
            "The base-only partner/non-partner behavior is the assay positive/negative control.",
            "Identity and missing-key states are numerical and transplant-contract controls.",
        ],
    )
    for path in (config_path, base_weight, archive, Path(args.gtf), Path(args.fasta)):
        manifest.inputs.append(file_record(path))
    for checkpoint in checkpoints.values():
        manifest.inputs.append(file_record(checkpoint))
    for seed in sorted(expected_seeds):
        model_dir = derived_root / f"seed-{seed}"
        manifest.inputs.extend(
            [
                file_record(model_dir / "transplant.json"),
                file_record(model_dir / "transplant.manifest.json"),
                file_record(_weight_file(model_dir)),
            ]
        )
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(f"{result['status']}: {result['n_passed']}/{result['n_checks']} checks")
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
