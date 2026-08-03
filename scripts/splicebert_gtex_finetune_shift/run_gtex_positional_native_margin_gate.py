#!/usr/bin/env python
"""Screen inherited positional carriers on the released GTEx classifier task.

This is an exploratory, classifier-native gate on a reused cohort.  It tests
the two named pretrained carriers (period 3 and approximately 10.63 nt) with
deletion, half deletion, base-carrier replacement, and three equal-RMS
same-band phase controls.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import (  # noqa: E402
    ExperimentManifest,
    file_record,
    sha256_file,
)
from gfmcd.rigor import benjamini_hochberg  # noqa: E402
from gfmcd.splicing.circuit import _dataset, eligible, pick_device  # noqa: E402
from gfmcd.splicing.coupling import MOTIF_CLEAN_DONOR_ALT, CouplingExample, substitute  # noqa: E402
from gfmcd.splicing.grouped_stats import (  # noqa: E402
    grouped_bootstrap,
    grouped_signflip_pvalue,
)
from gfmcd.splicing.gtex_finetune import (  # noqa: E402
    load_gtex_state_dict,
    load_gtex_token_classifier,
)
from gfmcd.splicing.native_classifier import (  # noqa: E402
    ACCEPTOR_LABEL,
    DONOR_LABEL,
    NativeClassifierTask,
)
from gfmcd.splicing.positional_spectrum import (  # noqa: E402
    centered_spectrum,
    phase_rotate_band,
    replace_band,
    scale_band,
    validate_table,
)

EXPECTED_SEEDS = frozenset(range(2020, 2025))
EXPECTED_N = 64
EXPECTED_BOOT = 5000
EXPECTED_PERM = 9999
EXPECTED_NAMED_BANDS = {
    "period3": tuple(range(339, 346)),
    "period10": tuple(range(94, 101)),
}
PHASE_ANGLE = np.pi / 3.0
PHASE_PATTERNS = {
    "all_positive": (1, 1, 1, 1, 1, 1, 1),
    "alternating": (1, -1, 1, -1, 1, -1, 1),
    "mixed": (1, 1, -1, 1, -1, -1, 1),
}
ENDPOINTS = ("donor_margin", "acceptor_margin", "donor_coupling")
PRIMARY_ESTIMANDS = (
    "deletion_damage",
    "base_band_rescue",
    "deletion_minus_phase",
)
SPECTRAL_SUPPORT_TOLERANCE = 1e-10
PHASE_POWER_TOLERANCE = 1e-10
PHASE_DOSE_TOLERANCE = 1e-10
MIN_INTACT_LOCAL_DESTRUCTION = 5.0
MIN_HEALTH_RETENTION = 0.80
MIN_SCREEN_EFFECT = 0.05
MAX_Q = 0.05


def parse_checkpoints(values: Sequence[str]) -> dict[int, Path]:
    """Parse and require the complete released five-checkpoint family."""

    checkpoints: dict[int, Path] = {}
    for value in values:
        try:
            seed_text, path_text = value.split("=", 1)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError("--checkpoint must be SEED=PATH") from exc
        if seed in checkpoints:
            raise ValueError(f"duplicate checkpoint seed {seed}")
        checkpoints[seed] = Path(path_text).expanduser().resolve()
    if set(checkpoints) != EXPECTED_SEEDS:
        raise ValueError(f"complete seed family required; found {sorted(checkpoints)}")
    return checkpoints


def _weight_file(model_dir: Path) -> Path:
    for name in ("pytorch_model.bin", "model.safetensors"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no model weights in {model_dir}")


def _position_table_from_weight_file(path: Path) -> np.ndarray:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state: Mapping[str, Any] = load_file(str(path), device="cpu")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError("base model weights are not a state mapping")
        state = payload
    hits = [
        tensor
        for name, tensor in state.items()
        if name.endswith("position_embeddings.weight")
    ]
    if len(hits) != 1 or not isinstance(hits[0], torch.Tensor):
        raise ValueError("weights must expose exactly one position embedding table")
    return validate_table(
        hits[0].detach().cpu().to(dtype=torch.float64).numpy(),
        shape=(1026, 512),
    )


def _position_table_from_checkpoint(path: Path) -> np.ndarray:
    state = load_gtex_state_dict(path)
    hits = [
        tensor
        for name, tensor in state.items()
        if name.endswith("position_embeddings.weight")
    ]
    if len(hits) != 1:
        raise ValueError("checkpoint must expose exactly one position embedding table")
    return validate_table(
        hits[0].detach().cpu().to(dtype=torch.float64).numpy(),
        shape=(1026, 512),
    )


def _embedding_table_cpu_float64(weight: torch.Tensor) -> np.ndarray:
    """Transfer off accelerator before the float64 conversion.

    MPS does not support float64, so the CPU transfer must be a separate first
    operation.  This ordering is part of the assay's numerical contract.
    """

    return validate_table(
        weight.detach().cpu().to(dtype=torch.float64).numpy(),
        shape=(1026, 512),
    )


def _copy_embedding_table(model: Any, table: np.ndarray) -> None:
    embedding = model.bert.embeddings.position_embeddings
    tensor = torch.as_tensor(
        table,
        dtype=embedding.weight.dtype,
        device=embedding.weight.device,
    )
    if tensor.shape != embedding.weight.shape:
        raise ValueError("surgery table does not match native position embedding")
    with torch.no_grad():
        embedding.weight.copy_(tensor)


def _arm_tables(
    original: np.ndarray,
    base: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Construct the frozen intervention family."""

    tables: dict[str, np.ndarray] = {"intact": original.copy()}
    specs: dict[str, dict[str, Any]] = {
        "intact": {"band": None, "intervention": "identity", "bins": []}
    }
    for band_name, bins in EXPECTED_NAMED_BANDS.items():
        arms = {
            f"{band_name}__delete": scale_band(original, bins, 0.0),
            f"{band_name}__half": scale_band(original, bins, 0.5),
            f"{band_name}__base_replace": replace_band(original, base, bins),
        }
        tables.update(arms)
        specs[f"{band_name}__delete"] = {
            "band": band_name,
            "intervention": "delete",
            "bins": list(bins),
        }
        specs[f"{band_name}__half"] = {
            "band": band_name,
            "intervention": "half_delete",
            "bins": list(bins),
        }
        specs[f"{band_name}__base_replace"] = {
            "band": band_name,
            "intervention": "base_replace",
            "bins": list(bins),
        }
        for pattern_name, signs in PHASE_PATTERNS.items():
            arm_name = f"{band_name}__phase_{pattern_name}"
            phases = [float(sign) * PHASE_ANGLE for sign in signs]
            tables[arm_name] = phase_rotate_band(original, bins, phases)
            specs[arm_name] = {
                "band": band_name,
                "intervention": "phase_control",
                "phase_pattern": pattern_name,
                "phase_signs": list(signs),
                "phase_angle_radians": float(PHASE_ANGLE),
                "bins": list(bins),
            }
    return tables, specs


def _support_audit(
    original: np.ndarray,
    edited: np.ndarray,
    bins: Sequence[int],
) -> dict[str, Any]:
    base_spectrum = centered_spectrum(original)
    edited_spectrum = centered_spectrum(edited)
    delta = edited_spectrum - base_spectrum
    outside = np.ones(delta.shape[0], dtype=bool)
    outside[list(bins)] = False
    scale = max(float(np.max(np.abs(base_spectrum))), np.finfo(np.float64).tiny)
    relative_error = float(np.max(np.abs(delta[outside]))) / scale
    return {
        "outside_band_max_relative_delta": relative_error,
        "outside_band_pass": relative_error <= SPECTRAL_SUPPORT_TOLERANCE,
        "edited_is_real": bool(np.isrealobj(edited)),
        "edited_finite": bool(np.isfinite(edited).all()),
    }


def _band_power(table: np.ndarray, bins: Sequence[int]) -> float:
    spectrum = centered_spectrum(table)
    return float(np.square(np.abs(spectrum[list(bins)])).sum())


def _mechanical_audits(
    original: np.ndarray,
    tables: Mapping[str, np.ndarray],
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    audits: dict[str, dict[str, Any]] = {}
    for name, table in tables.items():
        if name == "intact":
            continue
        bins = tuple(int(value) for value in specs[name]["bins"])
        row = _support_audit(original, table, bins)
        row["table_rms_delta"] = float(np.sqrt(np.mean(np.square(table - original))))
        if specs[name]["intervention"] == "phase_control":
            original_power = _band_power(original, bins)
            edited_power = _band_power(table, bins)
            deletion = scale_band(original, bins, 0.0)
            deletion_dose = float(
                np.sqrt(np.mean(np.square(deletion - original)))
            )
            phase_dose = float(np.sqrt(np.mean(np.square(table - original))))
            row.update(
                {
                    "band_power_relative_error": abs(
                        edited_power - original_power
                    )
                    / max(original_power, np.finfo(np.float64).tiny),
                    "dose_relative_error_vs_deletion": abs(
                        phase_dose - deletion_dose
                    )
                    / max(deletion_dose, np.finfo(np.float64).tiny),
                }
            )
            row["band_power_pass"] = (
                row["band_power_relative_error"] <= PHASE_POWER_TOLERANCE
            )
            row["dose_match_pass"] = (
                row["dose_relative_error_vs_deletion"] <= PHASE_DOSE_TOLERANCE
            )
        audits[name] = row
    return audits


def _load_replayed_cohort(
    gtf: Path,
    fasta: Path,
    cohort_source: Mapping[str, Any],
) -> tuple[list[CouplingExample], dict[str, bool]]:
    cohort = cohort_source.get("cohort", {})
    expected_ids = [str(value) for value in cohort.get("example_ids", [])[:EXPECTED_N]]
    expected_genes = [str(value) for value in cohort.get("gene_ids", [])[:EXPECTED_N]]
    expected_chromosomes = [
        str(value) for value in cohort.get("chromosomes", [])[:EXPECTED_N]
    ]
    if not (
        len(expected_ids)
        == len(expected_genes)
        == len(expected_chromosomes)
        == EXPECTED_N
    ):
        raise RuntimeError("cohort source does not contain the frozen first 64 records")
    candidates = eligible(_dataset(str(gtf), str(fasta)), "train")
    by_id: dict[str, CouplingExample] = {}
    duplicate_ids: set[str] = set()
    for example in candidates:
        if example.example_id in by_id:
            duplicate_ids.add(example.example_id)
        by_id[example.example_id] = example
    if duplicate_ids.intersection(expected_ids):
        raise RuntimeError("frozen cohort contains duplicate annotation IDs")
    missing = [example_id for example_id in expected_ids if example_id not in by_id]
    if missing:
        raise RuntimeError(f"could not reconstruct {len(missing)} frozen examples")
    examples = [by_id[example_id] for example_id in expected_ids]
    checks = {
        "n": len(examples) == EXPECTED_N,
        "example_ids_exact": (
            [example.example_id for example in examples] == expected_ids
        ),
        "gene_ids_exact": (
            [example.gene_id for example in examples] == expected_genes
        ),
        "chromosomes_exact": (
            [example.chrom for example in examples] == expected_chromosomes
        ),
        "one_example_per_gene": len({example.gene_id for example in examples})
        == EXPECTED_N,
        "train_split": all(example.split == "train" for example in examples),
        "sequence_length_1024": all(len(example.sequence) == 1024 for example in examples),
        "local_gt_intact": all(
            example.sequence[example.donor_pos : example.donor_pos + 2] == "GT"
            for example in examples
        ),
        "partner_ag_intact": all(
            example.next_acceptor_pos is not None
            and example.sequence[
                int(example.next_acceptor_pos) - 2 : int(example.next_acceptor_pos)
            ]
            == "AG"
            for example in examples
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen cohort replay failed: {checks}")
    return examples, checks


def _score_arm(
    task: NativeClassifierTask,
    clean_sequences: Sequence[str],
    local_sequences: Sequence[str],
    paired_sequences: Sequence[str],
    donor_positions: Sequence[int],
    acceptor_positions: Sequence[int],
) -> dict[str, np.ndarray]:
    combined = task.logits([*clean_sequences, *local_sequences, *paired_sequences])
    n = len(clean_sequences)
    clean_logits = combined[:n]
    local_logits = combined[n : 2 * n]
    paired_logits = combined[2 * n :]
    donor = task.score(clean_logits, donor_positions, label=DONOR_LABEL)
    acceptor = task.score(clean_logits, acceptor_positions, label=ACCEPTOR_LABEL)
    local_margin = task.score(
        local_logits, donor_positions, label=DONOR_LABEL
    ).margin
    paired_margin = task.score(
        paired_logits, donor_positions, label=DONOR_LABEL
    ).margin
    return {
        "donor_margin": donor.margin,
        "donor_probability": donor.probability,
        "acceptor_margin": acceptor.margin,
        "acceptor_probability": acceptor.probability,
        "paired_donor_margin": paired_margin,
        "donor_coupling": donor.margin - paired_margin,
        "local_destruction": donor.margin - local_margin,
    }


def run_seed(
    base_model: Path,
    checkpoint: Path,
    base_table: np.ndarray,
    clean_sequences: Sequence[str],
    local_sequences: Sequence[str],
    paired_sequences: Sequence[str],
    donor_positions: Sequence[int],
    acceptor_positions: Sequence[int],
    *,
    device: str,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    model, tokenizer, report = load_gtex_token_classifier(
        base_model,
        checkpoint,
        device=device,
        attn_implementation="eager",
    )
    task = NativeClassifierTask(
        model,
        tokenizer,
        device=device,
        batch_size=batch_size,
    )
    original = _embedding_table_cpu_float64(
        model.bert.embeddings.position_embeddings.weight
    )
    raw_checkpoint = _position_table_from_checkpoint(checkpoint)
    transfer_error = float(np.max(np.abs(original - raw_checkpoint)))
    tables, specs = _arm_tables(original, base_table)
    audits = _mechanical_audits(original, tables, specs)

    raw: dict[str, dict[str, np.ndarray]] = {}
    for name, table in tables.items():
        print(f"    arm {name}", flush=True)
        _copy_embedding_table(model, table)
        raw[name] = _score_arm(
            task,
            clean_sequences,
            local_sequences,
            paired_sequences,
            donor_positions,
            acceptor_positions,
        )
    _copy_embedding_table(model, original)

    intact_local = float(np.mean(raw["intact"]["local_destruction"]))
    health: dict[str, dict[str, Any]] = {}
    for name, metrics in raw.items():
        local_mean = float(np.mean(metrics["local_destruction"]))
        health[name] = {
            "local_destruction_mean": local_mean,
            "local_retention_fraction": local_mean / intact_local,
            "donor_margin_mean": float(np.mean(metrics["donor_margin"])),
            "donor_positive_fraction": float(np.mean(metrics["donor_margin"] > 0)),
            "acceptor_margin_mean": float(np.mean(metrics["acceptor_margin"])),
            "acceptor_positive_fraction": float(
                np.mean(metrics["acceptor_margin"] > 0)
            ),
            "finite": bool(
                all(np.isfinite(values).all() for values in metrics.values())
            ),
        }
    phase_audits = [
        row
        for name, row in audits.items()
        if specs[name]["intervention"] == "phase_control"
    ]
    checks = {
        "device_table_exact": transfer_error == 0.0,
        "all_tables_finite_and_real": all(
            row["edited_finite"] and row["edited_is_real"] for row in audits.values()
        ),
        "spectral_support": all(
            row["outside_band_pass"] for row in audits.values()
        ),
        "phase_power": all(row["band_power_pass"] for row in phase_audits),
        "phase_dose": all(row["dose_match_pass"] for row in phase_audits),
        "intact_local_destruction": intact_local
        >= MIN_INTACT_LOCAL_DESTRUCTION,
        "all_arm_health": all(
            row["finite"]
            and row["local_retention_fraction"] >= MIN_HEALTH_RETENTION
            for row in health.values()
        ),
        "all_logits_finite": all(row["finite"] for row in health.values()),
    }
    preflight = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "checkpoint": report.to_dict(),
        "device_table_transfer_max_abs_error": transfer_error,
        "arm_specs": specs,
        "mechanical_audits": audits,
        "arm_health": health,
        "thresholds": {
            "spectral_support_max_relative_error": SPECTRAL_SUPPORT_TOLERANCE,
            "phase_power_max_relative_error": PHASE_POWER_TOLERANCE,
            "phase_dose_max_relative_error": PHASE_DOSE_TOLERANCE,
            "minimum_intact_local_destruction": MIN_INTACT_LOCAL_DESTRUCTION,
            "minimum_arm_health_retention": MIN_HEALTH_RETENTION,
        },
    }
    del task, model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return preflight, raw


def effect_arrays(
    raw: Mapping[str, Mapping[str, np.ndarray]],
    band_name: str,
    endpoint: str,
) -> dict[str, np.ndarray]:
    """Compute all frozen paired estimands for one band and endpoint."""

    intact = raw["intact"][endpoint]
    deleted = raw[f"{band_name}__delete"][endpoint]
    half = raw[f"{band_name}__half"][endpoint]
    base_replaced = raw[f"{band_name}__base_replace"][endpoint]
    phase_values = np.stack(
        [
            raw[f"{band_name}__phase_{pattern_name}"][endpoint]
            for pattern_name in PHASE_PATTERNS
        ],
        axis=0,
    )
    phase_mean = phase_values.mean(axis=0)
    deletion_damage = intact - deleted
    phase_damage = intact - phase_mean
    return {
        "deletion_damage": deletion_damage,
        "half_deletion_damage": intact - half,
        "base_band_rescue": base_replaced - deleted,
        "base_rewrite": intact - base_replaced,
        "mean_phase_damage": phase_damage,
        "deletion_minus_phase": deletion_damage - phase_damage,
    }


def _summary(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    bootstrap_seed: int,
    permutation_seed: int | None,
) -> dict[str, Any]:
    result = grouped_bootstrap(
        values,
        groups,
        n_boot=EXPECTED_BOOT,
        seed=bootstrap_seed,
    )
    if permutation_seed is not None:
        result["p_signflip_add_one"] = grouped_signflip_pvalue(
            values,
            groups,
            n_perm=EXPECTED_PERM,
            seed=permutation_seed,
        )
    return result


def infer(
    raw_by_seed: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    groups: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Average checkpoints within example, infer the frozen 18-test family."""

    inference: dict[str, Any] = {}
    pvalue_locations: list[tuple[str, str, str]] = []
    pvalues: list[float] = []
    counter = 0
    for band_name in EXPECTED_NAMED_BANDS:
        inference[band_name] = {}
        for endpoint in ENDPOINTS:
            by_seed = {
                seed: effect_arrays(raw_by_seed[seed], band_name, endpoint)
                for seed in sorted(raw_by_seed)
            }
            row: dict[str, Any] = {
                "per_seed_means": {
                    str(seed): {
                        estimand: float(np.mean(values))
                        for estimand, values in by_seed[seed].items()
                    }
                    for seed in sorted(by_seed)
                },
                "all_five_positive": {
                    estimand: all(
                        float(np.mean(by_seed[seed][estimand])) > 0
                        for seed in by_seed
                    )
                    for estimand in PRIMARY_ESTIMANDS
                },
                "aggregate": {},
            }
            for estimand in (
                "deletion_damage",
                "half_deletion_damage",
                "base_band_rescue",
                "base_rewrite",
                "mean_phase_damage",
                "deletion_minus_phase",
            ):
                values = np.mean(
                    np.stack(
                        [by_seed[seed][estimand] for seed in sorted(by_seed)],
                        axis=0,
                    ),
                    axis=0,
                )
                primary = estimand in PRIMARY_ESTIMANDS
                summary = _summary(
                    values,
                    groups,
                    bootstrap_seed=740_000 + counter * 10 + 1,
                    permutation_seed=(
                        840_000 + counter * 10 + 2 if primary else None
                    ),
                )
                row["aggregate"][estimand] = summary
                if primary:
                    pvalue_locations.append((band_name, endpoint, estimand))
                    pvalues.append(float(summary["p_signflip_add_one"]))
                counter += 1
            inference[band_name][endpoint] = row

    qvalues = benjamini_hochberg(pvalues)
    for (band_name, endpoint, estimand), qvalue in zip(
        pvalue_locations, qvalues, strict=True
    ):
        inference[band_name][endpoint]["aggregate"][estimand][
            "q_bh_18_tests"
        ] = float(qvalue)

    screens: dict[str, dict[str, Any]] = {}
    for band_name in EXPECTED_NAMED_BANDS:
        screens[band_name] = {}
        for endpoint in ENDPOINTS:
            row = inference[band_name][endpoint]
            delete = row["aggregate"]["deletion_damage"]
            rescue = row["aggregate"]["base_band_rescue"]
            contrast = row["aggregate"]["deletion_minus_phase"]
            native_use = bool(
                row["all_five_positive"]["deletion_damage"]
                and row["all_five_positive"]["base_band_rescue"]
                and delete["estimate"] >= MIN_SCREEN_EFFECT
                and rescue["estimate"] >= MIN_SCREEN_EFFECT
                and delete["ci_low"] > 0
                and rescue["ci_low"] > 0
                and delete["q_bh_18_tests"] <= MAX_Q
                and rescue["q_bh_18_tests"] <= MAX_Q
            )
            specificity = bool(
                native_use
                and contrast["ci_low"] > 0
                and contrast["q_bh_18_tests"] <= MAX_Q
            )
            screens[band_name][endpoint] = {
                "native_use_screen": native_use,
                "deletion_specificity_screen": specificity,
            }
    return inference, screens


def _json_ready_raw(
    raw: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, list[float]]]:
    return {
        arm: {
            metric: [float(value) for value in values]
            for metric, values in metrics.items()
        }
        for arm, metrics in raw.items()
    }


def _write(
    output: Path,
    manifest_path: Path,
    payload: dict[str, Any],
    manifest: ExperimentManifest,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest.status = str(payload["status"])
    manifest.outputs = [file_record(output)]
    manifest.write(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrum", required=True)
    parser.add_argument("--cohort-source", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=EXPECTED_N)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=EXPECTED_BOOT)
    parser.add_argument("--n-perm", type=int, default=EXPECTED_PERM)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != EXPECTED_N:
        raise ValueError(f"cohort is frozen at n={EXPECTED_N}")
    if args.n_boot != EXPECTED_BOOT or args.n_perm != EXPECTED_PERM:
        raise ValueError(
            f"inference is frozen at {EXPECTED_BOOT} bootstraps and "
            f"{EXPECTED_PERM} sign flips"
        )
    torch.set_grad_enabled(False)
    spectrum_path = Path(args.spectrum).expanduser().resolve()
    cohort_path = Path(args.cohort_source).expanduser().resolve()
    spectrum = json.loads(spectrum_path.read_text())
    cohort_source = json.loads(cohort_path.read_text())
    if spectrum.get("status") != "COMPLETE":
        raise RuntimeError("spectrum artifact is not complete")
    if not spectrum.get("preflight", {}).get("passed"):
        raise RuntimeError("spectrum preflight did not pass")
    named_bands = spectrum.get("analysis", {}).get("named_bands")
    if named_bands != {
        name: list(bins) for name, bins in EXPECTED_NAMED_BANDS.items()
    }:
        raise RuntimeError("spectrum named bands do not match the frozen gate")
    if cohort_source.get("status") != "COMPLETE":
        raise RuntimeError("cohort source is not complete")
    if not cohort_source.get("preflight", {}).get("passed"):
        raise RuntimeError("cohort source preflight did not pass")

    base_model = Path(args.base_model).expanduser().resolve()
    base_weight = _weight_file(base_model)
    base_table = _position_table_from_weight_file(base_weight)
    checkpoints = parse_checkpoints(args.checkpoint)
    if sha256_file(base_weight) != spectrum["sources"]["base"]["sha256"]:
        raise RuntimeError("base checkpoint hash does not match spectrum artifact")
    for seed, checkpoint in checkpoints.items():
        expected_hash = spectrum["sources"]["fine_tuned"][str(seed)]["sha256"]
        if sha256_file(checkpoint) != expected_hash:
            raise RuntimeError(f"checkpoint hash mismatch for seed {seed}")

    output = guard_output_path(Path(args.out).expanduser().resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    gtf = Path(args.gtf).expanduser().resolve()
    fasta = Path(args.fasta).expanduser().resolve()
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_POSITIONAL_NATIVE_MARGIN_GATE_V1.md"
    )
    device = pick_device(args.device)
    examples, cohort_checks = _load_replayed_cohort(
        gtf, fasta, cohort_source
    )
    clean_sequences = [example.sequence for example in examples]
    local_sequences = [
        substitute(example.sequence, int(example.donor_pos), MOTIF_CLEAN_DONOR_ALT)
        for example in examples
    ]
    paired_sequences = [
        substitute(example.sequence, int(example.next_acceptor_pos) - 2, "CC")
        for example in examples
    ]
    donor_positions = [int(example.donor_pos) - 1 for example in examples]
    acceptor_positions = [
        int(example.next_acceptor_pos) for example in examples
    ]

    manifest = ExperimentManifest.create(
        "gtex-positional-native-margin-gate-v1",
        sys.argv,
        repo=REPO,
        config={
            "protocol_sha256": sha256_file(protocol),
            "spectrum_sha256": sha256_file(spectrum_path),
            "cohort_source_sha256": sha256_file(cohort_path),
            "named_bands": {
                name: list(bins) for name, bins in EXPECTED_NAMED_BANDS.items()
            },
            "phase_patterns": {
                name: list(pattern) for name, pattern in PHASE_PATTERNS.items()
            },
            "phase_angle_radians": float(PHASE_ANGLE),
            "n": EXPECTED_N,
            "batch": args.batch,
            "n_boot": EXPECTED_BOOT,
            "n_perm": EXPECTED_PERM,
            "thresholds": {
                "spectral_support_tolerance": SPECTRAL_SUPPORT_TOLERANCE,
                "phase_power_tolerance": PHASE_POWER_TOLERANCE,
                "phase_dose_tolerance": PHASE_DOSE_TOLERANCE,
                "minimum_intact_local_destruction": MIN_INTACT_LOCAL_DESTRUCTION,
                "minimum_health_retention": MIN_HEALTH_RETENTION,
                "minimum_screen_effect": MIN_SCREEN_EFFECT,
                "maximum_q": MAX_Q,
            },
        },
        seeds={
            "bootstrap_base": 740_000,
            "permutation_base": 840_000,
        },
    )
    for path in [
        spectrum_path,
        cohort_path,
        protocol,
        base_weight,
        gtf,
        fasta,
        *checkpoints.values(),
    ]:
        manifest.add_input(path)

    common = {
        "experiment_id": "gtex-positional-native-margin-gate-v1",
        "protocol": file_record(protocol),
        "spectrum_source": file_record(spectrum_path),
        "cohort_source": file_record(cohort_path),
        "device": device,
        "exploratory_reused_cohort": True,
        "named_bands": {
            name: list(bins) for name, bins in EXPECTED_NAMED_BANDS.items()
        },
        "cohort": {
            "n": len(examples),
            "checks": cohort_checks,
            "example_ids": [example.example_id for example in examples],
            "gene_ids": [example.gene_id for example in examples],
            "chromosomes": [example.chrom for example in examples],
        },
    }
    preflight_by_seed: dict[str, Any] = {}
    raw_by_seed: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for seed, checkpoint in sorted(checkpoints.items()):
        print(f"seed {seed}: native margin gate", flush=True)
        preflight, raw = run_seed(
            base_model,
            checkpoint,
            base_table,
            clean_sequences,
            local_sequences,
            paired_sequences,
            donor_positions,
            acceptor_positions,
            device=device,
            batch_size=args.batch,
        )
        preflight_by_seed[str(seed)] = preflight
        raw_by_seed[seed] = raw
        print(f"seed {seed}: preflight passed={preflight['passed']}", flush=True)

    preflight_passed = all(
        row["passed"] for row in preflight_by_seed.values()
    )
    if not preflight_passed:
        payload = {
            **common,
            "status": "STOPPED_INTERVENTION_PREFLIGHT",
            "preflight": {
                "passed": False,
                "by_seed": preflight_by_seed,
            },
            "raw": {
                str(seed): _json_ready_raw(raw)
                for seed, raw in raw_by_seed.items()
            },
            "inference": None,
            "screens": {
                band_name: {
                    endpoint: {
                        "native_use_screen": False,
                        "deletion_specificity_screen": False,
                    }
                    for endpoint in ENDPOINTS
                }
                for band_name in EXPECTED_NAMED_BANDS
            },
            "strongest_claim": (
                "At least one named-carrier intervention failed the frozen "
                "instrument or classifier-health gate; target interpretation "
                "is withheld and the anomaly is preserved."
            ),
        }
        _write(output, manifest_path, payload, manifest)
        raise SystemExit("named-carrier intervention preflight failed")

    groups = np.asarray([example.gene_id for example in examples], dtype=object)
    inference, screens = infer(raw_by_seed, groups)
    payload = {
        **common,
        "status": "COMPLETE",
        "preflight": {
            "passed": True,
            "by_seed": preflight_by_seed,
        },
        "raw": {
            str(seed): _json_ready_raw(raw)
            for seed, raw in raw_by_seed.items()
        },
        "inference": inference,
        "screens": screens,
        "multiplicity": {
            "method": "benjamini_hochberg",
            "family": (
                "2 named bands x 3 native endpoints x "
                "3 primary estimands = 18 tests"
            ),
            "n_tests": 18,
        },
        "strongest_claim": (
            "A passing screen is exploratory evidence that the named positional "
            "carrier contributes to the released fine-tuned classifier endpoint "
            "on the reused 64-example cohort. A failed screen is bounded to the "
            "tested carrier intervention and endpoint."
        ),
    }
    _write(output, manifest_path, payload, manifest)
    compact = {
        band_name: {
            endpoint: {
                "deletion_damage": inference[band_name][endpoint]["aggregate"][
                    "deletion_damage"
                ],
                "base_band_rescue": inference[band_name][endpoint]["aggregate"][
                    "base_band_rescue"
                ],
                "deletion_minus_phase": inference[band_name][endpoint][
                    "aggregate"
                ]["deletion_minus_phase"],
                **screens[band_name][endpoint],
            }
            for endpoint in ENDPOINTS
        }
        for band_name in EXPECTED_NAMED_BANDS
    }
    print(
        json.dumps(
            {
                "status": payload["status"],
                "preflight_passed": True,
                "results": compact,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
