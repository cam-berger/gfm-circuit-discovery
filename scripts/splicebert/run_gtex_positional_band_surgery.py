#!/usr/bin/env python
"""Causally test spectrum-frozen GTEx positional bands on native coupling.

The input spectrum artifact fixes targets and controls without classifier
outcomes.  This runner first opens only clean and local-GT→CC controls for
every intervention in every checkpoint.  Paired-acceptor coupling is opened
only if that hard preflight passes as preregistered.
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
from gfmcd.splicing.biological_cohort import (  # noqa: E402
    collect_prior_gene_exclusion,
)
from gfmcd.splicing.circuit import _dataset, eligible, pick_device  # noqa: E402
from gfmcd.splicing.coupling import (  # noqa: E402
    CouplingConfig,
    CouplingExample,
    stable_rng,
    substitute,
)
from gfmcd.splicing.grouped_stats import (  # noqa: E402
    grouped_bootstrap,
    grouped_signflip_pvalue,
)
from gfmcd.splicing.gtex_finetune import (  # noqa: E402
    load_gtex_token_classifier,
)
from gfmcd.splicing.native_classifier import (  # noqa: E402
    DONOR_LABEL,
    NativeClassifierTask,
)
from gfmcd.splicing.positional_spectrum import (  # noqa: E402
    centered_spectrum,
    delete_band,
    replace_band,
    validate_table,
)

EXPECTED_SEEDS = frozenset(range(2020, 2025))
EXPECTED_N = 200
EXPECTED_BOOT = 5000
EXPECTED_PERM = 9999
COHORT_NAMESPACE = "gtex-positional-spectrum-native-v1"
FORBIDDEN_CHROMOSOMES = frozenset(
    {
        "chr8",
        "chr9",
        "chr11",
        "chr16",
        "chr19",
        "chrX",
        "chrY",
        "chrM",
        "chrMT",
        "MT",
    }
)
IDENTITY_LOGIT_TOLERANCE = 2e-5
SPECTRAL_SUPPORT_TOLERANCE = 1e-10
MIN_INTACT_LOCAL_DESTRUCTION = 5.0
MIN_HEALTH_RETENTION = 0.80
MIN_CAUSAL_EFFECT = 0.05
MAX_Q = 0.05
ESTIMANDS = ("necessity", "base_band_rescue", "rewrite_contrast")


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
        raise ValueError("base weights must expose exactly one position embedding table")
    return validate_table(
        hits[0].detach().to(dtype=torch.float64, device="cpu").numpy(),
        shape=(1026, 512),
    )


def load_fresh_cohort(
    gtf: Path,
    fasta: Path,
    *,
    excluded_gene_ids: Sequence[str],
    n: int,
) -> tuple[list[CouplingExample], dict[str, int]]:
    """Outcome-blind train cohort with one example per never-analyzed gene."""

    excluded = set(excluded_gene_ids)
    counts = {
        "eligible_train": 0,
        "after_prior_and_chromosome_exclusion": 0,
        "unique_genes_after_exclusion": 0,
    }
    candidates: list[CouplingExample] = []
    for example in eligible(_dataset(str(gtf), str(fasta)), "train"):
        counts["eligible_train"] += 1
        if (
            example.gene_id in excluded
            or example.chrom in FORBIDDEN_CHROMOSOMES
        ):
            continue
        counts["after_prior_and_chromosome_exclusion"] += 1
        candidates.append(example)
    stable_rng(COHORT_NAMESPACE, CouplingConfig().seed).shuffle(candidates)
    selected: list[CouplingExample] = []
    seen: set[str] = set()
    for example in candidates:
        if example.gene_id in seen:
            continue
        seen.add(example.gene_id)
        selected.append(example)
        if len(selected) == n:
            break
    counts["unique_genes_after_exclusion"] = len(
        {example.gene_id for example in candidates}
    )
    if len(selected) != n:
        raise RuntimeError(f"only {len(selected)} fresh unique genes; asked for {n}")
    return selected, counts


def _full_reconstruction(table: np.ndarray) -> np.ndarray:
    values = validate_table(table)
    mean = values.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(values - mean, axis=0)
    return mean + np.fft.irfft(spectrum, n=values.shape[0], axis=0)


def _support_audit(
    original: np.ndarray,
    edited: np.ndarray,
    bins: Sequence[int],
) -> dict[str, Any]:
    """Audit that a surgery's Fourier delta has only nominated support."""

    base_spectrum = centered_spectrum(original)
    edited_spectrum = centered_spectrum(edited)
    delta = edited_spectrum - base_spectrum
    outside = np.ones(delta.shape[0], dtype=bool)
    outside[list(bins)] = False
    scale = max(
        float(np.max(np.abs(base_spectrum))),
        np.finfo(np.float64).tiny,
    )
    outside_error = float(np.max(np.abs(delta[outside]))) / scale
    return {
        "bins": [int(value) for value in bins],
        "outside_band_max_relative_delta": outside_error,
        "outside_band_pass": outside_error <= SPECTRAL_SUPPORT_TOLERANCE,
        "edited_is_real": bool(np.isrealobj(edited)),
        "edited_finite": bool(np.isfinite(edited).all()),
    }


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
    targets: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Construct all exact target and matched-control arm tables."""

    tables: dict[str, np.ndarray] = {"intact": original.copy()}
    specs: dict[str, dict[str, Any]] = {
        "intact": {"intervention": "identity", "bins": []}
    }
    for target in targets:
        target_name = str(target["name"])
        target_bins = tuple(int(value) for value in target["bins"])
        delete_name = f"{target_name}__delete"
        replace_name = f"{target_name}__base_replace"
        tables[delete_name] = delete_band(original, target_bins)
        tables[replace_name] = replace_band(original, base, target_bins)
        specs[delete_name] = {
            "target": target_name,
            "kind": "target",
            "intervention": "delete",
            "bins": list(target_bins),
        }
        specs[replace_name] = {
            "target": target_name,
            "kind": "target",
            "intervention": "base_replace",
            "bins": list(target_bins),
        }
        target_controls = controls.get(target_name)
        if not isinstance(target_controls, list) or len(target_controls) != 3:
            raise ValueError(f"{target_name} does not have three frozen controls")
        for index, control_bins_raw in enumerate(target_controls, start=1):
            control_bins = tuple(int(value) for value in control_bins_raw)
            control_delete = f"{target_name}__control_{index}__delete"
            control_replace = f"{target_name}__control_{index}__base_replace"
            tables[control_delete] = delete_band(original, control_bins)
            tables[control_replace] = replace_band(original, base, control_bins)
            specs[control_delete] = {
                "target": target_name,
                "kind": "control",
                "control_index": index,
                "intervention": "delete",
                "bins": list(control_bins),
            }
            specs[control_replace] = {
                "target": target_name,
                "kind": "control",
                "control_index": index,
                "intervention": "base_replace",
                "bins": list(control_bins),
            }
    return tables, specs


def _donor_margin(
    task: NativeClassifierTask,
    logits: np.ndarray,
    donor_label_positions: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    scores = task.score(logits, donor_label_positions, label=DONOR_LABEL)
    return scores.margin, scores.probability


def _score_clean_and_local(
    task: NativeClassifierTask,
    clean_sequences: Sequence[str],
    local_sequences: Sequence[str],
    donor_label_positions: Sequence[int],
) -> dict[str, np.ndarray]:
    combined = task.logits([*clean_sequences, *local_sequences])
    split = len(clean_sequences)
    clean_logits = combined[:split]
    local_logits = combined[split:]
    clean_margin, clean_probability = _donor_margin(
        task, clean_logits, donor_label_positions
    )
    local_margin, _ = _donor_margin(task, local_logits, donor_label_positions)
    return {
        "clean_logits": clean_logits,
        "clean_margin": clean_margin,
        "clean_probability": clean_probability,
        "local_margin": local_margin,
        "local_destruction": clean_margin - local_margin,
    }


def preflight_seed(
    base_model: Path,
    checkpoint: Path,
    base_table: np.ndarray,
    targets: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    clean_sequences: Sequence[str],
    local_sequences: Sequence[str],
    donor_label_positions: Sequence[int],
    *,
    device: str,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Run mechanical, clean-health, and local-positive controls for one seed."""

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
    embedding = model.bert.embeddings.position_embeddings
    original = validate_table(
        embedding.weight.detach().to(dtype=torch.float64, device="cpu").numpy(),
        shape=base_table.shape,
    )
    tables, arm_specs = _arm_tables(original, base_table, targets, controls)
    spectral_audits = {
        name: _support_audit(original, table, arm_specs[name]["bins"])
        for name, table in tables.items()
        if name != "intact"
    }
    full_reconstruction = _full_reconstruction(original)
    table_reconstruction_error = float(
        np.max(np.abs(full_reconstruction - original))
    )

    raw: dict[str, dict[str, np.ndarray]] = {}
    _copy_embedding_table(model, original)
    raw["intact"] = _score_clean_and_local(
        task,
        clean_sequences,
        local_sequences,
        donor_label_positions,
    )
    intact_logits = raw["intact"]["clean_logits"]

    _copy_embedding_table(model, full_reconstruction)
    identity_logits = task.logits(clean_sequences)
    identity_logit_error = float(np.max(np.abs(identity_logits - intact_logits)))

    for name, table in tables.items():
        if name == "intact":
            continue
        print(f"    preflight arm {name}", flush=True)
        _copy_embedding_table(model, table)
        raw[name] = _score_clean_and_local(
            task,
            clean_sequences,
            local_sequences,
            donor_label_positions,
        )
    _copy_embedding_table(model, original)

    intact_local = float(np.mean(raw["intact"]["local_destruction"]))
    intact_positive = float(np.mean(raw["intact"]["clean_margin"] > 0))
    arm_health: dict[str, dict[str, Any]] = {}
    for name, values in raw.items():
        local_mean = float(np.mean(values["local_destruction"]))
        positive_fraction = float(np.mean(values["clean_margin"] > 0))
        finite = bool(
            all(np.isfinite(array).all() for array in values.values())
        )
        arm_health[name] = {
            "clean_margin_mean": float(np.mean(values["clean_margin"])),
            "clean_donor_positive_fraction": positive_fraction,
            "local_destruction_mean": local_mean,
            "local_retention_fraction": local_mean / intact_local,
            "positive_fraction_retention": (
                positive_fraction / intact_positive
                if intact_positive > 0
                else float("nan")
            ),
            "finite": finite,
        }
    nonintact_health_pass = all(
        row["local_retention_fraction"] >= MIN_HEALTH_RETENTION
        and row["positive_fraction_retention"] >= MIN_HEALTH_RETENTION
        and row["finite"]
        for name, row in arm_health.items()
        if name != "intact"
    )
    checks = {
        "identity_logits": identity_logit_error <= IDENTITY_LOGIT_TOLERANCE,
        "full_table_reconstruction": table_reconstruction_error <= 1e-10,
        "spectral_support": all(
            row["outside_band_pass"]
            and row["edited_is_real"]
            and row["edited_finite"]
            for row in spectral_audits.values()
        ),
        "intact_local_destruction": (
            intact_local >= MIN_INTACT_LOCAL_DESTRUCTION
        ),
        "all_arm_health": nonintact_health_pass,
        "all_finite": all(row["finite"] for row in arm_health.values()),
    }
    result = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "checkpoint": report.to_dict(),
        "identity_logit_max_abs_error": identity_logit_error,
        "full_table_reconstruction_max_abs_error": table_reconstruction_error,
        "spectral_support_audits": spectral_audits,
        "arm_specs": arm_specs,
        "arm_health": arm_health,
        "thresholds": {
            "identity_logit_max_abs_error": IDENTITY_LOGIT_TOLERANCE,
            "spectral_support_max_relative_error": SPECTRAL_SUPPORT_TOLERANCE,
            "minimum_intact_local_destruction": MIN_INTACT_LOCAL_DESTRUCTION,
            "minimum_arm_health_retention": MIN_HEALTH_RETENTION,
        },
    }
    for values in raw.values():
        values.pop("clean_logits", None)
    del task, model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return result, raw


def paired_seed(
    base_model: Path,
    checkpoint: Path,
    base_table: np.ndarray,
    targets: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    partner_sequences: Sequence[str],
    donor_label_positions: Sequence[int],
    *,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Open paired-acceptor outcomes for every frozen arm in one seed."""

    model, tokenizer, _ = load_gtex_token_classifier(
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
    embedding = model.bert.embeddings.position_embeddings
    original = validate_table(
        embedding.weight.detach().to(dtype=torch.float64, device="cpu").numpy(),
        shape=base_table.shape,
    )
    tables, _ = _arm_tables(original, base_table, targets, controls)
    paired_margins: dict[str, np.ndarray] = {}
    for name, table in tables.items():
        print(f"    target arm {name}", flush=True)
        _copy_embedding_table(model, table)
        logits = task.logits(partner_sequences)
        paired_margins[name], _ = _donor_margin(
            task, logits, donor_label_positions
        )
    _copy_embedding_table(model, original)
    del task, model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return paired_margins


def _effect_arrays(
    coupling: Mapping[str, np.ndarray],
    target: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    target_name = str(target["name"])
    intact = coupling["intact"]
    target_delete = coupling[f"{target_name}__delete"]
    target_replace = coupling[f"{target_name}__base_replace"]
    target_effects = {
        "necessity": intact - target_delete,
        "base_band_rescue": target_replace - target_delete,
        "rewrite_contrast": intact - target_replace,
    }
    per_control: list[dict[str, np.ndarray]] = []
    for index in range(1, 4):
        deleted = coupling[f"{target_name}__control_{index}__delete"]
        replaced = coupling[f"{target_name}__control_{index}__base_replace"]
        per_control.append(
            {
                "necessity": intact - deleted,
                "base_band_rescue": replaced - deleted,
                "rewrite_contrast": intact - replaced,
            }
        )
    control_effects = {
        estimand: np.mean(
            np.stack([record[estimand] for record in per_control], axis=0),
            axis=0,
        )
        for estimand in ESTIMANDS
    }
    return {
        estimand: {
            "target": target_effects[estimand],
            "matched_control": control_effects[estimand],
            "target_minus_control": (
                target_effects[estimand] - control_effects[estimand]
            ),
        }
        for estimand in ESTIMANDS
    }


def _summary(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    bootstrap_seed: int,
    permutation_seed: int,
    with_pvalue: bool,
) -> dict[str, Any]:
    result = grouped_bootstrap(
        values,
        groups,
        n_boot=EXPECTED_BOOT,
        seed=bootstrap_seed,
    )
    if with_pvalue:
        result["p_signflip_add_one"] = grouped_signflip_pvalue(
            values,
            groups,
            n_perm=EXPECTED_PERM,
            seed=permutation_seed,
        )
    return result


def infer_effects(
    effects_by_seed: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    targets: Sequence[Mapping[str, Any]],
    groups: np.ndarray,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Average checkpoints within example, infer one frozen BH family, license."""

    inference: dict[str, Any] = {}
    pvalue_locations: list[tuple[str, str, str]] = []
    pvalues: list[float] = []
    counter = 0
    for target in targets:
        target_name = str(target["name"])
        inference[target_name] = {}
        for estimand in ESTIMANDS:
            seed_arrays = {
                seed: effects_by_seed[seed][target_name][estimand]
                for seed in sorted(effects_by_seed)
            }
            aggregate = {
                component: np.mean(
                    np.stack(
                        [seed_arrays[seed][component] for seed in sorted(seed_arrays)],
                        axis=0,
                    ),
                    axis=0,
                )
                for component in (
                    "target",
                    "matched_control",
                    "target_minus_control",
                )
            }
            row = {
                "per_seed_means": {
                    str(seed): {
                        component: float(np.mean(seed_arrays[seed][component]))
                        for component in (
                            "target",
                            "matched_control",
                            "target_minus_control",
                        )
                    }
                    for seed in sorted(seed_arrays)
                },
                "all_five_target_positive": all(
                    float(np.mean(seed_arrays[seed]["target"])) > 0
                    for seed in seed_arrays
                ),
                "aggregate": {},
            }
            for component, values in aggregate.items():
                with_pvalue = component in {"target", "target_minus_control"}
                summary = _summary(
                    values,
                    groups,
                    bootstrap_seed=730_000 + counter * 10 + 1,
                    permutation_seed=830_000 + counter * 10 + 2,
                    with_pvalue=with_pvalue,
                )
                row["aggregate"][component] = summary
                if with_pvalue:
                    pvalue_locations.append((target_name, estimand, component))
                    pvalues.append(float(summary["p_signflip_add_one"]))
                counter += 1
            inference[target_name][estimand] = row
    qvalues = benjamini_hochberg(pvalues)
    for (target_name, estimand, component), qvalue in zip(
        pvalue_locations, qvalues, strict=True
    ):
        inference[target_name][estimand]["aggregate"][component][
            "q_bh_12_tests"
        ] = float(qvalue)

    licenses: dict[str, bool] = {}
    for target in targets:
        target_name = str(target["name"])
        passed = True
        for estimand in ("necessity", "base_band_rescue"):
            row = inference[target_name][estimand]
            target_summary = row["aggregate"]["target"]
            contrast_summary = row["aggregate"]["target_minus_control"]
            passed &= bool(row["all_five_target_positive"])
            passed &= target_summary["estimate"] >= MIN_CAUSAL_EFFECT
            passed &= target_summary["ci_low"] > 0
            passed &= contrast_summary["ci_low"] > 0
            passed &= target_summary["q_bh_12_tests"] <= MAX_Q
            passed &= contrast_summary["q_bh_12_tests"] <= MAX_Q
        licenses[target_name] = bool(passed)
    return inference, licenses


def _json_ready_raw(
    values: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            metric: [float(value) for value in array]
            for metric, array in metrics.items()
        }
        for name, metrics in values.items()
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
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--prior-results", required=True)
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
    spectrum = json.loads(spectrum_path.read_text())
    if spectrum.get("status") != "COMPLETE":
        raise RuntimeError("spectrum artifact is not complete")
    if not spectrum.get("preflight", {}).get("passed"):
        raise RuntimeError("spectrum preflight did not pass")
    package = spectrum.get("surgery_package", {})
    if (
        not package.get("authorized")
        or package.get("selection_used_classifier_outputs") is not False
        or package.get("authorization_semantics")
        != "target_specific_stop_per_preregistration_section_6"
    ):
        raise RuntimeError("spectrum artifact does not authorize frozen surgery")
    targets = package.get("authorized_targets")
    if not isinstance(targets, list) or not targets:
        raise RuntimeError("spectrum artifact has no authorized targets")
    if any(target.get("kind") != "scan_candidate" for target in targets):
        raise RuntimeError("only spectrum-authorized scan candidates may be opened")

    base_model = Path(args.base_model).expanduser().resolve()
    base_weight = _weight_file(base_model)
    base_table = _position_table_from_weight_file(base_weight)
    checkpoints = parse_checkpoints(args.checkpoint)
    for seed, checkpoint in checkpoints.items():
        expected_hash = (
            spectrum["sources"]["fine_tuned"][str(seed)]["sha256"]
        )
        if sha256_file(checkpoint) != expected_hash:
            raise RuntimeError(f"checkpoint hash mismatch for seed {seed}")
    if sha256_file(base_weight) != spectrum["sources"]["base"]["sha256"]:
        raise RuntimeError("base checkpoint hash does not match spectrum artifact")

    output = guard_output_path(Path(args.out).expanduser().resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    gtf = Path(args.gtf).expanduser().resolve()
    fasta = Path(args.fasta).expanduser().resolve()
    prior_root = Path(args.prior_results).expanduser().resolve()
    protocol = (
        REPO
        / "studies/splicebert_gtex_finetune_shift/"
        "PREREGISTRATION_POSITIONAL_SPECTRUM_V1.md"
    )
    device = pick_device(args.device)

    prior = collect_prior_gene_exclusion(prior_root)
    examples, cohort_counts = load_fresh_cohort(
        gtf,
        fasta,
        excluded_gene_ids=prior.gene_ids,
        n=args.n,
    )
    if len({example.gene_id for example in examples}) != len(examples):
        raise RuntimeError("cohort is not one-example-per-gene")
    if any(example.chrom in FORBIDDEN_CHROMOSOMES for example in examples):
        raise RuntimeError("cohort opened a forbidden chromosome")
    clean_sequences = [example.sequence for example in examples]
    local_sequences = [
        substitute(example.sequence, int(example.donor_pos), "CC")
        for example in examples
    ]
    partner_sequences = [
        substitute(example.sequence, int(example.next_acceptor_pos) - 2, "CC")
        for example in examples
    ]
    donor_label_positions = [int(example.donor_pos) - 1 for example in examples]
    annotation_checks = {
        "n": len(examples) == EXPECTED_N,
        "one_example_per_gene": (
            len({example.gene_id for example in examples}) == EXPECTED_N
        ),
        "train_split": all(example.split == "train" for example in examples),
        "forbidden_chromosomes_absent": all(
            example.chrom not in FORBIDDEN_CHROMOSOMES for example in examples
        ),
        "prior_genes_absent": all(
            example.gene_id not in set(prior.gene_ids) for example in examples
        ),
        "local_gt_intact": all(
            example.sequence[
                int(example.donor_pos) : int(example.donor_pos) + 2
            ]
            == "GT"
            for example in examples
        ),
        "partner_ag_intact": all(
            example.sequence[
                int(example.next_acceptor_pos) - 2 : int(example.next_acceptor_pos)
            ]
            == "AG"
            for example in examples
        ),
        "sequence_length_1024": all(
            len(example.sequence) == 1024 for example in examples
        ),
    }
    if not all(annotation_checks.values()):
        raise RuntimeError(f"cohort annotation preflight failed: {annotation_checks}")

    manifest = ExperimentManifest.create(
        "gtex-positional-band-surgery-v1",
        sys.argv,
        repo=REPO,
        config={
            "protocol_sha256": sha256_file(protocol),
            "spectrum_sha256": sha256_file(spectrum_path),
            "targets": targets,
            "cohort_namespace": COHORT_NAMESPACE,
            "forbidden_chromosomes": sorted(FORBIDDEN_CHROMOSOMES),
            "n": EXPECTED_N,
            "batch": args.batch,
            "n_boot": EXPECTED_BOOT,
            "n_perm": EXPECTED_PERM,
            "thresholds": {
                "identity_logit_tolerance": IDENTITY_LOGIT_TOLERANCE,
                "spectral_support_tolerance": SPECTRAL_SUPPORT_TOLERANCE,
                "minimum_intact_local_destruction": MIN_INTACT_LOCAL_DESTRUCTION,
                "minimum_health_retention": MIN_HEALTH_RETENTION,
                "minimum_causal_effect": MIN_CAUSAL_EFFECT,
                "maximum_q": MAX_Q,
            },
        },
        seeds={"cohort": CouplingConfig().seed},
    )
    for path in [
        spectrum_path,
        protocol,
        base_weight,
        gtf,
        fasta,
        *checkpoints.values(),
    ]:
        manifest.add_input(path)
    for source in prior.source_files:
        manifest.inputs.append(dict(source))

    common = {
        "experiment_id": "gtex-positional-band-surgery-v1",
        "protocol": file_record(protocol),
        "spectrum_source": file_record(spectrum_path),
        "device": device,
        "targets": targets,
        "cohort": {
            "namespace": COHORT_NAMESPACE,
            "n": len(examples),
            "counts": cohort_counts,
            "prior_gene_exclusion": {
                "n_genes": len(prior.gene_ids),
                "aggregate_sha256": prior.aggregate_sha256,
                "source_files": list(prior.source_files),
            },
            "annotation_checks": annotation_checks,
            "example_ids": [example.example_id for example in examples],
            "gene_ids": [example.gene_id for example in examples],
            "chromosomes": [example.chrom for example in examples],
        },
    }
    preflight_results: dict[str, Any] = {}
    preflight_raw: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    controls_by_seed = package["matched_controls_by_seed"]
    for seed, checkpoint in sorted(checkpoints.items()):
        print(
            f"seed {seed}: surgery preflight on {len(examples)} examples",
            flush=True,
        )
        result, raw = preflight_seed(
            base_model,
            checkpoint,
            base_table,
            targets,
            controls_by_seed[str(seed)],
            clean_sequences,
            local_sequences,
            donor_label_positions,
            device=device,
            batch_size=args.batch,
        )
        preflight_results[str(seed)] = result
        preflight_raw[seed] = raw
        print(f"seed {seed}: preflight passed={result['passed']}", flush=True)

    preflight_passed = all(
        result["passed"] for result in preflight_results.values()
    )
    if not preflight_passed:
        payload = {
            **common,
            "status": "STOPPED_SURGERY_PREFLIGHT",
            "preflight": {
                "passed": False,
                "by_seed": preflight_results,
                "raw": {
                    str(seed): _json_ready_raw(raw)
                    for seed, raw in preflight_raw.items()
                },
            },
            "paired_acceptor_outcome_opened": False,
            "licenses": {str(target["name"]): False for target in targets},
            "strongest_claim": (
                "At least one positional-band intervention failed the frozen "
                "classifier-surgery preflight; paired coupling was not opened."
            ),
        }
        _write(output, manifest_path, payload, manifest)
        raise SystemExit("surgery preflight failed; paired outcome remained sealed")

    paired_by_seed: dict[int, dict[str, np.ndarray]] = {}
    effects_by_seed: dict[
        int, dict[str, dict[str, dict[str, np.ndarray]]]
    ] = {}
    coupling_by_seed: dict[int, dict[str, np.ndarray]] = {}
    for seed, checkpoint in sorted(checkpoints.items()):
        print(f"seed {seed}: opening paired-acceptor outcome", flush=True)
        paired = paired_seed(
            base_model,
            checkpoint,
            base_table,
            targets,
            controls_by_seed[str(seed)],
            partner_sequences,
            donor_label_positions,
            device=device,
            batch_size=args.batch,
        )
        paired_by_seed[seed] = paired
        coupling = {
            arm: preflight_raw[seed][arm]["clean_margin"] - paired_margin
            for arm, paired_margin in paired.items()
        }
        coupling_by_seed[seed] = coupling
        effects_by_seed[seed] = {
            str(target["name"]): _effect_arrays(coupling, target)
            for target in targets
        }

    groups = np.asarray(
        [example.gene_id for example in examples],
        dtype=object,
    )
    inference, licenses = infer_effects(effects_by_seed, targets, groups)
    payload = {
        **common,
        "status": "COMPLETE",
        "preflight": {
            "passed": True,
            "by_seed": preflight_results,
            "raw": {
                str(seed): _json_ready_raw(raw)
                for seed, raw in preflight_raw.items()
            },
        },
        "paired_acceptor_outcome_opened": True,
        "raw": {
            str(seed): {
                "paired_margin": {
                    arm: [float(value) for value in values]
                    for arm, values in paired_by_seed[seed].items()
                },
                "coupling": {
                    arm: [float(value) for value in values]
                    for arm, values in coupling_by_seed[seed].items()
                },
                "effects": {
                    target_name: {
                        estimand: {
                            component: [float(value) for value in array]
                            for component, array in components.items()
                        }
                        for estimand, components in estimands.items()
                    }
                    for target_name, estimands in effects_by_seed[seed].items()
                },
            }
            for seed in sorted(checkpoints)
        },
        "inference": inference,
        "licenses": licenses,
        "multiplicity": {
            "method": "benjamini_hochberg",
            "family": (
                "2 targets x 3 estimands x "
                "(target and target-minus-control) = 12 tests"
            ),
            "n_tests": 12,
        },
        "strongest_claim": (
            "A target with license=true is a causal positional carrier for "
            "native donor coupling on this analysis-fresh train cohort. A "
            "false license is a bounded/null result for this behavior only."
        ),
    }
    _write(output, manifest_path, payload, manifest)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "preflight_passed": True,
                "licenses": licenses,
                "inference": inference,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
