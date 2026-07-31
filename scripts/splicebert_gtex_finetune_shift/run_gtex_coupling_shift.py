#!/usr/bin/env python
"""Paired behavior-only comparison of base and GTEx-fine-tuned encoders.

This reuses the unchanged masked-donor coupling, distance-matched other-donor,
and non-partner-AG arms from ``run_sealed_test.py`` on the same validation
cohort.  It deliberately stops before head or representation tracing.

The runner was finalized after the seed-2020 mechanics smoke, so its output is
verification tier rather than confirmatory.  It nevertheless freezes the full
five-seed family, writes per-example paired values, uses gene-grouped
uncertainty, add-one sign-flip p-values, and one correction family spanning all
three scanned behavioral contrasts across all five seeds.
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

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.rigor import benjamini_hochberg  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CORRUPT_ALT,
    CouplingTask,
    load_adapter,
    load_examples,
    matched_decoy_donor,
    pick_device,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402

EXPECTED_SEEDS = frozenset(range(2020, 2025))
BASE_REPLAY_ATOL = 0.005
SCANNED_METRICS = (
    "coupling",
    "specificity_vs_other_donor",
    "specificity_vs_nonpartner_ag",
)


def json_default(value: Any) -> Any:
    """Serialize NumPy scalar/array values without lossy string fallbacks."""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def grouped_bootstrap(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    """Gene-grouped percentile interval using the study's exon-weighted estimand."""

    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    if values.size == 0:
        raise ValueError("no finite values")
    unique = np.unique(groups)
    by_group = {group: values[groups == group] for group in unique}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        boot[index] = np.concatenate([by_group[group] for group in sampled]).mean()
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n": int(values.size),
        "n_groups": int(len(unique)),
        "n_boot": n_boot,
    }


def grouped_signflip_pvalue(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> float:
    """Two-sided gene-level sign-flip test with the exact add-one correction."""

    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    unique, inverse = np.unique(groups, return_inverse=True)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(unique))
        exceed += abs(float(np.mean(values * signs[inverse]))) >= observed
    return float((exceed + 1) / (n_perm + 1))


def grouped_ratio_bootstrap(
    numerator: np.ndarray,
    denominator: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    """Gene-grouped interval for a ratio of paired means."""

    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    keep = np.isfinite(numerator) & np.isfinite(denominator)
    numerator, denominator, groups = (
        numerator[keep],
        denominator[keep],
        groups[keep],
    )
    unique = np.unique(groups)
    by_group = {
        group: (numerator[groups == group], denominator[groups == group])
        for group in unique
    }
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    eps = np.finfo(float).eps
    for index in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        num = np.concatenate([by_group[group][0] for group in sampled]).mean()
        den = np.concatenate([by_group[group][1] for group in sampled]).mean()
        boot[index] = num / max(den, eps)
    estimate = numerator.mean() / max(denominator.mean(), eps)
    return {
        "estimate": float(estimate),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n": int(numerator.size),
        "n_groups": int(len(unique)),
        "n_boot": n_boot,
    }


def parse_fine_models(values: list[str]) -> dict[int, Path]:
    models: dict[int, Path] = {}
    for value in values:
        try:
            seed_text, path_text = value.split("=", 1)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError("--fine-model must be SEED=PATH") from exc
        if seed in models:
            raise ValueError(f"duplicate seed {seed}")
        models[seed] = Path(path_text).expanduser().resolve()
    if set(models) != set(EXPECTED_SEEDS):
        raise ValueError(
            f"the complete seed family is required: expected {sorted(EXPECTED_SEEDS)}, "
            f"found {sorted(models)}"
        )
    return models


def _model_weight(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = model_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"model weights not found in {model_dir}")


def score_model(
    model_path: Path,
    examples: list[Any],
    *,
    device: str,
    batch_size: int,
    health_n: int,
) -> dict[str, Any]:
    adapter = load_adapter(str(model_path), device, eager=False)
    task = CouplingTask(adapter, batch=batch_size)
    coupling = task.coupling(examples)
    other = np.full(len(examples), np.nan, dtype=np.float64)
    nonpartner = np.full(len(examples), np.nan, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            other_positions = [matched_decoy_donor(example) for example in chunk]
            ag_positions = []
            for example in chunk:
                candidates = [pos for pos in example.other_ag if pos >= 3]
                ag_positions.append(
                    min(
                        candidates,
                        key=lambda pos: abs(pos - example.next_acceptor_pos),
                    )
                    if candidates
                    else None
                )
            keep = [
                index
                for index in range(len(chunk))
                if other_positions[index] is not None and ag_positions[index] is not None
            ]
            if not keep:
                continue
            selected = [chunk[index] for index in keep]
            donors = [example.donor_pos for example in selected]
            decoy_donors = [other_positions[index] for index in keep]
            decoy_ags = [ag_positions[index] for index in keep]
            clean_sequences = [example.sequence for example in selected]
            corrupt_sequences = [
                substitute(
                    example.sequence,
                    example.next_acceptor_pos - 2,
                    CORRUPT_ALT,
                )
                for example in selected
            ]
            ag_corrupt_sequences = [
                substitute(example.sequence, ag - 2, CORRUPT_ALT)
                for example, ag in zip(selected, decoy_ags)
            ]
            clean_own = task.margin(
                adapter.model(**task.encode(clean_sequences, donors)).logits.float(),
                donors,
            )
            corrupt_own = task.margin(
                adapter.model(**task.encode(corrupt_sequences, donors)).logits.float(),
                donors,
            )
            ag_corrupt = task.margin(
                adapter.model(**task.encode(ag_corrupt_sequences, donors)).logits.float(),
                donors,
            )
            clean_other = task.margin(
                adapter.model(
                    **task.encode(clean_sequences, decoy_donors)
                ).logits.float(),
                decoy_donors,
            )
            corrupt_other = task.margin(
                adapter.model(
                    **task.encode(corrupt_sequences, decoy_donors)
                ).logits.float(),
                decoy_donors,
            )
            for local, source_index in enumerate(keep):
                destination = start + source_index
                other[destination] = float(clean_other[local] - corrupt_other[local])
                nonpartner[destination] = float(clean_own[local] - ag_corrupt[local])
            # Memorial: the standalone coupling path and the specificity path must agree.
            expected = coupling[[start + index for index in keep]]
            observed = (clean_own - corrupt_own).detach().cpu().numpy()
            if not np.allclose(expected, observed, atol=2e-5, rtol=0):
                raise RuntimeError("coupling metric changed between canonical code paths")

    health = task.mlm_health(examples[:health_n], n_probe=40, seed=0)
    result = {
        "coupling": coupling,
        "other_donor": other,
        "nonpartner_ag": nonpartner,
        "specificity_vs_other_donor": coupling - other,
        "specificity_vs_nonpartner_ag": coupling - nonpartner,
        "mlm_health": health,
        "adapter_metadata": adapter.metadata().__dict__,
    }
    del task, adapter
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return result


def summarize_model(
    result: dict[str, Any],
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    summary = {
        metric: grouped_bootstrap(
            result[metric],
            groups,
            n_boot=n_boot,
            seed=seed + index,
        )
        for index, metric in enumerate(
            (
                "coupling",
                "other_donor",
                "nonpartner_ag",
                "specificity_vs_other_donor",
                "specificity_vs_nonpartner_ag",
            )
        )
    }
    summary["mlm_health"] = result["mlm_health"]
    summary["adapter_metadata"] = result["adapter_metadata"]
    return summary


def _assert_base_replay(
    summary: dict[str, Any],
    reference: dict[str, Any],
    *,
    n: int,
) -> dict[str, Any]:
    observed = {
        "n": summary["coupling"]["n"],
        "coupling": summary["coupling"]["estimate"],
        "other_donor": summary["other_donor"]["estimate"],
        "nonpartner_ag": summary["nonpartner_ag"]["estimate"],
    }
    expected = {
        "n": reference["behaviour"]["n"],
        "coupling": reference["behaviour"]["coupling"],
        "other_donor": reference["behaviour"]["other_donor"],
        "nonpartner_ag": reference["behaviour"]["nonpartner_ag"],
    }
    checks = {
        "n": observed["n"] == expected["n"] == n,
        "coupling": abs(observed["coupling"] - expected["coupling"])
        <= BASE_REPLAY_ATOL,
        "other_donor": abs(observed["other_donor"] - expected["other_donor"])
        <= BASE_REPLAY_ATOL,
        "nonpartner_ag": abs(observed["nonpartner_ag"] - expected["nonpartner_ag"])
        <= BASE_REPLAY_ATOL,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"base replay failed; observed={observed}, expected={expected}, "
            f"atol={BASE_REPLAY_ATOL}"
        )
    return {
        "atol": BASE_REPLAY_ATOL,
        "observed": observed,
        "expected": expected,
        "checks": checks,
        "PASS": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--base-reference", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--fine-model", action="append", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--health-n", type=int, default=100)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--n-perm", type=int, default=9999)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preflight_path = Path(args.preflight).resolve()
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "PASS":
        raise RuntimeError("preflight did not pass; target behavior is frozen")
    if args.n != 400:
        raise ValueError("the verification cohort is frozen at n=400")
    fine_models = parse_fine_models(args.fine_model)
    base_model = Path(args.base_model).expanduser().resolve()
    reference_path = Path(args.base_reference).resolve()
    reference = json.loads(reference_path.read_text())
    output = guard_output_path(Path(args.out).resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    device = pick_device(args.device)

    # The base loader defines the exact annotation-only cohort used by sealed_val.
    base_adapter = load_adapter(str(base_model), device, eager=False)
    base_task = CouplingTask(base_adapter, batch=args.batch)
    examples, _ = load_examples(
        args.gtf,
        args.fasta,
        "val",
        args.n,
        task=base_task,
        tag="sealed-unc-val",
        min_effect=None,
    )
    del base_task, base_adapter
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    groups = np.asarray([example.gene_id for example in examples], dtype=object)

    print(f"base: scoring {len(examples)} fixed validation examples")
    results: dict[str, dict[str, Any]] = {
        "base": score_model(
            base_model,
            examples,
            device=device,
            batch_size=args.batch,
            health_n=args.health_n,
        )
    }
    summaries: dict[str, Any] = {
        "base": summarize_model(
            results["base"],
            groups,
            n_boot=args.n_boot,
            seed=100,
        )
    }
    base_replay = _assert_base_replay(
        summaries["base"],
        reference,
        n=args.n,
    )

    for seed, model_path in sorted(fine_models.items()):
        print(f"seed {seed}: scoring {len(examples)} fixed validation examples")
        name = str(seed)
        results[name] = score_model(
            model_path,
            examples,
            device=device,
            batch_size=args.batch,
            health_n=args.health_n,
        )
        summaries[name] = summarize_model(
            results[name],
            groups,
            n_boot=args.n_boot,
            seed=seed,
        )

    shifts: dict[str, dict[str, Any]] = {}
    pvalue_entries: list[tuple[str, str, float]] = []
    for seed in sorted(EXPECTED_SEEDS):
        name = str(seed)
        shifts[name] = {}
        for metric_index, metric in enumerate(SCANNED_METRICS):
            delta = results[name][metric] - results["base"][metric]
            stat = grouped_bootstrap(
                delta,
                groups,
                n_boot=args.n_boot,
                seed=seed + 100 * metric_index,
            )
            pvalue = grouped_signflip_pvalue(
                delta,
                groups,
                n_perm=args.n_perm,
                seed=seed + 1000 * metric_index,
            )
            stat["p_add_one_two_sided"] = pvalue
            shifts[name][metric] = stat
            pvalue_entries.append((name, metric, pvalue))
        shifts[name]["coupling_retention"] = grouped_ratio_bootstrap(
            results[name]["coupling"],
            results["base"]["coupling"],
            groups,
            n_boot=args.n_boot,
            seed=seed + 7000,
        )

    corrected = benjamini_hochberg([entry[2] for entry in pvalue_entries])
    for (name, metric, _), qvalue in zip(pvalue_entries, corrected):
        shifts[name][metric]["q_bh_15_tests"] = float(qvalue)

    family: dict[str, Any] = {}
    family_pvalues: list[float] = []
    fine_names = [str(seed) for seed in sorted(EXPECTED_SEEDS)]
    for metric_index, metric in enumerate(SCANNED_METRICS):
        seed_deltas = np.stack(
            [results[name][metric] - results["base"][metric] for name in fine_names]
        )
        mean_delta = np.nanmean(seed_deltas, axis=0)
        stat = grouped_bootstrap(
            mean_delta,
            groups,
            n_boot=args.n_boot,
            seed=9000 + metric_index,
        )
        pvalue = grouped_signflip_pvalue(
            mean_delta,
            groups,
            n_perm=args.n_perm,
            seed=10000 + metric_index,
        )
        stat["p_add_one_two_sided"] = pvalue
        stat["seed_estimates"] = {
            name: shifts[name][metric]["estimate"] for name in fine_names
        }
        stat["n_same_direction_as_family"] = int(
            sum(
                np.sign(value) == np.sign(stat["estimate"])
                for value in stat["seed_estimates"].values()
            )
        )
        family[metric] = stat
        family_pvalues.append(pvalue)
    for metric, qvalue in zip(
        SCANNED_METRICS,
        benjamini_hochberg(family_pvalues),
    ):
        family[metric]["q_bh_3_family_tests"] = float(qvalue)

    raw = {}
    for name, result in results.items():
        raw[name] = [
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "coupling": float(result["coupling"][index]),
                "other_donor": (
                    float(result["other_donor"][index])
                    if np.isfinite(result["other_donor"][index])
                    else None
                ),
                "nonpartner_ag": (
                    float(result["nonpartner_ag"][index])
                    if np.isfinite(result["nonpartner_ag"][index])
                    else None
                ),
            }
            for index, example in enumerate(examples)
        ]

    artifact = {
        "study": "splicebert_gtex_finetune_shift",
        "stage": "five-seed behavior verification",
        "status": "VERIFICATION_OUTCOME_AWARE",
        "scope": (
            "Derived fixed-MLM-readout checkpoints on the existing validation cohort; "
            "no native classifier or mechanism claim."
        ),
        "base_replay": base_replay,
        "models": summaries,
        "paired_shifts_vs_base": shifts,
        "family_mean_shift": family,
        "multiplicity": {
            "individual_family": "BH across 3 behavioral contrasts x 5 seeds = 15",
            "family_mean": "BH across the 3 family-mean behavioral contrasts",
            "permutation": (
                f"{args.n_perm} gene-level sign flips; exact add-one correction"
            ),
        },
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, default=json_default) + "\n")

    manifest = ExperimentManifest.create(
        "splicebert_gtex_finetune_shift:five_seed_behavior",
        sys.argv,
        repo=REPO,
        status="VERIFICATION_OUTCOME_AWARE",
        config={
            "split": "val",
            "n": args.n,
            "batch": args.batch,
            "health_n": args.health_n,
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "device": device,
            "cohort_tag": "sealed-unc-val",
            "scanned_metrics": list(SCANNED_METRICS),
            "base_replay_atol": BASE_REPLAY_ATOL,
        },
        seeds={
            "bootstrap_base": 100,
            "bootstrap_finetunes": 2020,
            "family": 9000,
        },
        notes=[
            "Runner finalized after the seed-2020 mechanics smoke; verification tier.",
            "All five published fine-tuning seeds are included; no seed selection.",
            "Behavior and specificity only. Mechanism tracing remains frozen.",
        ],
    )
    for path in (
        preflight_path,
        reference_path,
        _model_weight(base_model),
        Path(args.gtf),
        Path(args.fasta),
    ):
        manifest.inputs.append(file_record(path))
    for model_path in fine_models.values():
        manifest.inputs.extend(
            [
                file_record(_model_weight(model_path)),
                file_record(model_path / "transplant.json"),
                file_record(model_path / "transplant.manifest.json"),
            ]
        )
    manifest.outputs.append(file_record(output))
    manifest.write(manifest_path)
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
