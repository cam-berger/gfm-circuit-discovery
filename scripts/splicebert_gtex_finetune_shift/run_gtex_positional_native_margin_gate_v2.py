#!/usr/bin/env python
"""Health-preserving named-carrier interventions on native GTEx margins."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gtex_positional_native_margin_gate as v1  # noqa: E402
from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.rigor import benjamini_hochberg  # noqa: E402
from gfmcd.splicing.circuit import _dataset, eligible, pick_device  # noqa: E402
from gfmcd.splicing.coupling import CouplingExample, substitute  # noqa: E402
from gfmcd.splicing.gtex_finetune import load_gtex_token_classifier  # noqa: E402
from gfmcd.splicing.native_classifier import NativeClassifierTask  # noqa: E402
from gfmcd.splicing.positional_spectrum import (  # noqa: E402
    phase_rotate_band,
    replace_band,
    scale_band,
)

COHORT_START = 64
COHORT_STOP = 128
REMOVAL_FRACTIONS = {
    "period3": 0.5,
    "period10": 1.0,
}
PRIMARY_ESTIMANDS = (
    "attenuation_damage",
    "base_band_rescue",
    "attenuation_minus_phase",
)


def _phase_angle(removal_fraction: float) -> float:
    if not 0.0 <= removal_fraction <= 2.0:
        raise ValueError("removal fraction must lie in [0, 2]")
    return float(2.0 * np.arcsin(removal_fraction / 2.0))


def _arm_tables(
    original: np.ndarray,
    base: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    tables: dict[str, np.ndarray] = {"intact": original.copy()}
    specs: dict[str, dict[str, Any]] = {
        "intact": {"band": None, "intervention": "identity", "bins": []}
    }
    for band_name, bins in v1.EXPECTED_NAMED_BANDS.items():
        removal = REMOVAL_FRACTIONS[band_name]
        retained_scale = 1.0 - removal
        attenuate_name = f"{band_name}__attenuate"
        replace_name = f"{band_name}__base_replace"
        tables[attenuate_name] = scale_band(original, bins, retained_scale)
        tables[replace_name] = replace_band(original, base, bins)
        specs[attenuate_name] = {
            "band": band_name,
            "intervention": "attenuate",
            "removal_fraction": removal,
            "retained_scale": retained_scale,
            "bins": list(bins),
        }
        specs[replace_name] = {
            "band": band_name,
            "intervention": "base_replace",
            "bins": list(bins),
        }
        angle = _phase_angle(removal)
        for pattern_name, signs in v1.PHASE_PATTERNS.items():
            arm_name = f"{band_name}__phase_{pattern_name}"
            phases = [float(sign) * angle for sign in signs]
            tables[arm_name] = phase_rotate_band(original, bins, phases)
            specs[arm_name] = {
                "band": band_name,
                "intervention": "phase_control",
                "removal_fraction": removal,
                "phase_pattern": pattern_name,
                "phase_signs": list(signs),
                "phase_angle_radians": angle,
                "bins": list(bins),
            }
    return tables, specs


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
        row = v1._support_audit(original, table, bins)
        row["table_rms_delta"] = float(np.sqrt(np.mean(np.square(table - original))))
        if specs[name]["intervention"] == "phase_control":
            original_power = v1._band_power(original, bins)
            edited_power = v1._band_power(table, bins)
            band_name = str(specs[name]["band"])
            attenuation = scale_band(
                original,
                bins,
                1.0 - REMOVAL_FRACTIONS[band_name],
            )
            attenuation_dose = float(
                np.sqrt(np.mean(np.square(attenuation - original)))
            )
            phase_dose = float(np.sqrt(np.mean(np.square(table - original))))
            row.update(
                {
                    "band_power_relative_error": abs(
                        edited_power - original_power
                    )
                    / max(original_power, np.finfo(np.float64).tiny),
                    "dose_relative_error_vs_attenuation": abs(
                        phase_dose - attenuation_dose
                    )
                    / max(attenuation_dose, np.finfo(np.float64).tiny),
                }
            )
            row["band_power_pass"] = (
                row["band_power_relative_error"] <= v1.PHASE_POWER_TOLERANCE
            )
            row["dose_match_pass"] = (
                row["dose_relative_error_vs_attenuation"]
                <= v1.PHASE_DOSE_TOLERANCE
            )
        audits[name] = row
    return audits


def _load_replayed_cohort(
    gtf: Path,
    fasta: Path,
    cohort_source: Mapping[str, Any],
) -> tuple[list[CouplingExample], dict[str, bool]]:
    cohort = cohort_source.get("cohort", {})
    expected_ids = [
        str(value)
        for value in cohort.get("example_ids", [])[COHORT_START:COHORT_STOP]
    ]
    expected_genes = [
        str(value)
        for value in cohort.get("gene_ids", [])[COHORT_START:COHORT_STOP]
    ]
    expected_chromosomes = [
        str(value)
        for value in cohort.get("chromosomes", [])[COHORT_START:COHORT_STOP]
    ]
    if not (
        len(expected_ids)
        == len(expected_genes)
        == len(expected_chromosomes)
        == v1.EXPECTED_N
    ):
        raise RuntimeError("cohort source does not contain records 65-128")
    candidates = eligible(_dataset(str(gtf), str(fasta)), "train")
    by_id: dict[str, CouplingExample] = {}
    duplicate_ids: set[str] = set()
    for example in candidates:
        if example.example_id in by_id:
            duplicate_ids.add(example.example_id)
        by_id[example.example_id] = example
    if duplicate_ids.intersection(expected_ids):
        raise RuntimeError("V2 cohort contains duplicate annotation IDs")
    missing = [example_id for example_id in expected_ids if example_id not in by_id]
    if missing:
        raise RuntimeError(f"could not reconstruct {len(missing)} V2 examples")
    examples = [by_id[example_id] for example_id in expected_ids]
    checks = {
        "n": len(examples) == v1.EXPECTED_N,
        "example_ids_exact": [example.example_id for example in examples]
        == expected_ids,
        "gene_ids_exact": [example.gene_id for example in examples]
        == expected_genes,
        "chromosomes_exact": [example.chrom for example in examples]
        == expected_chromosomes,
        "one_example_per_gene": len({example.gene_id for example in examples})
        == v1.EXPECTED_N,
        "disjoint_from_v1_gate": not set(expected_ids).intersection(
            cohort.get("example_ids", [])[:COHORT_START]
        ),
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
        raise RuntimeError(f"V2 cohort replay failed: {checks}")
    return examples, checks


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
    original = v1._embedding_table_cpu_float64(
        model.bert.embeddings.position_embeddings.weight
    )
    raw_checkpoint = v1._position_table_from_checkpoint(checkpoint)
    transfer_error = float(np.max(np.abs(original - raw_checkpoint)))
    tables, specs = _arm_tables(original, base_table)
    audits = _mechanical_audits(original, tables, specs)

    raw: dict[str, dict[str, np.ndarray]] = {}
    for name, table in tables.items():
        print(f"    arm {name}", flush=True)
        v1._copy_embedding_table(model, table)
        raw[name] = v1._score_arm(
            task,
            clean_sequences,
            local_sequences,
            paired_sequences,
            donor_positions,
            acceptor_positions,
        )
    v1._copy_embedding_table(model, original)

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
        >= v1.MIN_INTACT_LOCAL_DESTRUCTION,
        "all_arm_health": all(
            row["finite"]
            and row["local_retention_fraction"] >= v1.MIN_HEALTH_RETENTION
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
            "spectral_support_max_relative_error": v1.SPECTRAL_SUPPORT_TOLERANCE,
            "phase_power_max_relative_error": v1.PHASE_POWER_TOLERANCE,
            "phase_dose_max_relative_error": v1.PHASE_DOSE_TOLERANCE,
            "minimum_intact_local_destruction": v1.MIN_INTACT_LOCAL_DESTRUCTION,
            "minimum_arm_health_retention": v1.MIN_HEALTH_RETENTION,
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
    intact = raw["intact"][endpoint]
    attenuated = raw[f"{band_name}__attenuate"][endpoint]
    base_replaced = raw[f"{band_name}__base_replace"][endpoint]
    phase_mean = np.mean(
        np.stack(
            [
                raw[f"{band_name}__phase_{pattern_name}"][endpoint]
                for pattern_name in v1.PHASE_PATTERNS
            ],
            axis=0,
        ),
        axis=0,
    )
    attenuation_damage = intact - attenuated
    phase_damage = intact - phase_mean
    return {
        "attenuation_damage": attenuation_damage,
        "base_band_rescue": base_replaced - attenuated,
        "base_rewrite": intact - base_replaced,
        "mean_phase_damage": phase_damage,
        "attenuation_minus_phase": attenuation_damage - phase_damage,
    }


def infer(
    raw_by_seed: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    groups: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    inference: dict[str, Any] = {}
    pvalue_locations: list[tuple[str, str, str]] = []
    pvalues: list[float] = []
    counter = 0
    for band_name in v1.EXPECTED_NAMED_BANDS:
        inference[band_name] = {}
        for endpoint in v1.ENDPOINTS:
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
                "attenuation_damage",
                "base_band_rescue",
                "base_rewrite",
                "mean_phase_damage",
                "attenuation_minus_phase",
            ):
                values = np.mean(
                    np.stack(
                        [by_seed[seed][estimand] for seed in sorted(by_seed)],
                        axis=0,
                    ),
                    axis=0,
                )
                primary = estimand in PRIMARY_ESTIMANDS
                summary = v1._summary(
                    values,
                    groups,
                    bootstrap_seed=750_000 + counter * 10 + 1,
                    permutation_seed=(
                        850_000 + counter * 10 + 2 if primary else None
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
    for band_name in v1.EXPECTED_NAMED_BANDS:
        screens[band_name] = {}
        for endpoint in v1.ENDPOINTS:
            row = inference[band_name][endpoint]
            attenuation = row["aggregate"]["attenuation_damage"]
            rescue = row["aggregate"]["base_band_rescue"]
            contrast = row["aggregate"]["attenuation_minus_phase"]
            native_use = bool(
                row["all_five_positive"]["attenuation_damage"]
                and row["all_five_positive"]["base_band_rescue"]
                and attenuation["estimate"] >= v1.MIN_SCREEN_EFFECT
                and rescue["estimate"] >= v1.MIN_SCREEN_EFFECT
                and attenuation["ci_low"] > 0
                and rescue["ci_low"] > 0
                and attenuation["q_bh_18_tests"] <= v1.MAX_Q
                and rescue["q_bh_18_tests"] <= v1.MAX_Q
            )
            specificity = bool(
                native_use
                and contrast["ci_low"] > 0
                and contrast["q_bh_18_tests"] <= v1.MAX_Q
            )
            screens[band_name][endpoint] = {
                "native_use_screen": native_use,
                "attenuation_specificity_screen": specificity,
            }
    return inference, screens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrum", required=True)
    parser.add_argument("--cohort-source", required=True)
    parser.add_argument("--v1-result", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--n", type=int, default=v1.EXPECTED_N)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=v1.EXPECTED_BOOT)
    parser.add_argument("--n-perm", type=int, default=v1.EXPECTED_PERM)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n != v1.EXPECTED_N:
        raise ValueError(f"cohort is frozen at n={v1.EXPECTED_N}")
    if args.n_boot != v1.EXPECTED_BOOT or args.n_perm != v1.EXPECTED_PERM:
        raise ValueError("inference settings do not match the frozen protocol")
    torch.set_grad_enabled(False)
    spectrum_path = Path(args.spectrum).expanduser().resolve()
    cohort_path = Path(args.cohort_source).expanduser().resolve()
    v1_path = Path(args.v1_result).expanduser().resolve()
    spectrum = json.loads(spectrum_path.read_text())
    cohort_source = json.loads(cohort_path.read_text())
    v1_result = json.loads(v1_path.read_text())
    if spectrum.get("status") != "COMPLETE" or not spectrum.get(
        "preflight", {}
    ).get("passed"):
        raise RuntimeError("spectrum source did not pass")
    named_bands = spectrum.get("analysis", {}).get("named_bands")
    if named_bands != {
        name: list(bins) for name, bins in v1.EXPECTED_NAMED_BANDS.items()
    }:
        raise RuntimeError("spectrum named bands do not match V2")
    if cohort_source.get("status") != "COMPLETE" or not cohort_source.get(
        "preflight", {}
    ).get("passed"):
        raise RuntimeError("cohort source did not pass")
    if v1_result.get("status") != "STOPPED_INTERVENTION_PREFLIGHT":
        raise RuntimeError("V1 must be the preserved stopped health diagnosis")

    expected_v1_health = {
        "period3__delete": False,
        "period3__half": True,
        "period10__delete": True,
    }
    for arm_name, expected_all_seed_health in expected_v1_health.items():
        observed = all(
            seed_result["arm_health"][arm_name]["local_retention_fraction"]
            >= v1.MIN_HEALTH_RETENTION
            for seed_result in v1_result["preflight"]["by_seed"].values()
        )
        if observed is not expected_all_seed_health:
            raise RuntimeError(f"V1 health diagnosis mismatch for {arm_name}")

    base_model = Path(args.base_model).expanduser().resolve()
    base_weight = v1._weight_file(base_model)
    base_table = v1._position_table_from_weight_file(base_weight)
    checkpoints = v1.parse_checkpoints(args.checkpoint)
    if sha256_file(base_weight) != spectrum["sources"]["base"]["sha256"]:
        raise RuntimeError("base checkpoint hash mismatch")
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
        "PREREGISTRATION_POSITIONAL_NATIVE_MARGIN_GATE_V2.md"
    )
    device = pick_device(args.device)
    examples, cohort_checks = _load_replayed_cohort(
        gtf, fasta, cohort_source
    )
    clean_sequences = [example.sequence for example in examples]
    local_sequences = [
        substitute(example.sequence, int(example.donor_pos), "CC")
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
        "gtex-positional-native-margin-gate-v2",
        sys.argv,
        repo=REPO,
        config={
            "protocol_sha256": sha256_file(protocol),
            "spectrum_sha256": sha256_file(spectrum_path),
            "cohort_source_sha256": sha256_file(cohort_path),
            "v1_result_sha256": sha256_file(v1_path),
            "cohort_slice_zero_based": [COHORT_START, COHORT_STOP],
            "removal_fractions": REMOVAL_FRACTIONS,
            "phase_angles_radians": {
                name: _phase_angle(removal)
                for name, removal in REMOVAL_FRACTIONS.items()
            },
            "phase_patterns": {
                name: list(pattern)
                for name, pattern in v1.PHASE_PATTERNS.items()
            },
            "n": v1.EXPECTED_N,
            "batch": args.batch,
            "n_boot": v1.EXPECTED_BOOT,
            "n_perm": v1.EXPECTED_PERM,
            "thresholds": {
                "spectral_support_tolerance": v1.SPECTRAL_SUPPORT_TOLERANCE,
                "phase_power_tolerance": v1.PHASE_POWER_TOLERANCE,
                "phase_dose_tolerance": v1.PHASE_DOSE_TOLERANCE,
                "minimum_intact_local_destruction": (
                    v1.MIN_INTACT_LOCAL_DESTRUCTION
                ),
                "minimum_health_retention": v1.MIN_HEALTH_RETENTION,
                "minimum_screen_effect": v1.MIN_SCREEN_EFFECT,
                "maximum_q": v1.MAX_Q,
            },
        },
        seeds={
            "bootstrap_base": 750_000,
            "permutation_base": 850_000,
        },
    )
    for path in [
        spectrum_path,
        cohort_path,
        v1_path,
        protocol,
        base_weight,
        gtf,
        fasta,
        *checkpoints.values(),
    ]:
        manifest.add_input(path)

    common = {
        "experiment_id": "gtex-positional-native-margin-gate-v2",
        "protocol": file_record(protocol),
        "spectrum_source": file_record(spectrum_path),
        "cohort_source": file_record(cohort_path),
        "v1_health_source": file_record(v1_path),
        "device": device,
        "outcome_fresh_for_named_carriers": True,
        "prior_use_of_cohort": "scan-candidate positional-band study",
        "removal_fractions": REMOVAL_FRACTIONS,
        "cohort": {
            "slice_zero_based": [COHORT_START, COHORT_STOP],
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
        print(f"seed {seed}: V2 native margin gate", flush=True)
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
            "preflight": {"passed": False, "by_seed": preflight_by_seed},
            "raw": {
                str(seed): v1._json_ready_raw(raw)
                for seed, raw in raw_by_seed.items()
            },
            "inference": None,
            "screens": None,
            "strongest_claim": (
                "The health-preserving named-carrier intervention failed at "
                "least one frozen V2 preflight check; interpretation is withheld."
            ),
        }
        v1._write(output, manifest_path, payload, manifest)
        raise SystemExit("V2 named-carrier intervention preflight failed")

    groups = np.asarray([example.gene_id for example in examples], dtype=object)
    inference, screens = infer(raw_by_seed, groups)
    payload = {
        **common,
        "status": "COMPLETE",
        "preflight": {"passed": True, "by_seed": preflight_by_seed},
        "raw": {
            str(seed): v1._json_ready_raw(raw)
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
            "A passing screen is confirmatory behavioral evidence that the "
            "named positional carrier contributes to the released fine-tuned "
            "classifier endpoint at the largest V1-health-preserving dose."
        ),
    }
    v1._write(output, manifest_path, payload, manifest)
    compact = {
        band_name: {
            endpoint: {
                "attenuation_damage": inference[band_name][endpoint][
                    "aggregate"
                ]["attenuation_damage"],
                "base_band_rescue": inference[band_name][endpoint][
                    "aggregate"
                ]["base_band_rescue"],
                "attenuation_minus_phase": inference[band_name][endpoint][
                    "aggregate"
                ]["attenuation_minus_phase"],
                **screens[band_name][endpoint],
            }
            for endpoint in v1.ENDPOINTS
        }
        for band_name in v1.EXPECTED_NAMED_BANDS
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
