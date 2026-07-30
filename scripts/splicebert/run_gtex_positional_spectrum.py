#!/usr/bin/env python
"""Preflight and run the frozen five-seed GTEx positional-spectrum assay.

This runner reads no model behavior.  It first validates the spectral
instrument on the base checkpoint and synthetic controls.  Only a fully
passing preflight authorizes the fine-tuned spectrum and freezes candidate and
matched-control bands for the separately gated native-classifier surgery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

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
from gfmcd.splicing.gtex_finetune import (  # noqa: E402
    checkpoint_seed,
    load_gtex_state_dict,
)
from gfmcd.splicing.positional_spectrum import (  # noqa: E402
    band_coefficient_matrix,
    band_component,
    band_summary,
    centered_spectrum,
    compare_band,
    continuous_coefficients,
    continuous_peak,
    parseval_relative_error,
    power_spectrum,
    scan_windows,
    select_power_matched_controls,
    select_scan_candidates,
    subspace_comparison,
    validate_table,
)

EXPECTED_SHAPE = (1026, 512)
EXPECTED_BASE_HASH = "2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d"
EXPECTED_CHECKPOINT_HASHES = {
    2020: "2552486a0870e2466a223fcce34f0477c40526a09172622619f0d1672a1af45e",
    2021: "2cb06d2a17d941857485329604d526b465eabda319632c932000b85d8d40cda6",
    2022: "556e38c92319ce87fb15ca390743ae20c539f4c5400c4cca82a83cdf3d46bd0a",
    2023: "03c25882ae3e4b48198265bf8447ef543308e40ec9fd700b4834d829dcb6100b",
    2024: "d1caccec4db1784537178e355351a3e0b67cde4e70185bf6b6a6d63f792198f1",
}
EXPECTED_PRIOR_HASHES = {
    "carrier_subbin": "8a79ff0ba261020a439fc54797679a24382dcf0254f055b4450efd96c29bf7fb",
    "carrier_geometry": "2a46ebffc44b82f625adeba0ea0ff2bdada6a0ff8e8cb1233e8c3d678db248ef",
    "fig_phase_data": "91d7f51e113fecd7d8972d6b3a1f87278de9c3efe22fbc600f82e9584ebe9316",
}
EXPECTED_BASE_BAND_OVER_MEDIAN = {
    "period3": 802.4380295,
    "period10": 851.8638377,
}
NAMED_BANDS = {
    "period3": tuple(range(339, 346)),
    "period10": tuple(range(94, 101)),
}
PERIOD6_BAND = tuple(range(168, 175))
CONTINUOUS_SPECS = {
    "period3": {
        "period_low": 2.95,
        "period_high": 3.05,
        "step": 0.001,
        "nominated_period": 3.0,
    },
    "period10": {
        "period_low": 10.4,
        "period_high": 10.9,
        "step": 0.001,
        "nominated_period": 10.652435732380217,
    },
}


def _position_table(state: Mapping[str, Any]) -> tuple[str, np.ndarray]:
    hits = [
        (key, value)
        for key, value in state.items()
        if key.endswith("position_embeddings.weight")
    ]
    if len(hits) != 1:
        raise ValueError(
            "expected exactly one position_embeddings.weight tensor, "
            f"found {[key for key, _ in hits]}"
        )
    key, tensor = hits[0]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{key} is not a tensor")
    table = tensor.detach().to(dtype=torch.float64, device="cpu").numpy()
    return key, validate_table(table, shape=EXPECTED_SHAPE)


def _load_base(path: Path) -> tuple[str, np.ndarray]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("base checkpoint is not a state-dict mapping")
    return _position_table(state)


def _roundtrip_error(table: np.ndarray) -> float:
    centered = table - table.mean(axis=0, keepdims=True)
    reconstructed = np.fft.irfft(
        np.fft.rfft(centered, axis=0),
        n=table.shape[0],
        axis=0,
    )
    return float(np.max(np.abs(centered - reconstructed)))


def _relative_power_error(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.maximum(np.abs(left), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(left - right) / denominator))


def _synthetic_preflight() -> dict[str, Any]:
    n_positions = EXPECTED_SHAPE[0]
    hidden = 16
    positions = np.arange(n_positions, dtype=np.float64)
    rng = np.random.default_rng(271828)
    directions, _ = np.linalg.qr(rng.normal(size=(hidden, hidden)))
    p3_rows = directions[:, :2].T
    p10_rows = directions[:, 2:4].T
    table = (
        1.5 * np.cos(2 * np.pi * positions[:, None] / 3.0) * p3_rows[0]
        + 1.5 * np.sin(2 * np.pi * positions[:, None] / 3.0) * p3_rows[1]
        + 1.1
        * np.cos(2 * np.pi * positions[:, None] / 10.652435732380217)
        * p10_rows[0]
        + 1.1
        * np.sin(2 * np.pi * positions[:, None] / 10.652435732380217)
        * p10_rows[1]
    )
    scaled = (positions - positions.mean()) / positions.max()
    table += 0.03 * scaled[:, None] * directions[:, 4][None, :]
    permutation = np.random.default_rng(314159).permutation(n_positions)
    negative = table[permutation]
    records: dict[str, Any] = {}
    for name, truth_rows in (("period3", p3_rows), ("period10", p10_rows)):
        spec = CONTINUOUS_SPECS[name]
        peak = continuous_peak(table, **spec)
        fitted = continuous_coefficients(table, spec["nominated_period"])
        geometry = subspace_comparison(truth_rows, fitted)
        positive_power = float(
            peak["nominated_power"]
        )
        negative_power = float(
            continuous_peak(negative, **spec)["nominated_power"]
        )
        records[name] = {
            "truth_period": float(spec["nominated_period"]),
            "recovered_peak_period": float(peak["peak_period"]),
            "absolute_period_error": abs(
                float(peak["peak_period"]) - float(spec["nominated_period"])
            ),
            "coefficient_subspace_overlap": float(
                geometry["symmetric_projection_overlap"]
            ),
            "positive_power": positive_power,
            "row_permuted_negative_power": negative_power,
            "positive_exceeds_negative": positive_power > negative_power,
        }
    return records


def _spectrum_preflight(
    base: np.ndarray,
    *,
    base_hash: str,
    base_key: str,
    checkpoint_records: Mapping[int, dict[str, Any]],
    prior_hashes: Mapping[str, str],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    observed: dict[str, Any] = {}
    checks["P1_base_hash"] = base_hash == EXPECTED_BASE_HASH
    checks["P1_checkpoint_family"] = set(checkpoint_records) == set(
        EXPECTED_CHECKPOINT_HASHES
    )
    checks["P1_checkpoint_hashes"] = all(
        checkpoint_records[seed]["sha256"] == expected
        for seed, expected in EXPECTED_CHECKPOINT_HASHES.items()
    )
    checks["P1_tensor_keys"] = (
        base_key == "bert.embeddings.position_embeddings.weight"
        and all(
            record["tensor_key"] == "bert.embeddings.position_embeddings.weight"
            for record in checkpoint_records.values()
        )
    )
    checks["P1_shapes_finite"] = base.shape == EXPECTED_SHAPE and all(
        record["shape"] == list(EXPECTED_SHAPE) and record["finite"]
        for record in checkpoint_records.values()
    )
    checks["P1_prior_hashes"] = dict(prior_hashes) == EXPECTED_PRIOR_HASHES

    roundtrip = _roundtrip_error(base)
    parseval = parseval_relative_error(base)
    observed["roundtrip_max_abs_error"] = roundtrip
    observed["parseval_relative_error"] = parseval
    checks["P2_fft_roundtrip"] = roundtrip <= 1e-10
    checks["P2_parseval"] = parseval <= 1e-10

    identity = {
        name: compare_band(base, base, bins) for name, bins in NAMED_BANDS.items()
    }
    observed["identity"] = {
        name: {
            "normalized_power_ratio": record["normalized_power_ratio"],
            "reconstruction_cosine": record["reconstruction_cosine"],
            "subspace_overlap": record["subspace"]["symmetric_projection_overlap"],
        }
        for name, record in identity.items()
    }
    checks["P3_identity"] = all(
        abs(metric - 1.0) <= 1e-10
        for record in observed["identity"].values()
        for metric in record.values()
    )

    rng = np.random.default_rng(161803)
    rotation, _ = np.linalg.qr(rng.normal(size=(base.shape[1], base.shape[1])))
    base_power = power_spectrum(base)
    rotated_power = power_spectrum(base @ rotation)
    rotation_error = _relative_power_error(base_power[1:], rotated_power[1:])
    observed["dense_orthogonal_rotation_max_relative_power_error"] = rotation_error
    checks["P4_orthogonal_power_invariance"] = rotation_error <= 1e-10

    base_bands = {
        name: band_summary(base, bins) for name, bins in NAMED_BANDS.items()
    }
    observed["base_band_replay"] = {
        name: {
            "observed_band_over_median": summary["band_over_median"],
            "expected_band_over_median": EXPECTED_BASE_BAND_OVER_MEDIAN[name],
            "relative_error": abs(
                summary["band_over_median"] - EXPECTED_BASE_BAND_OVER_MEDIAN[name]
            )
            / EXPECTED_BASE_BAND_OVER_MEDIAN[name],
        }
        for name, summary in base_bands.items()
    }
    checks["P5_base_period3_replay"] = (
        observed["base_band_replay"]["period3"]["relative_error"] <= 1e-4
    )
    checks["P5_base_period10_replay"] = (
        observed["base_band_replay"]["period10"]["relative_error"] <= 1e-4
    )

    base_p10_peak = continuous_peak(base, **CONTINUOUS_SPECS["period10"])
    observed["base_period10_continuous_replay"] = base_p10_peak
    checks["P6_base_period10_subbin"] = (
        abs(base_p10_peak["peak_period"] - 10.6524357) <= 0.01
    )

    synthetic = _synthetic_preflight()
    observed["synthetic"] = synthetic
    checks["P7_synthetic_period_recovery"] = all(
        record["absolute_period_error"] <= 0.01 for record in synthetic.values()
    )
    checks["P7_synthetic_subspace_recovery"] = all(
        record["coefficient_subspace_overlap"] >= 0.99
        for record in synthetic.values()
    )
    checks["P8_row_permuted_negative"] = all(
        record["positive_exceeds_negative"] for record in synthetic.values()
    )

    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": observed,
    }


def _family_labels(seed_records: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(seed_records[seed]["normalized_power_ratio"])
        for seed in sorted(seed_records)
    ]
    overlaps = [
        float(
            seed_records[seed]["subspace"]["symmetric_projection_overlap"]
        )
        for seed in sorted(seed_records)
    ]
    if all(value >= 2.0 for value in ratios):
        power = "power_amplified"
    elif all(value <= 0.1 for value in ratios):
        power = "power_strongly_attenuated"
    elif all(value >= 0.5 for value in ratios):
        power = "power_retained"
    else:
        power = None
    if all(value >= 0.5 for value in overlaps):
        coordinates = "coordinate_retained"
    elif all(ratio >= 0.5 for ratio in ratios) and all(
        overlap < 0.25 for overlap in overlaps
    ):
        coordinates = "coordinate_rewritten"
    else:
        coordinates = None
    return {
        "power_family_label": power,
        "coordinate_family_label": coordinates,
        "normalized_power_ratios": ratios,
        "subspace_overlaps": overlaps,
    }


def _write_artifact(
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
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--protocol",
        default=str(
            REPO
            / "studies/splicebert_gtex_finetune_shift/"
            "PREREGISTRATION_POSITIONAL_SPECTRUM_V1.md"
        ),
    )
    parser.add_argument(
        "--carrier-subbin",
        default=str(REPO / "results/phase-lattice/carrier_subbin.json"),
    )
    parser.add_argument(
        "--carrier-geometry",
        default=str(REPO / "results/phase-lattice/carrier_geometry.json"),
    )
    parser.add_argument(
        "--fig-phase-data",
        default=str(REPO / "results/phase-lattice/fig_phase_data.json"),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_weights).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    prior_paths = {
        "carrier_subbin": Path(args.carrier_subbin).expanduser().resolve(),
        "carrier_geometry": Path(args.carrier_geometry).expanduser().resolve(),
        "fig_phase_data": Path(args.fig_phase_data).expanduser().resolve(),
    }
    output = guard_output_path(Path(args.out).expanduser().resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    manifest = ExperimentManifest.create(
        "gtex-positional-spectrum-v1",
        sys.argv,
        repo=REPO,
        config={
            "protocol_sha256": sha256_file(protocol_path),
            "named_bands": {
                name: list(bins) for name, bins in NAMED_BANDS.items()
            },
            "period6_exclusion": list(PERIOD6_BAND),
            "continuous_specs": CONTINUOUS_SPECS,
            "scan": {"width": 7, "start_min": 3, "start_max": 507},
        },
        seeds={"synthetic": 271828, "row_permutation": 314159, "rotation": 161803},
    )
    for path in [base_path, protocol_path, *prior_paths.values()]:
        manifest.add_input(path)

    base_hash = sha256_file(base_path)
    base_key, base = _load_base(base_path)
    checkpoint_paths = {
        checkpoint_seed(path): path
        for path in sorted(checkpoint_dir.glob("*.pt"))
        if checkpoint_seed(path) is not None
    }
    checkpoint_records: dict[int, dict[str, Any]] = {}
    tuned_tables: dict[int, np.ndarray] = {}
    for seed, path in sorted(checkpoint_paths.items()):
        assert seed is not None
        manifest.add_input(path)
        state = load_gtex_state_dict(path)
        tensor_key, table = _position_table(state)
        checkpoint_records[seed] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "tensor_key": tensor_key,
            "shape": list(table.shape),
            "finite": bool(np.isfinite(table).all()),
        }
        tuned_tables[seed] = table

    prior_hashes = {name: sha256_file(path) for name, path in prior_paths.items()}
    preflight = _spectrum_preflight(
        base,
        base_hash=base_hash,
        base_key=base_key,
        checkpoint_records=checkpoint_records,
        prior_hashes=prior_hashes,
    )
    common: dict[str, Any] = {
        "experiment_id": "gtex-positional-spectrum-v1",
        "protocol": file_record(protocol_path),
        "evidence_level": "position_table_weight_space_only",
        "sources": {
            "base": {
                "path": str(base_path),
                "sha256": base_hash,
                "tensor_key": base_key,
                "shape": list(base.shape),
            },
            "fine_tuned": {
                str(seed): record
                for seed, record in sorted(checkpoint_records.items())
            },
            "prior_hashes": prior_hashes,
        },
        "preflight": preflight,
    }
    if not preflight["passed"]:
        payload = {
            **common,
            "status": "STOPPED_PREFLIGHT",
            "target_spectrum_opened": False,
            "surgery_authorized": False,
            "strongest_claim": (
                "The positional-spectrum instrument failed its frozen preflight; "
                "no fine-tuned spectrum was interpreted."
            ),
        }
        _write_artifact(output, manifest_path, payload, manifest)
        raise SystemExit("spectrum preflight failed; wrote stopped artifact")

    base_records: dict[str, Any] = {}
    comparisons: dict[str, dict[int, dict[str, Any]]] = {
        name: {} for name in NAMED_BANDS
    }
    continuous: dict[str, dict[str, Any]] = {}
    for name, bins in NAMED_BANDS.items():
        base_records[name] = band_summary(base, bins)
        continuous[name] = {"base": continuous_peak(base, **CONTINUOUS_SPECS[name])}
        for seed, table in sorted(tuned_tables.items()):
            comparisons[name][seed] = compare_band(base, table, bins)
            continuous[name][str(seed)] = continuous_peak(
                table, **CONTINUOUS_SPECS[name]
            )

    scan = scan_windows(base, tuned_tables, width=7, start_min=3, start_max=507)
    protected_for_candidates = [
        NAMED_BANDS["period3"],
        NAMED_BANDS["period10"],
        PERIOD6_BAND,
    ]
    candidates = select_scan_candidates(
        scan,
        protected_bands=protected_for_candidates,
        maximum_per_direction=2,
        minimum_fold=2.0,
    )
    targets: list[dict[str, Any]] = [
        {"name": name, "kind": "named", "bins": list(bins)}
        for name, bins in NAMED_BANDS.items()
    ]
    targets.extend(
        {
            "name": candidate["name"],
            "kind": "scan_candidate",
            "direction": candidate["direction"],
            "bins": candidate["bins"],
        }
        for candidate in candidates
    )
    all_target_bands = [target["bins"] for target in targets]
    protected_for_controls = [*all_target_bands, PERIOD6_BAND]
    controls = {
        str(seed): select_power_matched_controls(
            table,
            targets,
            protected_bands=protected_for_controls,
            n_controls=3,
            tolerance=0.25,
            start_min=3,
            start_max=507,
        )
        for seed, table in sorted(tuned_tables.items())
    }
    missing_controls = [
        {"seed": seed, "target": target["name"]}
        for seed, seed_controls in controls.items()
        for target in targets
        if seed_controls[target["name"]] is None
    ]
    surgery_authorized = not missing_controls
    family_labels = {
        name: _family_labels(seed_records)
        for name, seed_records in comparisons.items()
    }
    payload = {
        **common,
        "status": "COMPLETE",
        "target_spectrum_opened": True,
        "analysis": {
            "named_bands": {
                name: list(bins) for name, bins in NAMED_BANDS.items()
            },
            "base": base_records,
            "named_band_comparisons": {
                name: {
                    str(seed): record
                    for seed, record in sorted(seed_records.items())
                }
                for name, seed_records in comparisons.items()
            },
            "continuous_frequency": continuous,
            "family_labels": family_labels,
            "full_scan": {
                "width": 7,
                "start_min": 3,
                "start_max": 507,
                "n_tests": len(scan),
                "records": scan,
                "candidates": candidates,
                "inference_note": (
                    "Five-sign exact add-one inference is resolution-limited; "
                    "scan q-values are reported but do not license selection."
                ),
            },
        },
        "surgery_package": {
            "authorized": surgery_authorized,
            "target_order": targets,
            "matched_controls_by_seed": controls,
            "missing_control_sets": missing_controls,
            "selection_used_classifier_outputs": False,
        },
        "surgery_authorized": surgery_authorized,
        "strongest_claim": (
            "The artifact licenses only frozen-threshold statements about "
            "fine-tuned positional-table spectral power and hidden-coordinate "
            "geometry; it does not establish attention use, causality, or biology."
        ),
    }
    _write_artifact(output, manifest_path, payload, manifest)
    print(json.dumps({
        "status": payload["status"],
        "preflight_passed": preflight["passed"],
        "family_labels": family_labels,
        "candidates": candidates,
        "surgery_authorized": surgery_authorized,
        "missing_control_sets": missing_controls,
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
