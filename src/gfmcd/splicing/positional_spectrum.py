"""Pure estimators for fine-tuned positional-carrier spectrum assays.

All functions operate on finite NumPy position tables and make no model
forward pass.  They deliberately keep Fourier power, hidden-coordinate
geometry, and candidate selection separate from behavioral intervention.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np

from gfmcd.crossmodel.carrier_spectrum import leakage_aware_band_power, periodogram
from gfmcd.rigor import benjamini_hochberg


def validate_table(table: np.ndarray, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Return a float64 finite position table after enforcing its contract."""

    values = np.asarray(table, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("position table must be a two-dimensional nontrivial matrix")
    if shape is not None and values.shape != shape:
        raise ValueError(f"position table shape {values.shape} does not match {shape}")
    if not np.isfinite(values).all():
        raise ValueError("position table contains non-finite values")
    return values


def centered_spectrum(table: np.ndarray) -> np.ndarray:
    """Real FFT of a table after subtracting the per-dimension row mean."""

    values = validate_table(table)
    return np.fft.rfft(values - values.mean(axis=0, keepdims=True), axis=0)


def power_spectrum(table: np.ndarray) -> np.ndarray:
    """Per-bin Fourier power summed over hidden dimensions."""

    spectrum = centered_spectrum(table)
    return np.square(np.abs(spectrum)).sum(axis=1)


def parseval_relative_error(table: np.ndarray) -> float:
    """Relative error in the real-FFT Parseval identity."""

    values = validate_table(table)
    centered = values - values.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    frequency_energy = np.square(np.abs(spectrum))
    weighted = frequency_energy[0].sum()
    if values.shape[0] % 2 == 0:
        weighted += frequency_energy[-1].sum()
        weighted += 2.0 * frequency_energy[1:-1].sum()
    else:
        weighted += 2.0 * frequency_energy[1:].sum()
    frequency_total = float(weighted / values.shape[0])
    spatial_total = float(np.square(centered).sum())
    return abs(frequency_total - spatial_total) / max(
        spatial_total, np.finfo(np.float64).tiny
    )


def _validated_bins(bins: Sequence[int], n_positions: int) -> tuple[int, ...]:
    selected = tuple(sorted(set(int(index) for index in bins)))
    if not selected:
        raise ValueError("at least one Fourier bin is required")
    if selected[0] <= 0 or selected[-1] > n_positions // 2:
        raise ValueError(f"Fourier bins must lie in [1, {n_positions // 2}]")
    return selected


def band_component(table: np.ndarray, bins: Sequence[int]) -> np.ndarray:
    """Real, mean-zero table component supported only on ``bins``."""

    values = validate_table(table)
    selected_bins = _validated_bins(bins, values.shape[0])
    spectrum = centered_spectrum(values)
    selected = np.zeros_like(spectrum)
    selected[list(selected_bins)] = spectrum[list(selected_bins)]
    return np.fft.irfft(selected, n=values.shape[0], axis=0)


def delete_band(table: np.ndarray, bins: Sequence[int]) -> np.ndarray:
    """Delete a nominated mean-zero Fourier component while retaining the mean."""

    values = validate_table(table)
    return values - band_component(values, bins)


def replace_band(
    table: np.ndarray,
    source: np.ndarray,
    bins: Sequence[int],
) -> np.ndarray:
    """Replace ``table``'s nominated component with ``source``'s component."""

    values = validate_table(table)
    source_values = validate_table(source, shape=values.shape)
    return values - band_component(values, bins) + band_component(source_values, bins)


def scale_band(
    table: np.ndarray,
    bins: Sequence[int],
    scale: float,
) -> np.ndarray:
    """Scale one mean-zero Fourier component without changing other bins."""

    values = validate_table(table)
    if not np.isfinite(scale):
        raise ValueError("scale must be finite")
    return values + (float(scale) - 1.0) * band_component(values, bins)


def phase_rotate_band(
    table: np.ndarray,
    bins: Sequence[int],
    phases: Sequence[float],
) -> np.ndarray:
    """Rotate nominated rFFT coefficients while preserving their magnitudes."""

    values = validate_table(table)
    selected = _validated_bins(bins, values.shape[0])
    angles = np.asarray(phases, dtype=np.float64)
    if angles.shape != (len(selected),) or not np.isfinite(angles).all():
        raise ValueError("phases must be one finite angle per selected bin")
    mean = values.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(values - mean, axis=0)
    spectrum[list(selected)] *= np.exp(1j * angles[:, None])
    return mean + np.fft.irfft(spectrum, n=values.shape[0], axis=0)


def band_coefficient_matrix(table: np.ndarray, bins: Sequence[int]) -> np.ndarray:
    """Stack real and imaginary coefficient vectors for hidden-space geometry."""

    values = validate_table(table)
    selected = _validated_bins(bins, values.shape[0])
    coefficients = centered_spectrum(values)[list(selected)]
    return np.concatenate([coefficients.real, coefficients.imag], axis=0)


def _row_basis(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    values = np.asarray(matrix, dtype=np.float64)
    _, singular, vh = np.linalg.svd(values, full_matrices=False)
    rank = int(np.linalg.matrix_rank(values))
    return vh[:rank], singular, rank


def subspace_comparison(left: np.ndarray, right: np.ndarray) -> dict:
    """Symmetric projection overlap and principal cosines of two row spaces."""

    left_basis, left_singular, left_rank = _row_basis(left)
    right_basis, right_singular, right_rank = _row_basis(right)
    if left_rank == 0 or right_rank == 0:
        overlap = 1.0 if left_rank == right_rank == 0 else 0.0
        principal = np.empty(0, dtype=np.float64)
        left_covered = float(overlap)
        right_covered = float(overlap)
    else:
        cross = left_basis @ right_basis.T
        squared = float(np.square(cross).sum())
        left_covered = squared / left_rank
        right_covered = squared / right_rank
        overlap = 0.5 * (left_covered + right_covered)
        principal = np.linalg.svd(cross, compute_uv=False)
    return {
        "symmetric_projection_overlap": float(overlap),
        "left_covered_by_right": float(left_covered),
        "right_covered_by_left": float(right_covered),
        "principal_cosines": [float(value) for value in principal],
        "left_rank": left_rank,
        "right_rank": right_rank,
        "left_singular_values": [float(value) for value in left_singular],
        "right_singular_values": [float(value) for value in right_singular],
    }


def dominant_subspace_comparison(
    left: np.ndarray,
    right: np.ndarray,
    ranks: Sequence[int],
) -> dict[str, dict]:
    """Compare leading energy-ranked row subspaces at fixed SVD ranks.

    This is a sensitivity descriptor for coefficient matrices whose numerical
    full rank includes low-energy directions. It must not silently replace a
    preregistered full-rank subspace estimand.
    """

    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    _, left_singular, left_vh = np.linalg.svd(left_values, full_matrices=False)
    _, right_singular, right_vh = np.linalg.svd(right_values, full_matrices=False)
    maximum = min(left_vh.shape[0], right_vh.shape[0])
    selected = tuple(sorted(set(int(rank) for rank in ranks)))
    if not selected or selected[0] < 1 or selected[-1] > maximum:
        raise ValueError(f"ranks must lie in [1, {maximum}]")
    left_energy = np.square(left_singular)
    right_energy = np.square(right_singular)
    result: dict[str, dict] = {}
    for rank in selected:
        cross = left_vh[:rank] @ right_vh[:rank].T
        principal = np.linalg.svd(cross, compute_uv=False)
        result[str(rank)] = {
            "rank": rank,
            "symmetric_projection_overlap": float(np.square(cross).sum() / rank),
            "principal_cosines": [float(value) for value in principal],
            "left_cumulative_energy_fraction": float(
                left_energy[:rank].sum() / left_energy.sum()
            ),
            "right_cumulative_energy_fraction": float(
                right_energy[:rank].sum() / right_energy.sum()
            ),
        }
    return result


def reconstruction_cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Raw cosine between two table reconstructions."""

    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(
        np.linalg.vector_norm(left_values) * np.linalg.vector_norm(right_values)
    )
    if denominator <= np.finfo(np.float64).tiny:
        return 1.0 if np.array_equal(left_values, right_values) else 0.0
    return float(np.dot(left_values, right_values) / denominator)


def band_summary(table: np.ndarray, bins: Sequence[int]) -> dict:
    """Weight-only power and hidden-coordinate summary for one exact-bin band."""

    values = validate_table(table)
    selected = _validated_bins(bins, values.shape[0])
    power = power_spectrum(values)
    band_power = float(power[list(selected)].sum())
    non_dc_total = float(power[1:].sum())
    coefficient = band_coefficient_matrix(values, selected)
    _, singular, rank = _row_basis(coefficient)
    singular_energy = np.square(singular)
    singular_fraction = singular_energy / max(
        float(singular_energy.sum()), np.finfo(np.float64).tiny
    )
    peak_local = int(selected[int(np.argmax(power[list(selected)]))])
    return {
        "bins": list(selected),
        "raw_power": band_power,
        "median_non_dc_bin_power": float(np.median(power[1:])),
        "band_over_median": band_power / float(np.median(power[1:])),
        "non_dc_power_fraction": band_power / non_dc_total,
        "peak_bin": peak_local,
        "peak_bin_power": float(power[peak_local]),
        "rank": rank,
        "singular_values": [float(value) for value in singular],
        "singular_energy_fractions": [float(value) for value in singular_fraction],
    }


def compare_band(base: np.ndarray, tuned: np.ndarray, bins: Sequence[int]) -> dict:
    """Compare one exact-bin carrier between base and fine-tuned tables."""

    base_values = validate_table(base)
    tuned_values = validate_table(tuned, shape=base_values.shape)
    selected = _validated_bins(bins, base_values.shape[0])
    base_summary = band_summary(base_values, selected)
    tuned_summary = band_summary(tuned_values, selected)
    base_component = band_component(base_values, selected)
    tuned_component = band_component(tuned_values, selected)
    update = tuned_values - base_values
    update_power = power_spectrum(update)
    update_total = float(update_power[1:].sum())
    return {
        "base": base_summary,
        "tuned": tuned_summary,
        "raw_power_ratio": tuned_summary["raw_power"] / base_summary["raw_power"],
        "normalized_power_ratio": (
            tuned_summary["non_dc_power_fraction"]
            / base_summary["non_dc_power_fraction"]
        ),
        "centered_update_energy_fraction": (
            float(update_power[list(selected)].sum()) / update_total
            if update_total > 0
            else 0.0
        ),
        "reconstruction_cosine": reconstruction_cosine(
            base_component, tuned_component
        ),
        "subspace": subspace_comparison(
            band_coefficient_matrix(base_values, selected),
            band_coefficient_matrix(tuned_values, selected),
        ),
    }


def continuous_coefficients(
    table: np.ndarray, period: float, *, envdeg: int = 6
) -> np.ndarray:
    """Cosine/sine hidden coefficient rows from the leakage-aware joint fit."""

    values = validate_table(table)
    n_positions = values.shape[0]
    positions = np.arange(n_positions, dtype=np.float64)
    scaled = (
        (positions - (n_positions - 1) / 2.0)
        / ((n_positions - 1) / 2.0)
    )
    columns = [scaled**degree for degree in range(envdeg + 1)]
    omega = 2.0 * np.pi / float(period)
    columns.extend([np.cos(omega * positions), np.sin(omega * positions)])
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return coefficients[-2:]


def continuous_peak(
    table: np.ndarray,
    *,
    period_low: float,
    period_high: float,
    step: float,
    nominated_period: float,
    envdeg: int = 6,
) -> dict:
    """Leakage-aware nominated power and maximum on a frozen period grid."""

    if not (0 < period_low <= nominated_period <= period_high and step > 0):
        raise ValueError("invalid continuous-period grid")
    count = int(round((period_high - period_low) / step))
    periods = period_low + step * np.arange(count + 1, dtype=np.float64)
    powers = periodogram(validate_table(table), periods, envdeg=envdeg)
    best = int(np.argmax(powers))
    return {
        "period_low": float(period_low),
        "period_high": float(period_high),
        "step": float(step),
        "n_periods": int(periods.size),
        "peak_period": float(periods[best]),
        "peak_power": float(powers[best]),
        "nominated_period": float(nominated_period),
        "nominated_power": float(
            leakage_aware_band_power(table, nominated_period, envdeg=envdeg)
        ),
        "envdeg": int(envdeg),
    }


def exact_signflip_pvalue(values: Sequence[float]) -> float:
    """Two-sided exhaustive sign-flip p-value with the frozen add-one rule."""

    observed_values = np.asarray(values, dtype=np.float64)
    if observed_values.ndim != 1 or observed_values.size == 0:
        raise ValueError("values must be a nonempty vector")
    observed = abs(float(observed_values.mean()))
    exceed = 0
    total = 0
    for signs in product((-1.0, 1.0), repeat=observed_values.size):
        total += 1
        statistic = abs(float(np.mean(observed_values * np.asarray(signs))))
        exceed += statistic >= observed - 1e-15
    return float((exceed + 1) / (total + 1))


def scan_windows(
    base: np.ndarray,
    tuned_by_seed: Mapping[int, np.ndarray],
    *,
    width: int = 7,
    start_min: int = 3,
    start_max: int = 507,
) -> list[dict]:
    """Complete frozen scan of normalized power change in contiguous windows."""

    base_values = validate_table(base)
    seeds = tuple(sorted(int(seed) for seed in tuned_by_seed))
    if not seeds:
        raise ValueError("at least one tuned table is required")
    tables = {
        int(seed): validate_table(tuned_by_seed[seed], shape=base_values.shape)
        for seed in seeds
    }
    if width < 1 or start_min < 1 or start_max + width - 1 > base_values.shape[0] // 2:
        raise ValueError("invalid scan bounds")
    base_power = power_spectrum(base_values)
    base_total = float(base_power[1:].sum())
    tuned_power = {seed: power_spectrum(table) for seed, table in tables.items()}
    tuned_total = {seed: float(power[1:].sum()) for seed, power in tuned_power.items()}
    records: list[dict] = []
    for start in range(start_min, start_max + 1):
        bins = tuple(range(start, start + width))
        base_fraction = float(base_power[list(bins)].sum()) / base_total
        ratios = {
            seed: (
                float(tuned_power[seed][list(bins)].sum()) / tuned_total[seed]
            )
            / base_fraction
            for seed in seeds
        }
        log_ratios = np.log2([ratios[seed] for seed in seeds])
        records.append(
            {
                "start": start,
                "end": start + width - 1,
                "bins": list(bins),
                "base_non_dc_power_fraction": base_fraction,
                "normalized_power_ratios": {
                    str(seed): float(ratios[seed]) for seed in seeds
                },
                "log2_normalized_power_ratios": {
                    str(seed): float(value)
                    for seed, value in zip(seeds, log_ratios, strict=True)
                },
                "median_normalized_power_ratio": float(
                    np.median([ratios[seed] for seed in seeds])
                ),
                "median_log2_normalized_power_ratio": float(np.median(log_ratios)),
                "all_same_nonzero_sign": bool(
                    np.all(log_ratios > 0) or np.all(log_ratios < 0)
                ),
                "p_exact_add_one": exact_signflip_pvalue(log_ratios),
            }
        )
    qvalues = benjamini_hochberg(
        [record["p_exact_add_one"] for record in records]
    )
    for record, qvalue in zip(records, qvalues, strict=True):
        record["q_bh_full_scan"] = float(qvalue)
    return records


def _overlaps(left: Iterable[int], right: Iterable[int]) -> bool:
    return not set(left).isdisjoint(right)


def select_scan_candidates(
    records: Sequence[dict],
    *,
    protected_bands: Sequence[Sequence[int]],
    maximum_per_direction: int = 2,
    minimum_fold: float = 2.0,
) -> list[dict]:
    """Apply the frozen no-outcome candidate rule to full-scan records."""

    protected = [tuple(int(value) for value in band) for band in protected_bands]
    selected: list[dict] = []
    for direction in ("gain", "loss"):
        eligible: list[dict] = []
        for record in records:
            ratio = float(record["median_normalized_power_ratio"])
            direction_ok = ratio >= minimum_fold if direction == "gain" else ratio <= 1 / minimum_fold
            if (
                direction_ok
                and bool(record["all_same_nonzero_sign"])
                and not any(_overlaps(record["bins"], band) for band in protected)
            ):
                eligible.append(record)
        eligible.sort(
            key=lambda record: (
                -abs(float(record["median_log2_normalized_power_ratio"])),
                int(record["start"]),
            )
        )
        chosen = 0
        for record in eligible:
            if any(_overlaps(record["bins"], item["bins"]) for item in selected):
                continue
            selected.append(
                {
                    "name": f"scan_{direction}_{chosen + 1}",
                    "direction": direction,
                    "bins": list(record["bins"]),
                    "start": int(record["start"]),
                    "end": int(record["end"]),
                    "median_normalized_power_ratio": float(
                        record["median_normalized_power_ratio"]
                    ),
                    "median_log2_normalized_power_ratio": float(
                        record["median_log2_normalized_power_ratio"]
                    ),
                    "all_same_nonzero_sign": True,
                    "p_exact_add_one": float(record["p_exact_add_one"]),
                    "q_bh_full_scan": float(record["q_bh_full_scan"]),
                }
            )
            chosen += 1
            if chosen == maximum_per_direction:
                break
    return selected


def select_power_matched_controls(
    table: np.ndarray,
    targets: Sequence[Mapping[str, object]],
    *,
    protected_bands: Sequence[Sequence[int]],
    n_controls: int = 3,
    tolerance: float = 0.25,
    start_min: int = 3,
    start_max: int = 507,
) -> dict[str, list[list[int]] | None]:
    """Deterministically freeze nonoverlapping same-width power controls."""

    values = validate_table(table)
    power = power_spectrum(values)
    unavailable = [tuple(int(value) for value in band) for band in protected_bands]
    result: dict[str, list[list[int]] | None] = {}
    for target in targets:
        name = str(target["name"])
        bins = tuple(int(value) for value in target["bins"])  # type: ignore[arg-type]
        width = len(bins)
        target_power = float(power[list(bins)].sum())
        choices: list[tuple[float, int, tuple[int, ...]]] = []
        for start in range(start_min, start_max + 1):
            candidate = tuple(range(start, start + width))
            if candidate[-1] > values.shape[0] // 2:
                continue
            if any(_overlaps(candidate, band) for band in unavailable):
                continue
            candidate_power = float(power[list(candidate)].sum())
            ratio = candidate_power / target_power
            if 1.0 - tolerance <= ratio <= 1.0 + tolerance:
                choices.append((abs(float(np.log(ratio))), start, candidate))
        choices.sort(key=lambda item: (item[0], item[1]))
        selected: list[tuple[int, ...]] = []
        for _, _, candidate in choices:
            if any(_overlaps(candidate, existing) for existing in selected):
                continue
            selected.append(candidate)
            if len(selected) == n_controls:
                break
        if len(selected) != n_controls:
            result[name] = None
            continue
        result[name] = [list(candidate) for candidate in selected]
        unavailable.extend(selected)
    return result


__all__ = [
    "band_coefficient_matrix",
    "band_component",
    "band_summary",
    "centered_spectrum",
    "compare_band",
    "continuous_coefficients",
    "continuous_peak",
    "delete_band",
    "dominant_subspace_comparison",
    "exact_signflip_pvalue",
    "parseval_relative_error",
    "phase_rotate_band",
    "power_spectrum",
    "reconstruction_cosine",
    "replace_band",
    "scale_band",
    "scan_windows",
    "select_power_matched_controls",
    "select_scan_candidates",
    "subspace_comparison",
    "validate_table",
]
