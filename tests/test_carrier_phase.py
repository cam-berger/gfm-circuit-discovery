"""Memorials for the phase-sensitive carrier readout.

The load-bearing test in this file is
``test_a_phase_rotation_is_invisible_to_power_and_visible_to_band_rotation``: it plants a known
rotation, shows the power readout cannot see it, and shows this module recovers it exactly. That
pairing is the whole reason the module exists -- the pilot's equal period-3 band power across arms
was briefly read as "the carrier did not change", and a rotation is precisely the change that
reading cannot detect.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gfmcd.splicing.carrier_phase import (
    P3_BINS_1026,
    PROTECTED_PERIODS,
    band_coefficients,
    band_phase_profile,
    band_rotation,
    bin_energy,
    dimension_energy,
    dominant_dimensions,
    equivalent_position_shift,
    protected_period_bins,
    redistribution,
    retention_ratio,
    wrap_to_pi,
)
from gfmcd.splicing.lineage_pretraining import period3_band_power
from gfmcd.splicing.positional_spectrum import phase_rotate_band, scale_band

N_POSITIONS = 1026
HIDDEN = 24


def _table(seed: int = 0, n: int = N_POSITIONS, hidden: int = HIDDEN) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, hidden))


def _carrier_table(seed: int = 1) -> np.ndarray:
    """A table with a genuine period-3 carrier planted on top of noise."""
    rng = np.random.default_rng(seed)
    positions = np.arange(N_POSITIONS, dtype=np.float64)
    amplitude = rng.normal(size=(1, HIDDEN))
    carrier = np.cos(2.0 * np.pi * positions / 3.0)[:, None] * amplitude
    return carrier + 0.05 * rng.normal(size=(N_POSITIONS, HIDDEN))


# ---------------------------------------------------------------------------------------
# The memorial this module exists for
# ---------------------------------------------------------------------------------------
def test_a_phase_rotation_is_invisible_to_power_and_visible_to_band_rotation():
    table = _table(seed=7)
    angle = 0.7
    rotated = phase_rotate_band(table, P3_BINS_1026, [angle] * len(P3_BINS_1026))

    # 1. The power readout used by the pilot cannot see it.
    before = period3_band_power(table, P3_BINS_1026)
    after = period3_band_power(rotated, P3_BINS_1026)
    assert abs(after - before) / before < 1e-12

    # 2. The table really did move -- so "power unchanged" is not "table unchanged".
    assert np.abs(rotated - table).max() > 1e-3

    # 3. This module recovers the planted angle, on every bin, essentially exactly.
    for row in band_rotation(table, rotated, P3_BINS_1026):
        assert row["rotation_radians"] == pytest.approx(angle, abs=1e-9)
        assert row["coherence"] == pytest.approx(1.0, abs=1e-12)
        assert row["amplitude_ratio"] == pytest.approx(1.0, abs=1e-12)
        assert row["residual_fraction"] < 1e-12


def test_per_bin_rotations_are_recovered_independently():
    """A different angle per bin must not be smeared into one average."""
    table = _table(seed=11)
    angles = [0.1, -0.9, 2.5, 0.0, -3.0, 1.234, 0.6]
    assert len(angles) == len(P3_BINS_1026)
    rotated = phase_rotate_band(table, P3_BINS_1026, angles)
    recovered = [row["rotation_radians"] for row in band_rotation(table, rotated, P3_BINS_1026)]
    for expected, observed in zip(angles, recovered):
        assert observed == pytest.approx(float(wrap_to_pi(expected)), abs=1e-9)


# ---------------------------------------------------------------------------------------
# band_rotation: specificity, scaling, and the meaning of coherence
# ---------------------------------------------------------------------------------------
def test_identity_comparison_reports_no_rotation():
    table = _table(seed=3)
    for row in band_rotation(table, table, P3_BINS_1026):
        assert row["rotation_radians"] == pytest.approx(0.0, abs=1e-12)
        assert row["coherence"] == pytest.approx(1.0, abs=1e-12)
        assert row["amplitude_ratio"] == pytest.approx(1.0, abs=1e-12)
        assert row["equivalent_position_shift"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("scale", [0.25, 0.5, 0.75, 2.0])
def test_positive_scaling_moves_amplitude_and_not_phase(scale):
    table = _table(seed=5)
    scaled = scale_band(table, P3_BINS_1026, scale)
    for row in band_rotation(table, scaled, P3_BINS_1026):
        assert row["amplitude_ratio"] == pytest.approx(scale, rel=1e-10)
        assert row["rotation_radians"] == pytest.approx(0.0, abs=1e-10)
        assert row["coherence"] == pytest.approx(1.0, abs=1e-12)


def test_negative_scaling_is_a_half_turn():
    table = _table(seed=6)
    flipped = scale_band(table, P3_BINS_1026, -1.0)
    for row in band_rotation(table, flipped, P3_BINS_1026):
        assert abs(row["rotation_radians"]) == pytest.approx(math.pi, abs=1e-9)
        assert row["amplitude_ratio"] == pytest.approx(1.0, abs=1e-10)


def test_scaling_by_one_is_the_identity_to_the_bit():
    """The dose ladder's 1.0 rung must be the untouched table, not merely a close one."""
    table = _table(seed=9)
    assert np.array_equal(scale_band(table, P3_BINS_1026, 1.0), table)


def test_coherence_falls_when_the_change_is_not_a_rotation():
    """An unrelated band replacement is not a rotation, and coherence must say so."""
    table = _table(seed=13)
    other = _table(seed=14)
    spectrum = np.fft.rfft(table - table.mean(axis=0, keepdims=True), axis=0)
    other_spectrum = np.fft.rfft(other - other.mean(axis=0, keepdims=True), axis=0)
    spectrum[list(P3_BINS_1026)] = other_spectrum[list(P3_BINS_1026)]
    replaced = table.mean(axis=0, keepdims=True) + np.fft.irfft(
        spectrum, n=table.shape[0], axis=0
    )
    coherences = [row["coherence"] for row in band_rotation(table, replaced, P3_BINS_1026)]
    assert max(coherences) < 0.5


def test_band_rotation_rejects_a_mismatched_table():
    with pytest.raises(ValueError):
        band_rotation(_table(seed=1), _table(seed=2, hidden=HIDDEN + 1), P3_BINS_1026)


# ---------------------------------------------------------------------------------------
# The translation convention, anchored against an actual roll
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("shift", [1, 2, -1, 3])
def test_equivalent_position_shift_matches_a_real_translation(shift):
    """Roll the table by a known number of positions and read the shift back off the phase."""
    table = _carrier_table()
    rolled = np.roll(table, shift, axis=0)
    row = next(r for r in band_rotation(table, rolled, [342]) )
    # A rotation only determines the shift modulo N / b = 3.0 positions at the period-3 bin.
    expected = ((shift + 1.5) % 3.0) - 1.5
    assert row["equivalent_position_shift"] == pytest.approx(expected, abs=1e-6)


def test_equivalent_position_shift_exact_arithmetic_anchor():
    """A third of a turn at the exact period-3 bin is exactly one position."""
    assert equivalent_position_shift(2.0 * math.pi / 3.0, 342, 1026) == pytest.approx(
        -1.0, abs=1e-12
    )
    assert equivalent_position_shift(-2.0 * math.pi / 3.0, 342, 1026) == pytest.approx(
        1.0, abs=1e-12
    )
    assert equivalent_position_shift(0.0, 342, 1026) == 0.0


def test_equivalent_position_shift_rejects_nonsense():
    with pytest.raises(ValueError):
        equivalent_position_shift(0.1, 0, 1026)
    with pytest.raises(ValueError):
        equivalent_position_shift(0.1, 342, 0)
    with pytest.raises(ValueError):
        equivalent_position_shift(float("nan"), 342, 1026)


# ---------------------------------------------------------------------------------------
# wrap_to_pi
# ---------------------------------------------------------------------------------------
def test_wrap_to_pi_half_open_interval():
    assert wrap_to_pi(0.0) == 0.0
    assert wrap_to_pi(math.pi) == pytest.approx(math.pi)
    assert wrap_to_pi(-math.pi) == pytest.approx(math.pi)   # +pi, never -pi
    assert wrap_to_pi(3.0 * math.pi) == pytest.approx(math.pi)
    assert wrap_to_pi(1.5 * math.pi) == pytest.approx(-0.5 * math.pi)
    wrapped = wrap_to_pi(np.array([0.0, 4.0 * math.pi, -2.5 * math.pi]))
    assert wrapped.shape == (3,)
    assert wrapped[0] == pytest.approx(0.0)
    assert wrapped[1] == pytest.approx(0.0)
    assert wrapped[2] == pytest.approx(-0.5 * math.pi)


# ---------------------------------------------------------------------------------------
# Amplitude / phase profile
# ---------------------------------------------------------------------------------------
def test_profile_amplitudes_square_to_the_band_power():
    """The phase readout and the power readout must describe the same band."""
    table = _carrier_table()
    profile = band_phase_profile(table, P3_BINS_1026)
    total = sum(row["amplitude"] ** 2 for row in profile)
    assert total == pytest.approx(period3_band_power(table, P3_BINS_1026), rel=1e-12)


def test_profile_periods_are_the_bin_periods():
    profile = band_phase_profile(_table(), [342])
    assert profile[0]["period"] == pytest.approx(3.0)


def test_profile_reference_dimensions_are_honoured():
    table = _carrier_table()
    dimensions = dominant_dimensions(table, P3_BINS_1026)
    assert len(dimensions) == len(P3_BINS_1026)
    profile = band_phase_profile(table, P3_BINS_1026, reference_dimensions=dimensions)
    for row, dimension in zip(profile, dimensions):
        assert row["reference_dimension"] == dimension
    forced = band_phase_profile(table, P3_BINS_1026, reference_dimensions=[0] * 7)
    assert all(row["reference_dimension"] == 0 for row in forced)


def test_profile_absolute_phase_tracks_a_planted_rotation():
    table = _carrier_table()
    dimensions = dominant_dimensions(table, P3_BINS_1026)
    angle = 0.4
    rotated = phase_rotate_band(table, P3_BINS_1026, [angle] * len(P3_BINS_1026))
    before = band_phase_profile(table, P3_BINS_1026, reference_dimensions=dimensions)
    after = band_phase_profile(rotated, P3_BINS_1026, reference_dimensions=dimensions)
    for left, right in zip(before, after):
        moved = wrap_to_pi(right["phase_radians"] - left["phase_radians"])
        assert moved == pytest.approx(angle, abs=1e-9)


def test_profile_rejects_bad_reference_dimensions():
    table = _table()
    with pytest.raises(ValueError):
        band_phase_profile(table, P3_BINS_1026, reference_dimensions=[0, 1])
    with pytest.raises(ValueError):
        band_phase_profile(table, P3_BINS_1026, reference_dimensions=[HIDDEN] * 7)


def test_band_coefficients_are_complex_and_shaped_by_the_band():
    coefficients = band_coefficients(_table(), P3_BINS_1026)
    assert coefficients.shape == (len(P3_BINS_1026), HIDDEN)
    assert np.iscomplexobj(coefficients)


@pytest.mark.parametrize("bins", [[], [0], [514], [-1]])
def test_bins_outside_the_rfft_range_are_refused(bins):
    with pytest.raises(ValueError):
        band_coefficients(_table(), bins)


# ---------------------------------------------------------------------------------------
# Redistribution: the second thing a scalar band power conceals
# ---------------------------------------------------------------------------------------
def test_energy_profiles_both_sum_to_the_band_power():
    table = _carrier_table()
    total = period3_band_power(table, P3_BINS_1026)
    assert bin_energy(table, P3_BINS_1026).sum() == pytest.approx(total, rel=1e-12)
    assert dimension_energy(table, P3_BINS_1026).sum() == pytest.approx(total, rel=1e-12)
    assert bin_energy(table, P3_BINS_1026).shape == (len(P3_BINS_1026),)
    assert dimension_energy(table, P3_BINS_1026).shape == (HIDDEN,)


def test_aggregate_band_power_conceals_a_pure_dimension_redistribution():
    """Move all of one dimension's band energy into another: total unchanged, shape not.

    This is the second blindness of a scalar power number, alongside the phase blindness above,
    and it is why "aggregate period-3 band power was similar" is the only summary such a number
    licenses.
    """
    table = _carrier_table()
    spectrum = np.fft.rfft(table - table.mean(axis=0, keepdims=True), axis=0)
    rows = list(P3_BINS_1026)
    swapped = spectrum.copy()
    swapped[rows, 0] = spectrum[rows, 1]
    swapped[rows, 1] = spectrum[rows, 0]
    moved = table.mean(axis=0, keepdims=True) + np.fft.irfft(swapped, n=table.shape[0], axis=0)

    assert period3_band_power(moved, P3_BINS_1026) == pytest.approx(
        period3_band_power(table, P3_BINS_1026), rel=1e-12
    )
    report = redistribution(
        dimension_energy(table, P3_BINS_1026), dimension_energy(moved, P3_BINS_1026)
    )
    assert report["total_ratio"] == pytest.approx(1.0, rel=1e-12)
    assert report["total_variation"] > 1e-6


def test_redistribution_of_a_profile_against_itself_is_zero():
    energy = dimension_energy(_carrier_table(), P3_BINS_1026)
    report = redistribution(energy, energy)
    assert report["total_variation"] == pytest.approx(0.0, abs=1e-15)
    assert report["cosine"] == pytest.approx(1.0, abs=1e-12)
    assert report["total_ratio"] == pytest.approx(1.0, rel=1e-15)
    assert report["top_k_overlap"] == 1.0


def test_redistribution_separates_amplitude_from_shape():
    energy = dimension_energy(_carrier_table(), P3_BINS_1026)
    report = redistribution(energy, 9.0 * energy)
    assert report["total_ratio"] == pytest.approx(9.0, rel=1e-12)
    assert report["total_variation"] == pytest.approx(0.0, abs=1e-12)


def test_redistribution_rejects_bad_profiles():
    with pytest.raises(ValueError):
        redistribution(np.ones(4), np.ones(5))
    with pytest.raises(ValueError):
        redistribution(np.empty(0), np.empty(0))
    with pytest.raises(ValueError):
        redistribution(np.asarray([-1.0, 1.0]), np.ones(2))


# ---------------------------------------------------------------------------------------
# Control-band exclusion
# ---------------------------------------------------------------------------------------
def test_protected_period_bins_covers_the_carrier_and_its_representable_harmonic():
    protected = protected_period_bins(1026)
    assert set(P3_BINS_1026) <= set(protected["blocked_bins"])
    centres = {band["centre_bin"] for band in protected["bands"]}
    assert centres == {342, 171}
    # Period 1.5 lands at bin 684, above the 513-bin Nyquist limit, so it does not exist.
    assert protected["unrepresentable_periods"] == [1.5]
    assert protected["nyquist_bin"] == 513


def test_protected_period_bins_half_width_is_honoured():
    narrow = protected_period_bins(1026, [3.0], half_width=0)
    assert narrow["blocked_bins"] == [342]
    wide = protected_period_bins(1026, [3.0], half_width=3)
    assert wide["blocked_bins"] == list(range(339, 346))


def test_protected_period_bins_rejects_nonsense():
    with pytest.raises(ValueError):
        protected_period_bins(1)
    with pytest.raises(ValueError):
        protected_period_bins(1026, [0.0])
    with pytest.raises(ValueError):
        protected_period_bins(1026, [3.0], half_width=-1)


def test_protected_periods_constant_is_the_frozen_triple():
    assert PROTECTED_PERIODS == (3.0, 1.5, 6.0)


# ---------------------------------------------------------------------------------------
# Retention arithmetic
# ---------------------------------------------------------------------------------------
def test_retention_ratio_is_a_plain_ratio():
    assert retention_ratio(0.5, 2.0) == pytest.approx(0.25)
    assert retention_ratio(-0.5, 2.0) == pytest.approx(-0.25)
    assert retention_ratio(0.0, 2.0) == 0.0


def test_retention_ratio_of_a_zero_baseline_is_nan_not_an_exception():
    assert math.isnan(retention_ratio(0.3, 0.0))


def test_retention_ratio_against_a_negative_baseline_keeps_its_sign():
    """The sign trap: a *more negative* value against a negative baseline reads as >1."""
    assert retention_ratio(-0.2, -0.1) == pytest.approx(2.0)
