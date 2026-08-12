"""Phase-sensitive readouts for the learned period-3 carrier, and dose-retention arithmetic.

WHY THIS MODULE EXISTS
----------------------
``lineage_pretraining.period3_band_power`` sums squared moduli over the carrier's rFFT bins.
That number is **blind to phase by construction**: rotating every coefficient in the band by a
common angle -- this study's own carrier operator, ``positional_spectrum.phase_rotate_band`` --
leaves it unchanged to floating-point noise, while the table is materially different. The pilot
artifact's equal band power across arms (198,761 vs 198,592) was briefly read as "the corpus
manipulation did not go through the carrier"; that reading was withdrawn
(``studies/splicebert_positional_proxy_modifiability/DECISIONS.md``, 2026-08-12, "bounding the
entry above"), and the blindness is memorialised in
``tests/test_lineage_pretraining.py::test_period3_band_power_is_blind_to_a_phase_rotation``.

This module supplies the readout that is **not** blind: the complex band coefficients, a per-bin
rotation angle relating one table's carrier to another's, and the coherence that says whether a
single rotation explains the difference at all. Amplitude alone is the mistake that made this
necessary.

Deliberately model-free: NumPy only, no ``torch``, no checkpoint, no genome. Every function here
is a pure function of arrays and is tested without the ``splicebert`` extra.

THE ESTIMATOR, AND WHY IT IS THIS ONE
-------------------------------------
At rFFT bin ``b`` a position table of shape ``(N, H)`` contributes a complex vector
``c[b] in C^H`` -- one coefficient per hidden dimension, not a scalar. "The phase of the carrier"
is therefore not defined until a reference is named. Two readings are supplied and they answer
different questions:

* ``band_rotation`` compares two tables. The angle is
  ``arg <c_ref[b], c_tab[b]>`` with the Hermitian product ``<u, v> = sum_h conj(u_h) v_h``. If
  ``c_tab = exp(i*theta) c_ref`` -- exactly what ``phase_rotate_band`` produces -- this returns
  ``theta`` on the nose, and it is the least-squares best global rotation otherwise. The
  companion ``coherence`` (in ``[0, 1]``) says how much of the difference a single rotation can
  explain: ``1`` means the two bins differ by a complex scalar and nothing else, and a small
  value means the reported angle summarises a change that is *not* a rotation.
* ``band_phase_profile`` reports an absolute phase per bin, read at one nominated hidden
  dimension. It needs a reference dimension supplied by the caller (normally the base table's
  dominant one) so that arms are read at the *same* coordinate and their absolute phases are
  comparable.

SIGN CONVENTION, PINNED
-----------------------
NumPy's ``rfft`` uses ``X[b] = sum_n x[n] exp(-2i*pi*b*n/N)``. Translating a table by ``s``
positions (``x'[n] = x[n - s]``, i.e. ``numpy.roll(x, s, axis=0)``) multiplies bin ``b`` by
``exp(-2i*pi*b*s/N)``. So a measured rotation ``theta`` at bin ``b`` corresponds to a
translation of ``s = -theta*N / (2*pi*b)`` positions -- the sign is easy to get backwards, so
``equivalent_position_shift`` owns it and ``tests/test_carrier_phase.py`` anchors it against an
actual ``numpy.roll``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from gfmcd.splicing.positional_spectrum import centered_spectrum, validate_table

#: The learned period-3 carrier's exact rFFT bins in SpliceBERT's 1026-row position table.
#: Same object as ``pbc_carrier_factorial.P3_BINS_1026`` and ``p3_phase_surgery.P3_BAND_1026``;
#: re-exported here so a phase readout and a power readout cannot drift onto different bands.
P3_BINS_1026: tuple[int, ...] = tuple(range(339, 346))

#: Periods whose bins may never serve as a control band for a period-3 intervention: the carrier
#: itself and its neighbouring harmonics. Frozen by
#: ``studies/splicebert_positional_proxy_modifiability/DECISIONS.md`` (2026-08-12, "later").
PROTECTED_PERIODS: tuple[float, ...] = (3.0, 1.5, 6.0)

__all__ = [
    "P3_BINS_1026",
    "PROTECTED_PERIODS",
    "band_coefficients",
    "band_phase_profile",
    "band_rotation",
    "bin_energy",
    "dimension_energy",
    "dominant_dimensions",
    "equivalent_position_shift",
    "protected_period_bins",
    "redistribution",
    "retention_ratio",
    "wrap_to_pi",
]


def _selected_bins(bins: Sequence[int], n_positions: int) -> tuple[int, ...]:
    selected = tuple(sorted({int(value) for value in bins}))
    if not selected:
        raise ValueError("at least one Fourier bin is required")
    if selected[0] <= 0 or selected[-1] > n_positions // 2:
        raise ValueError(f"Fourier bins must lie in [1, {n_positions // 2}]")
    return selected


def band_coefficients(table: np.ndarray, bins: Sequence[int]) -> np.ndarray:
    """The **complex** rFFT coefficients of the mean-centred table at ``bins``.

    Shape ``(len(bins), hidden)``. Built on ``positional_spectrum.centered_spectrum`` rather than
    on a fresh ``np.fft.rfft`` so that this module and the power readouts can never disagree
    about what "the spectrum of the position table" means (mean-centring included).
    """
    values = validate_table(table)
    selected = _selected_bins(bins, values.shape[0])
    return centered_spectrum(values)[list(selected)]


def dominant_dimensions(table: np.ndarray, bins: Sequence[int]) -> tuple[int, ...]:
    """Per bin, the hidden dimension carrying the largest coefficient modulus.

    Used to fix a *coordinate* at which absolute phases are read, so that two tables are read at
    the same place. Ties resolve to the lowest index, which is ``argmax``'s own rule -- stated
    rather than left implicit, because a tie broken differently in two calls would silently
    compare two different coordinates.
    """
    coefficients = band_coefficients(table, bins)
    return tuple(int(index) for index in np.argmax(np.abs(coefficients), axis=1))


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle, or an array of angles, into ``(-pi, pi]``.

    ``(-pi, pi]`` rather than ``[-pi, pi)``: ``np.angle`` returns ``+pi`` for a negative real
    number, and a wrapper that mapped that to ``-pi`` would report a sign flip as a rotation of
    the opposite sense.
    """
    values = np.asarray(angle, dtype=np.float64)
    wrapped = np.mod(values + np.pi, 2.0 * np.pi) - np.pi
    wrapped = np.where(np.isclose(wrapped, -np.pi), np.pi, wrapped)
    return float(wrapped) if np.ndim(angle) == 0 else wrapped


def equivalent_position_shift(angle: float, bin_index: int, n_positions: int) -> float:
    """Positions of translation equivalent to a rotation of ``angle`` at ``bin_index``.

    With NumPy's ``rfft`` convention a translation by ``s`` positions multiplies bin ``b`` by
    ``exp(-2i*pi*b*s/N)``, so ``s = -angle*N / (2*pi*b)``. The result is **not** wrapped to a
    single period: a rotation only determines the shift modulo ``N / b`` (3.0 positions at the
    period-3 bin of a 1026-row table), and this returns the representative in
    ``(-N/(2b), N/(2b)]`` implied by an angle already wrapped to ``(-pi, pi]``.
    """
    if bin_index <= 0:
        raise ValueError(f"bin_index must be positive, got {bin_index}")
    if n_positions <= 0:
        raise ValueError(f"n_positions must be positive, got {n_positions}")
    if not math.isfinite(float(angle)):
        raise ValueError("angle must be finite")
    return float(-float(angle) * float(n_positions) / (2.0 * math.pi * float(bin_index)))


def band_phase_profile(
    table: np.ndarray,
    bins: Sequence[int],
    *,
    reference_dimensions: Sequence[int] | None = None,
) -> list[dict]:
    """Per-bin amplitude and an absolute phase read at a nominated hidden dimension.

    ``amplitude`` is the Euclidean norm of the bin's complex coefficient vector, so
    ``sum(amplitude**2)`` over the band is exactly ``period3_band_power``'s quantity -- the two
    readouts stay commensurable, and the point of this module is that amplitude is only half of
    the story.

    ``reference_dimensions`` fixes the coordinate at which ``phase_radians`` is read; pass the
    base table's ``dominant_dimensions`` when comparing arms, or leave it ``None`` to use each
    table's own dominant dimension (which is **not** comparable across tables).
    """
    values = validate_table(table)
    selected = _selected_bins(bins, values.shape[0])
    coefficients = band_coefficients(values, selected)
    if reference_dimensions is None:
        dimensions = np.argmax(np.abs(coefficients), axis=1)
    else:
        dimensions = np.asarray([int(d) for d in reference_dimensions], dtype=int)
        if dimensions.shape != (len(selected),):
            raise ValueError("reference_dimensions must supply one dimension per bin")
        if dimensions.min() < 0 or dimensions.max() >= values.shape[1]:
            raise ValueError("reference_dimensions fall outside the table's hidden width")
    profile: list[dict] = []
    for row, (bin_index, dimension) in enumerate(zip(selected, dimensions)):
        vector = coefficients[row]
        amplitude = float(np.linalg.norm(vector))
        phase = float(np.angle(vector[int(dimension)]))
        profile.append(
            {
                "bin": int(bin_index),
                "period": float(values.shape[0]) / float(bin_index),
                "amplitude": amplitude,
                "power": amplitude * amplitude,
                "reference_dimension": int(dimension),
                "phase_radians": phase,
                "phase_degrees": float(np.degrees(phase)),
                "reference_dimension_modulus": float(abs(vector[int(dimension)])),
            }
        )
    return profile


def band_rotation(reference: np.ndarray, table: np.ndarray, bins: Sequence[int]) -> list[dict]:
    """Per-bin rotation, coherence and amplitude ratio taking ``reference`` to ``table``.

    The angle is ``arg <c_ref, c_tab>`` under the Hermitian product, which is exact when the two
    bins differ by a global rotation and is the least-squares best rotation otherwise.

    **Read ``coherence`` before reading ``rotation_radians``.** Coherence is
    ``|<c_ref, c_tab>| / (||c_ref|| ||c_tab||)``, equal to 1 exactly when the two coefficient
    vectors are complex multiples of one another. When it is small the two bands differ in a way
    no single rotation describes, and the reported angle is then a summary of a direction, not a
    description of the change. ``residual_fraction`` is ``1 - coherence**2``: the fraction of the
    band's energy that the best rotation-and-scale cannot account for.
    """
    left = validate_table(reference)
    right = validate_table(table, shape=left.shape)
    selected = _selected_bins(bins, left.shape[0])
    left_coefficients = band_coefficients(left, selected)
    right_coefficients = band_coefficients(right, selected)

    tiny = float(np.finfo(np.float64).tiny)
    rows: list[dict] = []
    for row, bin_index in enumerate(selected):
        u = left_coefficients[row]
        v = right_coefficients[row]
        inner = complex(np.vdot(u, v))          # np.vdot conjugates its FIRST argument
        left_norm = float(np.linalg.norm(u))
        right_norm = float(np.linalg.norm(v))
        denominator = max(left_norm * right_norm, tiny)
        angle = float(wrap_to_pi(float(np.angle(inner))))
        coherence = float(min(abs(inner) / denominator, 1.0))
        rows.append(
            {
                "bin": int(bin_index),
                "period": float(left.shape[0]) / float(bin_index),
                "rotation_radians": angle,
                "rotation_degrees": float(np.degrees(angle)),
                "coherence": coherence,
                "residual_fraction": float(max(0.0, 1.0 - coherence * coherence)),
                "amplitude_reference": left_norm,
                "amplitude_table": right_norm,
                "amplitude_ratio": float(right_norm / max(left_norm, tiny)),
                "equivalent_position_shift": equivalent_position_shift(
                    angle, int(bin_index), int(left.shape[0])
                ),
            }
        )
    return rows


def bin_energy(table: np.ndarray, bins: Sequence[int]) -> np.ndarray:
    """Energy the band holds in each of its bins. Sums to the band power."""
    return np.square(np.abs(band_coefficients(table, bins))).sum(axis=1)


def dimension_energy(table: np.ndarray, bins: Sequence[int]) -> np.ndarray:
    """Energy the band holds in each hidden dimension. Also sums to the band power.

    A scalar band power is a sum over both axes and therefore conceals **redistribution** along
    either: energy can move from bin to bin inside the band, or from hidden dimension to hidden
    dimension, with the total unmoved. ``redistribution`` measures both.
    """
    return np.square(np.abs(band_coefficients(table, bins))).sum(axis=0)


def redistribution(reference_energy: np.ndarray, table_energy: np.ndarray, *, top: int = 8) -> dict:
    """How differently two energy profiles spread the same total, plus how the total moved.

    Reported on the **normalised** profiles, so this is redistribution and not amplitude: two
    tables whose band powers differ by a factor of ten but whose shape is identical score
    ``total_variation == 0`` and ``total_ratio == 10``. Both numbers are needed, and a scalar
    band power supplies neither.
    """
    left = np.asarray(reference_energy, dtype=np.float64).reshape(-1)
    right = np.asarray(table_energy, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"energy profiles must align: {left.shape} vs {right.shape}")
    if left.size == 0:
        raise ValueError("energy profiles must not be empty")
    if (left < 0).any() or (right < 0).any():
        raise ValueError("energies must be non-negative")
    tiny = float(np.finfo(np.float64).tiny)
    left_total, right_total = float(left.sum()), float(right.sum())
    left_norm = left / max(left_total, tiny)
    right_norm = right / max(right_total, tiny)
    difference = np.abs(left_norm - right_norm)
    denominator = max(
        float(np.linalg.norm(left_norm) * np.linalg.norm(right_norm)), tiny
    )
    width = int(min(top, left.size))
    left_top = set(int(i) for i in np.argsort(-left)[:width])
    right_top = set(int(i) for i in np.argsort(-right)[:width])
    return {
        "total_reference": left_total,
        "total_table": right_total,
        "total_ratio": float(right_total / max(left_total, tiny)),
        "total_variation": float(0.5 * difference.sum()),
        "max_abs_share_difference": float(difference.max()),
        "cosine": float(np.dot(left_norm, right_norm) / denominator),
        "top_k": width,
        "top_k_overlap": float(len(left_top & right_top) / width),
        "size": int(left.size),
    }


def protected_period_bins(
    n_positions: int,
    periods: Sequence[float] = PROTECTED_PERIODS,
    *,
    half_width: int = 3,
) -> dict:
    """Bins a control band may not touch, plus the periods that are not representable at all.

    A period ``p`` sits at rFFT bin ``round(N / p)``, which exists only when that bin is at or
    below Nyquist (``N // 2``). Period 1.5 in a 1026-row table maps to bin 684 and therefore has
    **no** rFFT bin -- it is aliased away rather than protected, and this function says so
    explicitly instead of silently protecting nothing.
    """
    if n_positions <= 1:
        raise ValueError(f"n_positions must exceed 1, got {n_positions}")
    if half_width < 0:
        raise ValueError(f"half_width must be non-negative, got {half_width}")
    nyquist = n_positions // 2
    bands: list[dict] = []
    unrepresentable: list[float] = []
    blocked: set[int] = set()
    for period in periods:
        value = float(period)
        if value <= 0:
            raise ValueError(f"periods must be positive, got {period}")
        centre = int(round(n_positions / value))
        if centre < 1 or centre > nyquist:
            unrepresentable.append(value)
            continue
        low = max(1, centre - half_width)
        high = min(nyquist, centre + half_width)
        band = tuple(range(low, high + 1))
        blocked.update(band)
        bands.append({"period": value, "centre_bin": centre, "bins": list(band)})
    return {
        "n_positions": int(n_positions),
        "nyquist_bin": int(nyquist),
        "half_width": int(half_width),
        "bands": bands,
        "blocked_bins": sorted(blocked),
        "unrepresentable_periods": unrepresentable,
    }


def retention_ratio(value: float, baseline: float) -> float:
    """``value / baseline``, or NaN when the baseline is exactly zero.

    Returned rather than raised because a dose ladder must still report the doses whose baseline
    happens to vanish; NaN propagates into the artifact where a reader can see it, whereas an
    exception would delete the whole ladder. **A signed ratio against a baseline near zero, or of
    the opposite sign, is not a retention** -- the caller is responsible for reporting the raw
    baseline beside it, which is why this function does not try to be clever about the sign.
    """
    denominator = float(baseline)
    if denominator == 0.0:
        return float("nan")
    return float(float(value) / denominator)
