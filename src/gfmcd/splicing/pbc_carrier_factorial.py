"""Pure machinery for the positional-boundary-coherence carrier factorial.

This module owns no cohort construction and performs no model forward pass.  It
keeps the position-table intervention, the C43-compatible behavioral endpoint,
the factorial arithmetic, and the gate logic independently testable.  Runtime
drivers live in ``scripts/splicebert_positional_boundary_coherence``.

The arm convention is fixed throughout: ``1`` means attenuated and ``0`` means
intact, so ``N3 = Y00 - Y10`` and ``J = Y11 - Y10 - Y01 + Y00``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import permutations
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gfmcd.rigor import benjamini_hochberg
from gfmcd.splicing.positional_spectrum import (
    band_component,
    centered_spectrum,
    phase_rotate_band,
    scale_band,
    validate_table,
)


DOSE_GRID: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
P3_BINS_1026: tuple[int, ...] = tuple(range(339, 346))
P10_BINS_1026: tuple[int, ...] = tuple(range(94, 101))
P10_READERS_1026: tuple[str, ...] = (
    "L2H6",
    "L2H12",
    "L3H3",
    "L3H5",
    "L3H8",
    "L3H9",
    "L3H11",
    "L3H13",
    "L4H13",
)
P10_PERIOD_1026: float = 1026.0 / 96.0
SESOI: float = 0.05
OFF_BAND_TOLERANCE: float = 1e-10
COMMUTATIVITY_TOLERANCE: float = 1e-12
RECONSTRUCTION_TOLERANCE: float = 1e-10
MIN_STRUCTURED_CONTROLS: int = 31
MIN_PSEUDO_PHASE_CONTROLS: int = 999
STRUCTURED_CONTROL_BASE_SEED: int = 20260810
P3_CONTROL_SEED: int = 2026081003
P10_CONTROL_SEED: int = 2026081004
JOINT_CONTROL_SEED: int = 2026081005
CLEAN_DIRECTION_ENDPOINTS: tuple[str, ...] = (
    "Dplus",
    "Dminus",
    "Psi0",
    "Psi1",
    "Psi2",
)


@dataclass(frozen=True)
class IntervalResult:
    estimate: float
    lower: float
    upper: float
    adjusted_p: float
    classification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci": [self.lower, self.upper],
            "adjusted_p": self.adjusted_p,
            "classification": self.classification,
        }


@dataclass(frozen=True)
class SimultaneousIntervals:
    names: tuple[str, ...]
    estimates: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    adjusted_p: tuple[float, ...]
    standard_errors: tuple[float, ...]
    critical_value: float
    n_blocks: int
    n_bootstrap: int

    def by_name(self, *, sesoi: float = SESOI) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for index, name in enumerate(self.names):
            classification = (
                classify_interaction_interval(
                    self.lower[index], self.upper[index], sesoi=sesoi
                )
                if name == "J"
                else classify_effect_interval(
                    self.lower[index], self.upper[index], sesoi=sesoi
                )
            )
            rows[name] = IntervalResult(
                estimate=self.estimates[index],
                lower=self.lower[index],
                upper=self.upper[index],
                adjusted_p=self.adjusted_p[index],
                classification=classification,
            ).to_dict()
        return rows


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping with one canonical serialization."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_dose(dose: float) -> float:
    value = float(dose)
    if value not in DOSE_GRID:
        raise ValueError(f"dose must be one of {DOSE_GRID}; found {value}")
    return value


def attenuate_band(
    table: np.ndarray, bins: Sequence[int], dose: float
) -> np.ndarray:
    """Remove a frozen fraction of one Fourier band."""

    amount = _validate_dose(dose)
    return scale_band(table, bins, 1.0 - amount)


def replay_original_table(table: np.ndarray) -> np.ndarray:
    """Return the exact original bytes used by the identity replay arm.

    This is deliberately an identity control rather than a behavioral rescue.
    Reconstruction of an attenuated band is audited separately below.
    """

    return np.array(validate_table(table), copy=True, order="C")


def reconstruct_attenuated_band(
    edited: np.ndarray,
    original: np.ndarray,
    bins: Sequence[int],
    dose: float,
) -> np.ndarray:
    """Restore the analytically removed component for a mechanical audit."""

    amount = _validate_dose(dose)
    edited_values = validate_table(edited, shape=validate_table(original).shape)
    return edited_values + amount * band_component(original, bins)


def build_world_tables(
    original: np.ndarray,
    *,
    p3_dose: float,
    p10_dose: float | None,
) -> dict[str, np.ndarray]:
    """Build the atomic target worlds, with a P3-only staged fallback."""

    base = validate_table(original)
    d3 = _validate_dose(p3_dose)
    worlds = {
        "Y00": replay_original_table(base),
        "Y10": attenuate_band(base, P3_BINS_1026, d3),
    }
    if p10_dose is None:
        return worlds
    d10 = _validate_dose(p10_dose)
    worlds["Y01"] = attenuate_band(base, P10_BINS_1026, d10)
    worlds["Y11"] = attenuate_band(
        worlds["Y10"], P10_BINS_1026, d10
    )
    return worlds


def _phase_angle(dose: float) -> float:
    """Angle whose coefficient displacement matches attenuation RMS."""

    amount = _validate_dose(dose)
    return float(2.0 * math.asin(amount / 2.0))


def _unique_sign_patterns(
    width: int, *, n_controls: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    if n_controls < MIN_STRUCTURED_CONTROLS:
        raise ValueError(
            f"at least {MIN_STRUCTURED_CONTROLS} structured controls are required"
        )
    if n_controls > 2**width:
        raise ValueError("requested more unique phase patterns than the band permits")
    rng = np.random.default_rng(seed)
    all_patterns = np.asarray(
        [
            [1 if (mask >> index) & 1 else -1 for index in range(width)]
            for mask in range(2**width)
        ],
        dtype=np.int8,
    )
    # Select complementary pairs first. Thus every coordinate has exactly equal
    # positive/negative signs for an even family and differs by one for an odd
    # family such as the frozen 31-member null.
    representatives = np.arange(2 ** (width - 1), dtype=int)
    pair_order = rng.permutation(representatives)
    selected: list[np.ndarray] = []
    for index in pair_order[: n_controls // 2]:
        pattern = all_patterns[int(index)]
        selected.extend((pattern, -pattern))
    if n_controls % 2:
        extra = all_patterns[int(pair_order[n_controls // 2])]
        selected.append(extra if rng.integers(2) else -extra)
    output = np.stack(selected)[rng.permutation(n_controls)]
    return tuple(tuple(int(value) for value in row) for row in output)


def phase_control_tables(
    original: np.ndarray,
    bins: Sequence[int],
    dose: float,
    *,
    n_controls: int = MIN_STRUCTURED_CONTROLS,
    seed: int,
) -> tuple[tuple[np.ndarray, ...], tuple[dict[str, Any], ...]]:
    """Same-band, equal-RMS phase rotations for one attenuation family."""

    base = validate_table(original)
    selected = tuple(int(value) for value in bins)
    angle = _phase_angle(dose)
    patterns = _unique_sign_patterns(
        len(selected), n_controls=n_controls, seed=seed
    )
    tables = tuple(
        phase_rotate_band(base, selected, [sign * angle for sign in pattern])
        for pattern in patterns
    )
    specs = tuple(
        {
            "control_id": index + 1,
            "seed": int(seed),
            "bins": list(selected),
            "dose": float(dose),
            "phase_angle_radians": angle,
            "phase_signs": list(pattern),
        }
        for index, pattern in enumerate(patterns)
    )
    return tables, specs


def hidden_axis_control_tables(
    original: np.ndarray,
    bins: Sequence[int],
    dose: float,
    *,
    n_controls: int = MIN_STRUCTURED_CONTROLS,
    seed: int,
) -> tuple[tuple[np.ndarray, ...], tuple[dict[str, Any], ...]]:
    """Signed-permutation hidden-axis controls with exact spatial Gram.

    If ``C`` is the complete nominated band component and ``Q`` is a signed
    permutation, each control is ``P - dose * C Q``.  Since ``Q Q^T = I``, the
    transformed component has exactly the native spatial Gram and RMS while the
    native learned carrier itself remains present in ``P``.
    """

    base = validate_table(original)
    amount = _validate_dose(dose)
    if n_controls < MIN_STRUCTURED_CONTROLS:
        raise ValueError(
            f"at least {MIN_STRUCTURED_CONTROLS} hidden-axis controls are required"
        )
    component = band_component(base, bins)
    width = base.shape[1]
    rng = np.random.default_rng(seed)
    tables: list[np.ndarray] = []
    specs: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    while len(tables) < n_controls:
        permutation = tuple(int(value) for value in rng.permutation(width))
        signs = tuple(int(value) for value in rng.choice((-1, 1), size=width))
        key = (permutation, signs)
        if key in seen:
            continue
        seen.add(key)
        transformed = component[:, permutation] * np.asarray(signs)[None, :]
        tables.append(base - amount * transformed)
        specs.append(
            {
                "control_id": len(tables),
                "seed": int(seed),
                "bins": [int(value) for value in bins],
                "dose": amount,
                "hidden_permutation": list(permutation),
                "hidden_signs": list(signs),
            }
        )
    return tuple(tables), tuple(specs)


def build_phase_control_family(
    original: np.ndarray,
    *,
    p3_dose: float,
    p10_dose: float | None,
    n_controls: int = MIN_STRUCTURED_CONTROLS,
    seed: int = STRUCTURED_CONTROL_BASE_SEED,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Build phase, hidden-axis, and joint nulls without pooling families.

    The historical function name is retained as a stable public API; the
    returned package now implements the full frozen structured-null contract.
    """

    if seed != STRUCTURED_CONTROL_BASE_SEED:
        raise ValueError(
            f"structured-control base seed must be {STRUCTURED_CONTROL_BASE_SEED}"
        )
    base = validate_table(original)
    p3_phase_tables, p3_phase_specs = phase_control_tables(
        base,
        P3_BINS_1026,
        p3_dose,
        n_controls=n_controls,
        seed=P3_CONTROL_SEED,
    )
    p3_axis_tables, p3_axis_specs = hidden_axis_control_tables(
        base,
        P3_BINS_1026,
        p3_dose,
        n_controls=n_controls,
        seed=P3_CONTROL_SEED,
    )
    tables: dict[str, np.ndarray] = {}
    specs: dict[str, dict[str, Any]] = {}
    for table, spec in zip(p3_phase_tables, p3_phase_specs, strict=True):
        label = f"p3_phase_control_{int(spec['control_id']):03d}"
        tables[label] = table
        specs[label] = {**spec, "family": "p3_phase", "world": "Y10"}
    for table, spec in zip(p3_axis_tables, p3_axis_specs, strict=True):
        label = f"p3_axis_control_{int(spec['control_id']):03d}"
        tables[label] = table
        specs[label] = {**spec, "family": "p3_axis", "world": "Y10"}
    if p10_dose is None:
        return tables, specs

    p10_phase_tables, p10_phase_specs = phase_control_tables(
        base,
        P10_BINS_1026,
        p10_dose,
        n_controls=n_controls,
        seed=P10_CONTROL_SEED,
    )
    p10_axis_tables, p10_axis_specs = hidden_axis_control_tables(
        base,
        P10_BINS_1026,
        p10_dose,
        n_controls=n_controls,
        seed=P10_CONTROL_SEED,
    )
    for p10_table, p10_spec in zip(
        p10_phase_tables, p10_phase_specs, strict=True
    ):
        index = int(p10_spec["control_id"])
        p10_label = f"p10_phase_control_{index:03d}"
        tables[p10_label] = p10_table
        specs[p10_label] = {**p10_spec, "family": "p10_phase", "world": "Y01"}
    for table, spec in zip(p10_axis_tables, p10_axis_specs, strict=True):
        label = f"p10_axis_control_{int(spec['control_id']):03d}"
        tables[label] = table
        specs[label] = {**spec, "family": "p10_axis", "world": "Y01"}

    joint_p3, joint_p3_specs = hidden_axis_control_tables(
        base,
        P3_BINS_1026,
        p3_dose,
        n_controls=n_controls,
        seed=JOINT_CONTROL_SEED,
    )
    joint_p10, joint_p10_specs = hidden_axis_control_tables(
        base,
        P10_BINS_1026,
        p10_dose,
        n_controls=n_controls,
        seed=JOINT_CONTROL_SEED,
    )
    for p3_table, p10_table, p3_spec, p10_spec in zip(
        joint_p3, joint_p10, joint_p3_specs, joint_p10_specs, strict=True
    ):
        index = int(p3_spec["control_id"])
        if int(p10_spec["control_id"]) != index:
            raise RuntimeError("joint-control IDs drifted")
        p3_label = f"joint_p3_control_{index:03d}"
        p10_label = f"joint_p10_control_{index:03d}"
        joint_label = f"joint_control_{index:03d}"
        tables[p3_label] = p3_table
        tables[p10_label] = p10_table
        tables[joint_label] = p3_table + p10_table - base
        specs[p3_label] = {
            **p3_spec,
            "family": "joint_p3",
            "world": "Y10",
        }
        specs[p10_label] = {
            **p10_spec,
            "family": "joint_p10",
            "world": "Y01",
        }
        specs[joint_label] = {
            "control_id": index,
            "seed": JOINT_CONTROL_SEED,
            "family": "joint",
            "world": "Y11",
            "p3_control": p3_label,
            "p10_control": p10_label,
            "bins": [*P3_BINS_1026, *P10_BINS_1026],
            "p3_dose": float(p3_dose),
            "p10_dose": float(p10_dose),
        }
    return tables, specs


def _band_amplitude(table: np.ndarray, bins: Sequence[int]) -> float:
    return float(np.linalg.norm(band_component(table, bins)))


def intervention_audit(
    original: np.ndarray,
    edited: np.ndarray,
    *,
    bins: Sequence[int],
    dose: float | None = None,
    protect_bins: Sequence[int] = (),
) -> dict[str, Any]:
    """Audit spectral support, delivered RMS dose, and cross-band preservation."""

    base = validate_table(original)
    changed = validate_table(edited, shape=base.shape)
    selected = tuple(int(value) for value in bins)
    base_spectrum = centered_spectrum(base)
    changed_spectrum = centered_spectrum(changed)
    delta = changed_spectrum - base_spectrum
    outside = np.ones(delta.shape[0], dtype=bool)
    outside[list(selected)] = False
    scale = max(float(np.max(np.abs(base_spectrum))), np.finfo(float).tiny)
    outside_error = float(np.max(np.abs(delta[outside]))) / scale
    table_rms_delta = float(np.sqrt(np.mean(np.square(changed - base))))
    original_component_rms = float(
        np.sqrt(np.mean(np.square(band_component(base, selected))))
    )
    delivered = table_rms_delta / max(original_component_rms, np.finfo(float).tiny)
    row: dict[str, Any] = {
        "bins": list(selected),
        "finite_real": bool(np.isfinite(changed).all() and np.isrealobj(changed)),
        "outside_band_max_relative_delta": outside_error,
        "outside_band_pass": outside_error <= OFF_BAND_TOLERANCE,
        "table_rms_delta": table_rms_delta,
        "delivered_component_rms_fraction": delivered,
    }
    if dose is not None:
        expected = float(dose)
        row["requested_dose"] = expected
        row["dose_absolute_error"] = abs(delivered - expected)
        row["dose_pass"] = row["dose_absolute_error"] <= 1e-10
    if protect_bins:
        before = _band_amplitude(base, protect_bins)
        after = _band_amplitude(changed, protect_bins)
        row["protected_band_amplitude_retention"] = after / max(
            before, np.finfo(float).tiny
        )
        row["protected_band_pass"] = (
            abs(row["protected_band_amplitude_retention"] - 1.0) <= 1e-10
        )
    return row


def mechanical_audit(
    original: np.ndarray,
    *,
    p3_dose: float,
    p10_dose: float | None,
    controls: Mapping[str, np.ndarray] | None = None,
    control_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Full mechanical gate for target worlds and structured controls."""

    base = validate_table(original)
    worlds = build_world_tables(base, p3_dose=p3_dose, p10_dose=p10_dose)
    p3 = intervention_audit(
        base,
        worlds["Y10"],
        bins=P3_BINS_1026,
        dose=p3_dose,
        protect_bins=P10_BINS_1026,
    )
    reconstructed_p3 = reconstruct_attenuated_band(
        worlds["Y10"], base, P3_BINS_1026, p3_dose
    )
    checks: dict[str, bool] = {
        "identity_replay_byte_exact": bool(
            replay_original_table(base).tobytes(order="C")
            == np.ascontiguousarray(base).tobytes(order="C")
        ),
        "p3_off_band": bool(p3["outside_band_pass"]),
        "p3_dose": bool(p3["dose_pass"]),
        "p3_preserves_p10": bool(p3["protected_band_pass"]),
        "p3_reconstructable": bool(
            np.max(np.abs(reconstructed_p3 - base)) <= RECONSTRUCTION_TOLERANCE
        ),
    }
    result: dict[str, Any] = {"worlds": {"Y10": p3}, "checks": checks}
    if p10_dose is not None:
        p10 = intervention_audit(
            base,
            worlds["Y01"],
            bins=P10_BINS_1026,
            dose=p10_dose,
            protect_bins=P3_BINS_1026,
        )
        reconstructed_p10 = reconstruct_attenuated_band(
            worlds["Y01"], base, P10_BINS_1026, p10_dose
        )
        reverse = attenuate_band(
            worlds["Y01"], P3_BINS_1026, p3_dose
        )
        commutativity_error = float(np.max(np.abs(worlds["Y11"] - reverse)))
        result["worlds"].update({"Y01": p10})
        result["commutativity_max_abs_error"] = commutativity_error
        checks.update(
            {
                "p10_off_band": bool(p10["outside_band_pass"]),
                "p10_dose": bool(p10["dose_pass"]),
                "p10_preserves_p3": bool(p10["protected_band_pass"]),
                "p10_reconstructable": bool(
                    np.max(np.abs(reconstructed_p10 - base))
                    <= RECONSTRUCTION_TOLERANCE
                ),
                "composition_commutes": commutativity_error
                <= COMMUTATIVITY_TOLERANCE,
            }
        )
    if controls is not None:
        if control_specs is None or set(controls) != set(control_specs):
            raise ValueError("control tables and specs must have identical labels")
        control_audits: dict[str, dict[str, Any]] = {}
        for label, table in controls.items():
            spec = control_specs[label]
            family = str(spec["family"])
            if family in {"p3_phase", "p3_axis", "joint_p3"}:
                bins = P3_BINS_1026
            elif family in {"p10_phase", "p10_axis", "joint_p10"}:
                bins = P10_BINS_1026
            elif family == "joint":
                bins = (*P3_BINS_1026, *P10_BINS_1026)
            else:
                raise ValueError(f"unknown structured-null family {family}")
            row = intervention_audit(base, table, bins=bins)
            if family in {"p3_phase", "p3_axis", "joint_p3"}:
                expected = float(p3_dose)
            elif family in {"p10_phase", "p10_axis", "joint_p10"}:
                if p10_dose is None:
                    raise ValueError("P10 control supplied in P3-only mode")
                expected = float(p10_dose)
            else:
                p3_rms = float(
                    np.sqrt(np.mean(np.square(worlds["Y10"] - base)))
                )
                p10_rms = float(
                    np.sqrt(np.mean(np.square(worlds["Y01"] - base)))
                )
                expected_rms = math.sqrt(p3_rms**2 + p10_rms**2)
                observed_rms = float(
                    np.sqrt(np.mean(np.square(table - base)))
                )
                row["joint_rms_relative_error"] = abs(
                    observed_rms - expected_rms
                ) / max(expected_rms, np.finfo(float).tiny)
                row["dose_pass"] = row["joint_rms_relative_error"] <= 1e-10
                control_audits[label] = row
                continue
            row["dose_absolute_error"] = abs(
                float(row["delivered_component_rms_fraction"]) - expected
            )
            row["dose_pass"] = row["dose_absolute_error"] <= 1e-10
            if family.endswith("_axis") or family in {"joint_p3", "joint_p10"}:
                component = band_component(base, bins)
                transformed = (base - table) / expected
                gram_scale = max(
                    float(np.max(np.abs(component @ component.T))),
                    np.finfo(float).tiny,
                )
                gram_error = float(
                    np.max(
                        np.abs(
                            transformed @ transformed.T
                            - component @ component.T
                        )
                    )
                    / gram_scale
                )
                row["spatial_gram_relative_error"] = gram_error
                row["spatial_gram_pass"] = gram_error <= 1e-10
                row["dose_pass"] = bool(
                    row["dose_pass"] and row["spatial_gram_pass"]
                )
            control_audits[label] = row
        family_counts = {
            family: sum(
                str(spec["family"]) == family for spec in control_specs.values()
            )
            for family in (
                "p3_phase",
                "p3_axis",
                "p10_phase",
                "p10_axis",
                "joint_p3",
                "joint_p10",
                "joint",
            )
        }
        active_families = (
            ("p3_phase", "p3_axis")
            if p10_dose is None
            else (
                "p3_phase",
                "p3_axis",
                "p10_phase",
                "p10_axis",
                "joint_p3",
                "joint_p10",
                "joint",
            )
        )
        checks["control_family_sizes"] = all(
            family_counts[family] >= MIN_STRUCTURED_CONTROLS
            for family in active_families
        )
        checks["control_support_and_dose"] = all(
            row["outside_band_pass"] and row["dose_pass"]
            for row in control_audits.values()
        )
        result["control_counts"] = family_counts
        result["controls"] = control_audits
    result["passed"] = bool(all(checks.values()))
    return result


def c43_margin(logits: Mapping[str, float] | Sequence[float]) -> float:
    """Exact C43 masked-donor readout: G minus the mean of A/C/T."""

    if isinstance(logits, Mapping):
        values = {base: float(logits[base]) for base in "ACGT"}
    else:
        array = np.asarray(logits, dtype=float)
        if array.shape != (4,):
            raise ValueError("sequence logits must be ordered [A,C,G,T]")
        values = dict(zip("ACGT", (float(value) for value in array), strict=True))
    if not np.isfinite(list(values.values())).all():
        raise ValueError("C43 logits must be finite")
    return values["G"] - float(np.mean([values["A"], values["C"], values["T"]]))


def bounded_attention_carrier_score(
    profile: Sequence[float],
    offsets: Sequence[int],
    *,
    period: float = P10_PERIOD_1026,
    envelope_degree: int = 3,
) -> float:
    """C37-compatible bounded carrier magnitude after a smooth envelope fit.

    The caller supplies the two-sided local profile and its signed offsets.
    Each negative/positive half receives its own Hann window so the gap around
    the query cannot create the target tone (the R15-hardened convention).
    """

    attention = np.asarray(profile, dtype=float)
    distance = np.asarray(offsets, dtype=int)
    if attention.ndim != 1 or distance.shape != attention.shape:
        raise ValueError("profile and offsets must be aligned one-dimensional arrays")
    if not np.isfinite(attention).all() or np.any(attention < 0):
        raise ValueError("attention profile must be finite and non-negative")
    keep = (np.abs(distance) >= 6) & (np.abs(distance) <= 60)
    attention, distance = attention[keep], distance[keep]
    left = distance < 0
    right = distance > 0
    if left.sum() < envelope_degree + 2 or right.sum() < envelope_degree + 2:
        raise ValueError("profile does not cover both offset halves")
    order = np.argsort(distance)
    attention, distance = attention[order], distance[order]
    safe = np.maximum(attention, 1e-12)
    baseline = np.exp(
        np.polyval(np.polyfit(distance, np.log(safe), envelope_degree), distance)
    )
    ratio = safe / baseline
    n_left = int(np.sum(distance < 0))
    n_right = int(np.sum(distance > 0))
    weights = np.concatenate([np.hanning(n_left), np.hanning(n_right)])
    numerator = np.sum(
        weights * ratio * np.exp(-2j * np.pi * distance / float(period))
    )
    denominator = float(np.sum(weights * ratio))
    if denominator <= 0:
        raise ValueError("attention carrier denominator is zero")
    score = float(abs(numerator / denominator))
    if score > 1.0 + 1e-9:
        raise RuntimeError("bounded attention score exceeded one")
    return min(score, 1.0)


def coupling_from_scores(intact_f: float, corrupt_f: float) -> float:
    values = np.asarray([intact_f, corrupt_f], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("coupling scores must be finite")
    return float(values[0] - values[1])


def block_psi(
    rows: Sequence[Mapping[str, Any]], *, value_key: str = "coupling_ac"
) -> dict[str, float]:
    """Compute the exact C43 M0/M+/M-/Psi statistic for one 3x3 block."""

    if len(rows) != 9:
        raise ValueError("a phase block must contain exactly nine cells")
    seen: set[tuple[int, int]] = set()
    by_delta: dict[int, list[float]] = {0: [], 1: [], 2: []}
    by_pair: dict[tuple[int, int], float] = {}
    for row in rows:
        donor = int(row["donor_phase"])
        acceptor = int(row["acceptor_phase"])
        if donor not in (0, 1, 2) or acceptor not in (0, 1, 2):
            raise ValueError("phase labels must lie in {0,1,2}")
        pair = (donor, acceptor)
        if pair in seen:
            raise ValueError(f"duplicate phase cell {pair}")
        seen.add(pair)
        value = float(row[value_key])
        if not math.isfinite(value):
            raise ValueError("phase-cell coupling must be finite")
        by_delta[(acceptor - donor) % 3].append(value)
        by_pair[pair] = value
    if seen != {(donor, acceptor) for donor in range(3) for acceptor in range(3)}:
        raise ValueError("phase block is not a complete 3x3 lattice")
    m0, mplus, mminus = (
        float(np.mean(by_delta[index])) for index in (0, 1, 2)
    )
    result = {
        "M0": m0,
        "Mplus": mplus,
        "Mminus": mminus,
        "Psi": m0 - 0.5 * (mplus + mminus),
        "Dplus": m0 - mplus,
        "Dminus": m0 - mminus,
    }
    for donor in range(3):
        off_diagonal = [
            by_pair[(donor, acceptor)]
            for acceptor in range(3)
            if acceptor != donor
        ]
        result[f"Psi{donor}"] = by_pair[(donor, donor)] - float(
            np.mean(off_diagonal)
        )
    return result


def factorial_contrasts(worlds: Mapping[str, float]) -> dict[str, float | None]:
    """Atomic contrast map for a single block or an aggregate."""

    required = {"Y00", "Y10"}
    if not required.issubset(worlds):
        raise ValueError("Y00 and Y10 are required")
    y00, y10 = float(worlds["Y00"]), float(worlds["Y10"])
    if not np.isfinite([y00, y10]).all():
        raise ValueError("factorial worlds must be finite")
    result: dict[str, float | None] = {
        "N3": y00 - y10,
        "N10": None,
        "Nboth": None,
        "J": None,
    }
    optional = {"Y01", "Y11"}
    if optional.intersection(worlds) and not optional.issubset(worlds):
        raise ValueError("full factorial requires both Y01 and Y11")
    if optional.issubset(worlds):
        y01, y11 = float(worlds["Y01"]), float(worlds["Y11"])
        if not np.isfinite([y01, y11]).all():
            raise ValueError("factorial worlds must be finite")
        result.update(
            {
                "N10": y00 - y01,
                "Nboth": y00 - y11,
                "J": y11 - y10 - y01 + y00,
            }
        )
    return result


def classify_effect_interval(
    lower: float, upper: float, *, sesoi: float = SESOI
) -> str:
    if lower > upper:
        raise ValueError("interval lower bound exceeds upper bound")
    if lower > sesoi:
        return "MATERIAL_SUPPORTIVE"
    if upper < -sesoi:
        return "MATERIAL_SUPPRESSIVE"
    if lower >= -sesoi and upper <= sesoi:
        return "BOUNDED_NULL"
    if lower > 0:
        return "DETECTABLE_MATERIALITY_UNRESOLVED"
    if upper < 0:
        return "DETECTABLE_SUPPRESSIVE_MATERIALITY_UNRESOLVED"
    return "INCONCLUSIVE"


def classify_interaction_interval(
    lower: float, upper: float, *, sesoi: float = SESOI
) -> str:
    if lower > upper:
        raise ValueError("interval lower bound exceeds upper bound")
    if upper < -sesoi:
        return "MATERIAL_SUPER_ADDITIVE_ATTENUATION_DAMAGE"
    if lower > sesoi:
        return "MATERIAL_SUB_ADDITIVE_ATTENUATION_DAMAGE"
    if lower >= -sesoi and upper <= sesoi:
        return "BOUNDED_ADDITIVE"
    if lower > 0 or upper < 0:
        return "DETECTABLE_INTERACTION_MATERIALITY_UNRESOLVED"
    return "INCONCLUSIVE"


def exact_add_one_upper_p(target: float, controls: Sequence[float]) -> float:
    """One-sided exact add-one rank p, keeping intervention families separate."""

    null = np.asarray(controls, dtype=float)
    if len(null) < MIN_STRUCTURED_CONTROLS or not np.isfinite(null).all():
        raise ValueError(
            f"at least {MIN_STRUCTURED_CONTROLS} finite controls are required"
        )
    if not math.isfinite(float(target)):
        raise ValueError("target statistic must be finite")
    return float((1 + np.sum(null >= float(target))) / (len(null) + 1))


def exact_add_one_absolute_p(target: float, controls: Sequence[float]) -> float:
    """Two-sided finite-control rank through absolute statistics."""

    return exact_add_one_upper_p(abs(float(target)), np.abs(controls))


def _stratified_bootstrap_means(
    matrix: np.ndarray,
    strata: Sequence[str],
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    labels = np.asarray(strata, dtype=object)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != len(labels):
        raise ValueError("bootstrap values and strata must align")
    if len(values) < 6 or not np.isfinite(values).all():
        raise ValueError("at least six finite block rows are required")
    if n_bootstrap < 999:
        raise ValueError("at least 999 bootstrap draws are required")
    groups = [np.flatnonzero(labels == label) for label in sorted(set(labels))]
    rng = np.random.default_rng(seed)
    output = np.empty((n_bootstrap, values.shape[1]), dtype=float)
    for draw in range(n_bootstrap):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        output[draw] = values[indices].mean(axis=0)
    return output


def simultaneous_stratified_intervals(
    matrix: np.ndarray,
    *,
    names: Sequence[str],
    strata: Sequence[str],
    n_bootstrap: int = 20_000,
    seed: int = 20260810,
    alpha: float = 0.05,
) -> SimultaneousIntervals:
    """Stratified standardized-max bootstrap intervals and add-one max-T p-values."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("matrix columns must match endpoint names")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    estimates = values.mean(axis=0)
    bootstrap = _stratified_bootstrap_means(
        values, strata, n_bootstrap=n_bootstrap, seed=seed
    )
    standard_errors = bootstrap.std(axis=0, ddof=1)
    if np.any(standard_errors <= 0):
        raise ValueError("every endpoint requires nonzero bootstrap variance")
    centered_t = (bootstrap - estimates) / standard_errors
    max_t = np.max(np.abs(centered_t), axis=1)
    critical = float(np.quantile(max_t, 1.0 - alpha, method="higher"))
    lower = estimates - critical * standard_errors
    upper = estimates + critical * standard_errors
    observed = np.abs(estimates / standard_errors)
    adjusted_p = np.asarray(
        [
            (1 + np.sum(max_t >= statistic)) / (len(max_t) + 1)
            for statistic in observed
        ],
        dtype=float,
    )
    return SimultaneousIntervals(
        names=tuple(str(name) for name in names),
        estimates=tuple(float(value) for value in estimates),
        lower=tuple(float(value) for value in lower),
        upper=tuple(float(value) for value in upper),
        adjusted_p=tuple(float(value) for value in adjusted_p),
        standard_errors=tuple(float(value) for value in standard_errors),
        critical_value=critical,
        n_blocks=len(values),
        n_bootstrap=n_bootstrap,
    )


def percentile_stratified_interval(
    values: Sequence[float],
    strata: Sequence[str],
    *,
    n_bootstrap: int = 20_000,
    seed: int = 20260810,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    bootstrap = _stratified_bootstrap_means(
        array, strata, n_bootstrap=n_bootstrap, seed=seed
    )[:, 0]
    return {
        "estimate": float(array.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "n_blocks": int(len(array)),
        "n_bootstrap": int(n_bootstrap),
    }


def mde80_from_bootstrap(values: Sequence[float], strata: Sequence[str]) -> float:
    """Normal-approximation MDE80 using stratum-centered block residuals."""

    array = np.asarray(values, dtype=float)
    labels = np.asarray(strata, dtype=object)
    if array.shape != labels.shape or len(array) < 6:
        raise ValueError("MDE values and strata must align")
    centered = array.copy()
    for label in set(labels):
        mask = labels == label
        centered[mask] -= centered[mask].mean()
    standard_error = centered.std(ddof=1) / math.sqrt(len(centered))
    return float((1.96 + 0.8416212335729143) * standard_error)


def projected_mde80(
    values: Sequence[float],
    strata: Sequence[str],
    *,
    target_blocks: int = 600,
    n_bootstrap: int = 20_000,
    seed: int = 20260810,
) -> dict[str, Any]:
    """Project a target-mean MDE with centered stratified whole-block draws."""

    array = np.asarray(values, dtype=float)
    labels = np.asarray(strata, dtype=object)
    if array.shape != labels.shape or len(array) < 6 or not np.isfinite(array).all():
        raise ValueError("projected-MDE values and strata must align and be finite")
    if target_blocks < len(set(labels)) or n_bootstrap < 20_000:
        raise ValueError("projected MDE requires the frozen 20,000 draws and target support")
    unique = tuple(sorted(set(labels)))
    groups = [np.flatnonzero(labels == label) for label in unique]
    centered = array.copy()
    for group in groups:
        centered[group] -= centered[group].mean()
    raw_counts = np.asarray([len(group) for group in groups], dtype=float)
    allocation = np.floor(target_blocks * raw_counts / raw_counts.sum()).astype(int)
    remainder = target_blocks - int(allocation.sum())
    fractions = target_blocks * raw_counts / raw_counts.sum() - allocation
    for index in np.argsort(-fractions, kind="stable")[:remainder]:
        allocation[index] += 1
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for draw in range(n_bootstrap):
        sampled = np.concatenate(
            [
                rng.choice(group, size=int(count), replace=True)
                for group, count in zip(groups, allocation, strict=True)
            ]
        )
        draws[draw] = centered[sampled].mean()
    multiplier = 1.96 + 0.8416212335729143
    mde = float(multiplier * draws.std(ddof=1))
    batches = np.array_split(draws, 20)
    batch_mde = np.asarray(
        [multiplier * values.std(ddof=1) for values in batches], dtype=float
    )
    monte_carlo_se = float(batch_mde.std(ddof=1) / math.sqrt(len(batch_mde)))
    return {
        "mde80": mde,
        "monte_carlo_se": monte_carlo_se,
        "n_bootstrap": int(n_bootstrap),
        "target_blocks": int(target_blocks),
        "stratum_target_allocation": {
            str(label): int(count)
            for label, count in zip(unique, allocation, strict=True)
        },
        "method": "centered_stratified_whole_block_bootstrap_projection",
    }


def balanced_pseudo_phase_assignments(
    n_blocks: int,
    *,
    n_controls: int = MIN_PSEUDO_PHASE_CONTROLS,
    seed: int = 20260810,
) -> np.ndarray:
    """Freeze independently seeded balanced S3 labels for every null draw.

    The returned integer matrix has shape ``(n_controls, n_blocks)`` and stores
    an index into the lexicographically ordered six acceptor-label
    permutations. When ``n_blocks`` is divisible by six, every row contains
    each label exactly ``n_blocks / 6`` times. Child ``SeedSequence`` streams
    make draws reproducible without recycling a six-assignment orbit.
    """

    if n_controls < MIN_PSEUDO_PHASE_CONTROLS:
        raise ValueError(
            f"at least {MIN_PSEUDO_PHASE_CONTROLS} pseudo-phase controls are required"
        )
    if n_blocks < 1:
        raise ValueError("at least one phase block is required")
    quotient, remainder = divmod(int(n_blocks), 6)
    children = np.random.SeedSequence(seed).spawn(n_controls)
    assignments = np.empty((n_controls, n_blocks), dtype=np.uint8)
    for draw, child in enumerate(children):
        rng = np.random.default_rng(child)
        counts = np.full(6, quotient, dtype=int)
        if remainder:
            counts[rng.permutation(6)[:remainder]] += 1
        labels = np.repeat(np.arange(6, dtype=np.uint8), counts)
        assignments[draw] = labels[rng.permutation(n_blocks)]
    return assignments


def balanced_pseudo_phase_assignment_audit(
    n_blocks: int,
    *,
    n_controls: int = MIN_PSEUDO_PHASE_CONTROLS,
    seed: int = 20260810,
) -> dict[str, Any]:
    """Summarize balance, diversity, and the immutable assignment hash."""

    assignments = balanced_pseudo_phase_assignments(
        n_blocks, n_controls=n_controls, seed=seed
    )
    counts = np.stack(
        [np.bincount(row, minlength=6) for row in assignments], axis=0
    )
    row_hashes = tuple(sha256(row.tobytes()).hexdigest() for row in assignments)
    return {
        "generator": "numpy.SeedSequence.spawn+PCG64",
        "seed": int(seed),
        "n_controls": int(n_controls),
        "n_blocks": int(n_blocks),
        "permutation_labels": [list(value) for value in permutations((0, 1, 2))],
        "minimum_label_count": int(counts.min()),
        "maximum_label_count": int(counts.max()),
        "exactly_balanced": bool(
            n_blocks % 6 == 0 and np.all(counts == n_blocks // 6)
        ),
        "unique_assignment_count": int(len(set(row_hashes))),
        "all_assignments_unique": len(set(row_hashes)) == n_controls,
        "assignment_matrix_sha256": sha256(assignments.tobytes()).hexdigest(),
    }


def balanced_pseudo_phase_matrix(
    block_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    value_key: str = "coupling_ac",
    n_controls: int = MIN_PSEUDO_PHASE_CONTROLS,
    seed: int = 20260810,
) -> np.ndarray:
    """Balanced whole-lattice acceptor-label controls for clean Psi.

    Every draw independently assigns the six S3 permutations as evenly as the
    block count allows. Cell values never move.
    """

    lattices = [list(rows) for rows in block_rows]
    if not lattices:
        raise ValueError("at least one phase block is required")
    permutation_family = tuple(permutations((0, 1, 2)))
    assignments = balanced_pseudo_phase_assignments(
        len(lattices), n_controls=n_controls, seed=seed
    )
    results = np.empty((n_controls, len(lattices)), dtype=float)
    for draw in range(n_controls):
        for block_index, assignment in enumerate(assignments[draw]):
            permutation = permutation_family[int(assignment)]
            relabeled = [
                {
                    **row,
                    "acceptor_phase": permutation[int(row["acceptor_phase"])],
                }
                for row in lattices[int(block_index)]
            ]
            results[draw, int(block_index)] = block_psi(
                relabeled, value_key=value_key
            )["Psi"]
    return results


def balanced_pseudo_phase_values(
    block_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    value_key: str = "coupling_ac",
    n_controls: int = MIN_PSEUDO_PHASE_CONTROLS,
    seed: int = 20260810,
) -> np.ndarray:
    """Aggregate values from the balanced whole-block control matrix."""

    return balanced_pseudo_phase_matrix(
        block_rows,
        value_key=value_key,
        n_controls=n_controls,
        seed=seed,
    ).mean(axis=1)


def _gate_flag(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, Mapping):
        return bool(value.get("pass", False))
    return bool(value)


def evaluate_dose_row(
    band: str,
    row: Mapping[str, Any],
    *,
    require_cross_carrier: bool = True,
) -> tuple[bool, dict[str, bool]]:
    """Evaluate one dose without consulting phase labels or target Psi."""

    control_families = row.get("control_families", {})
    required_families = ("phase", "axis")
    control_gates = {
        f"{family}_controls": bool(
            isinstance(control_families.get(family), Mapping)
            and int(control_families[family].get("n_controls", 0))
            >= MIN_STRUCTURED_CONTROLS
            and control_families[family].get("target_exceeds_every_control") is True
            and float(control_families[family].get("add_one_p", np.inf))
            <= 1.0 / 32.0
            and float(
                control_families[family].get("paired_excess_ci", [-np.inf])[0]
            )
            > 0
        )
        for family in required_families
    }
    common = {
        "mechanical": _gate_flag(row, "mechanical_pass"),
        "health": _gate_flag(row, "health_pass"),
        "all_control_health": _gate_flag(row, "all_control_health_pass"),
        "rescue_forward_replay": float(
            row.get("rescue_forward_max_abs_error", np.inf)
        )
        <= 1e-6,
        "rescue_ratio_denominator_stable": _gate_flag(row, "rescue_ratio_health"),
        "rescue_point_at_least_90pct": float(row.get("rescue_recovery", -np.inf))
        >= 0.90,
        "rescue_lower_above_80pct": float(
            row.get("rescue_recovery_ci", [-np.inf])[0]
        )
        > 0.80,
        **control_gates,
    }
    if band == "p3":
        gates = {
            **common,
            "cross_carrier_ratio_denominator_stable": _gate_flag(
                row, "cross_carrier_ratio_health"
            ),
            "effect_ci_above_0_05": float(row.get("capability_ci", [-np.inf])[0])
            > 0.05,
            "p10_retained": float(row.get("cross_carrier_retention", -np.inf))
            >= 0.80,
            "p10_retention_ci_above_0_80": float(
                row.get("cross_carrier_retention_ci", [-np.inf])[0]
            )
            > 0.80,
        }
    elif band == "p10":
        gates = {
            **common,
            "capability_ratio_denominator_stable": _gate_flag(
                row, "capability_ratio_health"
            ),
            "intact_carrier_lower_above_zero": float(
                row.get("intact_carrier_score_ci", [-np.inf])[0]
            )
            > 0,
            "denominator_finite_sign_stable_95pct": float(
                row.get("denominator_finite_sign_stable_fraction", -np.inf)
            )
            >= 0.95,
            "reduction_at_least_50pct": float(
                row.get("capability_fraction_reduced", -np.inf)
            )
            >= 0.50,
            "reduction_ci_above_25pct": float(
                row.get("capability_fraction_reduced_ci", [-np.inf])[0]
            )
            > 0.25,
        }
        if require_cross_carrier:
            gates.update(
                {
                    "joint_health": _gate_flag(row, "joint_health_pass"),
                    "p3_retained": float(
                        row.get("cross_carrier_retention", -np.inf)
                    )
                    >= 0.80,
                    "p3_retention_ci_above_0_80": float(
                        row.get("cross_carrier_retention_ci", [-np.inf])[0]
                    )
                    > 0.80,
                }
            )
    else:
        raise ValueError("band must be p3 or p10")
    return bool(all(gates.values())), gates


def select_largest_passing_dose(
    band: str,
    rows: Mapping[str, Mapping[str, Any]],
    *,
    require_cross_carrier: bool = True,
) -> dict[str, Any]:
    """Choose the largest capable dose on the frozen grid."""

    expected = {f"{dose:.2f}" for dose in DOSE_GRID}
    if set(rows) != expected:
        raise ValueError(f"dose rows must be exactly {sorted(expected)}")
    evaluated: dict[str, dict[str, Any]] = {}
    passing: list[float] = []
    for dose in DOSE_GRID:
        label = f"{dose:.2f}"
        passed, gates = evaluate_dose_row(
            band, rows[label], require_cross_carrier=require_cross_carrier
        )
        evaluated[label] = {"passed": passed, "gates": gates}
        if passed:
            passing.append(dose)
    return {
        "band": band,
        "selected_dose": max(passing) if passing else None,
        "passed": bool(passing),
        "evaluated": evaluated,
        "selection_rule": (
            "largest frozen dose passing individual capability and health plus "
            "the frozen joint cross-carrier gate; no phase labels or Psi"
            if require_cross_carrier
            else (
                "largest frozen dose passing individual capability and health; "
                "no phase labels or Psi"
            )
        ),
    }


def evaluate_preflight(
    measurements: Mapping[str, Any],
    *,
    development_mde: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Apply the asymmetric P3-mandatory/P10-optional authorization rule."""

    shared = measurements.get("shared_gates")
    dose_rows = measurements.get("dose_rows")
    if not isinstance(shared, Mapping) or not isinstance(dose_rows, Mapping):
        raise ValueError("preflight measurements need shared_gates and dose_rows")
    required_shared = (
        "identity_and_reconstruction",
        "native_source_positive",
        "phase_blind_crossed_positive",
        "bridge_negative",
        "ac_at_invariant",
        "protected_base_health",
        "masked_accuracy_health",
        "donor_margin_health",
        "acceptor_margin_health",
        "determinism",
        "construction_audit",
        "synthetic_familywise_fpr",
        "planted_power",
    )
    missing = [key for key in required_shared if key not in shared]
    if missing:
        raise ValueError(f"missing shared gates: {missing}")
    shared_pass = bool(all(_gate_flag(shared, key) for key in required_shared))
    p3 = select_largest_passing_dose("p3", dose_rows.get("p3", {}))
    p10 = select_largest_passing_dose("p10", dose_rows.get("p10", {}))
    mde = dict(development_mde or {})
    def mde_field(name: str, field: str, default: float) -> float:
        row = mde.get(name)
        if isinstance(row, Mapping):
            return float(row.get(field, default))
        return float(row) if field == "mde80" and row is not None else default

    mde_gates = {
        "clean_psi": mde_field("clean_psi", "mde80", np.inf) <= 0.04,
        "N3": mde_field("N3", "mde80", np.inf) <= 0.04,
        "N10": (
            mde_field("N10", "mde80", np.inf) <= 0.05 if p10["passed"] else True
        ),
        "J": mde_field("J", "mde80", np.inf) <= 0.05 if p10["passed"] else True,
    }
    mde_monte_carlo_gates = {
        name: bool(
            mde_field(name, "monte_carlo_se", np.inf) <= 0.0002
            and mde_field(name, "n_bootstrap", -1) == 20_000
            and mde_field(name, "target_blocks", -1) == 600
        )
        for name in ("clean_psi", "N3", "N10", "J")
        if name in mde
    }
    p3_resolution_pass = bool(
        mde_gates["clean_psi"]
        and mde_gates["N3"]
        and mde_monte_carlo_gates.get("clean_psi", False)
        and mde_monte_carlo_gates.get("N3", False)
    )
    p10_resolution_pass = bool(
        mde_gates["N10"]
        and mde_gates["J"]
        and (
            not p10["passed"]
            or (
                mde_monte_carlo_gates.get("N10", False)
                and mde_monte_carlo_gates.get("J", False)
            )
        )
    )
    mandatory_pass = bool(shared_pass and p3["passed"] and p3_resolution_pass)
    p10_authorized = bool(p10["passed"] and p10_resolution_pass)
    if not mandatory_pass:
        status = "P3_INSTRUMENT_UNRESOLVED"
        target_mode = "SEALED"
    elif p10_authorized:
        status = "FULL_FACTORIAL_TARGET_AUTHORIZED"
        target_mode = "FULL_FACTORIAL"
    else:
        status = "P3_ONLY_TARGET_AUTHORIZED"
        target_mode = "P3_ONLY"
    return {
        "status": status,
        "passed": mandatory_pass,
        "target_mode": target_mode,
        "shared_pass": shared_pass,
        "shared_gates": dict(shared),
        "p3": p3,
        "p10": p10,
        "development_mde": mde,
        "development_mde_gates": mde_gates,
        "development_mde_monte_carlo_gates": mde_monte_carlo_gates,
        "p3_resolution_pass": p3_resolution_pass,
        "p10_resolution_pass": p10_resolution_pass,
        "p10_authorized": p10_authorized,
        "p10_status": (
            "P10_TARGET_AUTHORIZED"
            if p10_authorized
            else "OPTIONAL_P10_INSTRUMENT_UNRESOLVED"
        ),
        "p10_failure_is_not_a_null": not p10_authorized,
    }


def make_target_seal(
    preflight: Mapping[str, Any],
    *,
    pinned_files: Mapping[str, Mapping[str, Any]],
    control_seed: int,
    structured_control_hashes: Mapping[str, Mapping[str, Any]],
    preflight_authorization: Mapping[str, Any],
    target_outputs: Mapping[str, Any],
    frozen_revision: str,
) -> dict[str, Any]:
    """Create a deterministic, outcome-free target authorization seal."""

    if not preflight.get("passed") or preflight.get("target_mode") == "SEALED":
        raise RuntimeError("target cannot be sealed from a failed preflight")
    forbidden = {"target_rows", "target_psi", "N3", "N10", "Nboth", "J"}
    if forbidden.intersection(preflight):
        raise RuntimeError("preflight contains target outcomes")
    p3_dose = preflight.get("p3", {}).get("selected_dose")
    p10_dose = preflight.get("p10", {}).get("selected_dose")
    mode = str(preflight["target_mode"])
    if (
        preflight_authorization.get("audit_schema")
        != "gfmcd.pbc.carrier_factorial_preflight_audit.v1"
        or preflight_authorization.get("audit_status") != "PASS"
        or preflight_authorization.get("target_mode") != mode
    ):
        raise RuntimeError("target seal requires a passing independent preflight audit")
    required_hashes = (
        "preflight_sha256",
        "raw_sha256",
        "analysis_sha256",
        "preflight_manifest_sha256",
        "audit_sha256",
        "audit_manifest_sha256",
        "construction_audit_sha256",
        "construction_audit_manifest_sha256",
    )
    if any(
        len(str(preflight_authorization.get(name, ""))) != 64
        for name in required_hashes
    ):
        raise RuntimeError("preflight authorization chain has an invalid hash")
    revision = str(frozen_revision)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("target seal requires a clean, full Git revision")
    shard_count = int(target_outputs.get("shard_count", -1))
    shards = target_outputs.get("shards")
    assembled = target_outputs.get("assembled")
    if (
        shard_count < 1
        or not isinstance(shards, list)
        or len(shards) != shard_count
        or len({str(path) for path in shards}) != shard_count
        or not all(Path(str(path)).is_absolute() for path in shards)
        or not isinstance(assembled, str)
        or not Path(assembled).is_absolute()
    ):
        raise RuntimeError("target output plan must bind unique absolute shard paths")
    payload: dict[str, Any] = {
        "schema": "gfmcd.pbc.carrier_factorial_target_seal.v1",
        "stage": "target",
        "status": "TARGET_AUTHORIZED",
        "target_mode": mode,
        "authorized_split_role": "confirmation",
        "selected_doses": {
            "p3": p3_dose,
            "p10": p10_dose if mode == "FULL_FACTORIAL" else None,
        },
        "bands": {"p3": list(P3_BINS_1026), "p10": list(P10_BINS_1026)},
        "dose_grid": list(DOSE_GRID),
        "sesoi": SESOI,
        "minimum_structured_controls_per_family": MIN_STRUCTURED_CONTROLS,
        "structured_controls_per_family": MIN_STRUCTURED_CONTROLS,
        "structured_control_seed": int(control_seed),
        "structured_control_hashes": dict(structured_control_hashes),
        "preflight_authorization": dict(preflight_authorization),
        "target_outputs": {
            "shard_count": shard_count,
            "shards": [str(path) for path in shards],
            "assembled": str(assembled),
        },
        "frozen_revision": revision,
        "tracked_tree_clean": True,
        "pinned_files": dict(pinned_files),
        "outcome_firewall": {
            "phase_labels_used_for_dose_selection": False,
            "psi_used_for_dose_selection": False,
            "target_outcomes_read": False,
        },
    }
    payload["seal_sha256"] = canonical_json_sha256(payload)
    return payload


def verify_target_seal(seal: Mapping[str, Any]) -> None:
    if seal.get("schema") != "gfmcd.pbc.carrier_factorial_target_seal.v1":
        raise RuntimeError("wrong target-seal schema")
    if seal.get("stage") != "target":
        raise RuntimeError("target seal does not declare the target stage")
    expected = str(seal.get("seal_sha256", ""))
    body = dict(seal)
    body.pop("seal_sha256", None)
    if canonical_json_sha256(body) != expected:
        raise RuntimeError("target-seal hash mismatch")
    firewall = seal.get("outcome_firewall", {})
    if (
        firewall.get("target_outcomes_read") is not False
        or firewall.get("phase_labels_used_for_dose_selection") is not False
        or firewall.get("psi_used_for_dose_selection") is not False
    ):
        raise RuntimeError("target-seal outcome firewall is not intact")
    mode = seal.get("target_mode")
    if seal.get("authorized_split_role") != "confirmation":
        raise RuntimeError("target seal does not authorize the confirmation role")
    authorization = seal.get("preflight_authorization", {})
    required_hashes = (
        "preflight_sha256",
        "raw_sha256",
        "analysis_sha256",
        "preflight_manifest_sha256",
        "audit_sha256",
        "audit_manifest_sha256",
        "construction_audit_sha256",
        "construction_audit_manifest_sha256",
    )
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("audit_schema")
        != "gfmcd.pbc.carrier_factorial_preflight_audit.v1"
        or authorization.get("audit_status") != "PASS"
        or authorization.get("target_mode") != seal.get("target_mode")
        or any(len(str(authorization.get(name, ""))) != 64 for name in required_hashes)
    ):
        raise RuntimeError("target seal lacks an audited preflight authorization chain")
    pinned = seal.get("pinned_files", {})
    if not isinstance(pinned, Mapping):
        raise RuntimeError("target seal lacks pinned authorization artifacts")

    def pinned_path(digest: str) -> Path | None:
        matches = [
            Path(str(record.get("path", ""))).expanduser().resolve()
            for record in pinned.values()
            if isinstance(record, Mapping) and str(record.get("sha256", "")) == digest
        ]
        return (
            matches[0]
            if len(matches) == 1
            and matches[0].is_file()
            and sha256(matches[0].read_bytes()).hexdigest() == digest
            else None
        )

    preflight_path = pinned_path(str(authorization["preflight_sha256"]))
    preflight_manifest_path = pinned_path(
        str(authorization["preflight_manifest_sha256"])
    )
    audit_path = pinned_path(str(authorization["audit_sha256"]))
    audit_manifest_path = pinned_path(str(authorization["audit_manifest_sha256"]))
    construction_audit_path = pinned_path(
        str(authorization["construction_audit_sha256"])
    )
    construction_audit_manifest_path = pinned_path(
        str(authorization["construction_audit_manifest_sha256"])
    )
    if any(
        path is None
        for path in (
            preflight_path,
            preflight_manifest_path,
            audit_path,
            audit_manifest_path,
            construction_audit_path,
            construction_audit_manifest_path,
        )
    ):
        raise RuntimeError("target seal does not pin the complete audit artifact chain")
    try:
        preflight_payload = json.loads(preflight_path.read_text())  # type: ignore[union-attr]
        audit_payload = json.loads(audit_path.read_text())  # type: ignore[union-attr]
        construction_audit_payload = json.loads(
            construction_audit_path.read_text()  # type: ignore[union-attr]
        )
        construction_audit_manifest = json.loads(
            construction_audit_manifest_path.read_text()  # type: ignore[union-attr]
        )
        preflight_manifest = json.loads(
            preflight_manifest_path.read_text()  # type: ignore[union-attr]
        )
        audit_manifest = json.loads(audit_manifest_path.read_text())  # type: ignore[union-attr]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("target seal authorization artifacts are unreadable") from exc

    def manifest_binds(manifest: Mapping[str, Any], digest: str) -> bool:
        outputs = manifest.get("outputs")
        return bool(
            isinstance(outputs, list)
            and any(
                isinstance(record, Mapping) and str(record.get("sha256", "")) == digest
                for record in outputs
            )
        )

    def manifest_binds_input(manifest: Mapping[str, Any], digest: str) -> bool:
        inputs = manifest.get("inputs")
        return bool(
            isinstance(inputs, list)
            and any(
                isinstance(record, Mapping)
                and str(record.get("sha256", "")) == digest
                for record in inputs
            )
        )

    construction_provenance = preflight_payload.get(
        "construction_audit_provenance"
    )
    construction_manifest_config = construction_audit_manifest.get("config")

    if (
        preflight_payload.get("schema")
        != "gfmcd.pbc.carrier_factorial_preflight.v1"
        or preflight_payload.get("target_mode") != seal.get("target_mode")
        or preflight_payload.get("raw_sha256") != authorization.get("raw_sha256")
        or preflight_payload.get("analysis_sha256")
        != authorization.get("analysis_sha256")
        or audit_payload.get("schema")
        != "gfmcd.pbc.carrier_factorial_preflight_audit.v1"
        or audit_payload.get("status") != "PASS"
        or audit_payload.get("target_mode") != seal.get("target_mode")
        or audit_payload.get("preflight_sha256")
        != authorization.get("preflight_sha256")
        or audit_payload.get("raw_sha256") != authorization.get("raw_sha256")
        or audit_payload.get("analysis_sha256")
        != authorization.get("analysis_sha256")
        or not manifest_binds(
            preflight_manifest, str(authorization["preflight_sha256"])
        )
        or not manifest_binds(audit_manifest, str(authorization["audit_sha256"]))
        or not isinstance(construction_provenance, Mapping)
        or construction_provenance.get("experiment_id")
        != "splicebert-positional-boundary-coherence:composition-balanced-cohort-audit-v1"
        or construction_provenance.get("artifact_sha256")
        != authorization.get("construction_audit_sha256")
        or construction_provenance.get("manifest_sha256")
        != authorization.get("construction_audit_manifest_sha256")
        or not manifest_binds_input(
            preflight_manifest,
            str(authorization["construction_audit_sha256"]),
        )
        or not manifest_binds_input(
            preflight_manifest,
            str(authorization["construction_audit_manifest_sha256"]),
        )
        or construction_audit_payload.get("schema")
        != "gfmcd.positional_boundary_coherence.cohort_audit.v1"
        or construction_audit_payload.get("stage") != "construction"
        or construction_audit_payload.get("status")
        != "AUDIT_PASS_TARGET_AUTHORIZED"
        or construction_audit_payload.get("passed") is not True
        or construction_audit_payload.get("target_execution_authorized") is not True
        or construction_audit_manifest.get("experiment_id")
        != "splicebert-positional-boundary-coherence:composition-balanced-cohort-audit-v1"
        or construction_audit_manifest.get("status")
        != "AUDIT_PASS_TARGET_AUTHORIZED"
        or not isinstance(construction_manifest_config, Mapping)
        or construction_manifest_config.get("stage") != "construction"
        or not manifest_binds(
            construction_audit_manifest,
            str(authorization["construction_audit_sha256"]),
        )
    ):
        raise RuntimeError("target seal audit artifact chain is internally inconsistent")
    revision = str(seal.get("frozen_revision", ""))
    if (
        seal.get("tracked_tree_clean") is not True
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError("target seal is not bound to a clean tracked revision")
    outputs = seal.get("target_outputs", {})
    try:
        shard_count = int(outputs.get("shard_count", -1))
        shards = outputs.get("shards")
        assembled = outputs.get("assembled")
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("target seal has an invalid output plan") from exc
    if (
        shard_count < 1
        or not isinstance(shards, list)
        or len(shards) != shard_count
        or len({str(path) for path in shards}) != shard_count
        or not all(Path(str(path)).is_absolute() for path in shards)
        or not isinstance(assembled, str)
        or not Path(assembled).is_absolute()
    ):
        raise RuntimeError("target seal has an invalid output plan")
    doses = seal.get("selected_doses", {})
    _validate_dose(float(doses.get("p3")))
    if mode == "FULL_FACTORIAL":
        _validate_dose(float(doses.get("p10")))
        required_families = {
            "p3_phase",
            "p3_axis",
            "p10_phase",
            "p10_axis",
            "joint_p3",
            "joint_p10",
            "joint",
        }
    elif mode == "P3_ONLY":
        if doses.get("p10") is not None:
            raise RuntimeError("P3-only seal must not carry a P10 dose")
        required_families = {"p3_phase", "p3_axis"}
    else:
        raise RuntimeError("target mode must be P3_ONLY or FULL_FACTORIAL")
    control_hashes = seal.get("structured_control_hashes", {})
    if int(seal.get("structured_controls_per_family", -1)) != MIN_STRUCTURED_CONTROLS:
        raise RuntimeError("target seal changes the frozen structured-control count")
    if not isinstance(control_hashes, Mapping):
        raise RuntimeError("target seal lacks structured-control hashes")
    family_counts = {
        family: sum(
            str(record.get("family")) == family
            and len(str(record.get("sha256", ""))) == 64
            for record in control_hashes.values()
        )
        for family in required_families
    }
    if any(
        family_counts[family] < MIN_STRUCTURED_CONTROLS
        for family in required_families
    ):
        raise RuntimeError("target seal has incomplete structured-control hashes")


def correction_qvalues(pvalues: Mapping[str, float]) -> dict[str, float]:
    """BH-correct the active target family while retaining named mapping."""

    names = tuple(pvalues)
    if not names:
        return {}
    values = [float(pvalues[name]) for name in names]
    adjusted = benjamini_hochberg(values)
    return {name: float(value) for name, value in zip(names, adjusted, strict=True)}


def phase3_authorization(
    *,
    clean_classification: str,
    endpoints: Mapping[str, Mapping[str, Any]],
    specificity: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    """Separate the P3-only route from the optional joint route."""

    clean_material = clean_classification == "MATERIAL_SUPPORTIVE"
    clean_specific = specificity.get("Psi00", {})
    n3 = endpoints.get("N3", {})
    n3_specific = specificity.get("N3", {})
    p3 = bool(
        clean_material
        and clean_specific.get("corrected_pass") is True
        and n3.get("classification") == "MATERIAL_SUPPORTIVE"
        and n3_specific.get("corrected_pass") is True
    )
    n10 = endpoints.get("N10", {})
    interaction = endpoints.get("J", {})
    n10_specific = specificity.get("N10", {})
    j_specific = specificity.get("J", {})
    joint = bool(
        p3
        and n10.get("classification") == "MATERIAL_SUPPORTIVE"
        and str(interaction.get("classification", "")).startswith("MATERIAL_")
        and n10_specific.get("corrected_pass") is True
        and j_specific.get("corrected_pass") is True
    )
    return {
        "authorize_p3_phase3": p3,
        "authorize_joint_phase3": joint,
    }


def carrier_phase2_decision(
    *,
    clean_behavior_classification: str,
    endpoints: Mapping[str, Mapping[str, Any]],
    specificity: Mapping[str, Mapping[str, Any]],
    target_mode: str,
    target_resolution: Mapping[str, Mapping[str, Any]],
    clean_directional_point_estimates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Single frozen Phase-II decision engine used by target and fixture analyses."""

    directional_rows = clean_directional_point_estimates or {}

    def point_estimate(name: str) -> float:
        value = directional_rows.get(name, float("nan"))
        if isinstance(value, Mapping):
            value = value.get("estimate", float("nan"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def lower_bound(name: str) -> float:
        row = endpoints.get(name, {})
        interval = row.get("ci") if isinstance(row, Mapping) else None
        if not isinstance(interval, Sequence) or len(interval) != 2:
            return float("nan")
        try:
            return float(interval[0])
        except (TypeError, ValueError):
            return float("nan")

    def resolution_at_most(name: str, threshold: float) -> bool:
        row = target_resolution.get(name, {})
        try:
            mde = float(row.get("mde80", math.inf))
        except (AttributeError, TypeError, ValueError):
            return False
        return bool(
            row.get("pass") is True and math.isfinite(mde) and mde <= threshold
        )

    directional_scoped = bool(
        lower_bound("Psi00") > 0
        and lower_bound("N3") > 0
        and all(point_estimate(name) > 0 for name in CLEAN_DIRECTION_ENDPOINTS)
        and specificity.get("Psi00", {}).get("corrected_pass") is True
        and specificity.get("N3", {}).get("corrected_pass") is True
        and resolution_at_most("Psi00", 0.04)
        and resolution_at_most("N3", 0.04)
    )

    p3 = bool(
        clean_behavior_classification == "BALANCED_PREFERENCE_MATERIAL"
        and endpoints.get("N3", {}).get("classification") == "MATERIAL_SUPPORTIVE"
        and specificity.get("N3", {}).get("corrected_pass") is True
        and target_resolution.get("N3", {}).get("pass") is True
    )
    joint = bool(
        target_mode == "FULL_FACTORIAL"
        and p3
        and endpoints.get("N10", {}).get("classification") == "MATERIAL_SUPPORTIVE"
        and str(endpoints.get("J", {}).get("classification", "")).startswith(
            "MATERIAL_"
        )
        and specificity.get("N10", {}).get("corrected_pass") is True
        and specificity.get("J", {}).get("corrected_pass") is True
        and target_resolution.get("N10", {}).get("pass") is True
        and target_resolution.get("J", {}).get("pass") is True
    )
    if p3:
        p3_state = "P3_NECESSITY_SUPPORTED"
    elif directional_scoped:
        p3_state = "P3_DIRECTIONAL_SCOPED_SUPPORT"
    elif (
        endpoints.get("N3", {}).get("classification") == "BOUNDED_NULL"
        and target_resolution.get("N3", {}).get("pass") is True
    ):
        p3_state = "P3_ATTENUATION_BOUNDED_NULL"
    elif endpoints.get("N3", {}).get("classification") == "MATERIAL_SUPPRESSIVE":
        p3_state = "P3_SUPPRESSIVE"
    else:
        p3_state = "P3_INCONCLUSIVE"

    if p3:
        if target_mode == "P3_ONLY":
            verdict = "P3_ONLY_CLOSURE_PATH"
        elif joint:
            verdict = "DUAL_SCALE_INTERACTION_SUPPORTED"
        elif endpoints.get("N10", {}).get("classification") == "BOUNDED_NULL":
            verdict = "P3_ONLY_CLOSURE_PATH"
        else:
            verdict = "P3_NECESSITY_SUPPORTED"
    elif directional_scoped:
        verdict = "P3_DIRECTIONAL_SCOPED_SUPPORT"
    elif clean_behavior_classification == "BALANCED_PREFERENCE_BOUNDED_NULL":
        verdict = "BALANCED_PREFERENCE_BOUNDED_NULL"
    elif clean_behavior_classification != "BALANCED_PREFERENCE_MATERIAL":
        verdict = "BALANCED_PREFERENCE_INCONCLUSIVE"
    else:
        verdict = p3_state if p3_state != "P3_INCONCLUSIVE" else "INCONCLUSIVE"
    support_tiers = []
    if directional_scoped:
        support_tiers.append("P3_DIRECTIONAL_SCOPED_SUPPORT")
    if p3:
        support_tiers.append("P3_NECESSITY_SUPPORTED")
    return {
        "verdict": verdict,
        "directional_scoped_support": directional_scoped,
        "support_tiers": support_tiers,
        "p3_phase3_statistical_criteria": p3,
        "joint_phase3_statistical_criteria": joint,
        "authorize_p3_phase3": p3,
        "authorize_joint_phase3": joint,
    }


__all__ = [
    "COMMUTATIVITY_TOLERANCE",
    "CLEAN_DIRECTION_ENDPOINTS",
    "DOSE_GRID",
    "MIN_PSEUDO_PHASE_CONTROLS",
    "MIN_STRUCTURED_CONTROLS",
    "STRUCTURED_CONTROL_BASE_SEED",
    "P3_CONTROL_SEED",
    "P10_CONTROL_SEED",
    "JOINT_CONTROL_SEED",
    "OFF_BAND_TOLERANCE",
    "P10_BINS_1026",
    "P10_READERS_1026",
    "P10_PERIOD_1026",
    "P3_BINS_1026",
    "SESOI",
    "attenuate_band",
    "balanced_pseudo_phase_assignment_audit",
    "balanced_pseudo_phase_assignments",
    "balanced_pseudo_phase_values",
    "balanced_pseudo_phase_matrix",
    "bounded_attention_carrier_score",
    "block_psi",
    "build_phase_control_family",
    "build_world_tables",
    "c43_margin",
    "canonical_json_sha256",
    "carrier_phase2_decision",
    "classify_effect_interval",
    "classify_interaction_interval",
    "correction_qvalues",
    "coupling_from_scores",
    "evaluate_dose_row",
    "evaluate_preflight",
    "exact_add_one_upper_p",
    "exact_add_one_absolute_p",
    "factorial_contrasts",
    "hidden_axis_control_tables",
    "intervention_audit",
    "make_target_seal",
    "mechanical_audit",
    "mde80_from_bootstrap",
    "percentile_stratified_interval",
    "phase3_authorization",
    "phase_control_tables",
    "projected_mde80",
    "reconstruct_attenuated_band",
    "replay_original_table",
    "select_largest_passing_dose",
    "simultaneous_stratified_intervals",
    "verify_target_seal",
]
