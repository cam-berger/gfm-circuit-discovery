"""Period-3 register-relocation estimators for positional boundary coherence.

This module contains only the model-independent mechanics and statistics for the
conditional Phase-III experiment.  Attention or spectral movement is not a behavioral
result: the primary input is the native masked-donor coupling value for every member of a
complete donor-phase x acceptor-phase x imposed-register lattice.

The experiment is deliberately asymmetric.  An intact-P10 lattice can close the period-3
question on its own.  A second P10 state and ``Jreg`` are admitted only for a separately
authorized modifier arm; absence or failure of that arm cannot invalidate a capable
period-3 result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from gfmcd.splicing.positional_spectrum import band_component, validate_table


PHASES = (0, 1, 2)
REGISTERS = (-1, 0, 1)
P10_INTACT = "intact"
P10_ATTENUATED = "attenuated"
TARGET_COMPONENT = "p3"
PARTIAL_SPECIFICITY_SEED = 2026081010
OPPOSITE_PARTIAL_SPECIFICITY_SEED = 2026081012


@dataclass(frozen=True)
class SimultaneousIntervals:
    """Studentized simultaneous intervals from whole-block stratified resampling."""

    names: tuple[str, ...]
    means: tuple[float, ...]
    standard_errors: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    adjusted_p: tuple[float, ...]
    critical_value: float
    n_blocks: int
    n_bootstrap: int

    def as_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "means": dict(zip(self.names, self.means)),
            "standard_errors": dict(zip(self.names, self.standard_errors)),
            "lower": dict(zip(self.names, self.lower)),
            "upper": dict(zip(self.names, self.upper)),
            "adjusted_p": dict(zip(self.names, self.adjusted_p)),
            "critical_value": self.critical_value,
            "n_blocks": self.n_blocks,
            "n_bootstrap": self.n_bootstrap,
        }


def phase_delta(donor_phase: int, acceptor_phase: int) -> int:
    """Return the acceptor-minus-donor annotated phase difference modulo three."""

    donor = int(donor_phase)
    acceptor = int(acceptor_phase)
    if donor not in PHASES or acceptor not in PHASES:
        raise ValueError("donor and acceptor phases must be 0, 1, or 2")
    return (acceptor - donor) % 3


def preferred_delta(register: int) -> int:
    """The biological phase difference predicted by an imposed carrier register."""

    value = int(register)
    if value not in REGISTERS:
        raise ValueError("register must be -1, 0, or +1")
    return (-value) % 3


def opposite_preferred_delta(register: int) -> int:
    """Return the prespecified opposite-handed phase relation for falsification."""

    value = int(register)
    if value not in REGISTERS:
        raise ValueError("register must be -1, 0, or +1")
    return value % 3


def register_endpoint_name(register: int, state: str) -> str:
    """Stable JSON key for one signed-register contrast."""

    value = int(register)
    if value not in REGISTERS:
        raise ValueError("register must be -1, 0, or +1")
    return f"s_{value}_{state}"


def relocate_band_within_span(
    table: np.ndarray,
    bins: Sequence[int],
    *,
    start: int,
    end: int,
    register: int,
    direction: str = "right",
) -> np.ndarray:
    """Apply a relative phase ramp to one carrier inside a half-open row span.

    ``register=0`` is an exact replay.  At the first row the carrier phase is unchanged;
    at the final row its positive-frequency analytic representation is rotated by
    ``-2*pi*register/3``, with a linear phase ramp between those endpoints.  One common
    phase multiplies the complete frozen multi-bin analytic component.  Direct Fourier
    evaluation, rather than row interpolation, keeps that definition exact.

    Every other carrier component, the row mean, and every row outside the span are
    retained algebraically.  The clipped edit itself is not claimed to be globally
    band-limited.
    """

    values = validate_table(table)
    amount = int(register)
    if amount not in REGISTERS:
        raise ValueError("register must be -1, 0, or +1")
    if direction not in {"left", "right"}:
        raise ValueError("direction must be left or right")
    if not (0 <= int(start) < int(end) <= values.shape[0]):
        raise ValueError("relocation span lies outside the position table")
    if amount == 0:
        return values.copy()
    target = np.arange(int(start), int(end))
    if len(target) < 2:
        raise ValueError("relocation span needs at least two rows")
    sign = 1.0 if direction == "left" else -1.0
    phases = sign * (2.0 * np.pi * amount / 3.0) * np.linspace(0.0, 1.0, len(target))
    component = band_component(values, bins)
    analytic = _analytic_band_rows(component, bins, target)
    shifted = np.real(analytic * np.exp(1j * phases[:, None]))
    edited = values.copy()
    edited[target] += shifted - component[target]
    return edited


def _analytic_band_rows(
    component: np.ndarray,
    bins: Sequence[int],
    coordinates: np.ndarray,
) -> np.ndarray:
    """Evaluate the positive-frequency analytic selected-band component."""

    values = validate_table(component)
    selected = tuple(sorted(set(int(value) for value in bins)))
    if not selected or selected[0] <= 0 or selected[-1] >= values.shape[0] / 2:
        raise ValueError("analytic carrier evaluation requires non-DC, non-Nyquist bins")
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 1 or not np.isfinite(points).all():
        raise ValueError("carrier coordinates must be a finite vector")
    spectrum = np.fft.rfft(values, axis=0)[list(selected)]
    angles = 2 * np.pi * points[:, None] * np.asarray(selected)[None, :] / values.shape[0]
    phasors = np.exp(1j * angles)
    return (2.0 / values.shape[0]) * (phasors @ spectrum)


def relocation_mechanics(
    table: np.ndarray,
    bins: Sequence[int],
    *,
    start: int,
    end: int,
    direction: str = "right",
    atol: float = 1e-10,
) -> dict[str, object]:
    """Audit exact replay, locality, residual retention, and nonzero register dose.

    These are construction invariants, not empirical capability gates.  A behavioral
    positive control must still show that the intervention moves an established P3 use.
    """

    values = validate_table(table)
    component = band_component(values, bins)
    rows: dict[str, dict[str, object]] = {}
    for register in REGISTERS:
        edited = relocate_band_within_span(
            values,
            bins,
            start=start,
            end=end,
            register=register,
            direction=direction,
        )
        target = np.arange(start, end)
        sign = 1.0 if direction == "left" else -1.0
        phases = (
            sign
            * (2.0 * np.pi * register / 3.0)
            * np.linspace(0.0, 1.0, len(target))
        )
        analytic = _analytic_band_rows(component, bins, target)
        shifted_component = np.real(analytic * np.exp(1j * phases[:, None]))
        changed = edited - values
        outside = np.r_[0:start, end : values.shape[0]]
        expected_residual = values[target] - component[target]
        observed_residual = edited[target] - shifted_component
        rows[str(register)] = {
            "register": register,
            "changed_rms": float(np.sqrt(np.mean(np.square(changed[target])))),
            "outside_max_abs": (
                0.0 if outside.size == 0 else float(np.max(np.abs(changed[outside])))
            ),
            "residual_max_abs": float(
                np.max(np.abs(observed_residual - expected_residual))
            ),
            "donor_endpoint_delta_max_abs": float(np.max(np.abs(changed[start]))),
            "acceptor_endpoint_phase_radians": float(phases[-1]),
            "expected_acceptor_endpoint_phase_radians": float(
                sign * 2.0 * np.pi * register / 3.0
            ),
            "acceptor_endpoint_phase_error_radians": float(
                abs(phases[-1] - sign * 2.0 * np.pi * register / 3.0)
            ),
            "phase_profile_monotone": bool(
                np.all(np.diff(phases) >= 0)
                if phases[-1] >= phases[0]
                else np.all(np.diff(phases) <= 0)
            ),
            "predicted_annotated_phase_delta": preferred_delta(register),
            "exact_replay": bool(register != 0 or np.array_equal(edited, values)),
        }
    passed = (
        bool(rows["0"]["exact_replay"])
        and all(float(row["outside_max_abs"]) <= atol for row in rows.values())
        and all(float(row["residual_max_abs"]) <= atol for row in rows.values())
        and all(float(row["donor_endpoint_delta_max_abs"]) <= atol for row in rows.values())
        and all(
            float(row["acceptor_endpoint_phase_error_radians"]) <= 1e-6
            and bool(row["phase_profile_monotone"])
            for row in rows.values()
        )
        and float(rows["-1"]["changed_rms"]) > atol
        and float(rows["1"]["changed_rms"]) > atol
    )
    return {
        "kind": "construction_invariants",
        "rows": rows,
        "tolerance": float(atol),
        "passed": bool(passed),
    }


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def block_register_contrasts(
    rows: Sequence[Mapping[str, object]],
    *,
    component: str = TARGET_COMPONENT,
    p10_states: Sequence[str] = (P10_INTACT,),
) -> tuple[dict[str, object], ...]:
    """Reduce complete cell lattices to block-level ``S_r``, ``Omega3``, and ``Jreg``.

    Required row keys are ``block_id``, ``chrom``, ``strand``, ``component``,
    ``p10_state``, ``register``, ``donor_phase``, ``acceptor_phase``, and ``coupling``.
    Repeated rows in one cell are averaged only after their phase identity is checked.
    Every donor/acceptor cell must be present for every requested register and P10 state.
    """

    states = tuple(str(value) for value in p10_states)
    if not states or len(set(states)) != len(states):
        raise ValueError("p10_states must be a nonempty unique sequence")
    if any(state not in {P10_INTACT, P10_ATTENUATED} for state in states):
        raise ValueError("unknown P10 state")
    selected = [row for row in rows if str(row.get("component")) == component]
    if not selected:
        raise ValueError(f"no rows for component {component!r}")

    by_block: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        block_id = str(row.get("block_id") or "")
        if not block_id:
            raise ValueError("every row needs a block_id")
        by_block[block_id].append(row)

    output: list[dict[str, object]] = []
    expected_cells = {
        (state, register, donor, acceptor)
        for state in states
        for register in REGISTERS
        for donor in PHASES
        for acceptor in PHASES
    }
    for block_id in sorted(by_block):
        local = by_block[block_id]
        chroms = {str(row.get("chrom") or "") for row in local}
        strands = {str(row.get("strand") or "") for row in local}
        if len(chroms) != 1 or "" in chroms:
            raise ValueError(f"block {block_id} does not have one chromosome")
        if len(strands) != 1 or next(iter(strands)) not in {"+", "-"}:
            raise ValueError(f"block {block_id} does not have one valid strand")
        cell_values: dict[tuple[str, int, int, int], list[float]] = defaultdict(list)
        for row in local:
            state = str(row.get("p10_state"))
            register = int(row.get("register", -1))
            donor = int(row.get("donor_phase", -1))
            acceptor = int(row.get("acceptor_phase", -1))
            key = (state, register, donor, acceptor)
            if key not in expected_cells:
                raise ValueError(f"unexpected cell in block {block_id}: {key}")
            declared_delta = row.get("phase_delta")
            if declared_delta is not None and int(declared_delta) != phase_delta(donor, acceptor):
                raise ValueError(f"phase_delta disagrees with phases in block {block_id}")
            cell_values[key].append(_finite_float(row.get("coupling"), "coupling"))
        if set(cell_values) != expected_cells:
            missing = sorted(expected_cells - set(cell_values))
            extra = sorted(set(cell_values) - expected_cells)
            raise ValueError(
                f"block {block_id} has an incomplete register lattice; "
                f"missing={missing[:3]} extra={extra[:3]}"
            )

        result: dict[str, object] = {
            "block_id": block_id,
            "chrom": next(iter(chroms)),
            "strand": next(iter(strands)),
            "component": component,
        }
        for state in states:
            s_values = []
            opposite_values = []
            for register in REGISTERS:
                delta_values: dict[int, list[float]] = {delta: [] for delta in PHASES}
                for donor in PHASES:
                    for acceptor in PHASES:
                        delta_values[phase_delta(donor, acceptor)].append(
                            float(np.mean(cell_values[(state, register, donor, acceptor)]))
                        )
                delta_means = {
                    delta: float(np.mean(values)) for delta, values in delta_values.items()
                }
                preferred = preferred_delta(register)
                other = [delta_means[delta] for delta in PHASES if delta != preferred]
                score = delta_means[preferred] - float(np.mean(other))
                result[register_endpoint_name(register, state)] = float(score)
                s_values.append(score)
                opposite = opposite_preferred_delta(register)
                opposite_other = [
                    delta_means[delta] for delta in PHASES if delta != opposite
                ]
                opposite_score = delta_means[opposite] - float(
                    np.mean(opposite_other)
                )
                result[f"opposite_s_{register}_{state}"] = float(opposite_score)
                opposite_values.append(opposite_score)
            result[f"omega3_{state}"] = float(np.mean(s_values))
            result[f"opposite_omega3_{state}"] = float(
                np.mean(opposite_values)
            )
        if P10_INTACT in states and P10_ATTENUATED in states:
            result["jreg"] = (
                float(result[f"omega3_{P10_INTACT}"])
                - float(result[f"omega3_{P10_ATTENUATED}"])
            )
        output.append(result)
    return tuple(output)


def _stratified_standard_error(
    values: np.ndarray,
    strata: np.ndarray,
    *,
    allow_zero: bool = False,
) -> np.ndarray:
    n = values.shape[0]
    variance = np.zeros(values.shape[1], dtype=float)
    for label in sorted(set(strata.tolist())):
        local = values[strata == label]
        if len(local) < 2:
            raise ValueError(f"sampling stratum {label!r} needs at least two blocks")
        weight = len(local) / n
        variance += weight * weight * local.var(axis=0, ddof=1) / len(local)
    result = np.sqrt(variance)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("every endpoint needs finite nonnegative stratified variance")
    if not allow_zero and np.any(result == 0):
        raise ValueError("every endpoint needs nonzero finite stratified variance")
    return result


def stratified_studentized_max_t(
    matrix: np.ndarray,
    *,
    names: Sequence[str],
    strata: Sequence[str],
    seed: int,
    n_bootstrap: int = 20_000,
    alpha: float = 0.05,
) -> SimultaneousIntervals:
    """Whole-block, within-stratum max-|t| inference with exact add-one p-values."""

    values = np.asarray(matrix, dtype=float)
    labels = np.asarray([str(value) for value in strata], dtype=object)
    endpoint_names = tuple(str(name) for name in names)
    if values.ndim != 2 or values.shape[1] != len(endpoint_names):
        raise ValueError("matrix columns must match endpoint names")
    if len(values) != len(labels) or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("inference needs at least three finite block rows and matching strata")
    if n_bootstrap < 99 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap count or alpha")

    means = values.mean(axis=0)
    standard_errors = _stratified_standard_error(values, labels, allow_zero=True)
    variable = standard_errors > 0
    by_stratum = [np.flatnonzero(labels == label) for label in sorted(set(labels.tolist()))]
    rng = np.random.default_rng(int(seed))
    max_t = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        draw = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in by_stratum]
        )
        sample = values[draw]
        sample_labels = labels[draw]
        try:
            sample_se = _stratified_standard_error(
                sample, sample_labels, allow_zero=True
            )
        except ValueError:
            max_t[index] = np.nan
            continue
        active = variable & (sample_se > 0)
        max_t[index] = (
            float(
                np.max(
                    np.abs(
                        (sample.mean(axis=0)[active] - means[active])
                        / sample_se[active]
                    )
                )
            )
            if np.any(active)
            else 0.0
        )
    finite = max_t[np.isfinite(max_t)]
    if len(finite) < int(0.95 * n_bootstrap):
        raise RuntimeError("too many degenerate stratified bootstrap draws")
    critical = float(np.quantile(finite, 1 - alpha, method="higher"))
    lower = means - critical * standard_errors
    upper = means + critical * standard_errors
    observed = np.empty_like(means)
    observed[variable] = np.abs(means[variable] / standard_errors[variable])
    observed[~variable] = np.where(np.abs(means[~variable]) > 0, np.inf, 0.0)
    adjusted = np.asarray(
        [(1 + int(np.sum(finite >= value))) / (len(finite) + 1) for value in observed]
    )
    return SimultaneousIntervals(
        names=endpoint_names,
        means=tuple(float(value) for value in means),
        standard_errors=tuple(float(value) for value in standard_errors),
        lower=tuple(float(value) for value in lower),
        upper=tuple(float(value) for value in upper),
        adjusted_p=tuple(float(value) for value in adjusted),
        critical_value=critical,
        n_blocks=len(values),
        n_bootstrap=int(n_bootstrap),
    )


def analyze_register_component(
    rows: Sequence[Mapping[str, object]],
    *,
    component: str = TARGET_COMPONENT,
    joint: bool = False,
    seed: int,
    n_bootstrap: int = 20_000,
) -> dict[str, object]:
    """Analyze one intervention direction without mixing it with control families."""

    states = (P10_INTACT, P10_ATTENUATED) if joint else (P10_INTACT,)
    blocks = block_register_contrasts(rows, component=component, p10_states=states)
    primary_names = (f"omega3_{P10_INTACT}",) + tuple(
        register_endpoint_name(register, P10_INTACT) for register in REGISTERS
    )
    matrix = np.asarray(
        [[float(row[name]) for name in primary_names] for row in blocks]
    )
    strata = [f"{row['chrom']}:{row['strand']}" for row in blocks]
    primary_inference = stratified_studentized_max_t(
        matrix,
        names=primary_names,
        strata=strata,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    lower = dict(zip(primary_inference.names, primary_inference.lower))
    upper = dict(zip(primary_inference.names, primary_inference.upper))
    means = dict(zip(primary_inference.names, primary_inference.means))
    required_intact = [
        register_endpoint_name(register, P10_INTACT) for register in REGISTERS
    ]
    required_joint = [
        register_endpoint_name(register, state)
        for state in states
        for register in REGISTERS
    ]
    omega = {
        state: float(np.mean([row[f"omega3_{state}"] for row in blocks]))
        for state in states
    }
    result: dict[str, object] = {
        "component": component,
        "joint": bool(joint),
        "n_blocks": len(blocks),
        "states": list(states),
        "inference": primary_inference.as_dict(),
        "omega3": omega,
        "omega3_material_positive": bool(
            lower[f"omega3_{P10_INTACT}"] > 0.05
        ),
        # The period-3 verdict is always the intact-P10 lattice.  A failed optional
        # modifier state therefore cannot veto period-3 closure.
        "all_registers_positive": bool(
            all(lower[name] > 0 for name in required_intact)
        ),
        # A rough boundary link is deliberately weaker than cyclic relocation:
        # all three prespecified register point estimates must agree in sign, while
        # only two need simultaneous lower bounds above zero.  Aggregate materiality
        # remains a separate, unchanged >0.05 requirement in the adjudicator.
        "positive_register_count": int(
            sum(lower[name] > 0 for name in required_intact)
        ),
        "all_register_point_estimates_positive": bool(
            all(means[name] > 0 for name in required_intact)
        ),
        "all_joint_state_registers_positive": bool(
            all(lower.get(name, -math.inf) > 0 for name in required_joint)
        ),
        "all_registers_bounded_null": bool(
            all(
                lower[name] >= -0.05 and upper[name] <= 0.05
                for name in primary_names
            )
        ),
        "per_block": list(blocks),
    }
    opposite_names = (f"opposite_omega3_{P10_INTACT}",) + tuple(
        f"opposite_s_{register}_{P10_INTACT}" for register in REGISTERS
    )
    opposite_inference = stratified_studentized_max_t(
        np.asarray(
            [[float(row[name]) for name in opposite_names] for row in blocks]
        ),
        names=opposite_names,
        strata=strata,
        seed=seed + 3,
        n_bootstrap=n_bootstrap,
    )
    opposite_lower = dict(
        zip(opposite_inference.names, opposite_inference.lower)
    )
    opposite_means = dict(
        zip(opposite_inference.names, opposite_inference.means)
    )
    opposite_required = [
        f"opposite_s_{register}_{P10_INTACT}" for register in REGISTERS
    ]
    result["opposite_handed_inference"] = opposite_inference.as_dict()
    result["opposite_omega3_material_positive"] = bool(
        opposite_lower[f"opposite_omega3_{P10_INTACT}"] > 0.05
    )
    result["opposite_all_registers_positive"] = bool(
        all(
            opposite_lower[name] > 0 for name in opposite_required
        )
    )
    result["opposite_positive_register_count"] = int(
        sum(opposite_lower[name] > 0 for name in opposite_required)
    )
    result["opposite_all_register_point_estimates_positive"] = bool(
        all(opposite_means[name] > 0 for name in opposite_required)
    )
    expected_partial = bool(
        result["omega3_material_positive"]
        and result["all_register_point_estimates_positive"]
        and int(result["positive_register_count"]) >= 2
    )
    opposite_partial = bool(
        result["opposite_omega3_material_positive"]
        and result["opposite_all_register_point_estimates_positive"]
        and int(result["opposite_positive_register_count"]) >= 2
    )
    result["expected_partial_concordance_pattern"] = expected_partial
    result["opposite_partial_concordance_pattern"] = opposite_partial
    if joint:
        attenuated_names = (f"omega3_{P10_ATTENUATED}",) + tuple(
            register_endpoint_name(register, P10_ATTENUATED)
            for register in REGISTERS
        )
        attenuated = stratified_studentized_max_t(
            np.asarray(
                [[float(row[name]) for name in attenuated_names] for row in blocks]
            ),
            names=attenuated_names,
            strata=strata,
            seed=seed + 1,
            n_bootstrap=n_bootstrap,
        )
        optional = stratified_studentized_max_t(
            np.asarray([[float(row["jreg"])] for row in blocks]),
            names=("jreg",),
            strata=strata,
            seed=seed + 2,
            n_bootstrap=n_bootstrap,
        )
        attenuated_lower = dict(zip(attenuated.names, attenuated.lower))
        result["attenuated_register_inference"] = attenuated.as_dict()
        result["optional_modifier_inference"] = optional.as_dict()
        result["all_joint_state_registers_positive"] = bool(
            all(
                attenuated_lower[
                    register_endpoint_name(register, P10_ATTENUATED)
                ]
                > 0
                for register in REGISTERS
            )
        )
    return result


def exact_add_one_upper_p(target: float, controls: Sequence[float]) -> float:
    """One-sided add-one rank p-value for a complete structured-null family."""

    values = [_finite_float(value, "control") for value in controls]
    target_value = _finite_float(target, "target")
    if not values:
        raise ValueError("at least one control is required")
    return float((1 + sum(value >= target_value for value in values)) / (len(values) + 1))


def register_specificity(
    analyses: Mapping[str, Mapping[str, object]],
    *,
    target_component: str = TARGET_COMPONENT,
    minimum_controls: int = 31,
) -> dict[str, object]:
    """Compare P3 only with its same-size, same-family structured controls."""

    if target_component not in analyses:
        raise ValueError("target component analysis is absent")
    control_labels = sorted(label for label in analyses if label != target_component)
    if len(control_labels) < minimum_controls:
        raise ValueError(
            f"specificity needs at least {minimum_controls} controls; got {len(control_labels)}"
        )
    target = float(analyses[target_component]["omega3"][P10_INTACT])
    controls = [float(analyses[label]["omega3"][P10_INTACT]) for label in control_labels]
    p_value = exact_add_one_upper_p(target, controls)
    return {
        "target_component": target_component,
        "target_omega3": target,
        "control_omega3": dict(zip(control_labels, controls)),
        "n_controls": len(controls),
        "target_exceeds_every_control": bool(target > max(controls)),
        "exact_add_one_p_one_sided": p_value,
        "passed": bool(target > max(controls) and p_value <= 0.05),
    }


def paired_register_specificity_interval(
    component_blocks: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    target_component: str = TARGET_COMPONENT,
    seed: int,
    n_bootstrap: int = 20_000,
    endpoint: str = f"omega3_{P10_INTACT}",
) -> dict[str, object]:
    """Interval for target ``Omega3`` minus the mean same-family control per block."""

    if target_component not in component_blocks:
        raise ValueError("target block table is absent")
    controls = sorted(label for label in component_blocks if label != target_component)
    if not controls:
        raise ValueError("at least one control block table is required")
    indexed = {
        label: {str(row["block_id"]): row for row in rows}
        for label, rows in component_blocks.items()
    }
    target_ids = sorted(indexed[target_component])
    if any(sorted(indexed[label]) != target_ids for label in controls):
        raise ValueError("target and controls must contain identical blocks")
    differences = []
    strata = []
    for block_id in target_ids:
        target_row = indexed[target_component][block_id]
        target = float(target_row[endpoint])
        control = float(
            np.mean(
                [indexed[label][block_id][endpoint] for label in controls]
            )
        )
        differences.append(target - control)
        strata.append(f"{target_row['chrom']}:{target_row['strand']}")
    inference = stratified_studentized_max_t(
        np.asarray(differences, dtype=float)[:, None],
        names=("p3_minus_mean_control",),
        strata=strata,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    result = inference.as_dict()
    return {
        "endpoint": endpoint,
        "inference": result,
        "passed": bool(result["lower"]["p3_minus_mean_control"] > 0),
    }


def paired_named_control_specificity(
    component_blocks: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    control_components: Sequence[str],
    target_component: str = TARGET_COMPONENT,
    seed: int,
    n_bootstrap: int = 20_000,
) -> dict[str, object]:
    """Simultaneous target-minus-control ``Omega3`` intervals for named controls."""

    controls = tuple(str(value) for value in control_components)
    if not controls or len(set(controls)) != len(controls):
        raise ValueError("named controls must be a nonempty unique sequence")
    required = {target_component, *controls}
    if not required.issubset(component_blocks):
        raise ValueError("target or named-control block table is absent")
    indexed = {
        label: {str(row["block_id"]): row for row in component_blocks[label]}
        for label in required
    }
    target_ids = sorted(indexed[target_component])
    if any(sorted(indexed[label]) != target_ids for label in controls):
        raise ValueError("target and named controls must contain identical blocks")
    names = tuple(f"p3_minus_{label}" for label in controls)
    matrix = []
    strata = []
    for block_id in target_ids:
        target_row = indexed[target_component][block_id]
        target = float(target_row[f"omega3_{P10_INTACT}"])
        matrix.append(
            [
                target - float(indexed[label][block_id][f"omega3_{P10_INTACT}"])
                for label in controls
            ]
        )
        strata.append(f"{target_row['chrom']}:{target_row['strand']}")
    inference = stratified_studentized_max_t(
        np.asarray(matrix, dtype=float),
        names=names,
        strata=strata,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    result = inference.as_dict()
    return {
        "controls": list(controls),
        "inference": result,
        "passed": bool(all(result["lower"][name] > 0 for name in names)),
    }


def partial_handedness_specificity_family(
    component_blocks: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    control_components: Sequence[str],
    target_component: str = TARGET_COMPONENT,
    seed: int,
    n_bootstrap: int = 20_000,
) -> dict[str, object]:
    """Joint max-T family for partial handedness and named-control specificity."""

    controls = tuple(str(value) for value in control_components)
    if not controls or len(set(controls)) != len(controls):
        raise ValueError("partial controls must be a nonempty unique sequence")
    required = {target_component, *controls}
    if not required.issubset(component_blocks):
        raise ValueError("partial target or named-control block table is absent")
    indexed = {
        label: {str(row["block_id"]): row for row in component_blocks[label]}
        for label in required
    }
    target_ids = sorted(indexed[target_component])
    if any(sorted(indexed[label]) != target_ids for label in controls):
        raise ValueError("partial target and named controls must contain identical blocks")
    handedness_name = "omega3_expected_minus_opposite"
    control_names = tuple(f"p3_minus_{label}" for label in controls)
    names = (handedness_name, *control_names)
    matrix = []
    strata = []
    for block_id in target_ids:
        target_row = indexed[target_component][block_id]
        target_omega = float(target_row[f"omega3_{P10_INTACT}"])
        matrix.append(
            [
                target_omega
                - float(target_row[f"opposite_omega3_{P10_INTACT}"]),
                *[
                    target_omega
                    - float(indexed[label][block_id][f"omega3_{P10_INTACT}"])
                    for label in controls
                ],
            ]
        )
        strata.append(f"{target_row['chrom']}:{target_row['strand']}")
    inference = stratified_studentized_max_t(
        np.asarray(matrix, dtype=float),
        names=names,
        strata=strata,
        seed=seed,
        n_bootstrap=n_bootstrap,
    ).as_dict()
    lower = inference["lower"]
    upper = inference["upper"]
    return {
        "controls": list(controls),
        "members": list(names),
        "seed": int(seed),
        "inference": inference,
        "handedness_lower_positive": bool(lower[handedness_name] > 0),
        "reverse_handedness_upper_negative": bool(upper[handedness_name] < 0),
        "named_controls_lower_positive": bool(
            all(lower[name] > 0 for name in control_names)
        ),
        "passed": bool(all(lower[name] > 0 for name in names)),
    }


def opposite_partial_handedness_specificity_family(
    component_blocks: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    control_components: Sequence[str],
    target_component: str = TARGET_COMPONENT,
    seed: int,
    n_bootstrap: int = 20_000,
) -> dict[str, object]:
    """Joint max-T family for the opposite-only partial falsification path."""

    controls = tuple(str(value) for value in control_components)
    if not controls or len(set(controls)) != len(controls):
        raise ValueError("opposite partial controls must be a nonempty unique sequence")
    required = {target_component, *controls}
    if not required.issubset(component_blocks):
        raise ValueError("opposite partial target or named-control block table is absent")
    indexed = {
        label: {str(row["block_id"]): row for row in component_blocks[label]}
        for label in required
    }
    target_ids = sorted(indexed[target_component])
    if any(sorted(indexed[label]) != target_ids for label in controls):
        raise ValueError(
            "opposite partial target and named controls must contain identical blocks"
        )
    handedness_name = "omega3_opposite_minus_expected"
    control_names = tuple(f"opposite_p3_minus_{label}" for label in controls)
    names = (handedness_name, *control_names)
    matrix = []
    strata = []
    for block_id in target_ids:
        target_row = indexed[target_component][block_id]
        target_opposite = float(
            target_row[f"opposite_omega3_{P10_INTACT}"]
        )
        matrix.append(
            [
                target_opposite - float(target_row[f"omega3_{P10_INTACT}"]),
                *[
                    target_opposite
                    - float(
                        indexed[label][block_id][
                            f"opposite_omega3_{P10_INTACT}"
                        ]
                    )
                    for label in controls
                ],
            ]
        )
        strata.append(f"{target_row['chrom']}:{target_row['strand']}")
    inference = stratified_studentized_max_t(
        np.asarray(matrix, dtype=float),
        names=names,
        strata=strata,
        seed=seed,
        n_bootstrap=n_bootstrap,
    ).as_dict()
    lower = inference["lower"]
    return {
        "controls": list(controls),
        "members": list(names),
        "seed": int(seed),
        "inference": inference,
        "reverse_handedness_lower_positive": bool(lower[handedness_name] > 0),
        "named_controls_lower_positive": bool(
            all(lower[name] > 0 for name in control_names)
        ),
        "passed": bool(all(lower[name] > 0 for name in names)),
    }


def adjudicate_register_relocation(
    target: Mapping[str, object],
    specificity: Mapping[str, object],
    *,
    capability_passed: bool,
    phase2_p3_authorized: bool,
    joint_requested: bool = False,
    phase2_joint_authorized: bool = False,
    jreg_direction: str | None = None,
    sesoi: float = 0.05,
) -> dict[str, str]:
    """Apply the asymmetric frozen decision hierarchy.

    The P3 conclusion is decided before the optional modifier arm.  ``Jreg`` can add a
    modifier clause but cannot turn a capable P3 relocation into a failure merely because
    P10 is unresolved.
    """

    if not phase2_p3_authorized:
        return {
            "code": "P3_REGISTER_NOT_AUTHORIZED",
            "licensed_claim": "Phase II did not authorize a period-3 register-relocation target.",
        }
    if not capability_passed:
        return {
            "code": "P3_REGISTER_INSTRUMENT_UNRESOLVED",
            "licensed_claim": (
                "No register interpretation; a frozen capability or health gate failed."
            ),
        }
    strong_relocation = bool(target.get("all_registers_positive")) and bool(
        target.get("omega3_material_positive")
    )
    if strong_relocation and not bool(specificity.get("passed")):
        return {
            "code": "P3_REGISTER_INSTRUMENT_UNRESOLVED",
            "licensed_claim": (
                "The imposed-register lattice is not specific to the learned period-3 "
                "hidden direction."
            ),
        }
    if not strong_relocation:
        if bool(target.get("opposite_all_registers_positive")) and bool(
            target.get("opposite_omega3_material_positive")
        ) and not bool(target.get("expected_partial_concordance_pattern")):
            if not bool(specificity.get("passed")):
                return {
                    "code": "P3_REGISTER_INSTRUMENT_UNRESOLVED",
                    "licensed_claim": "The opposite-handed pattern failed frozen specificity.",
                }
            return {
                "code": "P3_REGISTER_HANDEDNESS_FALSIFIED",
                "licensed_claim": (
                    "The material cyclic relocation followed the prespecified opposite "
                    "orientation; "
                    "the frozen biological handedness is falsified."
                ),
            }
        expected_partial = bool(
            target.get("expected_partial_concordance_pattern")
        )
        opposite_partial = bool(
            target.get("opposite_partial_concordance_pattern")
        )
        partial_specificity = specificity.get("partial_concordance")
        opposite_specificity = specificity.get("opposite_partial_concordance")
        partial_non_handedness = bool(
            isinstance(partial_specificity, Mapping)
            and partial_specificity.get("non_handedness_passed") is True
        )
        opposite_non_handedness = bool(
            isinstance(opposite_specificity, Mapping)
            and opposite_specificity.get("non_handedness_passed") is True
        )
        if expected_partial and opposite_partial:
            if not (
                partial_non_handedness
                and opposite_non_handedness
            ):
                return {
                    "code": "P3_REGISTER_INSTRUMENT_UNRESOLVED",
                    "licensed_claim": (
                        "The bidirectional partial pattern failed frozen specificity."
                    ),
                }
            return {
                "code": "P3_REGISTER_PARTIAL_HANDEDNESS_AMBIGUOUS",
                "licensed_claim": (
                    "Both prespecified orientations met the partial-concordance pattern; "
                    "no directional register interpretation is licensed."
                ),
            }
        if opposite_partial:
            if not opposite_non_handedness:
                return {
                    "code": "P3_REGISTER_INSTRUMENT_UNRESOLVED",
                    "licensed_claim": "The opposite partial pattern failed frozen specificity.",
                }
            if not bool(
                isinstance(opposite_specificity, Mapping)
                and opposite_specificity.get("reverse_handedness_resolved") is True
            ):
                return {
                    "code": "P3_REGISTER_PARTIAL_HANDEDNESS_AMBIGUOUS",
                    "licensed_claim": (
                        "The opposite 2-of-3 pattern was present, but the corrected "
                        "expected-minus-opposite handedness contrast was unresolved."
                    ),
                }
            if not bool(
                isinstance(opposite_specificity, Mapping)
                and opposite_specificity.get("passed") is True
            ):
                return {
                    "code": "P3_REGISTER_INSTRUMENT_UNRESOLVED",
                    "licensed_claim": (
                        "The opposite partial pattern failed its joint frozen specificity "
                        "family."
                    ),
                }
            return {
                "code": "P3_REGISTER_PARTIAL_HANDEDNESS_FALSIFIED",
                "licensed_claim": (
                    "Only the prespecified opposite orientation met the material partial-"
                    "concordance rule; the expected handedness is not supported."
                ),
            }
        if expected_partial:
            if not partial_non_handedness:
                return {
                    "code": "P3_REGISTER_INSTRUMENT_UNRESOLVED",
                    "licensed_claim": "The expected partial pattern failed frozen specificity.",
                }
            if not bool(target.get("handedness_expected_lower_positive")):
                return {
                    "code": "P3_REGISTER_PARTIAL_HANDEDNESS_AMBIGUOUS",
                    "licensed_claim": (
                        "The expected 2-of-3 pattern was present, but the corrected "
                        "expected-minus-opposite handedness contrast was unresolved."
                    ),
                }
            if (
                isinstance(partial_specificity, Mapping)
                and partial_specificity.get("passed") is True
                and bool(target.get("partial_concordance_pattern"))
            ):
                return {
                    "code": "P3_REGISTER_PARTIAL_CONCORDANCE_SUPPORTED",
                    "licensed_claim": (
                        "The period-3 intervention produced a material aggregate boundary-phase "
                        "effect with concordant point estimates in all three imposed registers "
                        "and simultaneous positive support in at least two. This supports a rough "
                        "period-3-to-boundary link, not full cyclic relocation or path mediation."
                    ),
                }
        if bool(target.get("all_registers_bounded_null")):
            return {
                "code": "P3_REGISTER_BOUNDED_NULL",
                "licensed_claim": (
                    "No register-relocation effect larger than +/-0.05 was detected "
                    "in the frozen domain."
                ),
            }
        if not bool(specificity.get("passed")):
            return {
                "code": "P3_REGISTER_INSTRUMENT_UNRESOLVED",
                "licensed_claim": "The unresolved register pattern failed frozen specificity.",
            }
        return {
            "code": "P3_REGISTER_INCONCLUSIVE",
            "licensed_claim": (
                "The relocation family did not jointly clear the material Omega3 and "
                "all-register criteria."
            ),
        }

    base_code = "P3_REGISTER_RELOCATION_SUPPORTED"
    base_claim = (
        "A span-wide period-3 carrier intervention cyclically relocated the preferred annotated "
        "boundary-phase relation in the frozen SpliceBERT assay."
    )
    if not joint_requested:
        return {"code": base_code, "licensed_claim": base_claim}
    if not phase2_joint_authorized:
        return {
            "code": base_code + "_P10_NOT_AUTHORIZED",
            "licensed_claim": (
                base_claim
                + " The approximately-10.63 modifier arm was not authorized."
            ),
        }
    if not bool(target.get("all_joint_state_registers_positive")):
        return {
            "code": base_code + "_P10_MODIFIER_UNRESOLVED",
            "licensed_claim": (
                base_claim
                + " The attenuated approximately-10.63 lattice did not retain all three "
                "register contrasts, so its modifier role is unresolved."
            ),
        }
    if jreg_direction not in {"positive", "negative"}:
        raise ValueError("joint adjudication requires a preregistered Jreg direction")
    inference = target.get("optional_modifier_inference")
    if not isinstance(inference, Mapping):
        raise ValueError("target inference is absent")
    lower = inference.get("lower")
    upper = inference.get("upper")
    if not isinstance(lower, Mapping) or not isinstance(upper, Mapping):
        raise ValueError("target Jreg interval is absent")
    low = float(lower["jreg"])
    high = float(upper["jreg"])
    modifier = low > sesoi if jreg_direction == "positive" else high < -sesoi
    if modifier:
        return {
            "code": base_code + "_P10_MODIFIED",
            "licensed_claim": (
                base_claim
                + " Health-preserving approximately-10.63 attenuation modified that relocation "
                "in the preregistered direction; this is joint use, not a directed pathway."
            ),
        }
    return {
        "code": base_code + "_P10_MODIFIER_UNRESOLVED",
        "licensed_claim": base_claim + " The approximately-10.63 modifier contrast was unresolved.",
    }


__all__ = [
    "PHASES",
    "REGISTERS",
    "P10_INTACT",
    "P10_ATTENUATED",
    "TARGET_COMPONENT",
    "PARTIAL_SPECIFICITY_SEED",
    "OPPOSITE_PARTIAL_SPECIFICITY_SEED",
    "SimultaneousIntervals",
    "adjudicate_register_relocation",
    "analyze_register_component",
    "block_register_contrasts",
    "exact_add_one_upper_p",
    "phase_delta",
    "opposite_preferred_delta",
    "paired_register_specificity_interval",
    "paired_named_control_specificity",
    "partial_handedness_specificity_family",
    "opposite_partial_handedness_specificity_family",
    "preferred_delta",
    "register_specificity",
    "register_endpoint_name",
    "relocate_band_within_span",
    "relocation_mechanics",
    "stratified_studentized_max_t",
]
