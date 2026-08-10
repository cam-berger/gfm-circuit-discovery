from __future__ import annotations

import numpy as np
import pytest

from gfmcd.splicing.pbc_register_relocation import (
    P10_ATTENUATED,
    P10_INTACT,
    REGISTERS,
    adjudicate_register_relocation,
    analyze_register_component,
    block_register_contrasts,
    paired_register_specificity_interval,
    opposite_preferred_delta,
    phase_delta,
    preferred_delta,
    register_specificity,
    relocate_band_within_span,
    relocation_mechanics,
)


def _periodic_table(length: int = 30) -> np.ndarray:
    position = np.arange(length, dtype=float)
    return np.stack(
        (
            np.cos(2 * np.pi * position / 3),
            np.sin(2 * np.pi * position / 3),
            0.1 * position,
        ),
        axis=1,
    )


def _lattice_rows(
    *,
    blocks: int = 24,
    component: str = "p3",
    states: tuple[str, ...] = (P10_INTACT,),
    effect: float = 0.30,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block in range(blocks):
        chrom = f"chr{1 + block % 2}"
        strand = "+" if (block // 2) % 2 == 0 else "-"
        block_offset = (block - (blocks - 1) / 2) * 0.002
        for state in states:
            local_effect = (
                effect + block_offset
                if state == P10_INTACT
                else (effect + block_offset) / 2
            )
            for register in REGISTERS:
                for donor in range(3):
                    for acceptor in range(3):
                        delta = phase_delta(donor, acceptor)
                        coupling = block_offset / 3
                        if delta == preferred_delta(register):
                            coupling += local_effect
                        rows.append(
                            {
                                "block_id": f"b{block:03d}",
                                "chrom": chrom,
                                "strand": strand,
                                "component": component,
                                "p10_state": state,
                                "register": register,
                                "donor_phase": donor,
                                "acceptor_phase": acceptor,
                                "phase_delta": delta,
                                "coupling": coupling,
                            }
                        )
    return rows


def test_phase_and_register_conventions_are_explicit():
    assert phase_delta(2, 0) == 1
    assert [preferred_delta(register) for register in REGISTERS] == [1, 0, 2]
    assert [opposite_preferred_delta(register) for register in REGISTERS] == [2, 0, 1]


def test_relocation_changes_only_declared_carrier_and_span():
    table = _periodic_table()
    edited = relocate_band_within_span(
        table, (10,), start=5, end=20, register=1, direction="left"
    )
    assert np.array_equal(edited[:5], table[:5])
    assert np.array_equal(edited[20:], table[20:])
    assert not np.array_equal(edited[5:20], table[5:20])
    replay = relocate_band_within_span(
        table, (10,), start=5, end=20, register=0, direction="left"
    )
    assert np.array_equal(replay, table)
    audit = relocation_mechanics(table, (10,), start=5, end=20, atol=1e-9)
    assert audit["passed"]
    assert audit["kind"] == "construction_invariants"


def test_positive_register_has_negative_endpoint_phase_on_a_pure_tone():
    """Truth-table the ramp sign without consulting preferred_delta()."""

    length = 12
    positions = np.arange(length, dtype=float)
    table = np.column_stack(
        (
            np.cos(2 * np.pi * positions / 3),
            np.sin(2 * np.pi * positions / 3),
        )
    )
    edited = relocate_band_within_span(
        table,
        (length // 3,),
        start=0,
        end=4,
        register=1,
    )
    # At the final endpoint phi=-2pi/3, so the analytic tone is evaluated at j-1.
    assert np.allclose(edited[3], table[2], atol=1e-10)


def test_block_contrasts_require_the_complete_three_by_three_lattice():
    rows = _lattice_rows(blocks=1)
    block = block_register_contrasts(rows)[0]
    assert block["s_-1_intact"] == pytest.approx(0.30)
    assert block["s_0_intact"] == pytest.approx(0.30)
    assert block["s_1_intact"] == pytest.approx(0.30)
    assert block["omega3_intact"] == pytest.approx(0.30)
    with pytest.raises(ValueError, match="incomplete register lattice"):
        block_register_contrasts(rows[:-1])


def test_joint_analysis_estimates_modifier_without_changing_p3_definition():
    rows = _lattice_rows(states=(P10_INTACT, P10_ATTENUATED))
    analysis = analyze_register_component(
        rows, joint=True, seed=41, n_bootstrap=199
    )
    assert analysis["all_registers_positive"]
    assert analysis["omega3_material_positive"]
    assert analysis["inference"]["names"] == [
        "omega3_intact",
        "s_-1_intact",
        "s_0_intact",
        "s_1_intact",
    ]
    assert analysis["omega3"][P10_INTACT] == pytest.approx(0.30)
    assert analysis["omega3"][P10_ATTENUATED] == pytest.approx(0.15)
    assert analysis["optional_modifier_inference"]["means"]["jreg"] == pytest.approx(0.15)


def test_specificity_uses_one_complete_control_family():
    analyses = {
        "p3": {"omega3": {P10_INTACT: 0.30}},
        **{
            f"signed_{index:02d}": {"omega3": {P10_INTACT: 0.01 + index / 10_000}}
            for index in range(31)
        },
    }
    result = register_specificity(analyses)
    assert result["n_controls"] == 31
    assert result["exact_add_one_p_one_sided"] == 1 / 32
    assert result["passed"]


def test_paired_specificity_keeps_whole_blocks_aligned():
    component_blocks = {
        "p3": [
            {
                "block_id": f"b{index}",
                "chrom": f"chr{1 + index % 2}",
                "strand": "+" if (index // 2) % 2 == 0 else "-",
                "omega3_intact": 0.28 + index * 0.002,
            }
            for index in range(24)
        ],
        "signed_00": [
            {
                "block_id": f"b{index}",
                "chrom": f"chr{1 + index % 2}",
                "strand": "+" if (index // 2) % 2 == 0 else "-",
                "omega3_intact": 0.02 - index * 0.001,
            }
            for index in range(24)
        ],
    }
    result = paired_register_specificity_interval(
        component_blocks, seed=9, n_bootstrap=199
    )
    assert result["passed"]
    assert result["inference"]["lower"]["p3_minus_mean_control"] > 0


def test_p10_non_authorization_cannot_veto_period3_closure():
    target = {
        "all_registers_positive": True,
        "omega3_material_positive": True,
        "all_registers_bounded_null": False,
    }
    specificity = {"passed": True}
    result = adjudicate_register_relocation(
        target,
        specificity,
        capability_passed=True,
        phase2_p3_authorized=True,
        joint_requested=True,
        phase2_joint_authorized=False,
    )
    assert result["code"] == "P3_REGISTER_RELOCATION_SUPPORTED_P10_NOT_AUTHORIZED"
    assert "cyclically relocated" in result["licensed_claim"]


def test_capability_failure_is_never_relabelled_as_a_null():
    result = adjudicate_register_relocation(
        {
            "all_registers_positive": False,
            "omega3_material_positive": False,
            "all_registers_bounded_null": True,
        },
        {"passed": True},
        capability_passed=False,
        phase2_p3_authorized=True,
    )
    assert result["code"] == "P3_REGISTER_INSTRUMENT_UNRESOLVED"


def test_opposite_handed_material_result_is_falsification_not_positive():
    result = adjudicate_register_relocation(
        {
            "all_registers_positive": False,
            "omega3_material_positive": False,
            "opposite_all_registers_positive": True,
            "opposite_omega3_material_positive": True,
            "all_registers_bounded_null": False,
        },
        {"passed": True},
        capability_passed=True,
        phase2_p3_authorized=True,
    )
    assert result["code"] == "P3_REGISTER_HANDEDNESS_FALSIFIED"
