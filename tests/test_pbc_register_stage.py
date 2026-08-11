from __future__ import annotations

import copy

import pytest

from gfmcd.provenance import file_record
from gfmcd.splicing.pbc_register_stage import (
    PHASE2_ANALYSIS_SCHEMA,
    PHASE2_AUDIT_SCHEMA,
    REGISTER_PREFLIGHT_ANALYSIS_SCHEMA,
    REGISTER_PREFLIGHT_AUDIT_SCHEMA,
    RegisterStageError,
    canonical_json_sha256,
    make_register_target_seal,
    projected_mde80,
    register_preflight_decision,
    validate_phase2_authorization,
    verify_register_target_seal,
)
from pbc_candidate_fixture import candidate_bindings, prerequisite_manifest_chain


def _phase2(*, joint: bool = False):
    analysis = {
        "schema": PHASE2_ANALYSIS_SCHEMA,
        "status": "COMPLETE",
        "authorization": {
            "authorize_p3_phase3": True,
            "authorize_joint_phase3": joint,
        },
        "endpoints": {"J": {"estimate": -0.12 if joint else None}},
    }
    audit = {
        "schema": PHASE2_AUDIT_SCHEMA,
        "status": "PASS",
        "analysis_sha256": "analysis-hash",
    }
    return analysis, audit


def _health(*, joint: bool) -> dict:
    p10_states = ("intact", "attenuated") if joint else ("intact",)
    components = (
        "p3",
        *(f"p3_signed_{index:02d}" for index in range(31)),
        "p3_common_mode",
        "p3_decoy_span",
    )
    labels = {
        f"{component}__{state}__r{register}"
        for component in components
        for state in p10_states
        for register in (-1, 0, 1)
    }
    states = {label: {"passed": True} for label in labels}
    tables = {
        label: {"pass": True, "raw_blocks": [{} for _ in range(120)]}
        for label in labels
    }
    return {
        "schema": "gfmcd.pbc.p3_register_table_health.v1",
        "stage": "development",
        "authorized_table_health": {
            "schema": "gfmcd.pbc.carrier_factorial_table_health.v1",
            "target_mode": "REGISTER_JOINT" if joint else "REGISTER_P3_ONLY",
            "n_blocks": 120,
            "n_tables": len(labels),
            "tables": tables,
            "all_tables_pass": True,
        },
        "states": states,
        "all_required_states_passed": True,
        "minimum_p10_reader_retention": 0.91,
        "joint_state_passed": joint,
    }


def _preflight_decision(
    *,
    joint: bool = False,
    relocation_status: str = "P3_REGISTER_RELOCATION_SUPPORTED",
):
    return register_preflight_decision(
        development_analysis={"status": relocation_status},
        construction_mechanics={
            "passed": True,
            "register_zero_no_hook_max_abs": 0.0,
            "order_replay_max_abs": 0.0,
        },
        health=_health(joint=joint),
        resolution={
            "omega3_intact": 0.02,
            "s_-1_intact": 0.03,
            "s_0_intact": 0.03,
            "s_1_intact": 0.03,
            "D_path": 0.04,
            "R_path": 0.04,
            "Q_path": 0.04,
            "jreg": 0.04 if joint else float("inf"),
        },
        fixture_calibration={
            "all_required_cases_passed": True,
            "familywise_false_positive_rate": 0.05,
            "corrected_power_at_0_075": 0.82,
        },
        phase2_authorization={
            "phase2_p3_authorized": True,
            "phase2_joint_authorized": joint,
        },
    )


def test_partial_concordance_can_open_reserve_but_not_joint_arm():
    result = _preflight_decision(
        relocation_status="P3_REGISTER_PARTIAL_CONCORDANCE_SUPPORTED"
    )
    assert result["status"] == "P3_REGISTER_PREFLIGHT_PASS"
    assert result["authorization"]["authorize_p3_register_reserve"]
    assert not result["authorization"]["authorize_joint_register_reserve"]


def test_phase2_p3_authorization_is_independent_of_optional_p10():
    analysis, audit = _phase2(joint=False)
    result = validate_phase2_authorization(
        analysis, audit, analysis_sha256="analysis-hash"
    )
    assert result == {
        "phase2_p3_authorized": True,
        "phase2_joint_authorized": False,
        "jreg_direction": None,
    }
    decision = _preflight_decision(joint=False)
    assert decision["status"] == "P3_REGISTER_PREFLIGHT_PASS"
    assert decision["authorization"]["authorize_p3_register_reserve"]
    assert not decision["authorization"]["authorize_joint_register_reserve"]


def test_joint_direction_is_frozen_from_corrected_phase2_j():
    analysis, audit = _phase2(joint=True)
    result = validate_phase2_authorization(
        analysis, audit, analysis_sha256="analysis-hash"
    )
    assert result["jreg_direction"] == "negative"


def test_projected_mde_uses_stratum_centered_development_variance():
    rows = [
        {
            "chrom": f"chr{1 + index % 2}",
            "strand": "+" if (index // 2) % 2 == 0 else "-",
            "omega3_intact": 0.2 + index * 0.002,
            "s_-1_intact": 0.2 + index * 0.003,
        }
        for index in range(24)
    ]
    result = projected_mde80(
        rows,
        endpoint_names=("omega3_intact", "s_-1_intact"),
        reserve_blocks=384,
    )
    assert 0 < result["omega3_intact"] < result["s_-1_intact"] < 0.05


def test_target_seal_binds_files_and_refuses_design_tampering(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    preflight = _preflight_decision(joint=False)
    assert preflight["schema"] == REGISTER_PREFLIGHT_ANALYSIS_SCHEMA
    audit = {
        "schema": REGISTER_PREFLIGHT_AUDIT_SCHEMA,
        "status": "PASS",
    }
    seal = make_register_target_seal(
        phase2_authorization={
            "phase2_p3_authorized": True,
            "phase2_joint_authorized": False,
            "jreg_direction": None,
        },
        preflight_analysis=preflight,
        preflight_audit=audit,
        bindings={"cohort": file_record(source), **candidate_bindings(tmp_path)},
        include_p10=False,
        p3_attenuation_dose=0.5,
        p10_attenuation_dose=None,
        output_path=str(tmp_path / "target.json"),
        prerequisite_manifest_chain=prerequisite_manifest_chain(),
    )
    verify_register_target_seal(seal)
    assert seal["relocation_outputs"]["expected_partial_specificity_seed"] == 2026081010
    assert seal["relocation_outputs"]["opposite_partial_specificity_seed"] == 2026081012
    tampered = copy.deepcopy(seal)
    tampered["design"]["direction"] = "left"
    try:
        verify_register_target_seal(tampered)
    except RegisterStageError:
        pass
    else:
        raise AssertionError("tampered target seal was accepted")
    for field in (
        "expected_partial_specificity_seed",
        "opposite_partial_specificity_seed",
    ):
        tampered = copy.deepcopy(seal)
        tampered["relocation_outputs"][field] += 1
        tampered_body = dict(tampered)
        tampered_body.pop("seal_sha256")
        tampered["seal_sha256"] = canonical_json_sha256(tampered_body)
        with pytest.raises(RegisterStageError, match="inference bindings drifted"):
            verify_register_target_seal(tampered)
