from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

from gfmcd.splicing.pbc_carrier_factorial import (
    DOSE_GRID,
    P10_BINS_1026,
    P3_BINS_1026,
    attenuate_band,
    balanced_pseudo_phase_assignment_audit,
    balanced_pseudo_phase_assignments,
    block_psi,
    build_phase_control_family,
    build_world_tables,
    c43_margin,
    carrier_phase2_decision,
    classify_effect_interval,
    classify_interaction_interval,
    evaluate_preflight,
    exact_add_one_absolute_p,
    exact_add_one_upper_p,
    factorial_contrasts,
    hidden_axis_control_tables,
    intervention_audit,
    make_target_seal,
    mechanical_audit,
    phase_control_tables,
    projected_mde80,
    reconstruct_attenuated_band,
    verify_target_seal,
)
from gfmcd.splicing.positional_spectrum import band_component
from gfmcd.provenance import file_record, sha256_file


def _table(hidden: int = 8) -> np.ndarray:
    rng = np.random.default_rng(17)
    return rng.normal(size=(1026, hidden))


def test_directional_scoped_support_is_nonmaterial_and_cannot_authorize_phase3():
    result = carrier_phase2_decision(
        clean_behavior_classification="BALANCED_PREFERENCE_BOUNDED_NULL",
        endpoints={
            "Psi00": {"ci": [0.01, 0.04], "classification": "BOUNDED_NULL"},
            "N3": {"ci": [0.005, 0.035], "classification": "BOUNDED_NULL"},
            "N10": {"status": "NOT_RUN"},
            "J": {"status": "NOT_RUN"},
        },
        specificity={
            "Psi00": {"corrected_pass": True},
            "N3": {"corrected_pass": True},
        },
        target_mode="P3_ONLY",
        target_resolution={
            "Psi00": {"pass": True, "mde80": 0.035},
            "N3": {"pass": True, "mde80": 0.038},
        },
        clean_directional_point_estimates={
            name: 0.01
            for name in ("Dplus", "Dminus", "Psi0", "Psi1", "Psi2")
        },
    )
    assert result["verdict"] == "P3_DIRECTIONAL_SCOPED_SUPPORT"
    assert result["directional_scoped_support"]
    assert result["support_tiers"] == ["P3_DIRECTIONAL_SCOPED_SUPPORT"]
    assert not result["authorize_p3_phase3"]
    assert not result["authorize_joint_phase3"]


def test_directional_scoped_support_requires_all_five_point_signs():
    result = carrier_phase2_decision(
        clean_behavior_classification="BALANCED_PREFERENCE_BOUNDED_NULL",
        endpoints={
            "Psi00": {"ci": [0.01, 0.04], "classification": "BOUNDED_NULL"},
            "N3": {"ci": [0.005, 0.035], "classification": "BOUNDED_NULL"},
        },
        specificity={
            "Psi00": {"corrected_pass": True},
            "N3": {"corrected_pass": True},
        },
        target_mode="P3_ONLY",
        target_resolution={
            "Psi00": {"pass": True, "mde80": 0.035},
            "N3": {"pass": True, "mde80": 0.038},
        },
        clean_directional_point_estimates={
            "Dplus": 0.01,
            "Dminus": 0.01,
            "Psi0": 0.01,
            "Psi1": 0.01,
            "Psi2": -0.001,
        },
    )
    assert result["verdict"] == "BALANCED_PREFERENCE_BOUNDED_NULL"
    assert not result["directional_scoped_support"]


def _lattice_rows() -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    diagonal = (0.3, 0.4, 0.5)
    for donor in range(3):
        for acceptor in range(3):
            delta = (acceptor - donor) % 3
            value = diagonal[donor] if delta == 0 else (0.02 if delta == 1 else -0.02)
            rows.append(
                {
                    "donor_phase": donor,
                    "acceptor_phase": acceptor,
                    "coupling_ac": value,
                }
            )
    return rows


def _passing_dose_row(*, p10: bool = False, cross: bool = True) -> dict[str, object]:
    row: dict[str, object] = {
        "mechanical_pass": True,
        "health_pass": True,
        "all_control_health_pass": True,
        "rescue_forward_max_abs_error": 0.0,
        "rescue_ratio_health": {"pass": True},
        "rescue_recovery": 1.0,
        "rescue_recovery_ci": [0.95, 1.0],
        "cross_carrier_retention": 0.95,
        "cross_carrier_retention_ci": [0.90, 1.0],
        "cross_carrier_ratio_health": {"pass": True},
        "joint_health_pass": cross,
        "control_families": {
            family: {
                "n_controls": 31,
                "target_exceeds_every_control": True,
                "add_one_p": 1 / 32,
                "paired_excess_ci": [0.01, 0.2],
            }
            for family in ("phase", "axis")
        },
    }
    if p10:
        row.update(
            {
                "intact_carrier_score_ci": [0.02, 0.10],
                "denominator_finite_sign_stable_fraction": 1.0,
                "capability_ratio_health": {"pass": True},
                "capability_fraction_reduced": 0.70,
                "capability_fraction_reduced_ci": [0.40, 0.90],
            }
        )
    else:
        row["capability_ci"] = [0.08, 0.20]
    return row


def _shared_gates() -> dict[str, bool]:
    return {
        name: True
        for name in (
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
    }


def _mde(value: float) -> dict[str, float | int]:
    return {
        "mde80": value,
        "monte_carlo_se": 0.0001,
        "n_bootstrap": 20_000,
        "target_blocks": 600,
    }


def test_band_attenuation_delivers_exact_dose_and_preserves_cross_band():
    table = _table()
    edited = attenuate_band(table, P3_BINS_1026, 0.75)
    audit = intervention_audit(
        table,
        edited,
        bins=P3_BINS_1026,
        dose=0.75,
        protect_bins=P10_BINS_1026,
    )
    assert audit["outside_band_pass"]
    assert audit["dose_pass"]
    assert audit["protected_band_pass"]
    rescued = reconstruct_attenuated_band(edited, table, P3_BINS_1026, 0.75)
    assert np.allclose(rescued, table, atol=1e-10)


def test_four_world_composition_commutes_and_contrasts_use_frozen_arm_coding():
    table = _table()
    worlds = build_world_tables(table, p3_dose=0.5, p10_dose=0.75)
    reverse = attenuate_band(worlds["Y01"], P3_BINS_1026, 0.5)
    assert np.allclose(worlds["Y11"], reverse, atol=1e-12)
    audit = mechanical_audit(table, p3_dose=0.5, p10_dose=0.75)
    assert audit["passed"]
    assert factorial_contrasts({"Y00": 1.0, "Y10": 0.7, "Y01": 0.8, "Y11": 0.4}) == {
        "N3": pytest.approx(0.3),
        "N10": pytest.approx(0.2),
        "Nboth": pytest.approx(0.6),
        "J": pytest.approx(-0.1),
    }


def test_structured_controls_are_sign_balanced_and_match_frozen_geometry():
    table = _table()
    phase_tables, phase_specs = phase_control_tables(
        table, P3_BINS_1026, 0.5, seed=2026081003
    )
    signs = np.asarray([row["phase_signs"] for row in phase_specs])
    assert len(phase_tables) == 31
    assert np.all(np.abs(signs.sum(axis=0)) == 1)
    target_rms = np.sqrt(
        np.mean(np.square(attenuate_band(table, P3_BINS_1026, 0.5) - table))
    )
    assert np.sqrt(np.mean(np.square(phase_tables[0] - table))) == pytest.approx(
        target_rms, rel=1e-10
    )
    before_power = np.linalg.norm(band_component(table, P3_BINS_1026))
    after_power = np.linalg.norm(band_component(phase_tables[0], P3_BINS_1026))
    assert after_power == pytest.approx(before_power, rel=1e-10)

    axis_tables, _ = hidden_axis_control_tables(
        table, P3_BINS_1026, 0.5, seed=2027081003
    )
    native = band_component(table, P3_BINS_1026)
    transformed = (table - axis_tables[0]) / 0.5
    assert np.allclose(transformed @ transformed.T, native @ native.T, atol=1e-8)
    assert not np.allclose(axis_tables[0], attenuate_band(table, P3_BINS_1026, 0.5))


def test_control_package_keeps_all_seven_families_separate_and_reproducible():
    table = _table(hidden=4)
    controls, specs = build_phase_control_family(
        table, p3_dose=0.5, p10_dose=0.75, seed=20260810
    )
    counts = Counter(str(row["family"]) for row in specs.values())
    assert counts == {
        "p3_phase": 31,
        "p3_axis": 31,
        "p10_phase": 31,
        "p10_axis": 31,
        "joint_p3": 31,
        "joint_p10": 31,
        "joint": 31,
    }
    replay, replay_specs = build_phase_control_family(
        table, p3_dose=0.5, p10_dose=0.75, seed=20260810
    )
    assert specs == replay_specs
    assert all(np.array_equal(controls[label], replay[label]) for label in controls)
    family_seeds = {
        family: {int(row["seed"]) for row in specs.values() if row["family"] == family}
        for family in counts
    }
    assert family_seeds == {
        "p3_phase": {2026081003},
        "p3_axis": {2026081003},
        "p10_phase": {2026081004},
        "p10_axis": {2026081004},
        "joint_p3": {2026081005},
        "joint_p10": {2026081005},
        "joint": {2026081005},
    }


def test_c43_and_clean_lattice_endpoints_are_exact():
    assert c43_margin({"A": 1.0, "C": 2.0, "G": 7.0, "T": 3.0}) == 5.0
    summary = block_psi(_lattice_rows())
    assert summary["M0"] == pytest.approx(0.4)
    assert summary["Mplus"] == pytest.approx(0.02)
    assert summary["Mminus"] == pytest.approx(-0.02)
    assert summary["Psi"] == pytest.approx(0.4)
    assert summary["Dplus"] == pytest.approx(0.38)
    assert summary["Dminus"] == pytest.approx(0.42)
    assert summary["Psi0"] == pytest.approx(0.3)
    assert summary["Psi1"] == pytest.approx(0.4)
    assert summary["Psi2"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="exactly nine"):
        block_psi(_lattice_rows()[:-1])


def test_exact_add_one_and_interval_classes_cover_negative_interaction():
    controls = np.linspace(-0.03, 0.03, 31)
    assert exact_add_one_upper_p(0.20, controls) == 1 / 32
    assert exact_add_one_absolute_p(-0.20, controls) == 1 / 32
    assert classify_effect_interval(0.06, 0.10) == "MATERIAL_SUPPORTIVE"
    assert classify_effect_interval(-0.04, 0.04) == "BOUNDED_NULL"
    assert (
        classify_interaction_interval(-0.12, -0.06)
        == "MATERIAL_SUPER_ADDITIVE_ATTENUATION_DAMAGE"
    )


def test_pseudo_phase_assignments_are_fresh_reproducible_and_exactly_balanced():
    first = balanced_pseudo_phase_assignments(600, seed=109)
    replay = balanced_pseudo_phase_assignments(600, seed=109)
    changed = balanced_pseudo_phase_assignments(600, seed=110)
    assert np.array_equal(first, replay)
    assert not np.array_equal(first, changed)
    assert all(Counter(row) == {index: 100 for index in range(6)} for row in first)
    audit = balanced_pseudo_phase_assignment_audit(600, seed=109)
    assert audit["exactly_balanced"]
    assert audit["unique_assignment_count"] == 999
    assert audit["all_assignments_unique"]


def test_development_mde_projection_uses_frozen_target_size_and_mc_precision():
    values = np.linspace(-0.02, 0.02, 120)
    strata = [f"chr{1 + index % 3}|{'+' if index % 2 else '-'}" for index in range(120)]
    result = projected_mde80(values, strata, seed=77)
    assert result["n_bootstrap"] == 20_000
    assert result["target_blocks"] == 600
    assert sum(result["stratum_target_allocation"].values()) == 600
    assert result["mde80"] < 0.04
    assert result["monte_carlo_se"] <= 0.0002


def test_preflight_is_p3_mandatory_and_p10_optional():
    p3_rows = {f"{dose:.2f}": _passing_dose_row() for dose in DOSE_GRID}
    p10_rows = {
        f"{dose:.2f}": _passing_dose_row(p10=True, cross=False) for dose in DOSE_GRID
    }
    result = evaluate_preflight(
        {"shared_gates": _shared_gates(), "dose_rows": {"p3": p3_rows, "p10": p10_rows}},
        development_mde={"clean_psi": _mde(0.03), "N3": _mde(0.03)},
    )
    assert result["status"] == "P3_ONLY_TARGET_AUTHORIZED"
    assert result["target_mode"] == "P3_ONLY"
    assert result["p10_status"] == "OPTIONAL_P10_INSTRUMENT_UNRESOLVED"

    for row in p10_rows.values():
        row["joint_health_pass"] = True
    full = evaluate_preflight(
        {"shared_gates": _shared_gates(), "dose_rows": {"p3": p3_rows, "p10": p10_rows}},
        development_mde={
            "clean_psi": _mde(0.03),
            "N3": _mde(0.03),
            "N10": _mde(0.04),
            "J": _mde(0.04),
        },
    )
    assert full["status"] == "FULL_FACTORIAL_TARGET_AUTHORIZED"
    assert full["target_mode"] == "FULL_FACTORIAL"

    broken = dict(_shared_gates())
    broken["construction_audit"] = False
    failed = evaluate_preflight(
        {"shared_gates": broken, "dose_rows": {"p3": p3_rows, "p10": p10_rows}},
        development_mde={
            "clean_psi": _mde(0.0),
            "N3": _mde(0.0),
            "N10": _mde(0.0),
            "J": _mde(0.0),
        },
    )
    assert failed["status"] == "P3_INSTRUMENT_UNRESOLVED"
    assert failed["target_mode"] == "SEALED"


def test_target_seal_hash_binds_control_seed_and_every_control(tmp_path: Path):
    p3_rows = {f"{dose:.2f}": _passing_dose_row() for dose in DOSE_GRID}
    p10_rows = {
        f"{dose:.2f}": _passing_dose_row(p10=True) for dose in DOSE_GRID
    }
    preflight = evaluate_preflight(
        {"shared_gates": _shared_gates(), "dose_rows": {"p3": p3_rows, "p10": p10_rows}},
        development_mde={
            "clean_psi": _mde(0.02),
            "N3": _mde(0.02),
            "N10": _mde(0.03),
            "J": _mde(0.03),
        },
    )
    hashes = {
        f"{family}_{index:03d}": {
            "family": family,
            "control_id": index,
            "world": "Y10",
            "sha256": "0" * 64,
        }
        for family in (
            "p3_phase",
            "p3_axis",
            "p10_phase",
            "p10_axis",
            "joint_p3",
            "joint_p10",
            "joint",
        )
        for index in range(1, 32)
    }
    raw_hash, analysis_hash = "2" * 64, "3" * 64
    construction_source_path = tmp_path / "construction.source.json"
    construction_source_path.write_text(json.dumps({"fixture": True}))
    construction_audit_path = tmp_path / "construction.audit.json"
    construction_audit_path.write_text(
        json.dumps(
            {
                "schema": "gfmcd.positional_boundary_coherence.cohort_audit.v1",
                "stage": "construction",
                "status": "AUDIT_PASS_TARGET_AUTHORIZED",
                "passed": True,
                "target_execution_authorized": True,
            }
        )
    )
    construction_manifest_path = tmp_path / "construction.audit.manifest.json"
    construction_manifest_path.write_text(
        json.dumps(
            {
                "experiment_id": (
                    "splicebert-positional-boundary-coherence:"
                    "composition-balanced-cohort-audit-v1"
                ),
                "status": "AUDIT_PASS_TARGET_AUTHORIZED",
                "config": {"stage": "construction"},
                "inputs": [file_record(construction_source_path)],
                "outputs": [file_record(construction_audit_path)],
            }
        )
    )
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                "schema": "gfmcd.pbc.carrier_factorial_preflight.v1",
                "target_mode": "FULL_FACTORIAL",
                "raw_sha256": raw_hash,
                "analysis_sha256": analysis_hash,
                "construction_audit_provenance": {
                    "artifact_sha256": sha256_file(construction_audit_path),
                    "manifest_sha256": sha256_file(construction_manifest_path),
                    "experiment_id": (
                        "splicebert-positional-boundary-coherence:"
                        "composition-balanced-cohort-audit-v1"
                    ),
                },
            }
        )
    )
    preflight_manifest_path = tmp_path / "preflight.manifest.json"
    preflight_manifest_path.write_text(
        json.dumps(
            {
                "inputs": [
                    file_record(construction_audit_path),
                    file_record(construction_manifest_path),
                ],
                "outputs": [file_record(preflight_path)],
            }
        )
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": "gfmcd.pbc.carrier_factorial_preflight_audit.v1",
                "status": "PASS",
                "target_mode": "FULL_FACTORIAL",
                "preflight_sha256": sha256_file(preflight_path),
                "raw_sha256": raw_hash,
                "analysis_sha256": analysis_hash,
            }
        )
    )
    audit_manifest_path = tmp_path / "audit.manifest.json"
    audit_manifest_path.write_text(json.dumps({"outputs": [file_record(audit_path)]}))
    conformance_path = tmp_path / "conformance.audit.json"
    conformance_path.write_text(
        json.dumps(
            {
                "schema": "gfmcd.pbc.protocol_code_conformance_audit.v1",
                "stage": "development",
                "status": "PASS",
                "passed": True,
                "target_outcomes_read": False,
                "tracked_tree_clean": True,
                "git_revision": "f" * 40,
            }
        )
    )
    conformance_manifest_path = tmp_path / "conformance.audit.manifest.json"
    conformance_manifest_path.write_text(
        json.dumps(
            {
                "experiment_id": "pbc-protocol-code-conformance-audit-v1",
                "status": "PASS",
                "outputs": [file_record(conformance_path)],
            }
        )
    )
    chain_paths = (
        construction_source_path,
        construction_audit_path,
        construction_manifest_path,
        preflight_path,
        preflight_manifest_path,
        audit_path,
        audit_manifest_path,
        conformance_path,
        conformance_manifest_path,
    )
    authorization = {
        "audit_schema": "gfmcd.pbc.carrier_factorial_preflight_audit.v1",
        "audit_status": "PASS",
        "target_mode": "FULL_FACTORIAL",
        "preflight_sha256": sha256_file(preflight_path),
        "raw_sha256": raw_hash,
        "analysis_sha256": analysis_hash,
        "preflight_manifest_sha256": sha256_file(preflight_manifest_path),
        "audit_sha256": sha256_file(audit_path),
        "audit_manifest_sha256": sha256_file(audit_manifest_path),
        "construction_audit_sha256": sha256_file(construction_audit_path),
        "construction_audit_manifest_sha256": sha256_file(
            construction_manifest_path
        ),
        "conformance_audit_sha256": sha256_file(conformance_path),
        "conformance_audit_manifest_sha256": sha256_file(
            conformance_manifest_path
        ),
    }
    seal = make_target_seal(
        preflight,
        pinned_files={path.name: file_record(path) for path in chain_paths},
        control_seed=20260810,
        structured_control_hashes=hashes,
        preflight_authorization=authorization,
        runtime={"device": "cpu", "batch_size": 4, "dtype": "float32"},
        target_outputs={
            "shard_count": 1,
            "shards": ["/tmp/pbc-target.json"],
            "assembled": "/tmp/pbc-target.json",
        },
        frozen_revision="f" * 40,
    )
    verify_target_seal(seal)
    tampered = {**seal, "structured_control_seed": 9}
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_target_seal(tampered)
    self_sealed = dict(seal)
    self_sealed["preflight_authorization"] = {
        **authorization,
        "audit_sha256": "9" * 64,
    }
    body = dict(self_sealed)
    body.pop("seal_sha256")
    from gfmcd.splicing.pbc_carrier_factorial import canonical_json_sha256

    self_sealed["seal_sha256"] = canonical_json_sha256(body)
    with pytest.raises(RuntimeError, match="complete audit artifact chain"):
        verify_target_seal(self_sealed)
    construction_manifest_path.write_text(json.dumps({"substituted": True}))
    with pytest.raises(RuntimeError, match="complete audit artifact chain"):
        verify_target_seal(seal)
