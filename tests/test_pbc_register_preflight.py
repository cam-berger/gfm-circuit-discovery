from __future__ import annotations

import importlib.util
from hashlib import sha256
import json
import math
from pathlib import Path

from gfmcd.provenance import file_record, sha256_file
from gfmcd.splicing.pbc_register_relocation import REGISTERS, phase_delta, preferred_delta
from gfmcd.splicing.pbc_register_stage import (
    REGISTER_CONTROL_SEEDS,
    REGISTER_PREFLIGHT_ANALYSIS_SCHEMA,
    REGISTER_PREFLIGHT_AUDIT_SCHEMA,
    REGISTER_PREFLIGHT_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts/splicebert_positional_boundary_coherence" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN = _load("run_p3_register_preflight")
ANALYZE = _load("analyze_p3_register_preflight")
AUDIT = _load("audit_p3_register_preflight")


def _fixture_calibration():
    return {
        "schema": ANALYZE.FIXTURE_SCHEMA,
        "status": "PASS",
        "analysis_engine": "analyze_p3_register_relocation.analyze_payload",
        "decision_engine": "pbc_register_relocation.adjudicate_register_relocation",
        "required_cases": {
            name: {"status": "EXPECTED", "passed": True}
            for name in ANALYZE.FIXTURE_CASES
        },
        "all_required_cases_passed": True,
        "familywise_false_positive_rate": 0.05,
        "corrected_power_at_0_075": 0.80,
        "replicates": 1_000,
        "bootstrap_draws_per_fixture": 99,
        "seed": 17,
        "planted_effect": 0.075,
    }


def _path_mde():
    return {
        "schema": ANALYZE.PATH_MDE_SCHEMA,
        "status": "COMPLETE",
        "reserve_blocks": 384,
        "mde80": {"D_path": 0.02, "R_path": 0.025, "Q_path": 0.03},
        "self_sha256": "d" * 64,
        "candidate_freeze_sha256": "e" * 64,
    }


def _path_mde_audit():
    return {
        "schema": ANALYZE.PATH_MDE_AUDIT_SCHEMA,
        "status": "PASS",
        "path_mde_sha256": "d" * 64,
        "candidate_freeze_sha256": "e" * 64,
        "candidate_freeze_validation_passed": True,
    }


def _mechanics():
    handedness = {}
    for register in REGISTERS:
        phase = -2 * math.pi * register / 3
        handedness[str(register)] = {
            "register": register,
            "measured_acceptor_phase_radians": phase,
            "requested_acceptor_phase_radians": phase,
            "phase_profile_max_abs_error_radians": 0.0,
            "observed_source_displacement": -register,
            "predicted_annotated_phase_delta": (-register) % 3,
            "passed": True,
        }
    return {
        "construction_invariants": {"passed": True},
        "direction": "right",
        "handedness": handedness,
        "handedness_passed": True,
        "p10_independent_reconstruction_relative_max_abs": 0.0,
        "pointwise_non_p3_residual_max_abs": 0.0,
        "p10_explicit_component_retention_relative_max_abs": 0.0,
        "rms_match_max_relative_error": 0.0,
        "register_zero_no_hook_max_abs": 0.0,
        "batch_order_replay_max_abs": 0.0,
        "corruption_order_replay_max_abs": 0.0,
        "ramp_forward_replay_max_abs": 0.0,
        "order_replay_max_abs": 0.0,
        "passed": True,
    }


def _health():
    raw_blocks = [
        {
            "block_id": f"b{block:03d}",
            "stratum": f"chr{1 + block % 2}|{'+' if (block // 2) % 2 == 0 else '-'}",
            "native_source_ac": 1.0,
            "crossed_ac": 1.0,
            "bridge_effect": 0.0,
            "crossed_ac_minus_at": 0.0,
            "protected_log_probability_delta": 0.0,
            "protected_intact_correct": 1,
            "protected_table_correct": 1,
            "protected_total": 1,
            "donor_destruction_response": 1.0,
            "acceptor_destruction_response": 1.0,
            "baseline_donor_destruction_response": 1.0,
            "baseline_acceptor_destruction_response": 1.0,
            "numerical_complete": True,
        }
        for block in range(120)
    ]
    components = (
        "p3",
        *(f"p3_signed_{index:02d}" for index in range(31)),
        "p3_common_mode",
        "p3_decoy_span",
    )
    states = {}
    tables = {}
    for component in components:
        family = (
            "target"
            if component == "p3"
            else "signed_hidden_axis"
            if component.startswith("p3_signed_")
            else "common_mode"
            if component == "p3_common_mode"
            else "decoy_span"
        )
        control_id = (
            int(component.rsplit("_", 1)[1]) + 1
            if component.startswith("p3_signed_")
            else None
        )
        for register in REGISTERS:
            label = f"{component}__intact__r{register}"
            spec = {
                "family": family,
                "component": component,
                "control_id": control_id,
                "p10_state": "intact",
                "register": register,
                "world": "register_p3",
            }
            table_hash = sha256(label.encode("ascii")).hexdigest()
            seed = 20261000 + int(table_hash[:8], 16) % 100_000_000

            def endpoint(value: float, offset: int):
                return {
                    "estimate": value,
                    "ci": [value, value],
                    "n_bootstrap": 999,
                    "seed": seed + offset,
                }

            gates = {
                "native_source_positive": True,
                "phase_blind_crossed_positive": True,
                "bridge_negative": True,
                "ac_at_invariant": True,
                "protected_base_health": True,
                "masked_accuracy_health": True,
                "donor_margin_health": True,
                "acceptor_margin_health": True,
                "numerical_complete": True,
            }
            tables[label] = {
                "spec": spec,
                "table_sha256": table_hash,
                "inference_seed": seed,
                "raw_blocks": raw_blocks,
                "endpoints": {
                    "native_source_ac": endpoint(1.0, 1),
                    "phase_blind_crossed_ac": endpoint(1.0, 2),
                    "fixed_bridge_effect": endpoint(0.0, 3),
                    "phase_blind_crossed_ac_minus_at": endpoint(0.0, 4),
                    "protected_true_base_log_probability_delta": endpoint(0.0, 5),
                    "donor_destruction_response": endpoint(1.0, 6),
                    "acceptor_destruction_response": endpoint(1.0, 7),
                },
                "protected_accuracy": {
                    "intact_correct": 120,
                    "table_correct": 120,
                    "total": 120,
                    "intact_accuracy": 1.0,
                    "table_accuracy": 1.0,
                    "retention": 1.0,
                },
                "destruction_response_retention": {
                    "donor": 1.0,
                    "acceptor": 1.0,
                    "baseline_donor": 1.0,
                    "baseline_acceptor": 1.0,
                },
                "gates": gates,
                "pass": True,
            }
            states[label] = {
                "spec": spec,
                "phase2_table_health_pass": True,
                "p10_reader_score": 1.0,
                "p10_reader_retention": 0.95,
                "passed": True,
            }
    controls = {
        label: row
        for label, row in states.items()
        if row["spec"]["family"] != "target"
    }
    return {
        "schema": "gfmcd.pbc.p3_register_table_health.v1",
        "stage": "development",
        "authorized_table_health": {
            "schema": "gfmcd.pbc.carrier_factorial_table_health.v1",
            "target_mode": "REGISTER_P3_ONLY",
            "candidate_package_sha256": "b" * 64,
            "n_blocks": 120,
            "n_tables": len(tables),
            "tables": tables,
            "all_tables_pass": True,
        },
        "baseline_p10_reader_score": 1.0,
        "states": states,
        "control_states": controls,
        "all_control_states_passed": True,
        "all_required_states_passed": True,
        "minimum_p10_reader_retention": 0.95,
        "joint_state_passed": False,
    }


def _rows(*, failed_common_control: bool = False):
    rows = []
    labels = ("p3", *ANALYZE.CONTROL_LABELS, *ANALYZE.DIAGNOSTIC_LABELS)
    for label_index, component in enumerate(labels):
        for block in range(120):
            chrom = f"chr{1 + block % 2}"
            strand = "+" if (block // 2) % 2 == 0 else "-"
            centered = (block - 59.5) * 0.0001
            for register in REGISTERS:
                if component == "p3":
                    effect = 0.30 + centered
                elif component == "p3_common_mode" and failed_common_control:
                    effect = 0.30 + centered
                elif component in ANALYZE.DIAGNOSTIC_LABELS:
                    effect = {-1: -0.02, 0: 0.04, 1: -0.01}[register] + centered / 10
                else:
                    effect = 0.01 + label_index * 0.00001 - centered / 3
                for donor in range(3):
                    for acceptor in range(3):
                        delta = phase_delta(donor, acceptor)
                        coupling = effect if delta == preferred_delta(register) else 0.0
                        rows.append(
                            {
                                "component": component,
                                "control_family": (
                                    "target"
                                    if component == "p3"
                                    else (
                                        "signed_hidden_axis"
                                        if component in ANALYZE.CONTROL_LABELS
                                        else (
                                            "common_mode"
                                            if component == "p3_common_mode"
                                            else "decoy_span"
                                        )
                                    )
                                ),
                                "p10_state": "intact",
                                "register": register,
                                "block_id": f"b{block:03d}",
                                "chrom": chrom,
                                "strand": strand,
                                "donor_phase": donor,
                                "acceptor_phase": acceptor,
                                "phase_delta": delta,
                                "intact_margin": 1.0 + coupling,
                                "corrupt_margin": 1.0,
                                "coupling": coupling,
                            }
                        )
    return rows


def _raw(*, failed_common_control: bool = False):
    ids = [f"b{block:03d}" for block in range(120)]
    return {
        "schema": REGISTER_PREFLIGHT_SCHEMA,
        "status": RUN.RAW_STATUS,
        "stage": "development",
        "mode": "P3_ONLY",
        "split_role": "development",
        "n_blocks": 120,
        "n_cells": 1080,
        "selection": {
            "method": "sha256_rank_whole_block_identity",
            "salt": RUN.DEVELOPMENT_SELECTION_SALT,
            "candidate_blocks": 120,
            "selected_blocks": 120,
            "selected_block_ids": ids,
            "selected_block_ids_sha256": "a" * 64,
        },
        "authorization": {
            "phase2_p3_authorized": True,
            "phase2_joint_authorized": False,
            "jreg_direction": None,
        },
        "construction_binding": {
            "candidate_package_canonical_sha256": "b" * 64,
            "candidate_content_sha256": "c" * 64,
            "candidate_audit_status": "AUDIT_PASS",
        },
        "design": {
            "p3_bins": list(range(339, 346)),
            "p10_bins": list(range(94, 101)),
            "registers": list(REGISTERS),
            "direction": "right",
            "span_token": [393, 513],
            "decoy_span_token": [153, 273],
            "control_seeds": list(REGISTER_CONTROL_SEEDS),
            "p10_attenuation_dose": None,
            "candidate_package_canonical_sha256": "b" * 64,
        },
        "mechanics": _mechanics(),
        "health": _health(),
        "model": {"checkpoint": "synthetic"},
        "rows": _rows(failed_common_control=failed_common_control),
    }


def _analyze(raw, *, path=True):
    return ANALYZE.analyze_preflight(
        raw,
        preflight_sha256="a" * 64,
        path_mde_evidence=_path_mde() if path else None,
        path_mde_audit=_path_mde_audit() if path else None,
        fixture_calibration=_fixture_calibration(),
        seed=29,
        n_bootstrap=99,
    )


def test_positive_preflight_authorizes_only_after_all_gates_pass():
    result = _analyze(_raw())
    assert result["schema"] == REGISTER_PREFLIGHT_ANALYSIS_SCHEMA
    assert result["status"] == "P3_REGISTER_PREFLIGHT_PASS"
    assert result["authorization"]["authorize_p3_register_reserve"]
    assert not result["authorization"]["authorize_joint_register_reserve"]
    assert result["development_relocation"]["status"] == "P3_REGISTER_RELOCATION_SUPPORTED"
    assert result["partial_specificity_seeds"] == {
        "expected": 2026081010,
        "opposite": 2026081012,
    }
    assert (
        result["development_relocation"]["partial_specificity_seeds"]
        == result["partial_specificity_seeds"]
    )


def test_missing_health_is_instrument_unresolved():
    raw = _raw()
    raw["health"] = {}
    result = _analyze(raw)
    assert result["status"] == "P3_REGISTER_INSTRUMENT_UNRESOLVED"
    assert not result["gates"]["health_and_p10_retention"]


def test_named_control_that_reproduces_relocation_fails_preflight():
    result = _analyze(_raw(failed_common_control=True))
    assert result["status"] == "P3_REGISTER_INSTRUMENT_UNRESOLVED"
    assert not result["gates"]["development_relocation"]
    assert not result["development_relocation"]["specificity"]["passed"]


def test_missing_path_mde_is_explicitly_unresolved():
    result = _analyze(_raw(), path=False)
    assert result["status"] == "P3_REGISTER_INSTRUMENT_UNRESOLVED"
    assert result["projected_resolution"]["path_evidence"]["status"] == "MISSING_UNRESOLVED"
    assert not result["resolution"]["D_path"]["passed"]
    assert not result["resolution"]["R_path"]["passed"]


def test_independent_audit_recomputes_and_rejects_hash_bound_tampering(tmp_path):
    raw = _raw()
    candidate_package = {"content_sha256": "c" * 64}
    candidate_hash = sha256(
        json.dumps(
            candidate_package,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    raw["construction_binding"]["candidate_package_canonical_sha256"] = candidate_hash
    raw["design"]["candidate_package_canonical_sha256"] = candidate_hash
    raw["health"]["authorized_table_health"][
        "candidate_package_sha256"
    ] = candidate_hash
    source_payloads = {
        "candidate_package": candidate_package,
        "candidate_construction_audit": {
            "schema": "gfmcd.positional_boundary_coherence.candidate_census_audit.v1",
            "stage": "construction",
            "status": "AUDIT_PASS",
            "passed": True,
            "checks": {"independent_reconstruction": True},
            "candidate_census_sha256": candidate_hash,
            "candidate_content_sha256": "c" * 64,
        },
    }
    for name in (
        "cohort",
        "candidate_package",
        "candidate_construction_audit",
        "phase2_analysis",
        "phase2_audit",
        "checkpoint",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(source_payloads.get(name, {})) + "\n")
        raw.setdefault("sources", {})[name] = file_record(path)
    raw_path = tmp_path / "preflight.json"
    raw_path.write_text(json.dumps(raw, sort_keys=True) + "\n")
    raw_hash = sha256_file(raw_path)
    analysis = ANALYZE.analyze_preflight(
        raw,
        preflight_sha256=raw_hash,
        path_mde_evidence=_path_mde(),
        path_mde_audit=_path_mde_audit(),
        fixture_calibration=_fixture_calibration(),
        seed=31,
        n_bootstrap=99,
    )
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, sort_keys=True) + "\n")
    audit = AUDIT.audit_payloads(
        raw,
        analysis,
        preflight_sha256=raw_hash,
        analysis_sha256=sha256_file(analysis_path),
    )
    assert audit["schema"] == REGISTER_PREFLIGHT_AUDIT_SCHEMA
    assert audit["status"] == "PASS"
    assert audit["analysis_sha256"] == sha256_file(analysis_path)
    assert audit["partial_specificity_seeds"] == analysis[
        "partial_specificity_seeds"
    ]

    tampered_seeds = json.loads(json.dumps(analysis))
    tampered_seeds["partial_specificity_seeds"]["opposite"] += 1
    seed_audit = AUDIT.audit_payloads(
        raw,
        tampered_seeds,
        preflight_sha256=raw_hash,
        analysis_sha256="d" * 64,
    )
    assert seed_audit["status"] == "FAIL"
    assert not seed_audit["checks"]["analysis_exact_recompute"]

    analysis["status"] = "P3_REGISTER_INSTRUMENT_UNRESOLVED"
    tampered = AUDIT.audit_payloads(
        raw,
        analysis,
        preflight_sha256=raw_hash,
        analysis_sha256="b" * 64,
    )
    assert tampered["status"] == "FAIL"
    assert not tampered["checks"]["analysis_exact_recompute"]

    raw["rows"][0]["coupling"] += 0.2
    raw_tamper = AUDIT.audit_payloads(
        raw,
        _analyze(_raw()),
        preflight_sha256="a" * 64,
        analysis_sha256="c" * 64,
    )
    assert raw_tamper["status"] == "FAIL"
    assert not raw_tamper["checks"]["exact_development_lattice"]
