#!/usr/bin/env python
"""Independent raw-lattice, arithmetic, gate, hash, and analysis audit."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_register_relocation import (  # noqa: E402
    OPPOSITE_PARTIAL_SPECIFICITY_SEED,
    PARTIAL_SPECIFICITY_SEED,
    REGISTERS,
    phase_delta,
)
from gfmcd.splicing.pbc_register_stage import (  # noqa: E402
    DECOY_SPAN_TOKEN,
    P10_BINS,
    P3_BINS,
    REGISTER_CONTROL_SEEDS,
    REGISTER_DIRECTION,
    REGISTER_PREFLIGHT_ANALYSIS_SCHEMA,
    REGISTER_PREFLIGHT_AUDIT_SCHEMA,
    REGISTER_PREFLIGHT_SCHEMA,
    RESERVE_BLOCKS,
    SPAN_TOKEN,
    manifest_binds_path,
    manifest_precedes,
    projected_mde80,
    validate_sibling_manifest,
)
from analyze_p3_register_relocation import (  # noqa: E402
    CONTROL_LABELS,
    DIAGNOSTIC_LABELS,
)
from audit_p3_register_relocation import _independent_recompute  # noqa: E402
from run_p3_register_preflight import (  # noqa: E402
    DEVELOPMENT_BLOCKS,
    DEVELOPMENT_SELECTION_SALT,
    RAW_STATUS,
)


FIXTURE_SCHEMA = "gfmcd.pbc.p3_register_fixture_calibration.v1"
PATH_MDE_SCHEMA = "gfmcd.pbc.p3_register_path_mde.v1"
PATH_MDE_AUDIT_SCHEMA = "gfmcd.pbc.p3_register_path_mde_audit.v1"
FIXTURE_CASES = (
    "planted_positive",
    "opposite_handed",
    "no_ramp",
    "common_mode",
    "wrong_support",
    "null",
)
UNRESOLVED_MDE = 1_000_000.0


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    if output == manifest:
        raise ValueError("audit and manifest paths must be distinct")
    return output, manifest


def _file_record_valid(value: object) -> bool:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        return False
    path = Path(str(value["path"])).expanduser()
    return bool(
        path.is_file()
        and type(value.get("size_bytes")) is int
        and path.stat().st_size == int(value["size_bytes"])
        and value.get("sha256") == sha256_file(path)
    )


def _construction_source_binding(payload: Mapping[str, Any]) -> bool:
    """Independently bind the consumed census and construction audit to raw output."""

    sources = payload.get("sources")
    binding = payload.get("construction_binding")
    if not isinstance(sources, Mapping) or not isinstance(binding, Mapping):
        return False
    package_record = sources.get("candidate_package")
    audit_record = sources.get("candidate_construction_audit")
    if not _file_record_valid(package_record) or not _file_record_valid(audit_record):
        return False
    try:
        package = json.loads(Path(str(package_record["path"])).read_text())
        audit = json.loads(Path(str(audit_record["path"])).read_text())
        encoded = json.dumps(
            package, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        package_hash = sha256(encoded).hexdigest()
        checks = audit.get("checks")
        return bool(
            isinstance(package, Mapping)
            and isinstance(audit, Mapping)
            and audit.get("schema")
            == "gfmcd.positional_boundary_coherence.candidate_census_audit.v1"
            and audit.get("stage") == "construction"
            and audit.get("status") == "AUDIT_PASS"
            and audit.get("passed") is True
            and isinstance(checks, Mapping)
            and checks
            and all(value is True for value in checks.values())
            and audit.get("candidate_census_sha256") == package_hash
            and audit.get("candidate_content_sha256")
            == package.get("content_sha256")
            and binding.get("candidate_package_canonical_sha256") == package_hash
            and binding.get("candidate_content_sha256")
            == package.get("content_sha256")
            and binding.get("candidate_audit_status") == "AUDIT_PASS"
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _raw_row_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return {"passed": False, "reason": "raw rows absent"}
    labels = {"p3", *CONTROL_LABELS, *DIAGNOSTIC_LABELS}
    states = {"intact", "attenuated"} if payload.get("mode") == "JOINT" else {"intact"}
    blocks: set[str] = set()
    keys: set[tuple[str, str, int, str, int, int]] = set()
    arithmetic_max = 0.0
    finite = True
    phase_valid = True
    family_valid = True
    expected_family = {
        "p3": "target",
        **{label: "signed_hidden_axis" for label in CONTROL_LABELS},
        "p3_common_mode": "common_mode",
        "p3_decoy_span": "decoy_span",
    }
    try:
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("non-object raw row")
            component = str(row["component"])
            state = str(row["p10_state"])
            register = int(row["register"])
            block = str(row["block_id"])
            donor = int(row["donor_phase"])
            acceptor = int(row["acceptor_phase"])
            key = (component, state, register, block, donor, acceptor)
            values = (
                float(row["intact_margin"]),
                float(row["corrupt_margin"]),
                float(row["coupling"]),
            )
            finite &= all(math.isfinite(value) for value in values)
            arithmetic_max = max(arithmetic_max, abs(values[2] - (values[0] - values[1])))
            phase_valid &= int(row["phase_delta"]) == phase_delta(donor, acceptor)
            family_valid &= str(row.get("control_family")) == expected_family.get(component)
            blocks.add(block)
            keys.add(key)
    except (KeyError, TypeError, ValueError) as exc:
        return {"passed": False, "reason": f"invalid raw row: {exc}"}
    expected = {
        (component, state, register, block, donor, acceptor)
        for component in labels
        for state in states
        for register in REGISTERS
        for block in blocks
        for donor in range(3)
        for acceptor in range(3)
    }
    passed = bool(
        len(blocks) == DEVELOPMENT_BLOCKS
        and len(keys) == len(rows)
        and keys == expected
        and finite
        and phase_valid
        and family_valid
        and arithmetic_max <= 1e-8
    )
    return {
        "passed": passed,
        "n_blocks": len(blocks),
        "n_rows": len(rows),
        "unique_keys": len(keys),
        "expected_rows": len(expected),
        "all_finite": bool(finite),
        "phase_convention_valid": bool(phase_valid),
        "control_family_valid": bool(family_valid),
        "coupling_arithmetic_max_abs": float(arithmetic_max),
    }


def _design_contract(payload: Mapping[str, Any]) -> bool:
    design = payload.get("design")
    construction = payload.get("construction_binding")
    if not isinstance(design, Mapping) or not isinstance(construction, Mapping):
        return False
    candidate_hash = str(
        construction.get("candidate_package_canonical_sha256", "")
    )
    content_hash = str(construction.get("candidate_content_sha256", ""))
    return bool(
        design.get("p3_bins") == list(P3_BINS)
        and design.get("p10_bins") == list(P10_BINS)
        and design.get("registers") == list(REGISTERS)
        and design.get("direction") == REGISTER_DIRECTION
        and design.get("span_token") == list(SPAN_TOKEN)
        and design.get("decoy_span_token") == list(DECOY_SPAN_TOKEN)
        and design.get("control_seeds") == list(REGISTER_CONTROL_SEEDS)
        and design.get("candidate_package_canonical_sha256") == candidate_hash
        and construction.get("candidate_audit_status") == "AUDIT_PASS"
        and len(candidate_hash) == 64
        and len(content_hash) == 64
        and all(
            character in "0123456789abcdef"
            for character in candidate_hash + content_hash
        )
    )


def _selection_contract(payload: Mapping[str, Any]) -> bool:
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        return False
    block_ids = selection.get("selected_block_ids")
    return bool(
        selection.get("method") == "sha256_rank_whole_block_identity"
        and selection.get("salt") == DEVELOPMENT_SELECTION_SALT
        and int(selection.get("selected_blocks", -1)) == DEVELOPMENT_BLOCKS
        and isinstance(block_ids, list)
        and len(block_ids) == DEVELOPMENT_BLOCKS
        and len(set(str(value) for value in block_ids)) == DEVELOPMENT_BLOCKS
    )


def _mechanics_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    mechanics = payload.get("mechanics")
    if not isinstance(mechanics, Mapping):
        return {"passed": False, "reason": "mechanics absent"}
    construction = mechanics.get("construction_invariants")
    handedness = mechanics.get("handedness")
    expected_delta = {"-1": 1, "0": 0, "1": 2}
    handedness_contract = bool(
        mechanics.get("direction") == REGISTER_DIRECTION
        and isinstance(handedness, Mapping)
        and set(handedness) == set(expected_delta)
        and all(
            type(handedness[key].get("passed")) is bool
            and handedness[key].get("passed")
            == bool(
                int(handedness[key].get("predicted_annotated_phase_delta", -1)) == value
                and abs(
                    float(handedness[key].get("measured_acceptor_phase_radians", math.inf))
                    - float(handedness[key].get("requested_acceptor_phase_radians", -math.inf))
                )
                <= 1e-6
            )
            for key, value in expected_delta.items()
            if isinstance(handedness[key], Mapping)
        )
    )
    handedness_scientific_pass = bool(
        handedness_contract
        and all(bool(handedness[key]["passed"]) for key in expected_delta)
    )
    numeric = {
        "register_zero_no_hook_max_abs": float(
            mechanics.get("register_zero_no_hook_max_abs", math.inf)
        ),
        "order_replay_max_abs": float(mechanics.get("order_replay_max_abs", math.inf)),
        "p10_independent_reconstruction_relative_max_abs": float(
            mechanics.get("p10_independent_reconstruction_relative_max_abs", math.inf)
        ),
        "p10_explicit_component_retention_relative_max_abs": float(
            mechanics.get("p10_explicit_component_retention_relative_max_abs", math.inf)
        ),
        "pointwise_non_p3_residual_max_abs": float(
            mechanics.get("pointwise_non_p3_residual_max_abs", math.inf)
        ),
        "rms_match_max_relative_error": float(
            mechanics.get("rms_match_max_relative_error", math.inf)
        ),
    }
    recomputed_scientific_pass = bool(
        isinstance(construction, Mapping)
        and construction.get("passed") is True
        and handedness_scientific_pass
        and numeric["register_zero_no_hook_max_abs"] <= 1e-8
        and numeric["order_replay_max_abs"] <= 1e-8
        and numeric["p10_independent_reconstruction_relative_max_abs"] <= 1e-10
        and numeric["p10_explicit_component_retention_relative_max_abs"] <= 1e-10
        and numeric["pointwise_non_p3_residual_max_abs"] <= 1e-10
        and numeric["rms_match_max_relative_error"] <= 1e-6
    )
    contract_valid = bool(
        isinstance(construction, Mapping)
        and type(construction.get("passed")) is bool
        and handedness_contract
        and type(mechanics.get("handedness_passed")) is bool
        and mechanics.get("handedness_passed") == handedness_scientific_pass
        and all(math.isfinite(value) for value in numeric.values())
        and type(mechanics.get("passed")) is bool
        and mechanics.get("passed") == recomputed_scientific_pass
    )
    return {
        "passed": contract_valid,
        "reported_scientific_pass": mechanics.get("passed") is True,
        "independent_scientific_gate_recompute": recomputed_scientific_pass,
        "handedness_contract_valid": handedness_contract,
        "handedness_scientific_pass": handedness_scientific_pass,
        **numeric,
    }


def _health_interval(
    values: list[float], strata: list[str], *, seed: int
) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    labels = np.asarray(strata, dtype=object)
    if data.shape != labels.shape or len(data) != DEVELOPMENT_BLOCKS:
        raise ValueError("health endpoint does not use the 120-block support")
    if np.all(data == data[0]):
        lower = upper = float(data[0])
    else:
        groups = [np.flatnonzero(labels == label) for label in sorted(set(labels))]
        rng = np.random.default_rng(seed)
        draws = np.empty(999, dtype=float)
        for draw in range(999):
            indices = np.concatenate(
                [rng.choice(group, size=len(group), replace=True) for group in groups]
            )
            draws[draw] = float(data[indices].mean())
        lower = float(np.quantile(draws, 0.025))
        upper = float(np.quantile(draws, 0.975))
    return {
        "estimate": float(data.mean()),
        "ci": [lower, upper],
        "n_bootstrap": 999,
        "seed": seed,
    }


def _recompute_health_table(row: Mapping[str, Any]) -> dict[str, Any]:
    blocks = row.get("raw_blocks")
    if not isinstance(blocks, list) or len(blocks) != DEVELOPMENT_BLOCKS:
        raise ValueError("health table lacks 120 primitive block rows")
    block_ids = [str(block.get("block_id")) for block in blocks]
    strata = [str(block.get("stratum")) for block in blocks]
    if len(set(block_ids)) != DEVELOPMENT_BLOCKS:
        raise ValueError("health table repeats a block")
    seed = int(row["inference_seed"])

    def endpoint(name: str, offset: int) -> dict[str, Any]:
        return _health_interval(
            [float(block[name]) for block in blocks], strata, seed=seed + offset
        )

    native = endpoint("native_source_ac", 1)
    crossed = endpoint("crossed_ac", 2)
    bridge = endpoint("bridge_effect", 3)
    ac_at = endpoint("crossed_ac_minus_at", 4)
    protected = endpoint("protected_log_probability_delta", 5)
    donor = endpoint("donor_destruction_response", 6)
    acceptor = endpoint("acceptor_destruction_response", 7)
    intact_correct = sum(int(block["protected_intact_correct"]) for block in blocks)
    table_correct = sum(int(block["protected_table_correct"]) for block in blocks)
    total = sum(int(block["protected_total"]) for block in blocks)
    intact_accuracy = intact_correct / total if total else 0.0
    table_accuracy = table_correct / total if total else 0.0
    accuracy_retention = table_accuracy / intact_accuracy if intact_accuracy > 0 else 0.0
    baseline_donor = float(
        np.mean([float(block["baseline_donor_destruction_response"]) for block in blocks])
    )
    baseline_acceptor = float(
        np.mean([float(block["baseline_acceptor_destruction_response"]) for block in blocks])
    )
    donor_retention = donor["estimate"] / baseline_donor if baseline_donor > 0 else -math.inf
    acceptor_retention = (
        acceptor["estimate"] / baseline_acceptor if baseline_acceptor > 0 else -math.inf
    )
    gates = {
        "native_source_positive": native["ci"][0] > 0.05,
        "phase_blind_crossed_positive": crossed["ci"][0] > 0.02,
        "bridge_negative": bridge["ci"][0] >= -0.05 and bridge["ci"][1] <= 0.05,
        "ac_at_invariant": ac_at["ci"][0] >= -0.05 and ac_at["ci"][1] <= 0.05,
        "protected_base_health": protected["ci"][0] > -0.10,
        "masked_accuracy_health": accuracy_retention >= 0.80,
        "donor_margin_health": donor["ci"][0] > 0.50 and donor_retention >= 0.80,
        "acceptor_margin_health": acceptor["ci"][0] > 0.50
        and acceptor_retention >= 0.80,
        "numerical_complete": all(block.get("numerical_complete") is True for block in blocks),
    }
    return {
        "endpoints": {
            "native_source_ac": native,
            "phase_blind_crossed_ac": crossed,
            "fixed_bridge_effect": bridge,
            "phase_blind_crossed_ac_minus_at": ac_at,
            "protected_true_base_log_probability_delta": protected,
            "donor_destruction_response": donor,
            "acceptor_destruction_response": acceptor,
        },
        "protected_accuracy": {
            "intact_correct": intact_correct,
            "table_correct": table_correct,
            "total": total,
            "intact_accuracy": intact_accuracy,
            "table_accuracy": table_accuracy,
            "retention": accuracy_retention,
        },
        "destruction_response_retention": {
            "donor": float(donor_retention),
            "acceptor": float(acceptor_retention),
            "baseline_donor": baseline_donor,
            "baseline_acceptor": baseline_acceptor,
        },
        "gates": gates,
        "pass": bool(all(gates.values())),
    }


def _health_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    health = payload.get("health")
    construction = payload.get("construction_binding")
    if not isinstance(health, Mapping):
        return {"passed": False, "reason": "full table health artifact absent"}
    authorized = health.get("authorized_table_health")
    states = health.get("states")
    if not isinstance(authorized, Mapping) or not isinstance(states, Mapping):
        return {"passed": False, "reason": "authorized table health rows absent"}
    mode = str(payload.get("mode"))
    p10_states = ("intact", "attenuated") if mode == "JOINT" else ("intact",)
    components = (
        "p3",
        *(f"p3_signed_{index:02d}" for index in range(31)),
        "p3_common_mode",
        "p3_decoy_span",
    )
    expected = {
        f"{component}__{state}__r{register}"
        for component in components
        for state in p10_states
        for register in REGISTERS
    }
    tables = authorized.get("tables")
    if not isinstance(tables, Mapping) or set(tables) != expected or set(states) != expected:
        return {"passed": False, "reason": "health table family is incomplete"}
    recomputed = {}
    contract = True
    try:
        for label in sorted(expected):
            table_row = tables[label]
            state = states[label]
            if not isinstance(table_row, Mapping) or not isinstance(state, Mapping):
                raise TypeError("health table/state row is not an object")
            component, p10_state, register_text = label.split("__")
            register = int(register_text[1:])
            family = (
                "target"
                if component == "p3"
                else "signed_hidden_axis"
                if component.startswith("p3_signed_")
                else "common_mode"
                if component == "p3_common_mode"
                else "decoy_span"
            )
            spec = table_row.get("spec")
            expected_spec = {
                "family": family,
                "component": component,
                "control_id": (
                    int(component.rsplit("_", 1)[1]) + 1
                    if component.startswith("p3_signed_")
                    else None
                ),
                "p10_state": p10_state,
                "register": register,
                "world": "register_p3" if p10_state == "intact" else "register_joint",
            }
            table_hash = str(table_row.get("table_sha256", ""))
            inference_seed = int(table_row.get("inference_seed", -1))
            table_identity = bool(
                len(table_hash) == 64
                and all(character in "0123456789abcdef" for character in table_hash)
                and inference_seed
                == 20261000 + int(table_hash[:8], 16) % 100_000_000
            )
            local = _recompute_health_table(table_row)
            recomputed[label] = local
            retention = float(state["p10_reader_retention"])
            contract &= bool(
                spec == expected_spec
                and state.get("spec") == expected_spec
                and table_identity
                and all(table_row.get(key) == value for key, value in local.items())
                and state.get("phase2_table_health_pass") == local["pass"]
                and math.isfinite(retention)
                and state.get("passed") == bool(local["pass"] and retention >= 0.80)
            )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"passed": False, "reason": f"health recomputation failed: {exc}"}
    retentions = [float(states[label]["p10_reader_retention"]) for label in expected]
    minimum = min(retentions)
    controls = {
        label: row for label, row in states.items() if row["spec"]["family"] != "target"
    }
    candidate_hash = (
        str(construction.get("candidate_package_canonical_sha256", ""))
        if isinstance(construction, Mapping)
        else ""
    )
    contract &= bool(
        health.get("schema") == "gfmcd.pbc.p3_register_table_health.v1"
        and health.get("stage") == "development"
        and authorized.get("schema") == "gfmcd.pbc.carrier_factorial_table_health.v1"
        and authorized.get("target_mode")
        == ("REGISTER_JOINT" if mode == "JOINT" else "REGISTER_P3_ONLY")
        and int(authorized.get("n_blocks", -1)) == DEVELOPMENT_BLOCKS
        and int(authorized.get("n_tables", -1)) == len(expected)
        and authorized.get("candidate_package_sha256") == candidate_hash
        and len(candidate_hash) == 64
        and all(character in "0123456789abcdef" for character in candidate_hash)
        and authorized.get("all_tables_pass")
        == all(row["pass"] for row in recomputed.values())
        and health.get("control_states") == controls
        and health.get("all_control_states_passed")
        == all(row["passed"] for row in controls.values())
        and health.get("all_required_states_passed")
        == all(row["passed"] for row in states.values())
        and abs(float(health.get("minimum_p10_reader_retention", math.inf)) - minimum)
        <= 1e-12
    )
    scientific = bool(contract and all(row["passed"] for row in states.values()))
    return {
        "passed": contract,
        "scientific_pass": scientific,
        "n_tables": len(expected),
        "recomputed_minimum": minimum,
        "all_inherited_gates_recomputed": True,
    }


def _path_resolution_independent(
    evidence: object,
    audit: object,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Validate path-power evidence without calling the preflight analyzer."""

    if not isinstance(evidence, Mapping) or not isinstance(audit, Mapping):
        return (
            {
                "D_path": UNRESOLVED_MDE,
                "R_path": UNRESOLVED_MDE,
                "Q_path": UNRESOLVED_MDE,
            },
            {
                "status": "MISSING_UNRESOLVED",
                "reason": "audited development path D/R/Q MDE evidence was not supplied",
                "required_schema": PATH_MDE_SCHEMA,
                "required_audit_schema": PATH_MDE_AUDIT_SCHEMA,
            },
        )
    values = evidence.get("mde80")
    valid = bool(
        evidence.get("schema") == PATH_MDE_SCHEMA
        and evidence.get("status") == "COMPLETE"
        and int(evidence.get("reserve_blocks", -1)) == RESERVE_BLOCKS
        and isinstance(values, Mapping)
        and set(values) >= {"D_path", "R_path", "Q_path"}
        and isinstance(evidence.get("self_sha256"), str)
        and len(str(evidence.get("self_sha256"))) == 64
        and isinstance(evidence.get("candidate_freeze_sha256"), str)
        and len(str(evidence.get("candidate_freeze_sha256"))) == 64
        and audit.get("schema") == PATH_MDE_AUDIT_SCHEMA
        and audit.get("status") == "PASS"
        and audit.get("path_mde_sha256") == evidence.get("self_sha256")
        and audit.get("candidate_freeze_sha256")
        == evidence.get("candidate_freeze_sha256")
        and audit.get("candidate_freeze_validation_passed") is True
    )
    parsed: dict[str, float] = {}
    if valid:
        try:
            parsed = {
                name: float(values[name])
                for name in ("D_path", "R_path", "Q_path")
            }
            valid = all(math.isfinite(value) and value >= 0 for value in parsed.values())
        except (TypeError, ValueError):
            valid = False
    if not valid:
        return (
            {
                "D_path": UNRESOLVED_MDE,
                "R_path": UNRESOLVED_MDE,
                "Q_path": UNRESOLVED_MDE,
            },
            {
                "status": "INVALID_UNRESOLVED",
                "reason": "path MDE evidence failed its frozen schema or numeric contract",
                "provided": dict(evidence),
                "audit": dict(audit),
            },
        )
    return parsed, {
        "status": "COMPLETE",
        "provided": dict(evidence),
        "audit": dict(audit),
        "candidate_freeze_dependency_enforced": True,
    }


def _fixture_validation_independent(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"validation_passed": False, "all_required_cases_passed": False}
    cases = value.get("required_cases")
    cases_valid = bool(
        isinstance(cases, Mapping)
        and set(cases) == set(FIXTURE_CASES)
        and all(
            isinstance(cases[name], Mapping)
            and type(cases[name].get("passed")) is bool
            for name in FIXTURE_CASES
        )
    )
    valid = bool(
        value.get("schema") == FIXTURE_SCHEMA
        and value.get("analysis_engine")
        == "analyze_p3_register_relocation.analyze_payload"
        and value.get("decision_engine")
        == "pbc_register_relocation.adjudicate_register_relocation"
        and cases_valid
        and value.get("all_required_cases_passed") is True
        and math.isfinite(float(value.get("familywise_false_positive_rate", math.inf)))
        and float(value.get("familywise_false_positive_rate", math.inf)) <= 0.06
        and math.isfinite(float(value.get("corrected_power_at_0_075", -math.inf)))
        and float(value.get("corrected_power_at_0_075", -math.inf)) >= 0.80
        and int(value.get("replicates", -1)) == 1_000
        and int(value.get("bootstrap_draws_per_fixture", -1)) >= 99
    )
    normalized = dict(value)
    normalized["validation_passed"] = valid
    if not valid:
        normalized["all_required_cases_passed"] = False
    return normalized


def _expected_health_labels(mode: str) -> set[str]:
    p10_states = ("intact", "attenuated") if mode == "JOINT" else ("intact",)
    components = (
        "p3",
        *(f"p3_signed_{index:02d}" for index in range(31)),
        "p3_common_mode",
        "p3_decoy_span",
    )
    return {
        f"{component}__{state}__r{register}"
        for component in components
        for state in p10_states
        for register in REGISTERS
    }


def _structural_health_gate(health: Mapping[str, Any]) -> bool:
    states = health.get("states")
    authorized = health.get("authorized_table_health")
    target_mode = (
        str(authorized.get("target_mode"))
        if isinstance(authorized, Mapping)
        else ""
    )
    expected_count = (
        204
        if target_mode == "REGISTER_JOINT"
        else 102
        if target_mode == "REGISTER_P3_ONLY"
        else -1
    )
    tables = authorized.get("tables") if isinstance(authorized, Mapping) else None
    return bool(
        health.get("schema") == "gfmcd.pbc.p3_register_table_health.v1"
        and health.get("stage") == "development"
        and isinstance(states, Mapping)
        and len(states) == expected_count
        and all(
            isinstance(row, Mapping) and row.get("passed") is True
            for row in states.values()
        )
        and isinstance(authorized, Mapping)
        and authorized.get("schema")
        == "gfmcd.pbc.carrier_factorial_table_health.v1"
        and int(authorized.get("n_blocks", -1)) == DEVELOPMENT_BLOCKS
        and int(authorized.get("n_tables", -1)) == expected_count
        and authorized.get("all_tables_pass") is True
        and isinstance(tables, Mapping)
        and set(tables) == set(states)
        and all(
            isinstance(row, Mapping)
            and row.get("pass") is True
            and isinstance(row.get("raw_blocks"), list)
            and len(row["raw_blocks"]) == DEVELOPMENT_BLOCKS
            for row in tables.values()
        )
        and health.get("all_required_states_passed") is True
        and float(health.get("minimum_p10_reader_retention", -math.inf)) >= 0.80
    )


def _raw_capability_independent(
    preflight: Mapping[str, Any], mechanics: Mapping[str, Any]
) -> bool:
    health = preflight.get("health")
    construction = preflight.get("construction_binding")
    mode = str(preflight.get("mode"))
    if (
        mode not in {"P3_ONLY", "JOINT"}
        or not isinstance(health, Mapping)
        or not isinstance(construction, Mapping)
    ):
        return False
    states = health.get("states")
    authorized = health.get("authorized_table_health")
    tables = authorized.get("tables") if isinstance(authorized, Mapping) else None
    expected = _expected_health_labels(mode)
    candidate_hash = str(
        construction.get("candidate_package_canonical_sha256", "")
    )
    return bool(
        mechanics.get("passed") is True
        and float(mechanics.get("register_zero_no_hook_max_abs", math.inf)) <= 1e-8
        and float(mechanics.get("order_replay_max_abs", math.inf)) <= 1e-8
        and construction.get("candidate_audit_status") == "AUDIT_PASS"
        and len(candidate_hash) == 64
        and all(character in "0123456789abcdef" for character in candidate_hash)
        and _structural_health_gate(health)
        and isinstance(states, Mapping)
        and set(states) == expected
        and isinstance(authorized, Mapping)
        and authorized.get("target_mode")
        == ("REGISTER_JOINT" if mode == "JOINT" else "REGISTER_P3_ONLY")
        and authorized.get("candidate_package_sha256") == candidate_hash
        and isinstance(tables, Mapping)
        and set(tables) == expected
        and health.get("all_control_states_passed") is True
    )


def _stage_decision_independent(
    *,
    relocation_status: str,
    mechanics: Mapping[str, Any],
    health: Mapping[str, Any],
    resolution: Mapping[str, float],
    fixture: Mapping[str, Any],
    phase2: Mapping[str, Any],
) -> dict[str, Any]:
    """Reapply the frozen gate hierarchy without importing its verdict helper."""

    required = (
        "omega3_intact",
        "s_-1_intact",
        "s_0_intact",
        "s_1_intact",
        "D_path",
        "R_path",
        "Q_path",
    )
    resolution_rows = {
        name: {
            "mde80": float(resolution.get(name, math.inf)),
            "threshold": 0.05,
            "passed": bool(
                math.isfinite(float(resolution.get(name, math.inf)))
                and float(resolution.get(name, math.inf)) <= 0.05
            ),
        }
        for name in required
    }
    mechanics_pass = bool(
        mechanics.get("passed") is True
        and float(mechanics.get("register_zero_no_hook_max_abs", math.inf)) <= 1e-8
        and float(mechanics.get("order_replay_max_abs", math.inf)) <= 1e-8
    )
    health_pass = _structural_health_gate(health)
    fixture_pass = bool(
        fixture.get("all_required_cases_passed") is True
        and float(fixture.get("familywise_false_positive_rate", math.inf)) <= 0.06
        and float(fixture.get("corrected_power_at_0_075", -math.inf)) >= 0.80
    )
    p3 = phase2.get("phase2_p3_authorized")
    joint_phase2 = phase2.get("phase2_joint_authorized")
    if type(p3) is not bool or type(joint_phase2) is not bool:
        raise ValueError("Phase-II authorization flags are absent")
    relocation_pass = relocation_status in {
        "P3_REGISTER_RELOCATION_SUPPORTED",
        "P3_REGISTER_PARTIAL_CONCORDANCE_SUPPORTED",
    }
    mandatory = bool(
        p3
        and relocation_pass
        and mechanics_pass
        and health_pass
        and fixture_pass
        and all(row["passed"] for row in resolution_rows.values())
    )
    jreg = float(resolution.get("jreg", math.inf))
    joint = bool(
        mandatory
        and joint_phase2
        and math.isfinite(jreg)
        and jreg <= 0.05
        and health.get("joint_state_passed") is True
    )
    return {
        "schema": REGISTER_PREFLIGHT_ANALYSIS_SCHEMA,
        "stage": "development",
        "status": (
            "P3_REGISTER_PREFLIGHT_PASS"
            if mandatory
            else "P3_REGISTER_INSTRUMENT_UNRESOLVED"
        ),
        "authorization": {
            "authorize_p3_register_reserve": mandatory,
            "authorize_joint_register_reserve": joint,
            "p10_failure_does_not_veto_p3": True,
        },
        "gates": {
            "phase2_p3_authorized": bool(p3),
            "development_relocation": bool(relocation_pass),
            "construction_and_replay": mechanics_pass,
            "health_and_p10_retention": health_pass,
            "fixture_calibration": fixture_pass,
            "resolution": all(row["passed"] for row in resolution_rows.values()),
        },
        "resolution": resolution_rows,
        "optional_jreg": {
            "mde80": jreg,
            "passed": bool(math.isfinite(jreg) and jreg <= 0.05),
            "authorized": joint,
        },
    }


def _independent_analysis_checks(
    preflight: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    preflight_sha256: str,
) -> dict[str, bool]:
    """Recompute inference, power, and authorization without analyzer verdict code."""

    seed = int(analysis["inference_seed"])
    draws = int(analysis["bootstrap_draws"])
    raw_authorization = preflight.get("authorization")
    mechanics = preflight.get("mechanics")
    raw_health = preflight.get("health")
    if not all(
        isinstance(value, Mapping)
        for value in (raw_authorization, mechanics, raw_health)
    ):
        raise ValueError("preflight authorization/mechanics/health are incomplete")
    capability = _raw_capability_independent(preflight, mechanics)
    mode = str(preflight.get("mode"))
    target_like = {
        "schema": "gfmcd.pbc.p3_register_target.v1",
        "status": "COMPLETE",
        "stage": "development",
        "mode": mode,
        "authorization": {
            "phase2_p3_authorized": bool(
                raw_authorization.get("phase2_p3_authorized")
            ),
            "phase2_joint_authorized": bool(
                raw_authorization.get("phase2_joint_authorized")
            ),
            "register_preflight_passed": capability,
        },
        "design": {
            "jreg_direction": (
                raw_authorization.get("jreg_direction") if mode == "JOINT" else None
            )
        },
        "mechanics": {
            "register_zero_no_hook_max_abs": float(
                mechanics.get("register_zero_no_hook_max_abs", math.inf)
            )
        },
        "rows": preflight.get("rows"),
    }
    relocation_partial = _independent_recompute(
        target_like, analysis["development_relocation"]
    )
    reported_relocation = analysis.get("development_relocation")
    if not isinstance(reported_relocation, Mapping):
        raise ValueError("development relocation analysis is absent")
    relocation_keys = (
        "status",
        "stage",
        "mode",
        "partial_specificity_seeds",
        "target",
        "controls",
        "diagnostic_controls",
        "specificity",
        "register_zero_replay",
        "authorization",
    )
    relocation_exact = all(
        relocation_partial[key] == reported_relocation.get(key)
        for key in relocation_keys
    ) and reported_relocation.get("target_sha256") == preflight_sha256
    decision = reported_relocation.get("decision")
    relocation_exact &= bool(
        isinstance(decision, Mapping)
        and decision.get("code") == relocation_partial["status"]
        and reported_relocation.get("inference_seed") == seed
        and reported_relocation.get("bootstrap_draws") == draws
        and reported_relocation.get("partial_specificity_seeds")
        == {
            "expected": PARTIAL_SPECIFICITY_SEED,
            "opposite": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
        }
    )

    per_block = relocation_partial["target"]["per_block"]
    resolution = projected_mde80(
        per_block,
        endpoint_names=(
            "omega3_intact",
            "s_-1_intact",
            "s_0_intact",
            "s_1_intact",
        ),
        reserve_blocks=RESERVE_BLOCKS,
    )
    path_values, path_record = _path_resolution_independent(
        analysis.get("path_mde_evidence"), analysis.get("path_mde_audit")
    )
    resolution.update(path_values)
    if mode == "JOINT" and all("jreg" in row for row in per_block):
        resolution.update(
            projected_mde80(
                per_block,
                endpoint_names=("jreg",),
                reserve_blocks=RESERVE_BLOCKS,
            )
        )
    else:
        resolution["jreg"] = UNRESOLVED_MDE
    fixture = _fixture_validation_independent(analysis.get("fixture_calibration"))
    control_pass = raw_health.get("all_control_states_passed") is True
    stage_health = dict(raw_health)
    raw_states = raw_health.get("states")
    if isinstance(raw_states, Mapping):
        stage_health["states"] = {
            str(label): {
                **dict(row),
                "passed": bool(
                    isinstance(row, Mapping)
                    and row.get("passed") is True
                    and control_pass
                ),
            }
            for label, row in raw_states.items()
            if isinstance(row, Mapping)
        }
    decision_expected = _stage_decision_independent(
        relocation_status=str(relocation_partial["status"]),
        mechanics=mechanics,
        health=stage_health,
        resolution=resolution,
        fixture=fixture,
        phase2=raw_authorization,
    )
    stage_exact = all(
        analysis.get(key) == decision_expected[key]
        for key in (
            "schema",
            "stage",
            "status",
            "authorization",
            "gates",
            "resolution",
            "optional_jreg",
        )
    )
    projected_expected = {
        "reserve_blocks": RESERVE_BLOCKS,
        "mde80": resolution,
        "path_evidence": path_record,
        "missing_path_evidence_is_unresolved": True,
    }
    support_exact = bool(
        analysis.get("preflight_sha256") == preflight_sha256
        and analysis.get("construction_mechanics") == dict(mechanics)
        and analysis.get("partial_specificity_seeds")
        == {
            "expected": PARTIAL_SPECIFICITY_SEED,
            "opposite": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
        }
        and analysis.get("health") == stage_health
        and analysis.get("projected_resolution") == projected_expected
        and analysis.get("fixture_calibration") == fixture
        and analysis.get("outcome_firewall")
        == {
            "phase3_reserve_read": False,
            "phase3_reserve_authorized_only_by_this_analysis_and_independent_audit": True,
        }
    )
    return {
        "relocation_statistics_independent": bool(relocation_exact),
        "power_projection_independent": bool(
            analysis.get("projected_resolution") == projected_expected
        ),
        "stage_decision_independent": bool(stage_exact),
        "analysis_support_fields_exact": bool(support_exact),
    }


def audit_payloads(
    preflight: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    preflight_sha256: str,
    analysis_sha256: str,
    provenance_checks: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    row_contract = _raw_row_contract(preflight)
    mechanics = _mechanics_contract(preflight)
    health = _health_contract(preflight)
    sources = preflight.get("sources")
    checks: dict[str, bool] = {
        "preflight_schema": bool(
            preflight.get("schema") == REGISTER_PREFLIGHT_SCHEMA
            and preflight.get("status") == RAW_STATUS
            and preflight.get("stage") == "development"
        ),
        "analysis_schema": analysis.get("schema") == REGISTER_PREFLIGHT_ANALYSIS_SCHEMA,
        "analysis_bound_to_preflight": analysis.get("preflight_sha256") == preflight_sha256,
        "exact_development_lattice": bool(row_contract.get("passed")),
        "frozen_design": _design_contract(preflight),
        "candidate_construction_bound": _construction_source_binding(preflight),
        "deterministic_selection": _selection_contract(preflight),
        "mechanics_recomputed": bool(mechanics.get("passed")),
        "health_recomputed": bool(health.get("passed")),
        "source_records_present": bool(
            isinstance(sources, Mapping)
            and set(sources)
            >= {
                "cohort",
                "candidate_package",
                "candidate_construction_audit",
                "phase2_analysis",
                "phase2_audit",
                "checkpoint",
            }
        ),
        "source_hashes_valid": bool(
            isinstance(sources, Mapping)
            and sources
            and all(_file_record_valid(record) for record in sources.values())
        ),
        "analysis_sha256_present": bool(
            len(str(analysis_sha256)) == 64
            and all(character in "0123456789abcdef" for character in str(analysis_sha256))
        ),
    }
    if provenance_checks is not None:
        checks.update(
            {
                f"producer_manifest_{name}": bool(value)
                for name, value in provenance_checks.items()
            }
        )
    recompute_error = None
    independent_checks: dict[str, bool] = {}
    try:
        independent_checks = _independent_analysis_checks(
            preflight,
            analysis,
            preflight_sha256=preflight_sha256,
        )
        checks.update(independent_checks)
        checks["analysis_exact_recompute"] = all(independent_checks.values())
    except Exception as exc:  # independent audits record failures instead of hiding them
        recompute_error = f"{type(exc).__name__}: {exc}"
        for name in (
            "relocation_statistics_independent",
            "power_projection_independent",
            "stage_decision_independent",
            "analysis_support_fields_exact",
        ):
            checks[name] = False
        checks["analysis_exact_recompute"] = False
    passed = all(checks.values())
    return {
        "schema": REGISTER_PREFLIGHT_AUDIT_SCHEMA,
        "stage": "development",
        "status": "PASS" if passed else "FAIL",
        "preflight_sha256": preflight_sha256,
        "analysis_sha256": analysis_sha256,
        "partial_specificity_seeds": {
            "expected": PARTIAL_SPECIFICITY_SEED,
            "opposite": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
        },
        "checks": checks,
        "row_contract": row_contract,
        "mechanics": mechanics,
        "health": health,
        "recompute_error": recompute_error,
        "recomputed_status": (
            analysis.get("status")
            if checks.get("stage_decision_independent") is True
            else None
        ),
        "reported_status": analysis.get("status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--preflight-manifest", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preflight_path = Path(args.preflight).expanduser().resolve()
    preflight_manifest_path = Path(args.preflight_manifest).expanduser().resolve()
    analysis_path = Path(args.analysis).expanduser().resolve()
    analysis_manifest_path = Path(args.analysis_manifest).expanduser().resolve()
    output, manifest_path = output_paths(args.out)
    preflight = json.loads(preflight_path.read_text())
    preflight_manifest = json.loads(preflight_manifest_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    analysis_manifest = json.loads(analysis_manifest_path.read_text())
    preflight_manifest_validation = validate_sibling_manifest(
        preflight_manifest,
        manifest_path=preflight_manifest_path,
        artifact_path=preflight_path,
    )
    analysis_manifest_validation = validate_sibling_manifest(
        analysis_manifest,
        manifest_path=analysis_manifest_path,
        artifact_path=analysis_path,
    )
    provenance = {
        "preflight_valid": bool(preflight_manifest_validation["passed"]),
        "analysis_valid": bool(analysis_manifest_validation["passed"]),
        "analysis_consumes_preflight": manifest_binds_path(
            analysis_manifest, preflight_path, field="inputs"
        ),
        "preflight_precedes_analysis": manifest_precedes(
            preflight_manifest, analysis_manifest
        ),
        "single_code_revision": preflight_manifest.get("git_commit")
        == analysis_manifest.get("git_commit"),
    }
    audit = audit_payloads(
        preflight,
        analysis,
        preflight_sha256=sha256_file(preflight_path),
        analysis_sha256=sha256_file(analysis_path),
        provenance_checks=provenance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        experiment_id="pbc-p3-register-development-audit",
        command=sys.argv,
        repo=REPO,
        status="AUDIT",
        config={
            "schema": REGISTER_PREFLIGHT_AUDIT_SCHEMA,
            "stage": "development",
            "expected_partial_specificity_seed": PARTIAL_SPECIFICITY_SEED,
            "opposite_partial_specificity_seed": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
        },
        seeds={
            "expected_partial_specificity": PARTIAL_SPECIFICITY_SEED,
            "opposite_partial_specificity": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
        },
        inputs=[
            file_record(preflight_path),
            file_record(preflight_manifest_path),
            file_record(analysis_path),
            file_record(analysis_manifest_path),
        ],
        outputs=[file_record(output)],
        notes=[
            "Independent complete-lattice/arithmetic/gate checks and deterministic analysis recomputation."
        ],
    )
    manifest.write(manifest_path)
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
