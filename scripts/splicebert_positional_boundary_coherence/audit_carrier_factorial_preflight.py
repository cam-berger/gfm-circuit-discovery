#!/usr/bin/env python
"""Independently audit Phase-II development measurements before target freeze.

This module deliberately does not import the preflight runner's gate or verdict
functions.  It recomputes the raw/analysis hashes, every dose decision, every
§7.4 table-health endpoint, and the final P3/P10 authorization state.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_carrier_factorial import (  # noqa: E402
    DOSE_GRID,
    MIN_STRUCTURED_CONTROLS,
)


PREFLIGHT_SCHEMA = "gfmcd.pbc.carrier_factorial_preflight.v1"
TABLE_HEALTH_SCHEMA = "gfmcd.pbc.carrier_factorial_table_health.v1"
AUDIT_SCHEMA = "gfmcd.pbc.carrier_factorial_preflight_audit.v1"
FIXTURE_CALIBRATION_SCHEMA = "gfmcd.pbc.carrier_factorial_fixture_calibration.v1"
CONSTRUCTION_AUDIT_SCHEMA = "gfmcd.positional_boundary_coherence.cohort_audit.v1"
CONSTRUCTION_AUDIT_STATUS = "AUDIT_PASS_TARGET_AUTHORIZED"
CONSTRUCTION_AUDIT_EXPERIMENT_ID = (
    "splicebert-positional-boundary-coherence:composition-balanced-cohort-audit-v1"
)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _raw_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema",
            "split_role",
            "dose_grid",
            "bands",
            "n_controls_per_family",
            "control_seed",
            "dose_selection_used_phase_labels",
            "dose_selection_used_psi",
            "development_resolution_opened_after_dose_selection",
            "construction_audit_provenance",
            "measurements",
            "provider",
        )
        if key in payload
    }


def _analysis_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in ("schema", "status", "target_mode", "evaluation")
        if key in payload
    }


def _file_record_valid(record: object) -> bool:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        return False
    path = Path(str(record["path"])).expanduser().resolve()
    return bool(
        path.is_file()
        and type(record.get("size_bytes")) is int
        and path.stat().st_size == int(record["size_bytes"])
        and str(record.get("sha256", "")) == sha256_file(path)
    )


def _manifest_contract(
    manifest: Mapping[str, Any], preflight_path: Path
) -> dict[str, bool]:
    outputs = manifest.get("outputs")
    inputs = manifest.get("inputs")
    return {
        "preflight_output_bound": bool(
            isinstance(outputs, list)
            and len(outputs) == 1
            and _file_record_valid(outputs[0])
            and Path(str(outputs[0]["path"])).expanduser().resolve()
            == preflight_path.resolve()
        ),
        "all_manifest_inputs_valid": bool(
            isinstance(inputs, list)
            and inputs
            and all(_file_record_valid(record) for record in inputs)
        ),
        "revision_recorded": bool(
            isinstance(manifest.get("git_commit"), str)
            and len(str(manifest["git_commit"])) == 40
            and type(manifest.get("git_dirty")) is bool
        ),
        "experiment_identity": bool(
            manifest.get("experiment_id") == "pbc-carrier-factorial-preflight-v1"
        ),
    }


def _construction_audit_contract(
    preflight: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, bool]:
    provenance = preflight.get("construction_audit_provenance")
    inputs = manifest.get("inputs")
    declared = bool(
        isinstance(provenance, Mapping)
        and provenance.get("experiment_id") == CONSTRUCTION_AUDIT_EXPERIMENT_ID
        and len(str(provenance.get("artifact_sha256", ""))) == 64
        and len(str(provenance.get("manifest_sha256", ""))) == 64
    )
    records = inputs if isinstance(inputs, list) else []

    def record_for(digest: str) -> Mapping[str, Any] | None:
        matches = [
            record
            for record in records
            if isinstance(record, Mapping)
            and str(record.get("sha256", "")) == digest
            and _file_record_valid(record)
        ]
        return matches[0] if len(matches) == 1 else None

    artifact_record = (
        record_for(str(provenance["artifact_sha256"]))
        if declared and isinstance(provenance, Mapping)
        else None
    )
    sibling_record = (
        record_for(str(provenance["manifest_sha256"]))
        if declared and isinstance(provenance, Mapping)
        else None
    )
    artifact: Mapping[str, Any] | None = None
    sibling: Mapping[str, Any] | None = None
    try:
        if artifact_record is not None:
            loaded = json.loads(Path(str(artifact_record["path"])).read_text())
            artifact = loaded if isinstance(loaded, Mapping) else None
        if sibling_record is not None:
            loaded = json.loads(Path(str(sibling_record["path"])).read_text())
            sibling = loaded if isinstance(loaded, Mapping) else None
    except (OSError, json.JSONDecodeError):
        artifact = None
        sibling = None

    sibling_outputs = sibling.get("outputs") if sibling is not None else None
    sibling_inputs = sibling.get("inputs") if sibling is not None else None
    sibling_config = sibling.get("config") if sibling is not None else None
    artifact_path = (
        Path(str(artifact_record["path"])).expanduser().resolve()
        if artifact_record is not None
        else None
    )
    return {
        "provenance_declared": declared,
        "artifact_is_preflight_input": artifact_record is not None,
        "sibling_manifest_is_preflight_input": sibling_record is not None,
        "artifact_authorizes_target": bool(
            artifact is not None
            and artifact.get("schema") == CONSTRUCTION_AUDIT_SCHEMA
            and artifact.get("stage") == "construction"
            and artifact.get("status") == CONSTRUCTION_AUDIT_STATUS
            and artifact.get("passed") is True
            and artifact.get("target_execution_authorized") is True
        ),
        "sibling_manifest_identity": bool(
            sibling is not None
            and sibling.get("experiment_id") == CONSTRUCTION_AUDIT_EXPERIMENT_ID
            and sibling.get("status") == CONSTRUCTION_AUDIT_STATUS
            and isinstance(sibling_config, Mapping)
            and sibling_config.get("stage") == "construction"
        ),
        "sibling_manifest_binds_artifact": bool(
            artifact_path is not None
            and isinstance(sibling_outputs, list)
            and len(sibling_outputs) == 1
            and _file_record_valid(sibling_outputs[0])
            and Path(str(sibling_outputs[0]["path"])).expanduser().resolve()
            == artifact_path
        ),
        "sibling_manifest_inputs_valid": bool(
            isinstance(sibling_inputs, list)
            and sibling_inputs
            and all(_file_record_valid(record) for record in sibling_inputs)
        ),
    }


def _stratified_interval(
    values: Sequence[float],
    strata: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    labels = np.asarray(strata, dtype=object)
    if data.ndim != 1 or data.shape != labels.shape or len(data) < 6:
        raise ValueError("health endpoint support is invalid")
    groups = [np.flatnonzero(labels == label) for label in sorted(set(labels))]
    rng = np.random.default_rng(seed)
    draws = np.empty(999, dtype=float)
    for draw in range(999):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        draws[draw] = float(data[indices].mean())
    return {
        "estimate": float(data.mean()),
        "ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "n_bootstrap": 999,
        "seed": seed,
    }


def _recompute_health_table(table: Mapping[str, Any]) -> dict[str, Any]:
    blocks = table.get("raw_blocks")
    if not isinstance(blocks, list) or len(blocks) != 120:
        raise ValueError("table health lacks the 120-block development support")
    block_ids = [str(row.get("block_id")) for row in blocks]
    strata = [str(row.get("stratum")) for row in blocks]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("table health repeats a block")
    seed = int(table["inference_seed"])

    def endpoint(field: str, offset: int) -> dict[str, Any]:
        return _stratified_interval(
            [float(row[field]) for row in blocks], strata, seed=seed + offset
        )

    native = endpoint("native_source_ac", 1)
    crossed = endpoint("crossed_ac", 2)
    bridge = endpoint("bridge_effect", 3)
    ac_at = endpoint("crossed_ac_minus_at", 4)
    protected = endpoint("protected_log_probability_delta", 5)
    donor = endpoint("donor_destruction_response", 6)
    acceptor = endpoint("acceptor_destruction_response", 7)
    intact_correct = sum(int(row["protected_intact_correct"]) for row in blocks)
    table_correct = sum(int(row["protected_table_correct"]) for row in blocks)
    total = sum(int(row["protected_total"]) for row in blocks)
    intact_accuracy = intact_correct / total if total else 0.0
    table_accuracy = table_correct / total if total else 0.0
    accuracy_retention = table_accuracy / intact_accuracy if intact_accuracy > 0 else 0.0
    baseline_donor = float(
        np.mean([float(row["baseline_donor_destruction_response"]) for row in blocks])
    )
    baseline_acceptor = float(
        np.mean([float(row["baseline_acceptor_destruction_response"]) for row in blocks])
    )
    donor_retention = donor["estimate"] / baseline_donor if baseline_donor > 0 else -math.inf
    acceptor_retention = (
        acceptor["estimate"] / baseline_acceptor if baseline_acceptor > 0 else -math.inf
    )
    gates = {
        "native_source_positive": native["ci"][0] > 0.05,
        "phase_blind_crossed_positive": crossed["ci"][0] > 0.02,
        "bridge_negative": bridge["ci"][0] >= -0.05
        and bridge["ci"][1] <= 0.05,
        "ac_at_invariant": ac_at["ci"][0] >= -0.05
        and ac_at["ci"][1] <= 0.05,
        "protected_base_health": protected["ci"][0] > -0.10,
        "masked_accuracy_health": accuracy_retention >= 0.80,
        "donor_margin_health": donor["ci"][0] > 0.50
        and donor_retention >= 0.80,
        "acceptor_margin_health": acceptor["ci"][0] > 0.50
        and acceptor_retention >= 0.80,
        "numerical_complete": all(row.get("numerical_complete") is True for row in blocks),
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


def _health_contract(preflight: Mapping[str, Any]) -> dict[str, Any]:
    measurements = preflight.get("measurements")
    health = measurements.get("authorized_table_health") if isinstance(measurements, Mapping) else None
    mode = str(preflight.get("target_mode"))
    if not isinstance(health, Mapping) or health.get("schema") != TABLE_HEALTH_SCHEMA:
        return {"passed": False, "reason": "authorized table-health artifact absent"}
    tables = health.get("tables")
    if not isinstance(tables, Mapping):
        return {"passed": False, "reason": "table-health rows absent"}
    required_families = (
        {"p3_phase", "p3_axis"}
        if mode == "P3_ONLY"
        else {
            "p3_phase",
            "p3_axis",
            "p10_phase",
            "p10_axis",
            "joint_p3",
            "joint_p10",
            "joint",
        }
    )
    required_worlds = {"Y00", "Y10"} if mode == "P3_ONLY" else {"Y00", "Y10", "Y01", "Y11"}
    specs: dict[str, Mapping[str, Any]] = {}
    recomputed: dict[str, Any] = {}
    table_contract = True
    try:
        for label, row in tables.items():
            if not isinstance(row, Mapping) or len(str(row.get("table_sha256", ""))) != 64:
                raise ValueError("invalid table health row/header")
            spec = row.get("spec")
            if not isinstance(spec, Mapping):
                raise ValueError("table health spec absent")
            specs[str(label)] = spec
            independently = _recompute_health_table(row)
            recomputed[str(label)] = independently
            table_contract &= all(row.get(key) == independently[key] for key in independently)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"passed": False, "reason": f"health recompute failed: {exc}"}
    target_worlds = {
        label for label, spec in specs.items() if str(spec.get("family")) == "target"
    }
    family_counts = Counter(
        str(spec.get("family"))
        for spec in specs.values()
        if str(spec.get("family")) != "target"
    )
    control_ids = {
        family: {
            int(spec.get("control_id"))
            for spec in specs.values()
            if str(spec.get("family")) == family
        }
        for family in required_families
    }
    coverage = bool(
        target_worlds == required_worlds
        and set(family_counts) == required_families
        and all(
            family_counts[family] == MIN_STRUCTURED_CONTROLS
            for family in required_families
        )
        and all(
            control_ids[family] == set(range(1, 32))
            for family in required_families
        )
        and int(health.get("n_tables", -1)) == len(tables)
        and int(health.get("n_blocks", -1)) == 120
        and health.get("target_mode") == mode
    )
    all_pass = bool(recomputed and all(row["pass"] for row in recomputed.values()))
    return {
        "passed": bool(
            coverage
            and table_contract
            and all_pass
            and health.get("all_tables_pass") is True
        ),
        "coverage_valid": coverage,
        "raw_recompute_matches": table_contract,
        "all_tables_scientifically_pass": all_pass,
        "n_tables": len(tables),
        "family_counts": dict(family_counts),
    }


def _fixture_endpoint(row: Mapping[str, Any], name: str) -> bool:
    endpoint = row.get(name)
    if not isinstance(endpoint, Mapping):
        return False
    try:
        ci = [float(value) for value in endpoint["simultaneous_ci"]]
        pvalues = [float(value) for value in endpoint["add_one_p"]]
        material = ci[0] > 0.05 if name != "J" else (ci[0] > 0.05 or ci[1] < -0.05)
        return bool(
            len(ci) == 2
            and len(pvalues) == (1 if name == "J" else 2)
            and material
            and all(value <= 1 / 32 for value in pvalues)
            and endpoint.get("target_exceeds_every_control") is True
            and endpoint.get("corrected_specificity_pass") is True
            and float(endpoint["paired_ci"][0]) > 0
            and float(endpoint["mde80"]) <= (0.04 if name == "N3" else 0.05)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _fixture_clean(row: Mapping[str, Any]) -> bool:
    clean = row.get("clean")
    if not isinstance(clean, Mapping):
        return False
    try:
        ci = [float(value) for value in clean["simultaneous_ci"]]
        return bool(
            len(ci) == 2
            and ci[0] > 0.05
            and float(clean["minimum_directional_lower_bound"]) > 0
            and float(clean["pseudo_add_one_p"]) <= 1 / 1000
            and clean.get("target_exceeds_every_pseudo") is True
            and clean.get("corrected_specificity_pass") is True
            and float(clean["paired_pseudo_ci"][0]) > 0
            and float(clean["mde80"]) <= 0.04
        )
    except (KeyError, TypeError, ValueError):
        return False


def _fixture_summary(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"passed": False, "reason": "fixture calibration absent"}
    null_rows = value.get("null_replicates")
    planted = value.get("planted_replicates")
    valid = bool(
        value.get("schema") == FIXTURE_CALIBRATION_SCHEMA
        and value.get("analysis_engine")
        == "analyze_carrier_factorial.analyze_fixture_arrays"
        and value.get("decision_engine")
        == "pbc_carrier_factorial.carrier_phase2_decision"
        and value.get("stage") == "development"
        and value.get("status") == "COMPLETE"
        and int(value.get("replicates", -1)) == 1_000
        and value.get("effects") == [0.05, 0.075, 0.10]
        and value.get("generator")
        == {
            "algorithm": "independent_gaussian_block_fixtures_v1",
            "base_seed": 2026081063,
            "bit_generator": "PCG64",
            "n_blocks": 600,
            "strata": [
                "chr1|+", "chr1|-", "chr2|+", "chr2|-", "chr3|+", "chr3|-",
                "chr4|+", "chr4|-", "chr5|+", "chr5|-", "chr6|+", "chr6|-",
            ],
            "noise_sd": 0.16,
            "sign_symmetric_null": True,
            "stored_inputs": "control_and_block_margins_sufficient_for_exact_families",
        }
        and value.get("analysis_parameters")
        == {"n_bootstrap": 20_000, "bootstrap_seed": 2026081001}
        and isinstance(null_rows, list)
        and len(null_rows) == 1_000
        and all(isinstance(row, Mapping) for row in null_rows)
        and isinstance(planted, Mapping)
        and set(planted) == {"0.050", "0.075", "0.100"}
        and all(
            isinstance(rows, list)
            and len(rows) == 1_000
            and all(isinstance(row, Mapping) for row in rows)
            for rows in planted.values()
        )
    )
    if not valid:
        return {"passed": False, "reason": "fixture calibration header/support invalid"}
    groups = [null_rows, *planted.values()]
    if any(
        [int(row.get("replicate", -1)) for row in rows] != list(range(1_000))
        for rows in groups
    ):
        return {"passed": False, "reason": "fixture replicate IDs invalid"}
    indexed_groups = (
        (0, null_rows),
        (1, planted["0.050"]),
        (2, planted["0.075"]),
        (3, planted["0.100"]),
    )

    def engine_trace(row: Mapping[str, Any], group: int) -> bool:
        replicate = int(row.get("replicate", -1))
        p3 = bool(_fixture_clean(row) and _fixture_endpoint(row, "N3"))
        joint = bool(
            p3
            and _fixture_endpoint(row, "N10")
            and _fixture_endpoint(row, "J")
        )
        decision = row.get("decision")
        return bool(
            row.get("analysis_engine")
            == "analyze_carrier_factorial.analyze_fixture_arrays"
            and row.get("analysis_parameters")
            == {
                "n_bootstrap": 20_000,
                "bootstrap_seed": 2026081001,
                "n_blocks": 600,
            }
            and int(row.get("generator_seed", -1))
            == 2026081063 + group * 1_000_000 + replicate
            and len(str(row.get("input_sha256", ""))) == 64
            and isinstance(decision, Mapping)
            and decision.get("authorize_p3_phase3") is p3
            and decision.get("authorize_joint_phase3") is joint
        )

    if any(not engine_trace(row, group) for group, rows in indexed_groups for row in rows):
        return {"passed": False, "reason": "fixture engine trace invalid"}
    contract = value.get("contract_fixtures")
    expected_contract = {
        "identity": (10, False, False),
        "no_op": (11, False, False),
        "planted_positive": (12, True, True),
        "sign_reversed": (13, False, False),
    }
    if not isinstance(contract, Mapping) or set(contract) != set(expected_contract):
        return {"passed": False, "reason": "contract fixtures absent"}
    for name, (group, expected_p3, expected_joint) in expected_contract.items():
        row = contract[name]
        decision = row.get("decision") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or not engine_trace(row, group)
            or not isinstance(decision, Mapping)
            or decision.get("authorize_p3_phase3") is not expected_p3
            or decision.get("authorize_joint_phase3") is not expected_joint
        ):
            return {"passed": False, "reason": f"contract fixture failed: {name}"}
    null_hits = sum(
        int(
            _fixture_clean(row)
            or any(_fixture_endpoint(row, name) for name in ("N3", "N10", "J"))
        )
        for row in null_rows
    )
    counts = {
        effect: {
            name: sum(
                int(_fixture_clean(row) and _fixture_endpoint(row, name))
                for row in rows
            )
            for name in ("N3", "N10", "J")
        }
        for effect, rows in planted.items()
    }
    power = {
        effect: {name: count / 1_000 for name, count in values.items()}
        for effect, values in counts.items()
    }
    fwer = null_hits / 1_000
    return {
        "passed": bool(fwer <= 0.06 and power["0.075"]["N3"] >= 0.80),
        "null_false_positive_count": null_hits,
        "familywise_false_positive_rate": fwer,
        "planted_success_counts": counts,
        "corrected_power": power,
        "n3_power_pass": power["0.075"]["N3"] >= 0.80,
        "optional_power_pass": bool(
            power["0.075"]["N10"] >= 0.80 and power["0.075"]["J"] >= 0.80
        ),
        "fwer_pass": fwer <= 0.06,
        "replicates": 1_000,
        "effects": [0.05, 0.075, 0.10],
    }


def _fixture_contract(
    preflight: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    shared = preflight.get("measurements", {}).get("shared_gates", {})
    diagnostic = shared.get("diagnostics", {}).get("fixture_calibration", {})
    source = diagnostic.get("source") if isinstance(diagnostic, Mapping) else None
    provenance = (
        diagnostic.get("provenance") if isinstance(diagnostic, Mapping) else None
    )
    recomputed = _fixture_summary(source)
    reported = diagnostic.get("summary") if isinstance(diagnostic, Mapping) else None
    input_hashes = {
        str(record.get("sha256", ""))
        for record in manifest.get("inputs", [])
        if isinstance(record, Mapping)
    }
    provenance_pass = bool(
        isinstance(provenance, Mapping)
        and provenance.get("experiment_id")
        == "pbc-carrier-factorial-fixture-calibration-v1"
        and len(str(provenance.get("artifact_sha256", ""))) == 64
        and len(str(provenance.get("manifest_sha256", ""))) == 64
        and provenance["artifact_sha256"] in input_hashes
        and provenance["manifest_sha256"] in input_hashes
    )
    return {
        "passed": bool(
            recomputed.get("passed")
            and (
                preflight.get("target_mode") != "FULL_FACTORIAL"
                or recomputed.get("optional_power_pass") is True
            )
            and reported == recomputed
            and provenance_pass
            and shared.get("synthetic_familywise_fpr")
            is (recomputed.get("fwer_pass") is True)
            and shared.get("planted_power")
            is (recomputed.get("n3_power_pass") is True)
        ),
        "recomputed": recomputed,
        "reported_matches": reported == recomputed,
        "producer_provenance_pass": provenance_pass,
    }


def _flag(row: Mapping[str, Any], name: str) -> bool:
    value = row.get(name)
    return bool(value.get("pass")) if isinstance(value, Mapping) else value is True


def _ratio_recomputed(
    row: Mapping[str, Any],
    *,
    raw_name: str,
    health_name: str,
    estimate_name: str,
    interval_name: str,
) -> bool:
    raw = row.get(raw_name)
    reported = row.get(health_name)
    if not isinstance(raw, list) or len(raw) != 120 or not isinstance(reported, Mapping):
        return False
    try:
        numerator = np.asarray([float(item["numerator"]) for item in raw])
        denominator = np.asarray([float(item["denominator"]) for item in raw])
        strata = np.asarray([str(item["stratum"]) for item in raw], dtype=object)
        block_ids = [str(item["block_id"]) for item in raw]
        seed = int(reported["seed"])
        groups = [np.flatnonzero(strata == label) for label in sorted(set(strata))]
        rng = np.random.default_rng(seed)
        ratios = np.empty(999, dtype=float)
        denominators = np.empty(999, dtype=float)
        for draw in range(999):
            indices = np.concatenate(
                [rng.choice(group, size=len(group), replace=True) for group in groups]
            )
            denominators[draw] = float(denominator[indices].mean())
            ratios[draw] = float(numerator[indices].mean() / denominators[draw])
        point_denominator = float(denominator.mean())
        point = float(numerator.mean() / point_denominator)
        finite = np.isfinite(ratios)
        sign = 1.0 if point_denominator >= 0 else -1.0
        stability = float(np.mean(sign * denominators > 0))
        interval = [
            float(np.quantile(ratios[finite], 0.025)),
            float(np.quantile(ratios[finite], 0.975)),
        ]
        expected_health = {
            "n_bootstrap": 999,
            "seed": seed,
            "finite_fraction": float(finite.mean()),
            "denominator_sign_stability": stability,
            "denominator_point": point_denominator,
            "pass": bool(finite.all() and stability >= 0.95),
        }
        return bool(
            len(block_ids) == len(set(block_ids))
            and np.isfinite(numerator).all()
            and np.isfinite(denominator).all()
            and reported == expected_health
            and math.isclose(
                float(row.get(estimate_name, math.inf)), point, rel_tol=0.0, abs_tol=1e-12
            )
            and all(
                math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
                for left, right in zip(row.get(interval_name, []), interval, strict=True)
            )
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def _dose_pass(band: str, row: Mapping[str, Any], *, joint: bool = True) -> bool:
    families = row.get("control_families")
    family_pass = bool(
        isinstance(families, Mapping)
        and set(families) == {"phase", "axis"}
        and all(
            int(value.get("n_controls", -1)) == 31
            and value.get("target_exceeds_every_control") is True
            and float(value.get("add_one_p", math.inf)) <= 1 / 32
            and float(value.get("paired_excess_ci", [-math.inf])[0]) > 0
            for value in families.values()
            if isinstance(value, Mapping)
        )
        and len(families) == 2
    )
    common = bool(
        row.get("mechanical_pass") is True
        and row.get("health_pass") is True
        and row.get("all_control_health_pass") is True
        and float(row.get("rescue_forward_max_abs_error", math.inf)) <= 1e-8
        and _flag(row, "rescue_ratio_health")
        and float(row.get("rescue_recovery", -math.inf)) >= 0.90
        and float(row.get("rescue_recovery_ci", [-math.inf])[0]) > 0.80
        and family_pass
    )
    if band == "p3":
        return bool(
            common
            and _ratio_recomputed(
                row,
                raw_name="cross_carrier_ratio_raw",
                health_name="cross_carrier_ratio_health",
                estimate_name="cross_carrier_retention",
                interval_name="cross_carrier_retention_ci",
            )
            and float(row.get("capability_ci", [-math.inf])[0]) > 0.05
            and float(row.get("cross_carrier_retention", -math.inf)) >= 0.80
            and float(row.get("cross_carrier_retention_ci", [-math.inf])[0]) > 0.80
        )
    if band != "p10":
        raise ValueError("unknown carrier band")
    individual = bool(
        common
        and _ratio_recomputed(
            row,
            raw_name="capability_ratio_raw",
            health_name="capability_ratio_health",
            estimate_name="capability_fraction_reduced",
            interval_name="capability_fraction_reduced_ci",
        )
        and float(row.get("intact_carrier_score_ci", [-math.inf])[0]) > 0
        and float(row.get("denominator_finite_sign_stable_fraction", -math.inf)) >= 0.95
        and float(row.get("capability_fraction_reduced", -math.inf)) >= 0.50
        and float(row.get("capability_fraction_reduced_ci", [-math.inf])[0]) > 0.25
    )
    if not joint:
        return individual
    return bool(
        individual
        and row.get("joint_health_pass") is True
        and float(row.get("cross_carrier_retention", -math.inf)) >= 0.80
        and float(row.get("cross_carrier_retention_ci", [-math.inf])[0]) > 0.80
    )


def _selection(band: str, rows: Mapping[str, Any]) -> dict[str, Any]:
    expected = {f"{dose:.2f}" for dose in DOSE_GRID}
    if set(rows) != expected:
        raise ValueError("dose grid is incomplete")
    passing = [
        dose
        for dose in DOSE_GRID
        if _dose_pass(band, rows[f"{dose:.2f}"], joint=True)
    ]
    return {"passed": bool(passing), "selected_dose": max(passing) if passing else None}


def _recompute_authorization(preflight: Mapping[str, Any]) -> dict[str, Any]:
    measurements = preflight.get("measurements")
    if not isinstance(measurements, Mapping):
        raise ValueError("preflight measurements absent")
    shared = measurements.get("shared_gates")
    dose_rows = measurements.get("dose_rows")
    mde = measurements.get("development_mde")
    if not isinstance(shared, Mapping) or not isinstance(dose_rows, Mapping) or not isinstance(mde, Mapping):
        raise ValueError("preflight measurement families absent")
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
    shared_pass = all(_flag(shared, name) for name in required_shared)
    p3 = _selection("p3", dose_rows.get("p3", {}))
    p10 = _selection("p10", dose_rows.get("p10", {}))

    def mde_pass(name: str, threshold: float) -> bool:
        row = mde.get(name)
        return bool(
            isinstance(row, Mapping)
            and float(row.get("mde80", math.inf)) <= threshold
            and float(row.get("monte_carlo_se", math.inf)) <= 0.0002
            and int(row.get("n_bootstrap", -1)) == 20_000
            and int(row.get("target_blocks", -1)) == 600
        )

    p3_resolution = mde_pass("clean_psi", 0.04) and mde_pass("N3", 0.04)
    p10_resolution = (
        mde_pass("N10", 0.05) and mde_pass("J", 0.05) if p10["passed"] else True
    )
    mandatory = bool(shared_pass and p3["passed"] and p3_resolution)
    p10_authorized = bool(p10["passed"] and p10_resolution)
    if not mandatory:
        status, mode = "P3_INSTRUMENT_UNRESOLVED", "SEALED"
    elif p10_authorized:
        status, mode = "FULL_FACTORIAL_TARGET_AUTHORIZED", "FULL_FACTORIAL"
    else:
        status, mode = "P3_ONLY_TARGET_AUTHORIZED", "P3_ONLY"
    return {
        "status": status,
        "target_mode": mode,
        "shared_pass": shared_pass,
        "p3": p3,
        "p10": p10,
        "p3_resolution_pass": p3_resolution,
        "p10_resolution_pass": p10_resolution,
    }


def audit_payloads(
    preflight: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    preflight_path: Path,
    preflight_sha256: str,
    preflight_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    raw_sha = _canonical_hash(_raw_view(preflight))
    analysis_sha = _canonical_hash(_analysis_view(preflight))
    manifest_checks = _manifest_contract(manifest, preflight_path)
    construction_audit = _construction_audit_contract(preflight, manifest)
    health = _health_contract(preflight)
    fixtures = _fixture_contract(preflight, manifest)
    recompute_error = None
    authorization: dict[str, Any] | None = None
    try:
        authorization = _recompute_authorization(preflight)
    except Exception as exc:  # an audit records malformed evidence as a failure
        recompute_error = f"{type(exc).__name__}: {exc}"
    evaluation = preflight.get("evaluation")
    checks = {
        "preflight_schema": preflight.get("schema") == PREFLIGHT_SCHEMA,
        "stage_declared": preflight.get("stage") == "development",
        "development_only": preflight.get("split_role") == "development",
        "outcome_firewall": bool(
            preflight.get("dose_selection_used_phase_labels") is False
            and preflight.get("dose_selection_used_psi") is False
        ),
        "raw_hash_recomputed": preflight.get("raw_sha256") == raw_sha,
        "analysis_hash_recomputed": preflight.get("analysis_sha256") == analysis_sha,
        "manifest_contract": all(manifest_checks.values()),
        "construction_audit_chain": all(construction_audit.values()),
        "per_table_health_recomputed": bool(health.get("passed")),
        "fixture_calibration_recomputed": bool(fixtures.get("passed")),
        "authorization_recomputed": bool(
            authorization is not None
            and isinstance(evaluation, Mapping)
            and authorization["status"] == preflight.get("status")
            and authorization["target_mode"] == preflight.get("target_mode")
            and authorization["shared_pass"] == evaluation.get("shared_pass")
            and authorization["p3"]["passed"] == evaluation.get("p3", {}).get("passed")
            and authorization["p3"]["selected_dose"]
            == evaluation.get("p3", {}).get("selected_dose")
            and authorization["p10"]["passed"] == evaluation.get("p10", {}).get("passed")
            and authorization["p10"]["selected_dose"]
            == evaluation.get("p10", {}).get("selected_dose")
            and authorization["p3_resolution_pass"]
            == evaluation.get("p3_resolution_pass")
            and authorization["p10_resolution_pass"]
            == evaluation.get("p10_resolution_pass")
        ),
    }
    passed = bool(all(checks.values()))
    return {
        "schema": AUDIT_SCHEMA,
        "stage": "development",
        "status": "PASS" if passed else "FAIL",
        "preflight_sha256": preflight_sha256,
        "raw_sha256": raw_sha,
        "analysis_sha256": analysis_sha,
        "preflight_manifest_sha256": (
            str(preflight_manifest_sha256)
            if preflight_manifest_sha256 is not None
            else _canonical_hash(manifest)
        ),
        "target_mode": None if authorization is None else authorization["target_mode"],
        "checks": checks,
        "manifest_checks": manifest_checks,
        "construction_audit": construction_audit,
        "table_health": health,
        "fixture_calibration": fixtures,
        "authorization_recompute": authorization,
        "recompute_error": recompute_error,
        "target_outcomes_read": False,
    }


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--preflight-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preflight_path = Path(args.preflight).expanduser().resolve()
    preflight_manifest_path = Path(args.preflight_manifest).expanduser().resolve()
    output, manifest_path = output_paths(args.out)
    preflight = json.loads(preflight_path.read_text())
    preflight_manifest = json.loads(preflight_manifest_path.read_text())
    audit = audit_payloads(
        preflight,
        preflight_manifest,
        preflight_path=preflight_path,
        preflight_sha256=sha256_file(preflight_path),
        preflight_manifest_sha256=sha256_file(preflight_manifest_path),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        "pbc-carrier-factorial-preflight-audit-v1",
        sys.argv,
        repo=REPO,
        status=audit["status"],
        config={"stage": "development", "target_mode": audit["target_mode"]},
    )
    for path in (
        preflight_path,
        preflight_manifest_path,
        REPO / "src/gfmcd/splicing/pbc_carrier_factorial.py",
        Path(__file__).resolve(),
    ):
        manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(json.dumps({"status": audit["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
