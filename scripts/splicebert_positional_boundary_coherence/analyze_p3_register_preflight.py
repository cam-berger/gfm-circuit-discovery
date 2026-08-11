#!/usr/bin/env python
"""Analyze the P3 register development preflight without opening the reserve."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

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
    preferred_delta,
)
from gfmcd.splicing.pbc_register_stage import (  # noqa: E402
    REGISTER_PREFLIGHT_ANALYSIS_SCHEMA,
    REGISTER_PREFLIGHT_SCHEMA,
    RESERVE_BLOCKS,
    projected_mde80,
    register_preflight_decision,
)
from analyze_p3_register_relocation import (  # noqa: E402
    CONTROL_LABELS,
    DIAGNOSTIC_LABELS,
    TARGET_SCHEMA,
    analyze_payload as analyze_relocation_payload,
)
from run_p3_register_preflight import (  # noqa: E402
    DEVELOPMENT_BLOCKS,
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
SUPPORTIVE_RELOCATION_STATUSES = frozenset(
    {
        "P3_REGISTER_RELOCATION_SUPPORTED",
        "P3_REGISTER_PARTIAL_CONCORDANCE_SUPPORTED",
    }
)


def _expected_health_labels(mode: str) -> set[str]:
    states = ("intact", "attenuated") if mode == "JOINT" else ("intact",)
    components = (
        "p3",
        *(f"p3_signed_{index:02d}" for index in range(31)),
        "p3_common_mode",
        "p3_decoy_span",
    )
    return {
        f"{component}__{state}__r{register}"
        for component in components
        for state in states
        for register in REGISTERS
    }


def output_paths(path: str | Path) -> tuple[Path, Path]:
    output = guard_output_path(Path(path).expanduser().resolve())
    manifest = guard_output_path(output.with_name(output.stem + ".manifest.json"))
    if output == manifest:
        raise ValueError("analysis and manifest paths must be distinct")
    return output, manifest


def _fixture_rows(
    case: str,
    *,
    seed: int,
    blocks: int = 8,
    planted_effect: float = 0.075,
) -> list[dict[str, object]]:
    if case not in FIXTURE_CASES:
        raise ValueError(f"unknown fixture case {case!r}")
    rng = np.random.default_rng(int(seed))
    labels = ("p3", *CONTROL_LABELS, *DIAGNOSTIC_LABELS)
    rows: list[dict[str, object]] = []
    block_noise = rng.normal(0.0, 0.0015, size=(blocks, 3))
    for label_index, component in enumerate(labels):
        for block in range(blocks):
            chrom = f"chr{1 + block % 2}"
            strand = "+" if (block // 2) % 2 == 0 else "-"
            for register_index, register in enumerate(REGISTERS):
                noise = float(block_noise[block, register_index])
                orientation = "expected"
                if component == "p3":
                    if case == "null":
                        effect = noise / 4
                    elif case == "opposite_handed":
                        effect = planted_effect + noise
                        orientation = "opposite"
                    elif case == "no_ramp" and register != 0:
                        effect = noise / 4
                    else:
                        effect = planted_effect + noise
                elif component in CONTROL_LABELS:
                    effect = 0.004 + label_index * 0.00001 - noise / 8
                else:
                    # Both named controls fail to reproduce all three registers in
                    # ordinary fixtures, while remaining non-degenerate.
                    effect = {-1: -0.006, 0: 0.010, 1: -0.004}[register] + noise / 8
                    if case == "common_mode" and component == "p3_common_mode":
                        effect = planted_effect + noise
                    if case == "wrong_support" and component == "p3_decoy_span":
                        effect = planted_effect + noise
                for donor in range(3):
                    for acceptor in range(3):
                        delta = phase_delta(donor, acceptor)
                        selected_delta = (
                            register % 3
                            if orientation == "opposite"
                            else preferred_delta(register)
                        )
                        coupling = effect if delta == selected_delta else 0.0
                        rows.append(
                            {
                                "component": component,
                                "control_family": (
                                    "target"
                                    if component == "p3"
                                    else (
                                        "signed_hidden_axis"
                                        if component in CONTROL_LABELS
                                        else component.removeprefix("p3_")
                                    )
                                ),
                                "p10_state": "intact",
                                "register": register,
                                "block_id": f"fixture-b{block:02d}",
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


def _fixture_payload(case: str, *, seed: int) -> dict[str, object]:
    return {
        "schema": TARGET_SCHEMA,
        "status": "COMPLETE",
        "mode": "P3_ONLY",
        "authorization": {
            "phase2_p3_authorized": True,
            "phase2_joint_authorized": False,
            "register_preflight_passed": True,
        },
        "design": {"jreg_direction": None},
        "mechanics": {"register_zero_no_hook_max_abs": 0.0},
        "rows": _fixture_rows(case, seed=seed),
        "self_sha256": f"fixture-{case}-{seed}",
    }


def _fixture_case_passed(case: str, analysis: Mapping[str, object]) -> bool:
    status = str(analysis.get("status"))
    target = analysis.get("target")
    specificity = analysis.get("specificity")
    if not isinstance(target, Mapping) or not isinstance(specificity, Mapping):
        return False
    if case == "planted_positive":
        return status == "P3_REGISTER_RELOCATION_SUPPORTED"
    if case == "opposite_handed":
        return bool(
            target.get("opposite_all_registers_positive")
            and target.get("opposite_omega3_material_positive")
            and status not in SUPPORTIVE_RELOCATION_STATUSES
        )
    if case == "no_ramp":
        return bool(
            not target.get("all_registers_positive")
            and status not in SUPPORTIVE_RELOCATION_STATUSES
        )
    if case == "common_mode":
        diagnostics = analysis.get("diagnostic_controls")
        return bool(
            isinstance(diagnostics, Mapping)
            and diagnostics["p3_common_mode"]["all_registers_positive"]
            and not specificity.get("passed")
        )
    if case == "wrong_support":
        diagnostics = analysis.get("diagnostic_controls")
        return bool(
            isinstance(diagnostics, Mapping)
            and diagnostics["p3_decoy_span"]["all_registers_positive"]
            and not specificity.get("passed")
        )
    return status not in SUPPORTIVE_RELOCATION_STATUSES


@lru_cache(maxsize=8)
def calibrate_fixture_suite(
    *,
    seed: int = 2026081011,
    n_bootstrap: int = 99,
    replicates: int = 1_000,
) -> dict[str, object]:
    """Exercise the real relocation analysis/decision on frozen synthetic cases."""

    if n_bootstrap < 99 or replicates != 1_000:
        raise ValueError(
            "fixture calibration needs >=99 bootstrap draws and exactly 1,000 replicates"
        )
    cases: dict[str, dict[str, object]] = {}
    for index, case in enumerate(FIXTURE_CASES):
        result = analyze_relocation_payload(
            _fixture_payload(case, seed=seed + index),
            seed=seed + 100 + index,
            n_bootstrap=n_bootstrap,
        )
        cases[case] = {
            "status": result["status"],
            "passed": _fixture_case_passed(case, result),
        }

    null_false_positives = 0
    planted_positives = 0
    for replicate in range(replicates):
        null = analyze_relocation_payload(
            _fixture_payload("null", seed=seed + 1_000 + replicate),
            seed=seed + 2_000 + replicate,
            n_bootstrap=n_bootstrap,
        )
        planted = analyze_relocation_payload(
            _fixture_payload("planted_positive", seed=seed + 3_000 + replicate),
            seed=seed + 4_000 + replicate,
            n_bootstrap=n_bootstrap,
        )
        null_false_positives += int(
            null["status"] in SUPPORTIVE_RELOCATION_STATUSES
        )
        planted_positives += int(
            planted["status"] == "P3_REGISTER_RELOCATION_SUPPORTED"
        )
    fpr = null_false_positives / replicates
    power = planted_positives / replicates
    all_cases = all(bool(row["passed"]) for row in cases.values())
    return {
        "schema": FIXTURE_SCHEMA,
        "status": "PASS" if all_cases and fpr <= 0.06 and power >= 0.80 else "FAIL",
        "analysis_engine": "analyze_p3_register_relocation.analyze_payload",
        "decision_engine": "pbc_register_relocation.adjudicate_register_relocation",
        "required_cases": cases,
        "all_required_cases_passed": bool(all_cases),
        "familywise_false_positive_rate": float(fpr),
        "corrected_power_at_0_075": float(power),
        "replicates": int(replicates),
        "bootstrap_draws_per_fixture": int(n_bootstrap),
        "seed": int(seed),
        "planted_effect": 0.075,
    }


def validate_fixture_calibration(value: Mapping[str, Any]) -> dict[str, Any]:
    cases = value.get("required_cases")
    valid_cases = bool(
        isinstance(cases, Mapping)
        and set(cases) == set(FIXTURE_CASES)
        and all(
            isinstance(cases[name], Mapping) and type(cases[name].get("passed")) is bool
            for name in FIXTURE_CASES
        )
    )
    valid = bool(
        value.get("schema") == FIXTURE_SCHEMA
        and value.get("analysis_engine")
        == "analyze_p3_register_relocation.analyze_payload"
        and value.get("decision_engine")
        == "pbc_register_relocation.adjudicate_register_relocation"
        and valid_cases
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


def _path_resolution(
    evidence: Mapping[str, Any] | None,
    audit: Mapping[str, Any] | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    if evidence is None or audit is None:
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


def _capability_from_raw(payload: Mapping[str, Any]) -> bool:
    mechanics = payload.get("mechanics")
    health = payload.get("health")
    construction = payload.get("construction_binding")
    if not all(
        isinstance(value, Mapping) for value in (mechanics, health, construction)
    ):
        return False
    mode = str(payload.get("mode"))
    if mode not in {"P3_ONLY", "JOINT"}:
        return False
    expected = _expected_health_labels(mode)
    states = health.get("states")
    authorized = health.get("authorized_table_health")
    tables = authorized.get("tables") if isinstance(authorized, Mapping) else None
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
        and health.get("schema") == "gfmcd.pbc.p3_register_table_health.v1"
        and health.get("stage") == "development"
        and isinstance(states, Mapping)
        and set(states) == expected
        and all(
            isinstance(row, Mapping) and row.get("passed") is True
            for row in states.values()
        )
        and isinstance(authorized, Mapping)
        and authorized.get("schema")
        == "gfmcd.pbc.carrier_factorial_table_health.v1"
        and authorized.get("target_mode")
        == ("REGISTER_JOINT" if mode == "JOINT" else "REGISTER_P3_ONLY")
        and authorized.get("candidate_package_sha256") == candidate_hash
        and int(authorized.get("n_blocks", -1)) == DEVELOPMENT_BLOCKS
        and int(authorized.get("n_tables", -1)) == len(expected)
        and authorized.get("all_tables_pass") is True
        and isinstance(tables, Mapping)
        and set(tables) == expected
        and all(
            isinstance(row, Mapping)
            and row.get("pass") is True
            and isinstance(row.get("raw_blocks"), list)
            and len(row["raw_blocks"]) == DEVELOPMENT_BLOCKS
            for row in tables.values()
        )
        and health.get("all_control_states_passed") is True
        and health.get("all_required_states_passed") is True
        and float(health.get("minimum_p10_reader_retention", -math.inf)) >= 0.80
    )


def _relocation_payload(payload: Mapping[str, Any], *, preflight_sha256: str) -> dict[str, Any]:
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("raw preflight lacks its Phase-II authorization")
    p3 = authorization.get("phase2_p3_authorized")
    joint = authorization.get("phase2_joint_authorized")
    if type(p3) is not bool or type(joint) is not bool:
        raise ValueError("raw Phase-II authorization flags must be booleans")
    mode = str(payload.get("mode"))
    if mode not in {"P3_ONLY", "JOINT"}:
        raise ValueError("raw preflight mode must be P3_ONLY or JOINT")
    mechanics = payload.get("mechanics")
    if not isinstance(mechanics, Mapping):
        raise ValueError("raw preflight lacks mechanics")
    return {
        "schema": TARGET_SCHEMA,
        "status": "COMPLETE",
        "stage": "development",
        "mode": mode,
        "authorization": {
            "phase2_p3_authorized": bool(p3),
            "phase2_joint_authorized": bool(joint),
            "register_preflight_passed": _capability_from_raw(payload),
        },
        "design": {
            "jreg_direction": authorization.get("jreg_direction") if mode == "JOINT" else None
        },
        "mechanics": {
            "register_zero_no_hook_max_abs": float(
                mechanics.get("register_zero_no_hook_max_abs", math.inf)
            )
        },
        "rows": payload.get("rows"),
        "self_sha256": preflight_sha256,
    }


def analyze_preflight(
    payload: Mapping[str, Any],
    *,
    preflight_sha256: str,
    path_mde_evidence: Mapping[str, Any] | None = None,
    path_mde_audit: Mapping[str, Any] | None = None,
    fixture_calibration: Mapping[str, Any] | None = None,
    seed: int = 2026081013,
    n_bootstrap: int = 20_000,
) -> dict[str, Any]:
    """Run real register inference, power projection, fixtures, and the stage gate."""

    if payload.get("schema") != REGISTER_PREFLIGHT_SCHEMA or payload.get("status") != RAW_STATUS:
        raise ValueError("raw register preflight is incomplete or has the wrong schema")
    if payload.get("stage") != "development" or int(payload.get("n_blocks", -1)) != DEVELOPMENT_BLOCKS:
        raise ValueError("register preflight must contain the exact 120-block development role")
    relocation = analyze_relocation_payload(
        _relocation_payload(payload, preflight_sha256=preflight_sha256),
        seed=int(seed),
        n_bootstrap=int(n_bootstrap),
    )
    per_block = relocation["target"]["per_block"]
    endpoint_names = (
        "omega3_intact",
        "s_-1_intact",
        "s_0_intact",
        "s_1_intact",
    )
    resolution = projected_mde80(
        per_block,
        endpoint_names=endpoint_names,
        reserve_blocks=RESERVE_BLOCKS,
    )
    path_values, path_record = _path_resolution(path_mde_evidence, path_mde_audit)
    resolution.update(path_values)
    if str(payload.get("mode")) == "JOINT" and all("jreg" in row for row in per_block):
        resolution.update(
            projected_mde80(
                per_block,
                endpoint_names=("jreg",),
                reserve_blocks=RESERVE_BLOCKS,
            )
        )
    else:
        resolution["jreg"] = UNRESOLVED_MDE

    calibration = validate_fixture_calibration(
        fixture_calibration
        if fixture_calibration is not None
        else calibrate_fixture_suite()
    )
    mechanics = payload.get("mechanics")
    health = payload.get("health")
    authorization = payload.get("authorization")
    assert isinstance(mechanics, Mapping)
    assert isinstance(health, Mapping)
    assert isinstance(authorization, Mapping)
    # Preserve the full exact-table family in the analyzed artifact.  The explicit
    # family gate is folded into every row so no failed control can authorize reserve.
    stage_health = dict(health)
    raw_states = health.get("states")
    control_health_pass = health.get("all_control_states_passed") is True
    if isinstance(raw_states, Mapping):
        stage_health["states"] = {
            str(label): {
                **dict(row),
                "passed": bool(
                    isinstance(row, Mapping)
                    and row.get("passed") is True
                    and control_health_pass
                ),
            }
            for label, row in raw_states.items()
            if isinstance(row, Mapping)
        }
    decision = register_preflight_decision(
        development_analysis=relocation,
        construction_mechanics=mechanics,
        health=stage_health,
        resolution=resolution,
        fixture_calibration=calibration,
        phase2_authorization=authorization,
    )
    return {
        **decision,
        "preflight_sha256": preflight_sha256,
        "inference_seed": int(seed),
        "partial_specificity_seeds": {
            "expected": PARTIAL_SPECIFICITY_SEED,
            "opposite": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
        },
        "bootstrap_draws": int(n_bootstrap),
        "development_relocation": relocation,
        "construction_mechanics": dict(mechanics),
        "health": stage_health,
        "projected_resolution": {
            "reserve_blocks": RESERVE_BLOCKS,
            "mde80": resolution,
            "path_evidence": path_record,
            "missing_path_evidence_is_unresolved": True,
        },
        "path_mde_evidence": None if path_mde_evidence is None else dict(path_mde_evidence),
        "path_mde_audit": None if path_mde_audit is None else dict(path_mde_audit),
        "fixture_calibration": calibration,
        "outcome_firewall": {
            "phase3_reserve_read": False,
            "phase3_reserve_authorized_only_by_this_analysis_and_independent_audit": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--path-mde")
    parser.add_argument("--path-mde-audit")
    parser.add_argument("--fixture-calibration")
    parser.add_argument("--seed", type=int, default=2026081013)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preflight_path = Path(args.preflight).expanduser().resolve()
    output, manifest_path = output_paths(args.out)
    payload = json.loads(preflight_path.read_text())
    path_evidence = None
    path_audit = None
    fixture = None
    inputs = [preflight_path]
    if args.path_mde:
        if not args.path_mde_audit:
            raise RuntimeError("--path-mde requires --path-mde-audit")
        path = Path(args.path_mde).expanduser().resolve()
        path_evidence = json.loads(path.read_text())
        path_evidence["self_sha256"] = sha256_file(path)
        inputs.append(path)
        audit_path = Path(args.path_mde_audit).expanduser().resolve()
        path_audit = json.loads(audit_path.read_text())
        inputs.append(audit_path)
    elif args.path_mde_audit:
        raise RuntimeError("--path-mde-audit requires --path-mde")
    if args.fixture_calibration:
        path = Path(args.fixture_calibration).expanduser().resolve()
        fixture = json.loads(path.read_text())
        inputs.append(path)
    analysis = analyze_preflight(
        payload,
        preflight_sha256=sha256_file(preflight_path),
        path_mde_evidence=path_evidence,
        path_mde_audit=path_audit,
        fixture_calibration=fixture,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        experiment_id="pbc-p3-register-development-analysis",
        command=sys.argv,
        repo=REPO,
        status="ANALYSIS",
        config={
            "schema": REGISTER_PREFLIGHT_ANALYSIS_SCHEMA,
            "stage": "development",
            "seed": args.seed,
            "expected_partial_specificity_seed": PARTIAL_SPECIFICITY_SEED,
            "opposite_partial_specificity_seed": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
            "bootstrap_draws": args.bootstrap,
            "reserve_blocks_for_mde": RESERVE_BLOCKS,
        },
        seeds={
            "whole_block_bootstrap": args.seed,
            "expected_partial_specificity": PARTIAL_SPECIFICITY_SEED,
            "opposite_partial_specificity": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
        },
        inputs=[file_record(path) for path in inputs],
        outputs=[file_record(output)],
        notes=[
            "Real relocation analysis reused unchanged; absent D/R/Q path MDE evidence is explicitly unresolved."
        ],
    )
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "authorize_p3_register_reserve": analysis["authorization"][
                    "authorize_p3_register_reserve"
                ],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
