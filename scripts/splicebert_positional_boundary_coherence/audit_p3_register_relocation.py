#!/usr/bin/env python
"""Independent arithmetic, lattice, provenance, and decision audit for Phase III."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_register_relocation import (  # noqa: E402
    P10_INTACT,
    REGISTERS,
    analyze_register_component,
    block_register_contrasts,
    paired_named_control_specificity,
    paired_register_specificity_interval,
    phase_delta,
    register_specificity,
)
from gfmcd.splicing.pbc_register_stage import (  # noqa: E402
    file_record_matches,
    manifest_binds_path,
    manifest_precedes,
    validate_sibling_manifest,
    verify_register_target_seal,
)


AUDIT_SCHEMA = "gfmcd.pbc.p3_register_audit.v1"
TARGET_SCHEMA = "gfmcd.pbc.p3_register_target.v1"
ANALYSIS_SCHEMA = "gfmcd.pbc.p3_register_analysis.v1"
INFERENCE_SEED = 2026081006
CONFIRMATORY_BOOTSTRAP_DRAWS = 20_000
CONFIRMATORY_BLOCKS = 384
CONFIRMATORY_CELLS = CONFIRMATORY_BLOCKS * 9
CONTROL_LABELS = tuple(f"p3_signed_{index:02d}" for index in range(31))
DIAGNOSTIC_LABELS = ("p3_common_mode", "p3_decoy_span")


def _file_record_valid(record: object) -> bool:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        return False
    path = Path(str(record["path"]))
    return (
        path.is_file()
        and type(record.get("size_bytes")) is int
        and path.stat().st_size == int(record["size_bytes"])
        and record.get("sha256") == sha256_file(path)
    )


def validate_confirmatory_provenance(
    target: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    target_path: Path,
    target_manifest: Mapping[str, Any],
    target_manifest_path: Path,
    analysis_path: Path,
    analysis_manifest: Mapping[str, Any],
    analysis_manifest_path: Path,
    audit_output_path: Path,
) -> tuple[dict[str, bool], tuple[Path, Path]]:
    """Reopen the reserve seal and rederive chronology and frozen path bindings."""

    sources = target.get("sources")
    seal_record = sources.get("reserve_seal") if isinstance(sources, Mapping) else None
    seal_manifest_record = (
        sources.get("reserve_seal_manifest") if isinstance(sources, Mapping) else None
    )
    if not (
        isinstance(seal_record, Mapping)
        and isinstance(seal_manifest_record, Mapping)
        and file_record_matches(seal_record)
        and file_record_matches(seal_manifest_record)
    ):
        raise RuntimeError("confirmatory relocation audit lacks its sealed provenance")
    seal_path = Path(str(seal_record["path"])).expanduser().resolve()
    seal_manifest_path = Path(
        str(seal_manifest_record["path"])
    ).expanduser().resolve()
    seal = json.loads(seal_path.read_text())
    seal_manifest = json.loads(seal_manifest_path.read_text())
    verify_register_target_seal(seal)
    seal_validation = validate_sibling_manifest(
        seal_manifest,
        manifest_path=seal_manifest_path,
        artifact_path=seal_path,
    )
    target_validation = validate_sibling_manifest(
        target_manifest,
        manifest_path=target_manifest_path,
        artifact_path=target_path,
    )
    analysis_validation = validate_sibling_manifest(
        analysis_manifest,
        manifest_path=analysis_manifest_path,
        artifact_path=analysis_path,
    )
    relocation = seal["relocation_outputs"]
    recorded = analysis.get("reserve_provenance")
    checks = {
        "seal_manifest_valid": bool(seal_validation["passed"]),
        "target_manifest_valid": bool(target_validation["passed"]),
        "analysis_manifest_valid": bool(analysis_validation["passed"]),
        "target_manifest_consumes_seal": manifest_binds_path(
            target_manifest, seal_path, field="inputs"
        ),
        "target_manifest_consumes_seal_manifest": manifest_binds_path(
            target_manifest, seal_manifest_path, field="inputs"
        ),
        "analysis_manifest_consumes_seal": manifest_binds_path(
            analysis_manifest, seal_path, field="inputs"
        ),
        "analysis_manifest_consumes_seal_manifest": manifest_binds_path(
            analysis_manifest, seal_manifest_path, field="inputs"
        ),
        "analysis_manifest_consumes_target": manifest_binds_path(
            analysis_manifest, target_path, field="inputs"
        ),
        "analysis_manifest_consumes_target_manifest": manifest_binds_path(
            analysis_manifest, target_manifest_path, field="inputs"
        ),
        "seal_precedes_target": manifest_precedes(seal_manifest, target_manifest),
        "target_precedes_analysis": manifest_precedes(
            target_manifest, analysis_manifest
        ),
        "single_code_revision": len(
            {
                str(manifest.get("git_commit"))
                for manifest in (seal_manifest, target_manifest, analysis_manifest)
            }
        )
        == 1,
        "target_seal_hash_bound": target.get("seal_sha256") == seal.get("seal_sha256"),
        "target_path_frozen": target_path.resolve()
        == Path(str(relocation["target"])).resolve(),
        "analysis_path_frozen": analysis_path.resolve()
        == Path(str(relocation["analysis"])).resolve(),
        "audit_path_frozen": audit_output_path.resolve()
        == Path(str(relocation["audit"])).resolve(),
        "confirmatory_inference_frozen": int(relocation["inference_seed"])
        == INFERENCE_SEED
        and int(relocation["bootstrap_draws"])
        == CONFIRMATORY_BOOTSTRAP_DRAWS,
        "analysis_recorded_seal_chain": bool(
            isinstance(recorded, Mapping)
            and recorded.get("reserve_seal_sha256") == seal_record.get("sha256")
            and recorded.get("reserve_seal_manifest_sha256")
            == seal_manifest_record.get("sha256")
            and recorded.get("target_manifest_sha256")
            == sha256_file(target_manifest_path)
            and isinstance(recorded.get("checks"), Mapping)
            and all(value is True for value in recorded["checks"].values())
        ),
    }
    if not all(checks.values()):
        failures = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"confirmatory relocation provenance failed: {failures}")
    return checks, (seal_path, seal_manifest_path)


def _row_contract(
    target: Mapping[str, object], *, enforce_confirmatory_contract: bool
) -> dict[str, object]:
    rows = target.get("rows")
    if not isinstance(rows, list) or not rows:
        return {"passed": False, "reason": "rows absent"}
    mode = str(target.get("mode"))
    states = {"intact", "attenuated"} if mode == "JOINT" else {"intact"}
    labels = {"p3", *CONTROL_LABELS, *DIAGNOSTIC_LABELS}
    keys = set()
    blocks = set()
    arithmetic_max = 0.0
    phase_valid = True
    finite = True
    for row in rows:
        if not isinstance(row, Mapping):
            return {"passed": False, "reason": "non-object row"}
        try:
            key = (
                str(row["component"]),
                str(row["p10_state"]),
                int(row["register"]),
                str(row["block_id"]),
                int(row["donor_phase"]),
                int(row["acceptor_phase"]),
            )
            values = [
                float(row["intact_margin"]),
                float(row["corrupt_margin"]),
                float(row["coupling"]),
            ]
            finite &= all(math.isfinite(value) for value in values)
            arithmetic_max = max(arithmetic_max, abs(values[2] - (values[0] - values[1])))
            phase_valid &= int(row["phase_delta"]) == phase_delta(key[4], key[5])
        except (KeyError, TypeError, ValueError):
            return {"passed": False, "reason": "row field invalid"}
        keys.add(key)
        blocks.add(key[3])
    expected = {
        (component, state, register, block, donor, acceptor)
        for component in labels
        for state in states
        for register in REGISTERS
        for block in blocks
        for donor in range(3)
        for acceptor in range(3)
    }
    confirmatory_counts = bool(
        not enforce_confirmatory_contract
        or (
            len(blocks) == CONFIRMATORY_BLOCKS
            and target.get("stage") == "phase3_reserve"
            and target.get("n_blocks") == CONFIRMATORY_BLOCKS
            and target.get("n_cells") == CONFIRMATORY_CELLS
        )
    )
    return {
        "passed": bool(
            len(keys) == len(rows)
            and keys == expected
            and finite
            and phase_valid
            and arithmetic_max <= 1e-8
            and confirmatory_counts
        ),
        "n_rows": len(rows),
        "n_blocks": len(blocks),
        "unique_keys": len(keys),
        "expected_rows": len(expected),
        "all_finite": bool(finite),
        "phase_convention_valid": bool(phase_valid),
        "coupling_arithmetic_max_abs": float(arithmetic_max),
        "confirmatory_counts": confirmatory_counts,
    }


def _independent_status(
    target: Mapping[str, object],
    specificity: Mapping[str, object],
    authorization: Mapping[str, object],
    *,
    replay_passed: bool,
    mode: str,
    design: Mapping[str, object],
) -> str:
    """Reapply the frozen hierarchy without importing the analysis verdict function."""

    if authorization.get("phase2_p3_authorized") is not True:
        return "P3_REGISTER_NOT_AUTHORIZED"
    if authorization.get("register_preflight_passed") is not True or not replay_passed:
        return "P3_REGISTER_INSTRUMENT_UNRESOLVED"
    if specificity.get("passed") is not True:
        return "P3_REGISTER_INSTRUMENT_UNRESOLVED"
    if not bool(target.get("all_registers_positive")) or not bool(
        target.get("omega3_material_positive")
    ):
        if bool(target.get("opposite_all_registers_positive")) and bool(
            target.get("opposite_omega3_material_positive")
        ):
            return "P3_REGISTER_HANDEDNESS_FALSIFIED"
        if bool(target.get("all_registers_bounded_null")):
            return "P3_REGISTER_BOUNDED_NULL"
        return "P3_REGISTER_INCONCLUSIVE"
    base = "P3_REGISTER_RELOCATION_SUPPORTED"
    if mode != "JOINT":
        return base
    if authorization.get("phase2_joint_authorized") is not True:
        return base + "_P10_NOT_AUTHORIZED"
    if not bool(target.get("all_joint_state_registers_positive")):
        return base + "_P10_MODIFIER_UNRESOLVED"
    direction = design.get("jreg_direction")
    inference = target.get("optional_modifier_inference")
    if direction not in {"positive", "negative"} or not isinstance(inference, Mapping):
        raise ValueError("joint target lacks the frozen Jreg direction/inference")
    lower = float(inference["lower"]["jreg"])
    upper = float(inference["upper"]["jreg"])
    modified = lower > 0.05 if direction == "positive" else upper < -0.05
    return base + ("_P10_MODIFIED" if modified else "_P10_MODIFIER_UNRESOLVED")


def _independent_recompute(
    target: Mapping[str, object],
    analysis: Mapping[str, object],
) -> dict[str, object]:
    """Recompute estimators and controls while treating analysis prose as untrusted."""

    rows = target["rows"]
    if not isinstance(rows, list):
        raise ValueError("target rows are absent")
    mode = str(target["mode"])
    joint = mode == "JOINT"
    seed = int(analysis["inference_seed"])
    draws = int(analysis["bootstrap_draws"])
    expected = tuple(sorted(("p3", *CONTROL_LABELS, *DIAGNOSTIC_LABELS)))
    states = ("intact", "attenuated") if joint else ("intact",)
    primary = analyze_register_component(
        rows,
        component="p3",
        joint=joint,
        seed=seed,
        n_bootstrap=draws,
    )
    component_blocks = {
        label: block_register_contrasts(rows, component=label, p10_states=states)
        for label in expected
    }
    compact = {}
    controls = {}
    for label, blocks in component_blocks.items():
        omega = {
            state: float(
                sum(float(row[f"omega3_{state}"]) for row in blocks) / len(blocks)
            )
            for state in states
        }
        if label == "p3" or label in CONTROL_LABELS:
            compact[label] = {"omega3": omega}
        if label in CONTROL_LABELS:
            controls[label] = {"n_blocks": len(blocks), "omega3": omega}
    rank = register_specificity(compact)
    paired = paired_register_specificity_interval(
        {label: component_blocks[label] for label in ("p3", *CONTROL_LABELS)},
        seed=seed + 1,
        n_bootstrap=draws,
    )
    diagnostics = {
        label: analyze_register_component(
            rows,
            component=label,
            joint=joint,
            seed=seed + 10 + index * 3,
            n_bootstrap=draws,
        )
        for index, label in enumerate(DIAGNOSTIC_LABELS)
    }
    named = paired_named_control_specificity(
        component_blocks,
        control_components=DIAGNOSTIC_LABELS,
        seed=seed + 30,
        n_bootstrap=draws,
    )
    fail_to_reproduce = all(
        not bool(diagnostics[label]["all_registers_positive"])
        for label in DIAGNOSTIC_LABELS
    )
    specificity = {
        **rank,
        "paired_target_minus_mean_control": paired,
        "paired_target_minus_named_controls": named,
        "diagnostics_fail_to_reproduce": bool(fail_to_reproduce),
        "passed": bool(
            rank["passed"] and paired["passed"] and named["passed"] and fail_to_reproduce
        ),
    }
    mechanics = target["mechanics"]
    replay_error = float(mechanics["register_zero_no_hook_max_abs"])
    replay = {
        "max_abs_cell_endpoint_error": replay_error,
        "threshold": 1e-8,
        "passed": replay_error <= 1e-8,
    }
    authorization = target["authorization"]
    design = target["design"]
    status = _independent_status(
        primary,
        specificity,
        authorization,
        replay_passed=bool(replay["passed"]),
        mode=mode,
        design=design,
    )
    return {
        "stage": str(target.get("stage") or "development"),
        "status": status,
        "mode": mode,
        "target": primary,
        "controls": controls,
        "diagnostic_controls": diagnostics,
        "specificity": specificity,
        "register_zero_replay": replay,
        "authorization": dict(authorization),
    }


def _mechanics_contract(target: Mapping[str, object]) -> dict[str, object]:
    mechanics = target.get("mechanics")
    design = target.get("design")
    if not isinstance(mechanics, Mapping) or not isinstance(design, Mapping):
        return {"passed": False, "reason": "mechanics/design absent"}
    construction = mechanics.get("construction_invariants")
    specs = mechanics.get("component_specs")
    rms = mechanics.get("carrier_delta_rms")
    labels = ("p3", *CONTROL_LABELS, *DIAGNOSTIC_LABELS)
    rms_error = 0.0
    rms_complete = isinstance(rms, Mapping)
    if rms_complete:
        try:
            for register in ("-1", "1"):
                target_rms = float(rms["p3"][register])
                for label in (*CONTROL_LABELS, *DIAGNOSTIC_LABELS):
                    rms_error = max(rms_error, abs(float(rms[label][register]) - target_rms))
        except (KeyError, TypeError, ValueError):
            rms_complete = False
    spec_pass = bool(
        isinstance(specs, Mapping)
        and set(specs) == set(labels)
        and all(specs[label].get("rms_match") is True for label in (*CONTROL_LABELS, *DIAGNOSTIC_LABELS))
        and specs["p3"].get("rms_match") is False
    )
    passed = bool(
        isinstance(construction, Mapping)
        and construction.get("passed") is True
        and float(mechanics.get("register_zero_no_hook_max_abs", math.inf)) <= 1e-8
        and design.get("direction") == "right"
        and design.get("span_token") == [393, 513]
        and design.get("decoy_span_token") == [153, 273]
        and len(design.get("control_seeds", [])) == 31
        and spec_pass
        and rms_complete
        and rms_error <= 1e-8
    )
    return {
        "passed": passed,
        "construction_passed": bool(
            isinstance(construction, Mapping) and construction.get("passed") is True
        ),
        "component_specs_passed": spec_pass,
        "carrier_rms_complete": rms_complete,
        "carrier_rms_max_abs_error": float(rms_error),
    }
def audit_payloads(
    target: Mapping[str, object],
    analysis: Mapping[str, object],
    *,
    target_sha256: str,
    enforce_confirmatory_contract: bool = True,
    analysis_sha256: str | None = None,
    provenance_checks: Mapping[str, bool] | None = None,
) -> dict[str, object]:
    checks: dict[str, bool] = {
        "target_schema_and_stage": target.get("schema") == TARGET_SCHEMA
        and target.get("status") == "COMPLETE"
        and (
            not enforce_confirmatory_contract
            or target.get("stage") == "phase3_reserve"
        ),
        "analysis_schema_and_stage": analysis.get("schema") == ANALYSIS_SCHEMA
        and (
            not enforce_confirmatory_contract
            or analysis.get("stage") == "phase3_reserve"
        ),
        "target_hash_bound": analysis.get("target_sha256") == target_sha256,
        "confirmatory_seed": bool(
            not enforce_confirmatory_contract
            or analysis.get("inference_seed") == INFERENCE_SEED
        ),
        "confirmatory_draws": bool(
            not enforce_confirmatory_contract
            or analysis.get("bootstrap_draws") == CONFIRMATORY_BOOTSTRAP_DRAWS
        ),
        "analysis_sha256_present": bool(
            not enforce_confirmatory_contract
            or (
                isinstance(analysis_sha256, str)
                and len(analysis_sha256) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in analysis_sha256
                )
            )
        ),
        "reserve_provenance_recomputed": bool(
            not enforce_confirmatory_contract
            or (
                isinstance(provenance_checks, Mapping)
                and provenance_checks
                and all(value is True for value in provenance_checks.values())
            )
        ),
    }
    sources = target.get("sources")
    checks["source_records_present"] = isinstance(sources, Mapping) and set(sources) >= {
        "cohort",
        "phase2_analysis",
        "phase2_audit",
        "register_preflight",
        "register_preflight_audit",
    }
    checks["source_hashes_valid"] = bool(
        isinstance(sources, Mapping)
        and sources
        and all(_file_record_valid(record) for record in sources.values())
    )
    shard_sources = target.get("shard_sources")
    checks["shard_hashes_valid"] = bool(
        shard_sources is None
        or (
            isinstance(shard_sources, list)
            and shard_sources
            and all(_file_record_valid(record) for record in shard_sources)
        )
    )
    row_contract = _row_contract(
        target, enforce_confirmatory_contract=enforce_confirmatory_contract
    )
    checks["row_contract"] = bool(row_contract.get("passed"))
    mechanics_contract = _mechanics_contract(target)
    checks["mechanics_contract"] = bool(mechanics_contract.get("passed"))
    recompute_error = None
    recomputed = None
    try:
        recomputed = _independent_recompute(target, analysis)
        checks["analysis_statistics_recomputed"] = all(
            recomputed[key] == analysis.get(key)
            for key in (
                "status",
                "stage",
                "mode",
                "target",
                "controls",
                "diagnostic_controls",
                "specificity",
                "register_zero_replay",
                "authorization",
            )
        )
    except Exception as exc:  # audit must record a failure rather than erase it
        recompute_error = f"{type(exc).__name__}: {exc}"
        checks["analysis_statistics_recomputed"] = False
    passed = all(checks.values())
    return {
        "schema": AUDIT_SCHEMA,
        "stage": "phase3_reserve",
        "status": "PASS" if passed else "FAIL",
        "target_sha256": target_sha256,
        "analysis_sha256": analysis_sha256,
        "checks": checks,
        "row_contract": row_contract,
        "mechanics_contract": mechanics_contract,
        "recompute_error": recompute_error,
        "recomputed_status": None if recomputed is None else recomputed["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    target_path = Path(args.target).resolve()
    target_manifest_path = Path(args.target_manifest).resolve()
    analysis_path = Path(args.analysis).resolve()
    analysis_manifest_path = Path(args.analysis_manifest).resolve()
    output_path = Path(args.out).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite register audit: {output_path}")
    target = json.loads(target_path.read_text())
    target_manifest = json.loads(target_manifest_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    analysis_manifest = json.loads(analysis_manifest_path.read_text())
    provenance, provenance_paths = validate_confirmatory_provenance(
        target,
        analysis,
        target_path=target_path,
        target_manifest=target_manifest,
        target_manifest_path=target_manifest_path,
        analysis_path=analysis_path,
        analysis_manifest=analysis_manifest,
        analysis_manifest_path=analysis_manifest_path,
        audit_output_path=output_path,
    )
    audit = audit_payloads(
        target,
        analysis,
        target_sha256=sha256_file(target_path),
        analysis_sha256=sha256_file(analysis_path),
        provenance_checks=provenance,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        experiment_id="pbc-p3-register-audit",
        command=sys.argv,
        repo=REPO,
        status="AUDIT",
        config={"stage": "phase3_reserve", "schema": AUDIT_SCHEMA},
        inputs=[
            file_record(path)
            for path in (
                *provenance_paths,
                target_path,
                target_manifest_path,
                analysis_path,
                analysis_manifest_path,
            )
        ],
        outputs=[file_record(output_path)],
        notes=["Independent raw-row lattice and arithmetic audit plus deterministic decision recomputation."],
    )
    manifest.write(manifest_path)
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
