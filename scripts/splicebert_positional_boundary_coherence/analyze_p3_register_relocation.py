#!/usr/bin/env python
"""Analyze a sealed P3 register-relocation target without loading the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_register_relocation import (  # noqa: E402
    OPPOSITE_PARTIAL_SPECIFICITY_SEED,
    P10_INTACT,
    PARTIAL_SPECIFICITY_SEED,
    adjudicate_register_relocation,
    analyze_register_component,
    block_register_contrasts,
    opposite_partial_handedness_specificity_family,
    partial_handedness_specificity_family,
    paired_named_control_specificity,
    paired_register_specificity_interval,
    register_specificity,
)
from gfmcd.splicing.pbc_register_stage import (  # noqa: E402
    file_record_matches,
    manifest_binds_path,
    manifest_precedes,
    validate_sibling_manifest,
    verify_register_target_seal,
)


TARGET_SCHEMA = "gfmcd.pbc.p3_register_target.v1"
ANALYSIS_SCHEMA = "gfmcd.pbc.p3_register_analysis.v1"
INFERENCE_SEED = 2026081006
CONFIRMATORY_BOOTSTRAP_DRAWS = 20_000
CONFIRMATORY_BLOCKS = 384
CONFIRMATORY_CELLS = CONFIRMATORY_BLOCKS * 9
CONTROL_LABELS = tuple(f"p3_signed_{index:02d}" for index in range(31))
DIAGNOSTIC_LABELS = ("p3_common_mode", "p3_decoy_span")


def _require_bool(record: Mapping[str, object], key: str) -> bool:
    value = record.get(key)
    if type(value) is not bool:
        raise ValueError(f"authorization.{key} must be boolean")
    return bool(value)


def validate_confirmatory_provenance(
    target: Mapping[str, Any],
    *,
    target_path: Path,
    target_manifest: Mapping[str, Any],
    target_manifest_path: Path,
    analysis_output_path: Path,
) -> tuple[Mapping[str, Any], dict[str, Any], tuple[Path, Path]]:
    """Reopen the reserve seal and validate target chronology/frozen output paths."""

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
        raise RuntimeError("confirmatory relocation target lacks its sealed provenance")
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
    relocation = seal["relocation_outputs"]
    checks = {
        "seal_manifest_valid": bool(seal_validation["passed"]),
        "target_manifest_valid": bool(target_validation["passed"]),
        "target_manifest_consumes_seal": manifest_binds_path(
            target_manifest, seal_path, field="inputs"
        ),
        "target_manifest_consumes_seal_manifest": manifest_binds_path(
            target_manifest, seal_manifest_path, field="inputs"
        ),
        "seal_precedes_target": manifest_precedes(seal_manifest, target_manifest),
        "single_code_revision": seal_manifest.get("git_commit")
        == target_manifest.get("git_commit"),
        "target_seal_hash_bound": target.get("seal_sha256") == seal.get("seal_sha256"),
        "target_path_frozen": target_path.resolve()
        == Path(str(relocation["target"])).resolve(),
        "analysis_path_frozen": analysis_output_path.resolve()
        == Path(str(relocation["analysis"])).resolve(),
        "confirmatory_inference_frozen": int(relocation["inference_seed"])
        == INFERENCE_SEED
        and int(relocation["bootstrap_draws"])
        == CONFIRMATORY_BOOTSTRAP_DRAWS,
        "partial_specificity_seeds_frozen": int(
            relocation["expected_partial_specificity_seed"]
        )
        == PARTIAL_SPECIFICITY_SEED
        and int(relocation["opposite_partial_specificity_seed"])
        == OPPOSITE_PARTIAL_SPECIFICITY_SEED,
    }
    if not all(checks.values()):
        failures = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"confirmatory relocation provenance failed: {failures}")
    return (
        seal,
        {
            "checks": checks,
            "reserve_seal_sha256": str(seal_record["sha256"]),
            "reserve_seal_manifest_sha256": str(seal_manifest_record["sha256"]),
            "target_manifest_sha256": sha256_file(target_manifest_path),
        },
        (seal_path, seal_manifest_path),
    )


def analyze_payload(
    payload: Mapping[str, object],
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, object]:
    """Recompute target, controls, specificity, and the asymmetric decision."""

    if payload.get("schema") != TARGET_SCHEMA or payload.get("status") != "COMPLETE":
        raise ValueError("register target is incomplete or has the wrong schema")
    stage = str(payload.get("stage") or "development")
    if stage not in {"development", "phase3_reserve"}:
        raise ValueError("register target has an invalid stage")
    mode = str(payload.get("mode"))
    if mode not in {"P3_ONLY", "JOINT"}:
        raise ValueError("register target mode must be P3_ONLY or JOINT")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("register target has no rows")
    labels = sorted({str(row.get("component")) for row in rows if isinstance(row, Mapping)})
    expected = sorted(("p3", *CONTROL_LABELS, *DIAGNOSTIC_LABELS))
    if labels != expected:
        raise ValueError("register target does not contain the frozen 31-control family")
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("register target lacks authorization")
    p3_authorized = _require_bool(authorization, "phase2_p3_authorized")
    joint_authorized = _require_bool(authorization, "phase2_joint_authorized")
    capability_passed = _require_bool(authorization, "register_preflight_passed")
    if mode == "JOINT" and not joint_authorized:
        raise ValueError("joint target was opened without joint Phase-III authorization")

    joint = mode == "JOINT"
    target = analyze_register_component(
        rows,
        component="p3",
        joint=joint,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    states = ("intact", "attenuated") if joint else ("intact",)
    component_blocks = {
        label: block_register_contrasts(rows, component=label, p10_states=states)
        for label in expected
    }
    compact_analyses: dict[str, dict[str, object]] = {}
    opposite_compact_analyses: dict[str, dict[str, object]] = {}
    controls: dict[str, dict[str, object]] = {}
    for label, blocks in component_blocks.items():
        omega = {
            state: float(
                sum(float(row[f"omega3_{state}"]) for row in blocks) / len(blocks)
            )
            for state in states
        }
        opposite_omega = {
            state: float(
                sum(
                    float(row[f"opposite_omega3_{state}"]) for row in blocks
                )
                / len(blocks)
            )
            for state in states
        }
        if label == "p3" or label in CONTROL_LABELS:
            compact_analyses[label] = {"omega3": omega}
            opposite_compact_analyses[label] = {"omega3": opposite_omega}
        if label in CONTROL_LABELS:
            controls[label] = {
                "n_blocks": len(blocks),
                "omega3": omega,
                "opposite_omega3": opposite_omega,
            }
    rank_specificity = register_specificity(compact_analyses)
    paired_specificity = paired_register_specificity_interval(
        {label: component_blocks[label] for label in ("p3", *CONTROL_LABELS)},
        seed=seed + 1,
        n_bootstrap=n_bootstrap,
    )
    opposite_rank_specificity = register_specificity(opposite_compact_analyses)
    opposite_paired_specificity = paired_register_specificity_interval(
        {label: component_blocks[label] for label in ("p3", *CONTROL_LABELS)},
        seed=seed + 2,
        n_bootstrap=n_bootstrap,
        endpoint=f"opposite_omega3_{P10_INTACT}",
    )
    diagnostics = {
        label: analyze_register_component(
            rows,
            component=label,
            joint=joint,
            seed=seed + 10 + index * 3,
            n_bootstrap=n_bootstrap,
        )
        for index, label in enumerate(DIAGNOSTIC_LABELS)
    }
    diagnostic_specificity = paired_named_control_specificity(
        component_blocks,
        control_components=DIAGNOSTIC_LABELS,
        seed=seed + 30,
        n_bootstrap=n_bootstrap,
    )
    partial_family = partial_handedness_specificity_family(
        component_blocks,
        control_components=DIAGNOSTIC_LABELS,
        seed=PARTIAL_SPECIFICITY_SEED,
        n_bootstrap=n_bootstrap,
    )
    opposite_partial_family = opposite_partial_handedness_specificity_family(
        component_blocks,
        control_components=DIAGNOSTIC_LABELS,
        seed=OPPOSITE_PARTIAL_SPECIFICITY_SEED,
        n_bootstrap=n_bootstrap,
    )
    target["handedness_specificity_family"] = partial_family
    target["opposite_handedness_specificity_family"] = opposite_partial_family
    target["handedness_expected_lower_positive"] = bool(
        partial_family["handedness_lower_positive"]
    )
    target["handedness_opposite_lower_positive"] = bool(
        opposite_partial_family["reverse_handedness_lower_positive"]
    )
    target["partial_concordance_pattern"] = bool(
        target["expected_partial_concordance_pattern"]
        and target["handedness_expected_lower_positive"]
        and not target["opposite_partial_concordance_pattern"]
    )
    diagnostics_fail_to_reproduce = all(
        not bool(diagnostics[label]["all_registers_positive"])
        for label in DIAGNOSTIC_LABELS
    )
    diagnostics_fail_to_reproduce_partial = all(
        not bool(diagnostics[label]["expected_partial_concordance_pattern"])
        for label in DIAGNOSTIC_LABELS
    )
    diagnostics_fail_to_reproduce_opposite_partial = all(
        not bool(diagnostics[label]["opposite_partial_concordance_pattern"])
        for label in DIAGNOSTIC_LABELS
    )
    partial_specificity = {
        "rank_target_exceeds_all_31": bool(rank_specificity["passed"]),
        "paired_target_minus_mean_control": bool(paired_specificity["passed"]),
        "common_mode_and_decoy_fail_to_reproduce": bool(
            diagnostics_fail_to_reproduce_partial
        ),
        "simultaneous_handedness_and_named_controls": bool(
            partial_family["passed"]
        ),
        "handedness_specificity_family": partial_family,
    }
    partial_specificity["passed"] = bool(
        partial_specificity["rank_target_exceeds_all_31"]
        and partial_specificity["paired_target_minus_mean_control"]
        and partial_specificity["common_mode_and_decoy_fail_to_reproduce"]
        and partial_specificity["simultaneous_handedness_and_named_controls"]
    )
    partial_specificity["non_handedness_passed"] = bool(
        partial_specificity["rank_target_exceeds_all_31"]
        and partial_specificity["paired_target_minus_mean_control"]
        and partial_specificity["common_mode_and_decoy_fail_to_reproduce"]
        and partial_family["named_controls_lower_positive"]
    )
    opposite_partial_specificity = {
        "rank_target_exceeds_all_31": bool(opposite_rank_specificity["passed"]),
        "paired_target_minus_mean_control": bool(
            opposite_paired_specificity["passed"]
        ),
        "common_mode_and_decoy_fail_to_reproduce": bool(
            diagnostics_fail_to_reproduce_opposite_partial
        ),
        "simultaneous_reverse_handedness_and_named_controls": bool(
            opposite_partial_family["passed"]
        ),
        "opposite_handedness_specificity_family": opposite_partial_family,
        "reverse_handedness_resolved": bool(
            opposite_partial_family["reverse_handedness_lower_positive"]
        ),
    }
    opposite_partial_specificity["non_handedness_passed"] = bool(
        opposite_partial_specificity["rank_target_exceeds_all_31"]
        and opposite_partial_specificity["paired_target_minus_mean_control"]
        and opposite_partial_specificity[
            "common_mode_and_decoy_fail_to_reproduce"
        ]
        and opposite_partial_family["named_controls_lower_positive"]
    )
    opposite_partial_specificity["passed"] = bool(
        opposite_partial_specificity["non_handedness_passed"]
        and opposite_partial_specificity[
            "simultaneous_reverse_handedness_and_named_controls"
        ]
    )
    specificity = {
        **rank_specificity,
        "paired_target_minus_mean_control": paired_specificity,
        "paired_target_minus_named_controls": diagnostic_specificity,
        "diagnostics_fail_to_reproduce": bool(diagnostics_fail_to_reproduce),
        "partial_concordance": partial_specificity,
        "opposite_partial_concordance": opposite_partial_specificity,
        "opposite_rank": opposite_rank_specificity,
        "opposite_paired_target_minus_mean_control": opposite_paired_specificity,
        "passed": bool(
            rank_specificity["passed"]
            and paired_specificity["passed"]
            and diagnostic_specificity["passed"]
            and diagnostics_fail_to_reproduce
        ),
    }
    mechanics = payload.get("mechanics")
    if not isinstance(mechanics, Mapping):
        raise ValueError("register target lacks mechanical replay endpoints")
    replay_error = float(mechanics.get("register_zero_no_hook_max_abs", float("inf")))
    replay_passed = replay_error <= 1e-8
    design = payload.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("register target lacks its frozen design")
    jreg_direction = design.get("jreg_direction")
    if jreg_direction is not None:
        jreg_direction = str(jreg_direction)
    decision = adjudicate_register_relocation(
        target,
        specificity,
        capability_passed=bool(capability_passed and replay_passed),
        phase2_p3_authorized=p3_authorized,
        joint_requested=joint,
        phase2_joint_authorized=joint_authorized,
        jreg_direction=jreg_direction,
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "status": decision["code"],
        "stage": stage,
        "mode": mode,
        "target_sha256": str(payload.get("self_sha256") or ""),
        "inference_seed": int(seed),
        "partial_specificity_seeds": {
            "expected": PARTIAL_SPECIFICITY_SEED,
            "opposite": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
        },
        "bootstrap_draws": int(n_bootstrap),
        "target": target,
        "controls": controls,
        "diagnostic_controls": diagnostics,
        "specificity": specificity,
        "register_zero_replay": {
            "max_abs_cell_endpoint_error": replay_error,
            "threshold": 1e-8,
            "passed": replay_passed,
        },
        "authorization": dict(authorization),
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=INFERENCE_SEED)
    parser.add_argument("--bootstrap", type=int, default=CONFIRMATORY_BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    if args.seed != INFERENCE_SEED or args.bootstrap != CONFIRMATORY_BOOTSTRAP_DRAWS:
        parser.error(
            "confirmatory register inference is frozen at seed "
            f"{INFERENCE_SEED} and {CONFIRMATORY_BOOTSTRAP_DRAWS} bootstrap draws"
        )
    target_path = Path(args.target).resolve()
    target_manifest_path = Path(args.target_manifest).resolve()
    output_path = Path(args.out).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite register analysis: {output_path}")
    payload = json.loads(target_path.read_text())
    target_manifest = json.loads(target_manifest_path.read_text())
    if (
        payload.get("stage") != "phase3_reserve"
        or payload.get("n_blocks") != CONFIRMATORY_BLOCKS
        or payload.get("n_cells") != CONFIRMATORY_CELLS
    ):
        raise RuntimeError(
            "confirmatory register target must contain exactly 384 complete blocks / "
            "3,456 crossed cells"
        )
    _, provenance, provenance_paths = validate_confirmatory_provenance(
        payload,
        target_path=target_path,
        target_manifest=target_manifest,
        target_manifest_path=target_manifest_path,
        analysis_output_path=output_path,
    )
    payload["self_sha256"] = sha256_file(target_path)
    analysis = analyze_payload(payload, seed=args.seed, n_bootstrap=args.bootstrap)
    analysis["reserve_provenance"] = provenance
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        experiment_id="pbc-p3-register-analysis",
        command=sys.argv,
        repo=REPO,
        status="ANALYSIS",
        config={
            "stage": "phase3_reserve",
            "seed": args.seed,
            "expected_partial_specificity_seed": PARTIAL_SPECIFICITY_SEED,
            "opposite_partial_specificity_seed": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
            "bootstrap": args.bootstrap,
            "schema": ANALYSIS_SCHEMA,
        },
        seeds={
            "bootstrap": args.seed,
            "expected_partial_specificity": PARTIAL_SPECIFICITY_SEED,
            "opposite_partial_specificity": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
        },
        inputs=[
            file_record(path)
            for path in (
                *provenance_paths,
                target_path,
                target_manifest_path,
            )
        ],
        outputs=[file_record(output_path)],
        notes=[
            "Model-free recomputation. Period-3 is adjudicated from the intact-P10 lattice before the optional modifier arm."
        ],
    )
    manifest.write(manifest_path)


if __name__ == "__main__":
    main()
