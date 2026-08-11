#!/usr/bin/env python
"""Independent pre-target audit of the PBC protocol-to-code contract.

This program reads construction identities and source text only.  It does not
import a preflight, analysis, or verdict function, load SpliceBERT, or read a
development/confirmation outcome.  Its passing artifact is a required input to
the Phase-II freezer, where it is hash-bound into the target seal.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record, sha256_file  # noqa: E402


CONFORMANCE_SCHEMA = "gfmcd.pbc.protocol_code_conformance_audit.v1"
CONFORMANCE_EXPERIMENT_ID = "pbc-protocol-code-conformance-audit-v1"
DEFAULT_SOURCE_SPEC = (
    REPO
    / "configs/splicebert_positional_boundary_coherence/freshness_sources_v1.json"
)
DEFAULT_FROZEN_VARIANCE = (
    REPO
    / "configs/splicebert_positional_boundary_coherence/c43_upper_variance_v1.json"
)
EXPECTED_WEIGHT_SHA256 = (
    "2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d"
)
EXPECTED_BIOLOGY_SHA256: dict[str, str] = {
    "fasta": "5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51",
    "fai": "3b425de206296a5c8053023fa5ca61da43cfe78c1737c12e58c83367c7e83c21",
    "basic_gtf": "5b9ad9aa71f6ab5d55123c51f74723644a895bc0cbaa9cee8fbc11bb9dec0bf7",
    "comprehensive_gtf": (
        "83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453"
    ),
    "maxent_donor_table": (
        "c64fa6aea3d8b71d6af69f5e3ece8f84de06f5450f6d4ba482fbbff6afcf8b5b"
    ),
    "maxent_acceptor_table": (
        "9e8a74dc795ae5c9c5b611d4343a1a5a70fff863956f4e00ae23a85f21beb3ae"
    ),
}
EXPECTED_SOURCE_CATEGORIES = {
    "c43_predecessor",
    "c43_replication",
    "c44_smoke",
    "c44_development",
    "c44_reserve",
    "gfs25_v1",
    "gfs25_v2",
    "source_protocol_registry",
}
EXPECTED_SOURCE_SPEC_SHA256 = (
    "c3a145aa70d52eeba8276456822ba6f8aed5acd7e65ae62f67e14d7aec6c40d4"
)
EXPECTED_VARIANCE_SHA256 = (
    "aaaf0b554ba1d11dc7c03485f1db583b886f13d786bc0508b729d09d0645584d"
)


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def conformance_code_files(repo: Path = REPO) -> tuple[Path, ...]:
    """Return every active PBC executable, reusable module, and memorial test."""

    script_dir = repo / "scripts/splicebert_positional_boundary_coherence"
    sources = [
        *script_dir.glob("*.py"),
        *(repo / "src/gfmcd/splicing").glob("pbc*.py"),
        repo / "src/gfmcd/splicing/positional_boundary_coherence.py",
        repo / "src/gfmcd/splicing/positional_spectrum.py",
        repo / "src/gfmcd/splicing/circuit.py",
        repo / "src/gfmcd/splicing/dataset.py",
        repo / "src/gfmcd/splicing/gtf.py",
        repo / "src/gfmcd/splicing/phase.py",
        repo / "src/gfmcd/crossmodel/carrier_spectrum.py",
        repo / "src/gfmcd/adapters/base.py",
        repo / "src/gfmcd/adapters/splicebert.py",
        repo / "src/gfmcd/artifact_guard.py",
        repo / "src/gfmcd/provenance.py",
        repo / "src/gfmcd/rigor.py",
        *(repo / "tests").glob("test_pbc*.py"),
        repo / "tests/pbc_candidate_fixture.py",
        repo / "tests/test_positional_boundary_coherence.py",
    ]
    return tuple(sorted({path.resolve() for path in sources if path.is_file()}))


def model_files(model: str | Path) -> tuple[Path, ...]:
    root = Path(model).expanduser().resolve(strict=True)
    names = (
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    )
    return tuple(root / name for name in names if (root / name).is_file())


def _clean_revision(repo: Path) -> tuple[bool, str]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    return not bool(status.strip()), revision


def _tracked(path: Path, repo: Path) -> bool:
    try:
        relative = path.resolve().relative_to(repo.resolve())
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (ValueError, subprocess.CalledProcessError):
        return False
    return True


def _record_matches(record: object, expected: str) -> tuple[bool, Path | None]:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        return False, None
    path = Path(str(record["path"])).expanduser().resolve()
    valid = bool(
        path.is_file()
        and str(record.get("sha256", "")) == expected
        and sha256_file(path) == expected
        and type(record.get("size_bytes")) is int
        and int(record["size_bytes"]) == path.stat().st_size
    )
    return valid, path if valid else None


def candidate_input_checks(
    candidate: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Path]]:
    """Independently verify candidate records and their current external bytes."""

    pins = candidate.get("input_pins")
    if not isinstance(pins, Mapping):
        return {name: False for name in EXPECTED_BIOLOGY_SHA256}, {}
    maxent = pins.get("maxent")
    maxent = maxent if isinstance(maxent, Mapping) else {}
    records = {
        "fasta": pins.get("fasta"),
        "fai": pins.get("fai"),
        "basic_gtf": pins.get("basic_gtf"),
        "comprehensive_gtf": pins.get("comprehensive_gtf"),
        "maxent_donor_table": maxent.get("donor_table"),
        "maxent_acceptor_table": maxent.get("acceptor_table"),
    }
    checks: dict[str, bool] = {}
    paths: dict[str, Path] = {}
    for label, expected in EXPECTED_BIOLOGY_SHA256.items():
        checks[label], path = _record_matches(records[label], expected)
        if path is not None:
            paths[label] = path
    basic = records["basic_gtf"]
    comprehensive = records["comprehensive_gtf"]
    checks["annotation_conventions"] = bool(
        isinstance(basic, Mapping)
        and isinstance(comprehensive, Mapping)
        and basic.get("release") == "GENCODE_v50"
        and basic.get("coordinate_convention")
        == "GTF_1_based_closed_parsed_as_0_based_half_open"
        and basic.get("phase_convention")
        == "GTF_frame_to_conventional_intron_phase"
        and comprehensive.get("release") == "GENCODE_v50"
        and comprehensive.get("role")
        == "local_coding_strand_and_phase_safety_audit"
    )
    checks["maxent_equations"] = bool(
        maxent.get("equations") == "Yeo_Burge_MaxEntScan_9mer_23mer_local_v1"
    )
    checks["construction_outcome_firewall"] = bool(
        candidate.get("stage") == "construction"
        and candidate.get("model_outcomes_read") is False
    )
    return checks, paths


def _has(path: Path, *tokens: str) -> bool:
    if not path.is_file():
        return False
    text = path.read_text()
    return all(token in text for token in tokens)


def _row(
    requirement_id: str,
    protocol_section: str,
    requirement: str,
    implementation: Sequence[str],
    tests: Sequence[str],
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "requirement_id": requirement_id,
        "protocol_section": protocol_section,
        "requirement": requirement,
        "implementation": list(implementation),
        "tests": list(tests),
        "checks": {name: bool(value) for name, value in checks.items()},
        "passed": bool(checks and all(checks.values())),
    }


def coverage_matrix(
    *,
    repo: Path,
    protocol_path: Path,
    source_spec: Mapping[str, Any],
    variance: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_checks: Mapping[str, bool],
    checkpoint_ok: bool,
    source_spec_sha256: str,
    variance_sha256: str,
) -> list[dict[str, Any]]:
    """Build the manually enumerated protocol-to-code requirement matrix."""

    protocol = protocol_path.read_text()
    scripts = repo / "scripts/splicebert_positional_boundary_coherence"
    core = repo / "src/gfmcd/splicing/pbc_carrier_factorial.py"
    pins = repo / "src/gfmcd/splicing/pbc_frozen_inputs.py"
    preflight = scripts / "run_carrier_factorial_preflight.py"
    preflight_audit = scripts / "audit_carrier_factorial_preflight.py"
    freezer = scripts / "freeze_carrier_factorial_target.py"
    target = scripts / "run_carrier_factorial_target.py"
    analysis = scripts / "analyze_carrier_factorial.py"
    result_audit = scripts / "audit_carrier_factorial.py"
    source_categories = {
        str(row.get("category"))
        for row in source_spec.get("sources", [])
        if isinstance(row, Mapping)
    }
    code_files = conformance_code_files(repo)
    return [
        _row(
            "PBC-C01",
            "§2.1; A4",
            "The official checkpoint is identified by weight bytes, not location.",
            [str(pins), str(preflight), str(freezer), str(target)],
            ["tests/test_pbc_conformance_closure.py"],
            {
                "protocol_hash": EXPECTED_WEIGHT_SHA256 in protocol,
                "checkpoint_bytes": checkpoint_ok,
                "pin_constant": _has(pins, EXPECTED_WEIGHT_SHA256),
                "alternate_weight_rejected": _has(pins, "model.safetensors"),
                "preflight_gate": _has(preflight, "validate_official_checkpoint(model_path)"),
                "freeze_gate": _has(freezer, "validate_official_checkpoint(model_path)"),
                "target_gate": _has(target, "validate_official_checkpoint(model_path)"),
            },
        ),
        _row(
            "PBC-C02",
            "§3.3; §4.3; A4",
            "FASTA, FAI, GENCODE-v50, and MaxEnt bytes and conventions are frozen.",
            [str(pins), str(freezer)],
            ["tests/test_pbc_conformance_closure.py"],
            dict(candidate_checks),
        ),
        _row(
            "PBC-C03",
            "§3.1; §4.2",
            "Historical inputs are limited to eight freshness categories and sign-free variance.",
            [str(DEFAULT_SOURCE_SPEC), str(DEFAULT_FROZEN_VARIANCE)],
            ["tests/test_pbc_phase1_real_data.py"],
            {
                "source_schema": source_spec.get("schema")
                == "gfmcd.positional_boundary_coherence.freshness_sources.v1",
                "source_spec_bytes": source_spec_sha256 == EXPECTED_SOURCE_SPEC_SHA256,
                "categories": source_categories == EXPECTED_SOURCE_CATEGORIES,
                "variance_schema": variance.get("schema")
                == "gfmcd.positional_boundary_coherence.frozen_variance.v1",
                "variance_bytes": variance_sha256 == EXPECTED_VARIANCE_SHA256,
                "variance_revision": variance.get("source", {}).get("revision")
                == "fd26cc1b",
                "target_count": "exact confirmation count is frozen at **600" in protocol,
            },
        ),
        _row(
            "PBC-C04",
            "§4.1–§4.5",
            "Construction is model-free, complete-lattice, balanced, and independently audited.",
            [
                str(scripts / "build_candidate_census.py"),
                str(scripts / "build_composition_balanced_cohort.py"),
                str(scripts / "audit_composition_balanced_cohort.py"),
            ],
            ["tests/test_pbc_phase1_real_data.py", "tests/test_pbc_cohort_runners.py"],
            {
                "candidate_outcome_firewall": candidate.get("model_outcomes_read") is False,
                "builder_no_model_import": not _has(
                    scripts / "build_candidate_census.py", "load_adapter("
                ),
                "cohort_counts": _has(
                    repo / "src/gfmcd/splicing/positional_boundary_coherence.py",
                    "confirmation",
                    "phase3_reserve",
                    "development",
                ),
                "independent_auditor": (
                    scripts / "audit_composition_balanced_cohort.py"
                ).is_file(),
            },
        ),
        _row(
            "PBC-C05",
            "§2.2; §8",
            "The model-native donor-mask coupling and equal-block Psi estimand are fixed.",
            [str(target), str(analysis), str(result_audit)],
            ["tests/test_pbc_factorial_runners.py"],
            {
                "readout": _has(target, "c43_margin", "coupling_from_scores"),
                "corruptions": _has(target, 'for state in ("intact", "ac", "at")'),
                "independent_recompute": _has(
                    result_audit, "_independent_statistical_recompute"
                ),
            },
        ),
        _row(
            "PBC-C06",
            "§6.1–§6.4",
            "P3/P10 bands, doses, worlds, rescue, and exact structured-null families are fixed.",
            [str(core), str(preflight), str(target)],
            ["tests/test_pbc_carrier_factorial.py", "tests/test_pbc_factorial_runners.py"],
            {
                "bands": _has(core, "P3_BINS_1026", "P10_BINS_1026", "DOSE_GRID"),
                "exact_band_bins": _has(
                    core, "tuple(range(339, 346))", "tuple(range(94, 101))"
                ),
                "exact_doses_and_sesoi": _has(
                    core,
                    "(0.25, 0.50, 0.75, 1.00)",
                    "SESOI: float = 0.05",
                ),
                "mechanical_tolerances": _has(
                    core,
                    "OFF_BAND_TOLERANCE: float = 1e-10",
                    "COMMUTATIVITY_TOLERANCE: float = 1e-12",
                    "RECONSTRUCTION_TOLERANCE: float = 1e-10",
                ),
                "worlds": _has(core, "build_world_tables", '"Y00"', '"Y11"'),
                "nulls": _has(
                    core,
                    "build_phase_control_family",
                    "MIN_STRUCTURED_CONTROLS: int = 31",
                    "MIN_PSEUDO_PHASE_CONTROLS: int = 999",
                ),
                "seeds": all(
                    seed in protocol
                    for seed in ("2026081003", "2026081004", "2026081005")
                ),
            },
        ),
        _row(
            "PBC-C07",
            "§7; A1",
            (
                "Dose selection is outcome-blind and all authorized tables receive "
                "failable health gates."
            ),
            [str(preflight), str(preflight_audit)],
            ["tests/test_pbc_factorial_runners.py"],
            {
                "phase_blind_selector": _has(preflight, "select_largest_passing_dose"),
                "table_health": _has(preflight, "authorized_table_health", "all_tables_pass"),
                "health_thresholds": _has(
                    preflight_audit,
                    '"native_source_positive": native["ci"][0] > 0.05',
                    '"phase_blind_crossed_positive": crossed["ci"][0] > 0.02',
                    '"masked_accuracy_health": accuracy_retention >= 0.80',
                    "donor_retention >= 0.80",
                    "acceptor_retention >= 0.80",
                ),
                "resolution_thresholds": _has(
                    preflight_audit,
                    'mde_pass("clean_psi", 0.04)',
                    'mde_pass("N3", 0.04)',
                    'mde_pass("N10", 0.05)',
                    'mde_pass("J", 0.05)',
                    'monte_carlo_se", math.inf)) <= 0.0002',
                ),
                "fixture_thresholds": _has(
                    preflight_audit,
                    "fwer <= 0.06",
                    'power["0.075"]["N3"] >= 0.80',
                    '"replicates": 1_000',
                ),
                "independent_audit": _has(
                    preflight_audit, "target_outcomes_read", "raw_sha256", "analysis_sha256"
                ),
            },
        ),
        _row(
            "PBC-C08",
            "§7.6; §11.3; A1",
            (
                "Development measurement, independent audit, and clean-revision "
                "freeze are separate stages."
            ),
            [str(preflight), str(preflight_audit), str(freezer)],
            ["tests/test_pbc_factorial_runners.py", "tests/test_pbc_conformance_closure.py"],
            {
                "separate_files": len({preflight, preflight_audit, freezer}) == 3,
                "clean_revision": _has(freezer, "clean_tracked_revision"),
                "output_plan": _has(freezer, "target_output_plan"),
                "audit_chain": _has(freezer, "preflight audit hash chain is invalid"),
            },
        ),
        _row(
            "PBC-C09",
            "§11.3; §14; A4",
            "An independent passing protocol-to-code audit is hash-bound into every target seal.",
            [str(Path(__file__).resolve()), str(freezer), str(core)],
            ["tests/test_pbc_conformance_closure.py"],
            {
                "freeze_argument": _has(freezer, "--conformance-audit"),
                "freeze_manifest_argument": _has(
                    freezer, "--conformance-audit-manifest"
                ),
                "seal_hashes": _has(
                    core,
                    "conformance_audit_sha256",
                    "conformance_audit_manifest_sha256",
                ),
                "target_reverification": _has(core, CONFORMANCE_SCHEMA),
            },
        ),
        _row(
            "PBC-C10",
            "§11.3–§11.4; A4",
            (
                "Device, batch size, dtype, revision, and immutable output paths "
                "are sealed and rechecked."
            ),
            [str(freezer), str(core), str(target)],
            ["tests/test_pbc_conformance_closure.py"],
            {
                "runtime_in_seal": _has(core, '"runtime"', '"batch_size"', '"dtype"'),
                "runtime_recheck": _has(target, "_require_sealed_runtime"),
                "revision_recheck": _has(target, "_require_runtime_revision"),
                "output_recheck": _has(target, "_require_sealed_output_path"),
            },
        ),
        _row(
            "PBC-C11",
            "§8.1–§8.5; A3",
            (
                "Phase-II multiplicity, materiality, controls, sensitivity, and "
                "directional tier are executable."
            ),
            [str(analysis), str(result_audit)],
            ["tests/test_pbc_factorial_runners.py"],
            {
                "bootstrap": _has(analysis, "CONFIRMATORY_BOOTSTRAP_DRAWS", "20_000"),
                "bootstrap_seed": _has(
                    analysis, "CONFIRMATORY_BOOTSTRAP_SEED = 2026081001"
                ),
                "pseudo_phase": _has(
                    analysis, "CONFIRMATORY_PSEUDO_CONTROLS", "999"
                ),
                "pseudo_seed": _has(
                    analysis, "CONFIRMATORY_PSEUDO_SEED = 2026081002"
                ),
                "resolution": _has(
                    analysis,
                    '"threshold": 0.04 if name in {"Psi00", "N3"} else 0.05',
                ),
                "directional_tier": _has(core, "P3_DIRECTIONAL_SCOPED_SUPPORT"),
                "audit_recompute": _has(result_audit, "simultaneous_stratified_intervals"),
            },
        ),
        _row(
            "PBC-C12",
            "§9–§10; A2–A3",
            (
                "Phase III has a separate candidate freeze, reserve seal, target, "
                "analysis, and audits."
            ),
            [str(scripts)],
            [
                "tests/test_pbc_register_runners.py",
                "tests/test_pbc_register_mediation_execution.py",
            ],
            {
                "candidate_freezer": (
                    scripts / "freeze_p3_register_candidates.py"
                ).is_file(),
                "reserve_freezer": (scripts / "freeze_p3_register_target.py").is_file(),
                "register_target": (scripts / "run_p3_register_target.py").is_file(),
                "mediation_target": (
                    scripts / "run_p3_register_mediation_target.py"
                ).is_file(),
            },
        ),
        _row(
            "PBC-C13",
            "§11.1–§11.5",
            (
                "All active PBC code/tests are tracked and result writers use "
                "guarded JSON+manifest paths."
            ),
            [str(path) for path in code_files],
            [str(path.relative_to(repo)) for path in code_files if "tests" in path.parts],
            {
                "inventory_nonempty": len(code_files) >= 20,
                "all_tracked": all(_tracked(path, repo) for path in code_files),
                "target_guard": _has(target, "guard_output_path", "ExperimentManifest"),
                "audit_guard": _has(result_audit, "guard_output_path", "ExperimentManifest"),
            },
        ),
    ]


def build_audit(
    *,
    protocol_path: Path,
    source_spec_path: Path,
    variance_path: Path,
    candidate_path: Path,
    model_path: Path,
    repo: Path = REPO,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    source_spec = json.loads(source_spec_path.read_text())
    variance = json.loads(variance_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if not all(
        isinstance(value, Mapping) for value in (source_spec, variance, candidate)
    ):
        raise RuntimeError("conformance inputs must be JSON objects")
    candidate_checks, biology_paths = candidate_input_checks(candidate)
    checkpoint = model_path.expanduser().resolve(strict=True) / "pytorch_model.bin"
    checkpoint_ok = bool(
        checkpoint.is_file()
        and sha256_file(checkpoint) == EXPECTED_WEIGHT_SHA256
        and not (model_path / "model.safetensors").exists()
    )
    matrix = coverage_matrix(
        repo=repo,
        protocol_path=protocol_path,
        source_spec=source_spec,
        variance=variance,
        candidate=candidate,
        candidate_checks=candidate_checks,
        checkpoint_ok=checkpoint_ok,
        source_spec_sha256=sha256_file(source_spec_path),
        variance_sha256=sha256_file(variance_path),
    )
    tracked_clean, revision = _clean_revision(repo)
    code_files = conformance_code_files(repo)
    inputs = tuple(
        dict.fromkeys(
            (
                protocol_path.resolve(),
                source_spec_path.resolve(),
                variance_path.resolve(),
                candidate_path.resolve(),
                *biology_paths.values(),
                *model_files(model_path),
                *code_files,
            )
        )
    )
    passed = bool(
        tracked_clean
        and len(revision) == 40
        and all(row["passed"] for row in matrix)
    )
    payload: dict[str, Any] = {
        "schema": CONFORMANCE_SCHEMA,
        "stage": "development",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "target_outcomes_read": False,
        "model_loaded": False,
        "tracked_tree_clean": tracked_clean,
        "git_revision": revision,
        "protocol_sha256": sha256_file(protocol_path),
        "source_spec_sha256": sha256_file(source_spec_path),
        "frozen_variance_sha256": sha256_file(variance_path),
        "candidate_package_sha256": sha256_file(candidate_path),
        "checkpoint_weight_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "external_input_sha256": dict(EXPECTED_BIOLOGY_SHA256),
        "coverage_matrix": matrix,
        "failures": [
            row["requirement_id"] for row in matrix if row["passed"] is not True
        ]
        + ([] if tracked_clean else ["TRACKED_TREE_DIRTY"]),
        "audited_inputs": [file_record(path) for path in inputs],
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload, inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--source-spec", default=str(DEFAULT_SOURCE_SPEC))
    parser.add_argument("--frozen-variance", default=str(DEFAULT_FROZEN_VARIANCE))
    parser.add_argument("--candidate-package", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = guard_output_path(Path(args.out).expanduser().resolve())
    manifest_path = guard_output_path(
        output.with_name(output.stem + ".manifest.json")
    )
    protocol = Path(args.protocol).expanduser().resolve(strict=True)
    source_spec = Path(args.source_spec).expanduser().resolve(strict=True)
    variance = Path(args.frozen_variance).expanduser().resolve(strict=True)
    candidate = Path(args.candidate_package).expanduser().resolve(strict=True)
    model = Path(args.model).expanduser().resolve(strict=True)
    audit, inputs = build_audit(
        protocol_path=protocol,
        source_spec_path=source_spec,
        variance_path=variance,
        candidate_path=candidate,
        model_path=model,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = ExperimentManifest.create(
        CONFORMANCE_EXPERIMENT_ID,
        sys.argv,
        repo=REPO,
        status=str(audit["status"]),
        config={
            "stage": "development",
            "target_outcomes_read": False,
            "git_revision": audit["git_revision"],
            "requirements": len(audit["coverage_matrix"]),
        },
        notes=[
            "Independent source-level protocol-to-code audit; no preflight verdict "
            "function imported and no model or target outcome loaded."
        ],
    )
    for path in inputs:
        manifest.add_input(path)
    manifest.add_output(output)
    manifest.write(manifest_path)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "failures": audit["failures"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
