from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from gfmcd.provenance import file_record, sha256_file
from gfmcd.splicing.pbc_register_relocation import REGISTERS, phase_delta, preferred_delta
from gfmcd.splicing.pbc_register_stage import (
    REGISTER_PREFLIGHT_AUDIT_SCHEMA,
    make_register_target_seal,
)
from pbc_candidate_fixture import candidate_bindings, prerequisite_manifest_chain


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts/splicebert_positional_boundary_coherence" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ANALYZE = _load("analyze_p3_register_relocation")
AUDIT = _load("audit_p3_register_relocation")


def _rows(blocks: int = 24):
    rows = []
    labels = ["p3", *ANALYZE.CONTROL_LABELS, *ANALYZE.DIAGNOSTIC_LABELS]
    for label_index, component in enumerate(labels):
        for block in range(blocks):
            chrom = f"chr{1 + block % 2}"
            strand = "+" if (block // 2) % 2 == 0 else "-"
            centered = (block - (blocks - 1) / 2) * 0.001
            for register in REGISTERS:
                if component == "p3":
                    effect = 0.30 + centered
                elif component in ANALYZE.DIAGNOSTIC_LABELS:
                    base = {-1: -0.02, 0: 0.04, 1: -0.01}[register]
                    effect = base * (1 + centered * 20)
                else:
                    effect = 0.01 + label_index * 0.0001 - centered / 3
                for donor in range(3):
                    for acceptor in range(3):
                        delta = phase_delta(donor, acceptor)
                        coupling = effect if delta == preferred_delta(register) else 0.0
                        rows.append(
                            {
                                "component": component,
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


def _sources(tmp_path):
    output = {}
    for name in (
        "cohort",
        "phase2_analysis",
        "phase2_audit",
        "register_preflight",
        "register_preflight_audit",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n")
        output[name] = file_record(path)
    return output


def _design():
    return {
        "jreg_direction": None,
        "direction": "right",
        "span_token": [393, 513],
        "decoy_span_token": [153, 273],
        "control_seeds": list(range(31)),
    }


def _mechanics():
    labels = ["p3", *ANALYZE.CONTROL_LABELS, *ANALYZE.DIAGNOSTIC_LABELS]
    return {
        "register_zero_no_hook_max_abs": 0.0,
        "construction_invariants": {"passed": True},
        "component_specs": {
            label: {"rms_match": label != "p3"} for label in labels
        },
        "carrier_delta_rms": {
            label: {"-1": 1.0, "0": 0.0, "1": 1.0} for label in labels
        },
    }


def _write_manifest(path, artifact, inputs, timestamp):
    payload = {
        "experiment_id": f"fixture-{artifact.stem}",
        "command": ["fixture"],
        "status": "COMPLETE",
        "config": {"stage": "phase3_reserve"},
        "seeds": {},
        "inputs": [file_record(value) for value in inputs],
        "outputs": [file_record(artifact)],
        "notes": [],
        "created_at_utc": timestamp,
        "git_commit": "d" * 40,
        "git_dirty": False,
        "environment": {},
    }
    path.write_text(json.dumps(payload) + "\n")
    return payload


def test_analysis_and_independent_audit_close_p3_without_joint_arm(tmp_path):
    target = {
        "schema": ANALYZE.TARGET_SCHEMA,
        "status": "COMPLETE",
        "stage": "phase3_reserve",
        "mode": "P3_ONLY",
        "authorization": {
            "phase2_p3_authorized": True,
            "phase2_joint_authorized": False,
            "register_preflight_passed": True,
        },
        "design": _design(),
        "sources": _sources(tmp_path),
        "mechanics": _mechanics(),
        "rows": _rows(),
        "self_sha256": "synthetic-target",
    }
    analysis = ANALYZE.analyze_payload(target, seed=71, n_bootstrap=199)
    assert analysis["status"] == "P3_REGISTER_RELOCATION_SUPPORTED"
    assert analysis["specificity"]["passed"]
    target.pop("self_sha256")
    audit = AUDIT.audit_payloads(
        target,
        analysis,
        target_sha256="synthetic-target",
        enforce_confirmatory_contract=False,
    )
    assert audit["status"] == "PASS"

    confirmatory_audit = AUDIT.audit_payloads(
        target,
        analysis,
        target_sha256="synthetic-target",
    )
    assert confirmatory_audit["status"] == "FAIL"
    assert not confirmatory_audit["checks"]["confirmatory_seed"]
    assert not confirmatory_audit["checks"]["confirmatory_draws"]
    assert not confirmatory_audit["row_contract"]["confirmatory_counts"]


def test_audit_rejects_coupling_not_equal_to_paired_margins(tmp_path):
    target = {
        "schema": ANALYZE.TARGET_SCHEMA,
        "status": "COMPLETE",
        "stage": "phase3_reserve",
        "mode": "P3_ONLY",
        "authorization": {
            "phase2_p3_authorized": True,
            "phase2_joint_authorized": False,
            "register_preflight_passed": True,
        },
        "design": _design(),
        "sources": _sources(tmp_path),
        "mechanics": _mechanics(),
        "rows": _rows(),
        "self_sha256": "synthetic-target",
    }
    analysis = ANALYZE.analyze_payload(target, seed=73, n_bootstrap=99)
    target.pop("self_sha256")
    target["rows"][0]["coupling"] += 0.2
    audit = AUDIT.audit_payloads(
        target,
        analysis,
        target_sha256="synthetic-target",
        enforce_confirmatory_contract=False,
    )
    assert audit["status"] == "FAIL"
    assert not audit["checks"]["row_contract"]


def test_confirmatory_analysis_and_audit_reopen_seal_and_enforce_paths_chronology(
    tmp_path,
):
    binding = tmp_path / "cohort.json"
    binding.write_text("{}\n")
    target_path = tmp_path / "relocation.json"
    analysis_path = tmp_path / "relocation.analysis.json"
    audit_path = tmp_path / "relocation.audit.json"
    seal = make_register_target_seal(
        phase2_authorization={
            "phase2_p3_authorized": True,
            "phase2_joint_authorized": False,
            "jreg_direction": None,
        },
        preflight_analysis={
            "schema": "gfmcd.pbc.p3_register_preflight_analysis.v1",
            "status": "P3_REGISTER_PREFLIGHT_PASS",
            "authorization": {
                "authorize_p3_register_reserve": True,
                "authorize_joint_register_reserve": False,
            },
        },
        preflight_audit={
            "schema": REGISTER_PREFLIGHT_AUDIT_SCHEMA,
            "status": "PASS",
        },
        bindings={
            "cohort": file_record(binding),
            **candidate_bindings(tmp_path),
        },
        include_p10=False,
        p3_attenuation_dose=0.5,
        p10_attenuation_dose=None,
        output_path=str(target_path),
        relocation_analysis_path=str(analysis_path),
        relocation_audit_path=str(audit_path),
        prerequisite_manifest_chain=prerequisite_manifest_chain(),
    )
    seal_path = tmp_path / "reserve-seal.json"
    seal_path.write_text(json.dumps(seal) + "\n")
    seal_manifest_path = seal_path.with_name(seal_path.stem + ".manifest.json")
    seal_manifest = _write_manifest(
        seal_manifest_path,
        seal_path,
        [binding],
        "2026-08-10T00:00:00+00:00",
    )
    target = {
        "schema": ANALYZE.TARGET_SCHEMA,
        "status": "COMPLETE",
        "stage": "phase3_reserve",
        "mode": "P3_ONLY",
        "seal_sha256": seal["seal_sha256"],
        "authorization": seal["authorization"],
        "design": seal["design"],
        "sources": {
            **_sources(tmp_path),
            "reserve_seal": file_record(seal_path),
            "reserve_seal_manifest": file_record(seal_manifest_path),
        },
        "mechanics": _mechanics(),
        "rows": _rows(),
    }
    target_path.write_text(json.dumps(target) + "\n")
    target_manifest_path = target_path.with_name(
        target_path.stem + ".manifest.json"
    )
    target_manifest = _write_manifest(
        target_manifest_path,
        target_path,
        [seal_path, seal_manifest_path],
        "2026-08-10T00:00:01+00:00",
    )
    _, provenance, provenance_paths = ANALYZE.validate_confirmatory_provenance(
        target,
        target_path=target_path,
        target_manifest=target_manifest,
        target_manifest_path=target_manifest_path,
        analysis_output_path=analysis_path,
    )
    assert all(provenance["checks"].values())
    assert provenance_paths == (seal_path, seal_manifest_path)
    with pytest.raises(RuntimeError, match="analysis_path_frozen"):
        ANALYZE.validate_confirmatory_provenance(
            target,
            target_path=target_path,
            target_manifest=target_manifest,
            target_manifest_path=target_manifest_path,
            analysis_output_path=tmp_path / "unfrozen-analysis.json",
        )

    analysis_input = {**target, "self_sha256": sha256_file(target_path)}
    analysis = ANALYZE.analyze_payload(
        analysis_input,
        seed=ANALYZE.INFERENCE_SEED,
        n_bootstrap=99,
    )
    analysis["reserve_provenance"] = provenance
    analysis_path.write_text(json.dumps(analysis) + "\n")
    analysis_manifest_path = analysis_path.with_name(
        analysis_path.stem + ".manifest.json"
    )
    analysis_manifest = _write_manifest(
        analysis_manifest_path,
        analysis_path,
        [seal_path, seal_manifest_path, target_path, target_manifest_path],
        "2026-08-10T00:00:02+00:00",
    )
    checks, _ = AUDIT.validate_confirmatory_provenance(
        target,
        analysis,
        target_path=target_path,
        target_manifest=target_manifest,
        target_manifest_path=target_manifest_path,
        analysis_path=analysis_path,
        analysis_manifest=analysis_manifest,
        analysis_manifest_path=analysis_manifest_path,
        audit_output_path=audit_path,
    )
    assert all(checks.values())
    out_of_order = {**analysis_manifest, "created_at_utc": "2026-08-09T23:59:59+00:00"}
    with pytest.raises(RuntimeError, match="target_precedes_analysis"):
        AUDIT.validate_confirmatory_provenance(
            target,
            analysis,
            target_path=target_path,
            target_manifest=target_manifest,
            target_manifest_path=target_manifest_path,
            analysis_path=analysis_path,
            analysis_manifest=out_of_order,
            analysis_manifest_path=analysis_manifest_path,
            audit_output_path=audit_path,
        )
    with pytest.raises(RuntimeError, match="audit_path_frozen"):
        AUDIT.validate_confirmatory_provenance(
            target,
            analysis,
            target_path=target_path,
            target_manifest=target_manifest,
            target_manifest_path=target_manifest_path,
            analysis_path=analysis_path,
            analysis_manifest=analysis_manifest,
            analysis_manifest_path=analysis_manifest_path,
            audit_output_path=tmp_path / "unfrozen-audit.json",
        )
