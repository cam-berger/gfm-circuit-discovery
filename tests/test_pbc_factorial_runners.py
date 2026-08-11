from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pytest


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "splicebert_positional_boundary_coherence"
)
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_carrier_factorial as analysis_runner  # noqa: E402
import audit_carrier_factorial as audit_runner  # noqa: E402
import audit_carrier_factorial_preflight as preflight_audit_runner  # noqa: E402
import calibrate_carrier_factorial as calibration_runner  # noqa: E402
import run_carrier_factorial_preflight as preflight_runner  # noqa: E402
import run_carrier_factorial_target as target_runner  # noqa: E402
from gfmcd.provenance import file_record, sha256_file  # noqa: E402
from gfmcd.splicing.pbc_carrier_factorial import (  # noqa: E402
    DOSE_GRID,
    build_phase_control_family,
    make_target_seal,
)


def _sequence_states() -> tuple[str, str, str]:
    intact = list("A" * 1022)
    intact[392:394] = list("GT")
    intact[435:437] = list("AG")
    intact[510:512] = list("AG")
    ac = list(intact)
    at = list(intact)
    ac[511] = "C"
    at[511] = "T"
    return "".join(intact), "".join(ac), "".join(at)


def _cohort(blocks: int, *, role: str) -> dict[str, Any]:
    intact, ac, at = _sequence_states()
    cells: list[dict[str, Any]] = []
    for block in range(blocks):
        for donor in range(3):
            for acceptor in range(3):
                cells.append(
                    {
                        "cell_id": f"b{block:03d}-d{donor}-a{acceptor}",
                        "block_id": f"b{block:03d}",
                        "split_role": role,
                        "chrom": f"chr{1 + block % 3}",
                        "strand": "+" if block % 2 == 0 else "-",
                        "donor_phase": donor,
                        "acceptor_phase": acceptor,
                        "donor_pos": 392,
                        "acceptor_pos": 512,
                        "span_nt": 120,
                        "bridge_acceptor_pos": 437,
                        "scaffold_source_id": f"scaffold-{block:03d}",
                        "donor_source_id": f"donor-{block:03d}-{donor}",
                        "acceptor_source_id": f"acceptor-{block:03d}-{acceptor}",
                        "sequence_intact": intact,
                        "sequence_acceptor_ac": ac,
                        "sequence_acceptor_at": at,
                    }
                )
    return {"schema": target_runner.COHORT_SCHEMA, "cells": cells}


class _FakeScorer:
    def __init__(self, table: np.ndarray) -> None:
        self.table = table

    def position_table(self) -> np.ndarray:
        return self.table.copy()

    def metadata(self) -> Mapping[str, Any]:
        return {"fixture": "deterministic"}

    def score_conditions(
        self,
        cells: Sequence[Mapping[str, Any]],
        tables: Mapping[str, np.ndarray],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        output: dict[str, list[dict[str, Any]]] = {}
        for condition_index, condition in enumerate(tables):
            if condition == "Y00":
                amplitude = 0.30
            elif condition == "Y10":
                amplitude = 0.15
            else:
                amplitude = 0.285 + condition_index * 1e-6
            rows = []
            for cell in cells:
                match = int(cell["donor_phase"]) == int(cell["acceptor_phase"])
                coupling = amplitude if match else 0.0
                rows.append(
                    {
                        "cell_id": str(cell["cell_id"]),
                        "block_id": str(cell["block_id"]),
                        "chrom": str(cell["chrom"]),
                        "strand": str(cell["strand"]),
                        "donor_phase": int(cell["donor_phase"]),
                        "acceptor_phase": int(cell["acceptor_phase"]),
                        "f_intact": 1.0,
                        "f_ac": 1.0 - coupling,
                        "f_at": 1.0 - coupling + 0.001,
                        "coupling_ac": coupling,
                        "coupling_at": coupling - 0.001,
                    }
                )
            output[condition] = rows
        return output


def _seal(table: np.ndarray, tmp_path: Path, *, full: bool = False) -> dict[str, Any]:
    p10 = 0.5 if full else None
    controls, specs = build_phase_control_family(
        table, p3_dose=0.5, p10_dose=p10, seed=20260810
    )
    hashes = {
        label: {
            "family": str(specs[label]["family"]),
            "control_id": int(specs[label]["control_id"]),
            "world": str(specs[label]["world"]),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
        for label, value in controls.items()
    }
    preflight = {
        "passed": True,
        "target_mode": "FULL_FACTORIAL" if full else "P3_ONLY",
        "p3": {"selected_dose": 0.5},
        "p10": {"selected_dose": p10},
    }
    mode = preflight["target_mode"]
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
                "schema": preflight_runner.PREFLIGHT_SCHEMA,
                "target_mode": mode,
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
    audit_path = tmp_path / "preflight.audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": "gfmcd.pbc.carrier_factorial_preflight_audit.v1",
                "status": "PASS",
                "target_mode": mode,
                "preflight_sha256": sha256_file(preflight_path),
                "raw_sha256": raw_hash,
                "analysis_sha256": analysis_hash,
            }
        )
    )
    audit_manifest_path = tmp_path / "preflight.audit.manifest.json"
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
    paths = (
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
    return make_target_seal(
        preflight,
        pinned_files={path.name: file_record(path) for path in paths},
        control_seed=20260810,
        structured_control_hashes=hashes,
        preflight_authorization={
            "audit_schema": "gfmcd.pbc.carrier_factorial_preflight_audit.v1",
            "audit_status": "PASS",
            "target_mode": preflight["target_mode"],
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
        },
        runtime={"device": "cpu", "batch_size": 4, "dtype": "float32"},
        target_outputs={
            "shard_count": 1,
            "shards": ["/tmp/pbc-target.json"],
            "assembled": "/tmp/pbc-target.json",
        },
        frozen_revision="f" * 40,
    )


def _synthetic_target(
    *, full: bool, directional_submaterial: bool = False
) -> dict[str, Any]:
    blocks = 12
    specs: dict[str, dict[str, Any]] = {
        "Y00": {"family": "target", "control_id": None, "world": "Y00"},
        "Y10": {"family": "target", "control_id": None, "world": "Y10"},
    }
    if full:
        specs.update(
            {
                "Y01": {"family": "target", "control_id": None, "world": "Y01"},
                "Y11": {"family": "target", "control_id": None, "world": "Y11"},
            }
        )
    for family, world in (("p3_phase", "Y10"), ("p3_axis", "Y10")):
        for control in range(1, 32):
            specs[f"{family}_{control:03d}"] = {
                "family": family,
                "control_id": control,
                "world": world,
            }
    if full:
        for family, world in (
            ("p10_phase", "Y01"),
            ("p10_axis", "Y01"),
            ("joint_p3", "Y10"),
            ("joint_p10", "Y01"),
            ("joint", "Y11"),
        ):
            for control in range(1, 32):
                specs[f"{family}_{control:03d}"] = {
                    "family": family,
                    "control_id": control,
                    "world": world,
                }

    def amplitude(condition: str, spec: Mapping[str, Any], block: int) -> float:
        noise = 0.0015 * np.sin((block + 1) * (sum(map(ord, condition)) % 11 + 1))
        if condition == "Y00":
            base = 0.04 if directional_submaterial else 0.30
        elif condition == "Y10":
            base = 0.02 if directional_submaterial else 0.15
        elif condition == "Y01":
            base = 0.17
        elif condition == "Y11":
            base = -0.08
        elif str(spec["family"]).startswith("p3_") and spec["family"] not in {
            "joint_p3"
        }:
            base = 0.282 + int(spec["control_id"]) * 1e-5
        elif str(spec["family"]).startswith("p10_") and spec["family"] not in {
            "joint_p10"
        }:
            base = 0.284 + int(spec["control_id"]) * 1e-5
        elif spec["family"] == "joint_p3":
            base = 0.282
        elif spec["family"] == "joint_p10":
            base = 0.283
        elif spec["family"] == "joint":
            base = 0.268
        else:
            raise AssertionError(spec)
        return float(base + noise)

    rows: list[dict[str, Any]] = []
    for condition, spec in specs.items():
        condition_code = sum(map(ord, condition)) % 13
        for block in range(blocks):
            amp = amplitude(condition, spec, block)
            for donor in range(3):
                for acceptor in range(3):
                    delta = (acceptor - donor) % 3
                    if delta == 0:
                        coupling = amp + (donor - 1) * 0.004
                    elif delta == 1:
                        coupling = 0.002 * np.sin(block + donor + 0.4)
                    else:
                        coupling = 0.002 * np.cos(block + donor + 0.7)
                    epsilon = (
                        0.0015
                        * np.sin((block + 1) * (condition_code + 1) + donor)
                        if delta == 0
                        else 0.0002 * np.cos(block + acceptor)
                    )
                    coupling_at = float(coupling - epsilon)
                    f_intact = 1.0 + block * 1e-5
                    rows.append(
                        {
                            "condition_id": condition,
                            "family": spec["family"],
                            "control_id": spec["control_id"],
                            "world": spec["world"],
                            "cell_id": f"b{block:03d}-d{donor}-a{acceptor}",
                            "block_id": f"b{block:03d}",
                            "chrom": f"chr{1 + block % 3}",
                            "strand": "+" if block % 2 == 0 else "-",
                            "donor_phase": donor,
                            "acceptor_phase": acceptor,
                            "f_intact": f_intact,
                            "f_ac": f_intact - coupling,
                            "f_at": f_intact - coupling_at,
                            "coupling_ac": float(coupling),
                            "coupling_at": coupling_at,
                        }
                    )
    raw = {
        "schema": target_runner.TARGET_SCHEMA,
        "stage": "target",
        "status": "TARGET_SCORED_UNANALYZED",
        "target_mode": "FULL_FACTORIAL" if full else "P3_ONLY",
        "target_seal_sha256": "a" * 64,
        "split_role": "confirmation",
        "n_blocks": blocks,
        "n_cells": blocks * 9,
        "n_controls_per_family": 31,
        "condition_specs": specs,
        "mechanical_audit": {"passed": True},
        "rows": rows,
    }
    return target_runner.compact_target_result(raw)


def _passing_measurement(*, p10: bool) -> dict[str, Any]:
    result = {
        "control_families": {
            family: {
                "n_controls": 31,
                "target_exceeds_every_control": True,
                "add_one_p": 1 / 32,
                "paired_excess_ci": [0.02, 0.2],
            }
            for family in ("phase", "axis")
        },
        "cross_carrier_retention": 0.95,
        "cross_carrier_retention_ci": [0.90, 1.0],
        "cross_carrier_ratio_health": {"pass": True},
        "rescue_recovery": 1.0,
        "rescue_recovery_ci": [0.95, 1.0],
        "rescue_forward_max_abs_error": 0.0,
        "rescue_ratio_health": {"pass": True},
        "health_pass": True,
        "all_control_health_pass": True,
    }
    if p10:
        result.update(
            {
                "intact_carrier_score_ci": [0.02, 0.10],
                "denominator_finite_sign_stable_fraction": 1.0,
                "capability_ratio_health": {"pass": True},
                "capability_fraction_reduced": 0.70,
                "capability_fraction_reduced_ci": [0.40, 0.90],
                "joint_health_pass": False,
            }
        )
    else:
        result["capability_ci"] = [0.08, 0.20]
    return result


class _FakePreflightProvider:
    def __init__(self) -> None:
        self.table = np.random.default_rng(9).normal(size=(1026, 4))

    def position_table(self) -> np.ndarray:
        return self.table.copy()

    def measure_dose(self, *, band: str, **_: Any) -> Mapping[str, Any]:
        return _passing_measurement(p10=band == "p10")

    def measure_joint_p10(self, *, p10_dose: float, **_: Any) -> Mapping[str, Any]:
        passed = p10_dose <= 0.75
        return {
            "cross_carrier_retention": 0.95 if passed else 0.70,
            "cross_carrier_retention_ci": [0.90, 1.0] if passed else [0.60, 0.80],
            "joint_health_pass": passed,
            "joint_k3_evaluated": True,
        }

    def measure_shared_gates(self, **_: Any) -> Mapping[str, Any]:
        gates = {
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
        gates["diagnostics"] = {
            "fixture_calibration": {"summary": {"optional_power_pass": True}}
        }
        return gates

    def measure_authorized_table_health(
        self,
        *,
        tables: Mapping[str, np.ndarray],
        table_specs: Mapping[str, Mapping[str, Any]],
        target_mode: str,
        **_: Any,
    ) -> Mapping[str, Any]:
        return {
            "schema": preflight_runner.TABLE_HEALTH_SCHEMA,
            "target_mode": target_mode,
            "candidate_package_sha256": "a" * 64,
            "n_blocks": 120,
            "n_tables": len(tables),
            "tables": {
                label: {"spec": dict(table_specs[label]), "pass": True}
                for label in tables
            },
            "all_tables_pass": True,
        }

    def development_mde(self, **_: Any) -> Mapping[str, Mapping[str, Any]]:
        return {
            name: {
                "mde80": value,
                "monte_carlo_se": 0.0001,
                "n_bootstrap": 20_000,
                "target_blocks": 600,
            }
            for name, value in (
                ("clean_psi", 0.03),
                ("N3", 0.03),
                ("N10", 0.04),
                ("J", 0.04),
            )
        }

    def metadata(self) -> Mapping[str, Any]:
        return {"fixture": True}


def test_target_shards_are_deterministic_disjoint_and_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(target_runner, "CONFIRMATION_BLOCKS", 12)
    monkeypatch.setattr(
        target_runner,
        "mechanical_audit",
        lambda *_args, **_kwargs: {"passed": True, "fixture": True},
    )
    table = np.random.default_rng(3).normal(size=(1026, 4))
    seal = _seal(table, tmp_path)
    cohort = _cohort(12, role="confirmation")
    scorer = _FakeScorer(table)
    shards = []
    for index in range(3):
        shard = target_runner.execute_target(
            seal=seal,
            cohort=cohort,
            scorer=scorer,
            split_role="confirmation",
            n_controls=31,
            control_seed=20260810,
            shard_index=index,
            shard_count=3,
        )
        shard["source_cohort_sha256"] = "b" * 64
        shard["source_seal_file_sha256"] = "c" * 64
        shards.append(shard)
    shard_manifest = target_runner.target_manifest_config(shards[0], seal)
    assert shard_manifest["compact_evidence_schema"] is None
    assert shard_manifest["raw_row_count"] == len(shards[0]["rows"])
    assert len(shard_manifest["raw_endpoint_stream_sha256"]) == 64
    assembled = target_runner.assemble_target_shards(shards)
    assert assembled["schema"] == target_runner.TARGET_SCHEMA
    assert assembled["status"] == "TARGET_SCORED_UNANALYZED"
    assert assembled["n_blocks"] == 12
    assert "rows" not in assembled
    assert (
        assembled["compact_evidence"]["schema"]
        == target_runner.COMPACT_EVIDENCE_SCHEMA
    )
    assert len(assembled["compact_evidence"]["blocks"]) == 12
    assert set(assembled["compact_evidence"]["condition_endpoints"]) == (
        set(assembled["condition_specs"]) - {"Y00"}
    )
    assert len(assembled["assembly"]["block_ids"]) == 12
    assert len(set(assembled["assembly"]["block_ids"])) == 12
    with pytest.raises(RuntimeError, match="every index exactly once"):
        target_runner.assemble_target_shards([shards[0], shards[0], shards[2]])


def test_preflight_steps_p10_down_without_changing_selected_p3(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        preflight_runner,
        "build_phase_control_family",
        lambda *_args, **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        preflight_runner,
        "mechanical_audit",
        lambda *_args, **_kwargs: {"passed": True},
    )
    result = preflight_runner.execute_preflight(
        cohort=_cohort(120, role="development"),
        candidate_package={},
        provider=_FakePreflightProvider(),
        split_role="development",
        shared_gate_source=None,
        construction_audit_pass=True,
        n_controls=31,
        control_seed=20260810,
    )
    assert result["status"] == "FULL_FACTORIAL_TARGET_AUTHORIZED"
    assert result["evaluation"]["p3"]["selected_dose"] == 1.0
    assert result["evaluation"]["p10"]["selected_dose"] == 0.75
    assert result["measurements"]["p10_joint_step_down"]["1.00"]["status"] == "FAIL_STEP_DOWN"
    assert result["measurements"]["p10_joint_step_down"]["0.75"]["status"] == "PASS"


def test_preflight_and_independent_audit_require_exactly_120_development_blocks():
    with pytest.raises(RuntimeError, match="exactly 120 complete blocks"):
        preflight_runner.execute_preflight(
            cohort=_cohort(121, role="development"),
            candidate_package={},
            provider=_FakePreflightProvider(),
            split_role="development",
            shared_gate_source=None,
            construction_audit_pass=True,
            n_controls=31,
            control_seed=20260810,
        )
    with pytest.raises(ValueError, match="120-block development support"):
        preflight_audit_runner._recompute_health_table({"raw_blocks": [{}] * 121})


def test_preflight_requires_the_exact_authorized_construction_audit_contract():
    valid = {
        "schema": "gfmcd.positional_boundary_coherence.cohort_audit.v1",
        "stage": "construction",
        "status": "AUDIT_PASS_TARGET_AUTHORIZED",
        "passed": True,
        "target_execution_authorized": True,
    }
    assert preflight_runner.construction_audit_authorizes_target(valid)
    assert not preflight_runner.construction_audit_authorizes_target(
        {**valid, "status": "PASS"}
    )
    assert not preflight_runner.construction_audit_authorizes_target(
        {**valid, "schema": "wrong"}
    )
    assert not preflight_runner.construction_audit_authorizes_target(
        {**valid, "target_execution_authorized": False}
    )


def test_preflight_requires_and_hash_binds_construction_audit_sibling_manifest(
    tmp_path: Path,
):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"source": True}))
    audit_path = tmp_path / "cohort.audit.json"
    audit = {
        "schema": "gfmcd.positional_boundary_coherence.cohort_audit.v1",
        "stage": "construction",
        "status": "AUDIT_PASS_TARGET_AUTHORIZED",
        "passed": True,
        "target_execution_authorized": True,
    }
    audit_path.write_text(json.dumps(audit))
    manifest_path = tmp_path / "cohort.audit.manifest.json"
    manifest = {
        "experiment_id": (
            "splicebert-positional-boundary-coherence:"
            "composition-balanced-cohort-audit-v1"
        ),
        "status": "AUDIT_PASS_TARGET_AUTHORIZED",
        "config": {"stage": "construction"},
        "inputs": [file_record(source)],
        "outputs": [file_record(audit_path)],
    }
    manifest_path.write_text(json.dumps(manifest))
    provenance = preflight_runner.validate_construction_audit_chain(
        audit,
        audit_path=audit_path,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    assert provenance == {
        "artifact_sha256": sha256_file(audit_path),
        "manifest_sha256": sha256_file(manifest_path),
        "experiment_id": (
            "splicebert-positional-boundary-coherence:"
            "composition-balanced-cohort-audit-v1"
        ),
    }
    preflight_manifest = {
        "inputs": [file_record(audit_path), file_record(manifest_path)]
    }
    independent_checks = preflight_audit_runner._construction_audit_contract(
        {"construction_audit_provenance": provenance}, preflight_manifest
    )
    assert all(independent_checks.values())
    substituted = {
        "construction_audit_provenance": {
            **provenance,
            "manifest_sha256": sha256_file(source),
        }
    }
    assert not all(
        preflight_audit_runner._construction_audit_contract(
            substituted, preflight_manifest
        ).values()
    )
    wrong_manifest = {**manifest, "outputs": [file_record(source)]}
    with pytest.raises(RuntimeError, match="sibling producer manifest"):
        preflight_runner.validate_construction_audit_chain(
            audit,
            audit_path=audit_path,
            manifest=wrong_manifest,
            manifest_path=manifest_path,
        )


def test_p3_only_analysis_authorizes_closure_with_all_clean_direction_checks():
    result = analysis_runner.analyze_factorial_rows(
        _synthetic_target(full=False),
        n_bootstrap=analysis_runner.CONFIRMATORY_BOOTSTRAP_DRAWS,
        bootstrap_seed=analysis_runner.CONFIRMATORY_BOOTSTRAP_SEED,
        n_pseudo=analysis_runner.CONFIRMATORY_PSEUDO_CONTROLS,
        pseudo_seed=analysis_runner.CONFIRMATORY_PSEUDO_SEED,
    )
    assert result["clean_behavior_classification"] == "BALANCED_PREFERENCE_MATERIAL"
    assert result["clean_directional_family"]["all_simultaneous_lower_bounds_above_zero"]
    assert result["mismatch_equivalence_family"]["simultaneous_equivalence_within_sesoi"]
    assert result["structured_null_specificity"]["N3"]["corrected_pass"]
    assert result["verdict"] == "P3_ONLY_CLOSURE_PATH"
    assert result["authorization"]["authorize_p3_phase3"]
    assert not result["authorization"]["authorize_joint_phase3"]
    assert (
        result["phase2_analysis_engine"]
        == "analyze_carrier_factorial.analyze_fixture_arrays"
    )


def test_directional_scoped_phase2_tier_is_independently_recomputed():
    target = _synthetic_target(full=False, directional_submaterial=True)
    result = analysis_runner.analyze_factorial_rows(
        target,
        n_bootstrap=analysis_runner.CONFIRMATORY_BOOTSTRAP_DRAWS,
        bootstrap_seed=analysis_runner.CONFIRMATORY_BOOTSTRAP_SEED,
        n_pseudo=analysis_runner.CONFIRMATORY_PSEUDO_CONTROLS,
        pseudo_seed=analysis_runner.CONFIRMATORY_PSEUDO_SEED,
    )
    assert result["verdict"] == "P3_DIRECTIONAL_SCOPED_SUPPORT"
    assert result["directional_scoped_support"]
    assert result["support_tiers"] == ["P3_DIRECTIONAL_SCOPED_SUPPORT"]
    assert not result["authorization"]["authorize_p3_phase3"]
    independent = audit_runner._independent_statistical_recompute(target, result)
    assert independent["passed"]
    assert independent["expected_verdict"] == "P3_DIRECTIONAL_SCOPED_SUPPORT"
    assert not independent["expected_p3_authorization"]


def test_compact_target_is_the_complete_independently_auditable_evidence():
    target = _synthetic_target(full=True)
    contract, worlds = audit_runner._row_contract(target)
    assert contract["passed"]
    assert contract["raw_rows_absent"]
    assert contract["n_y00_cells"] == 108
    assert set(worlds) == {"Y00", "Y10", "Y01", "Y11"}
    assert all(len(values) == 12 for values in worlds.values())
    assert "psi_at" not in target["compact_evidence"]["condition_endpoints"][
        "p3_phase_001"
    ]
    assert "psi_at" in target["compact_evidence"]["condition_endpoints"]["Y10"]

    contaminated = {**target, "rows": []}
    contaminated_contract, _ = audit_runner._row_contract(contaminated)
    assert not contaminated_contract["passed"]


def test_fixture_calibration_uses_shared_analysis_and_decision_engines():
    null = calibration_runner.analyze_replicate(
        replicate=0,
        effect=0.0,
        group_index=0,
        n_blocks=600,
        n_bootstrap=999,
    )
    planted = calibration_runner.analyze_replicate(
        replicate=0,
        effect=0.10,
        group_index=3,
        n_blocks=600,
        n_bootstrap=999,
    )
    result_engine = "analyze_carrier_factorial.analyze_fixture_arrays"
    assert null["analysis_engine"] == result_engine
    assert planted["analysis_engine"] == result_engine
    assert not null["decision"]["authorize_p3_phase3"]
    assert planted["decision"]["authorize_p3_phase3"]
    assert planted["decision"]["authorize_joint_phase3"]
    for row in (null, planted):
        expected = bool(row["decision"]["directional_scoped_support"])
        assert preflight_runner._fixture_directional_scoped_pass(row) is expected
        assert preflight_audit_runner._fixture_directional_scoped(row) is expected


def test_negative_joint_interaction_uses_absolute_rank_and_frozen_observed_sign():
    result = analysis_runner.analyze_factorial_rows(
        _synthetic_target(full=True),
        n_bootstrap=analysis_runner.CONFIRMATORY_BOOTSTRAP_DRAWS,
        bootstrap_seed=analysis_runner.CONFIRMATORY_BOOTSTRAP_SEED,
        n_pseudo=analysis_runner.CONFIRMATORY_PSEUDO_CONTROLS,
        pseudo_seed=analysis_runner.CONFIRMATORY_PSEUDO_SEED,
    )
    joint = result["structured_null_specificity"]["J"]
    assert result["endpoints"]["J"]["estimate"] < -0.05
    assert joint["observed_direction"] == "negative_super_additive"
    assert joint["absolute_add_one_p"] == 1 / 32
    assert joint["paired_signed_lower_above_zero"]
    assert joint["corrected_pass"]
    assert result["verdict"] == "DUAL_SCALE_INTERACTION_SUPPORTED"
    assert result["authorization"]["authorize_joint_phase3"]


def test_audit_contract_hash_binds_analysis_and_fails_a_tampered_target_hash():
    target = _synthetic_target(full=False)
    analysis = analysis_runner.analyze_factorial_rows(
        target,
        n_bootstrap=analysis_runner.CONFIRMATORY_BOOTSTRAP_DRAWS,
        bootstrap_seed=analysis_runner.CONFIRMATORY_BOOTSTRAP_SEED,
        n_pseudo=analysis_runner.CONFIRMATORY_PSEUDO_CONTROLS,
        pseudo_seed=analysis_runner.CONFIRMATORY_PSEUDO_SEED,
    )
    analysis["target_sha256"] = "wrong"
    preflight = {
        "schema": preflight_runner.PREFLIGHT_SCHEMA,
        "status": "P3_ONLY_TARGET_AUTHORIZED",
        "target_mode": "P3_ONLY",
        "evaluation": {"passed": True},
    }
    audit = audit_runner.audit_payloads(
        preflight,
        {"schema": audit_runner.SEAL_SCHEMA, "target_mode": "P3_ONLY", "pinned_files": {}},
        target,
        analysis,
        preflight_sha256="p" * 64,
        target_sha256="t" * 64,
        analysis_sha256="a" * 64,
        provenance_checks={"fixture_provenance": True},
    )
    assert audit["schema"] == "gfmcd.pbc.carrier_factorial_audit.v1"
    assert audit["status"] == "FAIL"
    assert audit["analysis_sha256"] == "a" * 64
    assert audit["target_mode"] == "P3_ONLY"
    assert "target_hash_bound" in audit["failures"]
    assert not audit["authorization"]["authorize_p3_phase3"]
