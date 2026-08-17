"""Focused memorials for the corrected R1c behavior-only successor."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from gfmcd.splicing.coupling import CouplingExample
from gfmcd.splicing.paired_component import FamilyDecision


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/splicebert_acceptor_coupling/run_r1c_sealed_reverse_behavior.py"
SPEC = importlib.util.spec_from_file_location("run_r1c_sealed_reverse_behavior", RUNNER)
assert SPEC is not None and SPEC.loader is not None
r1c = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r1c
SPEC.loader.exec_module(r1c)


def _example(index: int = 0, chrom: str = "chr11") -> CouplingExample:
    sequence = list("C" * 200)
    for position, motif in ((28, "AG"), (65, "GT"), (80, "GT"), (118, "AG"), (138, "AG")):
        sequence[position : position + 2] = motif
    return CouplingExample(
        example_id=f"example-{index}",
        sequence="".join(sequence),
        acceptor_pos=30,
        donor_pos=80,
        chrom=chrom,
        strand="+" if index % 2 == 0 else "-",
        gene_id=f"gene-{index}",
        split="test" if chrom in {"chr8", "chr9"} else "val",
        exon_length=50,
        intron_length_down=60,
        n_isoforms_donor=2,
        n_isoforms_acceptor=2,
        gene_type="protein_coding",
        other_ag=(120,),
        other_gt=(65,),
        next_acceptor_pos=140,
    )


def _build_valid_preflight_record() -> dict:
    n_genes = 40
    groups = [f"gene-{index % n_genes}" for index in range(r1c.TARGET_N)]
    family = np.tile(np.asarray([[0.40], [0.0], [0.40], [0.40]]), (1, r1c.TARGET_N))
    inference = r1c.r1c_boundary_classifier(
        family,
        groups,
        sesoi=r1c.SESOI,
        n_randomisation=r1c.PRODUCTION_RANDOMISATION,
        alpha=r1c.FAMILY_ALPHA,
        seed=r1c.PRODUCTION_SEED,
        namespace="r1c:preflight:production-boundary",
        min_genes=r1c.MIN_GENES,
        studentized=True,
    )
    robustness = r1c.r1c_gene_robustness_gate(
        family,
        groups,
        n_draws=r1c.PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
        seed=r1c.PRODUCTION_SEED,
        namespace="r1c:preflight:production-gene-robustness",
    )
    stack = r1c.downstream_r5_stack_identity(repo=REPO)
    null_inference = r1c.r1c_boundary_classifier(
        np.zeros_like(family),
        groups,
        sesoi=r1c.SESOI,
        n_randomisation=r1c.PRODUCTION_RANDOMISATION,
        alpha=r1c.FAMILY_ALPHA,
        seed=r1c.PRODUCTION_SEED,
        namespace="r1c:fixture:null-boundary",
        min_genes=r1c.MIN_GENES,
        studentized=True,
    )
    false_positive_counts = {
        path: {
            case_name: 0
            for case_name, case in r1c.least_favourable_boundary_cases().items()
            if path in case["null_paths"]
        }
        for path in r1c.PATH_NAMES
    }
    fpr_upper = r1c.one_sided_binomial_bounds(
        0, r1c.PRODUCTION_TRIALS, alpha=r1c.FAMILY_ALPHA
    )[1]
    power_lower = r1c.one_sided_binomial_bounds(
        r1c.PRODUCTION_TRIALS,
        r1c.PRODUCTION_TRIALS,
        alpha=r1c.FAMILY_ALPHA,
    )[0]
    def fixture_classifier_audit(namespace: str, source: dict) -> dict:
        audit = copy.deepcopy(r1c._classifier_trial_audit(source))
        audit["namespace"] = namespace
        boundary_namespace = f"{namespace}:shared-boundary-signs"
        for test in audit["boundary_tests"].values():
            test["namespace"] = boundary_namespace
        audit["shared_boundary_randomisation"]["namespace"] = boundary_namespace
        return audit

    case_names = list(r1c.least_favourable_boundary_cases())
    trial_records = []
    for index in range(r1c.PRODUCTION_TRIALS):
        prefix = "r1c:production-path-preflight"
        trial_records.append(
            {
                "trial": index,
                "outer_sign_namespace": f"{prefix}:centered-primitive-null:{index}",
                "least_favourable_cases": {
                    case_name: fixture_classifier_audit(
                        f"{prefix}:lfc:{case_name}:{index}", null_inference
                    )
                    for case_name in case_names
                },
                "coherent_plant": fixture_classifier_audit(
                    f"{prefix}:coherent-plant:{index}", inference
                ),
            }
        )
    calibration = {
        "randomisation_unit": "gene",
        "outer_trial_unit": "endpoint-centered 2x2 cell residual",
        "outer_trial_centering": "subtract each observed cell mean before sign flipping",
        "outer_trial_algebra": (
            "one shared sign per gene across Y_PD/Y_PN/Y_XD/Y_XN; "
            "derive S_P and I after flip"
        ),
        "translation_invariant": True,
        "classifier": (
            "exact one-sided boundary tests + Y_PN TOST + path IUT + Holm over exactly "
            "two named paths + strict simultaneous-CI gate on every LFC and plant trial"
        ),
        "classifier_schema": "gfmcd.r1c.boundary_classifier/1",
        "boundary_test_contract": r1c.boundary_test_contract(),
        "namespace": "r1c:production-path-preflight",
        "alpha": r1c.FAMILY_ALPHA,
        "sesoi": r1c.SESOI,
        "n_components": len(r1c.PRIMARY_FAMILY_NAMES),
        "n_examples": r1c.TARGET_N,
        "n_genes": n_genes,
        "minimum_genes_required": r1c.MIN_GENES,
        "n_calibration": r1c.PRODUCTION_RANDOMISATION,
        "n_trials": r1c.PRODUCTION_TRIALS,
        "reference_interval_calls": 1,
        "production_classifier_calls": r1c.PRODUCTION_TRIALS
        * (len(r1c.least_favourable_boundary_cases()) + 1),
        "simultaneous_half_width": 0.0,
        "simultaneous_half_widths": {
            name: 0.0 for name in r1c.PRIMARY_FAMILY_NAMES
        },
        "studentized": True,
        "interval_scaling": "fixed endpoint bootstrap SD; not nested bootstrap-t",
        "resolution_ok": True,
        "least_favourable_boundary_cases": r1c.least_favourable_boundary_cases(),
        "nuisance_continuum_control": r1c.nuisance_continuum_control(),
        "lfc_numeric_guard_toward_composite_null": r1c.LFC_NUMERIC_GUARD,
        "boundary_false_positive_counts": false_positive_counts,
        "worst_case_false_positives": {path: 0 for path in r1c.PATH_NAMES},
        "maximum_false_positives_allowed": r1c.MAX_FALSE_POSITIVES,
        "path_boundary_fpr": {path: 0.0 for path in r1c.PATH_NAMES},
        "path_boundary_fpr_upper": {
            path: fpr_upper for path in r1c.PATH_NAMES
        },
        "path_boundary_fpr_ok": {path: True for path in r1c.PATH_NAMES},
        "planted_effect": None,
        "plant_delta": r1c.PLANT_DELTA,
        "cell_plant_vector": r1c.cell_plant_vector().tolist(),
        "plant_vector": r1c.structured_plant_vector().tolist(),
        "planted_requirements": r1c._component_plant_requirements(),
        "decision_paths": r1c.decision_path_requirements(component_names=True),
        "path_holm_family": list(r1c.PATH_NAMES),
        "plant_successes": {
            path: r1c.PRODUCTION_TRIALS for path in r1c.PATH_NAMES
        },
        "minimum_power_successes_required": r1c.MIN_POWER_SUCCESSES,
        "path_planted_power": {path: 1.0 for path in r1c.PATH_NAMES},
        "path_planted_power_lower": {
            path: power_lower for path in r1c.PATH_NAMES
        },
        "path_planted_power_ok": {path: True for path in r1c.PATH_NAMES},
        "power_floor": r1c.PREFLIGHT_POWER_FLOOR,
        "binomial_bound": "one-sided 95% Clopper-Pearson",
        "trial_records": trial_records,
        "trial_records_sha256": r1c.canonical_sha256(trial_records),
    }
    rows = []
    for index, gene_id in enumerate(groups):
        rows.append(
            {
                "example_id": f"fixture-example-{index}",
                "gene_id": gene_id,
                "chrom": r1c.VALIDATION_CHROMOSOMES[index % 3],
                "strand": "+" if index % 2 == 0 else "-",
                "positions": {
                    "prior_real_acceptor": 30,
                    "paired_donor": 80,
                    "partner_acceptor": 140,
                    "matched_decoy_donor": 65,
                    "matched_decoy_donor_distance_mismatch": 15,
                    "dyad_distance_mismatch": 10,
                },
                "family": {
                    "Y_PD": 0.40,
                    "Y_PN": 0.0,
                    "S_P": 0.40,
                    "I": 0.40,
                },
                "descriptive_2x2": {"Y_XD": 0.0, "Y_XN": 0.0, "S_X": 0.0},
                "cell_effects": {
                    "Y_PD": 0.40,
                    "Y_PN": 0.0,
                    "Y_XD": 0.0,
                    "Y_XN": 0.0,
                },
                "margin_levels": {
                    "P_clean": 1.0,
                    "P_under_D": 0.6,
                    "P_under_N": 1.0,
                    "X_clean": 1.0,
                    "X_under_D": 1.0,
                    "X_under_N": 1.0,
                },
                "health": {
                    "clean_accuracy": 0.5,
                    "corrupt_accuracy": 0.5,
                    "damage": 0.0,
                },
            }
        )
    cohort_payload = r1c._cohort_payload_from_rows(rows)
    selected_example_ids = [row["example_id"] for row in rows]
    return {
        "schema": "gfmcd.r1c.reverse_behavior/3",
        "experiment": "R1c_corrected_reverse_behavior",
        "protocol_status": "FROZEN_HASH_PINNED",
        "freeze_authorization": {
            "protocol_declares_frozen": True,
            "protocol_hash_pinned": True,
            "instrument_fingerprint_pinned": True,
            "downstream_r5_stack_pinned": True,
            "strict_mps_runtime_verified": True,
            "authorization_ready": True,
        },
        "stage": "preflight",
        "status": "DEVELOPMENT",
        "evidence_rung": "BEHAVIOR_ONLY",
        "split": "val",
        "sealed_split_opened": False,
        "holdout_scope_disclosure": r1c.holdout_scope_disclosure(),
        "n": r1c.TARGET_N,
        "n_genes": n_genes,
        "cohort_sha256": r1c.canonical_sha256(cohort_payload),
        "all_gates_pass": True,
        "verdict": "PREFLIGHT_GO",
        "config": r1c._preflight_config_contract(),
        "cohort_audit": {
            "screened": 6000,
            "screened_example_ids_sha256": "a" * 64,
            "primary_comparator_complete": 1500,
            "primary_comparator_complete_example_ids_sha256": "b" * 64,
            "distance_matched_2x2_eligible": 425,
            "distance_matched_2x2_genes": 339,
            "distance_matched_2x2_example_ids_sha256": "c" * 64,
            "dyad_distance_definition": "abs(exon_length-intron_length_down)",
            "dyad_distance_caliper": r1c.DYAD_DISTANCE_CALIPER,
            "selected": r1c.TARGET_N,
            "selected_genes": n_genes,
            "selected_example_ids_sha256": r1c.canonical_sha256(
                selected_example_ids
            ),
            "selected_cohort_sha256": r1c.canonical_sha256(cohort_payload),
            "exclusions": {},
            "selection_reads_model_outputs": False,
        },
        "target_cohort_identity": r1c.target_cohort_identity(rows),
        "gates": {name: True for name in r1c.PREFLIGHT_GATE_NAMES},
        "family_decisions": copy.deepcopy(inference["family_decisions"]),
        "boundary_inference": inference,
        "gene_robustness": robustness,
        "observed_decision_paths": r1c.observed_decision_paths(inference),
        "family_calibration": copy.deepcopy(inference["family_calibration"]),
        "production_path_preflight": calibration,
        "health": {
            "n_probe_per_example_per_construct": r1c.PRODUCTION_HEALTH_PROBES,
            "clean_mean": 0.5,
            "corrupt_mean": 0.5,
            "damage_mean": 0.0,
            "floor": r1c.HEALTH_FLOOR,
            "damage_limit": r1c.HEALTH_DAMAGE_LIMIT,
        },
        "structured_plant": {
            "delta": r1c.PLANT_DELTA,
            "cell_vector": r1c.cell_plant_vector().tolist(),
            "vector": r1c.structured_plant_vector().tolist(),
            "requirements": {
                r1c.PRIMARY_FAMILY_NAMES[index]: label
                for index, label in r1c.PRIMARY_PLANT_REQUIREMENTS.items()
            },
        },
        "instrument_fingerprint": {
            "contract": r1c.frozen_instrument_contract(),
            "runtime_contract": {
                "n": r1c.TARGET_N,
                "batch": r1c.PRODUCTION_BATCH,
                "device": "mps",
                "seed": r1c.PRODUCTION_SEED,
                "n_randomisation": r1c.PRODUCTION_RANDOMISATION,
                "n_trials": r1c.PRODUCTION_TRIALS,
                "robust_bootstrap_draws": r1c.PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
                "health_probes": r1c.PRODUCTION_HEALTH_PROBES,
                "strict_runtime_policy": r1c.authorize_production_runtime(
                    "mps", environ={}
                ),
                "r5_stack_fingerprint_sha256": stack[
                    r1c.STACK_FINGERPRINT_FIELD
                ],
            },
            "fingerprint_sha256": "same",
        },
        r1c.STACK_BINDING_FIELD: stack[r1c.STACK_BINDING_FIELD],
        r1c.STACK_FINGERPRINT_FIELD: stack[r1c.STACK_FINGERPRINT_FIELD],
        "rows": rows,
    }


_VALID_PREFLIGHT_TEMPLATE: dict | None = None


def _valid_preflight_record() -> dict:
    global _VALID_PREFLIGHT_TEMPLATE
    if _VALID_PREFLIGHT_TEMPLATE is None:
        _VALID_PREFLIGHT_TEMPLATE = _build_valid_preflight_record()
    return copy.deepcopy(_VALID_PREFLIGHT_TEMPLATE)


def test_successor_exposes_model_free_target_package_verifier() -> None:
    assert callable(r1c.verify_successor_target_package)


def _fixture_sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _fixture_source(logical_name: str, label: str) -> dict:
    return {
        "logical_name": logical_name,
        "location_scope": "external",
        "size_bytes": len(label) + 1,
        "sha256": _fixture_sha(label),
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _package_identity(path: Path) -> dict:
    return {
        "logical_name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fixture_bound_instrument(stack_sha256: str) -> tuple[dict, list[dict]]:
    checkpoint_file = {
        "relative_path": "weights.bin",
        "size_bytes": 8,
        "sha256": _fixture_sha("checkpoint-weights"),
    }
    checkpoint_input = {
        "logical_name": "weights.bin",
        "location_scope": "external",
        "size_bytes": checkpoint_file["size_bytes"],
        "sha256": checkpoint_file["sha256"],
    }
    sources = {
        "gtf": _fixture_source("genes.gtf", "gtf"),
        "fasta": _fixture_source("genome.fa", "fasta"),
        "fasta_fai": _fixture_source("genome.fa.fai", "fasta-fai"),
        "protocol": _fixture_source("protocol.md", "protocol"),
        "statistics": _fixture_source("paired_component.py", "statistics"),
    }
    dependency = _fixture_source("coupling.py", "dependency")
    source = {
        "schema": "gfmcd.r1c.instrument_fingerprint/4",
        "checkpoint": {
            "root": "model",
            "n_files": 1,
            "files": [checkpoint_file],
            "bundle_sha256": r1c.canonical_sha256([checkpoint_file]),
        },
        **sources,
        "runner": {
            "logical_name": "scripts/run_r1c.py",
            "location_scope": "repository",
            "normalization": (
                "replace only top-level FROZEN_*_SHA256 assignment values"
            ),
            "normalized_size_bytes": 1,
            "normalized_sha256": _fixture_sha("runner"),
        },
        "measurement_dependencies": [dependency],
        "runtime": {"fixture": True},
        "contract": r1c.frozen_instrument_contract(),
    }
    source["fingerprint_sha256"] = r1c.canonical_sha256(source)
    runtime_contract = {
        "n": r1c.TARGET_N,
        "batch": r1c.PRODUCTION_BATCH,
        "device": "mps",
        "seed": r1c.PRODUCTION_SEED,
        "n_randomisation": r1c.PRODUCTION_RANDOMISATION,
        "n_trials": r1c.PRODUCTION_TRIALS,
        "robust_bootstrap_draws": r1c.PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
        "health_probes": r1c.PRODUCTION_HEALTH_PROBES,
        "strict_runtime_policy": r1c.authorize_production_runtime(
            "mps", environ={}
        ),
        "r5_stack_fingerprint_sha256": stack_sha256,
    }
    instrument = r1c.bind_runtime_contract(source, runtime_contract)
    return instrument, [checkpoint_input, *sources.values(), dependency]


def _build_valid_target_package(tmp_path: Path) -> dict[str, Path]:
    ledger_repo = tmp_path / "ledger-repo"
    ledger_repo.mkdir()
    subprocess.run(["git", "init", "-q", str(ledger_repo)], check=True)

    preflight = _valid_preflight_record()
    stack_sha256 = preflight[r1c.STACK_FINGERPRINT_FIELD]
    instrument, source_inputs = _fixture_bound_instrument(stack_sha256)
    protocol_sha256 = instrument["protocol"]["sha256"]
    preflight["protocol_sha256"] = protocol_sha256
    preflight["instrument_fingerprint"] = copy.deepcopy(instrument)
    preflight_inputs = sorted(
        source_inputs,
        key=lambda item: (
            item["location_scope"], item["logical_name"], item["sha256"]
        ),
    )
    preflight["input_snapshot_sha256"] = r1c.canonical_sha256(preflight_inputs)
    preflight["target_authorization"] = {
        "required": False,
        "pre_run_clean_worktree": True,
        "pre_run_git_commit": _fixture_sha("preflight-commit"),
        "explicit_unseal_acknowledgement": False,
        "production_authorization": {
            "stage": "preflight",
            "protocol_sha256": protocol_sha256,
            "instrument_fingerprint_pin_present": True,
            "r5_stack_fingerprint_sha256": stack_sha256,
            "preflight_artifact_pin_required": False,
            "explicit_unseal_acknowledgement": False,
        },
        "strict_runtime_policy": r1c.authorize_production_runtime(
            "mps", environ={}
        ),
        "model_loaded_from_private_staged_copy": True,
        "preflight_artifact_hash_pin_required": False,
        "preflight_artifact_hash_pin_matched": None,
    }

    preflight_path = tmp_path / "r1c-preflight.json"
    _write_json(preflight_path, preflight)
    preflight_manifest_path = preflight_path.with_suffix(".manifest.json")
    preflight_identity = _package_identity(preflight_path)
    _write_json(
        preflight_manifest_path,
        {
            "experiment_id": (
                "splicebert_acceptor_coupling:r1c_reverse_behavior"
            ),
            "inputs": [
                {
                    "path": item["logical_name"],
                    "location_scope": item["location_scope"],
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                }
                for item in preflight_inputs
            ],
            "outputs": [
                {
                    "path": f"/relocated/{preflight_path.name}",
                    "size_bytes": preflight_identity["size_bytes"],
                    "sha256": preflight_identity["sha256"],
                }
            ],
        },
    )
    preflight_package = r1c.verify_preflight_manifest(preflight_path)

    target = copy.deepcopy(preflight)
    for index, row in enumerate(target["rows"]):
        row["chrom"] = "chr8" if index % 2 == 0 else "chr9"
    family = np.asarray(
        [
            [row["family"][name] for row in target["rows"]]
            for name in r1c.PRIMARY_FAMILY_NAMES
        ],
        dtype=float,
    )
    groups = [row["gene_id"] for row in target["rows"]]
    inference = r1c.r1c_boundary_classifier(
        family,
        groups,
        sesoi=r1c.SESOI,
        n_randomisation=r1c.PRODUCTION_RANDOMISATION,
        alpha=r1c.FAMILY_ALPHA,
        seed=r1c.PRODUCTION_SEED,
        namespace="r1c:target:production-boundary",
        min_genes=r1c.MIN_GENES,
        studentized=True,
    )
    robustness = r1c.r1c_gene_robustness_gate(
        family,
        groups,
        n_draws=r1c.PRODUCTION_ROBUST_BOOTSTRAP_DRAWS,
        seed=r1c.PRODUCTION_SEED,
        namespace="r1c:target:production-gene-robustness",
    )
    cohort_identity = r1c.target_cohort_identity(target["rows"])
    target.update(
        {
            "stage": "target",
            "status": "CONFIRMATORY",
            "split": "test",
            "sealed_split_opened": True,
            "protocol_sha256": protocol_sha256,
            "preconstruction_allowed_chromosomes": ["chr8", "chr9"],
            "boundary_inference": inference,
            "gene_robustness": robustness,
            "family_decisions": copy.deepcopy(inference["family_decisions"]),
            "family_calibration": copy.deepcopy(inference["family_calibration"]),
            "observed_decision_paths": r1c.observed_decision_paths(inference),
            "production_path_preflight": None,
            "target_cohort_identity": cohort_identity,
            "cohort_sha256": cohort_identity["source_cohort_sha256"],
            "preflight_artifact": str(preflight_path),
            "preflight_package": preflight_package,
        }
    )
    target["cohort_audit"].update(
        {
            "selected": r1c.TARGET_N,
            "selected_genes": len(set(groups)),
            "selected_example_ids_sha256": r1c.canonical_sha256(
                [row["example_id"] for row in target["rows"]]
            ),
            "selected_cohort_sha256": cohort_identity["source_cohort_sha256"],
            "selection_reads_model_outputs": False,
        }
    )
    gates = {
        "minimum_30_genes": True,
        "zero_GT_to_AC_planted_motifs": True,
        "downstream_acceptor_clean_margin_floor": True,
        "exon_mate_acceptor_clean_margin_floor": True,
        "health_clean_floor": True,
        "health_corrupt_floor": True,
        "health_damage_bounded": True,
        "passing_exact_bound_preflight": True,
        "raw_trimmed_gene_robustness": bool(
            robustness["paths"]["core_paired_donor"]["pass"]
        ),
    }
    instrument_gates = {
        name: value
        for name, value in gates.items()
        if name != "raw_trimmed_gene_robustness"
    }
    verdict = r1c.production_verdict(
        inference, instrument_gates, "target", robustness
    )
    target["gates"] = gates
    target["all_gates_pass"] = all(gates.values())
    target["verdict"] = verdict

    pre_inputs = sorted(
        [
            *source_inputs,
            {
                "logical_name": preflight_path.name,
                "location_scope": "external",
                **{
                    key: value
                    for key, value in preflight_package["artifact"].items()
                    if key != "logical_name"
                },
            },
            {
                "logical_name": preflight_manifest_path.name,
                "location_scope": "external",
                **{
                    key: value
                    for key, value in preflight_package["manifest"].items()
                    if key != "logical_name"
                },
            },
        ],
        key=lambda item: (
            item["location_scope"], item["logical_name"], item["sha256"]
        ),
    )
    biological_identity = r1c.build_r1c_biological_target_identity(
        checkpoint_sha256=instrument["checkpoint"]["bundle_sha256"],
        gtf_sha256=instrument["gtf"]["sha256"],
        fasta_sha256=instrument["fasta"]["sha256"],
        fasta_fai_sha256=instrument["fasta_fai"]["sha256"],
    )
    artifact_path = tmp_path / "r1c-target.json"
    manifest_path = tmp_path / "r1c-target.manifest.json"
    completion_name = (
        f"{biological_identity['biological_target_key']}.completion.json"
    )
    receipt = r1c.reserve_r1c_target_attempt(
        repo=ledger_repo,
        identity=biological_identity,
        method_context={
            "protocol_sha256": protocol_sha256,
            "instrument_fingerprint_sha256": instrument[
                "fingerprint_sha256"
            ],
            "r5_stack_fingerprint_sha256": stack_sha256,
            "checkpoint_bundle_sha256": instrument["checkpoint"][
                "bundle_sha256"
            ],
            "construction_contract_sha256": r1c.construction_contract_sha256(),
            "endpoint_policy": r1c.TARGET_ENDPOINT_POLICY,
            "preflight_artifact": preflight_package["artifact"],
            "preflight_manifest": preflight_package["manifest"],
            "input_snapshot_sha256": r1c.canonical_sha256(pre_inputs),
            "requested_artifact_logical_name": artifact_path.name,
            "requested_manifest_logical_name": manifest_path.name,
            "declared_completion_logical_name": completion_name,
        },
    )
    receipt_path = Path(receipt["receipt_path"])
    receipt_base_identity = _package_identity(receipt_path)
    receipt_input = {
        "logical_name": receipt_path.name,
        "location_scope": "external",
        "size_bytes": receipt_base_identity["size_bytes"],
        "sha256": receipt_base_identity["sha256"],
    }
    full_inputs = sorted(
        [*pre_inputs, receipt_input],
        key=lambda item: (
            item["location_scope"], item["logical_name"], item["sha256"]
        ),
    )
    target["input_snapshot_sha256"] = r1c.canonical_sha256(full_inputs)
    target["target_attempt"] = receipt_base_identity | {
        "attempt_id": receipt["attempt_id"],
        "biological_target_key": receipt["biological_target_key"],
        "status": receipt["status"],
    }
    target["target_completion"] = {"logical_name": completion_name}
    target["target_attempt_disposition"] = (
        "SEALED_OUTCOMES_MAY_HAVE_BEEN_SPENT_EVEN_IF_THIS_RUN_CRASHES"
    )
    target["target_authorization"] = {
        "required": True,
        "pre_run_clean_worktree": True,
        "pre_run_git_commit": _fixture_sha("commit"),
        "explicit_unseal_acknowledgement": True,
        "production_authorization": {
            "stage": "target",
            "protocol_sha256": protocol_sha256,
            "instrument_fingerprint_pin_present": True,
            "r5_stack_fingerprint_sha256": stack_sha256,
            "preflight_artifact_pin_required": True,
            "explicit_unseal_acknowledgement": True,
        },
        "strict_runtime_policy": r1c.authorize_production_runtime(
            "mps", environ={}
        ),
        "model_loaded_from_private_staged_copy": True,
        "preflight_artifact_hash_pin_required": True,
        "preflight_artifact_hash_pin_matched": True,
    }
    _write_json(artifact_path, target)
    artifact_identity = _package_identity(artifact_path)
    _write_json(
        manifest_path,
        {
            "experiment_id": (
                "splicebert_acceptor_coupling:r1c_reverse_behavior"
            ),
            "inputs": [
                {
                    "path": item["logical_name"],
                    "location_scope": item["location_scope"],
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                }
                for item in full_inputs
            ],
            "outputs": [
                {
                    "path": f"/relocated/{artifact_path.name}",
                    "size_bytes": artifact_identity["size_bytes"],
                    "sha256": artifact_identity["sha256"],
                }
            ],
        },
    )
    completion = r1c.complete_r1c_target_attempt(
        receipt=receipt,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        cohort_digests=cohort_identity,
        method_fingerprint_sha256=instrument["fingerprint_sha256"],
        r5_stack_fingerprint_sha256=stack_sha256,
        official_verdict=verdict,
        all_gates_pass=all(gates.values()),
        gates=gates,
    )
    return {
        "artifact": artifact_path,
        "manifest": manifest_path,
        "attempt": receipt_path,
        "completion": Path(completion["completion_path"]),
        "preflight": preflight_path,
    }


def test_model_free_target_package_verifier_replays_science_and_seals(
    tmp_path: Path,
) -> None:
    package = _build_valid_target_package(tmp_path)
    handoff = r1c.verify_successor_target_package(
        package["artifact"],
        package["manifest"],
        package["attempt"],
        package["completion"],
        preflight_artifact_path=package["preflight"],
    )
    assert handoff["schema"] == "gfmcd.r1c.verified_target_handoff/1"
    assert handoff["verdict"] == "CONFIRMED_DOWNSTREAM_ACCEPTOR_PREFERENCE"
    assert handoff["all_gates_pass"] is True
    assert handoff["observed_decision_paths"] == {
        "core_paired_donor": True,
        "stronger_downstream_preference": True,
    }
    assert handoff["target_cohort_identity"][
        "ordered_partner_margin_levels_sha256"
    ]
    target_record = json.loads(package["artifact"].read_text())
    assert target_record["holdout_scope_disclosure"] == {
        "holdout_scope": "masked_acceptor_endpoint_outcomes_on_exact_R1c_target",
        "historical_chr8_chr9_sequence_materialization_disclosed": True,
        "globally_sequence_unseen": False,
        "known_prior_masked_acceptor_endpoint_scores_on_target_rows": 0,
        "licensed_term": "endpoint-outcome-held-out",
        "forbidden_terms": [
            "globally unseen",
            "sequence unseen",
            "virgin chromosome split",
        ],
    }


def test_model_free_preflight_package_verifier_returns_400_row_val_handoff(
    tmp_path: Path,
) -> None:
    package = _build_valid_target_package(tmp_path)
    verified = r1c.verify_successor_preflight_package(
        package["preflight"], package["preflight"].with_suffix(".manifest.json")
    )
    record = verified["record"]
    attestation = verified["attestation"]
    assert record["stage"] == "preflight"
    assert record["split"] == "val"
    assert len(record["rows"]) == r1c.TARGET_N
    assert {row["chrom"] for row in record["rows"]} <= set(
        r1c.VALIDATION_CHROMOSOMES
    )
    assert attestation["schema"] == "gfmcd.r1c.verified_preflight_handoff/1"
    assert attestation["verdict"] == "PREFLIGHT_GO"
    assert attestation["target_cohort_identity"][
        "ordered_partner_margin_levels_sha256"
    ]


def _rebind_tampered_target_publication(
    package: dict[str, Path], tamper: str
) -> None:
    artifact = json.loads(package["artifact"].read_text())
    if tamper == "margin":
        artifact["rows"][0]["margin_levels"]["P_clean"] += 0.01
    elif tamper == "robustness":
        artifact["gene_robustness"]["endpoints"]["Y_PD"]["raw"][
            "estimate"
        ] += 0.01
    elif tamper == "verdict":
        artifact["verdict"] = "NO_CONFIRMED_PAIRED_DONOR_PATH"
    elif tamper == "preflight":
        artifact["preflight_package"]["artifact"]["sha256"] = "0" * 64
    elif tamper == "holdout_scope":
        artifact["holdout_scope_disclosure"]["globally_sequence_unseen"] = True
    else:  # pragma: no cover - guarded by the parametrized test
        raise ValueError(tamper)
    _write_json(package["artifact"], artifact)

    manifest = json.loads(package["manifest"].read_text())
    artifact_identity = _package_identity(package["artifact"])
    manifest["outputs"] = [
        {
            "path": f"/relocated/{package['artifact'].name}",
            "size_bytes": artifact_identity["size_bytes"],
            "sha256": artifact_identity["sha256"],
        }
    ]
    _write_json(package["manifest"], manifest)

    completion = json.loads(package["completion"].read_text())
    completion["artifact"] = artifact_identity
    completion["manifest"] = _package_identity(package["manifest"])
    if tamper == "verdict":
        completion["official_verdict"] = artifact["verdict"]
    completion_without_id = {
        key: value for key, value in completion.items() if key != "completion_id"
    }
    completion["completion_id"] = r1c.canonical_sha256(completion_without_id)
    _write_json(package["completion"], completion)


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("margin", "primitive cells do not reconstruct"),
        ("robustness", "robustness does not reproduce"),
        ("verdict", "verdict, gates, or completion is stale"),
        ("preflight", "reviewed preflight link differs"),
        ("holdout_scope", "target holdout scope disclosure differs"),
    ],
)
def test_model_free_target_verifier_rejects_semantically_repackaged_tampering(
    tmp_path: Path, tamper: str, match: str
) -> None:
    package = _build_valid_target_package(tmp_path)
    _rebind_tampered_target_publication(package, tamper)
    with pytest.raises(r1c.TargetSealError, match=match):
        r1c.verify_successor_target_package(
            package["artifact"],
            package["manifest"],
            package["attempt"],
            package["completion"],
            preflight_artifact_path=package["preflight"],
        )


def test_model_free_target_verifier_rejects_receipt_and_completion_tampering(
    tmp_path: Path,
) -> None:
    package = _build_valid_target_package(tmp_path)
    attempt = json.loads(package["attempt"].read_text())
    attempt["model_loaded_before_record"] = True
    _write_json(package["attempt"], attempt)
    with pytest.raises(r1c.TargetSealError, match="attempt receipt does not replay"):
        r1c.verify_successor_target_package(
            package["artifact"],
            package["manifest"],
            package["attempt"],
            package["completion"],
            preflight_artifact_path=package["preflight"],
        )

    second = tmp_path / "second"
    second.mkdir()
    package = _build_valid_target_package(second)
    completion = json.loads(package["completion"].read_text())
    completion["official_verdict"] = "STALE"
    _write_json(package["completion"], completion)
    with pytest.raises(
        r1c.TargetSealError, match="completion does not authenticate"
    ):
        r1c.verify_successor_target_package(
            package["artifact"],
            package["manifest"],
            package["attempt"],
            package["completion"],
            preflight_artifact_path=package["preflight"],
        )


def test_successor_source_freeze_authorizes_only_a_fresh_preflight() -> None:
    assert r1c.HISTORICAL_ENGINEERING_PREFLIGHT_ARTIFACT_SHA256 == (
        "50cda24f0effcb8d9ce0ec9c766d3c6c20306ee8c07be27c182894e829fb6b6a"
    )
    assert r1c.FROZEN_PROTOCOL_SHA256 == (
        "2a2ac9dec6fb52b551cd87f96f7e3f07d6bc29431539011ebfaca2ce8c08da76"
    )
    assert r1c.FROZEN_INSTRUMENT_FINGERPRINT_SHA256 == (
        "c0ecd2be972b953c216ca3075530d520ea9cf448a07c385c40d1f1984a1b6dcb"
    )
    assert r1c.FROZEN_PREFLIGHT_ARTIFACT_SHA256 is None
    assert r1c.FROZEN_R5_STACK_FINGERPRINT_SHA256 == (
        "b89efd7c0d83d579aa7e0a2d36f8059c96913242a1c943ec64d87fe7590b00bc"
    )
    protocol_text = r1c.PROTOCOL.read_text()
    assert "> **Status: FROZEN FOR PREFLIGHT ONLY" in protocol_text
    authorized = r1c.authorize_production_stage(
        "preflight",
        protocol_sha256=r1c.sha256_file(r1c.PROTOCOL),
        protocol_text=protocol_text,
    )
    assert authorized["stage"] == "preflight"
    with pytest.raises(r1c.TargetSealError, match="preflight artifact"):
        r1c.authorize_production_stage(
            "target",
            protocol_sha256=r1c.sha256_file(r1c.PROTOCOL),
            protocol_text=protocol_text,
            acknowledgement=r1c.UNSEAL_ACKNOWLEDGEMENT,
        )


def test_protocol_freeze_order_does_not_require_future_package_pins() -> None:
    protocol = " ".join(r1c.PROTOCOL.read_text().split())
    assert "normalized full R1c→R5 source/method stack pin" in protocol
    assert "cannot depend on output-derived package pins whose bytes do not yet exist" in protocol
    assert "later R5a/R5b/R5c package pins are installed only after their producing stages" in protocol


def test_cli_target_stops_before_touching_missing_model_or_genome(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--stage",
            "target",
            "--gtf",
            str(tmp_path / "must-not-read.gtf"),
            "--fasta",
            str(tmp_path / "must-not-read.fa"),
            "--model",
            str(tmp_path / "must-not-read-model"),
            "--out",
            str(tmp_path / "must-not-write.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "target requires a reviewed successor preflight artifact pin" in completed.stderr
    assert "does not exist" not in completed.stderr
    assert not (tmp_path / "must-not-write.json").exists()


def test_preflight_rejects_parameter_drift_before_reading_inputs(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--stage",
            "preflight",
            "--n",
            str(r1c.TARGET_N - 1),
            "--gtf",
            str(tmp_path / "must-not-read.gtf"),
            "--fasta",
            str(tmp_path / "must-not-read.fa"),
            "--model",
            str(tmp_path / "must-not-read-model"),
            "--out",
            str(tmp_path / "must-not-write.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "production parameters are frozen" in completed.stderr
    assert "does not exist" not in completed.stderr


def test_frozen_preflight_stops_before_inputs_when_worktree_is_not_clean(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("do not overwrite\n")
    sentinel = REPO / "R1C_DIRTY_WORKTREE_TEST_SENTINEL"
    assert not sentinel.exists()
    sentinel.write_text("untracked on purpose\n")
    environment = os.environ.copy()
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--stage",
                "preflight",
                "--gtf",
                str(tmp_path / "must-not-read.gtf"),
                "--fasta",
                str(tmp_path / "must-not-read.fa"),
                "--model",
                str(tmp_path / "must-not-read-model"),
                "--out",
                str(output),
            ],
            capture_output=True,
            text=True,
            env=environment,
        )
    finally:
        sentinel.unlink(missing_ok=True)
    assert completed.returncode != 0
    assert "clean Git worktree" in completed.stderr
    assert output.read_text() == "do not overwrite\n"


def test_frozen_preflight_rejects_instrument_drift_before_external_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    touched: list[str] = []

    class _StagedCheckpoint:
        def cleanup(self) -> None:
            touched.append("checkpoint-cleanup")

    def inspect_output(path: Path) -> Path:
        touched.append("output")
        return path

    def stage_checkpoint(*args: object, **kwargs: object) -> tuple[object, Path, dict]:
        touched.append("model")
        return _StagedCheckpoint(), tmp_path / "staged-model", {
            "bundle_sha256": "checkpoint"
        }

    def construct_cohort(*args: object, **kwargs: object) -> list[object]:
        touched.append("cohort")
        raise AssertionError("cohort construction must follow frozen binding verification")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER),
            "--stage",
            "preflight",
            "--out",
            str(tmp_path / "preflight.json"),
        ],
    )
    monkeypatch.setattr(r1c, "require_clean_production_worktree", lambda *a, **k: "commit")
    monkeypatch.setattr(
        r1c,
        "downstream_r5_stack_identity",
        lambda **kwargs: {
            r1c.STACK_BINDING_FIELD: {},
            r1c.STACK_FINGERPRINT_FIELD: r1c.FROZEN_R5_STACK_FINGERPRINT_SHA256,
        },
    )
    monkeypatch.setattr(r1c, "verify_frozen_r5_stack", lambda stack: stack)
    monkeypatch.setattr(r1c, "pick_device", lambda requested: "mps")
    monkeypatch.setattr(
        r1c,
        "authorize_production_runtime",
        lambda device: {"device": device, "verified": True},
    )
    monkeypatch.setattr(r1c, "guard_output_path", inspect_output)
    monkeypatch.setattr(
        r1c,
        "checkpoint_files",
        lambda model: (tmp_path / "model", tmp_path / "model", []),
    )
    monkeypatch.setattr(r1c, "snapshot_input_paths", lambda paths: {})
    monkeypatch.setattr(
        r1c,
        "instrument_fingerprint",
        lambda *args, **kwargs: {
            "fingerprint_sha256": "source",
            "checkpoint": {"bundle_sha256": "checkpoint"},
        },
    )
    monkeypatch.setattr(
        r1c,
        "bind_runtime_contract",
        lambda fingerprint, runtime: {
            **fingerprint,
            "fingerprint_sha256": "0" * 64,
            "runtime_contract": runtime,
        },
    )
    monkeypatch.setattr(
        r1c,
        "snapshot_record",
        lambda snapshot, path: {
            "sha256": r1c.FROZEN_PROTOCOL_SHA256,
            "size_bytes": 1,
        },
    )
    monkeypatch.setattr(
        r1c, "read_snapshotted_bytes", lambda path, snapshot: r1c.PROTOCOL.read_bytes()
    )
    monkeypatch.setattr(r1c, "verify_checkpoint_inventory", lambda *a, **k: None)
    monkeypatch.setattr(r1c, "verify_input_snapshot", lambda snapshot: None)
    monkeypatch.setattr(r1c, "stage_checkpoint_bundle", stage_checkpoint)
    monkeypatch.setattr(r1c, "load_examples_unscored", construct_cohort)
    monkeypatch.setattr(
        r1c,
        "load_adapter",
        lambda *args, **kwargs: touched.append("model-adapter"),
    )

    with pytest.raises(SystemExit, match="fingerprint differs from the reviewed pin"):
        r1c.main()
    assert touched == []


def test_target_requires_both_frozen_hash_and_exact_acknowledgement() -> None:
    frozen_header = "# R1c\n\n> **Status: FROZEN 2026-08-12.**"
    with pytest.raises(r1c.TargetSealError, match="does not declare FROZEN"):
        r1c.authorize_target(
            "draft",
            r1c.UNSEAL_ACKNOWLEDGEMENT,
            frozen_protocol_sha256="draft",
            protocol_text="> **Status: DRAFT — NOT FROZEN.**",
        )
    with pytest.raises(r1c.TargetSealError, match="do not match"):
        r1c.authorize_target(
            "actual",
            r1c.UNSEAL_ACKNOWLEDGEMENT,
            frozen_protocol_sha256="frozen",
            protocol_text=frozen_header,
        )
    with pytest.raises(r1c.TargetSealError, match="exact"):
        r1c.authorize_target(
            "frozen", "yes", frozen_protocol_sha256="frozen", protocol_text=frozen_header
        )
    r1c.authorize_target(
        "frozen",
        r1c.UNSEAL_ACKNOWLEDGEMENT,
        frozen_protocol_sha256="frozen",
        protocol_text=frozen_header,
    )


def test_target_requires_exact_frozen_instrument_fingerprint() -> None:
    with pytest.raises(r1c.TargetSealError, match="reviewed instrument"):
        r1c.verify_frozen_instrument("observed", frozen_fingerprint_sha256=None)
    with pytest.raises(r1c.TargetSealError, match="differs"):
        r1c.verify_frozen_instrument(
            "observed", frozen_fingerprint_sha256="frozen"
        )
    r1c.verify_frozen_instrument("same", frozen_fingerprint_sha256="same")


def test_target_requires_literal_reviewed_preflight_artifact_hash() -> None:
    observed = "a" * 64
    with pytest.raises(r1c.TargetSealError, match="pinned in the runner"):
        r1c.verify_pinned_preflight_artifact(
            observed, frozen_artifact_sha256=None
        )
    with pytest.raises(r1c.TargetSealError, match="differ"):
        r1c.verify_pinned_preflight_artifact(
            observed, frozen_artifact_sha256="b" * 64
        )
    r1c.verify_pinned_preflight_artifact(
        observed, frozen_artifact_sha256=observed
    )


def test_stage_split_and_chromosome_guards_keep_chr8_9_out_of_preflight() -> None:
    assert r1c.stage_split("preflight") == "val"
    assert r1c.stage_split("target") == "test"
    r1c.assert_stage_chromosomes([_example(chrom="chr11")], "preflight")
    with pytest.raises(r1c.TargetSealError, match="sealed"):
        r1c.assert_stage_chromosomes([_example(chrom="chr8")], "preflight")
    r1c.assert_stage_chromosomes([_example(chrom="chr8"), _example(1, "chr9")], "target")
    with pytest.raises(r1c.TargetSealError, match="only sealed"):
        r1c.assert_stage_chromosomes([_example(chrom="chr8"), _example(1, "chr11")], "target")


def test_outcome_blind_cohort_has_both_acceptors_donor_edits_and_matched_dyads() -> None:
    records, audit = r1c.prepare_cohort([_example(i) for i in range(30)], 30)
    assert len(records) == 30
    assert audit["selected_genes"] == 30
    assert audit["selection_reads_model_outputs"] is False
    first = records[0]
    assert first.example.sequence[first.example.acceptor_pos - 2 : first.example.acceptor_pos] == "AG"
    assert first.example.sequence[first.matched_decoy_donor_pos : first.matched_decoy_donor_pos + 2] == "GT"
    assert first.dyad_distance_mismatch == 10
    assert audit["dyad_distance_caliper"] == 20
    assert audit["distance_matched_2x2_eligible"] == 30


def test_target_cohort_identity_binds_all_six_fields_row_order_and_gene_views() -> None:
    rows = _valid_preflight_record()["rows"][:3]
    rows[2]["gene_id"] = rows[0]["gene_id"]
    identity = r1c.target_cohort_identity(rows)
    six_position_rows = [
        {
            "example_id": row["example_id"],
            "gene_id": row["gene_id"],
            "chrom": row["chrom"],
            "strand": row["strand"],
            "positions": {
                name: row["positions"][name]
                for name in r1c.R1C_ROW_POSITION_FIELDS
            },
        }
        for row in rows
    ]
    assert identity["ordered_six_position_rows_sha256"] == r1c.canonical_sha256(
        six_position_rows
    )
    assert identity["ordered_example_ids_sha256"] == r1c.canonical_sha256(
        [row["example_id"] for row in rows]
    )
    assert identity["ordered_gene_ids_sha256"] == r1c.canonical_sha256(
        [row["gene_id"] for row in rows]
    )
    assert identity["target_gene_ids_sha256"] == r1c.canonical_sha256(
        sorted({row["gene_id"] for row in rows})
    )
    expected_construction = {
        "construction_version": r1c.CONSTRUCTION_VERSION,
        "cohort_order_version": r1c.COHORT_ORDER_VERSION,
        "ordered_six_position_rows": six_position_rows,
    }
    assert identity["construction_sha256"] == r1c.canonical_sha256(
        expected_construction
    )


def test_dyad_caliper_is_applied_before_fixed_n_selection_and_never_relaxed() -> None:
    outside = CouplingExample(
        **{**_example(0).__dict__, "exon_length": 50, "intron_length_down": 71}
    )
    inside = [_example(i + 1) for i in range(30)]
    records, audit = r1c.prepare_cohort([outside, *inside], 30)
    assert all(record.example.example_id != outside.example_id for record in records)
    assert audit["primary_comparator_complete"] == 31
    assert audit["distance_matched_2x2_eligible"] == 30
    assert audit["exclusions"]["exon_intron_dyad_outside_distance_caliper"] == 1

    with pytest.raises(RuntimeError, match="fixed 20-nt dyad caliper"):
        r1c.prepare_cohort([outside, *inside[:-1]], 30)


def test_cohort_enforces_30_independent_genes_not_30_rows() -> None:
    examples = [_example(i) for i in range(30)]
    examples = [
        CouplingExample(**{**example.__dict__, "gene_id": "one-gene"}) for example in examples
    ]
    with pytest.raises(RuntimeError, match="independent genes"):
        r1c.prepare_cohort(examples, 30)


def test_exact_2x2_algebra_and_ordered_primary_family() -> None:
    Y_PD = np.asarray([1.0, 2.0])
    Y_PN = np.asarray([0.1, 0.2])
    Y_XD = np.asarray([0.3, 0.5])
    Y_XN = np.asarray([-0.1, 0.1])
    family = r1c._primary_family_matrix(Y_PD, Y_PN, Y_XD, Y_XN)
    assert family.shape == (len(r1c.PRIMARY_FAMILY_NAMES), 2)
    assert r1c.PRIMARY_FAMILY_NAMES == ("Y_PD", "Y_PN", "S_P", "I")
    np.testing.assert_array_equal(family[0], Y_PD)
    np.testing.assert_array_equal(family[1], Y_PN)
    np.testing.assert_array_equal(family[2], Y_PD - Y_PN)
    np.testing.assert_array_equal(
        family[3], (Y_PD - Y_PN) - (Y_XD - Y_XN)
    )


def test_raw_and_trimmed_equal_gene_bootstrap_is_a_necessary_10k_robustness_gate() -> None:
    groups = [f"gene-{index}" for index in range(80)]
    stable_positive = np.linspace(0.11, 0.19, len(groups))
    stable = np.vstack(
        [stable_positive, np.zeros(len(groups)), stable_positive, stable_positive]
    )
    supported = r1c.r1c_gene_robustness_gate(
        stable, groups, n_draws=r1c.PRODUCTION_ROBUST_BOOTSTRAP_DRAWS
    )
    assert supported["n_draws"] >= 10_000
    assert supported["trim_fraction"] == 0.10
    assert supported["paths"]["core_paired_donor"]["pass"] is True
    assert supported["paths"]["stronger_downstream_preference"]["pass"] is True
    assert supported["all_required_robust"] is True

    tail_driven = np.full(len(groups), 0.03)
    tail_driven[-1] = 8.0
    fragile = np.vstack(
        [tail_driven, np.zeros(len(groups)), tail_driven, tail_driven]
    )
    unresolved = r1c.r1c_gene_robustness_gate(
        fragile, groups, n_draws=r1c.PRODUCTION_ROBUST_BOOTSTRAP_DRAWS
    )
    assert unresolved["endpoints"]["Y_PD"]["raw"]["estimate"] > r1c.SESOI
    assert unresolved["endpoints"]["Y_PD"]["trimmed"]["estimate"] == pytest.approx(0.03)
    assert unresolved["all_required_robust"] is False
    assert unresolved["failure_disposition"] == (
        "SCIENTIFIC_UNRESOLVED_TAIL_SENSITIVITY_NOT_INSTRUMENT_FAILURE"
    )


def test_structured_plant_is_coherent_and_frozen_at_point_four() -> None:
    np.testing.assert_array_equal(
        r1c.cell_plant_vector(), np.asarray([0.40, 0.0, 0.0, 0.0])
    )
    plant = r1c.structured_plant_vector()
    np.testing.assert_array_equal(plant, np.asarray([0.40, 0.0, 0.40, 0.40]))
    assert r1c.PLANT_DELTA == 0.40
    assert r1c.PRIMARY_PLANT_REQUIREMENTS[1] == "EQUIVALENT_WITHIN_SESOI"


def test_centered_cell_residuals_are_translation_invariant_and_preserve_2x2() -> None:
    cells = np.asarray(
        [
            [1.0, 1.2, 0.8, 1.1],
            [0.0, 0.1, -0.1, 0.0],
            [0.3, 0.4, 0.2, 0.5],
            [0.0, -0.1, 0.1, 0.0],
        ]
    )
    translations = np.asarray([10.0, -4.0, 2.5, 9.0])[:, None]
    residuals = r1c.endpoint_centered_cell_residuals(cells)
    shifted = r1c.endpoint_centered_cell_residuals(cells + translations)
    np.testing.assert_allclose(residuals, shifted, atol=1e-14)

    groups = ["gene-a", "gene-a", "gene-b", "gene-c"]
    flipped = r1c.gene_sign_flip(
        residuals, groups, seed=7, namespace="r1c-test-whole-gene"
    )
    observed_signs = []
    for column in range(residuals.shape[1]):
        nonzero = np.abs(residuals[:, column]) > 1e-12
        ratios = np.unique(flipped[nonzero, column] / residuals[nonzero, column])
        assert ratios.size == 1
        observed_signs.append(float(ratios[0]))
    assert observed_signs[0] == observed_signs[1]
    derived = r1c._family_from_cells(flipped)
    np.testing.assert_allclose(
        derived[3],
        (flipped[0] - flipped[1]) - (flipped[2] - flipped[3]),
        atol=1e-14,
    )


def test_boundary_classifier_uses_tost_iut_holm_and_serializes_replay_metadata() -> None:
    groups = [f"gene-{index}" for index in range(40)]
    family = np.tile(np.asarray([[0.40], [0.0], [0.40], [0.40]]), (1, len(groups)))
    inference = r1c.r1c_boundary_classifier(
        family,
        groups,
        sesoi=r1c.SESOI,
        n_randomisation=63,
        alpha=r1c.FAMILY_ALPHA,
        seed=7,
        namespace="r1c-test-boundary",
        min_genes=30,
        studentized=True,
    )

    assert inference["Y_PN_TOST"]["p_value"] == max(
        inference["Y_PN_TOST"]["lower_edge_test"]["p_value"],
        inference["Y_PN_TOST"]["upper_edge_test"]["p_value"],
    )
    assert inference["holm_adjusted_path_p_values"] == pytest.approx(
        {
            "core_paired_donor": 2 / 64,
            "stronger_downstream_preference": 2 / 64,
        }
    )
    assert r1c.observed_decision_paths(inference) == {
        "core_paired_donor": True,
        "stronger_downstream_preference": True,
    }
    assert inference["shared_boundary_randomisation"][
        "all_five_boundary_tests_share_sign_matrix"
    ] is True
    assert len(
        {
            test["sign_matrix_sha256"]
            for test in inference["boundary_tests"].values()
        }
    ) == 1
    assert len(
        {
            test["ordered_gene_labels_sha256"]
            for test in inference["boundary_tests"].values()
        }
    ) == 1
    for test in inference["boundary_tests"].values():
        assert test["p_value"] == pytest.approx(
            (1 + test["exceedances"]) / (test["B"] + 1)
        )
        assert test["namespace"].startswith("r1c-test-boundary:")
        for digest in (
            "null_draws_sha256",
            "shared_null_draws_sha256",
            "sign_matrix_sha256",
            "ordered_gene_labels_sha256",
        ):
            assert len(test[digest]) == 64


def test_strict_ci_boundary_touch_and_p_ci_disagreement_are_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = [f"gene-{index}" for index in range(40)]
    touching = np.tile(
        np.asarray([[r1c.SESOI], [0.0], [r1c.SESOI], [r1c.SESOI]]),
        (1, len(groups)),
    )
    touching_inference = r1c.r1c_boundary_classifier(
        touching,
        groups,
        sesoi=r1c.SESOI,
        n_randomisation=63,
        alpha=r1c.FAMILY_ALPHA,
        seed=3,
        namespace="r1c-touching",
        min_genes=30,
        studentized=True,
    )
    assert touching_inference["family_decisions"]["Y_PD"]["classification"] == (
        "UNRESOLVED"
    )
    assert not any(r1c.observed_decision_paths(touching_inference).values())
    assert touching_inference["path_tests"]["core_paired_donor"]["decision"] == (
        "NOT_DECLARED"
    )
    assert (
        r1c.production_verdict(touching_inference, {"all": True}, "target")
        == "NO_CONFIRMED_PAIRED_DONOR_PATH"
    )

    def fake_intervals(values, groups, names, **kwargs):
        estimates = np.asarray(values).mean(axis=1)
        intervals = {
            "Y_PD": (0.04, 0.50),  # Deliberately fails the strict CI gate.
            "Y_PN": (-0.01, 0.01),
            "S_P": (0.20, 0.50),
            "I": (0.20, 0.50),
        }
        decisions = {
            name: FamilyDecision(
                estimate=float(estimates[index]),
                simultaneous_ci=intervals[name],
                max_abs_p=1.0,
                classification="UNRESOLVED",
            )
            for index, name in enumerate(names)
        }
        widths = {
            name: max(
                decisions[name].estimate - intervals[name][0],
                intervals[name][1] - decisions[name].estimate,
            )
            for name in names
        }
        return decisions, {
            "studentized": True,
            "method": "fixture",
            "interval_scaling": "fixture",
            "simultaneous_half_width": max(widths.values()),
            "simultaneous_half_widths": widths,
            "resolution_ok": False,
        }

    monkeypatch.setattr(r1c, "simultaneous_family_decision", fake_intervals)
    strong = np.tile(np.asarray([[0.40], [0.0], [0.40], [0.40]]), (1, len(groups)))
    disagreement = r1c.r1c_boundary_classifier(
        strong,
        groups,
        sesoi=r1c.SESOI,
        n_randomisation=63,
        alpha=r1c.FAMILY_ALPHA,
        seed=4,
        namespace="r1c-disagreement",
        min_genes=30,
        studentized=True,
    )
    for path in r1c.PATH_NAMES:
        assert disagreement["path_tests"][path]["p_rejects"] is True
        assert disagreement["path_tests"][path]["strict_ci_passes"] is False
        assert disagreement["path_tests"][path]["p_ci_disagreement"] is True
        assert disagreement["path_tests"][path]["decision"] == "UNRESOLVED"
    assert (
        r1c.production_verdict(disagreement, {"all": True}, "target")
        == "PAIRED_DONOR_PATH_UNRESOLVED"
    )


def test_least_favourable_cases_are_primitive_2x2_boundary_surfaces() -> None:
    cases = r1c.least_favourable_boundary_cases()
    assert set(cases) == {
        "Y_PD_at_positive_boundary",
        "Y_PN_at_lower_equivalence_boundary",
        "Y_PN_at_upper_equivalence_boundary",
        "S_P_at_positive_boundary",
        "I_at_positive_boundary",
    }
    expected_boundary = {
        "Y_PD_at_positive_boundary": (0, r1c.SESOI),
        "Y_PN_at_lower_equivalence_boundary": (1, -r1c.SESOI),
        "Y_PN_at_upper_equivalence_boundary": (1, r1c.SESOI),
        "S_P_at_positive_boundary": (2, r1c.SESOI),
        "I_at_positive_boundary": (3, r1c.SESOI),
    }
    for name, case in cases.items():
        derived = r1c._family_from_cells(
            np.asarray(case["cell_vector"], dtype=float)[:, None]
        )[:, 0]
        np.testing.assert_allclose(derived, case["family_vector"], atol=1e-14)
        index, boundary = expected_boundary[name]
        assert derived[index] == pytest.approx(boundary)
    assert cases["I_at_positive_boundary"]["null_paths"] == [
        "stronger_downstream_preference"
    ]


def test_nuisance_continuum_is_controlled_by_event_inclusion_with_coherent_surfaces() -> None:
    control = r1c.nuisance_continuum_control()
    assert control["iut_property"] == (
        "the raw path p-value is the maximum constituent p-value"
    )
    assert "subseteq" in control["null_event_inclusion"]
    surfaces = control["representative_classifier_surfaces"]
    cases = r1c.least_favourable_boundary_cases()
    for face, rows in surfaces.items():
        assert len(rows) == 3
        assert sum(row["executed_in_outer_trials"] for row in rows) == 1
        for row in rows:
            derived = r1c._family_from_cells(
                np.asarray(row["cell_vector"], dtype=float)[:, None]
            )[:, 0]
            np.testing.assert_allclose(derived, row["family_vector"], atol=1e-14)
            assert -r1c.SESOI < derived[1] < r1c.SESOI
            assert derived[3] > r1c.SESOI
            boundary_index = 0 if row["boundary_constituent"] == "Y_PD" else 2
            assert derived[boundary_index] == pytest.approx(r1c.SESOI)
        executed_case = control["executed_surface_cases"][face]
        np.testing.assert_allclose(
            rows[1]["family_vector"], cases[executed_case]["family_vector"]
        )


def test_preflight_routes_every_lfc_and_plant_through_exact_classifier() -> None:
    groups = [f"gene-{index}" for index in range(30)]
    result = r1c.r1c_decision_path_preflight(
        np.zeros((4, len(groups))),
        groups,
        sesoi=r1c.SESOI,
        n_calibration=63,
        n_trials=2,
        alpha=r1c.FAMILY_ALPHA,
        seed=11,
        namespace="r1c-calibration-test",
        min_genes=30,
        power_floor=r1c.PREFLIGHT_POWER_FLOOR,
        studentized=True,
    )
    assert result["production_classifier_calls"] == 2 * (
        len(r1c.least_favourable_boundary_cases()) + 1
    )
    assert result["plant_successes"] == {
        "core_paired_donor": 2,
        "stronger_downstream_preference": 2,
    }
    assert all(
        count == 0
        for path_counts in result["boundary_false_positive_counts"].values()
        for count in path_counts.values()
    )
    assert result["trial_records_sha256"] == r1c.canonical_sha256(
        result["trial_records"]
    )
    first = result["trial_records"][0]
    assert first["outer_sign_namespace"] == (
        "r1c-calibration-test:centered-primitive-null:0"
    )
    for classifier in [
        *first["least_favourable_cases"].values(),
        first["coherent_plant"],
    ]:
        assert set(classifier["holm_adjusted_path_p_values"]) == set(r1c.PATH_NAMES)
        for test in classifier["boundary_tests"].values():
            assert "exceedances" in test and "namespace" in test
            assert len(test["sign_matrix_sha256"]) == 64


def test_measure_behavior_uses_the_same_D_and_N_edits_at_both_acceptor_readouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, _ = r1c.prepare_cohort([_example(i) for i in range(30)], 30)
    calls: list[tuple[list[int], list[str]]] = []

    def fake_margins(_task, positions, sequences):
        calls.append((list(positions), list(sequences)))
        return np.full(len(positions), 10.0 - len(calls), dtype=float)

    monkeypatch.setattr(r1c, "margins_at", fake_margins)
    family, levels, descriptive = r1c.measure_behavior(object(), records)
    assert len(calls) == 6
    assert calls[1][1] == calls[4][1]  # D is identical at P and X.
    assert calls[2][1] == calls[5][1]  # N is identical at P and X.
    donor_positions = [record.example.donor_pos for record in records]
    assert all(positions != donor_positions for positions, _ in calls)
    assert set(levels) == {"P_clean", "P_under_D", "P_under_N", "X_clean", "X_under_D", "X_under_N"}
    assert descriptive["cell_values"].shape == (4, 30)
    assert family.shape == (4, 30)

    source = RUNNER.read_text().split("def measure_behavior(", 1)[1].split(
        "def _health_positions", 1
    )[0]
    assert "donor_readouts" not in source
    assert "FORWARD_ACCEPTOR_ALT" not in source


def test_exact_fingerprint_hashes_every_checkpoint_file_and_all_bound_sources(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("config")
    (model / "weights.bin").write_bytes(b"weights")
    gtf = tmp_path / "genes.gtf"
    fasta = tmp_path / "genome.fa"
    fasta_fai = tmp_path / "genome.fa.fai"
    protocol = tmp_path / "protocol.md"
    runner = tmp_path / "runner.py"
    stats = tmp_path / "stats.py"
    for path, text in (
        (gtf, "gtf"),
        (fasta, "fasta"),
        (fasta_fai, "fai-v1"),
        (protocol, "protocol"),
        (runner, "runner"),
        (stats, "stats"),
    ):
        path.write_text(text)

    first = r1c.instrument_fingerprint(
        model, gtf, fasta, protocol=protocol, runner=runner, stats=stats
    )
    assert first["checkpoint"]["n_files"] == 2
    assert {item["relative_path"] for item in first["checkpoint"]["files"]} == {
        "config.json",
        "weights.bin",
    }
    assert first["fasta_fai"]["sha256"] == r1c.sha256_file(fasta_fai)
    fasta_fai.write_text("fai-v2")
    fai_changed = r1c.instrument_fingerprint(
        model, gtf, fasta, protocol=protocol, runner=runner, stats=stats
    )
    assert first["fingerprint_sha256"] != fai_changed["fingerprint_sha256"]
    (model / "weights.bin").write_bytes(b"changed")
    second = r1c.instrument_fingerprint(
        model, gtf, fasta, protocol=protocol, runner=runner, stats=stats
    )
    assert first["fingerprint_sha256"] != second["fingerprint_sha256"]


def test_runtime_choices_are_part_of_the_preflight_target_fingerprint() -> None:
    base = {"schema": "fixture", "fingerprint_sha256": "source"}
    first = r1c.bind_runtime_contract(base, {"n_randomisation": 4000, "seed": 0})
    second = r1c.bind_runtime_contract(base, {"n_randomisation": 3999, "seed": 0})
    assert first["source_fingerprint_sha256"] == "source"
    assert first["fingerprint_sha256"] != second["fingerprint_sha256"]


def test_successor_production_runtime_is_strict_mps_without_fallback() -> None:
    with pytest.raises(r1c.TargetSealError, match="must be mps"):
        r1c.authorize_production_runtime("cpu", environ={})
    with pytest.raises(r1c.TargetSealError, match="must be absent"):
        r1c.authorize_production_runtime(
            "mps", environ={"PYTORCH_ENABLE_MPS_FALLBACK": "0"}
        )
    policy = r1c.authorize_production_runtime("mps", environ={})
    assert policy["verified"] is True
    assert policy["device"] == "mps"
    assert policy["absent_environment_variables"] == [
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ]


def test_successor_binds_one_exact_downstream_r5_stack_and_requires_its_pin() -> None:
    stack = r1c.downstream_r5_stack_identity(repo=REPO)
    assert set(stack) == {
        r1c.STACK_BINDING_FIELD,
        r1c.STACK_FINGERPRINT_FIELD,
    }
    assert stack[r1c.STACK_BINDING_FIELD]["schema"] == r1c.STACK_SCHEMA
    with pytest.raises(r1c.TargetSealError, match="downstream R5 stack"):
        r1c.verify_frozen_r5_stack(stack, frozen_fingerprint_sha256=None)
    with pytest.raises(r1c.TargetSealError, match="differs"):
        r1c.verify_frozen_r5_stack(
            stack, frozen_fingerprint_sha256="0" * 64
        )
    verified = r1c.verify_frozen_r5_stack(
        stack,
        frozen_fingerprint_sha256=stack[r1c.STACK_FINGERPRINT_FIELD],
    )
    assert verified[r1c.STACK_FINGERPRINT_FIELD] == stack[
        r1c.STACK_FINGERPRINT_FIELD
    ]


def test_preflight_and_target_remain_sealed_until_the_whole_successor_is_frozen() -> None:
    with pytest.raises(r1c.TargetSealError, match="does not declare FROZEN"):
        r1c.authorize_production_stage(
            "preflight", protocol_sha256="a" * 64, protocol_text="> **Status: DRAFT"
        )

    with pytest.raises(r1c.TargetSealError, match="source/method freeze is incomplete"):
        r1c.authorize_production_stage(
            "preflight",
            protocol_sha256="a" * 64,
            protocol_text="> **Status: FROZEN FOR PREFLIGHT ONLY",
            frozen_protocol_sha256=None,
            frozen_instrument_fingerprint_sha256=None,
            frozen_r5_stack_fingerprint_sha256=None,
        )

    frozen = {
        "frozen_protocol_sha256": "a" * 64,
        "frozen_instrument_fingerprint_sha256": "b" * 64,
        "frozen_r5_stack_fingerprint_sha256": "c" * 64,
    }
    header = "> **Status: FROZEN FOR SUCCESSOR PREFLIGHT.**"
    authorized = r1c.authorize_production_stage(
        "preflight",
        protocol_sha256="a" * 64,
        protocol_text=header,
        **frozen,
    )
    assert authorized["stage"] == "preflight"
    assert authorized["preflight_artifact_pin_required"] is False

    with pytest.raises(r1c.TargetSealError, match="preflight artifact"):
        r1c.authorize_production_stage(
            "target",
            protocol_sha256="a" * 64,
            protocol_text=header,
            acknowledgement=r1c.UNSEAL_ACKNOWLEDGEMENT,
            **frozen,
        )
    target = r1c.authorize_production_stage(
        "target",
        protocol_sha256="a" * 64,
        protocol_text=header,
        acknowledgement=r1c.UNSEAL_ACKNOWLEDGEMENT,
        frozen_preflight_artifact_sha256="d" * 64,
        **frozen,
    )
    assert target["stage"] == "target"
    assert target["explicit_unseal_acknowledgement"] is True


def test_both_production_stages_require_a_clean_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "r1c@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "R1c Test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("frozen\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=repo, check=True)

    preflight = r1c.require_clean_production_worktree("preflight", repo=repo)
    target = r1c.require_clean_production_worktree("target", repo=repo)
    assert preflight == target

    tracked.write_text("dirty\n")
    for stage in ("preflight", "target"):
        with pytest.raises(r1c.TargetSealError, match="clean Git worktree"):
            r1c.require_clean_production_worktree(stage, repo=repo)


def test_repository_sources_use_repo_relative_logical_identities() -> None:
    runner = r1c.file_identity(RUNNER)
    protocol = r1c.file_identity(r1c.PROTOCOL)
    assert runner["logical_name"] == (
        "scripts/splicebert_acceptor_coupling/run_r1c_sealed_reverse_behavior.py"
    )
    assert protocol["logical_name"] == (
        "studies/splicebert_acceptor_coupling/preregistrations/"
        "DRAFT_R1C_SEALED_REVERSE_BEHAVIOR_CONFIRMATION.md"
    )
    assert runner["location_scope"] == protocol["location_scope"] == "repository"
    assert "path" not in runner
    assert str(r1c.REPO) not in json.dumps({"runner": runner, "protocol": protocol})


def test_frozen_contract_filters_chromosomes_before_dataset_construction() -> None:
    contract = r1c.frozen_instrument_contract()
    assert r1c.CONSTRUCTION_VERSION.endswith("boundary-inference-v4")
    assert contract["construction_version"] == r1c.CONSTRUCTION_VERSION
    assert '"schema": "gfmcd.r1c.reverse_behavior/3"' in RUNNER.read_text()
    assert contract["validation_chromosomes"] == ["chr11", "chr16", "chr19"]
    assert contract["target_chromosomes"] == ["chr8", "chr9"]
    assert contract["chromosome_filter_stage"] == (
        "before_GTF_materialization_and_FASTA_fetch"
    )
    assert contract["cohort_scan"] == (
        "exhaustive_allowed_chromosomes_before_first_n_selection"
    )
    assert contract["cohort_order_version"] == "r1c-single-g-primary-family-v2"
    assert contract["production_randomisation"] == 4000
    assert contract["preflight_artifact_hash_pin"] == r1c.PREFLIGHT_HASH_PIN_CONTRACT
    assert contract["nuisance_continuum_control"] == r1c.nuisance_continuum_control()
    source = RUNNER.read_text().split("def main() -> None:", 1)[1]
    assert "split,\n        None,\n        tag=" in source
    assert 'tag=f"r1c-{args.stage}-{COHORT_ORDER_VERSION}"' in source


def test_runner_fingerprint_normalizes_only_self_referential_seal_literals(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text(
        "FROZEN_PROTOCOL_SHA256: str | None = None\n"
        "FROZEN_INSTRUMENT_FINGERPRINT_SHA256: str | None = None\n"
        "FROZEN_PREFLIGHT_ARTIFACT_SHA256: str | None = None\n"
        "FROZEN_R5_STACK_FINGERPRINT_SHA256: str | None = None\n"
        "LOGIC = 1\n"
    )
    first = r1c.normalized_runner_identity(runner)
    runner.write_text(
        'FROZEN_PROTOCOL_SHA256: str | None = "abc"\n'
        'FROZEN_INSTRUMENT_FINGERPRINT_SHA256: str | None = "def"\n'
        'FROZEN_PREFLIGHT_ARTIFACT_SHA256: str | None = "ghi"\n'
        'FROZEN_R5_STACK_FINGERPRINT_SHA256: str | None = "jkl"\n'
        "LOGIC = 1\n"
    )
    assert r1c.normalized_runner_identity(runner)["normalized_sha256"] == first[
        "normalized_sha256"
    ]
    runner.write_text(runner.read_text().replace("LOGIC = 1", "LOGIC = 2"))
    assert r1c.normalized_runner_identity(runner)["normalized_sha256"] != first[
        "normalized_sha256"
    ]


def test_production_seal_literals_remain_single_line_normalizable(tmp_path: Path) -> None:
    source = RUNNER.read_text()
    matches = re.findall(
        r"^FROZEN_[A-Z0-9_]+_SHA256:[^\n]+$",
        source,
        flags=re.MULTILINE,
    )
    assert len(matches) == 4
    normalized = r1c.normalized_runner_identity(RUNNER)
    rewritten = re.sub(
        r"^(FROZEN_[A-Z0-9_]+_SHA256):[^=]*=.*$",
        r'\1: str | None = "0"',
        source,
        flags=re.MULTILINE,
    )
    assert rewritten.count('str | None = "0"') == 4
    rewritten_path = tmp_path / "rewritten_runner.py"
    rewritten_path.write_text(rewritten)
    assert (
        r1c.normalized_runner_identity(rewritten_path)["normalized_sha256"]
        == normalized["normalized_sha256"]
    )


def test_target_preflight_binding_rejects_gate_or_hash_drift() -> None:
    record = _valid_preflight_record()
    r1c.verify_preflight_binding(record, "same")
    with pytest.raises(r1c.TargetSealError, match="fingerprint"):
        r1c.verify_preflight_binding(record, "different")
    with pytest.raises(r1c.TargetSealError, match="did not pass"):
        r1c.verify_preflight_binding({**record, "all_gates_pass": False}, "same")


@pytest.mark.parametrize("missing", ["rows", "health", "cohort_sha256"])
def test_target_preflight_binding_requires_rows_health_and_selected_cohort_hash(
    missing: str,
) -> None:
    record = _valid_preflight_record()
    record.pop(missing)
    with pytest.raises(r1c.TargetSealError, match="primitive rows|health|cohort SHA"):
        r1c.verify_preflight_binding(record, "same")


@pytest.mark.parametrize("tamper", ["estimate", "statistic", "digest"])
def test_target_replays_observed_estimate_statistic_and_digest_from_rows(
    tamper: str,
) -> None:
    record = _valid_preflight_record()
    inference = record["boundary_inference"]
    test = inference["boundary_tests"]["Y_PD_gt_SESOI"]
    if tamper == "estimate":
        inference["family_decisions"]["Y_PD"]["estimate"] += 0.01
        record["family_decisions"]["Y_PD"]["estimate"] += 0.01
    elif tamper == "statistic":
        test["oriented_statistic"] += 0.01
    else:
        test["null_draws_sha256"] = "f" * 64
    with pytest.raises(
        r1c.TargetSealError,
        match="do not reproduce from primitive rows|top-level family decisions",
    ):
        r1c.verify_preflight_binding(record, "same")


def test_target_rederives_calibration_trial_decision_and_aggregate_count() -> None:
    record = _valid_preflight_record()
    trial = record["production_path_preflight"]["trial_records"][0]
    trial["least_favourable_cases"]["Y_PD_at_positive_boundary"]["path_tests"][
        "core_paired_donor"
    ]["decision"] = "DECLARED"
    record["production_path_preflight"]["trial_records_sha256"] = r1c.canonical_sha256(
        record["production_path_preflight"]["trial_records"]
    )
    with pytest.raises(r1c.TargetSealError, match="path decision does not reproduce"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    calibration = record["production_path_preflight"]
    path = "core_paired_donor"
    case = "Y_PD_at_positive_boundary"
    calibration["boundary_false_positive_counts"][path][case] = 1
    calibration["worst_case_false_positives"][path] = 1
    calibration["path_boundary_fpr"][path] = 1 / r1c.PRODUCTION_TRIALS
    calibration["path_boundary_fpr_upper"][path] = r1c.one_sided_binomial_bounds(
        1, r1c.PRODUCTION_TRIALS, alpha=r1c.FAMILY_ALPHA
    )[1]
    calibration["path_boundary_fpr_ok"][path] = True
    with pytest.raises(r1c.TargetSealError, match="do not reproduce from trial decisions"):
        r1c.verify_preflight_binding(record, "same")


def test_target_preflight_binding_requires_exact_config_and_gate_family() -> None:
    record = _valid_preflight_record()
    record["config"]["batch"] -= 1
    with pytest.raises(r1c.TargetSealError, match="complete production config"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    record["gates"].pop("observed_Y_PN_tost_and_strict_ci")
    with pytest.raises(r1c.TargetSealError, match="gate family"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    record["gates"]["health_damage_bounded"] = False
    with pytest.raises(r1c.TargetSealError, match="required gates did not pass"):
        r1c.verify_preflight_binding(record, "same")


def test_target_preflight_binding_requires_exact_endpoint_scale_normalized_family() -> None:
    record = _valid_preflight_record()
    record["family_calibration"]["studentized"] = False
    with pytest.raises(r1c.TargetSealError, match="family calibration"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    record["family_calibration"]["interval_scaling"] = "nested bootstrap-t"
    with pytest.raises(r1c.TargetSealError, match="family calibration"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    record["production_path_preflight"]["alpha"] = 0.10
    with pytest.raises(r1c.TargetSealError, match="production-path calibration alpha"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    record["family_calibration"]["simultaneous_half_width"] = 0.01
    with pytest.raises(r1c.TargetSealError, match="family calibration|exact maximum width"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    record["family_decisions"].pop("I")
    with pytest.raises(r1c.TargetSealError, match="family decisions|four-row family"):
        r1c.verify_preflight_binding(record, "same")


def test_target_preflight_binding_recomputes_boundary_holm_and_shared_sign_metadata() -> None:
    record = _valid_preflight_record()
    record["boundary_inference"]["boundary_tests"]["Y_PD_gt_SESOI"][
        "exceedances"
    ] += 1
    with pytest.raises(r1c.TargetSealError, match="exact add-one"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    record["boundary_inference"]["path_tests"]["core_paired_donor"][
        "holm_adjusted_p_value"
    ] = 0.90
    with pytest.raises(r1c.TargetSealError, match="IUT, Holm"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    trial = record["production_path_preflight"]["trial_records"][0]
    trial["coherent_plant"]["boundary_tests"]["Y_PD_gt_SESOI"][
        "sign_matrix_sha256"
    ] = "f" * 64
    record["production_path_preflight"]["trial_records_sha256"] = r1c.canonical_sha256(
        record["production_path_preflight"]["trial_records"]
    )
    with pytest.raises(r1c.TargetSealError, match="shared signs"):
        r1c.verify_preflight_binding(record, "same")


def test_target_preflight_binding_requires_exact_plant_boundary_cases_and_holm_family() -> None:
    record = _valid_preflight_record()
    record["production_path_preflight"]["plant_vector"][0] = 0.0
    with pytest.raises(r1c.TargetSealError, match="plant_vector"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    record["production_path_preflight"]["least_favourable_boundary_cases"][
        "I_at_positive_boundary"
    ]["family_vector"][-1] = 0.0
    with pytest.raises(r1c.TargetSealError, match="least_favourable_boundary_cases"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    record["production_path_preflight"]["path_holm_family"].append("extra")
    with pytest.raises(r1c.TargetSealError, match="path_holm_family"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    record["structured_plant"]["requirements"]["I"] = "UNRESOLVED"
    with pytest.raises(r1c.TargetSealError, match="named structured plant"):
        r1c.verify_preflight_binding(record, "same")


def test_target_preflight_binding_recomputes_exact_fpr_and_power_bounds() -> None:
    record = _valid_preflight_record()
    record["production_path_preflight"]["path_boundary_fpr_upper"][
        "core_paired_donor"
    ] = r1c.FAMILY_ALPHA
    with pytest.raises(r1c.TargetSealError, match="path_boundary_fpr_upper"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    path = "stronger_downstream_preference"
    failing_successes = r1c.MIN_POWER_SUCCESSES - 1
    record["production_path_preflight"]["plant_successes"][path] = failing_successes
    record["production_path_preflight"]["path_planted_power"][path] = (
        failing_successes / r1c.PRODUCTION_TRIALS
    )
    record["production_path_preflight"]["path_planted_power_lower"][path] = (
        r1c.one_sided_binomial_bounds(
            failing_successes, r1c.PRODUCTION_TRIALS, alpha=0.05
        )[0]
    )
    record["production_path_preflight"]["path_planted_power_ok"][path] = False
    with pytest.raises(r1c.TargetSealError, match="planted successes"):
        r1c.verify_preflight_binding(record, "same")


def test_target_preflight_binding_requires_frozen_status_and_runtime_contract() -> None:
    record = _valid_preflight_record()
    record["protocol_status"] = "DRAFT_NOT_FROZEN"
    with pytest.raises(r1c.TargetSealError, match="protocol_status"):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    record["freeze_authorization"]["instrument_fingerprint_pinned"] = False
    record["freeze_authorization"]["authorization_ready"] = False
    with pytest.raises(
        r1c.TargetSealError,
        match="after every reviewed successor pin is installed",
    ):
        r1c.verify_preflight_binding(record, "same")

    record = _valid_preflight_record()
    record["instrument_fingerprint"]["runtime_contract"]["n_trials"] -= 1
    with pytest.raises(r1c.TargetSealError, match="production-parameter drift"):
        r1c.verify_preflight_binding(record, "same")


def test_target_preflight_binding_does_not_restore_a_global_width_gate() -> None:
    record = _valid_preflight_record()
    assert all("resolution" not in name for name in record["gates"])
    assert all("width" not in name for name in record["gates"])
    assert record["boundary_inference"]["family_calibration"]["resolution_ok"] is True
    r1c.verify_preflight_binding(record, "same")


def test_target_verifies_preflight_bytes_against_sibling_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "r1c_preflight.json"
    artifact.write_text('{"stage":"preflight"}\n')
    manifest = artifact.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "experiment_id": "splicebert_acceptor_coupling:r1c_reverse_behavior",
                "outputs": [
                    {
                        # A manifest created in another checkout remains valid after
                        # relocation; logical name plus exact bytes are authoritative.
                        "path": f"/old/checkout/results/{artifact.name}",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": r1c.file_identity(artifact)["sha256"],
                    }
                ],
            }
        )
    )
    package = r1c.verify_preflight_manifest(artifact)
    assert package["artifact"] == {
        "logical_name": artifact.name,
        "size_bytes": artifact.stat().st_size,
        "sha256": r1c.file_identity(artifact)["sha256"],
    }
    assert package["manifest"]["logical_name"] == manifest.name
    artifact.write_text("tampered\n")
    with pytest.raises(r1c.TargetSealError, match="does not match"):
        r1c.verify_preflight_manifest(artifact)


def _path_inference(
    *, core: bool, stronger: bool, core_unresolved: bool = False
) -> dict:
    core_decision = "UNRESOLVED" if core_unresolved else (
        "DECLARED" if core else "NOT_DECLARED"
    )
    return {
        "path_tests": {
            "core_paired_donor": {
                "decision": core_decision,
                "p_ci_disagreement": core_unresolved,
            },
            "stronger_downstream_preference": {
                "decision": "DECLARED" if stronger else "NOT_DECLARED",
                "p_ci_disagreement": False,
            },
        }
    }


def test_exact_nested_declaration_paths_and_verdict_branches() -> None:
    with pytest.raises(ValueError, match="without core"):
        r1c.observed_decision_paths(_path_inference(core=False, stronger=True))
    malformed = _path_inference(core=False, stronger=False)
    malformed["path_tests"]["core_paired_donor"]["decision"] = "FAILED"
    with pytest.raises(ValueError, match="three-state"):
        r1c.observed_decision_paths(malformed)

    core = _path_inference(core=True, stronger=False)
    assert r1c.observed_decision_paths(core) == {
        "core_paired_donor": True,
        "stronger_downstream_preference": False,
    }
    assert r1c.production_verdict(core, {"all": True}, "target") == (
        "CONFIRMED_PAIRED_DONOR_ACCEPTOR_BEHAVIOR"
    )

    stronger = _path_inference(core=True, stronger=True)
    assert r1c.observed_decision_paths(stronger) == {
        "core_paired_donor": True,
        "stronger_downstream_preference": True,
    }
    assert r1c.production_verdict(stronger, {"all": True}, "target") == (
        "CONFIRMED_DOWNSTREAM_ACCEPTOR_PREFERENCE"
    )
    assert (
        r1c.production_verdict(
            _path_inference(core=False, stronger=False), {"all": True}, "target"
        )
        == "NO_CONFIRMED_PAIRED_DONOR_PATH"
    )
    assert (
        r1c.production_verdict(
            _path_inference(core=False, stronger=False, core_unresolved=True),
            {"all": True},
            "target",
        )
        == "PAIRED_DONOR_PATH_UNRESOLVED"
    )
    assert r1c.production_verdict(stronger, {"failed": False}, "target") == "INCONCLUSIVE"


def test_production_clopper_pearson_boundaries_use_stable_1000_trial_audit() -> None:
    assert r1c.PRODUCTION_RANDOMISATION == 4000
    assert 1 / (r1c.PRODUCTION_RANDOMISATION + 1) == pytest.approx(1 / 4001)
    assert r1c.PRODUCTION_TRIALS == 1000
    _, fpr_upper_38 = r1c.one_sided_binomial_bounds(38, 1000, alpha=0.05)
    _, fpr_upper_39 = r1c.one_sided_binomial_bounds(39, 1000, alpha=0.05)
    power_lower_915, _ = r1c.one_sided_binomial_bounds(915, 1000, alpha=0.05)
    power_lower_916, _ = r1c.one_sided_binomial_bounds(916, 1000, alpha=0.05)
    assert fpr_upper_38 <= 0.05 < fpr_upper_39
    assert power_lower_915 < 0.90 <= power_lower_916
    assert r1c.MAX_FALSE_POSITIVES == 38
    assert r1c.MIN_POWER_SUCCESSES == 916


def test_health_probe_coordinates_are_deterministic_and_exclude_both_real_boundaries() -> None:
    records, _ = r1c.prepare_cohort([_example(i) for i in range(30)], 30)
    first = r1c._health_positions(records, 8, 7)
    second = r1c._health_positions(records, 8, 7)
    assert first == second
    for record, positions in zip(records, first):
        excluded = (
            record.example.donor_pos,
            record.example.acceptor_pos - 1,
            record.example.next_acceptor_pos - 1,
        )
        assert all(all(abs(position - site) > 3 for site in excluded) for position in positions)


def test_bound_input_snapshot_detects_toctou(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("frozen\n")
    snapshot = r1c.snapshot_input_paths([source, source])
    r1c.verify_input_snapshot(snapshot)
    source.write_text("changed\n")
    with pytest.raises(RuntimeError, match="changed during execution"):
        r1c.verify_input_snapshot(snapshot)


def test_checkpoint_staging_is_private_and_rejects_late_inventory_additions(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    (model / "weights.bin").write_bytes(b"weights")
    _, _, inventory = r1c.checkpoint_files(model)
    snapshot = r1c.snapshot_input_paths(inventory)
    expected = r1c.checkpoint_bundle_identity(
        model, snapshot=snapshot, expected_files=inventory
    )
    staging, staged_model, staged_identity = r1c.stage_checkpoint_bundle(
        model, inventory, snapshot
    )
    try:
        assert staged_model != model
        assert staged_model.parent == Path(staging.name)
        assert staged_identity == expected
        assert r1c.checkpoint_bundle_identity(staged_model)["files"] == expected["files"]
    finally:
        staging.cleanup()

    (model / "late-tokenizer.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="inventory changed"):
        r1c.verify_checkpoint_inventory(model, inventory, snapshot)
    with pytest.raises(RuntimeError, match="inventory changed"):
        r1c.stage_checkpoint_bundle(model, inventory, snapshot)


def test_target_static_ordering_spends_before_cohort_model_and_output_inspection() -> None:
    source = RUNNER.read_text().split("def main() -> None:", 1)[1]
    authorization = source.index("authorize_production_stage(")
    snapshot = source.index("input_snapshot = snapshot_input_paths(")
    attempt = source.index("target_receipt = reserve_r1c_target_attempt(")
    output_inspection = source.index("output = guard_output_path(output)")
    cohort = source.index("examples = load_examples_unscored(")
    model = source.index("adapter = load_adapter(")
    cohort_snapshot_check = source.index(
        "verify_input_snapshot(input_snapshot)", cohort
    )
    artifact = source.index("atomic_write_new(output, result_bytes)")
    manifest = source.index("atomic_write_new(manifest_path, manifest_bytes)")
    completion = source.index("complete_r1c_target_attempt(")
    assert (
        authorization
        < snapshot
        < attempt
        < output_inspection
        < cohort
        < cohort_snapshot_check
        < model
        < artifact
        < manifest
        < completion
    )
    assert "load_adapter(str(checkpoint_model_path)" in source


def test_manifest_publication_uses_snapshot_records_and_atomic_create_only_writes() -> None:
    source = RUNNER.read_text().split("def main() -> None:", 1)[1]
    assert 'manifest.inputs = [' in source
    assert "manifest.add_input(" not in source
    assert "manifest.add_output(" not in source
    assert "atomic_write_new(output, result_bytes)" in source
    assert "atomic_write_new(manifest_path, manifest_bytes)" in source
    assert 'manifest.git_commit = str(pre_run_git["git_commit"])' in source


def test_instrument_binds_artifact_guard_source() -> None:
    assert r1c.REPO / "src/gfmcd/artifact_guard.py" in r1c.CORE_DEPENDENCIES
