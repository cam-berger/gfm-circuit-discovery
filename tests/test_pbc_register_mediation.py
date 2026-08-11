from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from gfmcd.artifact_guard import ArtifactExistsError
from gfmcd.provenance import file_record, sha256_file
from gfmcd.splicing.pbc_register_mediation import (
    INFERENCE_SEED,
    LEARNED_SET,
    LEARNED_STATES,
    MEDIATION_TARGET_SCHEMA,
    PATH_NAMES,
    STATE_OPERATIONS,
    STATE_TABLES,
    analyze_mediation,
    candidate_freeze_sha256,
    compact_mediation_rows,
    expected_null_labels,
    validate_candidate_freeze,
    validate_mediation_rows,
)
from gfmcd.splicing.pbc_register_relocation import REGISTERS, preferred_delta


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts/splicebert_positional_boundary_coherence" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ANALYZE = _load("analyze_p3_register_mediation")
AUDIT = _load("audit_p3_register_mediation")


def _component(component_id: str, norm: float = 1.0):
    return {
        "label": component_id,
        "components": [
            {
                "component_id": component_id,
                "layer": 3,
                "kind": "head",
                "hook_site": "attention.self.context_pre_output_projection",
                "patch_token_count": 120,
            }
        ],
        "joint_output_delta_norm": norm,
    }


def _freeze():
    families = {}
    for path_index, path_name in enumerate(PATH_NAMES):
        families[path_name] = []
        for index, label in enumerate(expected_null_labels(path_name)):
            spec = _component(f"null_L3H{path_index}_{index:02d}", 0.91 + index / 2_000)
            spec["label"] = label
            families[path_name].append(spec)
    learned = _component("L3H1", 1.0)
    learned["label"] = LEARNED_SET
    return {
        "token_support": [393, 513],
        "patch_tensor_sha256": "a" * 64,
        "candidate_screen": {
            "attention_heads": 96,
            "mlp_outputs": 6,
            "all_components_screened": True,
            "rank_metric": "same_state_scrub_loss_in_omega3",
            "prefixes_jointly_remeasured": True,
        },
        "path_target_output_delta_norms": {
            "d_path": 1.0,
            "r_path": 1.0,
            "q_path": 1.0,
        },
        "selection": {
            "candidate_count": 1,
            "jointly_remeasured_fraction": 0.84,
            "classification": "PREFIX_80",
            "smallest_prefix_reaches_80": True,
        },
        "learned_set": learned,
        "null_families": families,
    }


def _state_effects(block: int, blocks: int, bounded: bool):
    centered = block - (blocks - 1) / 2
    if bounded:
        d = 0.003 + centered * 0.00004
        r = 0.002 - centered * 0.00003
        q = 0.004 + centered * 0.00002
    else:
        d = 0.145 + centered * 0.0005
        r = 0.140 - centered * 0.0004
        q = 0.150 + centered * 0.0003
    raw = 0.30 + centered * 0.0002
    attenuated = 0.04 - centered * 0.0001
    no_ramp = centered * 0.00001
    return {
        "raw": raw,
        "scrub": raw - d,
        "restore": attenuated + r,
        "attenuated_host": attenuated,
        "sufficiency": no_ramp + q,
        "no_ramp_host": no_ramp,
    }


def _row(
    *,
    block: int,
    component_set: str,
    path_contrast: str,
    state: str,
    register: int,
    donor: int,
    acceptor: int,
    effect: float,
):
    delta = (acceptor - donor) % 3
    coupling = effect if delta == preferred_delta(register) else 0.0
    patched = state in {"scrub", "restore", "sufficiency"}
    identity = f"b{block:03d}:{donor}:{acceptor}"
    if component_set == LEARNED_SET:
        component_ids = ["L3H1"]
    else:
        path_index = PATH_NAMES.index(path_contrast)
        null_index = int(component_set.rsplit("_", 1)[1])
        component_ids = [f"null_L3H{path_index}_{null_index:02d}"]
    patch_provenance = None
    if patched:
        patch_provenance = {
            "components": component_ids,
            "hook_sites": ["attention.self.context_pre_output_projection"],
            "operation": (
                "inject_same_state_activation_delta"
                if state == "sufficiency"
                else "replace_same_state_activation"
            ),
            "token_support": [393, 513],
            "source_register": 0 if state == "scrub" else register,
            "source_baseline_register": 0 if state == "sufficiency" else None,
            "intact_patch_tensor_sha256": "c" * 64,
            "acceptor_ac_patch_tensor_sha256": "d" * 64,
            "combined_patch_tensor_sha256": "e" * 64,
        }
    return {
        "block_id": f"b{block:03d}",
        "chrom": f"chr{1 + block % 2}",
        "strand": "+" if (block // 2) % 2 == 0 else "-",
        "component_set": component_set,
        "path_contrast": path_contrast,
        "causal_state": state,
        "register": register,
        "donor_phase": donor,
        "acceptor_phase": acceptor,
        "phase_delta": delta,
        "intact_margin": 1.0 + coupling,
        "acceptor_ac_margin": 1.0,
        "coupling": coupling,
        "intact_host_corruption_state": "intact",
        "acceptor_ac_host_corruption_state": "acceptor_ac",
        "intact_source_corruption_state": "intact" if patched else None,
        "acceptor_ac_source_corruption_state": "acceptor_ac" if patched else None,
        "activation_operation": STATE_OPERATIONS[state],
        "host_table_state": STATE_TABLES[state],
        "intact_host_cell_id": identity,
        "acceptor_ac_host_cell_id": identity,
        "intact_host_sequence_sha256": "a" * 64,
        "acceptor_ac_host_sequence_sha256": "b" * 64,
        "intact_source_cell_id": identity if patched else None,
        "acceptor_ac_source_cell_id": identity if patched else None,
        "intact_source_sequence_sha256": "a" * 64 if patched else None,
        "acceptor_ac_source_sequence_sha256": "b" * 64 if patched else None,
        "patch_provenance": patch_provenance,
    }


def _rows(*, blocks: int = 24, bounded: bool = False):
    rows = []
    for block in range(blocks):
        state_effects = _state_effects(block, blocks, bounded)
        for state in LEARNED_STATES:
            for register in REGISTERS:
                for donor in range(3):
                    for acceptor in range(3):
                        rows.append(
                            _row(
                                block=block,
                                component_set=LEARNED_SET,
                                path_contrast="learned",
                                state=state,
                                register=register,
                                donor=donor,
                                acceptor=acceptor,
                                effect=state_effects[state],
                            )
                        )
        for path_index, path_name in enumerate(PATH_NAMES):
            state = {"d_path": "scrub", "r_path": "restore", "q_path": "sufficiency"}[
                path_name
            ]
            for index, label in enumerate(expected_null_labels(path_name)):
                null_effect = 0.005 + index * 0.00005 + (block % 3 - 1) * 0.00002
                if path_name == "d_path":
                    effect = state_effects["raw"] - null_effect
                elif path_name == "r_path":
                    effect = state_effects["attenuated_host"] + null_effect
                else:
                    effect = state_effects["no_ramp_host"] + null_effect
                for register in REGISTERS:
                    for donor in range(3):
                        for acceptor in range(3):
                            rows.append(
                                _row(
                                    block=block,
                                    component_set=label,
                                    path_contrast=path_name,
                                    state=state,
                                    register=register,
                                    donor=donor,
                                    acceptor=acceptor,
                                    effect=effect,
                                )
                            )
    return rows


def _write_relocation_sources(tmp_path: Path, rows):
    raw = [
        {
            "component": "p3",
            "p10_state": "intact",
            "block_id": row["block_id"],
            "chrom": row["chrom"],
            "strand": row["strand"],
            "register": row["register"],
            "donor_phase": row["donor_phase"],
            "acceptor_phase": row["acceptor_phase"],
            "intact_margin": row["intact_margin"],
            "corrupt_margin": row["acceptor_ac_margin"],
            "coupling": row["coupling"],
        }
        for row in rows
        if row["component_set"] == LEARNED_SET and row["causal_state"] == "raw"
    ]
    target_path = tmp_path / "relocation_target.json"
    target_path.write_text(
        json.dumps(
            {
                "schema": "gfmcd.pbc.p3_register_target.v1",
                "status": "COMPLETE",
                "rows": raw,
            },
            sort_keys=True,
        )
        + "\n"
    )
    target_hash = sha256_file(target_path)
    analysis_path = tmp_path / "relocation_analysis.json"
    status = "P3_REGISTER_RELOCATION_SUPPORTED"
    analysis_path.write_text(
        json.dumps(
            {
                "schema": "gfmcd.pbc.p3_register_analysis.v1",
                "status": status,
                "target_sha256": target_hash,
                "decision": {"code": status},
            },
            sort_keys=True,
        )
        + "\n"
    )
    audit_path = tmp_path / "relocation_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": "gfmcd.pbc.p3_register_audit.v1",
                "status": "PASS",
                "target_sha256": target_hash,
            },
            sort_keys=True,
        )
        + "\n"
    )
    sources = {
        "relocation_target": file_record(target_path),
        "relocation_analysis": file_record(analysis_path),
        "relocation_audit": file_record(audit_path),
    }
    authorization = {
        "relocation_target_sha256": sources["relocation_target"]["sha256"],
        "relocation_analysis_sha256": sources["relocation_analysis"]["sha256"],
        "relocation_audit_sha256": sources["relocation_audit"]["sha256"],
        "relocation_status": status,
        "relocation_audit_status": "PASS",
        "path_instrument_capable": True,
    }
    return sources, authorization


def _payload(tmp_path: Path, *, bounded: bool = False):
    freeze = _freeze()
    rows = _rows(bounded=bounded)
    sources, authorization = _write_relocation_sources(tmp_path, rows)
    compact = compact_mediation_rows(rows, freeze=freeze)
    return {
        "schema": MEDIATION_TARGET_SCHEMA,
        "status": "COMPLETE",
        "stage": "development",
        "split_role": "development",
        "reserve_blocks": 384,
        "reserve_cells": 3456,
        "n_blocks": 24,
        "n_cells": 216,
        "authorization": authorization,
        "sources": sources,
        "design": {
            "candidate_freeze": freeze,
            "candidate_freeze_sha256": candidate_freeze_sha256(freeze),
        },
        "mechanics": {
            "passed": True,
            "same_sequence_patching": True,
            "same_corruption_state_patching": True,
            "full_token_support": [393, 513],
            "raw_relocation_replay_intact_margin_max_abs": 0.0,
            "raw_relocation_replay_corrupt_margin_max_abs": 0.0,
            "raw_relocation_replay_coupling_max_abs": 0.0,
            "actual_hook_tensor_stream_sha256": "f" * 64,
            "actual_patch_forward_events": 1,
        },
        **compact,
    }


def test_synthetic_positive_supports_absolute_necessity_restore_and_sufficiency(tmp_path):
    payload = _payload(tmp_path)
    result = ANALYZE.analyze_payload(
        payload,
        target_sha256="synthetic",
        seed=INFERENCE_SEED,
        n_bootstrap=199,
        enforce_confirmatory_contract=False,
    )
    assert result["status"] == "P3_REGISTER_MEDIATED"
    assert all(result["path_inference"]["lower"][name] > 0.05 for name in PATH_NAMES)
    assert all(
        result["specificity"]["rank_tests"][name]["exact_add_one_p_one_sided"]
        == pytest.approx(1 / 32)
        for name in PATH_NAMES
    )
    assert result["reporting_firewall"]["fractional_mediation_reported"] is False
    assert result["evidence_firewall"]["output_mediation_adjudicated"] is True
    assert result["evidence_firewall"]["selector_claim_adjudicated"] is False


def test_synthetic_bounded_null_preserves_alternative_routes(tmp_path):
    payload = _payload(tmp_path, bounded=True)
    result = analyze_mediation(
        _rows(bounded=True),
        freeze=payload["design"]["candidate_freeze"],
        seed=INFERENCE_SEED,
        n_bootstrap=199,
    )


def test_small_synthetic_target_cannot_pass_confirmatory_contract(tmp_path):
    payload = _payload(tmp_path)
    with pytest.raises(ValueError, match="exact 384/3456 reserve"):
        ANALYZE.analyze_payload(
            payload,
            target_sha256="synthetic-small-target",
            seed=INFERENCE_SEED,
            n_bootstrap=20_000,
            enforce_confirmatory_contract=True,
        )


def test_wrong_corruption_state_is_rejected_before_inference():
    freeze = _freeze()
    rows = _rows(blocks=4)
    patched = next(row for row in rows if row["causal_state"] == "scrub")
    patched["intact_source_corruption_state"] = "acceptor_ac"
    with pytest.raises(ValueError, match="crosses corruption states"):
        validate_mediation_rows(rows, freeze=freeze)


def test_incomplete_component_null_lattice_is_rejected():
    freeze = _freeze()
    rows = _rows(blocks=4)
    missing_label = expected_null_labels("q_path")[-1]
    rows.remove(next(row for row in rows if row["component_set"] == missing_label))
    with pytest.raises(ValueError, match="incomplete lattice"):
        validate_mediation_rows(rows, freeze=freeze)


def test_freeze_rejects_incomplete_null_family():
    freeze = _freeze()
    freeze["null_families"]["r_path"].pop()
    with pytest.raises(ValueError, match="exactly 31"):
        validate_candidate_freeze(freeze)


def test_independent_audit_recomputes_positive_target(tmp_path):
    payload = _payload(tmp_path)
    analysis = ANALYZE.analyze_payload(
        payload,
        target_sha256="synthetic-target",
        seed=INFERENCE_SEED,
        n_bootstrap=199,
        enforce_confirmatory_contract=False,
    )
    audit = AUDIT.audit_payloads(
        payload,
        analysis,
        target_sha256="synthetic-target",
        enforce_confirmatory_contract=False,
    )
    assert audit["status"] == "PASS"
    assert audit["recomputed_status"] == "P3_REGISTER_MEDIATED"


def test_audit_rejects_post_analysis_endpoint_tampering(tmp_path):
    payload = _payload(tmp_path)
    analysis = ANALYZE.analyze_payload(
        payload,
        target_sha256="synthetic-target",
        seed=INFERENCE_SEED,
        n_bootstrap=99,
        enforce_confirmatory_contract=False,
    )
    payload["endpoint_rows"][0]["mean_coupling"] += 0.1
    audit = AUDIT.audit_payloads(
        payload,
        analysis,
        target_sha256="synthetic-target",
        enforce_confirmatory_contract=False,
    )
    assert audit["status"] == "FAIL"
    assert not audit["checks"]["complete_same_state_lattice"]


def test_cli_writers_refuse_existing_scientific_records(tmp_path):
    path = tmp_path / "mediation_analysis.json"
    path.write_text("{}\n")
    with pytest.raises(ArtifactExistsError):
        ANALYZE.output_paths(path)
    with pytest.raises(ArtifactExistsError):
        AUDIT.output_paths(path)
