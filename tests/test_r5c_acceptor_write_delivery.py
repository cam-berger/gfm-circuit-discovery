from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert_acceptor_coupling/run_r5c_acceptor_write_delivery.py"
)
SPEC = importlib.util.spec_from_file_location("run_r5c_acceptor_write_delivery", SCRIPT)
R5C = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = R5C
SPEC.loader.exec_module(R5C)

SYNTHETIC_GENE_ROWS = R5C.MIN_GENE_CLUSTERS


def test_v2_registry_orders_dense_write_before_optional_subspace_and_delivery():
    assert R5C.WRITER_SITE_ID == "L3H1.output_projected@acceptor_query"
    assert R5C.PRIMARY_WRITE_REQUIRES_LOW_RANK is False
    order = R5C.fixed_scientific_gate_order()
    assert order[:3] == (
        "R1C_PAIRED_DONOR_BEHAVIOR",
        "R5A_DONOR_LOCAL_READER",
        "R5C_DENSE_LOCAL_WRITE",
    )
    assert order.index("R5C_PRIMARY_PAIR_DELIVERY") < order.index(
        "R5C_LOW_RANK_STRENGTHENING"
    )
    registry = R5C.path_registry()
    assert "P_C1_DENSE_LOCAL_WRITE" in registry["C1"]
    assert "P_C2_LOW_RANK_STRENGTHENING" in registry["C2"]
    assert "P_C4_PRIMARY_PAIR_DELIVERY" in registry["C4"]
    assert "P_C4_DISTINCT_WITHIN_TESTED_SET" in registry["C4"]

    basic = registry["C4"]["P_C4_PRIMARY_PAIR_DELIVERY"]
    distinct = registry["C4"]["P_C4_DISTINCT_WITHIN_TESTED_SET"]
    donor_route_names = {
        R5C.ensemble_name(pair) for pair in R5C.DONOR_ROUTE_PAIRS
    }
    assert all(
        not any(route in requirement.row for route in donor_route_names)
        for requirement in basic
    )
    assert any(
        any(route in requirement.row for route in donor_route_names)
        for requirement in distinct
    )


def test_health_registry_covers_exact_production_arms_and_own_baselines():
    without_basis = R5C.production_health_arm_registry(
        include_optional_basis=False
    )
    with_basis = R5C.production_health_arm_registry(include_optional_basis=True)

    dense = {
        "C1|L3H1.output_projected|necessity": (
            "post_wo_writer_delta_subtraction",
            "clean",
        ),
        "C1|L3H1.output_projected|sufficiency": (
            "post_wo_writer_delta_addition",
            "corrupt",
        ),
    }
    delivery = {
        f"delivery|{R5C.ensemble_name(ensemble)}|{direction}": (
            "post_wo_writer_restoration_plus_corrupt_head_patch"
            if direction == "necessity"
            else "writer_restored_head_patch_into_corrupt",
            "writer_restored" if direction == "necessity" else "corrupt",
        )
        for ensemble in R5C.complete_downstream_registry()
        for direction in ("necessity", "sufficiency")
    }
    assert {
        name: (record["constructor"], record["baseline_state"])
        for name, record in without_basis.items()
    } == dense | delivery
    assert len(without_basis) == 2 + 2 * len(R5C.complete_downstream_registry())

    c2_names = {
        "C2|selected_global_basis|necessity",
        "C2|selected_global_basis|sufficiency",
        *{
            f"C2|random_global_basis_{draw:02d}|{direction}"
            for draw in range(32)
            for direction in ("necessity", "sufficiency")
        },
    }
    assert set(with_basis) - set(without_basis) == c2_names
    assert all(
        with_basis[name]["baseline_state"]
        == ("clean" if name.endswith("|necessity") else "corrupt")
        for name in c2_names
    )

    records = {
        source: {
            name: {
                **arm,
                "strict_noninferiority_pass": True,
            }
            for name, arm in without_basis.items()
        }
        for source in ("partner", "pseudo")
    }
    health = {
        "schema": "gfmcd.r5c.paired_noop_health/3",
        "candidate_outcomes_used": False,
        "optional_basis_executed": False,
        "arm_registry": without_basis,
        "arm_registry_sha256": R5C.canonical_sha256(without_basis),
        "records": records,
        "all_arms_ok": True,
    }
    assert R5C.validate_production_health_record(
        health,
        include_optional_basis=False,
    )["arm_count_per_source"] == len(without_basis)

    missing = deepcopy(health)
    missing["records"]["partner"].pop(next(iter(without_basis)))
    with pytest.raises(R5C.InstrumentError, match="exact production arm coverage"):
        R5C.validate_production_health_record(
            missing,
            include_optional_basis=False,
        )

    altered_registry = deepcopy(health)
    altered_registry["arm_registry"][next(iter(without_basis))][
        "baseline_state"
    ] = "corrupt"
    with pytest.raises(R5C.InstrumentError, match="health identity"):
        R5C.validate_production_health_record(
            altered_registry,
            include_optional_basis=False,
        )


def test_v2_common_leg_binds_the_local_writer_intervention_not_final_residual():
    corrupt = np.asarray([1.0, 2.0])
    restored = np.asarray([3.0, 5.0])
    record = R5C.local_writer_common_leg_record(
        "partner",
        corrupt,
        restored,
        writer_site_id=R5C.WRITER_SITE_ID,
        writer_intervention_id="dense-clean-into-corrupt",
    )
    replayed = R5C.validate_local_writer_common_legs(
        [record, deepcopy(record)],
        m_corrupt=corrupt,
        m_writer_restored=restored,
    )
    assert replayed["writer_site_id"] == R5C.WRITER_SITE_ID
    assert "final_residual" not in json.dumps(replayed)

    wrong = deepcopy(record)
    wrong["writer_intervention_id"] = "full-final-residual-rescue"
    wrong["common_leg_pair_sha256"] = R5C.canonical_sha256(
        {key: value for key, value in wrong.items() if key != "common_leg_pair_sha256"}
    )
    with pytest.raises(R5C.InstrumentError, match="common legs differ"):
        R5C.validate_local_writer_common_legs([record, wrong])


def test_local_writer_source_cache_changes_only_l3h1_at_the_acceptor_query():
    original = {3: R5C.torch.zeros((2, 16, 5, 4))}
    replacement = np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    source = R5C.local_writer_source_cache(
        original,
        query_token_indices=(1, 3),
        replacement=replacement,
    )
    changed = source[3] != original[3]
    assert R5C.torch.count_nonzero(changed) == 8
    assert R5C.torch.all(changed[0, 1, 1])
    assert R5C.torch.all(changed[1, 1, 3])
    changed[0, 1, 1] = False
    changed[1, 1, 3] = False
    assert R5C.torch.count_nonzero(changed) == 0
    assert np.array_equal(source[3][0, 1, 1].numpy(), replacement[0])
    assert np.array_equal(source[3][1, 1, 3].numpy(), replacement[1])
    assert R5C.torch.count_nonzero(original[3]) == 0


def test_l3h1_writer_is_bias_free_post_wo_contribution_and_reconstructs_dense():
    contexts = R5C.torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
    )
    weight = R5C.torch.arange(1.0, 17.0).reshape(4, 4)
    bias = R5C.torch.tensor([0.5, 1.0, 1.5, 2.0])
    projected = [
        R5C.output_projected_head_contribution(
            contexts[..., head * 2 : (head + 1) * 2],
            weight,
            head_index=head,
            num_heads=2,
        )
        for head in range(2)
    ]
    reconstructed = projected[0] + projected[1] + bias.numpy()
    expected = R5C.torch.nn.functional.linear(contexts, weight, bias).numpy()
    np.testing.assert_allclose(reconstructed, expected)
    assert R5C.WRITER_SITE_ID == "L3H1.output_projected@acceptor_query"


def test_post_wo_query_delta_hook_changes_only_exact_query_and_replays_delta():
    class AttentionOutput(R5C.torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = R5C.torch.nn.Linear(4, 4, bias=True)

    layers = [
        SimpleNamespace(attention=SimpleNamespace(output=AttentionOutput()))
        for _ in range(4)
    ]
    layer = layers[3]
    adapter = SimpleNamespace(
        encoder_layers=layers,
        device=R5C.torch.device("cpu"),
        num_layers=4,
    )
    values = R5C.torch.arange(24.0).reshape(2, 3, 4)
    mask = R5C.torch.tensor([[False, True, False], [False, False, True]])
    delta = R5C.torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    baseline = layer.attention.output.dense(values).detach()
    audit = {}
    with R5C.audited_add_attention_output_delta(
        adapter,
        layer=3,
        token_mask=mask,
        delta=delta,
        intervention_id="test-post-wo",
        role="dense_local_writer_sufficiency",
        component_id="L3H1",
        audit_sink=audit,
    ):
        observed = layer.attention.output.dense(values).detach()
    expected = baseline.clone()
    expected[mask] += delta
    R5C.torch.testing.assert_close(observed, expected)
    assert audit["schema"] == "gfmcd.r5c.post_wo_delta_audit/1"
    validated = R5C.validate_post_wo_delta_audit(
        audit,
        expected_role="dense_local_writer_sufficiency",
        expected_component_id="L3H1",
    )
    assert validated["requested_token_indices"] == [1, 2]


def _sha(character: str = "a") -> str:
    return character * 64


def test_v2_stage_pin_registry_binds_r1c_preflight_and_terminal_r5b_disposition():
    preflight = set(R5C.stage_pin_names("preflight"))
    target = set(R5C.stage_pin_names("target"))

    assert {
        "FROZEN_R5A_PREFLIGHT_ARTIFACT_SHA256",
        "FROZEN_R5A_PREFLIGHT_MANIFEST_SHA256",
        "FROZEN_R5B_DISPOSITION_ARTIFACT_SHA256",
        "FROZEN_R5B_DISPOSITION_MANIFEST_SHA256",
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256",
        "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256",
    } <= preflight
    assert not any("R1C_TARGET" in name or "R5A_TARGET" in name for name in preflight)

    assert {
        "FROZEN_R5C_PREFLIGHT_ARTIFACT_SHA256",
        "FROZEN_R5C_PREFLIGHT_MANIFEST_SHA256",
        "FROZEN_R1C_TARGET_ARTIFACT_SHA256",
        "FROZEN_R1C_TARGET_MANIFEST_SHA256",
        "FROZEN_R5A_TARGET_ARTIFACT_SHA256",
        "FROZEN_R5A_TARGET_MANIFEST_SHA256",
        "FROZEN_R5B_DISPOSITION_ARTIFACT_SHA256",
        "FROZEN_R5B_DISPOSITION_MANIFEST_SHA256",
    } <= target
    assert not any("R3C" in name or "BUCKET1" in name for name in target)


def test_preflight_authorization_rejects_noncanonical_shared_r1c_pins(monkeypatch):
    for name in R5C.stage_pin_names("preflight"):
        monkeypatch.setattr(R5C, name, _sha("a"))
    monkeypatch.setattr(R5C, "SHARED_R1C_PREFLIGHT_ARTIFACT_SHA256", _sha("b"))
    monkeypatch.setattr(R5C, "SHARED_R1C_PREFLIGHT_MANIFEST_SHA256", _sha("c"))
    monkeypatch.setattr(R5C, "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256", _sha("d"))
    monkeypatch.setattr(R5C, "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256", _sha("e"))

    with pytest.raises(R5C.TargetSealError, match="canonical shared R1c preflight"):
        R5C._authorize_scored_stage("preflight")


def test_v2_schema_and_decision_authority_are_explicit():
    assert R5C.SCHEMA == "gfmcd.r5c.acceptor_write_delivery/2"
    assert R5C.R5_ENTRY_CONTRACT_SCHEMA == "gfmcd.r5.entry_contract/2"
    assert R5C.R5B_BASIS_CONTRACT_SCHEMA == (
        "gfmcd.r5b.frozen_local_writer_basis/2"
    )
    assert R5C.PRODUCTION_BOOTSTRAP_DRAWS >= 10_000
    payload = R5C.instrument_binding_payload()
    assert payload["decision_authority"] == (
        "mean_and_10pct_trimmed_studentized_gene_cluster_bootstrap"
    )
    assert payload["sign_flip_role"] == "diagnostic_only"
    assert R5C.MIN_GENE_CLUSTERS == 30


def test_target_call_graph_excludes_residual_bucket_and_r3c_entry_loaders():
    graph = R5C.target_call_graph_registry()
    ordered = graph["target_only"]
    target = "\n".join(graph["target_only"])
    assert all(name not in target for name in graph["forbidden_target_edges"])
    assert "materialize_r5_capstone_rows" in target
    assert "400_rows_repeated_genes_allowed" in target
    assert ordered.index("reserve_r5_target_attempt") < ordered.index(
        "load_r1c_verified_target_common_legs:after_R5c_receipt"
    ) < ordered.index("load_examples_unscored:test_chr8_chr9_after_receipt")


def test_exact_package_loader_requires_paired_pins_and_is_relocation_safe(tmp_path):
    artifact = tmp_path / "entry.json"
    artifact.write_text(json.dumps({"schema": "test"}))
    artifact_sha256 = R5C.sha256_file(artifact)
    manifest_path = tmp_path / "renamed-manifest.json"
    manifest = {
        "git_dirty": False,
        "outputs": [
            {
                "path": str(tmp_path / "different.json"),
                "size_bytes": artifact.stat().st_size,
                "sha256": artifact_sha256,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(R5C.InstrumentError, match="manifest does not independently bind"):
        R5C.load_exact_json_package(
            artifact_path=artifact,
            manifest_path=manifest_path,
            expected_artifact_sha256=artifact_sha256,
            expected_manifest_sha256=R5C.sha256_file(manifest_path),
            label="test entry",
        )

    # A producing machine's absolute directory is not portable; the logical basename is.
    manifest["outputs"][0]["path"] = "/different/machine/entry.json"
    manifest_path.write_text(json.dumps(manifest))
    record, loaded_manifest, identity = R5C.load_exact_json_package(
        artifact_path=artifact,
        manifest_path=manifest_path,
        expected_artifact_sha256=artifact_sha256,
        expected_manifest_sha256=R5C.sha256_file(manifest_path),
        label="test entry",
    )
    assert record == {"schema": "test"}
    assert loaded_manifest == manifest
    assert identity["artifact"]["logical_name"] == "entry.json"
    assert identity["manifest"]["logical_name"] == "renamed-manifest.json"


def test_r5a_target_must_bind_the_exact_r1c_capstone_handoff():
    handoff = {
        "schema": R5C.R5_CAPSTONE_COHORT_HANDOFF_SCHEMA,
        "row_count": 400,
        "cohort_sha256": _sha("1"),
        "ordered_row_ids_sha256": _sha("2"),
        "ordered_gene_ids_sha256": _sha("3"),
        "sorted_unique_gene_ids_sha256": _sha("4"),
        "capstone_construction_sha256": _sha("5"),
    }
    contract = {
        "schema": R5C.R5_ENTRY_CONTRACT_SCHEMA,
        "instrument_gates_pass": True,
        "claim_eligible": True,
        "verdict": "L3H1_DONOR_LOCAL_READER_SUPPORTED",
        "r3c_used_as_scientific_entry": False,
        "masked_donor_endpoint_constructed": False,
        R5C.STACK_FINGERPRINT_FIELD: _sha("9"),
        "checkpoint_sha256": _sha("8"),
        "capstone_cohort_schema": handoff["schema"],
        "capstone_row_count": 400,
        "capstone_cohort_sha256": handoff["cohort_sha256"],
        "capstone_ordered_row_ids_sha256": handoff[
            "ordered_row_ids_sha256"
        ],
        "capstone_ordered_gene_ids_sha256": handoff[
            "ordered_gene_ids_sha256"
        ],
        "capstone_sorted_unique_gene_ids_sha256": handoff[
            "sorted_unique_gene_ids_sha256"
        ],
        "capstone_construction_sha256": handoff[
            "capstone_construction_sha256"
        ],
    }
    record = {
        "schema": "gfmcd.r5a.acceptor_address_binding/2",
        "stage": "target",
        "status": "L3H1_DONOR_LOCAL_READER_SUPPORTED",
        "all_instrument_gates_pass": True,
        "behavioral_endpoint": "masked_acceptor_G_margin_only",
        "masked_donor_endpoint_constructed": False,
        R5C.STACK_FINGERPRINT_FIELD: _sha("9"),
        "r5_entry_contract": contract,
    }
    validated = R5C.validate_r5a_stage_record(
        record,
        stage="target",
        expected_stack_fingerprint=_sha("9"),
        capstone_handoff=handoff,
    )
    assert validated["same_row_capstone_bound"] is True

    changed = deepcopy(record)
    changed["r5_entry_contract"]["capstone_cohort_sha256"] = _sha("0")
    with pytest.raises(R5C.InstrumentError, match="capstone handoff differs"):
        R5C.validate_r5a_stage_record(
            changed,
            stage="target",
            expected_stack_fingerprint=_sha("9"),
            capstone_handoff=handoff,
        )


def test_r5c_target_spend_key_uses_exact_rows_and_genes_but_not_method_labels():
    handoff = {
        "row_count": 400,
        "cohort_sha256": _sha("1"),
        "sorted_unique_gene_ids_sha256": _sha("2"),
        "capstone_construction_sha256": _sha("3"),
    }
    first = R5C.build_r5c_target_identity(
        handoff,
        checkpoint_sha256=_sha("4"),
    )
    renamed = R5C.build_r5c_target_identity(
        handoff | {"capstone_construction_sha256": _sha("5")},
        checkpoint_sha256=_sha("4"),
    )
    assert first["ordered_rows_sha256"] == handoff["cohort_sha256"]
    assert first["target_gene_ids_sha256"] == handoff[
        "sorted_unique_gene_ids_sha256"
    ]
    assert first["biological_target_key"] == renamed["biological_target_key"]


def test_r5c_preflight_uses_exact_r1c_validation_rows_and_no_future_dependencies():
    r1c_identity = {
        "artifact": {"logical_name": "r1c-preflight.json", "size_bytes": 1, "sha256": _sha("0")},
        "manifest": {"logical_name": "r1c-preflight.manifest.json", "size_bytes": 1, "sha256": _sha("6")},
    }
    r5a_identity = {
        "artifact": {"logical_name": "r5a-preflight.json", "size_bytes": 1, "sha256": _sha("1")},
        "manifest": {"logical_name": "r5a-preflight.manifest.json", "size_bytes": 1, "sha256": _sha("2")},
    }
    disposition_identity = {
        "artifact": {"logical_name": "r5b-disposition.json", "size_bytes": 1, "sha256": _sha("3")},
        "manifest": {"logical_name": "r5b-disposition.manifest.json", "size_bytes": 1, "sha256": _sha("4")},
    }
    r5b_status = "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS"
    basis_plan = R5C.optional_basis_execution_plan(status=r5b_status, runtime=None)
    record = {
        "schema": R5C.SCHEMA,
        "stage": "preflight",
        "status": "PREFLIGHT_INSTRUMENT_PASS",
        "instrument_fingerprint_sha256": _sha("9"),
        "sealed_chr8_chr9_opened": False,
        "residual_validation_bucket_1_used": False,
        "scientific_outcomes_claim_eligible": False,
        "cohort": {
            "role": "AUTHENTICATED_R1C_NONSEALED_VALIDATION_400_ROW_PREFLIGHT",
            "row_count": 400,
            "rows_filtered_or_reordered": False,
            "residual_validation_cohort_used": False,
            "r5c_capstone_authorized": False,
        },
        "preflight_inputs": {
            "r1c_validation_preflight": r1c_identity,
            "r5a_preflight": r5a_identity,
            "r5b_disposition": disposition_identity,
            "r5b_status": r5b_status,
            "r5b_basis_contract_sha256": None,
            "future_r1c_or_r5a_target_required": False,
        },
        "measurement": {
            "instrument_ok": True,
            "optional_r5b_basis": {
                "disposition_status": r5b_status,
                "execution_plan": basis_plan,
            },
            "observed_centered_robust_sensitivity": {"status": "PASS"},
            "families": {
                family: (
                    {
                        "executed": False,
                        "disposition": basis_plan["classification"],
                        "dense_primary_unaffected": True,
                    }
                    if family == "C2"
                    else {
                        "executed": True,
                        "inference": {
                            "B": R5C.PRODUCTION_BOOTSTRAP_DRAWS,
                            "inference_kind": "studentized_gene_bootstrap_asymptotic",
                        },
                        "sign_flip_diagnostic": {"decision_authority": False},
                    }
                )
                for family in R5C.FAMILY_NAMES
            },
        },
        "binding_suite": {"status": "PASS"},
        "engineering_calibration": {"status": "PASS", "decision_authority": False},
    }
    validated = R5C.validate_r5c_preflight_record(
        record,
        fingerprint=_sha("9"),
        r1c_preflight_identity=r1c_identity,
        r5a_preflight_identity=r5a_identity,
        r5b_disposition_identity=disposition_identity,
        r5b_basis_contract_sha256=None,
        r5b_status=r5b_status,
    )
    assert validated["future_target_dependency_used"] is False


def test_r1c_common_leg_replay_is_same_row_strict_and_never_drops_mismatches():
    references = {
        "P_clean": np.asarray([1.0, 2.0, 3.0]),
        "P_under_D": np.asarray([0.5, 1.5, 2.5]),
        "P_under_N": np.asarray([0.9, 1.9, 2.9]),
    }
    passed = R5C.validate_r1c_common_leg_margin_arrays(
        references,
        {name: values.copy() for name, values in references.items()},
    )
    assert passed["all_400_rows_required"] is True
    assert passed["rows_dropped"] == 0

    changed = {name: values.copy() for name, values in references.items()}
    changed["P_under_D"][1] += 10 * R5C.R1C_COMMON_LEG_ATOL
    with pytest.raises(R5C.InstrumentError, match="same-row common-leg replay differs"):
        R5C.validate_r1c_common_leg_margin_arrays(references, changed)


def test_target_common_leg_references_require_the_verified_post_receipt_package():
    identities = [
        {"example_id": f"example-{index:03d}", "gene_id": f"gene-{index // 2:03d}"}
        for index in range(R5C.R5C_EXACT_ROWS)
    ]
    common = [
        {
            "example_id": identity["example_id"],
            "gene_id": identity["gene_id"],
            "margin_levels": {
                "P_clean": float(index + 1),
                "P_under_D": float(index) + 0.25,
                "P_under_N": float(index) + 0.75,
            },
        }
        for index, identity in enumerate(identities)
    ]
    handoff = {
        "row_count": R5C.R5C_EXACT_ROWS,
        "rows": identities,
        "cohort_sha256": _sha("1"),
        "ordered_row_ids_sha256": R5C.canonical_sha256(
            [row["example_id"] for row in identities]
        ),
        "source_package_sha256": _sha("2"),
        "successor_attestation_sha256": _sha("3"),
    }
    package = {
        "schema": R5C.R5_VERIFIED_R1C_COMMON_LEGS_SCHEMA,
        "source_package_sha256": handoff["source_package_sha256"],
        "successor_attestation_sha256": handoff[
            "successor_attestation_sha256"
        ],
        "cohort_sha256": handoff["cohort_sha256"],
        "ordered_row_ids_sha256": handoff["ordered_row_ids_sha256"],
        "ordered_partner_margin_levels_sha256": R5C.canonical_sha256(common),
        "row_count": R5C.R5C_EXACT_ROWS,
        "rows": common,
        "model_outputs_present": True,
        "may_be_used_for_cohort_selection": False,
        "authorized_use": (
            "same-row R1c common-leg replay after the downstream target receipt only"
        ),
        "caller_must_verify_downstream_target_receipt_before_load": True,
    }
    package["record_sha256"] = R5C.canonical_sha256(package)

    references = R5C.r1c_verified_target_common_leg_references(package, handoff)
    assert references["P_clean"].shape == (R5C.R5C_EXACT_ROWS,)
    assert references["P_under_D"][17] == pytest.approx(17.25)

    reordered = deepcopy(package)
    reordered["rows"][0], reordered["rows"][1] = (
        reordered["rows"][1],
        reordered["rows"][0],
    )
    reordered["ordered_partner_margin_levels_sha256"] = R5C.canonical_sha256(
        reordered["rows"]
    )
    reordered["record_sha256"] = R5C.canonical_sha256(
        {key: value for key, value in reordered.items() if key != "record_sha256"}
    )
    with pytest.raises(R5C.InstrumentError, match="row order differs"):
        R5C.r1c_verified_target_common_leg_references(reordered, handoff)


def _passing_diagnostic_records():
    registry = R5C.calibration_registry()
    diagnostics = {}
    for case in registry["coherent_face_registry"]:
        requirement = next(
            requirement
            for requirements in R5C.path_registry()[case["family"]].values()
            for requirement in requirements
            if requirement.row == case["row"]
        )
        key = "|".join((requirement.decision, case["edge"]))
        if key not in diagnostics:
            diagnostics[key] = {
                "representative_row": case["row"],
                "representative_edge": case["edge"],
                "requiring_paths": [
                    path
                    for path, requirements in R5C.path_registry()[case["family"]].items()
                    if any(requirement.row == case["row"] for requirement in requirements)
                ],
                "inference_sha256": _sha("d"),
            }
    binding = {
        "schema": "gfmcd.r5c.constituent_binding_suite/2",
        "robust_authority_classifier_reused": True,
        "inference_kind": "studentized_gene_bootstrap_asymptotic",
        "B": R5C.PRODUCTION_BOOTSTRAP_DRAWS,
        "binding_gene_count": 64,
        "literal_faces_checked": registry["literal_edge_count"],
        "literal_face_matrices_sha256": _sha("a"),
        "robust_boundary_archetypes": diagnostics,
        "robust_boundary_archetype_count": len(diagnostics),
        "all_alternative_plant_inference_sha256": {
            family: _sha("b") for family in R5C.FAMILY_NAMES
        },
        "all_boundary_archetypes_rejected": True,
        "all_path_plants_supported": True,
        "decision_authority_exercised": True,
        "status": "PASS",
    }
    trials = R5C.ENGINEERING_DIAGNOSTIC_TRIALS
    _, false_upper = R5C.one_sided_binomial_bounds(0, trials, alpha=0.05)
    plant_lower, _ = R5C.one_sided_binomial_bounds(trials, trials, alpha=0.05)
    family_records = {}
    for family in R5C.FAMILY_NAMES:
        cases = [
            case
            for case in registry["coherent_face_registry"]
            if case["family"] == family
        ]
        family_records[family] = {
            "trials": trials,
            "false_positive_trials": 0,
            "false_positive_rate": 0.0,
            "one_sided_clopper_pearson_upper": false_upper,
            "boundary_gate_ok": bool(
                false_upper <= R5C.FAMILY_ALPHA + R5C.CALIBRATION_EPSILON
            ),
            "plant_success_trials": trials,
            "plant_success_rate": 1.0,
            "one_sided_clopper_pearson_lower": plant_lower,
            "plant_gate_ok": bool(plant_lower >= R5C.CALIBRATION_POWER_FLOOR),
            "false_positive_pattern_sha256": _sha("e"),
            "plant_success_pattern_sha256": _sha("f"),
            "balanced_case_cycle_sha256": R5C.canonical_sha256(
                [
                    [
                        cases[trial % len(cases)]["row"],
                        cases[trial % len(cases)]["edge"],
                    ]
                    for trial in range(trials)
                ]
            ),
        }
    calibration = {
        "schema": "gfmcd.r5c.engineering_calibration/2",
        "frozen_diagnostic_configuration": True,
        "N_cal": trials,
        "n_draws_per_classifier": R5C.ENGINEERING_DIAGNOSTIC_DRAWS,
        "family_records": family_records,
        "all_family_gates_pass": False,
        "engineering_not_scientific_fwer_evidence": True,
        "decision_authority": False,
        "authorizes_target": False,
        "status": "DIAGNOSTIC_FAIL",
    }
    return binding, calibration


def _sandwich(
    *,
    leg_id: str,
    component: str,
    m_minus: np.ndarray,
    m_c_plus: np.ndarray,
    effect: float,
):
    writer_intervention_id = "dense-clean-L3H1-output-into-corrupt-query"
    record = R5C.local_writer_delivery_sandwich_estimands(
        m_corrupt=m_minus,
        m_writer_restored=m_c_plus,
        m_writer_restored_with_d_corrupt=m_c_plus - effect,
        m_corrupt_with_d_from_writer_restored=m_minus + effect,
        leg_id=leg_id,
        component_id=component,
        writer_site_id=R5C.WRITER_SITE_ID,
        writer_intervention_id=writer_intervention_id,
    )
    record["common_leg"] = R5C.local_writer_common_leg_record(
        leg_id,
        m_minus,
        m_c_plus,
        writer_site_id=R5C.WRITER_SITE_ID,
        writer_intervention_id=writer_intervention_id,
    )
    return record


def _capture_identity(label: str):
    record = {
        "schema": "gfmcd.r5c.condition_capture_identity/1",
        "margin_sha256": R5C.canonical_sha256([label, "margin"]),
        "query_residual_sha256": R5C.canonical_sha256([label, "residual"]),
        "query_logits_sha256": R5C.canonical_sha256([label, "logits"]),
        "head_outputs_sha256_by_layer": {
            str(layer): R5C.canonical_sha256([label, "head", layer])
            for layer in range(6)
        },
        "full_mlm_margin_replay_max_abs_error": 0.0,
        "full_mlm_logits_replay_exact": True,
    }
    record["capture_sha256"] = R5C.canonical_sha256(record)
    return record


def _mutation_record(role, component, heads, *, n, query_index=1, self_patch=False):
    sequence_tokens = 3
    mask = np.zeros((n, sequence_tokens), dtype=bool)
    mask[:, query_index] = True
    layers = sorted({layer for layer, _ in heads})
    source = {
        layer: np.ones((n, 16, sequence_tokens, 1), dtype=np.float32)
        for layer in layers
    }
    before = {
        layer: np.zeros((n, 16, sequence_tokens, 1), dtype=np.float32)
        for layer in layers
    }
    if self_patch:
        source = {layer: value.copy() for layer, value in before.items()}
    after = {layer: value.copy() for layer, value in before.items()}
    for layer, head in heads:
        after[layer][:, head, query_index] = source[layer][:, head, query_index]
    return R5C._head_mutation_audit_record(
        intervention_id=f"{role}:{component}",
        role=role,
        component_id=component,
        heads=heads,
        token_mask=R5C.torch.from_numpy(mask),
        source_cache={
            layer: R5C.torch.from_numpy(value) for layer, value in source.items()
        },
        before_by_layer={
            layer: R5C.torch.from_numpy(value) for layer, value in before.items()
        },
        after_by_layer={
            layer: R5C.torch.from_numpy(value) for layer, value in after.items()
        },
    )


def _post_wo_mutation_record(role, component, *, n, query_index=1):
    mask = np.zeros((n, 3), dtype=np.uint8)
    mask[:, query_index] = 1
    record = {
        "schema": "gfmcd.r5c.post_wo_delta_audit/1",
        "intervention_id": f"{role}:{component}",
        "role": role,
        "component_id": component,
        "layer": R5C.L3_READER[0],
        "writer_site_id": R5C.WRITER_SITE_ID,
        "bias_included_in_component": False,
        "n_rows": n,
        "sequence_tokens": 3,
        "requested_token_indices": [query_index] * n,
        "requested_token_mask_sha256": R5C._boolean_array_sha256(mask),
        "requested_delta_sha256": _sha("1"),
        "observed_delta_sha256": _sha("2"),
        "before_dense_output_sha256": _sha("3"),
        "after_dense_output_sha256": _sha("4"),
        "delta_replay_max_abs_error": 0.0,
        "off_mask_exact": True,
        "hook_count": 1,
    }
    return record | {"audit_sha256": R5C.canonical_sha256(record)}


def _leg(leg_id: str, *, n: int = 5):
    partner = leg_id == "partner"
    m_clean = np.full(n, 0.30)
    m_minus = np.zeros(n) if partner else m_clean.copy()
    m_c_plus = m_clean.copy()
    u_effect = 0.25 if partner else 0.0
    dense_writer = R5C.local_writer_dense_estimands(
        m_clean=m_clean,
        m_corrupt=m_minus,
        m_corrupt_with_clean_writer=m_c_plus,
        m_clean_with_corrupt_writer=m_clean - u_effect,
        leg_id=leg_id,
        writer_site_id=R5C.WRITER_SITE_ID,
    )
    u = R5C.local_writer_subspace_estimands(
        m_clean=m_clean,
        m_corrupt=m_minus,
        m_corrupt_with_projected_writer=m_minus + u_effect,
        m_clean_without_projected_writer=m_clean - u_effect,
        leg_id=leg_id,
        writer_site_id=R5C.WRITER_SITE_ID,
        basis_id="selected-local-writer-basis",
    )
    random_u = []
    for draw in range(32):
        effect = np.full(n, 0.01 if partner else 0.0)
        item = R5C.local_writer_subspace_estimands(
            m_clean=m_clean,
            m_corrupt=m_minus,
            m_corrupt_with_projected_writer=m_minus + effect,
            m_clean_without_projected_writer=m_clean - effect,
            leg_id=leg_id,
            writer_site_id=R5C.WRITER_SITE_ID,
            basis_id=f"random-local-writer-basis-{draw:02d}",
        )
        item.update(
            {
                "draw": draw,
                "projector_sha256": f"{draw:064x}",
                "per_row_energy_match_max_abs_error": 0.0,
                "redrawn": False,
            }
        )
        random_u.append(item)
    sandwich = {}
    for ensemble in R5C.complete_downstream_registry():
        identity = R5C.ensemble_name(ensemble)
        is_candidate = bool(set(ensemble) & set(R5C.ACCEPTOR_CANDIDATES))
        effect = 0.20 if partner and is_candidate else 0.0
        sandwich[identity] = _sandwich(
            leg_id=leg_id,
            component=identity,
            m_minus=m_minus,
            m_c_plus=m_c_plus,
            effect=effect,
        )
    direct_cells = {
        "m_minus": m_minus,
        "m_direct_negative": m_minus.copy(),
    }
    minus_capture = _capture_identity(f"{leg_id}:minus")
    plus_capture = _capture_identity(f"{leg_id}:plus")
    aggregate_capture = {
        "m_minus_capture_sha256": R5C.canonical_sha256(
            [minus_capture["capture_sha256"]]
        ),
        "m_c_plus_capture_sha256": R5C.canonical_sha256(
            [plus_capture["capture_sha256"]]
        ),
    }
    common = R5C.local_writer_common_leg_record(
        leg_id,
        m_minus,
        m_c_plus,
        writer_site_id=R5C.WRITER_SITE_ID,
        writer_intervention_id="dense-clean-L3H1-output-into-corrupt-query",
        m_corrupt_capture_sha256=aggregate_capture["m_minus_capture_sha256"],
        m_writer_restored_capture_sha256=aggregate_capture[
            "m_c_plus_capture_sha256"
        ],
    )
    registered = tuple(
        sorted(
            {R5C.L3_READER, R5C.DIRECT_NEGATIVE}
            | {
                head
                for ensemble in R5C.complete_downstream_registry()
                for head in ensemble
            }
        )
    )
    mutation_records = [
        _mutation_record(
            "registered_clean_self_patch",
            "ALL_REGISTERED_HEADS",
            registered,
            n=n,
            self_patch=True,
        ),
        _mutation_record(
            "registered_minus_self_patch",
            "ALL_REGISTERED_HEADS",
            registered,
            n=n,
            self_patch=True,
        ),
        _mutation_record(
            "pre_wo_writer_equivalence_diagnostic",
            "L3H1",
            (R5C.L3_READER,),
            n=n,
        ),
        _post_wo_mutation_record(
            "dense_local_writer_sufficiency",
            "L3H1",
            n=n,
        ),
        _post_wo_mutation_record(
            "dense_local_writer_necessity",
            "L3H1",
            n=n,
        ),
        _post_wo_mutation_record(
            "selected_local_writer_subspace_sufficiency",
            "U_A",
            n=n,
        ),
        _post_wo_mutation_record(
            "selected_local_writer_subspace_necessity",
            "U_A",
            n=n,
        ),
        _mutation_record(
            "direct_negative", "L0H2", (R5C.DIRECT_NEGATIVE,), n=n
        ),
    ]
    for ensemble in R5C.complete_downstream_registry():
        identity = R5C.ensemble_name(ensemble)
        audits = {
            "writer_restoration": _post_wo_mutation_record(
                "delivery_writer_restoration",
                identity,
                n=n,
            ),
            "necessity": _mutation_record(
                "delivery_necessity", identity, ensemble, n=n
            ),
            "sufficiency": _mutation_record(
                "delivery_sufficiency", identity, ensemble, n=n
            ),
        }
        sandwich[identity]["common_leg"] = dict(common)
        sandwich[identity]["mutation_audit_sha256_by_chunk"] = {
            key: [value["audit_sha256"]] for key, value in audits.items()
        }
        sandwich[identity]["condition_capture_sha256"] = {
            "necessity": _sha("a"),
            "sufficiency": _sha("b"),
        }
        mutation_records.extend(audits.values())
    for draw in range(32):
        mutation_records.extend(
            (
                _post_wo_mutation_record(
                    "random_local_writer_subspace_sufficiency",
                    f"random_projector_{draw:02d}",
                    n=n,
                ),
                _post_wo_mutation_record(
                    "random_local_writer_subspace_necessity",
                    f"random_projector_{draw:02d}",
                    n=n,
                ),
            )
        )
    capture_records = [minus_capture, plus_capture]
    replay_summary = {
        "condition_count": len(capture_records),
        "margin_max_abs_error": 0.0,
        "full_logits_exact": True,
        "condition_capture_registry_sha256": R5C.canonical_sha256(
            [record["capture_sha256"] for record in capture_records]
        ),
    }
    return {
        "leg_id": leg_id,
        "readout": "single masked acceptor G hard margin",
        "masked_donor_endpoint_constructed": False,
        "common_leg": common,
        "dense_writer": dense_writer,
        "local_writer_subspace": u,
        "direct_negative": {
            "component_id": "L0H2",
            "cells": direct_cells,
            "estimands": {"direct_negative_rescue": np.zeros(n)},
            "margin_sha256": R5C.array_sha256(m_minus),
        },
        "random_local_writer_subspaces": random_u,
        "delivery": sandwich,
        "hook_capability": {
            "registered_heads": [R5C.head_name(head) for head in registered],
            "registered_query_self_patch_margin_max_abs_error": 0.0,
            "registered_query_self_patch_full_logits_exact": True,
            "l3_query_self_patch_included": True,
            "candidate_outcomes_used": False,
            "dense_full_residual_replay": replay_summary,
            "condition_capture_records": capture_records,
            "mutation_audit_chunks": [mutation_records],
            "expected_query_token_indices_by_chunk": [[1] * n],
        },
        "chunk_condition_hashes": [
            {
                "m_corrupt_capture_identity": minus_capture,
                "m_writer_restored_capture_identity": plus_capture,
            }
        ],
        "native_conditions": {
            "clean_margin": m_clean,
            "m_corrupt_margin": m_minus,
            "m_writer_restored_margin": m_c_plus,
            "m_clean_with_corrupt_writer_margin": m_clean - u_effect,
        },
    }


def _projector_capability_fixture(n: int = SYNTHETIC_GENE_ROWS):
    projector = np.repeat(
        np.diag([1.0, 0.0])[None, :, :], n, axis=0
    )
    selected = R5C.validate_broadcast_projectors(
        projector, selected_rank=1, expected_rows=n
    )
    random = [
        {
            "draw": draw,
            **R5C.validate_broadcast_projectors(
                projector, selected_rank=1, expected_rows=n
            ),
        }
        for draw in range(32)
    ]
    runtime_hashes = {
        "frozen_basis": _sha("1"),
        "random_bases": _sha("2"),
        "gene_generic_axis": _sha("3"),
        "readout_axis": _sha("4"),
        "selected_rank": _sha("5"),
    }
    record = {
        "schema": "gfmcd.r5c.global_projector_capability/2",
        "selected": selected,
        "random": random,
        "authenticated_runtime_array_sha256": runtime_hashes,
        "observed_runtime_array_sha256": dict(runtime_hashes),
        "all_projectors_validated": 33,
        "audit_sha256": R5C.canonical_sha256(
            {"selected": selected, "random": random}
        ),
        "application": (
            "one authenticated global basis broadcast unchanged; no target folds"
        ),
        "expected_rows": n,
    }
    return record, runtime_hashes


def _without_optional_basis(leg):
    dense_only = deepcopy(leg)
    dense_only["local_writer_subspace"] = None
    dense_only["random_local_writer_subspaces"] = []
    optional_roles = {
        "selected_local_writer_subspace_sufficiency",
        "selected_local_writer_subspace_necessity",
        "random_local_writer_subspace_sufficiency",
        "random_local_writer_subspace_necessity",
    }
    dense_only["hook_capability"]["mutation_audit_chunks"] = [
        [
            record
            for record in dense_only["hook_capability"]["mutation_audit_chunks"][0]
            if record["role"] not in optional_roles
        ]
    ]
    return dense_only


def _capability_fixture(
    partner,
    pseudo,
    *,
    basis_status="VALIDATED_BASIS",
):
    n = len(partner["dense_writer"]["cells"]["m_clean"])
    basis_executed = basis_status == "VALIDATED_BASIS"
    projector_audit, runtime_hashes = (
        _projector_capability_fixture(n) if basis_executed else (None, None)
    )
    residual = {
        "partner": R5C.validate_residual_replay_registry(partner),
        "pseudo": R5C.validate_residual_replay_registry(pseudo),
    }
    mutation = {
        "partner": R5C.validate_mutation_audit_registry(partner),
        "pseudo": R5C.validate_mutation_audit_registry(pseudo),
    }
    common = {
        "partner": R5C.validate_common_leg_capture_registry(partner),
        "pseudo": R5C.validate_common_leg_capture_registry(pseudo),
    }
    health_registry = R5C.production_health_arm_registry(
        include_optional_basis=basis_executed
    )
    health = {
        "schema": "gfmcd.r5c.paired_noop_health/3",
        "candidate_outcomes_used": False,
        "optional_basis_executed": basis_executed,
        "arm_registry": health_registry,
        "arm_registry_sha256": R5C.canonical_sha256(health_registry),
        "records": {
            source: {
                name: {**arm, "strict_noninferiority_pass": True}
                for name, arm in health_registry.items()
            }
            for source in ("partner", "pseudo")
        },
        "all_arms_ok": True,
    }
    evidence = {
        "readout_validation": {
            "acceptor_g_mismatch_count": 0,
            "rows_checked": n,
        },
        "residual_replay": residual,
        "self_patch": {
            "margin_max_abs_error": 0.0,
            "full_logits_exact": True,
            "l3_included": True,
            "registered_heads_exact": True,
        },
        "mutation_audit": mutation,
        "common_leg_capture": common,
        "optional_r5b_basis": {
            "disposition_status": basis_status,
            "executed": basis_executed,
            "execution_plan": R5C.optional_basis_execution_plan(
                status=basis_status,
                runtime=object() if basis_executed else None,
            ),
        },
        "global_basis_projectors": projector_audit,
        "random_projectors": {
            "energy_match_max_abs_error": 0.0 if basis_executed else None,
            "redrawn_count": 0,
            "draws_per_leg": 32 if basis_executed else 0,
        },
        "health": health,
        "diagnostics": {
            "binding_suite_status": "PASS",
            "binding_decision_authority_exercised": True,
            "engineering_calibration_status": "DIAGNOSTIC_FAIL",
            "engineering_calibration_authorizes_target": False,
            "observed_sensitivity_status": "PASS",
            "observed_sensitivity_sha256": _sha("6"),
        },
    }
    record = {
        "gates": {name: True for name in R5C.CAPABILITY_GATES},
        "evidence": evidence,
        "candidate_outcomes_used": False,
    }
    return record, runtime_hashes


def _as_execution_chunk(leg):
    chunk = deepcopy(leg)
    hook = chunk["hook_capability"]
    hook["mutation_audits"] = hook.pop("mutation_audit_chunks")[0]
    hook["expected_query_token_indices"] = hook.pop(
        "expected_query_token_indices_by_chunk"
    )[0]
    conditions = chunk["chunk_condition_hashes"][0]
    m_minus = np.asarray(
        chunk["dense_writer"]["cells"]["m_corrupt"], dtype=float
    )
    m_c_plus = np.asarray(
        chunk["dense_writer"]["cells"]["m_corrupt_with_clean_writer"],
        dtype=float,
    )
    chunk["conditions"] = {
        "clean_margin": np.asarray(
            chunk["dense_writer"]["cells"]["m_clean"], dtype=float
        ),
        "m_corrupt_margin": m_minus.copy(),
        "m_writer_restored_margin": m_c_plus.copy(),
        "m_clean_with_corrupt_writer_margin": np.asarray(
            chunk["dense_writer"]["cells"]["m_clean_with_corrupt_writer"],
            dtype=float,
        ),
        **conditions,
    }
    raw_capture = {
        "m_corrupt_capture_sha256": conditions["m_corrupt_capture_identity"][
            "capture_sha256"
        ],
        "m_writer_restored_capture_sha256": conditions[
            "m_writer_restored_capture_identity"
        ][
            "capture_sha256"
        ],
    }
    for envelope in chunk["delivery"].values():
        envelope["common_leg"] = R5C.local_writer_common_leg_record(
            chunk["leg_id"],
            envelope["cells"]["m_corrupt"],
            envelope["cells"]["m_writer_restored"],
            writer_site_id=R5C.WRITER_SITE_ID,
            writer_intervention_id=envelope["writer_intervention_id"],
            **raw_capture,
        )
        envelope["mutation_audit_sha256"] = {
            direction: values[0]
            for direction, values in envelope.pop(
                "mutation_audit_sha256_by_chunk"
            ).items()
        }
    chunk["common_leg"] = next(iter(chunk["delivery"].values()))["common_leg"]
    chunk.pop("chunk_condition_hashes")
    return chunk


def test_registry_contains_fixed_heads_and_excludes_them_from_every_random_ensemble():
    registry = R5C.random_ensemble_registry()
    assert {key: len(value) for key, value in registry.items()} == {
        "pairs_l4_l4": 8,
        "pairs_l4_l5": 8,
        "triples_l4_l4_l4": 8,
        "triples_l4_l4_l5": 8,
    }
    flattened = [ensemble for values in registry.values() for ensemble in values]
    assert len({R5C.ensemble_name(ensemble) for ensemble in flattened}) == 32
    assert all(
        not (set(ensemble) & R5C.REGISTERED_DOWNSTREAM_HEADS)
        for ensemble in flattened
    )
    assert R5C.ACCEPTOR_CANDIDATES == ((4, 11), (4, 9), (4, 7))
    assert R5C.DONOR_ROUTE_COMPARATORS == ((4, 0), (4, 2), (5, 11))


def test_complete_registry_has_all_singletons_all_three_donor_pairs_and_fixed_triples():
    registry = R5C.complete_downstream_registry()
    assert len(registry) == 44
    assert registry[:6] == tuple(
        (head,) for head in R5C.ACCEPTOR_CANDIDATES + R5C.DONOR_ROUTE_COMPARATORS
    )
    assert all(pair in registry for pair in R5C.DONOR_ROUTE_PAIRS)
    assert R5C.SECONDARY_ACCEPTOR_TRIPLE in registry
    assert R5C.DONOR_ROUTE_TRIPLE in registry
    assert len({R5C.ensemble_name(item) for item in registry}) == len(registry)


def test_intervention_size_families_are_exact_separate_and_have_closed_paths():
    rows = R5C.family_row_names()
    assert {family: len(names) for family, names in rows.items()} == {
        "C1": 9,
        "C2": 10,
        "C3": 72,
        "C4": 40,
        "C5": 20,
    }
    assert all(len(names) == len(set(names)) for names in rows.values())
    assert all(name.startswith("C1|") for name in rows["C1"])
    assert all(name.startswith("C2|") for name in rows["C2"])
    assert all(name.startswith("C3|") for name in rows["C3"])
    assert all(name.startswith("C4|") for name in rows["C4"])
    assert all(name.startswith("C5|") for name in rows["C5"])
    paths = R5C.path_registry()
    assert tuple(paths) == R5C.FAMILY_NAMES
    assert tuple(paths["C3"]) == ("P_C3_L4H11", "P_C3_L4H9", "P_C3_L4H7")
    for family in R5C.FAMILY_NAMES:
        assert all(
            requirement.row in set(rows[family])
            for requirements in paths[family].values()
            for requirement in requirements
        )
    for family in ("C3", "C4", "C5"):
        for requirements in paths[family].values():
            gaps: dict[str, set[str]] = {}
            for requirement in requirements:
                if requirement.decision != "SUPERIOR_TO_CONTROL":
                    continue
                prefix, endpoint = requirement.row.rsplit("|endpoint=", 1)
                gaps.setdefault(prefix, set()).add(endpoint)
            assert gaps
            assert all(value == {"partner", "specific"} for value in gaps.values())


def test_only_c4_random_pair_null_is_decision_bearing():
    paths = R5C.path_registry()
    random_requirements = {
        family: [
            requirement.row
            for requirements in paths[family].values()
            for requirement in requirements
            if "random" in requirement.row
        ]
        for family in R5C.FAMILY_NAMES
    }
    assert random_requirements["C4"]
    assert "mean_8_layer_matched_random_pairs" in "\n".join(
        random_requirements["C4"]
    )
    assert all(
        not rows for family, rows in random_requirements.items() if family != "C4"
    )


def test_partner_pseudo_specific_and_all_gaps_are_derived_from_raw_legs():
    partner = _leg("partner")
    pseudo = _leg("pseudo")
    families = R5C.derive_family_matrices(partner, pseudo)
    assert {name: value.shape for name, value in families.items()} == {
        "C1": (9, 5),
        "C2": (10, 5),
        "C3": (72, 5),
        "C4": (40, 5),
        "C5": (20, 5),
    }
    rows = R5C.family_row_names()
    c3 = dict(zip(rows["C3"], families["C3"]))
    for estimand in R5C.SANDWICH_ESTIMANDS:
        partner_name = R5C._sandwich_row("C3", "L4H11", estimand, "partner")
        pseudo_name = R5C._sandwich_row("C3", "L4H11", estimand, "pseudo")
        specific_name = R5C._sandwich_row("C3", "L4H11", estimand, "specific")
        np.testing.assert_allclose(
            c3[specific_name], c3[partner_name] - c3[pseudo_name]
        )
        partner_gap = R5C._gap_row(
            "C3", "L4H11", "L4H0", estimand, "partner"
        )
        specific_gap = R5C._gap_row(
            "C3", "L4H11", "L4H0", estimand, "specific"
        )
        np.testing.assert_allclose(c3[partner_gap], 0.20)
        np.testing.assert_allclose(c3[specific_gap], 0.20)
        target_specific = c3[
            R5C._sandwich_row("C3", "L4H11", estimand, "specific")
        ]
        route_specific = c3[
            R5C._sandwich_row("C3", "L4H0", estimand, "specific")
        ]
        np.testing.assert_allclose(
            c3[specific_gap], target_specific - route_specific
        )
    c4 = dict(zip(rows["C4"], families["C4"]))
    target = R5C.ensemble_name(R5C.PRIMARY_ACCEPTOR_PAIR)
    for estimand in R5C.SANDWICH_ESTIMANDS:
        for endpoint in ("partner", "specific"):
            mean_row = R5C._gap_row(
                "C4",
                target,
                R5C._delivery_random_mean_id("C4"),
                estimand,
                endpoint,
            )
            np.testing.assert_allclose(c4[mean_row], 0.20)

    primary_requirements = R5C.path_registry()["C4"][
        "P_C4_PRIMARY_PAIR_DELIVERY"
    ]
    primary_gaps = [
        requirement.row
        for requirement in primary_requirements
        if requirement.decision == "SUPERIOR_TO_CONTROL"
    ]
    assert len(primary_gaps) == 4
    assert all("mean_8_layer_matched_random_pairs" in row for row in primary_gaps)
    assert len(R5C.matched_random_pairs()) == 8
    assert len(R5C.matched_random_triples()) == 8
    assert not any(
        "UNIFORM_RANDOM" in path
        for paths in R5C.path_registry().values()
        for path in paths
    )


def test_primary_random_mean_uses_only_layer_matched_controls():
    partner = _leg("partner")
    pseudo = _leg("pseudo")
    unmatched = R5C.random_ensemble_registry()["pairs_l4_l5"][0]
    unmatched_name = R5C.ensemble_name(unmatched)
    unmatched_record = _sandwich(
        leg_id="partner",
        component=unmatched_name,
        m_minus=np.zeros(5),
        m_c_plus=np.full(5, 0.30),
        effect=100.0,
    )
    unmatched_record["common_leg"] = dict(partner["common_leg"])
    partner["delivery"][unmatched_name] = unmatched_record
    baseline = R5C.derive_family_matrices(_leg("partner"), pseudo)["C4"]
    contaminated = R5C.derive_family_matrices(partner, pseudo)["C4"]
    np.testing.assert_array_equal(contaminated, baseline)

    matched = R5C.matched_random_pairs()[0]
    matched_name = R5C.ensemble_name(matched)
    matched_record = _sandwich(
        leg_id="partner",
        component=matched_name,
        m_minus=np.zeros(5),
        m_c_plus=np.full(5, 0.30),
        effect=0.80,
    )
    matched_record["common_leg"] = dict(partner["common_leg"])
    partner["delivery"][matched_name] = matched_record
    changed = R5C.derive_family_matrices(partner, pseudo)["C4"]
    mean_rows = [
        index
        for index, name in enumerate(R5C.family_row_names()["C4"])
        if "mean_8_layer_matched_random_pairs" in name
    ]
    assert mean_rows
    assert any(
        not np.array_equal(changed[index], baseline[index]) for index in mean_rows
    )


def test_chunk_concatenation_replays_primitives_and_capture_identities():
    partner = _leg("partner")
    pseudo = _leg("pseudo")
    expected = R5C.derive_family_matrices(partner, pseudo)
    concatenated_partner = R5C.concatenate_leg_chunks(
        [_as_execution_chunk(partner)]
    )
    concatenated_pseudo = R5C.concatenate_leg_chunks([_as_execution_chunk(pseudo)])
    observed = R5C.derive_family_matrices(
        concatenated_partner, concatenated_pseudo
    )
    for family in R5C.FAMILY_NAMES:
        np.testing.assert_array_equal(observed[family], expected[family])


def test_chunk_concatenation_preserves_a_basis_free_dense_execution():
    partner = _without_optional_basis(_leg("partner"))
    pseudo = _without_optional_basis(_leg("pseudo"))
    concatenated_partner = R5C.concatenate_leg_chunks(
        [_as_execution_chunk(partner)]
    )
    concatenated_pseudo = R5C.concatenate_leg_chunks([_as_execution_chunk(pseudo)])
    assert concatenated_partner["local_writer_subspace"] is None
    assert concatenated_partner["random_local_writer_subspaces"] == []
    observed = R5C.derive_family_matrices(
        concatenated_partner,
        concatenated_pseudo,
        include_optional_basis=False,
    )
    assert tuple(observed) == ("C1", "C3", "C4", "C5")


def test_derived_families_reject_caller_supplied_specific_rows_and_common_leg_drift():
    partner = _leg("partner")
    pseudo = _leg("pseudo")
    # There is deliberately nowhere to inject a caller-supplied "specific" vector.
    partner["specific"] = np.full(5, 999.0)
    families = R5C.derive_family_matrices(partner, pseudo)
    assert not np.any(families["C1"] == 999.0)

    broken = deepcopy(partner)
    first = next(iter(broken["delivery"]))
    broken["delivery"][first]["common_leg"]["m_corrupt_sha256"] = _sha("f")
    with pytest.raises(R5C.InstrumentError, match="common legs differ"):
        R5C.derive_family_matrices(broken, pseudo)

    stored_delivery_tamper = deepcopy(partner)
    component = next(iter(stored_delivery_tamper["delivery"]))
    stored_delivery_tamper["delivery"][component]["estimands"]["N_d"][0] = 99.0
    with pytest.raises(R5C.InstrumentError, match="primitive replay failed"):
        R5C.derive_family_matrices(stored_delivery_tamper, pseudo)

    stored_u_tamper = deepcopy(partner)
    stored_u_tamper["local_writer_subspace"]["estimands"]["S_U"][0] = 99.0
    with pytest.raises(R5C.InstrumentError, match="subspace replay failed"):
        R5C.derive_family_matrices(stored_u_tamper, pseudo)


def test_dense_and_delivery_families_rederive_without_optional_r5b_basis():
    partner = _leg("partner")
    pseudo = _leg("pseudo")
    for leg in (partner, pseudo):
        leg["local_writer_subspace"] = None
        leg["random_local_writer_subspaces"] = []
    families = R5C.derive_family_matrices(
        partner,
        pseudo,
        include_optional_basis=False,
    )
    assert tuple(families) == ("C1", "C3", "C4", "C5")
    assert all(values.shape[1] == 5 for values in families.values())


def test_optional_basis_plan_skips_cleanly_unless_r5b_validated_it():
    runtime = object()
    assert R5C.optional_basis_execution_plan(
        status="VALIDATED_BASIS", runtime=runtime
    ) == {
        "execute_c2": True,
        "classification": "PENDING_TARGET_STRENGTHENING",
        "dense_primary_unaffected": True,
    }
    for status in (
        "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS",
        "BASIS_INSTRUMENT_UNRESOLVED",
    ):
        plan = R5C.optional_basis_execution_plan(status=status, runtime=None)
        assert plan["execute_c2"] is False
        assert plan["classification"].startswith("NOT_EVALUATED_")
        assert plan["dense_primary_unaffected"] is True
    with pytest.raises(R5C.InstrumentError, match="runtime presence differs"):
        R5C.optional_basis_execution_plan(
            status="NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS",
            runtime=runtime,
        )


def test_common_leg_helper_and_gap_helper_fail_on_different_baselines():
    base = np.asarray([0.0, 1.0])
    rescued = np.asarray([0.5, 1.5])
    first = _sandwich(
        leg_id="partner",
        component="L4H11",
        m_minus=base,
        m_c_plus=rescued,
        effect=0.2,
    )
    second = _sandwich(
        leg_id="partner",
        component="L4H0",
        m_minus=base,
        m_c_plus=rescued,
        effect=0.0,
    )
    np.testing.assert_allclose(
        R5C.local_writer_delivery_gaps(
            R5C._replay_delivery_envelope(first),
            R5C._replay_delivery_envelope(second),
        )[
            "N_candidate_minus_comparator"
        ],
        0.2,
    )
    second["cells"]["m_corrupt"] = np.asarray([0.0, 1.1])
    with pytest.raises(R5C.InstrumentError, match="primitive replay failed"):
        R5C._replay_delivery_envelope(second)


def test_global_projector_broadcast_is_symmetric_idempotent_and_rank_exact():
    projector = np.diag([1.0, 0.0, 0.0])
    projectors = np.repeat(projector[None, :, :], 400, axis=0)
    audit = R5C.validate_broadcast_projectors(
        projectors, selected_rank=1, expected_rows=400
    )
    assert audit["n_projectors"] == 400
    delta = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(
        R5C.row_project(delta, projectors[:2]), [[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    )
    broken = projectors.copy()
    broken[0, 0, 1] = 1.0
    with pytest.raises(R5C.InstrumentError, match="symmetry/idempotence/rank"):
        R5C.validate_broadcast_projectors(
            broken, selected_rank=1, expected_rows=400
        )
    with pytest.raises(R5C.InstrumentError, match="positive integer"):
        R5C.validate_broadcast_projectors(
            projectors, selected_rank=True, expected_rows=400
        )


def test_global_r5b_basis_bundle_is_exact_and_never_refit_or_folded(tmp_path):
    rank, width = 2, 5
    basis = np.eye(width, dtype="<f8")[:rank]
    random = np.repeat(basis[None, :, :], 32, axis=0)
    arrays = {
        "frozen_basis": basis,
        "random_bases": random,
        "gene_generic_axis": np.eye(width, dtype="<f8")[2],
        "readout_axis": np.eye(width, dtype="<f8")[3],
        "selected_rank": np.asarray(rank, dtype="<i8"),
    }
    bundle = tmp_path / "global-basis.npz"
    np.savez(bundle, **arrays)
    package_bindings = {
        name: {
            "artifact_sha256": _sha(character),
            "manifest_sha256": _sha(character.upper().lower()),
        }
        for name, character in zip(
            ("r5a_entry", "r5b_derivation", "r5b_validation"),
            ("1", "2", "3"),
        )
    }
    payload = {
        "schema": R5C.R5B_BASIS_CONTRACT_SCHEMA,
        "writer_site_id": R5C.WRITER_SITE_ID,
        "path_role": "nested_candidate_only",
        "selected_rank": rank,
        "basis_id": "sha256:" + R5C.frozen_basis_array_sha256(basis),
        "random_basis_ids": [
            "sha256:" + R5C.frozen_basis_array_sha256(random[draw])
            for draw in range(32)
        ],
        "source_buckets": [0, 1],
        "fit_timing": "after_bucket1_validation_locked",
        "development_only_source": True,
        "representation": "post_W_O_L3H1_additive_contribution_at_acceptor_query",
        "residual_width": width,
        "post_wo_writer_contract_sha256": _sha("8"),
        "bundle": {
            "file_name": bundle.name,
            "size_bytes": bundle.stat().st_size,
            "sha256": R5C.sha256_file(bundle),
        },
        "runtime_arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": R5C.frozen_basis_array_sha256(value),
            }
            for name, value in arrays.items()
        },
        "checkpoint_sha256": _sha("4"),
        "cohort_audit_sha256": _sha("5"),
        R5C.STACK_FINGERPRINT_FIELD: _sha("9"),
        "upstream_artifact_bindings": package_bindings,
        "target_policy": {
            "refit": "forbidden",
            "rank_selection": "forbidden",
            "basis_mutation": "forbidden",
            "basis_application": "exact authenticated arrays only",
        },
        "r5c_target_attempt_hooks": {
            "owner_stage": "R5c biological target",
            "identity_builder": (
                "gfmcd.splicing.r5_target_ledger.build_r5_biological_target_identity"
            ),
            "reserve_hook": (
                "gfmcd.splicing.r5_target_ledger.reserve_r5_target_attempt"
            ),
            "complete_hook": (
                "gfmcd.splicing.r5_target_ledger.complete_r5_target_attempt"
            ),
            "reserve_after": "all predecessor and upstream packages authenticated",
            "reserve_before": "cohort/path/model/output inspection",
            "complete_after": "immutable result and manifest publication",
            "r5b_calls_target_ledger": False,
        },
    }
    contract = payload | {"contract_sha256": R5C.canonical_sha256(payload)}
    runtime = R5C.load_frozen_r5b_global_basis(
        bundle,
        contract=contract,
        expected_stack_fingerprint=_sha("9"),
    )
    assert runtime.selected_rank == rank
    assert runtime.row_bases(range(400)).shape == (400, rank, width)
    assert len(runtime.random_row_bases(range(400))) == 32
    assert runtime.target_refit_performed is False
    assert runtime.target_fold_assignment is None

    changed = deepcopy(contract)
    changed["target_policy"]["refit"] = "allowed"
    changed["contract_sha256"] = R5C.canonical_sha256(
        {key: value for key, value in changed.items() if key != "contract_sha256"}
    )
    with pytest.raises(R5C.InstrumentError, match="target policy"):
        R5C.load_frozen_r5b_global_basis(
            bundle,
            contract=changed,
            expected_stack_fingerprint=_sha("9"),
        )


def test_query_patch_mask_selects_exactly_the_masked_acceptor_query_and_rejects_bool():
    mask = R5C._masked_query_token_mask([3, 5], sequence_tokens=9, token_offset=1)
    assert mask.tolist() == [
        [False, False, False, False, True, False, False, False, False],
        [False, False, False, False, False, False, True, False, False],
    ]
    with pytest.raises(TypeError, match="non-Boolean integers"):
        R5C._masked_query_token_mask([True], sequence_tokens=9, token_offset=1)


def test_corrected_boundary_tost_iut_classifier_supports_plant_and_rejects_touching_edge():
    names = R5C.family_row_names()["C1"]
    planted = np.zeros((len(names), R5C.MIN_GENES))
    index = {name: row for row, name in enumerate(names)}
    for estimand in ("S_writer", "N_writer"):
        for endpoint in ("partner", "specific"):
            planted[index[R5C._writer_row(estimand, endpoint)]] = 0.20
    groups = [f"gene-{item}" for item in range(R5C.MIN_GENES)]
    result = R5C.strict_family_inference(
        "C1", planted, groups, n_draws=199, namespace="r5c-test-plant"
    )
    assert result["paths"]["P_C1_DENSE_LOCAL_WRITE"]["classification"] == "SUPPORTED"
    assert result["boundary_randomisation_audit"][
        "all_decision_edges_share_one_sign_matrix"
    ]

    touching = planted.copy()
    touching[index[R5C._writer_row("S_writer", "partner")]] = R5C.SESOI
    unresolved = R5C.strict_family_inference(
        "C1", touching, groups, n_draws=199, namespace="r5c-test-boundary"
    )
    assert unresolved["paths"]["P_C1_DENSE_LOCAL_WRITE"]["classification"] != "SUPPORTED"


def test_robust_classifier_requires_raw_and_trimmed_gene_evidence_and_reports_tails():
    groups = [f"gene-{item:03d}" for item in range(R5C.MIN_GENES)]
    planted = R5C.coherent_synthetic_family(
        "C1", residual_scale=0.005, outer_trial=7
    )
    supported = R5C.robust_family_inference(
        "C1",
        planted,
        groups,
        n_draws=999,
        namespace="r5c-test-robust-plant",
    )
    path = supported["paths"]["P_C1_DENSE_LOCAL_WRITE"]
    assert path["classification"] == "SUPPORTED"
    assert supported["trim_fraction"] == pytest.approx(0.10)
    assert supported["inference_kind"] == "studentized_gene_bootstrap_asymptotic"

    tail_sensitive = planted.copy()
    row = R5C.family_row_names()["C1"].index(
        R5C._writer_row("S_writer", "partner")
    )
    tail_sensitive[row] = 0.02 + np.linspace(-0.001, 0.001, R5C.MIN_GENES)
    # Exactly the upper 10% carry the large tail. The raw gene mean crosses the
    # boundary, while the prespecified 10%-trimmed mean removes that tail.
    tail_sensitive[row, :3] = 2.0
    rejected = R5C.robust_family_inference(
        "C1",
        tail_sensitive,
        groups,
        n_draws=999,
        namespace="r5c-test-robust-tail",
    )
    row_test = rejected["row_tests"][R5C._writer_row("S_writer", "partner")][
        "test"
    ]
    assert row_test["raw"]["bound"] > R5C.SESOI
    assert row_test["trimmed"]["bound"] < R5C.SESOI
    assert row_test["joint_boundary_pass"] is False
    assert row_test["median"] < 0.021
    assert rejected["paths"]["P_C1_DENSE_LOCAL_WRITE"]["classification"] != "SUPPORTED"


def test_robust_classifier_clusters_repeated_rows_by_gene():
    base = R5C.coherent_synthetic_family(
        "C1", residual_scale=0.005, outer_trial=17
    )
    repeated = np.repeat(base, 2, axis=1)
    groups = [
        f"gene-{index:03d}"
        for index in range(R5C.MIN_GENE_CLUSTERS)
        for _ in range(2)
    ]
    result = R5C.robust_family_inference(
        "C1",
        repeated,
        groups,
        n_draws=399,
        namespace="r5c-test-repeated-gene-clusters",
    )
    row = result["row_tests"][R5C._writer_row("S_writer", "partner")]["test"]
    assert row["n_genes"] == R5C.MIN_GENE_CLUSTERS
    assert result["paths"]["P_C1_DENSE_LOCAL_WRITE"]["classification"] == "SUPPORTED"


def test_target_shaped_observed_centered_sensitivity_uses_robust_primary_paths(
    monkeypatch,
):
    monkeypatch.setattr(R5C, "PRODUCTION_BOOTSTRAP_DRAWS", 399)
    groups = [f"gene-{index % 321:03d}" for index in range(400)]
    families = {
        family: R5C.coherent_synthetic_family(family, n_examples=400)
        for family in ("C1", "C4")
    }
    record = R5C.run_observed_centered_robust_sensitivity(families, groups)
    assert record["status"] == "PASS"
    assert record["row_count"] == 400
    assert record["gene_count"] == 321
    assert record["orientation_successes"] == R5C.SENSITIVITY_ORIENTATIONS
    assert record["minimum_boundary_slack"] > 0.0
    assert R5C.validate_observed_centered_robust_sensitivity(record)[
        "status"
    ] == "PASS"

    tampered = deepcopy(record)
    tampered["decision_authority"] = "sign_flip"
    with pytest.raises(R5C.InstrumentError, match="robust sensitivity differs"):
        R5C.validate_observed_centered_robust_sensitivity(tampered)


def test_three_candidate_paths_are_holm_corrected_as_one_singleton_family():
    names = R5C.family_row_names()["C3"]
    matrix = np.zeros((len(names), R5C.MIN_GENES))
    index = {name: row for row, name in enumerate(names)}
    for candidate in R5C.ACCEPTOR_CANDIDATES:
        identity = R5C.head_name(candidate)
        for endpoint in ("partner", "specific"):
            for estimand in R5C.SANDWICH_ESTIMANDS:
                matrix[index[R5C._sandwich_row("C3", identity, estimand, endpoint)]] = 0.20
        for route in R5C.DONOR_ROUTE_COMPARATORS:
            for estimand in R5C.SANDWICH_ESTIMANDS:
                for endpoint in ("partner", "specific"):
                    matrix[
                        index[
                            R5C._gap_row(
                                "C3",
                                identity,
                                R5C.head_name(route),
                                estimand,
                                endpoint,
                            )
                        ]
                    ] = 0.20
    groups = [f"gene-{item}" for item in range(R5C.MIN_GENES)]
    result = R5C.strict_family_inference(
        "C3", matrix, groups, n_draws=399, namespace="r5c-test-holm"
    )
    assert set(result["paths"]) == {"P_C3_L4H11", "P_C3_L4H9", "P_C3_L4H7"}
    assert all(
        path["holm_adjusted_p"] >= path["iut_p_value"]
        and path["classification"] == "SUPPORTED"
        for path in result["paths"].values()
    )


def test_capability_is_candidate_independent_and_missing_or_false_gate_fails():
    partner = _leg("partner", n=SYNTHETIC_GENE_ROWS)
    pseudo = _leg("pseudo", n=SYNTHETIC_GENE_ROWS)
    record, runtime_hashes = _capability_fixture(partner, pseudo)
    assert (
        R5C.validate_capability(
            record,
            partner_leg=partner,
            pseudo_leg=pseudo,
            expected_runtime_array_sha256=runtime_hashes,
        )["status"]
        == "R5C_CAPABILITY_CONFIRMED"
    )
    bad = deepcopy(record)
    bad["gates"]["dense_full_residual_replay"] = False
    with pytest.raises(R5C.InstrumentError, match="do not rederive"):
        R5C.validate_capability(
            bad,
            partner_leg=partner,
            pseudo_leg=pseudo,
            expected_runtime_array_sha256=runtime_hashes,
        )
    leaked = deepcopy(record)
    leaked["candidate_outcomes_used"] = True
    with pytest.raises(R5C.InstrumentError, match="do not rederive"):
        R5C.validate_capability(
            leaked,
            partner_leg=partner,
            pseudo_leg=pseudo,
            expected_runtime_array_sha256=runtime_hashes,
        )


@pytest.mark.parametrize(
    "status",
    (
        "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS",
        "BASIS_INSTRUMENT_UNRESOLVED",
    ),
)
def test_dense_capability_remains_confirmed_when_optional_r5b_basis_is_skipped(status):
    partner = _without_optional_basis(_leg("partner", n=SYNTHETIC_GENE_ROWS))
    pseudo = _without_optional_basis(_leg("pseudo", n=SYNTHETIC_GENE_ROWS))
    record, runtime_hashes = _capability_fixture(
        partner,
        pseudo,
        basis_status=status,
    )
    assert runtime_hashes is None
    validated = R5C.validate_capability(
        record,
        partner_leg=partner,
        pseudo_leg=pseudo,
        expected_runtime_array_sha256=None,
    )
    assert validated["status"] == "R5C_CAPABILITY_CONFIRMED"
    assert validated["gates"]["optional_r5b_basis_execution_coherent"] is True


def test_mutation_audit_rejects_wrong_token_and_extra_head():
    audit = _mutation_record(
        "dense_local_writer_sufficiency", "L3H1", (R5C.L3_READER,), n=4
    )
    assert R5C.validate_head_mutation_audit(
        audit,
        expected_heads=(R5C.L3_READER,),
        expected_role="dense_local_writer_sufficiency",
        expected_component_id="L3H1",
    )["requested_token_indices"] == [1, 1, 1, 1]

    wrong_token = deepcopy(audit)
    wrong_token["requested_token_indices"][0] = 2
    wrong_token["audit_sha256"] = R5C.canonical_sha256(
        {key: value for key, value in wrong_token.items() if key != "audit_sha256"}
    )
    with pytest.raises(R5C.InstrumentError, match="token mask does not replay"):
        R5C.validate_head_mutation_audit(
            wrong_token, expected_heads=(R5C.L3_READER,)
        )

    extra_head = _mutation_record(
        "dense_local_writer_sufficiency",
        "L3H1",
        (R5C.L3_READER, (3, 2)),
        n=4,
    )
    with pytest.raises(R5C.InstrumentError, match="wrong or extra head"):
        R5C.validate_head_mutation_audit(
            extra_head, expected_heads=(R5C.L3_READER,)
        )


def test_audited_patch_observes_the_actual_hook_before_and_after_tensors():
    module = R5C.torch.nn.Identity()
    adapter = SimpleNamespace(
        num_layers=1,
        num_heads=2,
        attention_self_modules=[module],
    )
    source = {0: R5C.torch.ones((2, 2, 3, 1))}
    target_heads = R5C.torch.zeros((2, 2, 3, 1))
    target_context = target_heads.permute(0, 2, 1, 3).reshape(2, 3, 2)
    mask = R5C.torch.zeros((2, 3), dtype=R5C.torch.bool)
    mask[:, 1] = True
    audit = {}
    with R5C.audited_patch_heads(
        adapter,
        source,
        ((0, 1),),
        token_mask=mask,
        intervention_id="test-hook",
        role="test",
        component_id="L0H1",
        audit_sink=audit,
    ):
        output = module(target_context)
    split = output.view(2, 3, 2, 1).permute(0, 2, 1, 3)
    assert R5C.torch.count_nonzero(split[:, 1, 1]).item() == 2
    assert R5C.torch.count_nonzero(split[:, 1, [0, 2]]).item() == 0
    R5C.validate_head_mutation_audit(
        audit,
        expected_heads=((0, 1),),
        expected_role="test",
        expected_component_id="L0H1",
    )


def test_capability_rejects_altered_projector_and_common_leg_capture_divergence():
    partner = _leg("partner", n=SYNTHETIC_GENE_ROWS)
    pseudo = _leg("pseudo", n=SYNTHETIC_GENE_ROWS)
    record, runtime_hashes = _capability_fixture(partner, pseudo)

    altered_projector = deepcopy(record)
    altered_projector["evidence"]["global_basis_projectors"]["selected"][
        "idempotence_max_abs"
    ] = 1.0
    with pytest.raises(R5C.InstrumentError, match="projector diagnostic"):
        R5C.validate_capability(
            altered_projector,
            partner_leg=partner,
            pseudo_leg=pseudo,
            expected_runtime_array_sha256=runtime_hashes,
        )

    divergent = deepcopy(partner)
    component = next(iter(divergent["delivery"]))
    common = divergent["delivery"][component]["common_leg"]
    common["m_corrupt_capture_sha256"] = _sha("f")
    common["common_leg_pair_sha256"] = R5C.canonical_sha256(
        {
            key: value
            for key, value in common.items()
            if key != "common_leg_pair_sha256"
        }
    )
    with pytest.raises(R5C.InstrumentError, match="common legs differ"):
        R5C.validate_capability(
            record,
            partner_leg=divergent,
            pseudo_leg=pseudo,
            expected_runtime_array_sha256=runtime_hashes,
        )


def test_capability_rejects_health_summary_without_exact_arm_records():
    partner = _leg("partner", n=SYNTHETIC_GENE_ROWS)
    pseudo = _leg("pseudo", n=SYNTHETIC_GENE_ROWS)
    record, runtime_hashes = _capability_fixture(partner, pseudo)
    record["evidence"]["health"] = {
        "all_arms_ok": True,
        "candidate_outcomes_used": False,
    }

    with pytest.raises(R5C.InstrumentError, match="health identity"):
        R5C.validate_capability(
            record,
            partner_leg=partner,
            pseudo_leg=pseudo,
            expected_runtime_array_sha256=runtime_hashes,
        )


def test_calibration_registry_is_coherent_and_does_not_claim_impossible_isolated_faces():
    calibration = R5C.calibration_registry()
    assert calibration["engineering_not_scientific_fwer_evidence"] is True
    assert calibration["isolated_edge_maximum_claimed"] is False
    assert calibration["primitive_generator"].startswith("centered partner/pseudo")
    assert calibration["N_cal"] == R5C.ENGINEERING_DIAGNOSTIC_TRIALS
    assert calibration["authorizes_target"] is False
    assert calibration["literal_edge_count"] == len(calibration["coherent_face_registry"])
    assert all(
        case["primitive_face_policy"].endswith("linked null edges permitted")
        for case in calibration["coherent_face_registry"]
    )


def test_diagnostic_records_are_replayed_and_tampering_fails_closed():
    binding, calibration = _passing_diagnostic_records()
    result = R5C.validate_binding_and_calibration_records(binding, calibration)
    assert result["status"] == "R5C_BINDING_AND_CALIBRATION_RECORDS_VALID"

    tampered = deepcopy(calibration)
    tampered["family_records"]["C3"]["false_positive_trials"] = 1
    with pytest.raises(R5C.InstrumentError, match="C3 calibration replay differs"):
        R5C.validate_binding_and_calibration_records(binding, tampered)


def test_constituent_binding_suite_reaches_every_face_and_robust_archetype():
    suite = R5C.run_constituent_binding_suite(n_draws=399)
    assert suite["status"] == "PASS"
    assert suite["literal_faces_checked"] == R5C.calibration_registry()[
        "literal_edge_count"
    ]
    assert suite["all_boundary_archetypes_rejected"] is True
    assert suite["all_path_plants_supported"] is True
    assert suite["robust_authority_classifier_reused"] is True
    assert suite["inference_kind"] == "studentized_gene_bootstrap_asymptotic"


def test_scientific_verdict_never_relables_healthy_negative_as_instrument_failure():
    base = dict(
        instrument_ok=True,
        entry_behavior="PASS",
        dense_write="SUPPORTED",
        low_rank="SUPPORTED",
        c3_paths={"P_C3_L4H11": "SUPPORTED"},
        primary_delivery="SUPPORTED",
        distinct_tested_set="SUPPORTED",
        secondary_triple="SUPPORTED",
        controls_active=False,
    )
    assert R5C.scientific_verdict(**base).startswith("R5C_DENSE_WRITE")
    assert R5C.scientific_verdict(**(base | {"controls_active": True})) == (
        "R5C_PRIMARY_PAIR_WITH_LOW_RANK_STRENGTHENING_SUPPORTED"
    )
    assert R5C.scientific_verdict(**(base | {"entry_behavior": "NOT_REPLICATED"})) == (
        "ENTRY_BEHAVIOR_NOT_REPLICATED"
    )
    assert R5C.scientific_verdict(**(base | {"instrument_ok": False})) == (
        "INSTRUMENT_UNRESOLVED"
    )
    assert R5C.scientific_verdict(
        **(base | {"low_rank": "NOT_SUPPORTED_AT_FROZEN_THRESHOLDS"})
    ).startswith("R5C_DENSE_WRITE")
    assert R5C.scientific_verdict(
        **(base | {"dense_write": "NOT_SUPPORTED_AT_FROZEN_THRESHOLDS"})
    ) == "DENSE_LOCAL_WRITE_NOT_SUPPORTED"
    assert R5C.scientific_claim_eligible(
        status="R5C_DENSE_WRITE_AND_PRIMARY_PAIR_DELIVERY_SUPPORTED",
        instrument_ok=True,
    ) is True
    assert R5C.scientific_claim_eligible(
        status="R5C_STALE_SUPPORTED",
        instrument_ok=True,
    ) is False
    assert R5C.scientific_claim_eligible(
        status="R5C_DENSE_WRITE_AND_PRIMARY_PAIR_DELIVERY_SUPPORTED",
        instrument_ok=False,
    ) is False

    gates = R5C.ordered_gate_dispositions(
        instrument_ok=True,
        dense_write="SUPPORTED",
        primary_delivery="SUPPORTED",
        low_rank="NOT_SUPPORTED_AT_FROZEN_THRESHOLDS",
        distinct_tested_set="SUPPORTED",
        secondary_triple="SUPPORTED",
    )
    assert gates["R5C_LOW_RANK_STRENGTHENING"]["status"] == (
        "NOT_SUPPORTED_AT_FROZEN_THRESHOLDS"
    )
    assert gates["R5C_DISTINCT_TESTED_SET_STRENGTHENING"]["status"] == (
        "NOT_TESTED_DUE_TO_PRIOR_SCIENTIFIC_STOP"
    )
    assert gates["R5C_SECONDARY_H7_TRIPLE_STRENGTHENING"]["status"] == (
        "NOT_TESTED_DUE_TO_PRIOR_SCIENTIFIC_STOP"
    )


@pytest.mark.parametrize(
    "c2_disposition",
    [
        "NOT_EVALUATED_NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS",
        "NOT_EVALUATED_R5B_BASIS_INSTRUMENT_UNRESOLVED",
    ],
)
def test_optional_c2_skip_preserves_later_ordered_gate_results(c2_disposition):
    gates = R5C.ordered_gate_dispositions(
        instrument_ok=True,
        dense_write="SUPPORTED",
        primary_delivery="SUPPORTED",
        low_rank=c2_disposition,
        distinct_tested_set="SUPPORTED",
        secondary_triple="SUPPORTED",
    )

    assert gates["R5C_LOW_RANK_STRENGTHENING"] == {
        "status": c2_disposition,
        "tested_here": False,
    }
    assert gates["R5C_DISTINCT_TESTED_SET_STRENGTHENING"] == {
        "status": "SUPPORTED",
        "tested_here": True,
    }
    assert gates["R5C_SECONDARY_H7_TRIPLE_STRENGTHENING"] == {
        "status": "SUPPORTED",
        "tested_here": True,
    }


@pytest.mark.parametrize(
    "c2_disposition",
    [
        "NOT_EVALUATED_NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS",
        "NOT_EVALUATED_R5B_BASIS_INSTRUMENT_UNRESOLVED",
    ],
)
@pytest.mark.parametrize(
    ("secondary_triple", "expected"),
    [
        (
            "NOT_SUPPORTED_AT_FROZEN_THRESHOLDS",
            "R5C_DENSE_WRITE_PRIMARY_PAIR_AND_DISTINCT_TESTED_SET_SUPPORTED",
        ),
        (
            "SUPPORTED",
            "R5C_DENSE_WRITE_PRIMARY_PAIR_DISTINCT_TESTED_SET_AND_H7_TRIPLE_SUPPORTED",
        ),
    ],
)
def test_optional_c2_skip_preserves_supported_later_strengthening_verdicts(
    c2_disposition, secondary_triple, expected
):
    verdict = R5C.scientific_verdict(
        instrument_ok=True,
        entry_behavior="PASS",
        dense_write="SUPPORTED",
        low_rank=c2_disposition,
        c3_paths={"P_C3_L4H11": "SUPPORTED"},
        primary_delivery="SUPPORTED",
        distinct_tested_set="SUPPORTED",
        secondary_triple=secondary_triple,
        controls_active=False,
    )

    assert verdict == expected
    assert R5C.scientific_claim_eligible(
        status=verdict,
        instrument_ok=True,
    ) is True


def test_optional_uniform_random_failures_do_not_veto_primary_pair():
    assert R5C.scientific_verdict(
        instrument_ok=True,
        entry_behavior="PASS",
        dense_write="SUPPORTED",
        low_rank="NOT_SUPPORTED_AT_FROZEN_THRESHOLDS",
        c3_paths={},
        primary_delivery="SUPPORTED",
        distinct_tested_set="NOT_SUPPORTED_AT_FROZEN_THRESHOLDS",
        secondary_triple="NOT_SUPPORTED_AT_FROZEN_THRESHOLDS",
        controls_active=False,
    ) == "R5C_DENSE_WRITE_AND_PRIMARY_PAIR_DELIVERY_SUPPORTED"


def test_model_free_audit_binds_exact_r1c_handoffs_and_never_loads_data_or_model():
    audit = R5C.model_free_registry_audit()
    assert audit["status"] == "FROZEN_MODEL_FREE_REGISTRY_AUDIT_PASS"
    assert audit["model_loaded"] is False
    assert audit["annotation_read"] is False
    assert audit["fasta_read"] is False
    assert audit["cohort_read"] is False
    assert audit["chr8_chr9_opened"] is False
    assert audit["masked_donor_endpoint_constructed"] is False
    assert audit["cohort_contracts"]["preflight"] == {
        "schema": R5C.R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
        "row_count": 400,
        "chromosomes": ["chr11", "chr16", "chr19"],
        "source": "authenticated_R1c_validation_preflight_handoff",
        "rows_filtered_or_reordered": False,
    }
    assert audit["cohort_contracts"]["target"]["schema"] == (
        R5C.R5_CAPSTONE_COHORT_HANDOFF_SCHEMA
    )
    assert audit["cohort_contracts"]["target"]["chromosomes"] == ["chr8", "chr9"]


def test_source_freeze_pins_r5c_method_but_no_output_derived_packages():
    assert R5C.FROZEN_PROTOCOL_SHA256 == (
        "98913630c476d848586f59bd72cd209c0d943ecd4b57231441642d5873eede3d"
    )
    assert R5C.FROZEN_INSTRUMENT_FINGERPRINT_SHA256 == (
        "44ff7aad74cb08d6b3de7753fb8ee0acc8c8cc2e3c3032b2adc40210315f3324"
    )
    assert "> **Status: FROZEN FOR PREFLIGHT ONLY" in R5C.PROTOCOL.read_text()
    output_pins = {
        name: value
        for name, value in vars(R5C).items()
        if name.startswith("FROZEN_")
        and name.endswith("_SHA256")
        and name not in {"FROZEN_PROTOCOL_SHA256", "FROZEN_INSTRUMENT_FINGERPRINT_SHA256"}
    }
    assert output_pins
    assert all(value is None for value in output_pins.values())


def test_target_hard_stops_on_none_pins_before_user_paths_are_opened(tmp_path):
    sentinel = tmp_path / "must-not-be-created-or-read.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--stage",
            "target",
            "--r5c-preflight-artifact",
            str(sentinel),
            "--r5c-preflight-manifest",
            str(sentinel),
            "--r1c-artifact",
            str(sentinel),
            "--r1c-manifest",
            str(sentinel),
            "--r1c-attempt",
            str(sentinel),
            "--r1c-completion",
            str(sentinel),
            "--r5a-target-artifact",
            str(sentinel),
            "--r5a-target-manifest",
            str(sentinel),
            "--r5b-disposition-artifact",
            str(sentinel),
            "--r5b-disposition-manifest",
            str(sentinel),
            "--r5b-basis-bundle",
            str(sentinel),
            "--model",
            str(sentinel),
            "--gtf",
            str(sentinel),
            "--fasta",
            str(sentinel),
            "--out",
            str(sentinel),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "target is hard closed" in result.stderr
    assert "literal seals remain None" in result.stderr
    assert not sentinel.exists()


def test_preflight_hard_stops_on_none_pins_before_user_paths_are_opened(tmp_path):
    sentinel = tmp_path / "must-not-be-created-or-read.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--stage",
            "preflight",
            "--r5a-preflight-artifact",
            str(sentinel),
            "--r5a-preflight-manifest",
            str(sentinel),
            "--r5b-disposition-artifact",
            str(sentinel),
            "--r5b-disposition-manifest",
            str(sentinel),
            "--r5b-basis-bundle",
            str(sentinel),
            "--model",
            str(sentinel),
            "--gtf",
            str(sentinel),
            "--fasta",
            str(sentinel),
            "--out",
            str(sentinel),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "preflight is hard closed" in result.stderr
    assert "literal seals remain None" in result.stderr
    assert not sentinel.exists()


def test_authorized_cli_routes_to_the_end_to_end_model_executor(monkeypatch, tmp_path, capsys):
    authorization = {
        "FROZEN_PROTOCOL_SHA256": _sha("a"),
        "FROZEN_INSTRUMENT_FINGERPRINT_SHA256": _sha("b"),
        "FROZEN_R5A_PREFLIGHT_ARTIFACT_SHA256": _sha("c"),
        "FROZEN_R5A_PREFLIGHT_MANIFEST_SHA256": _sha("d"),
        "FROZEN_R5B_DISPOSITION_ARTIFACT_SHA256": _sha("e"),
        "FROZEN_R5B_DISPOSITION_MANIFEST_SHA256": _sha("f"),
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256": _sha("1"),
        "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256": _sha("2"),
    }
    calls = []
    monkeypatch.setattr(R5C, "authorize_preflight", lambda: authorization)

    def execute(args, *, authorization):
        calls.append((args.stage, dict(authorization)))
        return (
            {"status": "PREFLIGHT_INSTRUMENT_PASS"},
            {"output": str(tmp_path / "result.json"), "result_sha256": _sha("f")},
        )

    monkeypatch.setattr(R5C, "run_authorized_model_stage", execute)
    R5C.main(
        [
            "--stage",
            "preflight",
            "--r5a-preflight-artifact",
            "r5a-preflight.json",
            "--r5a-preflight-manifest",
            "r5a-preflight.manifest.json",
            "--r5b-disposition-artifact",
            "r5b-disposition.json",
            "--r5b-disposition-manifest",
            "r5b-disposition.manifest.json",
            "--r5b-basis-bundle",
            "global-basis.npz",
            "--out",
            str(tmp_path / "result.json"),
        ]
    )
    assert calls == [("preflight", authorization)]
    assert "PREFLIGHT_INSTRUMENT_PASS" in capsys.readouterr().out


def test_target_preflight_replay_rederives_families_from_primitive_legs(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(R5C, "PRODUCTION_BOOTSTRAP_DRAWS", 399)
    n = R5C.R5C_EXACT_ROWS
    partner = _without_optional_basis(_leg("partner", n=n))
    pseudo = _without_optional_basis(_leg("pseudo", n=n))
    partner["native_conditions"] = {
        "clean_margin": np.full(n, 0.30),
        "m_corrupt_margin": np.zeros(n),
        "m_writer_restored_margin": np.full(n, 0.30),
        "m_clean_with_corrupt_writer_margin": np.full(
            n, 0.05
        ),
    }
    pseudo["native_conditions"] = {
        "clean_margin": np.full(n, 0.30),
        "m_corrupt_margin": np.full(n, 0.30),
        "m_writer_restored_margin": np.full(n, 0.30),
        "m_clean_with_corrupt_writer_margin": np.full(
            n, 0.30
        ),
    }
    groups = [f"gene-{index:03d}" for index in range(n)]
    basis_status = "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS"
    basis_plan = R5C.optional_basis_execution_plan(
        status=basis_status,
        runtime=None,
    )
    matrices = R5C.derive_family_matrices(
        partner,
        pseudo,
        include_optional_basis=False,
    )
    family_records = {}
    for family in matrices:
        inference = R5C.robust_family_inference(
            family,
            matrices[family],
            groups,
            n_draws=R5C.PRODUCTION_BOOTSTRAP_DRAWS,
            namespace=f"r5c-robust-observed:{family}",
        )
        diagnostic = R5C.strict_family_inference(
            family,
            matrices[family],
            groups,
            n_draws=R5C.PRODUCTION_RANDOMISATION,
            namespace=f"r5c-observed:{family}",
        )
        diagnostic = {**diagnostic, "decision_authority": False}
        family_records[family] = {
            "executed": True,
            "row_names": list(R5C.family_row_names()[family]),
            "raw_values": matrices[family].tolist(),
            "raw_values_sha256": R5C.array_sha256(matrices[family]),
            "inference": inference,
            "inference_sha256": R5C.canonical_sha256(inference),
            "sign_flip_diagnostic": diagnostic,
            "sign_flip_diagnostic_sha256": R5C.canonical_sha256(diagnostic),
        }
    family_records["C2"] = {
        "executed": False,
        "disposition": basis_plan["classification"],
        "dense_primary_unaffected": True,
    }
    family_records = {
        family: family_records[family] for family in R5C.FAMILY_NAMES
    }
    native = R5C.native_behavior_inference(
        np.full(n, 0.30),
        np.zeros(n),
        groups,
        n_draws=R5C.PRODUCTION_RANDOMISATION,
        seed=R5C.PRODUCTION_SEED,
        namespace="r5c-native-entry-behavior",
    )
    capability, runtime_hashes = _capability_fixture(
        partner,
        pseudo,
        basis_status=basis_status,
    )
    assert runtime_hashes is None
    measurement = {
        "groups": groups,
        "ordered_gene_ids_sha256": R5C.canonical_sha256(groups),
        "native_behavior": native,
        "partner_leg": partner,
        "pseudo_leg": pseudo,
        "families": family_records,
        "gene_sufficient_statistics": {
            "one_row_per_gene": True,
            "repeated_genes_allowed": True,
            "n_rows": n,
            "n_genes": n,
            "raw_gene_values_stored_in_families": True,
            "family": {
                family: {
                    "row_sum": np.sum(matrices[family], axis=1).tolist(),
                    "row_sum_squares": np.sum(
                        np.square(matrices[family]), axis=1
                    ).tolist(),
                    "raw_values_sha256": R5C.array_sha256(matrices[family]),
                }
                for family in matrices
            },
        },
        "capability": capability,
        "capability_status": {"status": "R5C_CAPABILITY_CONFIRMED"},
        "instrument_ok": True,
        "optional_r5b_basis": {
            "disposition_status": basis_status,
            "execution_plan": basis_plan,
        },
    }
    binding = R5C.instrument_binding_payload()
    fingerprint = R5C.canonical_sha256(binding)
    binding_suite, engineering_calibration = _passing_diagnostic_records()
    record = {
        "schema": R5C.SCHEMA,
        "stage": "preflight",
        "instrument_fingerprint_sha256": fingerprint,
        "instrument_binding": binding,
        R5C.STACK_BINDING_FIELD: binding[R5C.STACK_BINDING_FIELD],
        R5C.STACK_FINGERPRINT_FIELD: binding[R5C.STACK_FINGERPRINT_FIELD],
        "registry_audit": R5C.model_free_registry_audit(),
        "sealed_chr8_chr9_opened": False,
        "cohort": {
            "role": "AUTHENTICATED_R1C_NONSEALED_VALIDATION_400_ROW_PREFLIGHT",
            "row_count": n,
            "rows_filtered_or_reordered": False,
        },
        "r5b_optional_basis": {
            "disposition_status": basis_status,
            "execution_plan": basis_plan,
            "contract": None,
            "runtime_bundle_sha256": None,
            "runtime_array_sha256": None,
        },
        "measurement": measurement,
        "binding_suite": binding_suite,
        "engineering_calibration": engineering_calibration,
    }
    replay = R5C.validate_preflight_replay(record, fingerprint=fingerprint)
    assert replay["primitive_family_rederivation_pass"] is True

    target_record = deepcopy(record)
    target_record["stage"] = "target"
    target_record["sealed_chr8_chr9_opened"] = True
    target_record["cohort"]["role"] = "AUTHENTICATED_R1C_400_ROW_SAME_ROW_CAPSTONE"
    target_record["r1c_same_row_common_leg_replay"] = {
        "all_400_rows_required": True,
        "rows_dropped": 0,
        "all_pass": True,
        "diagnostics": {
            name: {"rows": n, "mismatch_count": 0}
            for name in ("P_clean", "P_under_D", "P_under_N")
        },
    }
    upstream = {"schema": "gfmcd.r5c.test_upstream/1", "ordered": True}
    target_record["upstream_chain"] = upstream | {
        "chain_sha256": R5C.canonical_sha256(upstream)
    }
    path_classes = {
        family: {
            path: result["classification"]
            for path, result in family_records[family]["inference"]["paths"].items()
        }
        for family in ("C1", "C3", "C4", "C5")
    }
    path_classes["C2"] = {
        "P_C2_LOW_RANK_STRENGTHENING": basis_plan["classification"]
    }
    path_classes = {
        family: path_classes[family] for family in R5C.FAMILY_NAMES
    }
    target_record["measurement"]["path_classifications"] = path_classes
    target_record["measurement"]["controls_active"] = False
    verdict = R5C.scientific_verdict(
        instrument_ok=True,
        entry_behavior="PASS",
        dense_write=path_classes["C1"]["P_C1_DENSE_LOCAL_WRITE"],
        low_rank=path_classes["C2"]["P_C2_LOW_RANK_STRENGTHENING"],
        c3_paths=path_classes["C3"],
        primary_delivery=path_classes["C4"]["P_C4_PRIMARY_PAIR_DELIVERY"],
        distinct_tested_set=path_classes["C4"][
            "P_C4_DISTINCT_WITHIN_TESTED_SET"
        ],
        secondary_triple=path_classes["C5"]["P_C5_SECONDARY_TRIPLE"],
        controls_active=False,
    )
    target_record["measurement"]["verdict"] = verdict
    target_record["status"] = verdict
    target_record["fixed_scientific_gate_order"] = list(
        R5C.fixed_scientific_gate_order()
    )
    target_record["fixed_scientific_gate_dispositions"] = (
        R5C.ordered_gate_dispositions(
            instrument_ok=True,
            dense_write=path_classes["C1"]["P_C1_DENSE_LOCAL_WRITE"],
            primary_delivery=path_classes["C4"][
                "P_C4_PRIMARY_PAIR_DELIVERY"
            ],
            low_rank=path_classes["C2"]["P_C2_LOW_RANK_STRENGTHENING"],
            distinct_tested_set=path_classes["C4"][
                "P_C4_DISTINCT_WITHIN_TESTED_SET"
            ],
            secondary_triple=path_classes["C5"]["P_C5_SECONDARY_TRIPLE"],
        )
    )
    target_record["scientific_outcomes_claim_eligible"] = (
        R5C.scientific_claim_eligible(status=verdict, instrument_ok=True)
    )
    target_replay = R5C.validate_target_replay(
        target_record,
        fingerprint=fingerprint,
    )
    assert target_replay["stage"] == "target"
    assert target_replay["r1c_common_legs_replayed"] is True

    stale_status = deepcopy(target_record)
    stale_status["status"] = "R5C_STALE_SUPPORTED"
    stale_status["scientific_outcomes_claim_eligible"] = True
    with pytest.raises(R5C.InstrumentError, match="official scientific disposition"):
        R5C.validate_target_replay(stale_status, fingerprint=fingerprint)

    official_tampers = []
    stale_paths = deepcopy(target_record)
    stale_paths["measurement"]["path_classifications"]["C1"][
        "P_C1_DENSE_LOCAL_WRITE"
    ] = "TAMPERED"
    official_tampers.append(stale_paths)
    stale_verdict = deepcopy(target_record)
    stale_verdict["measurement"]["verdict"] = "R5C_STALE_SUPPORTED"
    official_tampers.append(stale_verdict)
    stale_gates = deepcopy(target_record)
    stale_gates["fixed_scientific_gate_dispositions"][
        "R5C_DENSE_LOCAL_WRITE"
    ]["status"] = "TAMPERED"
    official_tampers.append(stale_gates)
    stale_eligibility = deepcopy(target_record)
    stale_eligibility["scientific_outcomes_claim_eligible"] = not target_record[
        "scientific_outcomes_claim_eligible"
    ]
    official_tampers.append(stale_eligibility)
    for tampered_official in official_tampers:
        with pytest.raises(
            R5C.InstrumentError,
            match="official scientific disposition",
        ):
            R5C.validate_target_replay(
                tampered_official,
                fingerprint=fingerprint,
            )

    target_record = R5C._json_ready(target_record)
    target_record["cohort"].update(
        {
            "cohort_sha256": _sha("1"),
            "sorted_unique_gene_ids_sha256": _sha("2"),
            "capstone_construction_sha256": _sha("3"),
        }
    )
    target_record["checkpoint"] = {
        "checkpoint_aggregate_sha256": _sha("4")
    }
    target_record["preflight_package"] = {
        "artifact_sha256": _sha("5"),
        "manifest_sha256": _sha("6"),
    }
    identity = R5C.build_r5_biological_target_identity(
        stage="R5c",
        checkpoint_sha256=_sha("4"),
        target_gene_ids_sha256=_sha("2"),
        ordered_rows_sha256=_sha("1"),
        construction_sha256=_sha("3"),
        endpoint_policy="masked_acceptor_G_margin_only",
    )
    attempt_path = tmp_path / "r5c-attempt.json"
    completion_path = tmp_path / "r5c-attempt.completion.json"
    attempt = {
        "schema": R5C.RECEIPT_SCHEMA,
        "created_at_utc": "2026-08-17T00:00:00+00:00",
        "biological_target_key": identity["biological_target_key"],
        "identity": identity,
        "attempt_context": {
            "r5_stack_fingerprint_sha256": target_record[
                R5C.STACK_FINGERPRINT_FIELD
            ],
            "stage_preflight_artifact_sha256": _sha("5"),
            "stage_preflight_manifest_sha256": _sha("6"),
            "upstream_package_chain_sha256": target_record["upstream_chain"][
                "chain_sha256"
            ],
        },
        "policy": (
            "create-only before target cohort/path/model/output inspection; failures "
            "remain spent; executable repackaging cannot change the ledger key"
        ),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    target_record["target_attempt_receipt"] = attempt | {
        "receipt_path": str(attempt_path)
    }
    target_record["target_completion"] = {
        "logical_name": completion_path.name
    }
    artifact_path = tmp_path / "r5c-target.json"
    manifest_path = tmp_path / "r5c-target.manifest.json"
    artifact_path.write_text(
        json.dumps(target_record, indent=2, sort_keys=True) + "\n"
    )

    def published_identity(path):
        return {
            "logical_name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": R5C.sha256_file(path),
        }

    artifact_identity = published_identity(artifact_path)
    attempt_identity = published_identity(attempt_path)
    manifest = {
        "experiment_id": "splicebert_acceptor_coupling:r5c_acceptor_write_delivery",
        "git_dirty": False,
        "inputs": [
            {
                "path": f"/relocated/{attempt_path.name}",
                "size_bytes": attempt_identity["size_bytes"],
                "sha256": attempt_identity["sha256"],
            }
        ],
        "outputs": [
            {
                "path": f"/relocated/{artifact_path.name}",
                "size_bytes": artifact_identity["size_bytes"],
                "sha256": artifact_identity["sha256"],
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_identity = published_identity(manifest_path)
    completion = {
        "schema": R5C.COMPLETION_SCHEMA,
        "created_at_utc": "2026-08-17T00:01:00+00:00",
        "biological_target_key": identity["biological_target_key"],
        "receipt_sha256": attempt_identity["sha256"],
        "result": {
            "size_bytes": artifact_identity["size_bytes"],
            "sha256": artifact_identity["sha256"],
        },
        "manifest": {
            "size_bytes": manifest_identity["size_bytes"],
            "sha256": manifest_identity["sha256"],
        },
        "official_disposition": verdict,
        "status": "TARGET_PUBLICATION_COMPLETE",
    }
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    verified = R5C.verify_r5c_target_package(
        artifact_path,
        manifest_path,
        attempt_path,
        completion_path,
        fingerprint=fingerprint,
    )
    assert verified["status"] == "EXACT_R5C_TARGET_PACKAGE_VERIFIED"
    assert verified["official_disposition"] == verdict
    assert set(verified["package_identities"]) == {
        "artifact",
        "manifest",
        "attempt",
        "completion",
    }

    completion["official_disposition"] = "R5C_STALE_SUPPORTED"
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(R5C.InstrumentError, match="completion.*official"):
        R5C.verify_r5c_target_package(
            artifact_path,
            manifest_path,
            attempt_path,
            completion_path,
            fingerprint=fingerprint,
        )

    completion["official_disposition"] = verdict
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    attempt["biological_target_key"] = _sha("0")
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(R5C.InstrumentError, match="manifest.*exact artifact and attempt"):
        R5C.verify_r5c_target_package(
            artifact_path,
            manifest_path,
            attempt_path,
            completion_path,
            fingerprint=fingerprint,
        )

    tampered = deepcopy(record)
    tampered_values = np.asarray(
        tampered["measurement"]["families"]["C1"]["raw_values"], dtype=float
    )
    tampered_values[0, 0] += 1.0
    tampered["measurement"]["families"]["C1"][
        "raw_values"
    ] = tampered_values.tolist()
    tampered["measurement"]["families"]["C1"][
        "raw_values_sha256"
    ] = R5C.array_sha256(tampered_values)
    with pytest.raises(R5C.InstrumentError, match="do not rederive"):
        R5C.validate_preflight_replay(tampered, fingerprint=fingerprint)


def test_normalized_runner_hash_changes_for_code_but_not_literal_pin_values():
    source = SCRIPT.read_text()
    modified_pin = source.replace(
        (
            'FROZEN_PROTOCOL_SHA256: str | None = '
            '"98913630c476d848586f59bd72cd209c0d943ecd4b57231441642d5873eede3d"'
        ),
        f'FROZEN_PROTOCOL_SHA256: str | None = "{_sha("e")}"',
    )
    assert modified_pin != source
    assert R5C.normalized_runner_sha256(source) == R5C.normalized_runner_sha256(
        modified_pin
    )
    modified_code = source.replace("SESOI = 0.05", "SESOI = 0.051")
    assert R5C.normalized_runner_sha256(source) != R5C.normalized_runner_sha256(
        modified_code
    )


def test_audit_cli_writes_a_fresh_model_free_artifact_and_manifest(tmp_path):
    output = tmp_path / "r5c-audit.json"
    manifest_path = output.with_suffix(".manifest.json")
    subprocess.run(
        [sys.executable, str(SCRIPT), "--stage", "audit", "--out", str(output)],
        text=True,
        capture_output=True,
        check=True,
    )
    record = json.loads(output.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert record["schema"] == R5C.AUDIT_SCHEMA
    assert record["instrument_binding_payload"]["family_alpha"] == pytest.approx(0.05)
    assert (
        R5C.r5_stack_fingerprint(
            record["instrument_binding_payload"][R5C.STACK_BINDING_FIELD]
        )
        == record["instrument_binding_payload"][R5C.STACK_FINGERPRINT_FIELD]
    )
    assert manifest["config"]["model_loaded"] is False
    assert manifest["outputs"][0]["sha256"] == R5C.sha256_file(output)
    second = subprocess.run(
        [sys.executable, str(SCRIPT), "--stage", "audit", "--out", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode != 0
    assert "outputs are immutable" in second.stderr
