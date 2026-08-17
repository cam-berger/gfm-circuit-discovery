from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

from gfmcd.splicing.r5_stack_binding import (
    R1C_BEHAVIOR_SCHEMA,
    R1C_PROTOCOL_PATH,
    R1C_RUNNER_PATH,
    R3C_PREFLIGHT_ARTIFACT_SHA256,
    R3C_PREFLIGHT_MANIFEST_SHA256,
    R3C_RUNNER_PATH,
    R3C_ENTRY_SCHEMA,
    R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS,
    R3C_INSTRUMENT_REVISION,
    R3C_PATH_HEALTH_RULE,
    R3C_PATH_NAMES,
    R3C_TARGET_HEALTH_RULE,
    R3C_TARGET_HEALTH_SCHEMA,
    R3D_ENTRY_SCHEMA,
    R3D_RUNNER_PATH,
    R3D_TARGET_ARTIFACT_SHA256,
    R3D_TARGET_MANIFEST_SHA256,
    R5_ENTRY_SCHEMA,
    R5_COHORT_SCHEMA,
    R5_EXCLUSION_REGISTRY_SCHEMA,
    R5_PROTOCOL_PATHS,
    R5_RUNNER_PATHS,
    SHARED_R1C_PACKAGE_PIN_NAMES,
    STACK_SCHEMA,
    R5StackBindingError,
    build_r5_stack_binding,
    normalize_freeze_pins,
    normalized_literal_pin_source_sha256,
    normalized_source_record,
    r5_stack_fingerprint,
    validate_r5_stack_binding,
    validate_r5_runtime_policy,
    validate_r3c_target_health_contract,
    validate_r3c_target_manifest_contract,
    validate_serialized_r5_stack,
)


REPO = Path(__file__).resolve().parents[1]


def _source_with_installed_r1c_package_pins(source: str) -> str:
    installed = {
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_LOGICAL_NAME": repr("r1c_preflight.json"),
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_SIZE_BYTES": "901",
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256": repr("e" * 64),
        "FROZEN_R1C_PREFLIGHT_MANIFEST_LOGICAL_NAME": repr(
            "r1c_preflight.manifest.json"
        ),
        "FROZEN_R1C_PREFLIGHT_MANIFEST_SIZE_BYTES": "902",
        "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256": repr("f" * 64),
        "FROZEN_R1C_TARGET_ARTIFACT_LOGICAL_NAME": repr("r1c_target.json"),
        "FROZEN_R1C_TARGET_ARTIFACT_SIZE_BYTES": "1001",
        "FROZEN_R1C_TARGET_ARTIFACT_SHA256": repr("a" * 64),
        "FROZEN_R1C_TARGET_MANIFEST_LOGICAL_NAME": repr(
            "r1c_target.manifest.json"
        ),
        "FROZEN_R1C_TARGET_MANIFEST_SIZE_BYTES": "1002",
        "FROZEN_R1C_TARGET_MANIFEST_SHA256": repr("b" * 64),
        "FROZEN_R1C_TARGET_ATTEMPT_LOGICAL_NAME": repr("r1c_target_attempt.json"),
        "FROZEN_R1C_TARGET_ATTEMPT_SIZE_BYTES": "1003",
        "FROZEN_R1C_TARGET_ATTEMPT_SHA256": repr("c" * 64),
        "FROZEN_R1C_TARGET_COMPLETION_LOGICAL_NAME": repr(
            "r1c_target_attempt.completion.json"
        ),
        "FROZEN_R1C_TARGET_COMPLETION_SIZE_BYTES": "1004",
        "FROZEN_R1C_TARGET_COMPLETION_SHA256": repr("d" * 64),
    }
    rendered = source
    for name, value in installed.items():
        before = rendered
        rendered = rendered.replace(
            f"{name}: str | None = None", f"{name}: str | None = {value}"
        )
        rendered = rendered.replace(
            f"{name}: int | None = None", f"{name}: int | None = {value}"
        )
        assert rendered != before, name
    return rendered


def _load(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_joint_stack_binds_all_runners_protocols_and_transitive_helpers():
    binding = build_r5_stack_binding(repo=REPO)
    assert binding["schema"] == STACK_SCHEMA == "gfmcd.r5.stack_binding/2"
    assert binding["runner_roots"] == list(R5_RUNNER_PATHS)
    assert binding["scientific_entry_runner"] == R3D_RUNNER_PATH
    assert binding["dynamic_entry_runner"] == R3D_RUNNER_PATH
    assert binding["exclusion_provenance_runner"] == R3C_RUNNER_PATH
    assert set(binding["protocols"]) == set(R5_PROTOCOL_PATHS)
    closure = binding["local_python_closure"]["files"]
    assert set(R5_RUNNER_PATHS) < set(closure)
    assert R3D_RUNNER_PATH in closure
    assert R3C_RUNNER_PATH in closure
    assert R1C_RUNNER_PATH in closure
    assert binding["behavior_predecessor_source_present_and_bound"] is True
    assert "src/gfmcd/splicing/r5_stack_binding.py" in closure
    assert "src/gfmcd/splicing/r5_target_ledger.py" in closure
    assert "src/gfmcd/splicing/r5_upstream_provenance.py" in closure
    assert "src/gfmcd/adapters/splicebert.py" in closure
    assert "src/gfmcd/splicing/acceptor_coupling_mechanism.py" in closure
    assert "src/gfmcd/circuits/hf_bert.py" in closure
    assert "src/gfmcd/circuits/__init__.py" in closure
    assert "src/gfmcd/splicing/paired_component.py" in closure
    assert binding["schema_registry_sha256"]
    assert binding["executable_registries_sha256"]
    assert set(binding["executable_registries"]) == set(R5_RUNNER_PATHS)
    assert R5_ENTRY_SCHEMA == "gfmcd.r5.entry_contract/2"
    assert R5_EXCLUSION_REGISTRY_SCHEMA == "gfmcd.r5.exclusion_registry/2"
    assert binding["contract"]["handoff_schemas"]["scientific_entry"] == R3D_ENTRY_SCHEMA
    assert binding["contract"]["handoff_schemas"]["entry"] == R5_ENTRY_SCHEMA
    assert (
        binding["contract"]["handoff_schemas"]["cohort"]
        == R5_COHORT_SCHEMA
        == "gfmcd.r5.acceptor_mechanism_cohort/3"
    )
    assert (
        binding["contract"]["handoff_schemas"]["exclusion_registry"]
        == R5_EXCLUSION_REGISTRY_SCHEMA
    )
    assert binding["contract"]["scientific_entry"]["decision_path"] == (
        "L3H1_reused_component"
    )
    assert binding["contract"]["scientific_entry"]["path_b_can_gate_entry"] is False
    assert binding["contract"]["masked_donor_endpoint_constructed"] is False
    assert binding["contract"]["writer_site_semantics"] == {
        "causal_measurement_site": "post-WO attention output in residual space",
        "model_free_projection_primitive": (
            "gfmcd.splicing.acceptor_coupling_mechanism."
            "output_projected_head_contribution"
        ),
        "torch_projection_primitive": (
            "gfmcd.circuits.hf_bert.project_head_context_to_residual"
        ),
        "torch_intervention_primitive": (
            "gfmcd.circuits.hf_bert.add_attention_output_delta"
        ),
        "pre_WO_context_is_not_a_residual_writer_measurement": True,
    }
    assert binding["contract"]["residual_validation_cohort"] == {
        "schema": R5_COHORT_SCHEMA,
        "authorized_stage_consumers": ["R5b"],
        "bucket_roles": {
            "0": "R5b_derivation",
            "1": "R5b_validation_and_union_refit",
        },
        "rows_per_bucket": 128,
        "r5a_r5c_preflight_authorized": False,
        "r5c_capstone_authorized": False,
    }
    preflight = binding["contract"]["r5a_r5c_preflight_cohort"]
    assert preflight["handoff_schema"] == "gfmcd.r5.preflight_cohort_handoff/1"
    assert preflight["row_materialization_schema"] == (
        "gfmcd.r5.preflight_row_materialization/1"
    )
    assert preflight["source_preconstruction_allowed_chromosomes"] == [
        "chr11",
        "chr16",
        "chr19",
    ]
    assert preflight["residual_validation_cohort_may_be_substituted"] is False
    assert preflight["scientific_entry_authorized"] is False
    assert preflight["semantic_verifier"] == "verify_successor_preflight_package"
    assert set(preflight["prospective_source_package_pins"]) == {
        "artifact",
        "manifest",
    }
    capstone = binding["contract"]["r5c_capstone_cohort"]
    assert capstone["handoff_schema"] == "gfmcd.r5.capstone_cohort_handoff/1"
    assert capstone["row_materialization_schema"] == (
        "gfmcd.r5.capstone_row_materialization/1"
    )
    assert capstone["verified_common_leg_schema"] == (
        "gfmcd.r5.verified_r1c_target_common_legs/1"
    )
    assert capstone["verified_common_leg_loader"] == (
        "load_r1c_verified_target_common_legs"
    )
    assert "never cohort selection" in capstone["verified_common_leg_use"]
    assert capstone["semantic_verifier"] == "verify_successor_target_package"
    assert capstone["authenticated_input_count"] == 6
    assert capstone["may_reuse_residual_validation_bucket_1"] is False
    pins = capstone["prospective_source_package_pins"]
    assert set(pins) == {
        "artifact",
        "manifest",
        "attempt",
        "completion",
        "preflight_artifact",
        "preflight_manifest",
        "value_policy",
    }
    assert pins["preflight_artifact"]["sha256"] == {
        "pin_symbol": "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256",
        "value": "<FROZEN_PIN>",
    }
    assert (
        binding["contract"]["handoff_schemas"]["capstone_cohort_handoff"]
        == "gfmcd.r5.capstone_cohort_handoff/1"
    )
    assert (
        binding["contract"]["handoff_schemas"]["capstone_row_materialization"]
        == "gfmcd.r5.capstone_row_materialization/1"
    )
    assert (
        binding["contract"]["handoff_schemas"][
            "verified_R1c_target_common_legs"
        ]
        == "gfmcd.r5.verified_r1c_target_common_legs/1"
    )
    assert binding["contract"]["handoff_schemas"]["preflight_cohort_handoff"] == (
        "gfmcd.r5.preflight_cohort_handoff/1"
    )
    assert binding["contract"]["target_ledger_identity_policy"] == {
        "biological_key_schema": "gfmcd.r5.biological_target_identity/1",
        "biological_key_fields": [
            "schema",
            "program",
            "stage",
            "checkpoint_sha256",
            "target_gene_ids_sha256",
            "ordered_rows_sha256",
        ],
        "method_context_only_fields": [
            "construction_sha256",
            "endpoint_policy",
            "r5_stack_fingerprint_sha256",
            "stage_preflight_artifact_sha256",
            "stage_preflight_manifest_sha256",
            "upstream_package_chain_sha256",
        ],
        "method_or_package_changes_may_reset_spend": False,
    }
    assert binding["contract"]["sealed_chr8_chr9_access_scope"] == {
        "R1c_R5a_R5b_R5c_method_freeze": "not opened",
        "R5a_R5c_preflight_rows": (
            "authenticated R1c val chr11/16/19 cohort; never residual buckets"
        ),
        "R5b_method_rows": "residual bucket 0 derivation and bucket 1 validation",
        "R1c_behavior_target": "sealed target behavior read after all method freezes",
        "R5a_R5c_targets": (
            "authenticated handoff materialization only after each stage receipt"
        ),
        "R5b": "no chr8/chr9 target stage",
    }
    assert binding["contract"]["stage_order"] == [
        "R3d Path-A authenticated prior",
        "R1c nonsealed validation behavior preflight",
        "R5a authenticated R1c-validation preflight and instrument freeze",
        "R5b residual bucket-0 derivation, bucket-1 validation, and union-basis freeze",
        "R5c authenticated R1c-validation preflight and delivery-instrument freeze",
        "R1c sealed chr8/chr9 acceptor-only behavior target",
        "R5a authenticated capstone target",
        "R5c authenticated capstone target",
    ]
    assert binding["contract"]["method_freeze_order_before_chr8_chr9_unseal"] == [
        "R1c nonsealed validation behavior preflight",
        "R5a authenticated R1c-validation preflight and instrument freeze",
        "R5b residual bucket-0 derivation, bucket-1 validation, and union-basis freeze",
        "R5c authenticated R1c-validation preflight and delivery-instrument freeze",
    ]
    assert binding["contract"]["scientific_target_order_after_method_freeze"] == [
        "R1c sealed chr8/chr9 acceptor-only behavior target",
        "R5a authenticated capstone target",
        "R5c authenticated capstone target",
    ]
    assert binding["contract"]["non_entry_provenance_inputs"] == [
        "exact committed R3c preflight"
    ]
    assert binding["contract"]["required_behavior_predecessor"] == {
        "schema": R1C_BEHAVIOR_SCHEMA,
        "runner_path": R1C_RUNNER_PATH,
        "protocol_path": R1C_PROTOCOL_PATH,
        "split": "chr8/chr9 sealed target",
        "endpoint": "single masked acceptor G hard margin only",
        "must_pass_before_r5a_target": True,
        "must_not_gate_r5a_preflight_or_r5b_basis_freeze": True,
        "attempt_schema": "gfmcd.r1c.target_attempt/2",
        "completion_schema": "gfmcd.r1c.target_completion/1",
        "biological_target_identity_schema": (
            "gfmcd.r1c.biological_target_identity/1"
        ),
        "preflight_verifier": "verify_successor_preflight_package",
        "preflight_verifier_schema": "gfmcd.r1c.verified_preflight_handoff/1",
        "target_verifier": "verify_successor_target_package",
        "target_verifier_schema": "gfmcd.r1c.verified_target_handoff/1",
        "target_requires_exact_predecessor_preflight_pair": True,
    }
    basis = binding["contract"]["r5b_basis_disposition"]
    assert basis["schema"] == "gfmcd.r5b.local_writer_basis_disposition/2"
    assert basis["artifact_and_manifest_required_for_every_status"] is True
    assert basis["statuses"]["VALIDATED_BASIS"] == {
        "runtime_bundle_required": True,
        "selected_basis_fields_required": True,
        "r5c_low_rank_C2_enabled": True,
        "r5c_dense_C1_C4_enabled": True,
    }
    no_basis = basis["statuses"][
        "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS"
    ]
    assert no_basis["runtime_bundle_required"] is False
    assert no_basis["r5c_low_rank_C2_enabled"] is False
    assert no_basis["r5c_dense_C1_C4_enabled"] is True
    assert no_basis["blocks_r5c_primary_paths"] is False
    unresolved_basis = basis["statuses"]["BASIS_INSTRUMENT_UNRESOLVED"]
    assert unresolved_basis == {
        "runtime_bundle_required": False,
        "selected_basis_fields_required": False,
        "r5c_low_rank_C2_enabled": False,
        "r5c_dense_C1_C4_enabled": True,
        "blocks_r5c_primary_paths": False,
    }
    assert basis["r5c_target_scores_r5b_on_chr8_chr9"] is False
    assert binding["required_behavior_predecessor_runner"] == R1C_RUNNER_PATH
    assert binding["upstream_packages"]["r3d_target"]["artifact"]["sha256"] == (
        R3D_TARGET_ARTIFACT_SHA256
    )
    assert binding["upstream_packages"]["r3d_target"]["manifest"]["sha256"] == (
        R3D_TARGET_MANIFEST_SHA256
    )
    assert binding["upstream_packages"]["r3c_preflight"]["artifact"]["sha256"] == (
        R3C_PREFLIGHT_ARTIFACT_SHA256
    )
    assert binding["upstream_packages"]["r3c_preflight"]["manifest"]["sha256"] == (
        R3C_PREFLIGHT_MANIFEST_SHA256
    )
    assert set(binding["repository_runtime_files"]) == {"pyproject.toml", "uv.lock"}
    assert binding["execution_policy"] == {
        "device": "mps",
        "required_absent_environment_variables": [
            "PYTORCH_ENABLE_MPS_FALLBACK"
        ],
    }
    assert set(binding["runtime_identity"]["packages"]) == {
        "numpy",
        "scipy",
        "torch",
        "transformers",
        "tokenizers",
        "pyfaidx",
    }


def test_runtime_policy_requires_mps_and_an_absent_fallback_variable():
    assert validate_r5_runtime_policy(device="mps", environ={})["verified"] is True
    with pytest.raises(R5StackBindingError, match="device"):
        validate_r5_runtime_policy(device="cpu", environ={})
    with pytest.raises(R5StackBindingError, match="MPS fallback"):
        validate_r5_runtime_policy(
            device="mps", environ={"PYTORCH_ENABLE_MPS_FALLBACK": "0"}
        )


def test_freeze_normalization_changes_only_frozen_assignment_values():
    first = (
        "FROZEN_PROTOCOL_SHA256: str | None = None\n"
        "FROZEN_COUNT: int | None = None\n"
        "METHOD = 'same'\n"
    )
    second = (
        f"FROZEN_PROTOCOL_SHA256: str | None = {'a' * 64!r}\n"
        "FROZEN_COUNT: int | None = 128\n"
        "METHOD = 'same'\n"
    )
    normalized_first, names_first = normalize_freeze_pins(first)
    normalized_second, names_second = normalize_freeze_pins(second)
    assert normalized_first == normalized_second
    assert names_first == names_second == (
        "FROZEN_PROTOCOL_SHA256",
        "FROZEN_COUNT",
    )
    changed, _ = normalize_freeze_pins(second.replace("same", "different"))
    assert changed != normalized_first
    multiline, multiline_names = normalize_freeze_pins(
        "FROZEN_PAIR: tuple[str, str] | None = (\n    'a',\n    'b',\n)\nMETHOD = 1\n"
    )
    assert multiline_names == ("FROZEN_PAIR",)
    assert multiline.startswith(
        "FROZEN_PAIR: tuple[str, str] | None = <FROZEN_PIN>\nMETHOD = 1"
    )
    lhs_changed, _ = normalize_freeze_pins(first.replace(": str | None", ":str|None"))
    assert lhs_changed != normalized_first


def test_installing_all_r1c_package_pins_cannot_change_joint_stack_fingerprint(
    monkeypatch,
):
    cohort_path = (
        REPO / "src/gfmcd/splicing/acceptor_mechanism_cohort.py"
    ).resolve()
    original_read_text = Path.read_text
    source_snapshot = {}
    install_values = {"enabled": False}
    installed = {
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_LOGICAL_NAME": repr(
            "r1c_preflight.json"
        ),
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_SIZE_BYTES": "901",
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256": repr("e" * 64),
        "FROZEN_R1C_PREFLIGHT_MANIFEST_LOGICAL_NAME": repr(
            "r1c_preflight.manifest.json"
        ),
        "FROZEN_R1C_PREFLIGHT_MANIFEST_SIZE_BYTES": "902",
        "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256": repr("f" * 64),
        "FROZEN_R1C_TARGET_ARTIFACT_LOGICAL_NAME": repr("r1c_target.json"),
        "FROZEN_R1C_TARGET_ARTIFACT_SIZE_BYTES": "1001",
        "FROZEN_R1C_TARGET_ARTIFACT_SHA256": repr("a" * 64),
        "FROZEN_R1C_TARGET_MANIFEST_LOGICAL_NAME": repr(
            "r1c_target.manifest.json"
        ),
        "FROZEN_R1C_TARGET_MANIFEST_SIZE_BYTES": "1002",
        "FROZEN_R1C_TARGET_MANIFEST_SHA256": repr("b" * 64),
        "FROZEN_R1C_TARGET_ATTEMPT_LOGICAL_NAME": repr(
            "r1c_target_attempt.json"
        ),
        "FROZEN_R1C_TARGET_ATTEMPT_SIZE_BYTES": "1003",
        "FROZEN_R1C_TARGET_ATTEMPT_SHA256": repr("c" * 64),
        "FROZEN_R1C_TARGET_COMPLETION_LOGICAL_NAME": repr(
            "r1c_target_attempt.completion.json"
        ),
        "FROZEN_R1C_TARGET_COMPLETION_SIZE_BYTES": "1004",
        "FROZEN_R1C_TARGET_COMPLETION_SHA256": repr("d" * 64),
    }

    def read_with_installed_pins(path, *args, **kwargs):
        resolved = path.resolve()
        if resolved not in source_snapshot:
            source_snapshot[resolved] = original_read_text(path, *args, **kwargs)
        text = source_snapshot[resolved]
        if not install_values["enabled"] or resolved != cohort_path:
            return text
        for name, value in installed.items():
            text = text.replace(
                f"{name}: str | None = None", f"{name}: str | None = {value}"
            )
            text = text.replace(
                f"{name}: int | None = None", f"{name}: int | None = {value}"
            )
        return text

    monkeypatch.setattr(Path, "read_text", read_with_installed_pins)
    baseline = build_r5_stack_binding(repo=REPO)
    install_values["enabled"] = True
    after_install = build_r5_stack_binding(repo=REPO)
    assert r5_stack_fingerprint(after_install) == r5_stack_fingerprint(baseline)
    assert after_install == baseline


def test_installing_future_r1c_package_pins_cannot_change_r5a_instrument_fingerprint(
    monkeypatch,
):
    r5a = _load("r5a_temporal_pin_binding_test", R5_RUNNER_PATHS[0])
    cohort_path = (
        REPO / "src/gfmcd/splicing/acceptor_mechanism_cohort.py"
    ).resolve()
    source = cohort_path.read_text()
    state = {"text": source}
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    original_sha256_file = r5a.sha256_file

    def read_text(path, *args, **kwargs):
        if path.resolve() == cohort_path:
            return state["text"]
        return original_read_text(path, *args, **kwargs)

    def read_bytes(path, *args, **kwargs):
        if path.resolve() == cohort_path:
            return state["text"].encode("utf-8")
        return original_read_bytes(path, *args, **kwargs)

    def sha256_file(path):
        if Path(path).resolve() == cohort_path:
            return hashlib.sha256(state["text"].encode("utf-8")).hexdigest()
        return original_sha256_file(path)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(r5a, "sha256_file", sha256_file)
    baseline = r5a.canonical_sha256(r5a.instrument_binding_payload())
    state["text"] = _source_with_installed_r1c_package_pins(state["text"])
    installed_fingerprint = r5a.canonical_sha256(r5a.instrument_binding_payload())
    assert installed_fingerprint == baseline

    state["text"] += "\nR5_METHOD_SEMANTICS_CHANGED = True\n"
    assert r5a.canonical_sha256(r5a.instrument_binding_payload()) != baseline


def test_installing_future_r1c_package_pins_cannot_change_r5c_instrument_fingerprint(
    monkeypatch,
):
    r5c = _load("r5c_temporal_pin_binding_test", R5_RUNNER_PATHS[2])
    cohort_path = (
        REPO / "src/gfmcd/splicing/acceptor_mechanism_cohort.py"
    ).resolve()
    state = {"text": cohort_path.read_text()}
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes

    def read_text(path, *args, **kwargs):
        if path.resolve() == cohort_path:
            return state["text"]
        return original_read_text(path, *args, **kwargs)

    def read_bytes(path, *args, **kwargs):
        if path.resolve() == cohort_path:
            return state["text"].encode("utf-8")
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    baseline = r5c.canonical_sha256(r5c.instrument_binding_payload())
    state["text"] = _source_with_installed_r1c_package_pins(state["text"])
    installed_fingerprint = r5c.canonical_sha256(r5c.instrument_binding_payload())
    assert installed_fingerprint == baseline

    state["text"] += "\nR5_METHOD_SEMANTICS_CHANGED = True\n"
    assert r5c.canonical_sha256(r5c.instrument_binding_payload()) != baseline


def test_normalized_source_record_is_invariant_to_pin_value_and_exact_size(tmp_path):
    runner = tmp_path / "runner.py"
    runner.write_text("FROZEN_HASH: str | None = None\nMETHOD = 'fixed'\n")
    first = normalized_source_record(runner, repo=tmp_path)
    runner.write_text(
        f"FROZEN_HASH: str | None = {'a' * 64!r}\nMETHOD = 'fixed'\n"
    )
    second = normalized_source_record(runner, repo=tmp_path)
    assert first == second
    assert "normalized_size_bytes" in first
    assert "size_bytes" not in first


@pytest.mark.parametrize(
    ("source", "message"),
    (
        (
            "FROZEN_HASH: str | None = None\nFROZEN_HASH = 'later'\n",
            "assigned more than once",
        ),
        (
            "FROZEN_HASH: str | None = resolve_hash_at_runtime()\n",
            "RHS is not literal",
        ),
    ),
)
def test_direct_source_pin_normalization_fails_closed_on_duplicate_or_nonliteral_rhs(
    tmp_path,
    source,
    message,
):
    path = tmp_path / "pins.py"
    path.write_text(source)
    with pytest.raises(R5StackBindingError, match=message):
        normalized_literal_pin_source_sha256(
            path,
            permitted_pin_names=("FROZEN_HASH",),
        )


def test_shared_r1c_pin_allowlist_matches_the_complete_cohort_contract():
    source = (REPO / "src/gfmcd/splicing/acceptor_mechanism_cohort.py").read_text()
    declared = {
        line.split(":", 1)[0]
        for line in source.splitlines()
        if line.startswith("FROZEN_R1C_")
    }
    assert declared == set(SHARED_R1C_PACKAGE_PIN_NAMES)


def test_r5a_r5b_and_r5c_emit_one_identical_joint_stack_identity():
    r5a = _load("r5a_joint_stack_test", R5_RUNNER_PATHS[0])
    r5b = _load("r5b_joint_stack_test", R5_RUNNER_PATHS[1])
    r5c = _load("r5c_joint_stack_test", R5_RUNNER_PATHS[2])
    a = r5a.instrument_binding_payload()
    b = r5b.instrument_binding_payload()
    c = r5c.instrument_binding_payload()
    assert a["r5_stack_binding"] == b["r5_stack_binding"] == c["r5_stack_binding"]
    assert (
        a["r5_stack_fingerprint_sha256"]
        == b["r5_stack_fingerprint_sha256"]
        == c["r5_stack_fingerprint_sha256"]
    )
    assert a["r5_stack_fingerprint_sha256"] == r5_stack_fingerprint(
        a["r5_stack_binding"]
    )


def test_serialized_joint_stack_rejects_payload_and_fingerprint_tampering():
    binding = build_r5_stack_binding(repo=REPO)
    fingerprint = r5_stack_fingerprint(binding)
    record = {
        "r5_stack_binding": binding,
        "r5_stack_fingerprint_sha256": fingerprint,
    }
    assert validate_serialized_r5_stack(record, current=binding)["verified"]
    changed = copy.deepcopy(binding)
    changed["contract"]["masked_donor_endpoint_constructed"] = True
    with pytest.raises(R5StackBindingError):
        validate_r5_stack_binding(changed, fingerprint, current=binding)
    with pytest.raises(R5StackBindingError):
        validate_serialized_r5_stack(
            record | {"r5_stack_fingerprint_sha256": "0" * 64},
            current=binding,
        )


def _r3c_target_health_record():
    suppressed = set(R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS)
    authorized = [name for name in R3C_PATH_NAMES if name not in suppressed]
    paths = {
        name: {
            "classification": "HEALTH_FAILED" if name in suppressed else "SUPPORTED",
            "health": {
                "ok": name not in suppressed,
                "preflight_health_ok": name not in suppressed,
                "current_health_ok": True,
                "authorization_rule": R3C_PATH_HEALTH_RULE,
            },
        }
        for name in R3C_PATH_NAMES
    }
    return {
        "instrument_revision": R3C_INSTRUMENT_REVISION,
        "paths": paths,
        "path_health_authorization": {
            "schema": R3C_TARGET_HEALTH_SCHEMA,
            "rule": R3C_TARGET_HEALTH_RULE,
            "can_only_shrink_from_preflight": True,
            "preflight_authorized_paths": list(authorized),
            "preflight_suppressed_paths": list(
                R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS
            ),
            "current_healthy_paths": list(R3C_PATH_NAMES),
            "effective_authorized_paths": list(authorized),
            "effective_suppressed_paths": list(
                R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS
            ),
        },
        "family": {
            "production_path_preflight": {
                "all_health_eligible_path_plants_powered": True,
                "health_eligible_paths": list(R3C_PATH_NAMES),
            }
        },
    }


def test_r3c_target_health_contract_requires_revision_and_monotone_intersection():
    record = _r3c_target_health_record()
    validated = validate_r3c_target_health_contract(record)
    assert validated["verified"] is True
    assert validated["scientific_entry_authorized"] is False
    disagreement = copy.deepcopy(record)
    disagreement["paths"]["L3H1_reused_component"][
        "classification"
    ] = "INCONCLUSIVE_METHOD_DISAGREEMENT"
    with pytest.raises(R5StackBindingError, match="not health-authorized support"):
        validate_r3c_target_health_contract(disagreement)
    changed = copy.deepcopy(record)
    changed["instrument_revision"] = "superseded"
    with pytest.raises(R5StackBindingError):
        validate_r3c_target_health_contract(changed)
    changed = copy.deepcopy(record)
    changed["path_health_authorization"]["current_healthy_paths"].remove(
        "L3H1_reused_component"
    )
    with pytest.raises(R5StackBindingError):
        validate_r3c_target_health_contract(changed)
    revived = _r3c_target_health_record()
    for field in ("preflight_authorized_paths", "effective_authorized_paths"):
        revived["path_health_authorization"][field].insert(
            1, "L2H15_mean_component"
        )
    for field in ("preflight_suppressed_paths", "effective_suppressed_paths"):
        revived["path_health_authorization"][field] = []
    revived["paths"]["L2H15_mean_component"] = {
        "classification": "SUPPORTED",
        "health": {
            "ok": True,
            "preflight_health_ok": True,
            "current_health_ok": True,
            "authorization_rule": R3C_PATH_HEALTH_RULE,
        },
    }
    with pytest.raises(R5StackBindingError, match="revived"):
        validate_r3c_target_health_contract(revived)


def test_r3c_target_manifest_contract_cross_checks_artifact_context():
    record = {
        "instrument_fingerprint_sha256": "a" * 64,
        "construction": {"ordered_rows_sha256": "b" * 64},
        "input_snapshot_sha256": "c" * 64,
        "git_authorization_identity": {"commit": "d" * 40, "dirty": False},
    }
    manifest = {
        "experiment_id": "splicebert_acceptor_coupling:r3c_acceptor_head_confirmation",
        "status": "CONFIRMATORY",
        "git_dirty": False,
        "git_commit": "d" * 40,
        "config": {
            "stage": "target",
            "gene_bucket": 1,
            "instrument_fingerprint_sha256": "a" * 64,
            "construction_sha256": "b" * 64,
            "input_snapshot_sha256": "c" * 64,
        },
    }
    validated = validate_r3c_target_manifest_contract(record, manifest)
    assert validated["verified"] is True
    assert validated["scientific_entry_authorized"] is False
    manifest["config"]["stage"] = "preflight"
    with pytest.raises(R5StackBindingError):
        validate_r3c_target_manifest_contract(record, manifest)
