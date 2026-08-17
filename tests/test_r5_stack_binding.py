from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest

from gfmcd.splicing.r5_stack_binding import (
    R3C_RUNNER_PATH,
    R3C_ENTRY_SCHEMA,
    R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS,
    R3C_INSTRUMENT_REVISION,
    R3C_PATH_HEALTH_RULE,
    R3C_PATH_NAMES,
    R3C_TARGET_HEALTH_RULE,
    R3C_TARGET_HEALTH_SCHEMA,
    R5_PROTOCOL_PATHS,
    R5_RUNNER_PATHS,
    R5StackBindingError,
    build_r5_stack_binding,
    normalize_freeze_pins,
    normalized_source_record,
    r5_stack_fingerprint,
    validate_r5_stack_binding,
    validate_r3c_target_health_contract,
    validate_r3c_target_manifest_contract,
    validate_serialized_r5_stack,
)


REPO = Path(__file__).resolve().parents[1]


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
    assert binding["runner_roots"] == list(R5_RUNNER_PATHS)
    assert binding["dynamic_entry_runner"] == R3C_RUNNER_PATH
    assert set(binding["protocols"]) == set(R5_PROTOCOL_PATHS)
    closure = binding["local_python_closure"]["files"]
    assert set(R5_RUNNER_PATHS) < set(closure)
    assert R3C_RUNNER_PATH in closure
    assert "src/gfmcd/splicing/r5_stack_binding.py" in closure
    assert "src/gfmcd/adapters/splicebert.py" in closure
    assert "src/gfmcd/circuits/hf_bert.py" in closure
    assert "src/gfmcd/splicing/paired_component.py" in closure
    assert binding["schema_registry_sha256"]
    assert binding["executable_registries_sha256"]
    assert set(binding["executable_registries"]) == set(R5_RUNNER_PATHS)
    assert "gfmcd.r5.entry_contract/1" in binding["schema_registry"]
    assert binding["contract"]["handoff_schemas"]["r3c_entry"] == R3C_ENTRY_SCHEMA
    assert set(binding["runtime_identity"]["packages"]) == {
        "numpy",
        "scipy",
        "torch",
        "transformers",
        "tokenizers",
        "pyfaidx",
    }


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
    assert validate_r3c_target_health_contract(record)["verified"] is True
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
    assert validate_r3c_target_manifest_contract(record, manifest)["verified"] is True
    manifest["config"]["stage"] = "preflight"
    with pytest.raises(R5StackBindingError):
        validate_r3c_target_manifest_contract(record, manifest)
