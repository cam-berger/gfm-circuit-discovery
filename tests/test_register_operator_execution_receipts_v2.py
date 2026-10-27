"""Fail-closed A1 execution receipts with tiny files and no calibration/model work."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from gfmcd.provenance import sha256_file
from gfmcd.splicing import register_operator_io_v2 as io_v2
from gfmcd.splicing.register_operator_io import (
    manifest_path, read_json, verify_artifact_manifest, write_artifact,
)
from gfmcd.splicing.register_operator_statistics_v2 import METHOD

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts/splicebert_register_operator_replication"
sys.path.insert(0, str(SCRIPTS))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


smoke = load("execution_receipts_smoke_v2", "run_smoke_v2.py")
freezer = load("execution_receipts_freeze_v2", "freeze_protocol_v2.py")
recorder = load("execution_receipts_recorder_v2", "record_result_v2.py")


@pytest.fixture
def tiny_repo(tmp_path, monkeypatch):
    """Run real version/hash verifiers over a small synthetic parent instead of hg38."""
    source = tmp_path / "tests/test_frozen_fixture.py"
    source.parent.mkdir()
    source.write_text("# frozen synthetic test source\n")
    parent = tmp_path / "studies/splicebert_register_operator_replication/protocol_freeze_v1.json"
    parent.parent.mkdir(parents=True)
    parent.write_text(json.dumps({"status": "FROZEN", "files": {str(source): sha256_file(source)}}))
    base_sha = sha256_file(parent)
    monkeypatch.setattr(io_v2, "BASE_FREEZE_SHA256", base_sha)
    monkeypatch.setattr(freezer, "BASE_FREEZE_SHA256", base_sha)
    for module in (smoke, freezer, recorder):
        monkeypatch.setattr(module, "REPO", tmp_path)
    return {"root": tmp_path, "source": source, "parent": parent, "base_sha": base_sha}


def freeze_file(env, outputs, *, extra_files=()):
    source = env["source"]
    files = {str(source.relative_to(env["root"])): sha256_file(source)}
    files.update({str(path): sha256_file(path) for path in extra_files})
    value = {"status": "FROZEN_V2", "schema": "gfmcd.register_operator.freeze.v2",
        "inference_method": METHOD, "base_freeze_path": str(env["parent"]),
        "base_freeze_sha256": env["base_sha"], "files": files,
        "tests": [str(source.relative_to(env["root"]))], "outputs": outputs}
    path = env["root"] / "synthetic_v2_freeze.json"
    path.write_text(json.dumps(value))
    return path, value


def artifact(path, value, inputs):
    write_artifact(path, value, inputs=list(inputs), experiment="synthetic-a1-receipt-test")
    return Path(path)


@pytest.mark.parametrize("fault", ["mechanics_exception", "test_process_exception", "source_changed", "freeze_changed"])
def test_smoke_exceptions_and_changed_bindings_write_immutable_failures(tiny_repo, monkeypatch, fault):
    out = tiny_repo["root"] / "smoke_v2.json"
    fp, freeze = freeze_file(tiny_repo, {"smoke": str(out)})
    source_before = tiny_repo["source"].read_bytes()
    freeze_before = fp.read_bytes()
    initial_sha = sha256_file(fp)
    observed_test_commands = []
    original_run = smoke.subprocess.run

    def subprocess_call(command, *args, **kwargs):
        if command[1:4] == ["-m", "pytest", "-q"]:
            observed_test_commands.append(command)
            if fault == "test_process_exception":
                raise OSError("synthetic subprocess failure")
            return SimpleNamespace(returncode=0, stdout="synthetic tests passed", stderr="")
        return original_run(command, *args, **kwargs)

    def mechanics():
        if fault == "mechanics_exception":
            raise ValueError("synthetic mechanics failure")
        if fault == "source_changed":
            tiny_repo["source"].write_text("# changed while mechanics ran\n")
        if fault == "freeze_changed":
            fp.write_text(fp.read_text() + " ")
        return {"status": "SMOKE_PASS", "failures": []}

    monkeypatch.setattr(smoke.subprocess, "run", subprocess_call)
    monkeypatch.setattr(smoke, "run_smoke", mechanics)
    monkeypatch.setattr(sys, "argv", ["run_smoke_v2.py", "--freeze", str(fp)])
    with pytest.raises(SystemExit) as stopped:
        smoke.main()
    assert stopped.value.code == 1
    assert observed_test_commands[0][4:] == freeze["tests"]
    assert len(observed_test_commands) == 1
    report = read_json(out)
    assert report["status"] == "FAIL" and report["inference_method"] == METHOD
    assert report["freeze_sha256"] == initial_sha
    assert report["model_loaded"] is False and report["new_cc_outputs"] == 0
    expected_stage = "mechanics" if fault.endswith("exception") else "post_run_bindings"
    assert any(row["stage"] == expected_stage for row in report["failures"])
    assert report["completed_at_utc"] >= report["created_at_utc"]
    assert verify_artifact_manifest(out)["status"] == "FAIL"
    before = out.read_bytes(), manifest_path(out).read_bytes()
    # Restore only temporary fixture inputs, then prove that FAIL cannot be replaced by PASS.
    tiny_repo["source"].write_bytes(source_before)
    fp.write_bytes(freeze_before)
    monkeypatch.setattr(smoke, "run_smoke", lambda: {"status": "SMOKE_PASS"})
    with pytest.raises(FileExistsError):
        smoke.main()
    assert (out.read_bytes(), manifest_path(out).read_bytes()) == before
    assert len(observed_test_commands) == 1


def prerequisite(tiny_repo, *, kind, updates=None, omit_source=False):
    auditor = SCRIPTS / "audit_repair_design.py"
    source = tiny_repo["source"]
    value = {"status": "PASS", "passed": True, "inference_method": METHOD,
        "base_freeze_sha256": tiny_repo["base_sha"], "fresh_streams_opened": False,
        "point_and_pseudo_memorials_passed": 1768, "acceptance": {"all_passed": True},
        "counts": {"development": 256, "null": 1000, "plants": 512}}
    if kind == "design":
        value.update(auditor_sha256=sha256_file(auditor), files={str(source): sha256_file(source)})
    value.update(updates or {})
    path = tiny_repo["root"] / (kind + ".json")
    inputs = [tiny_repo["parent"], auditor] + ([] if omit_source else [source])
    artifact(path, value, inputs)
    return path, auditor


@pytest.mark.parametrize("kind", ["design", "development"])
@pytest.mark.parametrize("updates", [
    {"status": "FAIL"}, {"passed": False}, {"passed": 1},
    {"inference_method": "original_v1"}, {"base_freeze_sha256": "stale-parent"},
])
def test_freeze_prerequisite_rejects_failed_or_wrong_instrument_receipts(tiny_repo, kind, updates):
    path, auditor = prerequisite(tiny_repo, kind=kind, updates=updates)
    with pytest.raises(ValueError, match="A1 prerequisite failed"):
        freezer.validate_receipt(path, auditor=auditor if kind == "design" else None)


@pytest.mark.parametrize("kind", ["design", "development"])
def test_freeze_rejects_changed_actual_prerequisite_inputs(tiny_repo, kind):
    path, auditor = prerequisite(tiny_repo, kind=kind)
    tiny_repo["source"].write_text("stale source after receipt\n")
    with pytest.raises(ValueError, match="input has changed"):
        freezer.validate_receipt(path, auditor=auditor if kind == "design" else None)


@pytest.mark.parametrize("kind", ["design", "development"])
def test_freeze_requires_pass_in_the_manifest_as_well_as_body(tiny_repo, kind):
    path, auditor = prerequisite(tiny_repo, kind=kind)
    mp = manifest_path(path); manifest = read_json(mp); manifest["status"] = "FAIL"
    mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="A1 prerequisite failed"):
        freezer.validate_receipt(path, auditor=auditor if kind == "design" else None)


@pytest.mark.parametrize("fault", ["auditor_hash", "reviewed_hash", "missing_reviewed_input"])
def test_freeze_design_receipt_binds_current_auditor_and_reviewed_files(tiny_repo, fault):
    updates = ({"auditor_sha256": "stale"} if fault == "auditor_hash" else
               {"files": {str(tiny_repo["source"]): "stale"}} if fault == "reviewed_hash" else {})
    path, auditor = prerequisite(tiny_repo, kind="design", updates=updates,
                                omit_source=fault == "missing_reviewed_input")
    with pytest.raises(ValueError):
        freezer.validate_receipt(path, auditor=auditor)


@pytest.mark.parametrize("updates", [
    {"fresh_streams_opened": True}, {"fresh_streams_opened": 0},
    {"point_and_pseudo_memorials_passed": 1767}, {"acceptance": {"all_passed": False}},
    {"acceptance": {"all_passed": 1}}, {"counts": {"development": 256, "null": 999, "plants": 512}},
])
def test_freeze_main_refuses_incomplete_or_exposed_development_before_building_package(tiny_repo, monkeypatch, updates):
    design, _ = prerequisite(tiny_repo, kind="design")
    development, _ = prerequisite(tiny_repo, kind="development", updates=updates)
    out = tiny_repo["root"] / "never_frozen.json"
    def forbidden(*args, **kwargs):
        pytest.fail("source collection reached despite invalid development prerequisite")
    monkeypatch.setattr(freezer, "source_closure", forbidden)
    monkeypatch.setattr(sys, "argv", ["freeze_protocol_v2.py", "--design-audit", str(design),
        "--development", str(development), "--out", str(out)])
    with pytest.raises(ValueError, match="complete consumed-data"):
        freezer.main()
    assert not out.exists() and not manifest_path(out).exists()


def final_fixture(env, *, updates=None, omit_audit=False, omit_input=None,
                  model_files=False, omit_model_binding=None):
    auditor = env["root"] / "scripts/splicebert_register_operator_replication/audit_results_v2.py"
    auditor.parent.mkdir(parents=True)
    auditor.write_text("# synthetic frozen result auditor\n")
    names = ("result_audit", "final", "smoke", "calibration", "calibration_audit", "fidelity", "fidelity_audit",
             "native", "cells", "analysis", "measurement_audit")
    outputs = {name: str(env["root"] / (name + "_v2.json")) for name in names}
    fp, _ = freeze_file(env, outputs, extra_files=[auditor])
    audit = {"schema": "gfmcd.register_operator.result_audit.v2", "status": "PASS", "passed": True,
        "authorized_for_final_result": True, "inference_method": METHOD, "freeze_sha256": sha256_file(fp),
        "auditor_sha256": sha256_file(auditor), "model_evidence_complete": False,
        "classification": "INSTRUMENT_UNRESOLVED", "gates": {"construction": False, "independent_audit": None},
        "failed_gates": ["construction"], "unrun_gates": ["independent_audit"], **(updates or {})}
    model_inputs = []
    if model_files:
        for name in ("native", "cells", "analysis", "measurement_audit"):
            artifact(outputs[name], {"status": "SYNTHETIC_TEST_FIXTURE"}, [fp, auditor])
            if name != omit_model_binding:
                model_inputs.append(Path(outputs[name]))
    if not omit_audit:
        inputs = [p for name, p in (("freeze", fp), ("auditor", auditor)) if name != omit_input]
        artifact(outputs["result_audit"], audit, inputs + model_inputs)
    return fp, outputs


def invoke_final(fp, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["record_result_v2.py", "--freeze", str(fp)])
    recorder.main()


def test_final_recorder_accepts_an_audited_stop_and_preserves_it_immutably(tiny_repo, monkeypatch):
    fp, outputs = final_fixture(tiny_repo)
    invoke_final(fp, monkeypatch)
    final = Path(outputs["final"])
    record = read_json(final)
    assert record["classification"] == "INSTRUMENT_UNRESOLVED"
    assert record["model_evidence_complete"] is False and record["model_endpoints"] is None
    assert all(row["status"] == "NOT_RUN" for row in record["stages"].values())
    assert verify_artifact_manifest(final)["status"] == "INSTRUMENT_UNRESOLVED"
    before = final.read_bytes(), manifest_path(final).read_bytes()
    with pytest.raises(FileExistsError):
        invoke_final(fp, monkeypatch)
    assert (final.read_bytes(), manifest_path(final).read_bytes()) == before


def test_final_recorder_rejects_missing_independent_audit(tiny_repo, monkeypatch):
    fp, outputs = final_fixture(tiny_repo, omit_audit=True)
    with pytest.raises(FileNotFoundError):
        invoke_final(fp, monkeypatch)
    assert not Path(outputs["final"]).exists()


@pytest.mark.parametrize("updates", [
    {"status": "FAIL"}, {"passed": False}, {"passed": 1},
    {"authorized_for_final_result": False}, {"authorized_for_final_result": 1},
    {"schema": "gfmcd.register_operator.result_audit.v1"}, {"inference_method": "original_v1"},
    {"freeze_sha256": "stale"}, {"auditor_sha256": "stale"},
    {"classification": "BOUNDED_NULL"},
])
def test_final_recorder_rejects_incorrect_audit_receipts(tiny_repo, monkeypatch, updates):
    fp, outputs = final_fixture(tiny_repo, updates=updates)
    with pytest.raises(ValueError):
        invoke_final(fp, monkeypatch)
    assert not Path(outputs["final"]).exists()


@pytest.mark.parametrize("missing", ["freeze", "auditor"])
def test_final_recorder_requires_explicit_audit_manifest_input_bindings(tiny_repo, monkeypatch, missing):
    fp, outputs = final_fixture(tiny_repo, omit_input=missing)
    with pytest.raises(ValueError, match="does not bind"):
        invoke_final(fp, monkeypatch)
    assert not Path(outputs["final"]).exists()


def test_final_recorder_rejects_audit_manifest_failure(tiny_repo, monkeypatch):
    fp, outputs = final_fixture(tiny_repo)
    mp = manifest_path(outputs["result_audit"]); manifest = read_json(mp); manifest["status"] = "FAIL"
    mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        invoke_final(fp, monkeypatch)
    assert not Path(outputs["final"]).exists()


def test_complete_model_flag_cannot_replace_absent_model_evidence(tiny_repo, monkeypatch):
    fp, outputs = final_fixture(tiny_repo, updates={"model_evidence_complete": True, "classification": "BOUNDED_NULL"})
    with pytest.raises((ValueError, FileNotFoundError)):
        invoke_final(fp, monkeypatch)
    assert not Path(outputs["final"]).exists()


@pytest.mark.parametrize("name", ["native", "cells", "analysis", "measurement_audit"])
def test_complete_final_requires_every_model_artifact_bound_by_audit(tiny_repo, monkeypatch, name):
    fp, outputs = final_fixture(tiny_repo, model_files=True, omit_model_binding=name,
        updates={"model_evidence_complete": True, "classification": "BOUNDED_NULL", "recomputed": {}})
    with pytest.raises(ValueError, match="does not bind"):
        invoke_final(fp, monkeypatch)
    assert not Path(outputs["final"]).exists()


def test_complete_final_requires_reconstructed_endpoints(tiny_repo, monkeypatch):
    fp, outputs = final_fixture(tiny_repo, model_files=True,
        updates={"model_evidence_complete": True, "classification": "BOUNDED_NULL"})
    with pytest.raises(ValueError, match="reconstructed endpoints"):
        invoke_final(fp, monkeypatch)
    assert not Path(outputs["final"]).exists()
