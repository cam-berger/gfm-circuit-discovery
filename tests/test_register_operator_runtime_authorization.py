"""Real authorization functions must stop CC before constructing any request."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from gfmcd.provenance import sha256_file
from gfmcd.splicing.register_operator_io import (
    manifest_path, verify_artifact_manifest, verify_freeze, write_artifact,
)


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "register_operator_runtime_under_test",
    REPO / "scripts/splicebert_register_operator_replication/run_model.py",
)
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)
AUDITOR = "scripts/splicebert_register_operator_replication/audit_fidelity.py"


def environment(tmp_path, *, omit=(), audit_updates=None, native_pass=True,
                receipt_updates=None, runtime_updates=None):
    """All receipts are synthetic files, written by the actual immutable writer."""
    source = tmp_path / "source.py"
    source.write_text("frozen synthetic source\n")
    names = ("smoke", "calibration", "fidelity", "fidelity_audit", "native", "cells",
             "fidelity_attempt", "native_attempt", "cells_attempt")
    outputs = {name: str(tmp_path / (name + ".json")) for name in names}
    freeze = {
        "status": "FROZEN", "outputs": outputs,
        "files": {str(source): sha256_file(source), AUDITOR: sha256_file(REPO / AUDITOR)},
        "runtime": {"python":runtime.platform.python_version(),
                    "versions":{key:runtime.importlib.metadata.version(key)
                                for key in ("torch","transformers","numpy","scipy","pyfaidx")},
                    **(runtime_updates or {})},
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze))
    freeze_sha = sha256_file(freeze_path)
    updates = receipt_updates or {}

    def receipt(name, value, extra_inputs=()):
        if name in omit:
            return
        value = {"freeze_sha256": freeze_sha, **value, **updates.get(name, {})}
        write_artifact(outputs[name], value, inputs=[source, freeze_path, *extra_inputs],
                       experiment="synthetic-runtime-authorization-test:" + name)

    receipt("smoke", {"status": "PASS"})
    receipt("calibration", {"status": "PASS"})
    receipt("fidelity", {"status": "COMPLETE", "stage": "fidelity"})
    receipt("fidelity_audit", {
        "status": "PASS", "passed": True, "authorized_for_cc": True,
        "auditor_sha256": freeze["files"][AUDITOR],
        "fidelity_sha256": sha256_file(outputs["fidelity"]),
        **(audit_updates or {}),
    }, extra_inputs=[outputs["fidelity"], REPO / AUDITOR])
    receipt("native", {"status": "COMPLETE", "native_preflight": {"passed": native_pass}})
    return freeze, freeze_sha, freeze_path, source


@pytest.mark.parametrize("stage", ["fidelity", "native", "cells"])
@pytest.mark.parametrize("missing", ["smoke", "calibration"])
def test_no_stage_starts_without_both_model_free_pass_receipts(tmp_path, stage, missing):
    freeze, digest, _, _ = environment(tmp_path, omit=[missing])
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime.verify_stage_authorization(freeze, digest, stage)


@pytest.mark.parametrize("stage", ["native", "cells"])
def test_no_cc_stage_starts_without_full_fidelity_audit(tmp_path, stage):
    freeze, digest, _, _ = environment(tmp_path, omit=["fidelity_audit"])
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime.verify_stage_authorization(freeze, digest, stage)


@pytest.mark.parametrize("updates", [
    {"authorized_for_cc": False}, {"authorized_for_cc": 1}, {"passed": False},
    {"status": "FAIL"}, {"freeze_sha256": "stale-freeze"},
    {"auditor_sha256": "wrong-auditor"}, {"fidelity_sha256": "stale-raw"},
])
def test_status_alone_never_authorizes_cc(tmp_path, updates):
    freeze, digest, _, _ = environment(tmp_path, audit_updates=updates)
    with pytest.raises(ValueError):
        runtime.verify_stage_authorization(freeze, digest, "native")


@pytest.mark.parametrize("receipt_name", ["smoke", "calibration"])
@pytest.mark.parametrize("updates", [{"status": "FAIL"}, {"freeze_sha256": "stale"}])
def test_failed_or_wrong_freeze_preflight_blocks_cc(tmp_path, receipt_name, updates):
    freeze, digest, _, _ = environment(tmp_path, receipt_updates={receipt_name: updates})
    with pytest.raises(ValueError):
        runtime.verify_stage_authorization(freeze, digest, "native")


def test_failed_native_preflight_blocks_crossed_but_not_native_stage(tmp_path):
    freeze, digest, _, _ = environment(tmp_path, native_pass=False)
    assert len(runtime.verify_stage_authorization(freeze, digest, "native")) == 3
    with pytest.raises(ValueError, match="native CC"):
        runtime.verify_stage_authorization(freeze, digest, "cells")


def test_all_required_pass_receipts_authorize_each_stage(tmp_path):
    freeze, digest, _, _ = environment(tmp_path)
    assert len(runtime.verify_stage_authorization(freeze, digest, "fidelity")) == 2
    assert len(runtime.verify_stage_authorization(freeze, digest, "native")) == 3
    assert len(runtime.verify_stage_authorization(freeze, digest, "cells")) == 4


@pytest.mark.parametrize("fault", ["raw", "audit", "manifest", "source"])
def test_actual_manifest_verification_rejects_changed_bytes(tmp_path, fault):
    freeze, digest, _, source = environment(tmp_path)
    paths = freeze["outputs"]
    if fault == "source":
        source.write_text("changed synthetic source\n")
    elif fault == "manifest":
        path = manifest_path(paths["fidelity_audit"])
        data = json.loads(path.read_text())
        data["outputs"][0]["sha256"] = "stale-output"
        path.write_text(json.dumps(data))
    else:
        path = Path(paths["fidelity" if fault == "raw" else "fidelity_audit"])
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        runtime.verify_stage_authorization(freeze, digest, "native")


def test_frozen_source_change_is_rejected_before_receipts(tmp_path):
    _, _, freeze_path, source = environment(tmp_path)
    verify_freeze(freeze_path, REPO)
    source.write_text("changed source\n")
    with pytest.raises(ValueError, match="frozen file changed"):
        verify_freeze(freeze_path, REPO)


@pytest.mark.parametrize("changed", ["python","torch","transformers","numpy","scipy","pyfaidx"])
def test_installed_runtime_mismatch_stops_before_request_construction(tmp_path,monkeypatch,changed):
    profile = {"python":"0.0.0"} if changed == "python" else {"versions":{changed:"0.0.0"}}
    _, _, freeze_path, _ = environment(tmp_path,omit=["native"],runtime_updates=profile)
    monkeypatch.setattr(sys,"argv",["run_model.py","--freeze",str(freeze_path),"--stage","native"])
    def forbidden(*args,**kwargs):
        pytest.fail("request construction was reached with an unapproved runtime")
    monkeypatch.setattr(runtime,"requests_for_blocks",forbidden)
    monkeypatch.setattr(runtime,"Genome",forbidden)
    with pytest.raises(ValueError,match="installed runtime"):
        runtime.main()


@pytest.mark.parametrize("stage", ["native", "cells"])
def test_main_cannot_construct_cc_requests_before_audit_pass(tmp_path, monkeypatch, stage):
    _, _, freeze_path, _ = environment(tmp_path, omit=["native"] if stage == "native" else [],
                                      audit_updates={"authorized_for_cc": False})
    monkeypatch.setattr(sys, "argv", ["run_model.py", "--freeze", str(freeze_path), "--stage", stage])
    def forbidden(*args, **kwargs):
        pytest.fail("CC request planning was reached before authorization")
    monkeypatch.setattr(runtime, "requests_for_blocks", forbidden)
    monkeypatch.setattr(runtime, "Genome", forbidden)
    with pytest.raises(ValueError, match="fidelity audit"):
        runtime.main()


@pytest.mark.parametrize("which", ["native", "native_attempt"])
def test_existing_output_or_attempt_cannot_be_overwritten(tmp_path, monkeypatch, which):
    freeze, _, freeze_path, source = environment(tmp_path)
    path = Path(freeze["outputs"][which])
    if not path.exists():
        write_artifact(path, {"status": "STARTED"}, inputs=[source], experiment="synthetic-attempt")
    # For the attempt branch the native result destination must be absent.
    if which == "native_attempt":
        Path(freeze["outputs"]["native"]).unlink()
        manifest_path(freeze["outputs"]["native"]).unlink()
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["run_model.py", "--freeze", str(freeze_path), "--stage", "native"])
    with pytest.raises(FileExistsError):
        runtime.main()
    assert path.read_bytes() == before
    verify_artifact_manifest(path)
