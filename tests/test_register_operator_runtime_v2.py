"""Amended runtime guards and corrected native bounds, without model loading."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import numpy as np

from gfmcd.splicing import register_operator_io_v2 as io_v2
from gfmcd.splicing import register_operator_statistics as old_statistics
from gfmcd.splicing import register_operator_statistics_v2 as statistics

from gfmcd.provenance import sha256_file
from gfmcd.splicing.register_operator_io import (
    manifest_path, verify_artifact_manifest, verify_freeze, write_artifact,
)


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "register_operator_runtime_v2_under_test",
    REPO / "scripts/splicebert_register_operator_replication/run_model_v2.py",
)
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)
AUDITOR = "scripts/splicebert_register_operator_replication/audit_fidelity.py"
CAL_AUDITOR = "scripts/splicebert_register_operator_replication/audit_results_v2.py"


@pytest.fixture(autouse=True)
def synthetic_parent_freeze(tmp_path, monkeypatch):
    """Exercise both real version verifiers without repeatedly hashing the hg38 FASTA."""
    source = tmp_path / "parent_source.py"
    source.write_text("immutable synthetic parent")
    base = tmp_path / "parent_freeze.json"
    base.write_text(json.dumps({"status":"FROZEN","files":{str(source):sha256_file(source)}}))
    monkeypatch.setattr(io_v2,"BASE_FREEZE_SHA256",sha256_file(base))
    monkeypatch.setattr(runtime,"hardware_model",lambda:"synthetic-frozen-hardware")



def environment(tmp_path, *, omit=(), audit_updates=None, native_pass=True,
                receipt_updates=None, runtime_updates=None):
    """All receipts are synthetic files, written by the actual immutable writer."""
    source = tmp_path / "source.py"
    source.write_text("frozen synthetic source\n")
    names = ("smoke", "calibration", "calibration_audit", "fidelity", "fidelity_audit", "native", "cells",
             "fidelity_attempt", "native_attempt", "cells_attempt")
    outputs = {name: str(tmp_path / (name + "_v2.json")) for name in names}
    freeze = {
        "status": "FROZEN_V2", "schema":"gfmcd.register_operator.freeze.v2",
        "inference_method":runtime.METHOD, "outputs": outputs,
        "base_freeze_sha256":io_v2.BASE_FREEZE_SHA256,"base_freeze_path":str(tmp_path/"parent_freeze.json"),
        "calibration":{"frozen":"settings"},
        "files": {str(source): sha256_file(source), AUDITOR: sha256_file(REPO / AUDITOR),
                  CAL_AUDITOR:sha256_file(REPO/CAL_AUDITOR)},
        "runtime": {"python":runtime.platform.python_version(),
                    "platform":runtime.platform.platform(),"hardware_model":"synthetic-frozen-hardware",
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
        value = {"freeze_sha256": freeze_sha,
                 **({"inference_method":runtime.METHOD} if name != "fidelity_audit" else {}),
                 **value, **updates.get(name, {})}
        write_artifact(outputs[name], value, inputs=[source, freeze_path, *extra_inputs],
                       experiment="synthetic-runtime-authorization-test:" + name)

    receipt("smoke", {"status": "PASS"})
    receipt("calibration", {"status":"PASS","acceptance":{"all_passed":True},
                            "settings":freeze["calibration"]})
    receipt("calibration_audit", {"status":"PASS","passed":True,"calibration_passed":True,
        "authorized_for_fidelity":True,"auditor_sha256":freeze["files"][CAL_AUDITOR],
        "calibration_sha256":sha256_file(outputs["calibration"]) if "calibration" not in omit else "absent"},
        extra_inputs=([outputs["calibration"]] if "calibration" not in omit else [])+[REPO/CAL_AUDITOR])
    receipt("fidelity", {"status": "COMPLETE", "stage": "fidelity"})
    receipt("fidelity_audit", {
        "status": "PASS", "passed": True, "authorized_for_cc": True,
        "auditor_sha256": freeze["files"][AUDITOR],
        "fidelity_sha256": sha256_file(outputs["fidelity"]),
        **(audit_updates or {}),
    }, extra_inputs=[outputs["fidelity"], REPO / AUDITOR])
    receipt("native", {"schema":"gfmcd.register_operator.model.v2","status":"COMPLETE",
        "native_preflight":{"passed":native_pass,"inference_method":runtime.METHOD}})
    return freeze, freeze_sha, freeze_path, source


@pytest.mark.parametrize("stage", ["fidelity", "native", "cells"])
@pytest.mark.parametrize("missing", ["smoke", "calibration", "calibration_audit"])
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


@pytest.mark.parametrize("receipt_name", ["smoke", "calibration", "calibration_audit"])
@pytest.mark.parametrize("updates", [{"status": "FAIL"}, {"freeze_sha256": "stale"}])
def test_failed_or_wrong_freeze_preflight_blocks_cc(tmp_path, receipt_name, updates):
    freeze, digest, _, _ = environment(tmp_path, receipt_updates={receipt_name: updates})
    with pytest.raises(ValueError):
        runtime.verify_stage_authorization(freeze, digest, "native")


def test_failed_native_preflight_blocks_crossed_but_not_native_stage(tmp_path):
    freeze, digest, _, _ = environment(tmp_path, native_pass=False)
    assert len(runtime.verify_stage_authorization(freeze, digest, "native")) == 4
    with pytest.raises(ValueError, match="native CC"):
        runtime.verify_stage_authorization(freeze, digest, "cells")


def test_all_required_pass_receipts_authorize_each_stage(tmp_path):
    freeze, digest, _, _ = environment(tmp_path)
    assert len(runtime.verify_stage_authorization(freeze, digest, "fidelity")) == 3
    assert len(runtime.verify_stage_authorization(freeze, digest, "native")) == 4
    assert len(runtime.verify_stage_authorization(freeze, digest, "cells")) == 5


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
    runtime.verify_freeze(freeze_path, REPO)
    source.write_text("changed source\n")
    with pytest.raises(ValueError, match="frozen v2 file changed"):
        runtime.verify_freeze(freeze_path, REPO)


@pytest.mark.parametrize("changed", ["python","torch","transformers","numpy","scipy","pyfaidx"])
def test_installed_runtime_mismatch_stops_before_request_construction(tmp_path,monkeypatch,changed):
    profile = {"python":"0.0.0"} if changed == "python" else {"versions":{changed:"0.0.0"}}
    _, _, freeze_path, _ = environment(tmp_path,omit=["native"],runtime_updates=profile)
    monkeypatch.setattr(sys,"argv",["run_model_v2.py","--freeze",str(freeze_path),"--stage","native"])
    def forbidden(*args,**kwargs):
        pytest.fail("request construction was reached with an unapproved runtime")
    monkeypatch.setattr(runtime,"requests_for_blocks",forbidden)
    monkeypatch.setattr(runtime,"Genome",forbidden)
    with pytest.raises(ValueError,match="installed runtime"):
        runtime.main()


@pytest.mark.parametrize("changed",["platform","hardware_model"])
def test_host_profile_mismatch_stops_before_request_construction(tmp_path,monkeypatch,changed):
    _,_,path,_=environment(tmp_path,omit=["native"],runtime_updates={changed:"different-host"})
    monkeypatch.setattr(sys,"argv",["run_model_v2.py","--freeze",str(path),"--stage","native"])
    def forbidden(*args,**kwargs):
        pytest.fail("request construction reached under a different frozen host")
    monkeypatch.setattr(runtime,"requests_for_blocks",forbidden)
    monkeypatch.setattr(runtime,"Genome",forbidden)
    with pytest.raises(ValueError,match="frozen profile"):
        runtime.main()


def test_unavailable_hardware_identity_stops_before_request_construction(tmp_path,monkeypatch):
    _,_,path,_=environment(tmp_path,omit=["native"])
    monkeypatch.setattr(sys,"argv",["run_model_v2.py","--freeze",str(path),"--stage","native"])
    def missing():
        raise OSError("hardware query unavailable")
    monkeypatch.setattr(runtime,"hardware_model",missing)
    with pytest.raises(ValueError,match="cannot verify frozen hardware"):
        runtime.main()


@pytest.mark.parametrize("stage", ["native", "cells"])
def test_main_cannot_construct_cc_requests_before_audit_pass(tmp_path, monkeypatch, stage):
    _, _, freeze_path, _ = environment(tmp_path, omit=["native"] if stage == "native" else [],
                                      audit_updates={"authorized_for_cc": False})
    monkeypatch.setattr(sys, "argv", ["run_model_v2.py", "--freeze", str(freeze_path), "--stage", stage])
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
    monkeypatch.setattr(sys, "argv", ["run_model_v2.py", "--freeze", str(freeze_path), "--stage", "native"])
    with pytest.raises(FileExistsError):
        runtime.main()
    assert path.read_bytes() == before
    verify_artifact_manifest(path)


@pytest.mark.parametrize("name",["smoke","calibration","calibration_audit","native"])
def test_wrong_inference_method_cannot_authorize_execution(tmp_path,name):
    freeze,digest,_,_ = environment(tmp_path,receipt_updates={name:{"inference_method":"original_v1"}})
    with pytest.raises(ValueError):
        runtime.verify_stage_authorization(freeze,digest,"cells")


@pytest.mark.parametrize("updates",[
    {"passed":False},{"passed":1},{"calibration_passed":False},{"calibration_passed":1},
    {"authorized_for_fidelity":False},{"authorized_for_fidelity":1},
    {"auditor_sha256":"stale-auditor"},{"calibration_sha256":"stale-calibration"},
])
def test_independent_calibration_audit_must_authorize_fidelity_explicitly(tmp_path,updates):
    freeze,digest,_,_ = environment(tmp_path,receipt_updates={"calibration_audit":updates})
    with pytest.raises(ValueError,match="calibration audit"):
        runtime.verify_stage_authorization(freeze,digest,"fidelity")


@pytest.mark.parametrize("updates",[{"acceptance":{"all_passed":False}},
    {"acceptance":{"all_passed":1}},{"settings":{"different":"recipe"}}])
def test_calibration_status_is_insufficient_if_recipe_or_gate_fails(tmp_path,updates):
    freeze,digest,_,_ = environment(tmp_path,receipt_updates={"calibration":updates})
    with pytest.raises(ValueError,match="calibration"):
        runtime.verify_stage_authorization(freeze,digest,"fidelity")


def test_v1_verifier_rejects_new_freeze_before_any_model_path(tmp_path):
    _,_,path,_ = environment(tmp_path)
    with pytest.raises(ValueError,match="frozen package"):
        verify_freeze(path,REPO)
    assert runtime.verify_freeze(path,REPO)["inference_method"] == runtime.METHOD


@pytest.mark.parametrize("field,value",[("status","FROZEN"),("schema","gfmcd.register_operator.freeze.v1"),
    ("inference_method","original_v1"),("base_freeze_sha256","wrong")])
def test_v2_version_verifier_rejects_old_or_stale_receipts(tmp_path,field,value):
    freeze,_,path,_ = environment(tmp_path)
    freeze[field]=value;path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError,match="A1 v2"):
        runtime.verify_freeze(path,REPO)


@pytest.mark.parametrize("which",["receipt","manifest"])
def test_approval_snapshots_reject_even_semantically_equivalent_changes(tmp_path,which):
    freeze,digest,path,_ = environment(tmp_path)
    approvals = runtime.verify_stage_authorization(freeze,digest,"native",freeze_path=path)
    snapshot = runtime.prerequisite_hashes(approvals)
    target = Path(freeze["outputs"]["smoke"])
    if which == "manifest":
        target = manifest_path(target)
        target.write_text(target.read_text()+" ")
    else:
        # Update the manifest consistently: the receipt still passes semantic checks.
        target.write_text(target.read_text()+" ")
        mp=manifest_path(target);m=json.loads(mp.read_text())
        m["outputs"][0]["sha256"]=sha256_file(target);mp.write_text(json.dumps(m))
    with pytest.raises(ValueError,match="changed during execution"):
        runtime.verify_unchanged_approvals(freeze,digest,"native",snapshot,freeze_path=path)


def native_fixture(cc):
    blocks=[];rows=[]
    for i,coupling in enumerate(cc):
        block={"block_id":f"b{i}","chrom":"chr1" if i<2 else "chr2","strand":"+"}
        for role in ("donor","acceptor"):
            block[role+"s"]={}
            for phase in range(3):
                source=f"s{i}-{role}-{phase}";block[role+"s"][str(phase)]={"source_id":source}
                prefix=f"native:b{i}:{role}:p{phase}:{source}:"
                for state,value in (("donor_intact",2.+coupling),("donor_true_acceptor_cc",2.),
                                    ("donor_true_acceptor_ac",2.+.5*coupling),("acceptor_intact",1.+coupling)):
                    rows.append({"request_id":prefix+state,"max_margin":float(value),"linear_margin":float(value+.2)})
        blocks.append(block)
    return {"native_requests":rows,"determinism":{"native":{"passed":True}}},blocks


def direct_native_replicates(values,indices):
    values=np.asarray(values,dtype=float);transformed=values.copy()
    for take in (np.arange(2),np.arange(2,6)):
        center=values[take].mean()
        transformed[take]=center+np.sqrt(len(take)/(len(take)-1))*(values[take]-center)
    return transformed[indices].mean(axis=1)


def test_real_native_preflight_correction_can_fail_where_uncorrected_control_passes():
    ids=[f"b{i}" for i in range(6)];strata=["chr1:+"]*2+["chr2:+"]*4
    # This uses only the unchanged bootstrap seed, never a calibration root.
    old=old_statistics.make_inference_context(ids,strata)
    cc=np.array([-.08,.08,-.03,-.01,.01,.03])
    corrected_low=np.quantile(direct_native_replicates(cc,old.indices),.025)
    cc += .05-corrected_low-1e-5
    raw,blocks=native_fixture(cc)
    result=runtime.native_preflight(raw,blocks)
    assert result["passed"] is False
    assert np.quantile(cc[old.indices].mean(axis=1),.025) > .05
    assert result["inference_method"] == runtime.METHOD
    for name,values in result["block_values"].items():
        expected=old_statistics.interval(np.asarray(values),direct_native_replicates(values,old.indices))
        np.testing.assert_allclose(result["endpoints"][name]["ci95"],expected["ci95"],atol=2e-14,rtol=0)
        assert result["endpoints"][name]["mean"] == pytest.approx(np.mean(values),abs=1e-15)
    reduced={"block_ids":ids,"strata":strata,"synthetic":True,
        "matrices":{f"{op}_{margin}":np.zeros((6,3,3)) for op in ("cc","ac","at") for margin in ("max","linear")},
        "native":{k:np.asarray(v) for k,v in result["block_values"].items()},
        "health":{"logp_delta":np.zeros(6),"construct_correct":np.ones(6),"source_correct":np.ones(6),"probe_pairs":np.ones(6)},
        "phase_blind":{k:np.ones(6) for k in ("donor_max","acceptor_max","donor_linear","acceptor_linear","bridge_max")},
        "prerequisites":{},"numerics":{}}
    analyzed=statistics.analyze(reduced)
    for name, endpoint in result["endpoints"].items():
        for field, value in endpoint.items():
            np.testing.assert_allclose(analyzed["instrument"]["native"][name][field],
                                       value, rtol=0, atol=2e-14)
    assert analyzed["instrument"]["native_floors"] == result["floors"]


def test_native_helper_rejects_uncorrected_context(monkeypatch):
    raw,blocks=native_fixture([.1]*6)
    monkeypatch.setattr(statistics,"make_inference_context",old_statistics.make_inference_context)
    with pytest.raises(ValueError,match="corrected v2"):
        runtime.native_preflight(raw,blocks)


def test_original_historical_auditor_bytes_and_raw_schema_are_unchanged():
    assert sha256_file(REPO/AUDITOR) == "62133af0f56d8159bf7c9f75d258260cf78489c3d14c7c1e978ea5867cb1b637"
    source=(REPO/"scripts/splicebert_register_operator_replication/run_model_v2.py").read_text()
    assert '"gfmcd.splicebert_register_operator_replication.fidelity.v1" if args.stage == "fidelity"' in source


@pytest.mark.parametrize("stage,expected",[
    ("fidelity", {"native":6912,"cells":127872}),
    ("native", {"native":8208}), ("cells", {"cells":116964}),
])
def test_actual_stage_planner_keeps_all_historical_blocks_and_denies_cc_in_fidelity(monkeypatch,stage,expected):
    original=[{"block_id":f"b{i}"} for i in range(384)]
    package={"original_blocks":original,"selected_block_ids":[b["block_id"] for b in original[:342]]}
    calls=[]
    def requests(blocks,genome,*,stage,include_cc):
        calls.append((len(blocks),stage,include_cc))
        per_block=(24 if include_cc else 18) if stage=="native" else (342 if include_cc else 333)
        return [None]*(per_block*len(blocks))
    monkeypatch.setattr(runtime,"requests_for_blocks",requests)
    blocks,plans=runtime.prepare_requests(package,object(),stage)
    assert {key:len(value) for key,value in plans.items()} == expected
    assert all(cc is (stage!="fidelity") for _,_,cc in calls)
    assert blocks == original if stage=="fidelity" else blocks == original[:342]
    package["original_blocks"].pop()
    if stage=="fidelity":
        with pytest.raises(ValueError,match="block count"):
            runtime.prepare_requests(package,object(),stage)
