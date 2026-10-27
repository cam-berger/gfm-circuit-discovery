"""Adversarial checks for the independent, post-freeze CC measurement audit."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = load("independent_cc_measurement_audit",
             REPO/"scripts/splicebert_register_operator_replication/audit_cc_measurement.py")
old_fixtures = load("frozen_fidelity_fixture_definitions",REPO/"tests/test_register_operator_fidelity_audit.py")


def zero_repeat():
    return {"maximum_absolute_differences":{key:0.0 for key in (
        "linear_margin","max_margin","true_base_log_probability","masked_logits")},
        "accuracy_identical":True,"maximum":0.0,"threshold":1e-8,"passed":True}


def emitted(request):
    row = old_fixtures.emitted(request)
    row["masked_logits"] = np.asarray(row["masked_logits"],dtype=np.float32).tolist()
    row["base_logits"] = [row["masked_logits"][audit.IDS[b]] for b in "ACGT"]
    return row


@pytest.fixture
def valid():
    old = old_fixtures.old_shaped.__wrapped__()
    ids = [old["cohort"]["blocks"][0]["block_id"]]
    plan = audit.expected_requests(old["cohort"],ids,old["fetch"],expected_blocks=1)
    runtime = {"python":"frozen-python","platform":"frozen-platform","device":"mps",
               "dtype":"float32","attention_implementation":"sdpa",
               "versions":{k:"frozen-"+k for k in ("torch","transformers","numpy","pyfaidx")}}
    freeze = {"runtime":runtime,"checkpoint_files":{"weights":"frozen-checkpoint"}}
    common = {"schema":audit.RAW_SCHEMA,"status":"COMPLETE","freeze_sha256":"freeze",
        "cohort_sha256":audit.PINS["cohort"],"construction_sha256":"construction","block_ids":ids,
        "checkpoint_files":freeze["checkpoint_files"],"tokenizer_ids":dict(audit.IDS),"batch_size":16,
        "resolved_device":"mps","model_dtype":"torch.float32","attention_implementation":"sdpa",
        "environment":{"python":runtime["python"],"platform":runtime["platform"],**runtime["versions"]},
        "native_preflight":{"passed":True}}
    native = {**copy.deepcopy(common),"stage":"native","cell_requests":[],"cell_repeats":[],
              "native_requests":[emitted(r) for r in plan["native"]],
              "determinism":{"native":zero_repeat()}}
    native["native_repeats"] = copy.deepcopy(native["native_requests"][:16])
    cells = {**copy.deepcopy(common),"stage":"cells","native_result_sha256":"native",
             "cell_requests":[emitted(r) for r in plan["cell"]],
             "native_requests":copy.deepcopy(native["native_requests"]),
             "native_repeats":copy.deepcopy(native["native_repeats"]),
             "determinism":{"native":zero_repeat(),"cells":zero_repeat()}}
    cells["cell_repeats"] = copy.deepcopy(cells["cell_requests"][:16])
    return native,cells,plan,freeze


def inspect(valid):
    return audit.audit_payloads(*valid,freeze_sha="freeze",construction_sha="construction",native_sha="native")


def test_complete_selected_cc_measurements_pass(valid):
    report = inspect(valid)
    assert report["passed"],report
    assert report["request_audits"]["cells"]["requests"] == {"expected":342,"observed":342}
    assert report["request_audits"]["native"]["requests"] == {"expected":24,"observed":24}
    plan = valid[2]
    assert sum(r.request_id.endswith("donor_true_acceptor_cc") for r in plan["cell"]) == 9
    assert sum(r.request_id.endswith("donor_true_acceptor_cc") for r in plan["native"]) == 6
    assert not any("bridge_cc" in r.request_id for r in plan["cell"])


@pytest.mark.parametrize("stage",["cell","native"])
@pytest.mark.parametrize("fault",["missing","duplicate","order","extra","wrong_sequence","wrong_mask","wrong_true_base"])
def test_complete_request_identity_and_input_bytes_are_enforced(valid,stage,fault):
    payload = valid[1] if stage == "cell" else valid[0]
    rows = payload[stage+"_requests"]
    if fault == "missing": rows.pop()
    elif fault == "duplicate": rows[-1] = copy.deepcopy(rows[0])
    elif fault == "order": rows[0],rows[1] = rows[1],rows[0]
    elif fault == "extra": rows.append(copy.deepcopy(rows[-1]))
    elif fault == "wrong_sequence": rows[0]["sequence_sha256"] = "wrong-state"
    elif fault == "wrong_mask": rows[0]["input_ids_sha256"] = "wrong-input"
    else: rows[0]["true_base"] = "A"
    assert not inspect(valid)["passed"]


@pytest.mark.parametrize("field,value",[("stage","fidelity"),("status","INSTRUMENT_UNRESOLVED"),
    ("block_ids",["subset"]),("resolved_device","cpu"),("model_dtype","torch.float16"),
    ("attention_implementation","eager"),("tokenizer_ids",{}),("batch_size",8),
    ("native_result_sha256","stale-native"),("checkpoint_files",{})])
def test_raw_stage_membership_runtime_and_provenance_must_match(valid,field,value):
    valid[1][field] = value
    assert not inspect(valid)["passed"]


@pytest.mark.parametrize("field",["linear_margin","max_margin","true_base_log_probability"])
def test_scalar_score_must_recompute_from_full_logits(valid,field):
    valid[1]["cell_requests"][30][field] += .001
    assert not inspect(valid)["passed"]


@pytest.mark.parametrize("field",["masked_logits","base_logits","correct","all_forward_logits_finite"])
def test_full_vocabulary_logits_and_accuracy_are_not_replaceable_by_base_only_summary(valid,field):
    row = valid[1]["cell_requests"][30]
    if field == "masked_logits": row[field] = row[field][6:]
    elif field == "base_logits": row[field] = [0.0]*4
    else: row[field] = not row[field]
    assert not inspect(valid)["passed"]


def test_saved_logits_must_really_be_float32(valid):
    valid[1]["cell_requests"][30]["masked_logits"][0] += 1e-12
    assert not inspect(valid)["passed"]


@pytest.mark.parametrize("fault",["missing","order","scalar","logit","summary","boolean"])
def test_exact_first_sixteen_repeats_and_recorded_summary_are_checked(valid,fault):
    repeated = valid[1]["cell_repeats"]
    if fault == "missing": repeated.pop()
    elif fault == "order": repeated[0],repeated[1] = repeated[1],repeated[0]
    elif fault == "scalar": repeated[0]["max_margin"] += 1.1e-8
    elif fault == "logit": repeated[0]["masked_logits"][0] += .001
    elif fault == "summary": valid[1]["determinism"]["cells"]["maximum"] = .1
    else: valid[1]["determinism"]["cells"]["passed"] = 1
    assert not inspect(valid)["passed"]


def test_carried_native_records_and_native_pass_flag_are_exact(valid):
    valid[1]["native_requests"][0]["linear_margin"] += .001
    assert not inspect(valid)["passed"]
    valid[1]["native_requests"] = copy.deepcopy(valid[0]["native_requests"])
    valid[1]["native_preflight"]["passed"] = 1
    assert not inspect(valid)["passed"]


def test_cc_geometry_counts_changed_bases_and_all_flank_motifs():
    for left in "ACGT":
        for right in "ACGT":
            sequence = "A"*20+left+"AG"+right+"C"*20
            if left == "G":
                with pytest.raises(ValueError,match="overlapping"):
                    audit.replace_ag(sequence,21,"CC")
            else:
                edited = audit.replace_ag(sequence,21,"CC")
                assert [i for i,(a,b) in enumerate(zip(sequence,edited)) if a != b] == [21,22]


def test_bad_selected_membership_and_native_donor_geometry_fail():
    old = old_fixtures.old_shaped.__wrapped__()
    bid = old["cohort"]["blocks"][0]["block_id"]
    with pytest.raises(ValueError,match="selected"):
        audit.expected_requests(old["cohort"],[bid,bid],old["fetch"],expected_blocks=2)
    old["cohort"]["blocks"][0]["donors"]["0"]["native_donor_pos"] += 1
    with pytest.raises(ValueError,match="geometry"):
        audit.expected_requests(old["cohort"],[bid],old["fetch"],expected_blocks=1)


def artifact(path,data,inputs=(),time="2026-10-16T20:00:00+00:00"):
    path = Path(path)
    path.write_text(json.dumps(data))
    manifest = {"status":data["status"],"created_at_utc":time,
        "inputs":[{"path":str(p),"sha256":audit.sha256_file(p)} for p in inputs],
        "outputs":[{"path":str(path),"sha256":audit.sha256_file(path)}]}
    audit.sidecar(path).write_text(json.dumps(manifest))
    return manifest


def stamp(second):
    return f"2026-10-16T20:00:{second:02d}+00:00"


@pytest.fixture
def history(tmp_path):
    source = tmp_path/"source.py";source.write_text("synthetic source")
    package = tmp_path/"construction.json";package.write_text("{}")
    construction_audit = tmp_path/"construction.audit.json";construction_audit.write_text("{}")
    names = ("smoke","calibration","fidelity_audit","fidelity","native","cells",
             "fidelity_attempt","native_attempt","cells_attempt")
    outputs = {key:str(tmp_path/(key+".json")) for key in names}
    freeze = {"outputs":outputs,"frozen_at_utc":stamp(0),"construction_path":str(package),
              "construction_audit_path":str(construction_audit)}
    fp = tmp_path/"freeze.json";fp.write_text(json.dumps(freeze))
    fs = audit.sha256_file(fp)
    approvals = {"manifests":{},"freeze_manifest":{"created_at_utc":stamp(1)}}
    for key,time in (("smoke",2),("calibration",3),("fidelity_audit",8)):
        approvals["manifests"][key] = artifact(outputs[key],{"status":"PASS","freeze_sha256":fs},[fp,source],stamp(time))
    required = [fp,REPO/"scripts/splicebert_register_operator_replication/run_model.py",package,construction_audit]
    raw = {}
    for stage,seconds in (("fidelity",(4,5,6,7)),("native",(9,10,11,12)),("cells",(13,14,15,16))):
        started,saved,completed,manifest_time = seconds
        counts = {"native":6912,"cells":127872} if stage == "fidelity" else {stage:8208 if stage == "native" else 116964}
        dependencies = [outputs["smoke"],outputs["calibration"]]
        if stage != "fidelity": dependencies.append(outputs["fidelity_audit"])
        if stage == "cells": dependencies.append(outputs["native"])
        attempt = {"schema":"gfmcd.register_operator.attempt.v1","status":"STARTED","stage":stage,
            "freeze_sha256":fs,"new_cc":stage != "fidelity","out":outputs[stage],
            "request_counts":counts,"created_at_utc":stamp(started)}
        artifact(outputs[stage+"_attempt"],attempt,[*required,*dependencies],stamp(saved))
        raw[stage] = {"status":"COMPLETE","freeze_sha256":fs,
            "attempt_sha256":audit.sha256_file(outputs[stage+"_attempt"]),
            "completed_at_utc":stamp(completed),"runtime_seconds":.5}
        artifact(outputs[stage],raw[stage],[*required,*dependencies,outputs[stage+"_attempt"]],stamp(manifest_time))
    return freeze,fp,raw,approvals


def test_execution_history_requires_native_before_cells_and_fidelity_before_native(history):
    freeze,fp,raw,approvals = history
    report = audit.execution_history(freeze,fp,raw["native"],raw["cells"],audit.Inputs(),approvals)
    assert report["passed"]
    approvals["manifests"]["fidelity_audit"]["created_at_utc"] = stamp(14)
    with pytest.raises(ValueError,match="chronology"):
        audit.execution_history(freeze,fp,raw["native"],raw["cells"],audit.Inputs(),approvals)


@pytest.mark.parametrize("fault",["pre_freeze","duration","attempt_hash","coverage","new_cc","wrong_out","manifest_status"])
def test_execution_history_rejects_changed_attempts_and_impossible_timelines(history,fault):
    freeze,fp,raw,approvals = history
    if fault == "pre_freeze":
        approvals["manifests"]["smoke"]["created_at_utc"] = stamp(0)
    elif fault == "duration": raw["native"]["runtime_seconds"] = 100.0
    elif fault == "attempt_hash": raw["native"]["attempt_sha256"] = "wrong"
    else:
        path = Path(freeze["outputs"]["native_attempt"])
        value = json.loads(path.read_text())
        if fault == "coverage": value["request_counts"] = {"native":24}
        elif fault == "new_cc": value["new_cc"] = False
        elif fault == "wrong_out": value["out"] = str(path)
        else:
            m = audit.sidecar(path);data=json.loads(m.read_text());data["status"]="PASS";m.write_text(json.dumps(data))
        if fault != "manifest_status": path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        audit.execution_history(freeze,fp,raw["native"],raw["cells"],audit.Inputs(),approvals)


def test_manifest_verifier_checks_outputs_required_inputs_and_concurrent_changes(tmp_path):
    source = tmp_path/"source";source.write_text("source")
    missing = tmp_path/"missing";missing.write_text("other")
    out = tmp_path/"audit.json"
    artifact(out,{"status":"PASS"},[source])
    inputs = audit.Inputs()
    audit.verify_manifest(out,inputs,[source])
    with pytest.raises(ValueError,match="required manifest input"):
        audit.verify_manifest(out,inputs,[missing])
    source.write_text("changed")
    with pytest.raises(ValueError,match="input changed"):
        audit.verify_manifest(out,audit.Inputs())
    with pytest.raises(ValueError,match="changed during"):
        inputs.unchanged()


def test_cli_checks_prerequisites_before_reading_any_cc_and_writes_immutable_failure(tmp_path):
    freeze = tmp_path/"freeze.json";freeze.write_text("{}")
    out = tmp_path/"measurement.audit.json"
    # Invalid JSON would fail if either CC path were read before the prerequisite gate.
    native = tmp_path/"native.json";native.write_text("invalid-json")
    cells = tmp_path/"cells.json";cells.write_text("invalid-json")
    args = ["--freeze",str(freeze),"--native",str(native),"--cells",str(cells),"--out",str(out)]
    assert audit.main(args) == 1
    report = json.loads(out.read_text())
    assert report["cc_raw_read"] is False and report["authorized_for_inference"] is False
    assert "wrong prospectively frozen" in report["fatal_audit_error"]
    assert audit.sidecar(out).is_file()
    before = out.read_bytes()
    with pytest.raises(ValueError,match="immutable"):
        audit.main(args)
    assert out.read_bytes() == before
