#!/usr/bin/env python3
"""Independently validate selected CC measurements and their execution history.

This result auditor loads no model and imports no production measurement,
construction, or inference implementation. Only immutable historical-auditor
primitives and the shared provenance writer are reused. A passing report checks
measurement and provenance; the separate result auditor owns statistical claims.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRIMITIVE_PATH = HERE / "audit_fidelity.py"
PRIMITIVE_SHA = "62133af0f56d8159bf7c9f75d258260cf78489c3d14c7c1e978ea5867cb1b637"
if hashlib.sha256(PRIMITIVE_PATH.read_bytes()).hexdigest() != PRIMITIVE_SHA:
    raise ValueError("immutable independent audit primitives changed")
sys.path.insert(0,str(HERE))
from audit_fidelity import (  # noqa: E402
    Checks, ExpectedRequest, IDS, IndexedFasta, PINS, PROBES, REPEAT_TOLERANCE,
    RAW_TOLERANCE, LOG_PROBABILITY_TOLERANCE, digest, read_json, require,
    sequence_digest, write_exclusive,
)
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402

SCHEMA = "gfmcd.register_operator.cc_measurement_audit.v1"
RAW_SCHEMA = "gfmcd.register_operator.model.v1"
FREEZE_SHA = "82bc168e2246dff82519215203e02747f0209c289fa00c3e538613d95573d854"
SELECTED_IDS_SHA = "972d388cc031b6debdbb5b1c531afb7a8619630462f6bf4bb380d6a989417436"
BLOCKS = 342


def replace_ag(sequence: str, start: int, replacement: str) -> str:
    require(replacement in {"AC","AT","CC"} and sequence[start:start+2] == "AG",
            "invalid corruption source or operator")
    edited = sequence[:start] + replacement + sequence[start+2:]
    changed = [i for i,(left,right) in enumerate(zip(sequence,edited)) if left != right]
    require(changed == ([start,start+1] if replacement == "CC" else [start+1]),
            "intervention nucleotide count differs")
    for i in (start-1,start,start+1):
        require(not (edited[i:i+2] in {"AG","GT","GC"} and edited[i:i+2] != sequence[i:i+2]),
                "corruption creates an overlapping splice motif")
    return edited


def expected_requests(cohort: Mapping[str,Any], selected_ids: Sequence[str],
                      fetch: Callable[[str,int,int,str],str], *, expected_blocks: int = BLOCKS
                      ) -> dict[str,Any]:
    """Derive new CC ordering directly from frozen coordinates, without raw outcomes."""
    require(len(selected_ids) == len(set(selected_ids)) == expected_blocks, "selected IDs are incomplete or duplicate")
    wanted = set(selected_ids)
    blocks = [block for block in cohort["blocks"] if block["block_id"] in wanted]
    require(len(blocks) == expected_blocks and {b["block_id"] for b in blocks} == wanted,
            "selected cohort membership differs")
    output: dict[str,dict[str,ExpectedRequest]] = {"cell":{},"native":{}}

    def add(stage: str, key: str, seq: str, pos: int, base: str) -> None:
        require(len(seq) == 1022 and not set(seq)-set("ACGT") and 0 <= pos < 1022 and seq[pos] == base,
                "invalid expected masked sequence")
        request = ExpectedRequest(key,seq,pos,base)
        previous = output[stage].get(key)
        require(previous is None or previous == request, "inconsistent shared source probe")
        output[stage][key] = request

    def fragment(row: Mapping[str,Any], native: bool = False) -> str:
        left,right = ("window_start","window_end") if native else ("genomic_start","genomic_end")
        seq = fetch(row["chrom"],row[left],row[right],row["strand"])
        require(sequence_digest(seq) == row["sequence_sha256"], "reference fragment/window hash differs")
        return seq

    native_groups = []
    for block in blocks:
        bid = block["block_id"]
        bridge = fragment(block["bridge"])
        require(len(bridge) == 24 and bridge[0] == "A" and bridge[-1] == "T"
                and bridge[11:13] == "AG" and bridge.count("AG") == 1, "bridge geometry differs")
        donors,acceptors,sources,prefixes = {},{},{},[]
        for role,plural in (("donor","donors"),("acceptor","acceptors")):
            require(set(block[plural]) == {"0","1","2"}, "source phases are incomplete")
            for phase in range(3):
                source = block[plural][str(phase)]
                require(source["boundary_phase"] == phase and source["chrom"] == block["chrom"]
                        and source["strand"] == block["strand"], "source phase/stratum differs")
                donor = fragment(source["donor_fragment"])
                acceptor = fragment(source["acceptor_fragment"])
                native = fragment(source["native_window"],native=True)
                dpos = int(source["native_donor_pos"])
                require(len(donor) == 424 and len(acceptor) == 574 and len(native) == 1022
                        and dpos == 512-int(source["intron_length"])
                        and native[dpos:dpos+2] == "GT" and native[510:512] == "AG",
                        "native/source geometry differs")
                key = f"native:{bid}:{role}:p{phase}:{source['source_id']}"
                prefixes.append(key)
                for state,seq,pos in (
                    ("donor_intact",native,dpos),
                    ("donor_true_acceptor_cc",replace_ag(native,510,"CC"),dpos),
                    ("donor_true_acceptor_ac",replace_ag(native,510,"AC"),dpos),
                    ("acceptor_intact",native,511),
                ):
                    add("native",f"{key}:{state}",seq,pos,"G")
                if role == "donor":
                    donors[phase] = donor
                    boundary = int(source["donor_boundary"])
                    start = boundary-392 if source["strand"] == "+" else boundary+392-1022
                    aligned = fetch(source["chrom"],start,start+1022,source["strand"])
                    require(aligned[:424] == donor, "donor protected source alignment differs")
                    sources[role,phase] = aligned
                else:
                    acceptors[phase] = acceptor
                    require(native[448:] == acceptor, "acceptor protected source alignment differs")
                    sources[role,phase] = native
        native_groups.append(prefixes)
        public = {(row["p"],row["q"]):row for row in block["cells"]}
        require(len(public) == len(block["cells"]) == 9, "full phase lattice is absent")
        for p in range(3):
            for q in range(3):
                seq = donors[p]+bridge+acceptors[q]
                require(sequence_digest(seq) == public[p,q]["sequence_sha256"], "full cell sequence hash differs")
                require(seq[392:394] == "GT" and seq[435:437] == seq[510:512] == "AG"
                        and all(seq[i-1:i+1] not in {"AG","GT","GC"} for i in (424,448)),
                        "crossed site/seam geometry differs")
                key = f"{bid}:p{p}:q{q}"
                for state,state_seq,pos in (
                    ("donor_intact",seq,392),
                    ("donor_true_acceptor_cc",replace_ag(seq,510,"CC"),392),
                    ("donor_true_acceptor_ac",replace_ag(seq,510,"AC"),392),
                    ("donor_true_acceptor_at",replace_ag(seq,510,"AT"),392),
                    ("donor_bridge_ac",replace_ag(seq,435,"AC"),392),
                    ("acceptor_intact",seq,511),
                ):
                    add("cell",f"{key}:{state}",state_seq,pos,"G")
                pending = []
                for role,phase in (("donor",p),("acceptor",q)):
                    sid = block[role+"s"][str(phase)]["source_id"]
                    for pos in PROBES[role]:
                        require(sources[role,phase][pos] == seq[pos], "protected source base differs")
                        add("cell",f"{key}:health:{role}:{pos}:construct",seq,pos,seq[pos])
                        pending.append((f"source:{role}:{sid}:health:{pos}",sources[role,phase],pos,seq[pos]))
                for values in pending:
                    add("cell",*values)
    require(len(output["cell"]) == 342*expected_blocks and len(output["native"]) == 24*expected_blocks,
            "complete CC request counts differ")
    return {"cell":list(output["cell"].values()),"native":list(output["native"].values()),
            "block_ids":[b["block_id"] for b in blocks],"native_groups":native_groups,
            "strata":[f"{b['chrom']}:{b['strand']}" for b in blocks]}


def audit_request_records(rows: Any, repeats: Any, expected: Sequence[ExpectedRequest],
                          summary: Any, *, label: str) -> dict[str,Any]:
    checks = Checks()
    valid = isinstance(rows,list) and all(isinstance(r,Mapping) for r in rows)
    checks.check(valid,f"{label}:request schema")
    if not valid:
        return checks.result()
    ids = [r.request_id for r in expected]
    checks.check([r.get("request_id") for r in rows] == ids,f"{label}:exact full request order/coverage")
    by_id = {r["request_id"]:r for r in rows if isinstance(r.get("request_id"),str)}
    checks.check(len(by_id) == len(rows),f"{label}:unique request IDs")
    attention_sha = digest([1]*1024)
    sequence_shas = {}

    def record(row: Mapping[str,Any], request: ExpectedRequest) -> None:
        key,seq = request.request_id,request.sequence
        if seq not in sequence_shas:
            sequence_shas[seq] = sequence_digest(seq)
        tokens = [IDS["CLS"],*[IDS[b] for b in seq],IDS["SEP"]]
        tokens[request.position+1] = IDS["MASK"]
        for field,value in {"request_id":key,"sequence_sha256":sequence_shas[seq],"position":request.position,
            "token_position":request.position+1,"true_base":request.true_base,"token_count":1024,
            "input_ids_sha256":digest(tokens),"attention_mask_sha256":attention_sha}.items():
            checks.check(row.get(field) == value,f"{key}:{field}")
        checks.check(row.get("all_forward_logits_finite") is True,f"{key}:all forward logits finite")
        values = row.get("masked_logits")
        finite = isinstance(values,list) and len(values) == 10 and all(
            isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in values)
        checks.check(finite,f"{key}:full ten-token finite logits")
        if not finite:
            return
        logits = np.asarray(values,dtype=np.float32)
        checks.check(np.isfinite(logits).all(),f"{key}:finite float32 logits")
        checks.check(np.array_equal(np.asarray(values,dtype=float),logits.astype(float)),
                     f"{key}:saved logits are exact float32 values")
        bases = [values[IDS[b]] for b in "ACGT"]
        saved_bases = row.get("base_logits")
        checks.check(isinstance(saved_bases,list) and len(saved_bases) == 4
                     and all(type(x) in (int,float) and math.isfinite(x) for x in saved_bases)
                     and saved_bases == bases,f"{key}:exact four-base logit copy")
        nucleotide_logits = logits[[IDS[b] for b in "ACGT"]]
        true = logits[IDS[request.true_base]]
        linear = np.float32(true-np.float32(np.float32(nucleotide_logits.sum(dtype=np.float32)-true)/np.float32(3)))
        maximum = np.float32(true-max(logits[IDS[b]] for b in "ACGT" if b != request.true_base))
        full = np.asarray(values,dtype=float)
        shift = float(full.max())
        logp = float(full[IDS[request.true_base]]-shift-np.log(np.exp(full-shift).sum()))
        for field,value,tol in (("linear_margin",float(linear),RAW_TOLERANCE),
                                ("max_margin",float(maximum),RAW_TOLERANCE),
                                ("true_base_log_probability",logp,LOG_PROBABILITY_TOLERANCE)):
            checks.numeric(row.get(field),value,tol,f"{key}:{field} from full logits",field)
        checks.check(type(row.get("correct")) is bool
                     and row["correct"] == (int(np.argmax(full)) == IDS[request.true_base]),
                     f"{key}:exact full-vocabulary accuracy")

    for request in expected:
        if request.request_id in by_id:
            record(by_id[request.request_id],request)
    repeat_valid = isinstance(repeats,list) and all(isinstance(r,Mapping) for r in repeats)
    checks.check(repeat_valid and [r.get("request_id") for r in repeats] == ids[:16],
                 f"{label}:exact first sixteen repeated requests")
    differences = {field:0.0 for field in ("linear_margin","max_margin","true_base_log_probability","masked_logits")}
    identical = True
    if repeat_valid:
        lookup = {r.request_id:r for r in expected[:16]}
        for row in repeats:
            key = row.get("request_id")
            if not isinstance(key,str) or key not in lookup or key not in by_id:
                continue
            record(row,lookup[key])
            original = by_id[key]
            for field in differences:
                left,right = row.get(field),original.get(field)
                if field != "masked_logits":
                    left,right = [left],[right]
                good = isinstance(left,list) and isinstance(right,list) and len(left) == len(right) and all(
                    type(x) in (int,float) and math.isfinite(x) for x in [*left,*right])
                checks.check(good,f"{key}:repeat {field} shape/finite")
                if good:
                    difference = max((abs(a-b) for a,b in zip(left,right)),default=0.0)
                    differences[field] = max(differences[field],difference)
                    checks.numeric(difference,0.0,REPEAT_TOLERANCE,f"{key}:repeat {field}","repeat_"+field)
            same = type(row.get("correct")) is bool and row.get("correct") is original.get("correct")
            checks.check(same,f"{key}:repeat exact accuracy")
            identical = identical and same
    expected_summary = {"maximum_absolute_differences":differences,"accuracy_identical":identical,
                        "maximum":max(differences.values()),"threshold":REPEAT_TOLERANCE,
                        "passed":max(differences.values()) <= REPEAT_TOLERANCE and identical}
    checks.check(isinstance(summary,Mapping) and type(summary.get("passed")) is bool
                 and type(summary.get("accuracy_identical")) is bool and summary == expected_summary,
                 f"{label}:recorded determinism summary independently recomputes")
    return {**checks.result(),"requests":{"expected":len(expected),"observed":len(rows)},
            "repeat_recomputed":expected_summary}


def audit_payloads(native: Mapping[str,Any], cells: Mapping[str,Any], plan: Mapping[str,Any],
                   freeze: Mapping[str,Any], *, freeze_sha: str, construction_sha: str,
                   native_sha: str) -> dict[str,Any]:
    checks = Checks()
    runtime = freeze["runtime"]
    for stage,payload in (("native",native),("cells",cells)):
        expected_header = {"schema":RAW_SCHEMA,"stage":stage,"status":"COMPLETE","freeze_sha256":freeze_sha,
            "cohort_sha256":PINS["cohort"],"construction_sha256":construction_sha,"block_ids":plan["block_ids"],
            "checkpoint_files":freeze["checkpoint_files"],"tokenizer_ids":IDS,"batch_size":16,
            "resolved_device":runtime["device"],"model_dtype":"torch."+runtime["dtype"],
            "attention_implementation":runtime["attention_implementation"]}
        for field,value in expected_header.items():
            checks.check(payload.get(field) == value,f"{stage}:{field}")
        checks.check("failure" not in payload,f"{stage}:no instrument error")
        environment = {"python":runtime["python"],"platform":runtime["platform"],
                       **{k:runtime["versions"][k] for k in ("torch","transformers","numpy","pyfaidx")}}
        checks.check(payload.get("environment") == environment,f"{stage}:frozen recorded runtime profile")
        checks.check(payload.get("native_preflight",{}).get("passed") is True,f"{stage}:native control passed")
    checks.check(native.get("cell_requests") == [] and native.get("cell_repeats") == [],
                 "native stage contains no crossed outcomes")
    checks.check(cells.get("native_result_sha256") == native_sha,"cells:current native output binding")
    for field in ("native_requests","native_repeats","native_preflight"):
        checks.check(cells.get(field) == native.get(field),f"cells:exact carried {field}")
    checks.check(set(native.get("determinism",{})) == {"native"},"native determinism stages")
    checks.check(set(cells.get("determinism",{})) == {"native","cells"},"cells determinism stages")
    checks.check(cells.get("determinism",{}).get("native") == native.get("determinism",{}).get("native"),
                 "cells:exact carried native determinism")
    reports = {
        "native":audit_request_records(native.get("native_requests"),native.get("native_repeats"),
            plan["native"],native.get("determinism",{}).get("native"),label="native"),
        "cells":audit_request_records(cells.get("cell_requests"),cells.get("cell_repeats"),
            plan["cell"],cells.get("determinism",{}).get("cells"),label="cells"),
    }
    for label,result in reports.items():
        checks.check(result["passed"],f"{label}:all request-level measurement checks")
    return {**checks.result(),"request_audits":reports}


class Inputs:
    """Verify each byte stream once per unchanged stat signature; reject concurrent edits."""
    def __init__(self) -> None:
        self.records: dict[Path,tuple[tuple[int,...],str]] = {}

    @staticmethod
    def signature(path: Path) -> tuple[int,...]:
        stat = path.stat()
        return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns

    def sha(self,path: str | Path) -> str:
        path = Path(path).resolve(strict=True)
        before = self.signature(path)
        if path not in self.records or self.records[path][0] != before:
            value = sha256_file(path)
            require(self.signature(path) == before,f"input changed while hashing: {path}")
            self.records[path] = before,value
        return self.records[path][1]

    def unchanged(self) -> None:
        for path,(signature,_) in self.records.items():
            require(self.signature(path) == signature,f"audit input changed during verification: {path}")


def sidecar(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix == ".gz":
        p = p.with_suffix("")
    return p.with_suffix(".manifest.json")


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None,"execution timestamp lacks timezone")
    return parsed


def verify_manifest(path: str | Path, inputs: Inputs, required: Sequence[str | Path] = ()) -> dict:
    path = Path(path).resolve(strict=True)
    manifest_path = sidecar(path)
    inputs.sha(manifest_path)
    manifest,_ = read_json(manifest_path)
    outputs = [row for row in manifest["outputs"] if Path(row["path"]).resolve() == path]
    require(len(outputs) == 1 and outputs[0].get("sha256") == inputs.sha(path),"manifest output binding differs")
    records = manifest["inputs"]
    require(isinstance(records,list) and bool(records),"manifest has no source/input bindings")
    for row in records:
        require(inputs.sha(row["path"]) == row.get("sha256"),"manifest input changed")
    for item in required:
        p = Path(item).resolve(strict=True)
        matches = [row for row in records if Path(row["path"]).resolve() == p]
        require(len(matches) == 1 and matches[0].get("sha256") == inputs.sha(p),"required manifest input is absent or wrong")
    timestamp(manifest["created_at_utc"])
    return manifest


def prerequisites(freeze: Mapping[str,Any], freeze_path: Path, inputs: Inputs) -> dict:
    """This gate is evaluated before opening either CC raw output."""
    fs = inputs.sha(freeze_path)
    require(fs == FREEZE_SHA and freeze["status"] == "FROZEN", "wrong prospectively frozen protocol")
    freeze_manifest = verify_manifest(freeze_path,inputs)
    require(freeze_manifest["status"] == "FROZEN","freeze manifest status differs")
    require(freeze["owner_choices"]["span_nt"] == 120 and freeze["owner_choices"]["blocks"] == BLOCKS,
            "frozen owner choices differ")
    for name,sha in freeze["files"].items():
        path = Path(name) if Path(name).is_absolute() else REPO/name
        require(inputs.sha(path) == sha,f"frozen file changed: {name}")
    receipts,manifests = {},{}
    for name in ("smoke","calibration","fidelity_audit"):
        path = Path(freeze["outputs"][name])
        manifests[name] = verify_manifest(path,inputs,[freeze_path])
        receipts[name],_ = read_json(path)
        require(receipts[name].get("status") == "PASS" and receipts[name].get("freeze_sha256") == fs,
                f"{name} does not authorize CC")
        require(manifests[name]["status"] == "PASS",f"{name} manifest status differs")
    audit = receipts["fidelity_audit"]
    require(audit.get("authorized_for_cc") is True and audit.get("passed") is True
            and audit.get("auditor_sha256") == PRIMITIVE_SHA
            and audit.get("fidelity_sha256") == inputs.sha(freeze["outputs"]["fidelity"]),
            "historical fidelity PASS does not bind the frozen raw replay")
    require(freeze["files"]["scripts/splicebert_register_operator_replication/audit_fidelity.py"] == PRIMITIVE_SHA,
            "historical auditor source pin differs")
    return {"receipts":receipts,"manifests":manifests,"freeze_manifest":freeze_manifest}


def execution_history(freeze: Mapping[str,Any], freeze_path: Path, native: Mapping[str,Any],
                      cells: Mapping[str,Any], inputs: Inputs, approvals: Mapping[str,Any]) -> dict:
    fs = inputs.sha(freeze_path)
    outputs = freeze["outputs"]
    fidelity,_ = read_json(Path(outputs["fidelity"]))
    require(fidelity.get("status") == "COMPLETE" and fidelity.get("freeze_sha256") == fs,
            "historical replay completion differs")
    frozen_time = timestamp(freeze["frozen_at_utc"])
    timeline = {"freeze":freeze["frozen_at_utc"]}
    required = [freeze_path,REPO/"scripts/splicebert_register_operator_replication/run_model.py",
                freeze["construction_path"],freeze["construction_audit_path"]]
    freeze_saved = timestamp(approvals["freeze_manifest"]["created_at_utc"])
    preflights = [timestamp(approvals["manifests"][key]["created_at_utc"]) for key in ("smoke","calibration")]
    require(all(value >= freeze_saved >= frozen_time for value in preflights),"model-free preflight preceded the freeze")
    previous = max(preflights)
    for stage,raw in (("fidelity",fidelity),("native",native),("cells",cells)):
        path = Path(outputs[stage])
        apath = Path(outputs[stage+"_attempt"])
        attempt,_ = read_json(apath)
        stage_prerequisites = [outputs["smoke"],outputs["calibration"]]
        if stage != "fidelity":
            stage_prerequisites.append(outputs["fidelity_audit"])
        if stage == "cells":
            stage_prerequisites.append(outputs["native"])
        amanifest = verify_manifest(apath,inputs,[*required,*stage_prerequisites])
        manifest = verify_manifest(path,inputs,[*required,*stage_prerequisites,apath])
        require(amanifest["status"] == "STARTED" and manifest["status"] == raw.get("status") == "COMPLETE",
                "attempt/raw manifest status differs")
        expected_counts = ({"native":6912,"cells":127872} if stage == "fidelity" else
                           {"native":24*BLOCKS} if stage == "native" else {"cells":342*BLOCKS})
        require(attempt.get("schema") == "gfmcd.register_operator.attempt.v1"
                and attempt.get("status") == "STARTED" and attempt.get("stage") == stage
                and attempt.get("freeze_sha256") == fs
                and attempt.get("new_cc") is (stage != "fidelity")
                and Path(attempt["out"]).resolve() == path.resolve()
                and attempt.get("request_counts") == expected_counts,"attempt content/coverage differs")
        require(raw.get("attempt_sha256") == inputs.sha(apath),"raw does not bind its immutable attempt")
        started = timestamp(attempt["created_at_utc"])
        attempt_saved = timestamp(amanifest["created_at_utc"])
        completed = timestamp(raw["completed_at_utc"])
        saved = timestamp(manifest["created_at_utc"])
        require(previous <= started <= attempt_saved <= completed <= saved,"execution approval/attempt/completion chronology differs")
        seconds = raw.get("runtime_seconds")
        require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds >= 0
                and seconds <= (completed-started).total_seconds()+1.0,"runtime is inconsistent with timestamps")
        timeline[stage] = {"attempt_created":started.isoformat(),"attempt_saved":attempt_saved.isoformat(),
                           "completed":completed.isoformat(),"raw_manifest_created":saved.isoformat()}
        if stage == "fidelity":
            audited = timestamp(approvals["manifests"]["fidelity_audit"]["created_at_utc"])
            require(audited >= saved,"fidelity approval preceded raw replay completion")
            previous = audited
        else:
            previous = saved
    return {"passed":True,"timeline":timeline}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("freeze","native","cells","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args(argv)
    for path in (args.out,sidecar(args.out)):
        require(not path.exists() and not path.is_symlink(),f"immutable audit output already exists: {path}")
    report: dict[str,Any] = {"schema":SCHEMA,"status":"FAIL","passed":False,"authorized_for_inference":False,
        "expected_blocks":BLOCKS,"model_loaded":False,"model_forward_count":0,"cc_raw_read":False,
        "auditor_sha256":sha256_file(Path(__file__)),"historical_auditor_sha256":PRIMITIVE_SHA,
        "tolerances":{"score_abs":RAW_TOLERANCE,"log_probability_abs":LOG_PROBABILITY_TOLERANCE,
                      "repeat_abs":REPEAT_TOLERANCE}}
    inputs = Inputs()
    try:
        freeze,_ = read_json(args.freeze)
        report["freeze_sha256"] = inputs.sha(args.freeze)
        approvals = prerequisites(freeze,args.freeze,inputs)
        for key in ("native","cells"):
            path = getattr(args,key).resolve(strict=True)
            require(path == Path(freeze["outputs"][key]).resolve(strict=True),f"{key} is not the frozen output path")
            report[key+"_sha256"] = inputs.sha(path)
        cohort,cohort_sha = read_json(Path(freeze["historical_inputs"]["cohort"]))
        require(cohort_sha == PINS["cohort"],"frozen historical cohort differs")
        selection,_ = read_json(Path(freeze["selection_path"]))
        ids = selection["block_ids"]
        require(selection["block_ids_sha256"] == SELECTED_IDS_SHA and digest(ids) == SELECTED_IDS_SHA,
                "approved342 selected IDs differ")
        fasta = IndexedFasta(Path(freeze["fasta_path"]))
        try:
            plan = expected_requests(cohort,ids,fasta.fetch)
        finally:
            fasta.close()
        require(Counter(s.rsplit(":",1)[1] for s in plan["strata"]) == {"+":171,"-":171}
                and min(Counter(plan["strata"]).values()) >= 2,"approved stratification differs")
        native,_ = read_json(args.native)
        cells,_ = read_json(args.cells)
        report["cc_raw_read"] = True
        report["execution_history"] = execution_history(freeze,args.freeze,native,cells,inputs,approvals)
        report["measurement"] = audit_payloads(native,cells,plan,freeze,freeze_sha=report["freeze_sha256"],
            construction_sha=inputs.sha(freeze["construction_path"]),native_sha=report["native_sha256"])
        report["passed"] = report["measurement"]["passed"] and report["execution_history"]["passed"]
        report["authorized_for_inference"] = report["passed"]
        report["status"] = "PASS" if report["passed"] else "FAIL"
        report["selected_block_ids_sha256"] = digest(ids)
        inputs.unchanged()
    except (ValueError,KeyError,TypeError,OSError,IndexError,OverflowError) as error:
        report.update(status="FAIL",passed=False,authorized_for_inference=False,
                      fatal_audit_error=f"{type(error).__name__}: {error}")
    write_exclusive(args.out,report)
    manifest = ExperimentManifest.create("register-operator:independent-cc-measurement-audit",
        sys.argv if argv is None else [str(Path(__file__)),*argv],repo=REPO,status=report["status"],
        config={"expected_blocks":BLOCKS,"tolerances":report["tolerances"],"no_model_loaded":True})
    for path in sorted(set(inputs.records)|{Path(__file__).resolve(),PRIMITIVE_PATH,REPO/"src/gfmcd/provenance.py"}):
        manifest.add_input(path)
    manifest.add_output(args.out)
    write_exclusive(sidecar(args.out),manifest.to_dict())
    print(json.dumps({key:report[key] for key in ("status","passed","authorized_for_inference","cc_raw_read")},sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
