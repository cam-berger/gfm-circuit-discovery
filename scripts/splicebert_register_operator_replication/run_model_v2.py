#!/usr/bin/env python3
"""Version-two execution gates, original historical fidelity, and corrected native inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import sha256_file
from gfmcd.splicing.register_operator_construction import Genome, requests_for_blocks
from gfmcd.splicing.register_operator_io import guard_paths, read_json, write_artifact, verify_artifact_manifest, require_manifest_input, manifest_path
from gfmcd.splicing.register_operator_measurement import score_requests, repeat_difference
from gfmcd.splicing.register_operator_io_v2 import verify_freeze
from gfmcd.splicing.register_operator_statistics_v2 import METHOD, CorrectedInferenceContext


def passed_receipt(path: Path, freeze_sha: str, label: str, *, method: bool = True,
                   freeze_path: Path | None = None) -> dict:
    manifest = verify_artifact_manifest(path)
    if freeze_path is not None:
        require_manifest_input(manifest, freeze_path)
    receipt = read_json(path)
    if (receipt.get("status") != "PASS" or manifest.get("status") != "PASS"
            or receipt.get("freeze_sha256") != freeze_sha
            or (method and receipt.get("inference_method") != METHOD)):
        raise ValueError(f"{label} does not authorize this frozen v2 execution")
    return receipt


def verify_stage_authorization(freeze: dict, freeze_sha: str, stage: str,
                               *, freeze_path: Path | None = None) -> list[Path]:
    if (freeze.get("status") != "FROZEN_V2" or freeze.get("inference_method") != METHOD
            or freeze.get("schema") != "gfmcd.register_operator.freeze.v2"
            or stage not in {"fidelity", "native", "cells"}):
        raise ValueError("v2 execution requires the amended instrument")
    outputs = freeze["outputs"]
    required = [Path(outputs[k]) for k in ("smoke", "calibration", "calibration_audit")]
    receipts = {key: passed_receipt(Path(outputs[key]), freeze_sha, key, freeze_path=freeze_path)
                for key in ("smoke", "calibration", "calibration_audit")}
    calibration = receipts["calibration"]
    if (calibration.get("acceptance", {}).get("all_passed") is not True
            or calibration.get("settings") != freeze["calibration"]):
        raise ValueError("calibration does not pass every frozen v2 rule")
    auditor = "scripts/splicebert_register_operator_replication/audit_results_v2.py"
    audit = receipts["calibration_audit"]
    if (audit.get("passed") is not True or audit.get("calibration_passed") is not True
            or audit.get("authorized_for_fidelity") is not True
            or audit.get("auditor_sha256") != freeze["files"][auditor]
            or audit.get("calibration_sha256") != sha256_file(outputs["calibration"])):
        raise ValueError("calibration audit does not authorize fidelity for this instrument")
    calibration_manifest = verify_artifact_manifest(outputs["calibration_audit"])
    for path in (Path(outputs["calibration"]), REPO / auditor):
        require_manifest_input(calibration_manifest, path)
    if stage != "fidelity":
        required.append(Path(outputs["fidelity_audit"]))
        audit = passed_receipt(Path(outputs["fidelity_audit"]), freeze_sha, "fidelity audit",
                               method=False, freeze_path=freeze_path)
        auditor = "scripts/splicebert_register_operator_replication/audit_fidelity.py"
        if (audit.get("authorized_for_cc") is not True or audit.get("passed") is not True
                or audit.get("auditor_sha256") != freeze["files"][auditor]
                or audit.get("fidelity_sha256") != sha256_file(outputs["fidelity"])):
            raise ValueError("fidelity audit does not bind the complete current raw replay and auditor")
        fidelity_manifest = verify_artifact_manifest(outputs["fidelity_audit"])
        for path in (Path(outputs["fidelity"]), REPO / auditor):
            require_manifest_input(fidelity_manifest, path)
    if stage == "cells":
        native = Path(outputs["native"])
        manifest = verify_artifact_manifest(native)
        if freeze_path is not None:
            require_manifest_input(manifest, freeze_path)
        result = read_json(native)
        if (result.get("schema") != "gfmcd.register_operator.model.v2"
                or result.get("freeze_sha256") != freeze_sha or result.get("status") != "COMPLETE"
                or manifest.get("status") != "COMPLETE" or result.get("inference_method") != METHOD
                or result.get("native_preflight", {}).get("inference_method") != METHOD
                or result.get("native_preflight", {}).get("passed") is not True):
            raise ValueError("crossed scoring requires passing native CC controls/floors under v2")
        required.append(native)
    return required


def prerequisite_hashes(paths: list[Path]) -> dict[str, str]:
    """Bind approvals and their manifests before the first possible model load."""
    return {str(p.resolve()): sha256_file(p) for path in paths
            for p in (path, manifest_path(path))}


def verify_unchanged_approvals(freeze: dict, freeze_sha: str, stage: str,
                             snapshot: dict[str, str], *, freeze_path: Path) -> None:
    paths = verify_stage_authorization(freeze, freeze_sha, stage, freeze_path=freeze_path)
    if prerequisite_hashes(paths) != snapshot:
        raise ValueError("prerequisite receipt or manifest changed during execution")


def hardware_model() -> str:
    return subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()


def verify_runtime_profile(freeze: dict) -> dict:
    """Reject a different frozen host/runtime before constructing model requests."""
    profile = freeze["runtime"]
    if (platform.python_version() != profile["python"] or any(
            importlib.metadata.version(k) != v for k, v in profile["versions"].items())):
        raise ValueError("installed runtime differs from frozen profile")
    if platform.platform() != profile.get("platform"):
        raise ValueError("installed runtime platform differs from frozen profile")
    try:
        model = hardware_model()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("cannot verify frozen hardware profile") from error
    if not model or model != profile.get("hardware_model"):
        raise ValueError("installed hardware differs from frozen profile")
    return {"platform": platform.platform(), "hardware_model": model}


def native_preflight(raw: dict, blocks: list[dict]) -> dict:
    """Compute native floors and coupling with the frozen block bootstrap before cells."""
    import numpy as np
    from gfmcd.splicing.register_operator_statistics_v2 import make_inference_context, interval
    by_id = {row["request_id"]: row for row in raw["native_requests"]}
    values = {k: [] for k in ("cc_max", "ac_max", "donor_max", "acceptor_max", "cc_linear",
                              "ac_linear", "donor_linear", "acceptor_linear")}
    for block in blocks:
        block_values = {k: [] for k in values}
        for role in ("donor", "acceptor"):
            for phase in range(3):
                source = block[role + "s"][str(phase)]["source_id"]
                prefix = f"native:{block['block_id']}:{role}:p{phase}:{source}:"
                rows = {k: by_id[prefix + k] for k in ("donor_intact", "donor_true_acceptor_cc",
                                                       "donor_true_acceptor_ac", "acceptor_intact")}
                for margin in ("max", "linear"):
                    field = margin + "_margin"
                    intact = rows["donor_intact"][field]
                    block_values["donor_" + margin].append(intact)
                    block_values["acceptor_" + margin].append(rows["acceptor_intact"][field])
                    for op in ("cc", "ac"):
                        block_values[op + "_" + margin].append(intact - rows["donor_true_acceptor_" + op][field])
        for key in values:
            values[key].append(float(np.mean(block_values[key])))
    context = make_inference_context([b["block_id"] for b in blocks], [b["chrom"] + ":" + b["strand"] for b in blocks])
    if (not isinstance(context, CorrectedInferenceContext)
            or context.inference_method != METHOD):
        raise ValueError("native preflight requires corrected v2 context")
    endpoints = {k: interval(np.array(v), context.weights @ np.array(v)) for k, v in values.items()}
    floors = {k: min(0.0, endpoints[k]["ci95"][0]) for k in values if k.startswith(("donor", "acceptor"))}
    return {"passed": endpoints["cc_max"]["ci95"][0] > .05 and raw["determinism"]["native"]["passed"],
            "endpoints": endpoints, "floors": floors, "block_values": values,
            "bootstrap_draws": 20000, "bootstrap_seed": 20260803, "inference_method": METHOD}


def prepare_requests(package: dict, genome, stage: str) -> tuple[list[dict], dict]:
    """Keep the historical replay complete and CC-free on the actual scoring path."""
    if stage not in {"fidelity", "native", "cells"}:
        raise ValueError("unknown scoring stage")
    selected = set(package["selected_block_ids"])
    blocks = (package["original_blocks"] if stage == "fidelity" else
              [b for b in package["original_blocks"] if b["block_id"] in selected])
    if len(blocks) != (384 if stage == "fidelity" else 342):
        raise ValueError("frozen block count mismatch")
    stages = ("native", "cells") if stage == "fidelity" else (stage,)
    plans = {name: requests_for_blocks(blocks, genome, stage=name, include_cc=stage != "fidelity")
             for name in stages}
    expected = {"native": (18 if stage == "fidelity" else 24) * len(blocks),
                "cells": (333 if stage == "fidelity" else 342) * len(blocks)}
    if any(len(requests) != expected[name] for name, requests in plans.items()):
        raise ValueError("unique request count mismatch")
    return blocks, plans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--stage", choices=("fidelity", "native", "cells"), required=True)
    args = p.parse_args()
    freeze = verify_freeze(args.freeze, REPO)
    verified_runtime = verify_runtime_profile(freeze)
    fs = sha256_file(args.freeze)
    out = Path(freeze["outputs"][args.stage])
    guard_paths(out)
    attempt_path = Path(freeze["outputs"][args.stage + "_attempt"])
    guard_paths(attempt_path)
    approvals = verify_stage_authorization(freeze, fs, args.stage, freeze_path=args.freeze)
    approval_hashes = prerequisite_hashes(approvals)
    package = read_json(freeze["construction_path"])
    audit = read_json(freeze["construction_audit_path"])
    audit_manifest = verify_artifact_manifest(freeze["construction_audit_path"])
    require_manifest_input(audit_manifest, freeze["construction_path"])
    if audit.get("status") != "AUDIT_PASS" or audit.get("passed") is not True:
        raise ValueError("construction audit not PASS")
    model_path = Path(freeze["model_path"])
    for filename, digest in freeze["checkpoint_files"].items():
        if sha256_file(model_path / filename) != digest:
            raise ValueError(f"checkpoint file mismatch: {filename}")
    genome = Genome(freeze["fasta_path"])
    blocks, plans = prepare_requests(package, genome, args.stage)
    stages = tuple(plans)
    inputs = [args.freeze, Path(__file__), *approvals, Path(freeze["construction_path"]),
              Path(freeze["construction_audit_path"])]
    attempt = {"schema": "gfmcd.register_operator.attempt.v2", "status": "STARTED",
               "stage": args.stage, "freeze_sha256": fs, "out": str(out),
               "inference_method": METHOD, "prerequisite_sha256": approval_hashes,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "request_counts": {k: len(v) for k, v in plans.items()}, "new_cc": args.stage != "fidelity"}
    write_artifact(attempt_path, attempt, inputs=inputs, experiment="register-operator-v2:" + args.stage + "-attempt")
    started = time.monotonic()
    raw = {"schema": "gfmcd.splicebert_register_operator_replication.fidelity.v1" if args.stage == "fidelity" else "gfmcd.register_operator.model.v2",
           "stage": args.stage, "freeze_sha256": fs, "inference_method": METHOD,
           "historical_inference_method": "original_c43_v1", "prerequisite_sha256": approval_hashes,
           "cohort_sha256": package["inputs"]["frozen_cohort"]["sha256"],
           "construction_sha256": sha256_file(freeze["construction_path"]),
           "attempt_sha256": sha256_file(attempt_path), "status": "STARTED",
           "block_ids": [b["block_id"] for b in blocks], "cell_requests": [], "native_requests": [],
           "cell_repeats": [], "native_repeats": [], "determinism": {},
           "checkpoint_files": freeze["checkpoint_files"], "batch_size": 16,
           "hardware_model": verified_runtime["hardware_model"],
           "environment": {"platform": platform.platform(), "python": platform.python_version(),
              **{k: importlib.metadata.version(k) for k in ("torch", "transformers", "numpy", "pyfaidx")}}}
    try:
        import torch
        torch.set_num_threads(1)
        from gfmcd.splicing.circuit import load_adapter
        if not torch.backends.mps.is_available():
            raise ValueError("frozen execution requires MPS")
        adapter = load_adapter(str(model_path), "mps", eager=False)
        adapter.model.eval()
        raw["resolved_device"] = str(adapter.device)
        raw["model_dtype"] = str(next(adapter.model.parameters()).dtype)
        raw["attention_implementation"] = str(getattr(adapter.model.config, "_attn_implementation", None))
        if (raw["resolved_device"] != freeze["runtime"]["device"]
                or raw["model_dtype"] != "torch." + freeze["runtime"]["dtype"]
                or raw["attention_implementation"] != freeze["runtime"]["attention_implementation"]):
            raise ValueError("model runtime differs from frozen profile")
        raw["tokenizer_ids"] = {b: int(adapter.tokenizer.convert_tokens_to_ids(b)) for b in "ACGT"}
        raw["tokenizer_ids"].update({k: int(getattr(adapter.tokenizer, k.lower() + "_token_id")) for k in ("MASK", "CLS", "SEP")})
        if raw["tokenizer_ids"] != {"A": 6, "C": 7, "G": 8, "T": 9, "MASK": 4, "CLS": 2, "SEP": 3}:
            raise ValueError("checkpoint tokenizer mapping changed")
        for stage in stages:
            print(json.dumps({"stage": stage, "status": "SCORING", "requests": len(plans[stage])}), flush=True)
            def progress(value):
                print(json.dumps({"stage": stage, **value}), flush=True)
            field = "cell" if stage == "cells" else "native"
            rows = score_requests(adapter, plans[stage], progress=progress,
                                  output_records=raw[field + "_requests"])
            repeated = score_requests(adapter, plans[stage][:16], output_records=raw[field + "_repeats"])
            raw[field + "_requests"] = rows
            raw[field + "_repeats"] = repeated
            raw["determinism"][stage] = repeat_difference(rows[:16], repeated)
            if not raw["determinism"][stage]["passed"]:
                raise ValueError("deterministic repeated forward failed")
        if args.stage == "native":
            raw["native_preflight"] = native_preflight(raw, blocks)
        if args.stage == "cells":
            native_path = Path(freeze["outputs"]["native"])
            native = read_json(native_path)
            raw["native_requests"] = native["native_requests"]
            raw["native_repeats"] = native["native_repeats"]
            raw["determinism"]["native"] = native["determinism"]["native"]
            raw["native_preflight"] = native["native_preflight"]
            raw["native_result_sha256"] = sha256_file(native_path)
        verify_freeze(args.freeze, REPO)
        if sha256_file(args.freeze) != fs:
            raise ValueError("freeze receipt changed during execution")
        verify_unchanged_approvals(freeze, fs, args.stage, approval_hashes, freeze_path=args.freeze)
        raw["status"] = "COMPLETE"
    except Exception as error:
        raw["status"] = "INSTRUMENT_UNRESOLVED"
        raw["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        raw["runtime_seconds"] = time.monotonic() - started
        raw["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        receipt = write_artifact(out, raw, inputs=[*inputs, attempt_path],
            experiment="register-operator-v2:" + args.stage, config={"batch_size": 16, "stage": args.stage, "inference_method": METHOD},
            seeds={"bootstrap": 20260803})
        print(json.dumps({"status": raw["status"], **receipt}), flush=True)


if __name__ == "__main__":
    main()
