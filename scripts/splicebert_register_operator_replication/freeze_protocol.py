#!/usr/bin/env python3
"""Bind the approved design, audited construction, executable sources and one-shot paths."""
import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import sha256_file
from gfmcd.splicing.register_operator_io import guard_paths, read_json, write_artifact, verify_artifact_manifest, require_manifest_input
from calibrate_statistics import frozen_settings


def source_closure(initial):
    """Pin locally imported source, including lazy model-boundary imports."""
    found = {Path(p).resolve() for p in initial}
    pending = list(found)
    while pending:
        path = pending.pop()
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
                names.extend(node.module + "." + alias.name for alias in node.names)
        for name in names:
            if not name.startswith("gfmcd"):
                continue
            base = REPO / "src" / name.replace(".", "/")
            candidates = [base.with_suffix(".py"), base / "__init__.py"]
            candidates += [parent / "__init__.py" for parent in base.parents if parent.is_relative_to(REPO / "src")]
            for candidate in candidates:
                if candidate.is_file() and candidate not in found:
                    found.add(candidate); pending.append(candidate)
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--construction", type=Path, required=True)
    p.add_argument("--construction-audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--result-dir", type=Path, required=True)
    args = p.parse_args()
    guard_paths(args.out)
    study = REPO / "studies/splicebert_register_operator_replication"
    selection = study / "planning/fallback342_selection_v1.json"
    protocol = study / "PREREGISTRATION.md"
    specification = study / "STATISTICAL_CALIBRATION.md"
    for path in (args.construction, args.construction_audit, selection, protocol, specification):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = read_json(args.construction_audit)
    if audit.get("status") != "AUDIT_PASS" or audit.get("passed") is not True or not audit.get("reconstruction", {}).get("production_pseudo_family_present"):
        raise ValueError("construction must pass independent audit including all new pseudo hashes")
    audit_manifest = verify_artifact_manifest(args.construction_audit)
    require_manifest_input(audit_manifest, args.construction)
    package = read_json(args.construction)
    if package["selected_block_ids"] != read_json(selection)["block_ids"]:
        raise ValueError("construction does not use approved IDs")
    files = source_closure([
        *Path(__file__).parent.glob("*.py"),
        *(REPO / "src/gfmcd/splicing").glob("register_operator_*.py"),
        *(REPO / "tests").glob("test_register_operator*.py"), REPO / "tests/test_motif_cleanliness.py"])
    files.update({REPO / "pyproject.toml", REPO / "uv.lock", selection, protocol, specification,
                  args.construction.resolve(), args.construction_audit.resolve()})
    for record in package["inputs"].values():
        path = Path(record["path"]).resolve()
        if sha256_file(path) != record["sha256"]:
            raise ValueError("construction input binding changed")
        files.add(path)
    model = REPO / "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"
    names = ("pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
    checkpoint_files = {name: sha256_file(model / name) for name in names}
    if checkpoint_files["pytorch_model.bin"] != "2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d":
        raise ValueError("incorrect official checkpoint")
    files.update(model / name for name in names)
    inputs_dir = Path(package["inputs"]["frozen_cohort"]["path"]).parent
    historical = {"cohort": inputs_dir / "crossed_direct_edge_replication_cohort_v1.json",
        "native": inputs_dir / "crossed_direct_edge_replication_native.json",
        "cells": REPO / ".claude/worktrees/practical-hellman-b38d21/results/splicebert_exon_edge_compatibility/crossed_direct_edge_replication_cells.json.gz",
        "audit": REPO / "results/splicebert_exon_edge_compatibility/crossed_direct_edge_replication_audit.json"}
    files.update(historical.values())
    files.update(Path("/Users/cberg/Research/data/genomes") / name for name in
                 ("gencode.v50.basic.annotation.gtf.gz", "gencode.v50.annotation.gtf.gz"))
    result_dir = args.result_dir.resolve()
    filenames = {"smoke": "smoke_v1.json", "calibration": "statistical_calibration_v1.json",
        "calibration_records": "statistical_calibration_v1.records.jsonl.gz",
        "calibration_attempt": "statistical_calibration_v1.attempt.json",
        "calibration_manifest": "statistical_calibration_v1.manifest.json",
        "fidelity": "historical_fidelity_v1.json.gz", "fidelity_audit": "historical_fidelity_v1.audit.json",
        "native": "native_v1.json.gz", "cells": "cells_v1.json.gz", "analysis": "analysis_v1.json",
        "result_audit": "analysis_v1.audit.json", "final": "result_v1.json"}
    for stage in ("fidelity", "native", "cells"):
        filenames[stage + "_attempt"] = ("historical_fidelity" if stage == "fidelity" else stage) + "_v1.attempt.json"
    outputs = {key: str(result_dir / name) for key, name in filenames.items()}
    for key, name in outputs.items():
        if not key.endswith("_manifest"):
            guard_paths(name)
        elif Path(name).exists() or Path(name).is_symlink():
            raise FileExistsError(name)
    versions = {key: importlib.metadata.version(key) for key in ("torch", "transformers", "numpy", "scipy", "pyfaidx")}
    expected = {"torch": "2.13.0", "transformers": "5.13.0", "numpy": "2.5.1", "pyfaidx": "0.9.0.4"}
    if any(versions[key] != version for key, version in expected.items()) or platform.python_version() != "3.13.12":
        raise ValueError("frozen historical runtime versions changed")
    records = {(str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)): sha256_file(path) for path in sorted(files)}
    result = {"schema": "gfmcd.register_operator.freeze.v1", "status": "FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_choices": {"span_nt": 120, "blocks": 342, "exact_messages": ["120 only, any other decisions I have to make", "342"],
                          "cohort_tier": "outcome-aware operator verification", "resolution_limit_unchanged": .05},
        "files": records, "outputs": outputs, "construction_path": str(args.construction.resolve()),
        "construction_audit_path": str(args.construction_audit.resolve()), "selection_sha256": sha256_file(selection),
        "selection_path": str(selection), "specification_path": str(specification),
        "model_path": str(model), "fasta_path": str(Path(package["inputs"]["fasta"]["path"]).resolve()),
        "historical_inputs": {k: str(v) for k, v in historical.items()}, "checkpoint_files": checkpoint_files,
        "calibration": frozen_settings(), "runtime": {"versions": versions, "python": platform.python_version(),
            "platform": platform.platform(), "torch_num_threads": 1, "device": "mps", "dtype": "float32", "batch_size": 16,
            "attention_implementation": "sdpa",
            "hardware_model": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()},
        "numerical_thresholds": {"historical_raw_absolute": 1e-4, "historical_logp_absolute": 1e-4,
            "repeat_absolute": 1e-8, "historical_inference_atol_rtol": 1e-10, "resolution_mde80_max": .05},
        "exposure": {"new_model_loaded": False, "new_cc_rows": 0, "stochastic_calibration_started": False}}
    print(json.dumps(write_artifact(args.out, result, inputs=sorted(files),
        experiment="register-operator:protocol-freeze", config=result["owner_choices"],
        seeds={"bootstrap": 20260803, "calibration_development": 2026101601,
               "calibration_null": 2026101602, "calibration_plants": 2026101603})))


if __name__ == "__main__":
    main()
