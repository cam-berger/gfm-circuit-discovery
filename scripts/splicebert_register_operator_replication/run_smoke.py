#!/usr/bin/env python3
"""Post-freeze measurement-path memorials, with no real model request."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import sha256_file
from gfmcd.splicing.register_operator_io import guard_paths, verify_freeze, write_artifact
from calibrate_statistics import run_smoke


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    args = p.parse_args()
    freeze = verify_freeze(args.freeze, REPO)
    out = Path(freeze["outputs"]["smoke"])
    guard_paths(out)
    tests = sorted(str(p.relative_to(REPO)) for p in (REPO / "tests").glob("test_register_operator*.py"))
    tests.append("tests/test_motif_cleanliness.py")
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=REPO, text=True, capture_output=True)
    statistical = run_smoke()
    verify_freeze(args.freeze, REPO)
    result = {"schema": "gfmcd.register_operator.smoke.v1", "freeze_sha256": sha256_file(args.freeze),
        "status": "PASS" if run.returncode == 0 and statistical["status"] == "SMOKE_PASS" else "FAIL",
        "tests": {"paths": tests, "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr},
        "statistical_memorials": statistical, "model_loaded": False, "new_cc_outputs": 0}
    print(json.dumps(write_artifact(out, result, inputs=[args.freeze, Path(__file__), *[REPO / t for t in tests]],
        experiment="register-operator:smoke", seeds={"bootstrap_toy": 20260803})))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
