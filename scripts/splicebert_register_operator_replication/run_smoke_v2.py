#!/usr/bin/env python3
"""Post-freeze A1 mechanics; executes only the explicitly frozen test inventory."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import sha256_file
from gfmcd.splicing.register_operator_io import guard_paths, write_artifact
from gfmcd.splicing.register_operator_io_v2 import verify_freeze
from gfmcd.splicing.register_operator_statistics_v2 import METHOD
from calibrate_statistics_v2 import run_smoke


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    args = p.parse_args()
    freeze = verify_freeze(args.freeze, REPO)
    digest = sha256_file(args.freeze)
    out = Path(freeze["outputs"]["smoke"])
    guard_paths(out)
    tests = freeze["tests"]
    if not tests or len(tests) != len(set(tests)) or any(
            not name.startswith("tests/") or name not in freeze["files"] for name in tests):
        raise ValueError("smoke requires a nonempty source-bound test inventory")
    started = datetime.now(timezone.utc).isoformat()
    failures = []
    test_result = {"paths": tests, "exit_code": None, "stdout": "", "stderr": ""}
    statistical = {"status": "NOT_RUN"}
    try:
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests],
                             cwd=REPO, text=True, capture_output=True)
        test_result.update(exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr)
        statistical = run_smoke()
    except Exception as exc:
        failures.append({"stage": "mechanics", "error": f"{type(exc).__name__}: {exc}"})
    try:
        verify_freeze(args.freeze, REPO)
        if sha256_file(args.freeze) != digest:
            raise ValueError("freeze changed during smoke")
    except Exception as exc:
        failures.append({"stage": "post_run_bindings", "error": f"{type(exc).__name__}: {exc}"})
    result = {"schema": "gfmcd.register_operator.smoke.v2", "freeze_sha256": digest,
        "inference_method": METHOD, "created_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures and test_result["exit_code"] == 0 and statistical["status"] == "SMOKE_PASS" else "FAIL",
        "tests": test_result, "failures": failures,
        "statistical_memorials": statistical, "model_loaded": False, "new_cc_outputs": 0}
    print(json.dumps(write_artifact(out, result, inputs=[args.freeze, Path(__file__),
        *[REPO / t for t in tests]], experiment="register-operator:smoke-v2",
        seeds={"bootstrap_toy": 20260803})))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
