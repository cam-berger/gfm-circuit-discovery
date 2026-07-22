from __future__ import annotations
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/splicebert/run_p3_phase_surgery_preflight.py"
SPEC = importlib.util.spec_from_file_location("run_p3_phase_surgery_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_package_firewall_refuses_confirmation(tmp_path):
    package = tmp_path / "package.json"
    package.write_text(json.dumps({
        "schema": "gfmcd.p3_phase_surgery.cohort.v1",
        "model_outcomes_read": False,
        "records": [{"eligible": True, "role": "confirmation"}],
    }))
    with pytest.raises(RuntimeError, match="refuses"):
        MODULE.load_development_package(package)


def test_max_abs_difference_reads_all_primary_plumbing_metrics():
    row = {key: 1.0 for key in (
        "clean_margin", "corrupt_margin", "coupling", "phasor_real", "phasor_imag",
        "phasor_signed", "phasor_amplitude",
    )}
    other = dict(row)
    other["phasor_signed"] = 1.25
    assert MODULE.max_abs_difference([row], [other]) == 0.25
