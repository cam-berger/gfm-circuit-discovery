from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert/make_fig5_p3_phase_surgery_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location("make_fig5_p3_phase_surgery_confirmation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_figure_data_refuses_failed_instrument():
    with pytest.raises(RuntimeError, match="failed instrument"):
        MODULE.build_figure_data({
            "schema": "gfmcd.p3_phase_surgery.confirmation_analysis.v1",
            "status": "CONFIRMATORY_ADJUDICATED",
            "all_instrument_gates_pass": False,
        })
