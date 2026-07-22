from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert/analyze_p3_phase_surgery_controls.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_p3_phase_surgery_controls", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_add_one_rank_p_has_exact_resolution_and_tie_is_conservative():
    controls = [float(index) for index in range(19)]
    assert MODULE.add_one_rank_p(20.0, controls) == pytest.approx(0.05)
    assert MODULE.add_one_rank_p(18.0, controls) == pytest.approx(0.10)


def test_add_one_rank_p_refuses_too_few_controls():
    with pytest.raises(ValueError, match="at least 19"):
        MODULE.add_one_rank_p(1.0, [0.0] * 18)
