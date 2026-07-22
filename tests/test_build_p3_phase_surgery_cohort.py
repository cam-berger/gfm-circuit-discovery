from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest

from gfmcd.splicing.p3_closure import ContextBlock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/splicebert/build_p3_phase_surgery_cohort.py"
SPEC = importlib.util.spec_from_file_location("build_p3_phase_surgery_cohort", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_choose_blocks_is_shift_safe_and_spatially_distinct():
    candidates = tuple(ContextBlock(start, start + 36, "cds") for start in range(0, 80))
    chosen = MODULE.choose_shift_safe_blocks(candidates, n=2, maximum_shift=10, table_length=100)
    assert len(chosen) == 2
    assert chosen[0].start >= 9
    assert chosen[-1].end <= 89
    assert chosen[1].start - chosen[0].start >= 12


def test_role_parser_rejects_unknown_role():
    assert MODULE.parse_roles("development") == ("development",)
    with pytest.raises(Exception):
        MODULE.parse_roles("validation")
