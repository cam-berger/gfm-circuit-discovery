from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/splicebert/build_p3_closure_cohort.py"
SPEC = importlib.util.spec_from_file_location("build_p3_closure_cohort", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_role_parser_and_chromosome_normalization():
    assert MODULE.parse_roles("development,reserve") == ("development", "reserve")
    assert MODULE.normalized_chrom("1") == "chr1"
    assert MODULE.normalized_chrom("chr2") == "chr2"
    with pytest.raises(Exception):
        MODULE.parse_roles("test")


def test_stable_example_rank_is_deterministic():
    assert MODULE.stable_example_rank("example") == MODULE.stable_example_rank("example")
    assert MODULE.stable_example_rank("example") != MODULE.stable_example_rank("other")
