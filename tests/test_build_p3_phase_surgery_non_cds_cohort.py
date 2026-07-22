from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert/build_p3_phase_surgery_non_cds_cohort.py"
)
SPEC = importlib.util.spec_from_file_location("build_p3_phase_surgery_non_cds_cohort", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_builder_reuses_the_frozen_role_namespace():
    assert MODULE.NAMESPACE == "p3-closure-v1"
    assert dict(MODULE.ROLE_WEIGHTS) == {"development": 2, "confirmation": 2, "reserve": 1}
