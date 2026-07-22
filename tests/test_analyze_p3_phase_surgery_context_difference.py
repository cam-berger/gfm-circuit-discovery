from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert/analyze_p3_phase_surgery_context_difference.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_p3_phase_surgery_context_difference", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_context_family_is_fixed():
    assert MODULE.ENDPOINTS == ("coupling", "phasor_signed")
    assert MODULE.SESOI["coupling"] == 0.05
