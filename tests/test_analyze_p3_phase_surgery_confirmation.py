from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert/analyze_p3_phase_surgery_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_p3_phase_surgery_confirmation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_add_one_rank_is_exact_with_19_controls():
    assert MODULE.add_one_rank_p(1.0, [float(index) / 100 for index in range(19)]) == 0.05


def test_adjudication_never_uses_secondary_to_rescue_primary():
    primary = {"material_positive": False, "classification": "inconclusive", "ci": [-0.1, 0.2]}
    direction = {"specificity_pass": True}
    context = {"coding_enrichment_pass": True}
    result = MODULE.adjudicate(primary, direction, context, {"gate": True})
    assert result["code"] == "INCONCLUSIVE"


def test_single_endpoint_interval_uses_whole_interval_for_materiality():
    result = MODULE.single_endpoint_interval(
        np.linspace(0.08, 0.14, 30), seed=7, bootstrap=199
    )
    assert result["ci"][0] > 0.05
    assert result["material_positive"]
