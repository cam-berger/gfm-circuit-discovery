"""Planning must discard stratum means and must never masquerade as CC evidence."""
import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / (
    "scripts/splicebert_register_operator_replication/plan_power.py"
)
SPEC = importlib.util.spec_from_file_location("register_operator_power_plan", SCRIPT)
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


def rows(values):
    return [{"block_id": str(i), "stratum": str(i // 2), "ac": {"psi_match": x}}
            for i, x in enumerate(values)]


def test_centered_variance_is_invariant_to_stratum_offsets_and_sign():
    reference = plan.variance_summary(rows([1.0, 3.0, -5.0, 1.0]))
    shifted = plan.variance_summary(rows([1001.0, 1003.0, 45.0, 51.0]))
    reversed_sign = plan.variance_summary(rows([-1.0, -3.0, 5.0, -1.0]))
    assert reference == shifted == reversed_sign
    # Exact fixed-stratum bootstrap: within-pair SSE = 2 + 18, N = 4.
    assert reference["exact_empirical_bootstrap_se"] == pytest.approx(math.sqrt(20) / 4)


def test_between_stratum_differences_are_not_counted_as_resolution_noise():
    result = plan.variance_summary(rows([1.0, 1.0, 100.0, 100.0]))
    assert result["exact_empirical_bootstrap_mde80"] == 0


def test_no_false_resolution_from_duplicate_blocks_or_singletons():
    bad = rows([1.0, 2.0])
    bad[1]["block_id"] = bad[0]["block_id"]
    with pytest.raises(ValueError):
        plan.variance_summary(bad)
    bad = rows([1.0, 2.0])
    bad[1]["stratum"] = "different"
    with pytest.raises(ValueError):
        plan.variance_summary(bad)


def test_projection_preserves_sd_scaling_and_quadratic_sample_cost():
    assert plan.project_mde(0.05, 384, 1536, 2.0) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        plan.project_mde(0.05, 384, 0, 1.0)
