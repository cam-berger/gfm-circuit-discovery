from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "splicebert"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BG1 = _load("run_bg1_boundary_grammar_test", "run_bg1_boundary_grammar.py")
PREFLIGHT = _load("run_bg1_boundary_preflight_test", "run_bg1_boundary_preflight.py")


def test_primary_summary_and_verdict_are_deterministic_and_materiality_gated():
    rng = np.random.default_rng(7)
    matrix = rng.normal(loc=(0.20, 0.16), scale=(0.03, 0.04), size=(80, 2))
    kwargs = dict(
        names=("minus3", "plus1"),
        seed=19,
        n_bootstrap=199,
        n_permutation=199,
        p_two_sided=False,
    )
    first = BG1.simultaneous_summary(matrix, **kwargs)
    second = BG1.simultaneous_summary(matrix, **kwargs)
    assert first == second
    assert first["p_alternative"] == "greater"
    assert BG1.target_verdict(first) == "SUPPORTED_VERIFICATION"

    threshold_crossing = dict(first)
    threshold_crossing["simultaneous_95_ci"] = [[0.04, 0.20], [0.10, 0.20]]
    assert BG1.target_verdict(threshold_crossing) == "INCONCLUSIVE"


def test_localization_uses_holm_and_requires_material_attenuation():
    summaries = {
        "minus3_vs_minus4": {
            "inference_linear": {
                    "max_family_add_one_p": [0.03],
                "simultaneous_95_ci": [[0.06, 0.20]],
            }
        },
        "plus1_vs_plus2": {
            "inference_linear": {
                    "max_family_add_one_p": [0.04],
                "simultaneous_95_ci": [[0.06, 0.20]],
            }
        },
    }
    verdict = BG1.apply_attenuation_decisions(
        summaries,
        {"minus3_vs_minus4": True, "plus1_vs_plus2": True},
    )
    assert verdict == "JOINT_LOCALIZATION_INCONCLUSIVE"
    assert summaries["minus3_vs_minus4"]["holm_p_across_enabled_arms"] == pytest.approx(0.06)
    assert summaries["plus1_vs_plus2"]["holm_p_across_enabled_arms"] == pytest.approx(0.06)

    summaries["plus1_vs_plus2"]["inference_linear"]["max_family_add_one_p"] = [0.01]
    assert BG1.apply_attenuation_decisions(
        summaries,
        {"minus3_vs_minus4": True, "plus1_vs_plus2": False},
    ) == "JOINT_LOCALIZATION_INCONCLUSIVE"
    assert summaries["plus1_vs_plus2"]["verdict"] == "PRESPECIFIED_DIAGNOSTIC_ONLY"


def test_projected_mde_uses_target_sample_size():
    matrix = np.column_stack((np.arange(20, dtype=float), np.arange(20, dtype=float) * 2))
    small = PREFLIGHT.projected_mde80(matrix, critical_value=2.0, target_n=25)
    large = PREFLIGHT.projected_mde80(matrix, critical_value=2.0, target_n=100)
    assert np.asarray(small) == pytest.approx(2 * np.asarray(large))


def test_calibration_exercises_the_joint_interval_and_signflip_decision():
    rng = np.random.default_rng(4)
    matrix = rng.normal(0.0, 0.04, size=(32, 2))
    result = PREFLIGHT.analysis_calibration(
        matrix,
        critical_value=2.2,
        seed=31,
        replicates=40,
        n_permutation=99,
    )
    assert result["claim_threshold"] == 0.05
    assert result["planted_total_effect"] == 0.10
    assert result["permutation_draws_per_world"] == 99
    assert 0.0 <= result["endpoint_familywise_false_positive_rate"] <= 1.0
    assert 0.0 <= result["joint_claim_worst_case_false_positive_rate"] <= 1.0
    assert len(
        result["joint_claim_composite_null_false_positive_rate_by_boundary_endpoint"]
    ) == 2
    assert 0.0 <= result["planted_power"] <= 1.0


def test_smoke_pass_is_instrument_only_and_canonical_gates_are_unchanged():
    checks = {
        "panel_construction": {"PASS": True},
        "planted_sensitivity": {"PASS": False},
        "target_resolution": {"PASS": False},
    }
    smoke = PREFLIGHT.preflight_pass_semantics(checks, smoke=True)
    assert smoke["PASS"] is True
    assert smoke["instrument_PASS"] is True
    assert smoke["target_authorization_PASS"] is False
    assert smoke["failed_non_authorizing_smoke_diagnostics"] == [
        "planted_sensitivity",
        "target_resolution",
    ]

    canonical = PREFLIGHT.preflight_pass_semantics(checks, smoke=False)
    assert canonical["PASS"] is False
    assert canonical["instrument_PASS"] is False
    assert canonical["target_authorization_PASS"] is False

    broken_instrument = PREFLIGHT.preflight_pass_semantics(
        {**checks, "panel_construction": {"PASS": False}}, smoke=True
    )
    assert broken_instrument["PASS"] is False
    assert broken_instrument["instrument_PASS"] is False


def test_frozen_membership_detects_any_order_or_sequence_drift():
    examples = [
        SimpleNamespace(
            example_id=f"e{index}",
            gene_id=f"g{index}",
            chrom="chr11",
            strand="+" if index % 2 else "-",
            donor_pos=3,
            next_acceptor_pos=70,
            sequence="C" * 68 + "AG" + "A" * 10,
        )
        for index in range(4)
    ]
    frozen = BG1.membership_declaration(examples)
    preflight = {"target_cohorts": {"val": {"target_behavior": frozen}}}
    BG1.require_frozen_membership(preflight, "val", "target_behavior", examples)
    with pytest.raises(RuntimeError, match="differs"):
        BG1.require_frozen_membership(
            preflight, "val", "target_behavior", list(reversed(examples))
        )
