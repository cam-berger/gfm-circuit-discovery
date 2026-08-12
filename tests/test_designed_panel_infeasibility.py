"""Designed-panel infeasibility runner: pure-function coverage.

Covers ``scripts/splicebert_positional_clock/run_designed_panel_infeasibility.py``. Unlike the
real-panel sibling this runner's two inputs (the PBC v2 ladder artifact and the real-panel capacity
record) are small and committed, so most tests exercise the pure D1/D2/D3 functions directly against
hand-verifiable and regression-pinned values; a couple of tests also drive the small committed inputs
through the module's own input-reading/assertion helpers.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "splicebert_positional_clock"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load("run_designed_panel_infeasibility_test", "run_designed_panel_infeasibility.py")

_REAL_RHO = 0.368264006833713
_REAL_V9 = 0.14012168638599368


# -------------------------------------------------------------------------------------------
# D1: additive-capacity table -- regression-pinned against the audited literals.
# -------------------------------------------------------------------------------------------


def test_additive_capacity_table_matches_the_audited_literals():
    rows = RUNNER.additive_capacity_table(
        rho=_REAL_RHO,
        v9=_REAL_V9,
        real_usable_n=544,
        real_absolute_n=726,
        pool_union_size=3227,
    )
    RUNNER._assert_d1_matches_expected(rows)  # must not raise

    expected = [
        (440, 544, 3056, 0.053, "below the 0.25 rule"),
        (487, 544, 3148, 0.024, "below the 0.25 rule"),
        (440, 726, 3784, None, "infeasible"),
        (764, 544, 3704, None, "infeasible"),
    ]
    for row, (designed, real, spend, margin, verdict) in zip(rows, expected):
        assert row["designed_blocks_needed"] == designed
        assert row["real_blocks_needed"] == real
        assert row["union_spend"] == spend
        assert row["verdict"] == verdict
        if margin is None:
            assert row["margin"] is None
        else:
            assert row["margin"] == pytest.approx(margin, abs=5e-4)


def test_additive_capacity_row_uses_raw_not_ceiled_designed_blocks_for_spend():
    """The spend formula must use the pre-ceiling designed block count, not the displayed
    (ceiled) integer -- using the ceiled integer for row 2 (487, 544) gives 3150, not the
    audited 3148, a real (not rounding-noise) 2-unit gap. This is the specific defect this test
    guards against a future "simplification" that switches to the ceiled integer.
    """

    row = RUNNER.additive_capacity_row(
        row_label="x",
        designed_source="x",
        designed_raw_blocks=486.01933907208235,
        real_source="x",
        real_blocks=544,
        pool_union_size=3227,
    )
    assert row["designed_blocks_needed"] == 487  # displayed value is still the ceiling
    assert row["union_spend"] == 3148  # but spend is computed from the raw value
    wrong_spend_from_ceiling = 2 * 487 + 4 * 544
    assert wrong_spend_from_ceiling == 3150
    assert row["union_spend"] != wrong_spend_from_ceiling


def test_additive_capacity_row_verdict_boundaries():
    # Comfortably clears margin: small demand against a huge pool.
    clears = RUNNER.additive_capacity_row(
        row_label="x", designed_source="x", designed_raw_blocks=1.0, real_source="x",
        real_blocks=1, pool_union_size=1_000_000,
    )
    assert clears["verdict"] == "clears the 0.25 rule"
    assert clears["margin"] > 0.25

    # Below margin but still feasible: spend = 2*200 + 4*100 = 800 against pool 1000 -> margin 0.2.
    below = RUNNER.additive_capacity_row(
        row_label="x", designed_source="x", designed_raw_blocks=200.0, real_source="x",
        real_blocks=100, pool_union_size=1000,
    )
    assert below["verdict"] == "below the 0.25 rule"
    assert below["margin"] is not None and 0 <= below["margin"] < 0.25

    # Infeasible: spend exceeds the pool outright.
    infeasible = RUNNER.additive_capacity_row(
        row_label="x", designed_source="x", designed_raw_blocks=1000.0, real_source="x",
        real_blocks=1000, pool_union_size=10,
    )
    assert infeasible["verdict"] == "infeasible"
    assert infeasible["margin"] is None


def test_assert_d1_matches_expected_rejects_a_drifted_row():
    rows = RUNNER.additive_capacity_table(
        rho=_REAL_RHO, v9=_REAL_V9, real_usable_n=544, real_absolute_n=726, pool_union_size=3227,
    )
    RUNNER._assert_d1_matches_expected(rows)  # must not raise

    drifted = [dict(r) for r in rows]
    drifted[0]["union_spend"] = 1
    with pytest.raises(SystemExit, match="NEEDS_CONTEXT"):
        RUNNER._assert_d1_matches_expected(drifted)


# -------------------------------------------------------------------------------------------
# D2: corrected parent-pair variance band -- regression-pinned, including the one correction.
# -------------------------------------------------------------------------------------------


def test_designed_variance_band_matches_the_audited_literals_including_the_f05_correction():
    band = RUNNER.designed_variance_band(_REAL_RHO, _REAL_V9)
    by_f = {row["f"]: row for row in band}

    assert by_f[0.0]["n_psi"] == 487
    assert by_f[0.0]["n_m"] == 487
    assert by_f[0.25]["n_psi"] == 556
    assert by_f[0.25]["n_m"] == 579
    # CORRECTED: the task brief's own D2 table states 626 here; exact recomputation from the
    # brief's own formula gives 625 (verified three ways at the runner's assertion site). This
    # test pins the value the formula actually produces, not the brief's literal.
    assert by_f[0.5]["n_psi"] == 625
    assert by_f[0.5]["n_m"] == 672
    assert by_f[1.0]["n_psi"] == 764
    assert by_f[1.0]["n_m"] == 857

    for row in band:
        assert row["f_status"] == "unmeasured"


def test_no_f_makes_440_correct():
    """There is no f >= 0 at which the naive psi_full_3x3 figure (440) is correct for the
    designed panel: the f that would reproduce its variance is negative.
    """

    section = RUNNER.d2_corrected_variance_section(_REAL_RHO, _REAL_V9)
    assert section["no_f_makes_440_correct"] is True
    assert section["legacy_533_implicit_f"] == pytest.approx(0.125, abs=1e-3)


def test_implicit_f_solves_the_equation_it_claims_to():
    v9 = _REAL_V9
    rho = _REAL_RHO
    f = RUNNER.implicit_f(0.2, 0.5, rho, v9)
    reconstructed = 3 * rho * v9 + f * RUNNER.sig_c2(rho, v9) * 0.5
    assert reconstructed == pytest.approx(0.2, abs=1e-9)


def test_assert_d2_matches_expected_rejects_a_drifted_band():
    band = RUNNER.designed_variance_band(_REAL_RHO, _REAL_V9)
    RUNNER._assert_d2_matches_expected(band, 0.1256)  # must not raise

    drifted = [dict(row) for row in band]
    for row in drifted:
        if row["f"] == 0.5:
            row["n_psi"] = 1
    with pytest.raises(SystemExit, match="NEEDS_CONTEXT"):
        RUNNER._assert_d2_matches_expected(drifted, 0.1256)

    with pytest.raises(SystemExit, match="NEEDS_CONTEXT"):
        RUNNER._assert_d2_matches_expected(band, 0.9)


def test_var_psi_designed_and_var_dir_designed_agree_at_f_zero():
    """At f=0 the extra term vanishes for both formulas, so N_Psi and N_M coincide (487, 487)."""

    assert RUNNER.var_psi_designed(0.0, _REAL_RHO, _REAL_V9) == pytest.approx(
        RUNNER.var_dir_designed(0.0, _REAL_RHO, _REAL_V9)
    )


def test_designed_block_level_term_already_exceeds_v9_before_cell_noise():
    """3*rho*V9 (the designed panel's block-level term at f=0, before ANY cell noise) is
    already larger than V9 itself -- the headline fact D2 exists to record.
    """

    three_rho_v9 = 3.0 * _REAL_RHO * _REAL_V9
    assert three_rho_v9 > _REAL_V9
    assert three_rho_v9 / _REAL_V9 == pytest.approx(1.1049, abs=1e-3)


# -------------------------------------------------------------------------------------------
# D3: raw-periodicity confound -- the one arithmetic fact (span is a multiple of the period).
# -------------------------------------------------------------------------------------------


def test_d3_span_is_a_multiple_of_the_period():
    d3 = RUNNER.d3_raw_periodicity_confound()
    assert d3["span_nt"] == 120
    assert d3["period_nt"] == 3
    assert d3["span_is_multiple_of_period"] is True
    assert d3["span_cycles"] == 40
    assert d3["structural_not_a_power_problem"] is True


def test_d3_constructive_vs_destructive_magnitude_identity():
    """The trig identity D3's prose relies on: for two unit-magnitude components, in-phase sums
    to 2a; 120/240-degree-apart sums to a (sqrt(a^2 + a^2 - a*a) = a).
    """

    a = 1.0
    constructive = 2 * a
    destructive = math.sqrt(a**2 + a**2 - a * a)
    assert constructive == pytest.approx(2.0)
    assert destructive == pytest.approx(1.0)
    assert constructive > destructive


# -------------------------------------------------------------------------------------------
# D4: disposition.
# -------------------------------------------------------------------------------------------


def test_d4_disposition_fields():
    d4 = RUNNER.d4_disposition()
    assert d4["designed_panel_viable_as_primary_identifier"] is False
    assert d4["superseded_conclusion"] == "clears comfortably"
    assert "carrier-register" in d4["successor_primary_endpoint_note"]
    assert "not built yet" in d4["successor_not_further_described_note"]


# -------------------------------------------------------------------------------------------
# Input reading / verification against the two committed upstream artifacts.
# -------------------------------------------------------------------------------------------


def _fake_ladder_doc(min_cut_slots, feasible=False, min_cut_gene_count=3227):
    return {
        "rungs": [
            {
                "n_blocks": 909,
                "feasibility": {
                    "feasible": feasible,
                    "min_cut_slots": min_cut_slots,
                    "min_cut_gene_count": min_cut_gene_count,
                },
            }
        ]
    }


def test_read_pool_union_size_accepts_the_real_six_slot_certificate():
    doc = _fake_ladder_doc(RUNNER._LADDER_909_MIN_CUT_SLOTS_EXPECTED)
    pool_union_size, feas = RUNNER._read_pool_union_size(doc)
    assert pool_union_size == 3227
    assert feas["feasible"] is False


def test_read_pool_union_size_rejects_a_partial_min_cut():
    """If a future ladder rerun's N=909 min cut violates only a SUBSET of the six real slots,
    min_cut_gene_count would no longer equal the full pool union -- this must be caught, not
    silently trusted.
    """

    partial = ["donor_phase_0", "donor_phase_1"]
    doc = _fake_ladder_doc(partial)
    with pytest.raises(SystemExit, match="NEEDS_CONTEXT"):
        RUNNER._read_pool_union_size(doc)


def test_read_pool_union_size_rejects_a_feasible_rung():
    doc = _fake_ladder_doc(RUNNER._LADDER_909_MIN_CUT_SLOTS_EXPECTED, feasible=True)
    with pytest.raises(SystemExit, match="NEEDS_CONTEXT"):
        RUNNER._read_pool_union_size(doc)


def test_read_real_panel_inputs_extracts_the_expected_fields():
    doc = {
        "b1_measured_variance_components": {"rho": 0.5, "across_block_variance_psi_b": 0.2},
        "b2_real_panel_ceilings": {
            "ceiling_with_25pct_assignment_slack": 10,
            "exact_ceiling": 20,
            "union_of_four_phase01_module_slots": 30,
        },
        "b3_power_at_measured_components": {
            "psi_full_3x3": {"blocks_needed": 40},
            "m_pm_full_3x3": {"blocks_needed": 50},
        },
    }
    parsed = RUNNER._read_real_panel_inputs(doc)
    assert parsed == {
        "rho": 0.5,
        "v9": 0.2,
        "real_usable_n": 10,
        "real_absolute_n": 20,
        "real_four_slot_union": 30,
        "psi_full_3x3": 40,
        "m_pm_full_3x3": 50,
    }


def test_assert_inputs_match_expected_accepts_the_real_values_and_rejects_a_drifted_one():
    real_inputs = {
        "rho": 0.368264006833713,
        "v9": 0.14012168638599368,
        "real_usable_n": 544,
        "real_absolute_n": 726,
        "real_four_slot_union": 2904,
        "psi_full_3x3": 440,
        "m_pm_full_3x3": 533,
    }
    RUNNER._assert_inputs_match_expected(3227, real_inputs)  # must not raise

    with pytest.raises(SystemExit, match="NEEDS_CONTEXT"):
        RUNNER._assert_inputs_match_expected(1, real_inputs)

    drifted = dict(real_inputs, rho=0.9)
    with pytest.raises(SystemExit, match="NEEDS_CONTEXT"):
        RUNNER._assert_inputs_match_expected(3227, drifted)


def test_reads_the_real_committed_ladder_and_real_panel_artifacts():
    """End-to-end sanity against the actual committed inputs this runner is audited against."""

    ladder_doc = json.loads(RUNNER.DEFAULT_LADDER_PATH.read_text())
    real_panel_doc = json.loads(RUNNER.DEFAULT_REAL_PANEL_PATH.read_text())

    pool_union_size, _feas = RUNNER._read_pool_union_size(ladder_doc)
    real_inputs = RUNNER._read_real_panel_inputs(real_panel_doc)
    RUNNER._assert_inputs_match_expected(pool_union_size, real_inputs)  # must not raise

    assert pool_union_size == 3227
    assert real_inputs["rho"] == _REAL_RHO
    assert real_inputs["v9"] == _REAL_V9


# -------------------------------------------------------------------------------------------
# Artifact assembly -- honesty fields, JSON-safety, no gene IDs.
# -------------------------------------------------------------------------------------------


def test_assemble_artifact_is_json_safe_and_carries_the_required_honesty_fields():
    d1_rows = RUNNER.additive_capacity_table(
        rho=_REAL_RHO, v9=_REAL_V9, real_usable_n=544, real_absolute_n=726, pool_union_size=3227,
    )
    d2 = RUNNER.d2_corrected_variance_section(_REAL_RHO, _REAL_V9)
    d3 = RUNNER.d3_raw_periodicity_confound()
    d4 = RUNNER.d4_disposition()
    receipt = {"mode": "file", "path": "fake/path.json", "sha256": "0" * 64, "size_bytes": 1}

    artifact = RUNNER.assemble_artifact(
        ladder_receipt=receipt,
        real_panel_receipt=receipt,
        pool_union_size=3227,
        real_four_slot_union=2904,
        d1_rows=d1_rows,
        d2=d2,
        d3=d3,
        d4=d4,
    )

    assert artifact["model_execution"] is False
    assert artifact["checkpoint_loaded"] is None
    assert artifact["cohort_selection_attempted"] is False
    assert artifact["stage"] == "construction"
    assert artifact["designed_panel_viable_as_primary_identifier"] is False
    assert "not evidence about biology" in artifact["scope_note"]

    json.dumps(artifact)  # must succeed: no stray non-JSON-safe objects

    import re

    gene_id_pattern = re.compile(r"^ENSG\d")

    def check(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                check(key)
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)
        elif isinstance(node, str):
            assert not gene_id_pattern.match(node), f"gene id leaked into artifact: {node!r}"

    check(artifact)


def test_parse_args_requires_out_with_no_default():
    with pytest.raises(SystemExit):
        RUNNER.parse_args([])  # missing --out must fail, not silently default

    args = RUNNER.parse_args(["--out", "/tmp/x.json"])
    assert args.out == "/tmp/x.json"
