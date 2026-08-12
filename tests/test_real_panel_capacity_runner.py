"""Task 6 (Part B) real-panel capacity runner: pure-function coverage.

Covers ``scripts/splicebert_positional_clock/run_real_panel_capacity.py`` without needing the
2.34 GB census or the untracked C43 cells file: every layer (B1 variance decomposition, B2
max-flow ceilings, B3 power at measured components, B4 required cross-lineage correlation, B5
exclusion tallies) is a pure function taking small, hand-verifiable synthetic inputs. The runner's
own hard NEEDS_CONTEXT assertions against the task-6 brief's literals are exercised here too, with
both a matching and a deliberately wrong observation.
"""

from __future__ import annotations

import importlib.util
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


RUNNER = _load("run_real_panel_capacity_test", "run_real_panel_capacity.py")


# -------------------------------------------------------------------------------------------
# B1: sign-free variance decomposition
# -------------------------------------------------------------------------------------------


def _synthetic_cell_rows() -> list[dict]:
    """Two blocks, hand-verifiable variance components.

    Block A: matched c_ac = [1, 2, 3] (mean 2), mismatched = [0]*6 (mean 0) -> Psi_A = 2.
    Block B: matched c_ac = [4, 5, 6] (mean 5), mismatched = [1]*6 (mean 1) -> Psi_B = 4.

    sigma_c2 = (sum of squared within-group deviations) / df = (2+0+2+0) / (2+5+2+5) = 4/14.
    Var(Psi_b) = ((2-3)**2 + (4-3)**2) / (2-1) = 2.
    sigma_u2 = 2 - (4/14)*(1/3 + 1/6) = 2 - (4/14)*0.5.
    rho = sigma_u2 / 2.
    """

    def rows(block_id: str, matched: list[float], mismatched: list[float]) -> list[dict]:
        out = []
        for value in matched:
            out.append({"block_id": block_id, "compatible": True, "effects": {"c_ac": value}})
        for value in mismatched:
            out.append({"block_id": block_id, "compatible": False, "effects": {"c_ac": value}})
        return out

    return rows("A", [1, 2, 3], [0, 0, 0, 0, 0, 0]) + rows("B", [4, 5, 6], [1, 1, 1, 1, 1, 1])


def test_variance_components_from_cell_rows_matches_hand_computation():
    components = RUNNER.variance_components_from_cell_rows(_synthetic_cell_rows())

    assert components["n_cell_rows"] == 18
    assert components["n_blocks"] == 2
    assert components["pooled_within_block_within_group_df"] == 14
    assert components["pooled_within_block_within_group_variance"] == pytest.approx(4 / 14)
    assert components["across_block_variance_psi_b"] == pytest.approx(2.0)
    assert components["block_sd"] == pytest.approx(math.sqrt(2.0))
    expected_sigma_u2 = 2.0 - (4 / 14) * 0.5
    assert components["sigma_u2"] == pytest.approx(expected_sigma_u2)
    assert components["rho"] == pytest.approx(expected_sigma_u2 / 2.0)


def test_variance_components_never_returns_a_mean_sign_or_individual_psi_b():
    """Sign-free restriction: the returned dict carries only variance-shaped quantities."""

    components = RUNNER.variance_components_from_cell_rows(_synthetic_cell_rows())
    forbidden_keys = {"mean", "psi_b", "psi_b_values", "sign", "direction", "grand_mean"}
    assert not (set(components) & forbidden_keys)


def test_variance_components_rejects_a_block_with_the_wrong_cell_shape():
    bad_rows = [
        {"block_id": "A", "compatible": True, "effects": {"c_ac": 1.0}},
        {"block_id": "A", "compatible": False, "effects": {"c_ac": 0.0}},
    ]
    with pytest.raises(ValueError, match="expected exactly 3/6"):
        RUNNER.variance_components_from_cell_rows(bad_rows)


def test_assert_b1_matches_expected_accepts_the_real_values_and_rejects_a_drifted_one():
    real = {
        "n_cell_rows": 3456,
        "n_blocks": 384,
        "pooled_within_block_within_group_df": 2688,
        "pooled_within_block_within_group_variance": 0.17703982542638141,
        "across_block_variance_psi_b": 0.14012168638599368,
        "block_sd": 0.3743283136312209,
        "sigma_u2": 0.05160177367280297,
        "rho": 0.368264006833713,
    }
    RUNNER._assert_b1_matches_expected(real)  # must not raise

    drifted = dict(real, rho=0.5)
    with pytest.raises(SystemExit, match="NEEDS_CONTEXT"):
        RUNNER._assert_b1_matches_expected(drifted)


# -------------------------------------------------------------------------------------------
# B2: real-panel ceilings by exact max-flow
# -------------------------------------------------------------------------------------------


def test_largest_n_satisfying_finds_the_exact_boundary():
    assert RUNNER._largest_n_satisfying(lambda n: n <= 17, 1000) == 17
    assert RUNNER._largest_n_satisfying(lambda n: n <= 0, 1000) == 0
    assert RUNNER._largest_n_satisfying(lambda n: False, 1000) == 0


def _disjoint_plain_lattice_slot_sets() -> dict[str, set[str]]:
    """Four disjoint 100-gene phase slots (union 400) plus a large scaffold.

    Sized so `ceiling_with_assignment_slack` (bound by the four-slot union's own 25% margin,
    N <= 300/4 = 75) is strictly tighter than `exact_plain_ceiling` (bound by each disjoint slot's
    own 100 genes, N <= 100) -- demonstrating the two really are different constraints, not just
    coincidentally equal as they are on the real pool.
    """

    def genes(prefix: str, n: int) -> set[str]:
        return {f"{prefix}_{i}" for i in range(n)}

    return {
        "scaffold": genes("scaffold", 1000),
        "donor_phase_0": genes("donor0", 100),
        "donor_phase_1": genes("donor1", 100),
        "acceptor_phase_0": genes("acceptor0", 100),
        "acceptor_phase_1": genes("acceptor1", 100),
    }


def test_exact_plain_ceiling_and_assignment_slack_ceiling_are_genuinely_different_constraints():
    slot_sets = _disjoint_plain_lattice_slot_sets()

    assert RUNNER.exact_plain_ceiling(slot_sets) == 100
    assert RUNNER.ceiling_with_assignment_slack(slot_sets) == 75

    union4 = len(set().union(*(slot_sets[s] for s in RUNNER.PLAIN_LATTICE_PHASE_SLOTS)))
    assert union4 == 400


def test_rotated_ceiling_is_self_consistent_at_its_own_boundary():
    """Not a hand-predicted literal (allocate_blocks' largest-remainder rounding makes that
    error-prone to compute by hand for an arbitrary fixture) -- instead checks the property the
    binary search itself claims: feasible exactly at the boundary, infeasible one block later.
    """

    def genes(prefix: str, n: int) -> set[str]:
        return {f"{prefix}_{i}" for i in range(n)}

    slot_sets = {
        "scaffold": genes("scaffold", 10000),
        "donor_phase_0": genes("d0", 300),
        "donor_phase_1": genes("d1", 300),
        "donor_phase_2": genes("d2", 300),
        "acceptor_phase_0": genes("a0", 300),
        "acceptor_phase_1": genes("a1", 300),
        "acceptor_phase_2": genes("a2", 300),
    }

    ceiling = RUNNER.rotated_ceiling(slot_sets)
    assert ceiling > 0

    def is_feasible(n: int) -> bool:
        demands = RUNNER.rotated_demands(n)
        real_demands = {slot: demands[slot] for slot in RUNNER.SLOT_ORDER}
        return RUNNER.feasibility(slot_sets, real_demands)["feasible"]

    assert is_feasible(ceiling) is True
    assert is_feasible(ceiling + 1) is False


def test_assert_b2_matches_expected_accepts_real_values_and_rejects_a_drifted_one():
    RUNNER._assert_b2_matches_expected(2904, 726, {0: 1869, 1: 1370, 2: 800}, 647)  # must not raise
    with pytest.raises(SystemExit, match="NEEDS_CONTEXT"):
        RUNNER._assert_b2_matches_expected(2904, 700, {0: 1869, 1: 1370, 2: 800}, 647)


# -------------------------------------------------------------------------------------------
# B3: power at measured components -- regression-pinned against the real run's own rho/V9.
# -------------------------------------------------------------------------------------------

_REAL_RHO = 0.368264006833713
_REAL_V9 = 0.14012168638599368


def test_power_at_measured_components_matches_the_real_run():
    power = RUNNER.power_at_measured_components(_REAL_RHO, _REAL_V9)

    assert power["psi_plain_2x2"]["blocks_needed"] == 718
    assert power["m_pm_plain_2x2"]["blocks_needed"] == 996
    assert power["psi_full_3x3"]["blocks_needed"] == 440
    assert power["m_pm_full_3x3"]["blocks_needed"] == 533


def test_direction_endpoint_estimate_class_matches_a2s_calibration_rule():
    plain = RUNNER._direction_endpoint_block(996, matched=2, delta_cells=1)
    assert plain["estimate_class"] == "lower_bound_only"
    assert "lower_bound_reason" in plain

    full = RUNNER._direction_endpoint_block(533, matched=3, delta_cells=3)
    assert full["estimate_class"] == "calibrated"
    assert "lower_bound_reason" not in full


def test_assert_b3_matches_expected_accepts_real_values_and_rejects_a_drifted_one():
    power = RUNNER.power_at_measured_components(_REAL_RHO, _REAL_V9)
    RUNNER._assert_b3_matches_expected(power)  # must not raise

    drifted = dict(power)
    drifted["psi_plain_2x2"] = dict(power["psi_plain_2x2"], blocks_needed=1)
    with pytest.raises(SystemExit, match="NEEDS_CONTEXT"):
        RUNNER._assert_b3_matches_expected(drifted)


# -------------------------------------------------------------------------------------------
# B4: cross-lineage reframe -- required same-block correlation, no block count emitted.
# -------------------------------------------------------------------------------------------


def test_required_cross_lineage_correlation_matches_the_real_run():
    # The real transfer panel is the PLAIN {0,1} lattice, so the per-block variance is
    # (2 - rho) * V9, NOT C43's raw V9. Passing V9 here is the exact defect this test now guards:
    # it is the 3x3 value, and using it understates the required correlation as 0.382 / 0.175.
    var_plain = RUNNER._var_psi_at(2, 2, _REAL_RHO, _REAL_V9)
    r_544 = RUNNER.required_cross_lineage_correlation(544, var_plain)
    r_726 = RUNNER.required_cross_lineage_correlation(726, var_plain)
    assert r_544 == pytest.approx(0.621, abs=5e-4)
    assert r_726 == pytest.approx(0.494, abs=5e-4)


def test_using_the_three_by_three_variance_understates_the_required_correlation():
    """Regression guard for a corrected defect: V9 is the 3x3 value, not the plain-lattice one."""

    wrong = RUNNER.required_cross_lineage_correlation(544, _REAL_V9)
    right = RUNNER.required_cross_lineage_correlation(
        544, RUNNER._var_psi_at(2, 2, _REAL_RHO, _REAL_V9)
    )
    assert wrong == pytest.approx(0.382, abs=5e-4)
    assert right == pytest.approx(0.621, abs=5e-4)
    assert right > wrong


def test_cross_lineage_reframe_never_emits_a_delta_psi_block_count():
    reframe = RUNNER.cross_lineage_reframe(_REAL_V9, rho=_REAL_RHO, usable_ceiling=544, absolute_ceiling=726)

    assert reframe["required_same_block_correlation"]["544"]["status"] == "requires_pilot"
    assert reframe["required_same_block_correlation"]["726"]["status"] == "requires_pilot"
    delta_psi_field = reframe["delta_psi_blocks_needed"]
    assert isinstance(delta_psi_field, str)
    assert not any(ch.isdigit() for ch in delta_psi_field), "no block count may appear here"


# -------------------------------------------------------------------------------------------
# B5: exclusion tallies (sole-reason vs raw) and the phase-equality fact.
# -------------------------------------------------------------------------------------------


def test_exclusion_reason_tallies_separates_sole_from_multi_reason():
    exclusions = [
        {"reasons": ["freshness_registry_gene"]},
        {"reasons": ["freshness_registry_gene"]},
        {"reasons": ["comprehensive_local_coding_register_conflict"]},
        {"reasons": ["comprehensive_local_coding_register_conflict", "freshness_registry_gene"]},
        {"reasons": ["basic_comprehensive_phase_conflict", "comprehensive_coordinate_transcript_phase_conflict"]},
    ]

    tallies = RUNNER.exclusion_reason_tallies(exclusions)

    assert tallies["total"] == 5
    assert tallies["sole_reason_counts"] == {
        "freshness_registry_gene": 2,
        "comprehensive_local_coding_register_conflict": 1,
    }
    assert tallies["multi_reason_total"] == 2
    # raw counts every occurrence, including inside multi-reason records
    assert tallies["raw_reason_counts"]["freshness_registry_gene"] == 3
    assert tallies["raw_reason_counts"]["comprehensive_local_coding_register_conflict"] == 2
    assert tallies["raw_reason_counts"]["basic_comprehensive_phase_conflict"] == 1


def test_count_key_occurrences_recurses_dicts_and_lists():
    node = {
        "a": {"upstream_downstream_phase_equal": True, "nested": {"upstream_downstream_phase_equal": False}},
        "b": [{"upstream_downstream_phase_equal": True}, {"other": 1}],
        "c": "upstream_downstream_phase_equal",  # a STRING value must not be counted as a key
    }
    assert RUNNER._count_key_occurrences(node, "upstream_downstream_phase_equal") == 3


def test_tallying_candidates_yields_every_candidate_unchanged_while_tallying():
    candidates = [
        {"gene_id": "g1", "eligibility_checks": {"upstream_downstream_phase_equal": True}},
        {"gene_id": "g2", "eligibility_checks": {"upstream_downstream_phase_equal": True}},
    ]
    tally: dict[str, int] = {}
    passed_through = list(
        RUNNER._tallying_candidates(iter(candidates), tally, "upstream_downstream_phase_equal")
    )

    assert passed_through == candidates
    assert tally["upstream_downstream_phase_equal"] == 2


def test_assert_b5_matches_expected_accepts_real_values_and_rejects_a_drifted_one():
    tallies = {
        "total": 282057,
        "sole_reason_counts": dict(RUNNER._B5_EXPECTED_SOLE_REASONS),
        "raw_reason_counts": dict(RUNNER._B5_EXPECTED_DISAGREEMENT_RAW),
        "multi_reason_total": 28791,
    }
    RUNNER._assert_b5_matches_expected(tallies, 130922, 65461)  # must not raise

    with pytest.raises(SystemExit, match="NEEDS_CONTEXT"):
        RUNNER._assert_b5_matches_expected(tallies, 1, 65461)


def test_assert_b5_matches_expected_rejects_a_disagreement_reason_appearing_as_a_sole_reason():
    """The three annotation-disagreement reasons are expected to NEVER appear alone; if one did,
    that would itself be a fact worth stopping on, not silently absorbing.
    """

    tallies = {
        "total": 282057,
        "sole_reason_counts": dict(
            RUNNER._B5_EXPECTED_SOLE_REASONS, basic_comprehensive_phase_conflict=1
        ),
        "raw_reason_counts": dict(RUNNER._B5_EXPECTED_DISAGREEMENT_RAW),
        "multi_reason_total": 28790,
    }
    with pytest.raises(SystemExit, match="NEEDS_CONTEXT"):
        RUNNER._assert_b5_matches_expected(tallies, 130922, 65461)


# -------------------------------------------------------------------------------------------
# Honesty fields / no-gene-list rule on a full assembled artifact.
# -------------------------------------------------------------------------------------------


def test_assemble_artifact_is_json_safe_and_leaks_no_gene_ids():
    import json
    import re

    b1 = dict(
        RUNNER.variance_components_from_cell_rows(_synthetic_cell_rows()),
        sign_free_restriction_note="x",
        rho_status={"type": "point_estimate_no_interval", "note": "x"},
    )
    b2 = {
        "lattice": "plain_zero_one",
        "slots_demanded_per_block": list(RUNNER.PLAIN_LATTICE_PHASE_SLOTS) + ["scaffold"],
        "union_of_four_phase01_module_slots": 400,
        "exact_ceiling": 100,
        "ceiling_with_25pct_assignment_slack": 75,
        "ceiling_with_25pct_assignment_slack_note": "x",
        "donor_acceptor_phase_intersections": {"0": 1, "1": 2, "2": 3},
        "rotated_lattice_ceiling_for_comparison": 90,
        "rotated_design_withdrawn_reason": "x",
    }
    b3 = RUNNER.power_at_measured_components(_REAL_RHO, _REAL_V9)
    b4 = RUNNER.cross_lineage_reframe(_REAL_V9, rho=_REAL_RHO, usable_ceiling=75, absolute_ceiling=100)
    b5 = {
        "census_exclusions_total": 5,
        "sole_reason_counts": {"freshness_registry_gene": 2},
        "multi_reason_exclusions": 2,
        "load_bearing_note": "x",
        "upstream_downstream_phase_equal": {
            "occurrences": 2,
            "candidates": 1,
            "never_an_exclusion_reason": True,
            "note": "x",
        },
        "annotation_disagreement_reason_counts": {"basic_comprehensive_phase_conflict": 1},
        "annotation_disagreement_note": "x",
    }
    receipt = {"mode": "file", "path": "fake/path.json", "sha256": "0" * 64, "size_bytes": 1}

    artifact = RUNNER.assemble_artifact(
        census_receipt=receipt,
        certificate_receipt=receipt,
        cells_receipt=receipt,
        analysis_receipt=receipt,
        slot_hash_verification={"per_slot": {"scaffold": True}, "all_verified": True},
        b1=b1,
        b2=b2,
        b3=b3,
        b4=b4,
        b5=b5,
    )

    assert artifact["model_execution"] is False
    assert artifact["checkpoint_loaded"] is None
    assert artifact["cohort_selection_attempted"] is False
    assert artifact["stage"] == "construction"
    assert artifact["real_panel_constructible_at_sesoi"] is False

    json.dumps(artifact)  # must succeed: no stray set/other non-JSON-safe objects
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
