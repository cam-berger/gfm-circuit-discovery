"""PBC v2 capacity layers 2-4: rotated demand model, exact feasibility, power projection.

Three tests here are the load-bearing ones, and each fails against the implementation it replaces:

* `test_delta_balance_fails_for_a_plain_zero_one_lattice` -- the entire justification for rotating
  the phase pair. A plain {0,1} lattice gives one cell per mismatch direction and confounds delta
  with module identity.
* `test_hall_obstruction_is_invisible_to_counting_but_caught_by_flow` -- the entire justification
  for the max-flow layer. Per-slot counting returns PASS on that input.
* `test_three_by_three_calibration_is_exact_for_every_rho` -- an exact arithmetic anchor on the
  variance decomposition, but narrower than its name suggests -- see its own docstring (a review
  defect: it does not distinguish `1/k` from `(1/m + 1/u)` the way it once claimed to).

A few other tests here are near-vacuous -- forced by construction or a drift tripwire rather than
an independent check. They are kept (deleting a check silently is worse than a weak one) but each
says so in its own docstring: `test_every_phase_slot_is_demanded_by_two_thirds_of_blocks` and
`test_projection_reports_rho_and_cross_lineage_as_unidentifiable`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gfmcd.splicing.pbc_v2_capacity import (
    LEGACY_RESIDUAL_VARIANCE_UPPER_BOUND_DEV120,
    PAIR_TYPES,
    SESOI,
    allocate_blocks,
    blocks_for_sesoi,
    delta_balance_report,
    delta_class_phase_counts,
    feasibility,
    necessary_counts,
    power_projection,
    rotated_demands,
    var_direction,
    var_psi,
)

# ---------------------------------------------------------------------------------------------
# Layer 2 -- rotated 2x2 demand model
# ---------------------------------------------------------------------------------------------


def test_rotated_demands_at_the_capacity_ceiling_rung():
    demands = rotated_demands(909)
    assert demands["scaffold"] == 909
    for phase in (0, 1, 2):
        assert demands[f"donor_phase_{phase}"] == 606
        assert demands[f"acceptor_phase_{phase}"] == 606
    assert demands["donor_all_phases"] == 1818
    assert demands["acceptor_all_phases"] == 1818
    assert demands["all_roles"] == 4545


def test_allocate_blocks_is_exact_for_counts_not_divisible_by_three():
    for n_blocks in range(0, 40):
        allocation = allocate_blocks(n_blocks)
        assert sum(allocation.values()) == n_blocks, n_blocks
        assert max(allocation.values()) - min(allocation.values()) <= 1, n_blocks


def test_every_phase_slot_is_demanded_by_two_thirds_of_blocks():
    """FORCED BY CONSTRUCTION, not an independent check on the real pool.

    With three equal-weight pair types, each phase appears in exactly two of them, so
    `sum(donor_phase_p for p in 0,1,2) == 2*n_blocks` follows algebraically from `PAIR_TYPES` and
    `allocate_blocks` summing to `n_blocks` -- it cannot fail for any allocation respecting those
    two facts. Kept as a documented restatement of the design's own arithmetic, not as evidence
    the design (or its implementation) is correct.
    """
    # Each phase appears in exactly two of the three pair types.
    for n_blocks in (600, 729, 819, 909):
        demands = rotated_demands(n_blocks)
        total_phase_demand = sum(demands[f"donor_phase_{p}"] for p in (0, 1, 2))
        assert total_phase_demand == 2 * n_blocks


def test_delta_classes_are_phase_balanced_in_the_rotated_design():
    report = delta_balance_report(PAIR_TYPES)
    assert report["balanced"] is True
    counts = delta_class_phase_counts(PAIR_TYPES)
    for delta in (0, 1, 2):
        assert set(counts[delta]["donor"].values()) == ({2} if delta == 0 else {1})
        assert set(counts[delta]["acceptor"].values()) == ({2} if delta == 0 else {1})


def test_delta_balance_fails_for_a_plain_zero_one_lattice():
    """The design's whole justification: a plain {0,1} lattice confounds delta with cell identity."""

    report = delta_balance_report(((0, 1),))
    assert report["balanced"] is False
    # delta=1 is reachable only via the single cell (donor 0, acceptor 1)
    assert report["per_delta"][1]["donor_counts"] == {0: 1, 1: 0, 2: 0}
    assert report["per_delta"][2]["donor_counts"] == {0: 0, 1: 1, 2: 0}


def test_allocate_blocks_rejects_bad_inputs():
    with pytest.raises(ValueError):
        allocate_blocks(-1)
    with pytest.raises(ValueError):
        allocate_blocks(10, pair_weights=(1, 1))
    with pytest.raises(ValueError):
        allocate_blocks(10, pair_weights=(0, 0, 0))


# ---------------------------------------------------------------------------------------------
# Layer 3 -- exact feasibility by max-flow
# ---------------------------------------------------------------------------------------------


def test_planted_feasible_assignment_is_accepted():
    slot_sets = {"a": {"g1", "g2", "g3"}, "b": {"g4", "g5"}}
    out = feasibility(slot_sets, {"a": 2, "b": 2})
    assert out["feasible"] is True
    assert out["max_flow"] == out["total_demand"] == 4
    assert out["min_cut_slots"] == []


def test_hall_obstruction_is_invisible_to_counting_but_caught_by_flow():
    """Two slots, ten genes, but the SAME ten genes serve both, and each slot demands six.

    Per-slot counting passes (10 >= 6 for each). The pair is jointly infeasible (12 > 10). This is
    the input that justifies modelling the assignment exactly instead of counting.

    A third, NON-violating slot with its own five private genes is included specifically so
    `min_cut_gene_count` is tested for the property it actually publishes: with only the two
    violating slots present, `len(sink_side genes)` and "every gene in the graph" are the same
    number (10), so a buggy implementation returning the latter would pass by coincidence. With a
    third slot's disjoint, non-tight genes also in the graph (15 total), that coincidence is gone:
    the correct answer is still 10 (only the violating pair's own union), never 15.
    """

    shared = {f"g{i}" for i in range(10)}
    private = {f"p{i}" for i in range(5)}
    slot_sets = {
        "donor_phase_0": set(shared),
        "donor_phase_1": set(shared),
        "acceptor_phase_0": set(private),
    }
    demands = {"donor_phase_0": 6, "donor_phase_1": 6, "acceptor_phase_0": 2}

    counting = necessary_counts(slot_sets, demands)
    assert counting["entries"]["donor_phase_0"]["passed"] is True
    assert counting["entries"]["donor_phase_1"]["passed"] is True
    assert counting["entries"]["acceptor_phase_0"]["passed"] is True

    exact = feasibility(slot_sets, demands)
    assert exact["feasible"] is False
    assert set(exact["min_cut_slots"]) == {"donor_phase_0", "donor_phase_1"}
    assert exact["min_cut_gene_count"] == 10, "must be the violating pair's own union, not all 15 genes"
    assert exact["min_cut_demand"] == 12
    assert exact["shortfall"] == 2


def test_disjoint_pools_make_counting_and_flow_agree():
    slot_sets = {"a": {"g1", "g2"}, "b": {"g3", "g4"}}
    demands = {"a": 2, "b": 2}
    assert necessary_counts(slot_sets, demands)["passed"] is True
    assert feasibility(slot_sets, demands)["feasible"] is True


def test_feasibility_refuses_union_pseudo_slots():
    with pytest.raises(ValueError, match="no gene pool"):
        feasibility({"donor_phase_0": {"g1"}}, {"donor_phase_0": 1, "donor_all_phases": 2})


def test_capacity_margin_definition_is_available_minus_demand_over_available():
    out = necessary_counts({"donor_phase_2": {f"g{i}" for i in range(809)}}, {"donor_phase_2": 606})
    row = out["entries"]["donor_phase_2"]
    assert row["available_unique_genes"] == 809
    assert row["demand_unique_genes"] == 606
    assert row["margin"] == pytest.approx((809 - 606) / 809)
    assert row["deficit_unique_genes"] == 0


# ---------------------------------------------------------------------------------------------
# Layer 4 -- power projection
# ---------------------------------------------------------------------------------------------


def test_three_by_three_calibration_is_exact_for_every_rho():
    """Exact arithmetic anchor -- but narrower than it looks (a corrected review defect).

    This pins that `var_psi(3, 6, rho)` reproduces the frozen
    `LEGACY_RESIDUAL_VARIANCE_UPPER_BOUND_DEV120` for every rho -- i.e. it pins the PRODUCT of
    `_cell_variance(rho)` and the matched/mismatched cell term evaluated at m=3, u=6. It does NOT
    distinguish `(1/m + 1/u)` from `1/k` the way an earlier version of this docstring claimed:
    writing the cell term as `1/k` (k=9) together with a correspondingly rescaled
    `_cell_variance = 9*(1-rho)*V9` (instead of `2*(1-rho)*V9`) reproduces this exact same identity
    at m=3, u=6 -- both give a cell term of `(1-rho)*V9` there -- so this test alone would not catch
    that substitution. The 2x2 value that actually depends on `(1/m + 1/u)` vs `1/k` is pinned by
    `test_rotated_lattice_loses_precision_relative_to_the_three_by_three` below -- which restates
    the model's own arithmetic (it computes its expected value the same way the implementation
    does) rather than anchoring against an externally, independently sourced number.
    """

    for rho in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        assert var_psi(3, 6, rho) == pytest.approx(
            LEGACY_RESIDUAL_VARIANCE_UPPER_BOUND_DEV120, rel=1e-12
        )


def test_rotated_lattice_loses_precision_relative_to_the_three_by_three():
    # At rho=0 the 2x2 Psi variance is exactly twice the frozen 9-cell variance.
    assert var_psi(2, 2, 0.0) == pytest.approx(2.0 * LEGACY_RESIDUAL_VARIANCE_UPPER_BOUND_DEV120)
    # At rho=1 all variance is gene-level, so cell count cannot matter.
    assert var_psi(2, 2, 1.0) == pytest.approx(var_psi(3, 6, 1.0))
    # Directional endpoints degrade further: 1 cell per delta instead of 3.
    assert var_direction(1, 2, 0.0) > var_psi(2, 2, 0.0)


def test_psi_is_powered_for_every_rho_only_above_the_worst_case_floor():
    floor = blocks_for_sesoi(var_psi(2, 2, 0.0))
    assert power_projection(floor)["psi"]["verdict"] == "powered_for_every_rho"
    assert power_projection(floor - 200)["psi"]["verdict"] == "conditional_on_rho"


def test_directional_endpoints_are_not_powered_within_the_capacity_ceiling():
    """M+/M- need more blocks than the 25%-margin capacity ceiling (909) can supply."""

    assert blocks_for_sesoi(var_direction(1, 2, 0.0)) > 909
    assert power_projection(909)["direction"]["verdict"] == "conditional_on_rho"


def test_direction_block_is_a_floor_only_for_the_rotated_two_by_two():
    """A2: the rotated 2x2's M_delta blocks-needed figure is a floor, not a calibrated estimate.

    `var_direction`'s uncancelled module term (see its docstring) is structurally invisible to any
    lattice with fewer than 3 cells per delta class -- exactly the rotated 2x2 (1 cell per delta).
    The full 3x3 (3 cells per delta) draws from every donor and acceptor phase at each delta, so
    the module term cancels there and the estimate is calibrated.
    """
    rotated = power_projection(909, lattice="rotated_two_by_two")["direction"]
    assert rotated["estimate_class"] == "lower_bound_only"
    assert "lower_bound_reason" in rotated
    assert "module" in rotated["lower_bound_reason"]
    assert "blocks_needed_floor" in rotated
    assert "blocks_needed_worst_case" not in rotated

    full = power_projection(909, lattice="three_by_three")["direction"]
    assert full["estimate_class"] == "calibrated"
    assert "lower_bound_reason" not in full
    assert "blocks_needed_floor" in full


def test_psi_block_keeps_blocks_needed_worst_case_unrenamed():
    """Only the `direction` block's key was renamed to `blocks_needed_floor` (A2); `psi` has no
    uncancelled-module problem (module terms cancel in the 3x3 and the 2x2 Psi alike), so its key
    stays `blocks_needed_worst_case`.
    """
    psi = power_projection(909)["psi"]
    assert "blocks_needed_worst_case" in psi
    assert "estimate_class" not in psi


def test_projection_reports_rho_and_cross_lineage_as_unidentifiable():
    """DRIFT TRIPWIRE, not a measurement of anything.

    `rho_identifiable` and `cross_lineage["projectable"]` are hard-coded `False` literals in
    `power_projection` with no computed condition behind them (there is no rho estimate or
    cross-lineage variance anywhere in this module to be identifiable or not). This test only
    catches an accidental flip of one of those literals during a refactor; it is not evidence that
    rho actually is unidentifiable or that cross-lineage actually is unprojectable -- those are
    modelling facts recorded in `rho_unidentifiable_reason` / `cross_lineage["reason"]`, not
    something this assertion establishes.
    """
    projection = power_projection(909)
    assert projection["rho_identifiable"] is False
    assert projection["cross_lineage"]["projectable"] is False
    assert projection["cross_lineage"]["replicate"] == "training_seed"
    assert projection["n3"]["status"] == "unprojectable_pending_dose_calibration"


def test_projection_emits_no_number_for_the_unprojectable_endpoints():
    """A status string is the deliverable here; a number would be invented."""

    projection = power_projection(909)
    assert not any(isinstance(v, (int, float)) for v in projection["n3"].values())
    cross = projection["cross_lineage"]
    assert not any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in cross.values())


def test_mde80_band_is_ordered_and_straddles_the_sesoi_where_expected():
    projection = power_projection(729)
    assert projection["psi"]["mde80_best"] < projection["psi"]["mde80_worst"]
    assert projection["psi"]["mde80_worst"] == pytest.approx(SESOI, abs=5e-5)


def test_rho_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        var_psi(2, 2, 1.5)
    with pytest.raises(ValueError):
        var_direction(1, 2, -0.1)
