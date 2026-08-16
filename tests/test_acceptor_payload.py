"""M1 payload-identity primitives. No model or genome required."""

from __future__ import annotations

import pytest

from gfmcd.splicing.acceptor_payload import (
    LADDER_ARMS,
    REFERENCE_ALT,
    TREATMENT_ARM,
    TWIN_ARMS,
    allows_written_motif,
    arm_admissible,
    arm_dispositions,
)
from gfmcd.splicing.coupling import CouplingExample
from gfmcd.splicing.reverse_coupling import donor_alt_admissible

DONOR_POS = 21


def make_example(minus_one: str = "G", plus_three: str = "A", ex_id: str = "e0",
                 gene_id: str = "g0") -> CouplingExample:
    """A window whose donor GT sits at DONOR_POS.

    seq[DONOR_POS-1] is the last exon base; seq[DONOR_POS:DONOR_POS+2] is the GT;
    seq[DONOR_POS+2] is the +3 base. Those three positions drive every straddling register.
    """
    seq = "A" * (DONOR_POS - 1) + minus_one + "GT" + plus_three + "C" * 20
    assert seq[DONOR_POS:DONOR_POS + 2] == "GT"
    return CouplingExample(
        example_id=ex_id, sequence=seq, acceptor_pos=5, donor_pos=DONOR_POS,
        chrom="chr1", strand="+", gene_id=gene_id, split="train",
        exon_length=16, intron_length_down=100, n_isoforms_donor=1,
        n_isoforms_acceptor=1, gene_type="protein_coding", next_acceptor_pos=60,
    )


def test_frozen_arm_constants():
    assert REFERENCE_ALT == "AC"
    assert TREATMENT_ARM == "GC"
    assert TWIN_ARMS == ("GA", "GG")
    assert LADDER_ARMS == ("AC", "AT", "GA", "GG", "GC")
    assert allows_written_motif("GC") is True
    assert allows_written_motif("GG") is False


def test_gc_flags_only_the_written_register_on_a_clean_flank():
    x = make_example(minus_one="G", plus_three="A")
    assert arm_dispositions(x, "GC") == {"written": "GC"}
    assert arm_admissible(x, "GC", allow_written_motif=True) is True
    assert arm_admissible(x, "GC", allow_written_motif=False) is False


def test_gc_upstream_ag_still_disqualifies():
    """A -1 exon base of A makes upstream 'AG' -- a planted acceptor, not the treatment."""
    x = make_example(minus_one="A", plus_three="A")
    assert arm_dispositions(x, "GC")["upstream"] == "AG"
    assert arm_admissible(x, "GC", allow_written_motif=True) is False


def test_gg_downstream_gt_disqualifies():
    """alt='GG' with a +3 base of T plants a canonical GT one position downstream."""
    x = make_example(minus_one="G", plus_three="T")
    assert arm_dispositions(x, "GG")["downstream"] == "GT"
    assert arm_admissible(x, "GG", allow_written_motif=False) is False


def test_reference_arm_is_clean_on_a_clean_flank():
    x = make_example(minus_one="G", plus_three="A")
    assert arm_dispositions(x, REFERENCE_ALT) == {}
    assert arm_admissible(x, REFERENCE_ALT) is True


@pytest.mark.parametrize("minus_one", ["A", "C", "G", "T"])
@pytest.mark.parametrize("plus_three", ["A", "C", "G", "T"])
@pytest.mark.parametrize("alt", ["AC", "AT", "GA", "GG", "GC"])
def test_without_carveout_matches_the_existing_hygiene_rule(minus_one, plus_three, alt):
    """Regression guard: allow_written_motif=False must be exactly donor_alt_admissible."""
    x = make_example(minus_one=minus_one, plus_three=plus_three)
    assert arm_admissible(x, alt, allow_written_motif=False) == donor_alt_admissible(x, alt)


from gfmcd.splicing.acceptor_payload import admissible_fractions, common_admissible_rows


def test_common_rows_are_admissible_under_every_arm():
    exs = [
        make_example(minus_one="G", plus_three="A", ex_id="clean", gene_id="g1"),
        make_example(minus_one="A", plus_three="A", ex_id="bad_gc", gene_id="g2"),
        make_example(minus_one="G", plus_three="T", ex_id="bad_gg", gene_id="g3"),
    ]
    assert common_admissible_rows(exs, LADDER_ARMS) == [0]


def test_common_rows_empty_arms_returns_all():
    exs = [make_example(ex_id="a"), make_example(minus_one="A", ex_id="b")]
    assert common_admissible_rows(exs, ()) == [0, 1]


def test_admissible_fractions_are_per_arm_and_reported():
    exs = [
        make_example(minus_one="G", plus_three="A", ex_id="clean"),
        make_example(minus_one="A", plus_three="A", ex_id="bad_gc"),
    ]
    fracs = admissible_fractions(exs, ("AC", "GC"))
    assert fracs["AC"] == 1.0
    assert fracs["GC"] == 0.5


def test_admissible_fractions_empty_cohort_is_zero_not_a_crash():
    assert admissible_fractions([], ("AC",)) == {"AC": 0.0}


import numpy as np

from gfmcd.splicing.acceptor_payload import (
    classify_difference,
    gboot_rho_difference,
    pooled_twin_effect,
    rho,
)


def test_rho_is_ratio_of_sums_not_mean_of_ratios():
    """Near-zero denominators make per-example ratios explode; ratio-of-means does not."""
    v = np.array([1.0, 1.0])
    ref = np.array([2.0, 1e-12])
    assert rho(v, ref) == pytest.approx(2.0 / (2.0 + 1e-12))


def test_pooled_twin_effect_is_the_rowwise_mean():
    per_arm = {"GA": np.array([1.0, 3.0]), "GG": np.array([3.0, 5.0]), "GC": np.array([0.0, 0.0])}
    assert pooled_twin_effect(per_arm, ("GA", "GG")) == pytest.approx(np.array([2.0, 4.0]))


def test_identical_arms_give_exactly_zero_difference():
    rng = np.random.default_rng(0)
    v = rng.normal(1.0, 0.2, 60)
    ref = rng.normal(1.0, 0.2, 60)
    genes = [f"g{i // 3}" for i in range(60)]
    est, lo, hi = gboot_rho_difference(v, v, ref, genes, n_draws=200, seed=0)
    assert est == pytest.approx(0.0, abs=1e-12)
    assert lo <= 0.0 <= hi


def test_planted_difference_is_recovered():
    rng = np.random.default_rng(1)
    ref = np.abs(rng.normal(1.0, 0.05, 120)) + 0.5
    twin = ref * 0.85
    treat = ref * 0.35
    genes = [f"g{i // 4}" for i in range(120)]
    est, lo, hi = gboot_rho_difference(treat, twin, ref, genes, n_draws=400, seed=0)
    assert est == pytest.approx(0.35 - 0.85, abs=0.02)
    assert hi < 0.0


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(2)
    ref = np.abs(rng.normal(1.0, 0.1, 40)) + 0.5
    a, b = ref * 0.4, ref * 0.8
    genes = [f"g{i // 2}" for i in range(40)]
    first = gboot_rho_difference(a, b, ref, genes, n_draws=100, seed=7)
    second = gboot_rho_difference(a, b, ref, genes, n_draws=100, seed=7)
    assert first == second


def test_resampling_unit_is_the_gene_not_the_row():
    """One gene carrying every row must collapse the interval to a point.

    The 30 rows are given non-constant, per-row values on purpose. With a single gene,
    correct GENE-level resampling always draws that one gene and therefore always
    includes all 30 rows, so every draw is identical and the interval collapses exactly
    regardless of the row values. A WRONG row-level bootstrap, in contrast, draws
    different subsets of these now-varying rows on each iteration, so its sums vary
    across draws and the interval would NOT collapse (lo < est < hi). If the fixture
    rows were constant instead, row-level resampling would also collapse by coincidence
    and this test would fail to discriminate between the two implementations.
    """
    ref = np.linspace(0.5, 2.0, 30)
    a = np.linspace(0.1, 0.9, 30)
    b = np.linspace(0.3, 1.1, 30)
    est, lo, hi = gboot_rho_difference(a, b, ref, ["g0"] * 30, n_draws=50, seed=0)
    assert lo == pytest.approx(est)
    assert hi == pytest.approx(est)


@pytest.mark.parametrize(
    "lo,hi,expected",
    [
        (-0.50, -0.30, "MATERIAL_NEGATIVE"),
        (-0.04, +0.03, "EQUIVALENT_WITHIN_SESOI"),
        (-0.20, +0.10, "UNRESOLVED"),
        (-0.05, +0.01, "UNRESOLVED"),
        (+0.06, +0.20, "MATERIAL_POSITIVE"),
    ],
)
def test_classification_treats_a_touched_boundary_as_unresolved(lo, hi, expected):
    assert classify_difference(lo, hi, 0.05) == expected


from gfmcd.splicing.acceptor_payload import half_width, precision_gate


def test_half_width_is_half_the_interval():
    assert half_width(-0.10, +0.10) == pytest.approx(0.10)


def test_gate_proceeds_when_the_interval_can_clear_the_sesoi():
    assert precision_gate(-0.04, +0.04, 0.05) == "PROCEED"


def test_gate_stops_when_the_interval_cannot_resolve_the_sesoi():
    assert precision_gate(-0.30, +0.30, 0.05) == "NOT_POWERABLE_ON_THIS_DESIGN"


def test_gate_boundary_case_is_a_stop_not_a_proceed():
    """A half-width exactly at the SESOI cannot distinguish equivalence from material."""
    assert precision_gate(-0.05, +0.05, 0.05) == "NOT_POWERABLE_ON_THIS_DESIGN"
