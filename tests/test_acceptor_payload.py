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


def test_carveout_is_refused_for_an_arm_that_does_not_own_its_written_motif():
    """M4: the carve-out's scoping is enforced, not merely conventional.

    Only an arm whose planted `written` motif IS the treatment may waive it. A caller
    reaching past `common_admissible_rows` with `alt='GT'` would otherwise silently
    admit a genuinely planted canonical donor as if it were this experiment's treatment.
    """
    x = make_example(minus_one="G", plus_three="A")
    for alt in ("GT", "GA", "GG", "AC", "AT"):
        with pytest.raises(ValueError, match="allow_written_motif"):
            arm_admissible(x, alt, allow_written_motif=True)
    # The treatment arm itself is unaffected.
    assert arm_admissible(x, TREATMENT_ARM, allow_written_motif=True) is True


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
    DEFAULT_MIN_GENES,
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
    # 30 genes: the C1 cluster floor. `ref` is made strictly positive so the cohort
    # reference sum cannot trip the I3 denominator guard by chance.
    rng = np.random.default_rng(0)
    v = rng.normal(1.0, 0.2, 60)
    ref = np.abs(rng.normal(1.0, 0.2, 60)) + 0.5
    genes = [f"g{i // 2}" for i in range(60)]
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
    # 80 rows / 40 genes, up from 40 rows / 20 genes, to clear the C1 cluster floor
    # while keeping the rows clustered two-to-a-gene.
    rng = np.random.default_rng(2)
    ref = np.abs(rng.normal(1.0, 0.1, 80)) + 0.5
    a, b = ref * 0.4, ref * 0.8
    genes = [f"g{i // 2}" for i in range(80)]
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
    # min_genes=1 is deliberate: this test is ABOUT the single-cluster degenerate case
    # that the C1 floor otherwise refuses. It asserts the collapse rather than reading a
    # verdict off it, which is exactly the use the floor's docstring carves out.
    est, lo, hi = gboot_rho_difference(
        a, b, ref, ["g0"] * 30, n_draws=50, seed=0, min_genes=1
    )
    assert lo == pytest.approx(est)
    assert hi == pytest.approx(est)


def _gene_clustered_fixture(n_genes=40, per_gene=3, seed=17):
    """Both arms share one per-gene ratio LEVEL; only their offset and row noise differ.

    This is the structure real cohorts have -- genes differ in overall effect size, and
    the two arms move together within a gene -- and it is what makes preserving the
    resampling correlation matter. The independent per-row noise is deliberate: without
    it the arm difference would be exactly constant, the paired half-width would collapse
    to 0, and the comparison below would pass vacuously against any positive number.
    """
    rng = np.random.default_rng(seed)
    n = n_genes * per_gene
    ref = np.abs(rng.normal(1.0, 0.05, n)) + 0.5
    level = np.repeat(rng.uniform(-0.35, 0.35, n_genes), per_gene)
    treat = ref * (0.55 + level) + rng.normal(0.0, 0.05, n)
    twin = ref * (0.85 + level) + rng.normal(0.0, 0.05, n)
    genes = [f"g{i // per_gene}" for i in range(n)]
    return treat, twin, ref, genes


def _independent_draw_half_width(t, r, d, genes, *, n_draws, seed):
    """The WRONG estimator: each arm resampled on its OWN independent gene draw."""
    by: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        by.setdefault(str(g), []).append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        rows_t = [row for j in rng.choice(len(keys), len(keys), replace=True) for row in by[keys[j]]]
        rows_r = [row for j in rng.choice(len(keys), len(keys), replace=True) for row in by[keys[j]]]
        draws[i] = (t[rows_t].sum() / d[rows_t].sum()) - (r[rows_r].sum() / d[rows_r].sum())
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return (hi - lo) / 2.0


def test_both_arms_are_resampled_on_the_same_gene_draw():
    """The paired draw is the point of the estimator, so pin it against the unpaired one.

    With gene-level ratio heterogeneity shared by both arms, the per-gene DIFFERENCE is
    near-constant even though each arm's own ratio swings widely. Resampling both arms on
    one gene draw cancels that shared swing; drawing the arms independently does not, and
    the interval inflates by the heterogeneity the design was built to remove.
    """
    treat, twin, ref, genes = _gene_clustered_fixture()
    _, lo, hi = gboot_rho_difference(treat, twin, ref, genes, n_draws=600, seed=0)
    paired = (hi - lo) / 2.0
    independent = _independent_draw_half_width(
        treat, twin, ref, genes, n_draws=600, seed=0
    )
    assert paired > 0.0, "vacuous fixture: the paired interval must have real width"
    assert paired < 0.5 * independent, (paired, independent)


def test_bootstrap_refuses_a_cohort_below_the_gene_floor():
    """C1: a degenerate cohort must raise, not return a zero-width 'equivalence'.

    An empty or single-gene cohort makes every draw identical, so lo == hi == est, which
    classify_difference reads as EQUIVALENT_WITHIN_SESOI and precision_gate as PROCEED --
    a positive scientific finding manufactured from one cluster.
    """
    ref = np.abs(np.random.default_rng(4).normal(1.0, 0.05, 60)) + 0.5
    a, b = ref * 0.4, ref * 0.8
    for genes in ([], ["g0"] * 60, [f"g{i // 3}" for i in range(60)]):
        arrays = (a, b, ref) if genes else (a[:0], b[:0], ref[:0])
        with pytest.raises(ValueError, match="distinct genes"):
            gboot_rho_difference(*arrays, genes, n_draws=20, seed=0)


def test_bootstrap_accepts_a_cohort_at_the_gene_floor():
    """The floor is inclusive: exactly DEFAULT_MIN_GENES distinct genes must run."""
    ref = np.abs(np.random.default_rng(5).normal(1.0, 0.05, 60)) + 0.5
    a, b = ref * 0.4, ref * 0.8
    genes = [f"g{i // 2}" for i in range(60)]
    assert len(set(genes)) == DEFAULT_MIN_GENES == 30
    est, lo, hi = gboot_rho_difference(a, b, ref, genes, n_draws=50, seed=0)
    assert lo <= est <= hi


def test_bootstrap_refuses_a_non_positive_reference_sum():
    """I3: rho's 1e-9 floor sign-flips a negative denominator instead of guarding it.

    rho([1, 1], [-2, -3]) is 2.0e9, not an error, so an unguarded cohort could inject an
    astronomical draw and drag the percentile quantiles. The guard lives in the estimator;
    rho itself keeps the repo's circuit.ratio_of_means idiom.
    """
    assert rho(np.array([1.0, 1.0]), np.array([-2.0, -3.0])) == pytest.approx(2.0e9)
    genes = [f"g{i // 2}" for i in range(60)]
    ref = np.full(60, -1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        gboot_rho_difference(ref * 0.4, ref * 0.8, ref, genes, n_draws=20, seed=0)
    zero = np.zeros(60)
    with pytest.raises(ValueError, match="strictly positive"):
        gboot_rho_difference(zero, zero, zero, genes, n_draws=20, seed=0)


@pytest.mark.parametrize(
    "lo,hi,expected",
    [
        (-0.50, -0.30, "MATERIAL_NEGATIVE"),
        (-0.04, +0.03, "EQUIVALENT_WITHIN_SESOI"),
        (-0.20, +0.10, "UNRESOLVED"),
        (-0.05, +0.01, "UNRESOLVED"),
        (+0.06, +0.20, "MATERIAL_POSITIVE"),
        # I1: the MATERIAL side of both boundaries. A boundary that is merely TOUCHED
        # must never read as material -- `lo > sesoi` and `hi < -sesoi` are strict, and
        # relaxing either to >= / <= is only visible here.
        (+0.05, +0.20, "UNRESOLVED"),
        (-0.30, -0.05, "UNRESOLVED"),
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
    """A half-width exactly at the SESOI cannot distinguish equivalence from material.

    Unchanged by the I6 correction: this interval is UNRESOLVED (it touches both
    boundaries) AND its half-width is at the SESOI, so both stop conditions hold.
    """
    assert precision_gate(-0.05, +0.05, 0.05) == "NOT_POWERABLE_ON_THIS_DESIGN"


@pytest.mark.parametrize(
    "lo,hi",
    [
        (+0.06, +0.20),  # MATERIAL_POSITIVE, half-width 0.07 > sesoi
        (-0.90, -0.10),  # MATERIAL_NEGATIVE, half-width 0.40 >> sesoi
    ],
)
def test_gate_does_not_veto_a_material_verdict_for_being_wide(lo, hi):
    """I6: a half-width inside the SESOI certifies EQUIVALENCE; it does not gate material.

    An interval that already clears the SESOI has answered the question at whatever width
    it achieved. Stopping it would discard a real finding for being imprecise about the
    effect's SIZE, which is not what the gate is for.
    """
    assert classify_difference(lo, hi, 0.05).startswith("MATERIAL_")
    assert half_width(lo, hi) > 0.05
    assert precision_gate(lo, hi, 0.05) == "PROCEED"


def test_gate_proceeds_on_an_unresolved_interval_narrower_than_the_sesoi():
    """A near miss on a workable design is not a dead design.

    [-0.045, +0.051] straddles the +SESOI but is narrower than it, so a shifted point
    estimate at this precision could still resolve. Only UNRESOLVED *and* half-width at
    or above the SESOI is unpowerable.
    """
    assert classify_difference(-0.045, +0.051, 0.05) == "UNRESOLVED"
    assert half_width(-0.045, +0.051) < 0.05
    assert precision_gate(-0.045, +0.051, 0.05) == "PROCEED"


from gfmcd.splicing.acceptor_payload import permuted_label_fpr, planted_positive_check


def _ref_and_genes(n=120, per_gene=4, seed=3):
    rng = np.random.default_rng(seed)
    ref = np.abs(rng.normal(1.0, 0.05, n)) + 0.5
    genes = [f"g{i // per_gene}" for i in range(n)]
    return ref, genes


def test_planted_positive_is_recovered_as_material_negative():
    ref, genes = _ref_and_genes()
    out = planted_positive_check(
        ref, genes, planted_rho=0.35, twin_rho=0.85, sesoi=0.05, n_draws=300, seed=0
    )
    assert out["classification"] == "MATERIAL_NEGATIVE"
    assert out["PASS"] is True


def test_planted_positive_check_can_fail_when_nothing_is_planted():
    """Sensitivity must be falsifiable: an unplanted effect must not report PASS.

    With the C2 noise the difference is centred on zero with real width, so it resolves
    EQUIVALENT_WITHIN_SESOI or UNRESOLVED -- never MATERIAL_NEGATIVE.
    """
    ref, genes = _ref_and_genes()
    out = planted_positive_check(
        ref, genes, planted_rho=0.85, twin_rho=0.85, sesoi=0.05, n_draws=300, seed=0
    )
    assert out["PASS"] is False
    assert out["classification"] in ("EQUIVALENT_WITHIN_SESOI", "UNRESOLVED")


def test_planted_positive_check_interval_has_real_width():
    """C2: the noiseless construction was zero-variance and could not fail properly.

    `ref * planted_rho` vs `ref * twin_rho` makes every bootstrap draw return the
    identical difference (measured CI width ~4.4e-16), so the check only exercised the
    classifier's sign convention -- it reported PASS even for an estimator resampling
    ROWS instead of genes, or computing mean-of-ratios instead of ratio-of-sums. A
    non-degenerate interval is what makes this a power check on THIS cohort.
    """
    ref, genes = _ref_and_genes()
    out = planted_positive_check(
        ref, genes, planted_rho=0.35, twin_rho=0.85, sesoi=0.05, n_draws=300, seed=0
    )
    lo, hi = out["ci"]
    assert hi - lo > 1e-6, (lo, hi)
    assert out["noise_frac"] == 0.25


def test_planted_positive_check_records_a_custom_noise_frac():
    ref, genes = _ref_and_genes()
    wide = planted_positive_check(
        ref, genes, planted_rho=0.35, twin_rho=0.85, sesoi=0.05,
        n_draws=200, seed=0, noise_frac=1.0,
    )
    narrow = planted_positive_check(
        ref, genes, planted_rho=0.35, twin_rho=0.85, sesoi=0.05,
        n_draws=200, seed=0, noise_frac=0.05,
    )
    assert wide["noise_frac"] == 1.0
    assert narrow["noise_frac"] == 0.05
    assert (wide["ci"][1] - wide["ci"][0]) > (narrow["ci"][1] - narrow["ci"][0])


def test_permuted_label_fpr_is_near_alpha_under_the_null():
    ref, genes = _ref_and_genes()
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.01, len(ref))
    v_a = ref * 0.85 + noise
    v_b = ref * 0.85 - noise
    out = permuted_label_fpr(
        v_a, v_b, ref, genes, sesoi=0.05, n_permutations=60, n_draws=150, seed=0
    )
    assert out["fpr"] <= 0.15
    assert out["PASS"] is True


def test_permuted_label_fpr_actually_permutes_the_labels():
    """C3: discriminate a real per-gene swap from a no-op.

    The two existing FPR tests both feed null-true data, where the rate is 0 whether or
    not the labels move -- deleting the swap body is invisible to them. Here the arms
    differ STRONGLY (0.35 vs 0.85 of ref). A no-op swap would hand every permutation the
    same large negative contrast, so every one classifies MATERIAL_NEGATIVE and the rate
    is exactly 1.0. A real swap flips roughly half the genes, the aggregate contrast
    collapses toward zero, and the rate falls well below 1.0.

    This is NOT a PASS assertion: on non-null data a high rate is expected and correct.
    The claim is only that the swap moves the labels at all.
    """
    ref, genes = _ref_and_genes()
    out = permuted_label_fpr(
        ref * 0.35, ref * 0.85, ref, genes,
        sesoi=0.05, n_permutations=60, n_draws=120, seed=0,
    )
    assert out["fpr"] < 0.9, out["fpr"]


def test_permuted_label_fpr_rejects_a_non_positive_permutation_count():
    """A bare ZeroDivisionError is not a validation error; match n_draws' contract."""
    ref, genes = _ref_and_genes()
    for n in (0, -1):
        with pytest.raises(ValueError, match="n_permutations must be positive"):
            permuted_label_fpr(
                ref * 0.85, ref * 0.85, ref, genes,
                sesoi=0.05, n_permutations=n, n_draws=20, seed=0,
            )


def test_permuted_label_fpr_is_deterministic_under_a_fixed_seed():
    ref, genes = _ref_and_genes()
    v_a, v_b = ref * 0.4, ref * 0.85
    first = permuted_label_fpr(
        v_a, v_b, ref, genes, sesoi=0.05, n_permutations=20, n_draws=80, seed=5
    )
    second = permuted_label_fpr(
        v_a, v_b, ref, genes, sesoi=0.05, n_permutations=20, n_draws=80, seed=5
    )
    assert first == second
