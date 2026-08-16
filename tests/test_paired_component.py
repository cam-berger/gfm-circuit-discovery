from __future__ import annotations

from itertools import product

import numpy as np
import pytest

import gfmcd.splicing.paired_component as pc
from gfmcd.splicing.paired_component import (
    BoundarySignFlipInversion,
    BoundarySignFlipTest,
    CrossingBoundarySignFlipTest,
    FamilyDecision,
    TOSTEquivalenceTest,
    add_one_max_abs_pvalues,
    boundary_sign_flip_null_draws,
    boundary_sign_flip_tests,
    cluster_bootstrap_estimates,
    crossing_boundary_sign_flip_tests,
    decision_path_preflight,
    gene_sign_flip,
    gene_sign_flip_null_draws,
    gene_signs,
    holm_adjust_named,
    intersection_union_pvalue,
    invert_boundary_sign_flip_tests,
    invert_one_sided_boundary_sign_flip,
    one_sided_binomial_bounds,
    one_sided_boundary_sign_flip,
    simultaneous_family_decision,
    strict_interval_decision,
    tost_equivalence_test,
)


def _groups(n_genes: int = 80, per_gene: int = 3) -> list[str]:
    return [f"g{gene}" for gene in range(n_genes) for _ in range(per_gene)]


def test_gene_signs_are_stable_and_constant_within_gene():
    groups = ["g1", "g1", "g2", "g3", "g3"]
    a = gene_signs(groups, seed=7, namespace="test")
    b = gene_signs(groups, seed=7, namespace="test")
    assert np.array_equal(a, b)
    assert a[0] == a[1] and a[3] == a[4]
    assert set(a) <= {-1.0, 1.0}


def test_gene_sign_flip_never_splits_a_gene():
    groups = ["g1", "g1", "g2", "g2"]
    values = np.ones(4)
    flipped = gene_sign_flip(values, groups, seed=2, namespace="labels")
    assert flipped[0] == flipped[1]
    assert flipped[2] == flipped[3]


def test_add_one_max_abs_pvalues_use_the_full_family_maximum():
    null = np.asarray([[0.1, 0.2], [0.4, 0.05], [0.3, 0.1]])
    got = add_one_max_abs_pvalues([0.35, 0.15], null)
    assert np.allclose(got, [0.5, 1.0])  # (1+1)/(3+1), (1+3)/(3+1)


def test_boundary_tests_do_not_raw_max_mask_a_precise_heteroskedastic_endpoint():
    groups = _groups(n_genes=50, per_gene=1)
    rng = np.random.default_rng(3)
    family = np.vstack(
        (
            np.full(len(groups), 0.10),
            rng.normal(0.0, 100.0, len(groups)),
        )
    )

    tests = boundary_sign_flip_tests(
        family,
        groups,
        ["precise_signal", "high_variance_nuisance"],
        boundaries=0.0,
        alternatives="greater",
        n_draws=255,
        seed=7,
        namespace="heteroskedastic-boundary-family",
    )
    adjusted = holm_adjust_named(
        {name: result.p_value for name, result in tests.items()}
    )

    assert tests["precise_signal"].p_value == pytest.approx(1 / 256)
    assert adjusted["precise_signal"] == pytest.approx(2 / 256)
    assert adjusted["precise_signal"] < 0.05


def test_boundary_test_distinguishes_zero_from_a_positive_sesoi_boundary():
    groups = _groups(n_genes=40, per_gene=1)
    values = np.asarray([0.01] * 20 + [0.11] * 20)

    against_zero = one_sided_boundary_sign_flip(
        values,
        groups,
        boundary=0.0,
        alternative="greater",
        n_draws=1023,
        seed=9,
        namespace="theta-zero",
    )
    against_sesoi = one_sided_boundary_sign_flip(
        values,
        groups,
        boundary=0.05,
        alternative="greater",
        n_draws=1023,
        seed=9,
        namespace="theta-sesoi",
    )
    negative_against_zero = one_sided_boundary_sign_flip(
        -values,
        groups,
        boundary=0.0,
        alternative="less",
        n_draws=1023,
        seed=9,
        namespace="theta-zero-less",
    )

    assert against_zero.oriented_statistic == pytest.approx(0.06)
    assert against_sesoi.oriented_statistic == pytest.approx(0.01)
    assert negative_against_zero.oriented_statistic == pytest.approx(0.06)
    assert against_zero.p_value < 0.05
    assert against_sesoi.p_value > 0.05
    assert negative_against_zero.p_value < 0.05


def test_tost_uses_iut_max_and_strict_interval_boundaries():
    groups = _groups(n_genes=40, per_gene=1)
    equivalent = tost_equivalence_test(
        np.zeros(len(groups)),
        groups,
        sesoi=0.05,
        n_draws=255,
        seed=12,
        namespace="tost-equivalent",
    )
    touching = tost_equivalence_test(
        np.full(len(groups), 0.05),
        groups,
        sesoi=0.05,
        n_draws=255,
        seed=12,
        namespace="tost-touching",
    )

    assert isinstance(equivalent, TOSTEquivalenceTest)
    assert equivalent.p_value == max(
        equivalent.lower_edge.p_value,
        equivalent.upper_edge.p_value,
    )
    assert equivalent.p_value < 0.05
    assert touching.upper_edge.p_value == 1.0
    assert touching.p_value == 1.0
    assert strict_interval_decision(-0.049, 0.049, 0.05) == "EQUIVALENT_WITHIN_SESOI"
    assert strict_interval_decision(-0.05, 0.049, 0.05) == "UNRESOLVED"
    assert strict_interval_decision(0.05, 0.10, 0.05) == "UNRESOLVED"
    assert strict_interval_decision(0.050001, 0.10, 0.05) == "MATERIAL_POSITIVE"


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_exact_closed_null_boundary_is_a_tie_for_128_rows(alternative):
    groups = _groups(n_genes=128, per_gene=1)
    result = one_sided_boundary_sign_flip(
        np.full(128, 0.05),
        groups,
        boundary=0.05,
        alternative=alternative,
        n_draws=255,
        seed=14,
        namespace=f"exact-boundary-{alternative}",
    )

    assert np.mean(np.full(128, 0.05)) > 0.05  # regression precondition
    assert result.oriented_statistic == 0.0
    assert result.comparison_tolerance > 0.0
    assert result.exceedances == 255
    assert result.p_value == 1.0
    assert "comparison_tolerance" in result.pvalue_rule


def test_intersection_union_pvalue_is_the_constituent_maximum():
    assert intersection_union_pvalue([0.004, 0.031, 0.012]) == pytest.approx(0.031)
    with pytest.raises(ValueError, match="non-empty"):
        intersection_union_pvalue([])


def test_holm_adjustment_expands_over_every_named_path():
    original = holm_adjust_named({"core": 0.01, "stronger": 0.04})
    expanded = holm_adjust_named(
        {"core": 0.01, "stronger": 0.04, "exploratory": 0.90}
    )

    assert original == pytest.approx({"core": 0.02, "stronger": 0.04})
    assert expanded == pytest.approx(
        {"core": 0.03, "stronger": 0.08, "exploratory": 0.90}
    )
    assert expanded["core"] > original["core"]


def test_shared_gene_signs_preserve_partner_minus_pseudo_algebra():
    groups = _groups(n_genes=35, per_gene=2)
    rng = np.random.default_rng(17)
    partner = rng.normal(0.20, 0.30, len(groups))
    pseudo = rng.normal(0.01, 0.10, len(groups))
    specific = partner - pseudo
    draws = boundary_sign_flip_null_draws(
        np.vstack((partner, pseudo, specific)),
        groups,
        boundaries=[0.05, 0.0, 0.05],
        n_draws=127,
        seed=21,
        namespace="shared-algebra",
    )

    assert np.allclose(draws[:, 2], draws[:, 0] - draws[:, 1], atol=1e-14)


def test_boundary_sign_flip_records_add_one_and_replay_metadata():
    groups = _groups(n_genes=50, per_gene=1)
    result = one_sided_boundary_sign_flip(
        np.ones(len(groups)),
        groups,
        boundary=0.05,
        alternative="greater",
        n_draws=19,
        seed=23,
        namespace="metadata-replay",
    )
    replay = one_sided_boundary_sign_flip(
        np.ones(len(groups)),
        groups,
        boundary=0.05,
        alternative="greater",
        n_draws=19,
        seed=23,
        namespace="metadata-replay",
    )

    assert isinstance(result, BoundarySignFlipTest)
    assert result.exceedances == 0
    assert result.B == 19
    assert result.p_value == pytest.approx(1 / 20)
    assert result.p_value == pytest.approx(result.add_one_floor)
    assert result.seed == 23
    assert result.namespace == "metadata-replay"
    assert len(result.null_draws_sha256) == 64
    assert len(result.shared_null_draws_sha256) == 64
    assert len(result.sign_matrix_sha256) == 64
    assert len(result.ordered_gene_labels_sha256) == 64
    assert result.null_draws_sha256 == replay.null_draws_sha256
    assert result.shared_null_draws_sha256 == replay.shared_null_draws_sha256
    assert result.sign_matrix_sha256 == replay.sign_matrix_sha256
    assert result.ordered_gene_labels_sha256 == replay.ordered_gene_labels_sha256
    assert result.to_dict()["add_one_floor"] == pytest.approx(1 / 20)


@pytest.mark.parametrize(
    ("alternative", "bound_kind"),
    [("greater", "lower"), ("less", "upper")],
)
def test_inverted_boundary_is_strictly_dual_at_arbitrary_float_boundaries(
    alternative,
    bound_kind,
):
    groups = _groups(n_genes=45, per_gene=2)
    rng = np.random.default_rng(101)
    values = rng.normal(0.08, 0.24, len(groups))
    alpha = 0.05
    inversion = invert_one_sided_boundary_sign_flip(
        values,
        groups,
        alternative=alternative,
        alpha=alpha,
        n_draws=511,
        seed=31,
        namespace=f"strict-duality-{alternative}",
    )

    assert isinstance(inversion, BoundarySignFlipInversion)
    assert inversion.bound_kind == bound_kind
    assert inversion.finite_transition
    assert inversion.bound is not None
    assert inversion.transition_test is not None
    assert inversion.rejecting_adjacent_test is not None
    assert inversion.nonrejecting_adjacent_test is not None

    candidates = [
        -0.50,
        0.0,
        0.05,
        0.50,
        inversion.bound,
        inversion.rejecting_adjacent_test.boundary,
        inversion.nonrejecting_adjacent_test.boundary,
    ]
    for boundary in candidates:
        test = crossing_boundary_sign_flip_tests(
            values,
            groups,
            ["endpoint"],
            boundaries=boundary,
            alternatives=alternative,
            n_draws=511,
            seed=31,
            namespace=f"strict-duality-{alternative}",
        )["endpoint"]
        assert isinstance(test, CrossingBoundarySignFlipTest)
        assert (test.p_value < alpha) == inversion.rejects(boundary)
        if alternative == "greater":
            assert inversion.rejects(boundary) == (inversion.bound > boundary)
        else:
            assert inversion.rejects(boundary) == (inversion.bound < boundary)

    assert inversion.transition_test.boundary == inversion.bound
    assert inversion.transition_test.p_value >= alpha
    assert inversion.rejecting_adjacent_test.p_value < alpha
    assert inversion.nonrejecting_adjacent_test.p_value >= alpha
    if alternative == "greater":
        assert np.nextafter(
            inversion.rejecting_adjacent_test.boundary, np.inf
        ) == inversion.bound
        assert np.nextafter(inversion.bound, np.inf) == (
            inversion.nonrejecting_adjacent_test.boundary
        )
    else:
        assert np.nextafter(
            inversion.rejecting_adjacent_test.boundary, -np.inf
        ) == inversion.bound
        assert np.nextafter(inversion.bound, -np.inf) == (
            inversion.nonrejecting_adjacent_test.boundary
        )


def test_inverted_boundary_constant_multirow_closed_null_is_conservative():
    groups = [
        f"g{gene}"
        for gene in range(40)
        for _ in range(1 + gene % 4)
    ]
    values = np.full(len(groups), 0.05)
    tests = crossing_boundary_sign_flip_tests(
        np.vstack((values, values)),
        groups,
        ["greater", "less"],
        boundaries=[0.05, 0.05],
        alternatives=["greater", "less"],
        n_draws=255,
        seed=37,
        namespace="constant-closed-null",
    )
    inversions = invert_boundary_sign_flip_tests(
        np.vstack((values, values)),
        groups,
        ["greater", "less"],
        alternatives=["greater", "less"],
        alphas=0.05,
        n_draws=255,
        seed=37,
        namespace="constant-closed-null",
    )

    # Every randomized statistic is a machine-level closed-null tie.  Treating only
    # exact raw crossing ratios as ties can instead return the add-one floor here.
    assert tests["greater"].p_value == 1.0
    assert tests["less"].p_value == 1.0
    assert tests["greater"].exceedances == 255
    assert tests["less"].exceedances == 255
    assert tests["greater"].comparison_tolerance > 0.0
    assert tests["less"].comparison_tolerance > 0.0
    assert inversions["greater"].bound < 0.05
    assert inversions["less"].bound > 0.05
    assert not inversions["greater"].rejects(0.05)
    assert not inversions["less"].rejects(0.05)


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_crossing_boundary_matches_direct_shared_sign_test_away_from_ties(alternative):
    groups = [
        f"g{gene}"
        for gene in range(40)
        for _ in range(1 + gene % 4)
    ]
    values = np.random.default_rng(109).normal(0.10, 0.30, len(groups))
    direct = one_sided_boundary_sign_flip(
        values,
        groups,
        boundary=0.037,
        alternative=alternative,
        n_draws=511,
        seed=39,
        namespace="crossing-direct-replay",
    )
    crossing = crossing_boundary_sign_flip_tests(
        values,
        groups,
        ["endpoint"],
        boundaries=0.037,
        alternatives=alternative,
        n_draws=511,
        seed=39,
        namespace="crossing-direct-replay",
    )["endpoint"]

    assert crossing.exceedances == direct.exceedances
    assert crossing.p_value == direct.p_value
    assert crossing.sign_matrix_sha256 == direct.sign_matrix_sha256


def test_inverted_boundary_replays_shared_signs_and_transition_digests():
    groups = _groups(n_genes=50, per_gene=1)
    rng = np.random.default_rng(113)
    family = np.vstack(
        (
            rng.normal(0.20, 0.30, len(groups)),
            rng.normal(0.00, 0.08, len(groups)),
            rng.normal(-0.12, 0.20, len(groups)),
        )
    )
    kwargs = dict(
        alternatives=["greater", "less", "greater"],
        alphas=[0.05, 0.025, 0.05],
        n_draws=255,
        seed=41,
        namespace="inversion-replay",
    )
    first = invert_boundary_sign_flip_tests(
        family,
        groups,
        ["a", "b", "c"],
        **kwargs,
    )
    replay = invert_boundary_sign_flip_tests(
        family,
        groups,
        ["a", "b", "c"],
        **kwargs,
    )

    assert {item.sign_matrix_sha256 for item in first.values()} == {
        next(iter(first.values())).sign_matrix_sha256
    }
    assert {item.ordered_gene_labels_sha256 for item in first.values()} == {
        next(iter(first.values())).ordered_gene_labels_sha256
    }
    assert {item.ordered_endpoint_names_sha256 for item in first.values()} == {
        next(iter(first.values())).ordered_endpoint_names_sha256
    }
    assert {item.B for item in first.values()} == {255}
    assert {item.seed for item in first.values()} == {41}
    assert {item.namespace for item in first.values()} == {"inversion-replay"}
    assert {name: item.to_dict() for name, item in first.items()} == {
        name: item.to_dict() for name, item in replay.items()
    }
    for item in first.values():
        assert item.transition_test is not None
        assert item.rejecting_adjacent_test is not None
        assert item.nonrejecting_adjacent_test is not None
        for test in (
            item.transition_test,
            item.rejecting_adjacent_test,
            item.nonrejecting_adjacent_test,
        ):
            assert len(test.crossing_draws_sha256) == 64
            assert len(test.adjusted_crossing_draws_sha256) == 64
            assert len(test.exceedance_mask_sha256) == 64
            assert len(test.shared_crossing_draws_sha256) == 64
            assert len(test.shared_negative_example_counts_sha256) == 64
            assert len(test.always_tie_mask_sha256) == 64
        assert (
            item.transition_test.exceedance_mask_sha256
            != item.rejecting_adjacent_test.exceedance_mask_sha256
        )


def test_inverted_boundary_reports_vacuous_monte_carlo_resolution():
    groups = _groups(n_genes=30, per_gene=1)
    result = invert_one_sided_boundary_sign_flip(
        np.ones(len(groups)),
        groups,
        alternative="greater",
        alpha=0.05,
        n_draws=19,
        seed=43,
        namespace="strict-add-one-floor",
    )

    assert result.max_rejecting_exceedances == -1
    assert result.bound is None
    assert not result.finite_transition
    assert result.transition_status == "VACUOUS_NO_REJECTING_BOUNDARY"
    assert result.transition_test is None
    assert not result.rejects(-1e300)


def test_unequal_clusters_are_resampled_whole_and_keep_row_weighting():
    groups = ["a"] + ["b"] * 3 + ["c"] * 8
    values = np.asarray([[4.0] + [1.0] * 3 + [2.0] * 8])

    bootstrap = cluster_bootstrap_estimates(
        values,
        groups,
        n_draws=512,
        seed=5,
        namespace="unequal-bootstrap",
        min_genes=3,
    )[:, 0]
    bootstrap_support = {
        (4 * count_a + 3 * count_b + 16 * count_c)
        / (count_a + 3 * count_b + 8 * count_c)
        for count_a in range(4)
        for count_b in range(4 - count_a)
        for count_c in [3 - count_a - count_b]
    }
    assert {round(value, 12) for value in bootstrap} == {
        round(value, 12) for value in bootstrap_support
    }

    null = gene_sign_flip_null_draws(
        values,
        groups,
        n_draws=512,
        seed=5,
        namespace="unequal-sign-flip",
        min_genes=3,
    )[:, 0]
    raw_sum_support = {
        (4 * sign_a + 3 * sign_b + 16 * sign_c) / 12
        for sign_a, sign_b, sign_c in product((-1, 1), repeat=3)
    }
    assert {round(value, 12) for value in null} == {
        round(value, 12) for value in raw_sum_support
    }


def test_production_default_requires_thirty_independent_genes():
    groups = _groups(n_genes=29, per_gene=1)
    with pytest.raises(ValueError, match="at least 30 independent gene clusters"):
        simultaneous_family_decision(
            np.zeros((1, len(groups))),
            groups,
            ["zero"],
            sesoi=0.05,
            n_draws=20,
        )

    with pytest.raises(ValueError, match="min_genes must be at least 2"):
        simultaneous_family_decision(
            np.zeros((1, 1)),
            ["only-gene"],
            ["zero"],
            sesoi=0.05,
            n_draws=20,
            min_genes=1,
        )

    decisions, calibration = simultaneous_family_decision(
        np.zeros((1, 2)),
        ["g1", "g2"],
        ["zero"],
        sesoi=0.05,
        n_draws=20,
        min_genes=2,
    )
    assert calibration["minimum_genes_required"] == 2
    assert decisions["zero"].classification == "EQUIVALENT_WITHIN_SESOI"


def test_simultaneous_classifier_uses_generic_interval_labels_and_honest_names():
    groups = _groups()
    gene_offsets = np.where(np.arange(80) % 2 == 0, -0.10, 0.10)
    unresolved = np.repeat(0.055 + gene_offsets, 3)
    values = np.vstack(
        [
            np.zeros(len(groups)),
            np.full(len(groups), 0.18),
            np.full(len(groups), -0.18),
            unresolved,
        ]
    )
    out, calibration = simultaneous_family_decision(
        values,
        groups,
        ["zero", "positive", "negative", "unresolved"],
        sesoi=0.05,
        n_draws=1200,
        seed=3,
    )
    assert calibration["randomisation_unit"] == "gene"
    assert calibration["method"].endswith("max-absolute-deviation interval")
    assert "max-t" not in str(calibration).lower()
    assert out["zero"].classification == "EQUIVALENT_WITHIN_SESOI"
    assert out["positive"].classification == "MATERIAL_POSITIVE"
    assert out["negative"].classification == "MATERIAL_NEGATIVE"
    assert out["unresolved"].classification == "UNRESOLVED"
    assert "max_abs_p" in out["positive"].to_dict()
    assert "max_t_p" not in out["positive"].to_dict()


def test_studentized_family_keeps_endpoint_specific_simultaneous_widths():
    groups = _groups(n_genes=80, per_gene=1)
    rng = np.random.default_rng(19)
    values = np.vstack((rng.normal(0.4, 0.8, len(groups)), np.zeros(len(groups))))
    ordinary, ordinary_calibration = simultaneous_family_decision(
        values,
        groups,
        ["variable", "stable_control"],
        sesoi=0.05,
        n_draws=1000,
        seed=5,
        studentized=False,
    )
    studentized, studentized_calibration = simultaneous_family_decision(
        values,
        groups,
        ["variable", "stable_control"],
        sesoi=0.05,
        n_draws=1000,
        seed=5,
        studentized=True,
    )
    ordinary_widths = [
        decision.simultaneous_ci[1] - decision.estimate
        for decision in ordinary.values()
    ]
    assert ordinary_widths[0] == pytest.approx(ordinary_widths[1])
    studentized_widths = studentized_calibration["simultaneous_half_widths"]
    assert studentized_widths["variable"] > 0
    assert studentized_widths["stable_control"] == 0
    assert studentized["stable_control"].classification == "EQUIVALENT_WITHIN_SESOI"
    assert studentized_calibration["studentized"] is True
    assert "studentized" in studentized_calibration["method"]
    assert ordinary_calibration["studentized"] is False


def test_preflight_routes_every_row_level_null_and_plant_through_classifier(monkeypatch):
    classifier_calls: list[tuple[str, np.ndarray, int]] = []
    row_null_namespaces: list[str] = []
    real_gene_sign_flip = pc.gene_sign_flip

    def spy_gene_sign_flip(values, groups, *, seed, namespace):
        row_null_namespaces.append(namespace)
        return real_gene_sign_flip(
            values,
            groups,
            seed=seed,
            namespace=namespace,
        )

    def fake_classifier(values, groups, names, **kwargs):
        means = np.asarray(values).mean(axis=1)
        classifier_calls.append(
            (kwargs["namespace"], means.copy(), kwargs["min_genes"])
        )
        decisions = {
            name: FamilyDecision(
                estimate=float(mean),
                simultaneous_ci=(float(mean), float(mean)),
                max_abs_p=1.0,
                classification=(
                    "MATERIAL_POSITIVE"
                    if index == 0 and mean > 0.10
                    else "EQUIVALENT_WITHIN_SESOI"
                ),
            )
            for index, (name, mean) in enumerate(zip(names, means))
        }
        return decisions, {
            "simultaneous_half_width": 0.0,
            "simultaneous_half_widths": {name: 0.0 for name in names},
            "resolution_ok": True,
        }

    monkeypatch.setattr(pc, "gene_sign_flip", spy_gene_sign_flip)
    monkeypatch.setattr(pc, "simultaneous_family_decision", fake_classifier)
    result = decision_path_preflight(
        np.zeros((2, 30)),
        _groups(n_genes=30, per_gene=1),
        sesoi=0.05,
        n_calibration=7,
        n_trials=60,
        seed=4,
        namespace="route-test",
    )

    assert result["production_classifier_calls"] == 121
    assert len(classifier_calls) == 121
    assert row_null_namespaces == [f"route-test:row-null:{i}" for i in range(60)]
    assert classifier_calls[0][0] == "route-test:reference"
    assert [call[0] for call in classifier_calls[1:3]] == [
        "route-test:classifier:null:0",
        "route-test:classifier:plant:0",
    ]
    assert len({call[0] for call in classifier_calls}) == len(classifier_calls)
    component_zero_means = np.asarray([call[1][0] for call in classifier_calls])
    assert np.count_nonzero(np.isclose(component_zero_means, 0.0)) == 61
    assert np.count_nonzero(np.isclose(component_zero_means, 0.15)) == 60
    assert {call[2] for call in classifier_calls} == {30}
    assert result["familywise_fpr_ok"]
    assert result["planted_power_ok"]


def test_preflight_gates_use_conservative_one_sided_binomial_bounds():
    fpr_lower, fpr_upper = one_sided_binomial_bounds(0, 20)
    power_lower, power_upper = one_sided_binomial_bounds(20, 20)
    assert fpr_lower == 0.0 and fpr_upper > 0.05
    assert power_upper == 1.0 and power_lower < 0.90

    result = decision_path_preflight(
        np.zeros((1, 30)),
        _groups(n_genes=30, per_gene=1),
        sesoi=0.05,
        n_calibration=20,
        n_trials=20,
        seed=8,
        namespace="finite-trials",
    )
    assert result["familywise_fpr"] == 0.0
    assert result["planted_power"] == 1.0
    assert not result["familywise_fpr_ok"]
    assert not result["planted_power_ok"]


def test_preflight_supports_a_conjunctive_structured_plant():
    groups = _groups(n_genes=30, per_gene=1)
    result = decision_path_preflight(
        np.zeros((3, len(groups))),
        groups,
        sesoi=0.05,
        n_calibration=80,
        n_trials=80,
        seed=12,
        namespace="structured-plant",
        plant_vector=[0.15, 0.0, 0.15],
        planted_requirements={
            0: "MATERIAL_POSITIVE",
            1: "EQUIVALENT_WITHIN_SESOI",
            2: "MATERIAL_POSITIVE",
        },
    )
    assert result["plant_vector"] == [0.15, 0.0, 0.15]
    assert result["planted_requirements"] == {
        "component_0": "MATERIAL_POSITIVE",
        "component_1": "EQUIVALENT_WITHIN_SESOI",
        "component_2": "MATERIAL_POSITIVE",
    }
    assert result["planted_power"] == 1.0


def test_preflight_can_score_false_positives_on_predeclared_conjunctions(monkeypatch):
    def fake_classifier(values, groups, names, **kwargs):
        is_plant = ":plant:" in kwargs["namespace"]
        labels = ["MATERIAL_POSITIVE", "MATERIAL_POSITIVE" if is_plant else "UNRESOLVED"]
        decisions = {
            name: FamilyDecision(0.2, (0.1, 0.3), 0.01, labels[index])
            for index, name in enumerate(names)
        }
        return decisions, {
            "simultaneous_half_width": 0.1,
            "simultaneous_half_widths": {name: 0.1 for name in names},
            "resolution_ok": False,
        }

    monkeypatch.setattr(pc, "simultaneous_family_decision", fake_classifier)
    result = decision_path_preflight(
        np.zeros((2, 30)),
        _groups(n_genes=30, per_gene=1),
        sesoi=0.05,
        n_calibration=3,
        n_trials=60,
        planted_requirements={0: "MATERIAL_POSITIVE", 1: "MATERIAL_POSITIVE"},
        null_declaration_paths=[{0: "MATERIAL_POSITIVE", 1: "MATERIAL_POSITIVE"}],
    )
    assert result["familywise_false_positives"] == 0
    assert result["familywise_fpr_ok"]
    assert result["null_declaration_policy"] == "any predeclared classifier conjunction"
    assert result["null_declaration_paths"] == [
        {"component_0": "MATERIAL_POSITIVE", "component_1": "MATERIAL_POSITIVE"}
    ]


def test_preflight_fpr_and_plant_are_not_tautological_with_two_genes():
    result = decision_path_preflight(
        np.ones((1, 2)),
        ["g0", "g1"],
        sesoi=0.10,
        n_calibration=63,
        n_trials=80,
        seed=11,
        namespace="adversarial-two-gene",
        min_genes=2,
    )
    assert 0 < result["familywise_false_positives"] < result["n_trials"]
    assert 0 < result["planted_positive_trials"] < result["n_trials"]
    assert result["familywise_fpr"] > 0.20
    assert result["planted_power"] < 0.70
    assert not result["familywise_fpr_ok"]
    assert not result["planted_power_ok"]
