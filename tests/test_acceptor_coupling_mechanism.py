import copy
from dataclasses import replace

import numpy as np
import pytest

from gfmcd.splicing.acceptor_coupling_mechanism import (
    AV_CELL_ORDER,
    DONOR_WINDOW_OFFSETS,
    SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE,
    av_factorial_contrasts,
    av_four_cell_reconstruction,
    crossfit_project,
    crossfit_svd,
    delivery_candidate_route_gaps,
    delivery_sandwich_estimands,
    deterministic_group_folds,
    deterministic_nested_random_bases,
    deterministic_random_basis,
    donor_window_indices,
    donor_window_positions,
    energy_match_rows,
    fit_svd_grid,
    fixed_calibration_overlap_null,
    hard_g_margin_from_base_logits,
    l3_to_u_mediation_estimands,
    linear_mlm_base_logits,
    linear_mlm_g_margin,
    mean_squared_principal_cosine,
    orthogonal_projector,
    ordered_overlap_null_draw_identities,
    partner_pseudo_specific,
    project_rows,
    projector_diagnostics,
    random_rank_energy_control,
    reconstruct_attention_value,
    replay_delivery_sandwich_record,
    replay_l3_to_u_mediation_record,
    replace_projected_component,
    select_crossfit_outcome_rank,
    select_linear_mlm_margin_inputs,
    validate_svd_numerical_rank,
)


def test_donor_window_is_the_fixed_four_token_support_and_checks_bounds():
    assert DONOR_WINDOW_OFFSETS == (-1, 0, 1, 2)
    assert donor_window_positions(5, sequence_length=12) == (4, 5, 6, 7)
    assert donor_window_indices(
        np.asarray([2, 5]),
        sequence_length=10,
        token_offset=1,
    ).tolist() == [[2, 3, 4, 5], [5, 6, 7, 8]]

    with pytest.raises(ValueError, match="outside"):
        donor_window_positions(0, sequence_length=10)
    with pytest.raises(ValueError, match="outside"):
        donor_window_indices([8], sequence_length=10)


@pytest.mark.parametrize("invalid", [True, np.bool_(False), 5.0])
def test_coordinate_helpers_reject_boolean_and_nonintegral_coordinates(invalid):
    with pytest.raises((TypeError, ValueError), match="integer"):
        donor_window_positions(invalid)
    with pytest.raises((TypeError, ValueError), match="integer"):
        donor_window_positions(5, sequence_length=invalid)
    with pytest.raises((TypeError, ValueError), match="integer"):
        donor_window_indices([5], sequence_length=12, token_offset=invalid)


def test_attention_value_reconstruction_and_four_cell_factorial_are_exact():
    a_minus = np.asarray([[0.75, 0.25]])
    a_plus = np.asarray([[0.25, 0.75]])
    v_minus = np.asarray([[[1.0, 0.0], [0.0, 2.0]]])
    v_plus = np.asarray([[[3.0, 1.0], [1.0, 4.0]]])

    assert np.allclose(
        reconstruct_attention_value(a_minus, v_minus),
        [[0.75, 0.50]],
    )
    cells = av_four_cell_reconstruction(a_plus, a_minus, v_plus, v_minus)
    assert tuple(cells) == AV_CELL_ORDER
    assert np.allclose(cells["baseline"], [[0.75, 0.50]])
    assert np.allclose(cells["joint"], [[1.50, 3.25]])

    outcomes = {
        "baseline": np.asarray([1.0, 2.0]),
        "address_only": np.asarray([3.0, 4.0]),
        "value_only": np.asarray([5.0, 7.0]),
        "joint": np.asarray([10.0, 13.0]),
    }
    contrasts = av_factorial_contrasts(outcomes)
    assert np.array_equal(contrasts["joint_lift"], [9.0, 11.0])
    assert np.array_equal(contrasts["value_effect_at_address_plus"], [7.0, 9.0])
    assert np.array_equal(contrasts["interaction_AxV"], [3.0, 4.0])


def test_av_factorial_rejects_missing_or_broadcast_cells():
    with pytest.raises(ValueError, match="exactly the keys"):
        av_factorial_contrasts({"baseline": 0.0})
    with pytest.raises(ValueError, match="same shape"):
        av_factorial_contrasts(
            {
                "baseline": np.zeros(2),
                "address_only": np.zeros(2),
                "value_only": np.zeros(2),
                "joint": 0.0,
            }
        )


def test_partner_pseudo_specific_uses_native_minus_intervened_raw_cells():
    result = partner_pseudo_specific(
        native_partner=np.asarray([3.0, 4.0]),
        intervened_partner=np.asarray([2.0, 2.5]),
        native_pseudo=np.asarray([1.0, 2.0]),
        intervened_pseudo=np.asarray([0.9, 1.8]),
    )
    assert np.allclose(result["partner"], [1.0, 1.5])
    assert np.allclose(result["pseudo"], [0.1, 0.2])
    assert np.allclose(result["specific"], [0.9, 1.3])


def test_projector_and_component_replacement_are_orthogonal_and_exact():
    basis = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    projector = orthogonal_projector(basis)
    diagnostics = projector_diagnostics(projector)
    assert diagnostics["rank"] == 2
    assert diagnostics["symmetry_max_abs"] == 0.0
    assert diagnostics["idempotence_max_abs"] == 0.0

    target = np.asarray([[1.0, 2.0, 3.0]])
    source = np.asarray([[5.0, 7.0, 11.0]])
    assert np.array_equal(project_rows(target, projector), [[1.0, 2.0, 0.0]])
    assert np.array_equal(
        replace_projected_component(target, source, projector),
        [[5.0, 7.0, 3.0]],
    )
    with pytest.raises(ValueError, match="not orthonormal"):
        orthogonal_projector([[1.0, 0.0], [1.0, 0.0]])


def test_svd_energy_summary_is_diagnostic_and_retains_the_complete_basis_grid():
    rows = np.diag([4.0, 2.0, 1.0])
    fitted = fit_svd_grid(
        rows,
        candidate_ranks=(1, 2, 3),
        diagnostic_energy_target=0.90,
    )
    # cumulative energy: 16/21, 20/21, 21/21
    assert fitted.diagnostic_energy_rank == 2
    assert fitted.diagnostic_energy_target_reached
    assert fitted.cumulative_energy_by_rank[1][1] == pytest.approx(20 / 21)
    record = fitted.record()
    assert record["diagnostic_energy_summary_rule"].startswith(
        "NOT_THE_R5B_SCIENTIFIC_SELECTOR"
    )
    assert "selection_rule" not in record
    assert len(record["basis_sha256"]) == 64
    assert len(record["singular_values_sha256"]) == 64
    assert len(record["numerical_rank_diagnostics_sha256"]) == 64
    assert set(record["projector_sha256_by_rank"]) == {"1", "2", "3"}
    assert record["maximum_requested_rank"] == 3
    assert record["numerical_rank"] == 3
    assert record["numerical_rank_relative_tolerance"] == (
        SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE
    )
    assert record["minimum_required_singular_value_ratio"] == pytest.approx(0.25)
    assert "sigma_i > sigma_1" in record["numerical_rank_rule"]
    assert record["numerical_rank_diagnostics_sha256"] == (
        validate_svd_numerical_rank(
            fitted.singular_values,
            required_rank=3,
        )["numerical_rank_diagnostics_sha256"]
    )
    assert record["fitted_arrays_serialized"] is False
    assert "basis" not in record
    assert fitted.basis.shape == (3, 3)
    assert np.allclose(fitted.basis @ fitted.basis.T, np.eye(3))

    unresolved = fit_svd_grid(
        rows,
        candidate_ranks=(1,),
        diagnostic_energy_target=0.90,
    )
    assert unresolved.diagnostic_energy_rank == 1
    assert not unresolved.diagnostic_energy_target_reached


def test_svd_grid_rejects_exact_and_near_rank_deficiency_scale_relatively():
    exact_rank_one = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ]
    )
    with pytest.raises(ValueError, match="numerically rank deficient"):
        fit_svd_grid(
            exact_rank_one,
            candidate_ranks=(1, 2),
            diagnostic_energy_target=0.9,
        )

    near_deficient = np.diag(
        [1.0, 0.5 * SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE]
    )
    with pytest.raises(ValueError, match="relative_tolerance=1e-10"):
        fit_svd_grid(
            near_deficient,
            candidate_ranks=(1, 2),
            diagnostic_energy_target=0.9,
        )

    just_present = np.diag(
        [1.0, 2.0 * SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE]
    )
    fitted = fit_svd_grid(
        just_present * 1e-200,
        candidate_ranks=(1, 2),
        diagnostic_energy_target=1.0,
    )
    assert fitted.numerical_rank == 2
    assert fitted.minimum_required_singular_value_ratio == pytest.approx(
        2.0 * SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE
    )

    refit_diagnostics = validate_svd_numerical_rank(
        [4.0, 2.0, 1.0],
        required_rank=3,
    )
    assert refit_diagnostics["numerical_rank"] == 3
    assert refit_diagnostics["minimum_required_singular_value_ratio"] == 0.25
    assert len(refit_diagnostics["singular_values_sha256"]) == 64
    assert len(refit_diagnostics["numerical_rank_diagnostics_sha256"]) == 64


def test_crossfit_svd_keeps_genes_disjoint_and_projects_every_held_out_row():
    groups = ["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"]
    folds = [0, 0, 0, 0, 1, 1, 1, 1]
    rows = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    fitted = crossfit_svd(
        rows,
        groups,
        folds,
        candidate_ranks=(1, 2),
        diagnostic_energy_target=0.99,
        group_average=True,
    )
    assert len(fitted.folds) == 2
    for fold in fitted.folds:
        assert not (set(fold.derivation_groups) & set(fold.evaluation_groups))
        assert fold.svd.n_fit_rows == 2  # one mean row per derivation gene
        assert fold.svd.diagnostic_energy_rank == 2
    projected = crossfit_project(rows, fitted, rank=2)
    assert projected.shape == rows.shape
    assert np.allclose(projected, rows)
    assert fitted.record()["group_average"] is True
    assert fitted.record()["row_fold_ids"] == [str(fold) for fold in folds]
    assert len(fitted.record()["ordered_row_fold_ids_sha256"]) == 64
    reordered = replace(fitted, row_fold_ids=tuple(reversed(fitted.row_fold_ids)))
    assert (
        reordered.record()["ordered_row_fold_ids_sha256"]
        != fitted.record()["ordered_row_fold_ids_sha256"]
    )
    with pytest.raises(ValueError, match="positive integer"):
        crossfit_project(rows, fitted, rank=2.0)

    bad_folds = [0, 1, 0, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="may not cross folds"):
        crossfit_svd(
            rows,
            groups,
            bad_folds,
            candidate_ranks=(1,),
            diagnostic_energy_target=0.5,
        )

    overlapping = replace(
        fitted,
        folds=(
            fitted.folds[0],
            replace(
                fitted.folds[1],
                evaluation_indices=fitted.folds[0].evaluation_indices.copy(),
            ),
        ),
    )
    with pytest.raises(ValueError, match="exactly once"):
        crossfit_project(rows, overlapping, rank=2)

    # One held-out fold leaves two derivation genes whose means are collinear.  The
    # cross-fit constructor must propagate the same maximum-rank validity gate.
    deficient_rows = rows.copy()
    deficient_rows[4:8] = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    with pytest.raises(ValueError, match="numerically rank deficient"):
        crossfit_svd(
            deficient_rows,
            groups,
            folds,
            candidate_ranks=(1, 2),
            diagnostic_energy_target=0.99,
            group_average=True,
        )


def test_crossfit_outcome_rank_uses_ninety_percent_of_max_and_both_random_gates():
    ranks = (1, 2, 4, 8, 16, 32)
    sufficiency = {1: 0.60, 2: 0.91, 4: 1.00, 8: 0.98, 16: 0.97, 32: 0.96}
    overlap = {rank: True for rank in ranks}
    overlap[2] = False
    controls = {rank: rank >= 2 for rank in ranks}
    selected = select_crossfit_outcome_rank(
        sufficiency,
        overlap,
        controls,
        rank_grid=ranks,
    )
    assert selected.maximum_heldout_sufficiency == pytest.approx(1.0)
    assert selected.recovery_threshold == pytest.approx(0.90)
    assert selected.selected_rank == 4
    assert selected.status == "RANK_SELECTED"

    no_control = select_crossfit_outcome_rank(
        sufficiency,
        overlap,
        {rank: False for rank in ranks},
        rank_grid=ranks,
    )
    assert no_control.selected_rank is None
    assert no_control.status == "NO_U_A_RANK"

    with pytest.raises(ValueError, match="complete rank_grid"):
        select_crossfit_outcome_rank(
            {rank: sufficiency[rank] for rank in ranks[:-1]},
            overlap,
            controls,
            rank_grid=ranks,
        )
    with pytest.raises(TypeError, match="literal booleans"):
        select_crossfit_outcome_rank(
            sufficiency,
            {rank: ("false" if rank == 2 else True) for rank in ranks},
            controls,
            rank_grid=ranks,
        )
    with pytest.raises(TypeError, match="literal booleans"):
        select_crossfit_outcome_rank(
            sufficiency,
            {rank: np.bool_(True) for rank in ranks},
            controls,
            rank_grid=ranks,
        )
    with pytest.raises(ValueError, match="rank_grid"):
        select_crossfit_outcome_rank(
            {True: 1.0, 2: 1.0},
            {True: True, 2: True},
            {True: True, 2: True},
            rank_grid=(True, 2),
        )
    with pytest.raises(TypeError, match="keys must be integer ranks"):
        select_crossfit_outcome_rank(
            {
                True: sufficiency[1],
                2: sufficiency[2],
                4: sufficiency[4],
                8: sufficiency[8],
                16: sufficiency[16],
                32: sufficiency[32],
            },
            overlap,
            controls,
            rank_grid=ranks,
        )


def test_deterministic_group_folds_never_split_a_gene():
    groups = ["g2", "g1", "g2", "g3", "g1"]
    first = deterministic_group_folds(groups, n_folds=3, namespace="frozen")
    second = deterministic_group_folds(groups, n_folds=3, namespace="frozen")
    assert np.array_equal(first, second)
    assert first[0] == first[2]
    assert first[1] == first[4]
    with pytest.raises(TypeError, match="integer"):
        deterministic_group_folds(groups, n_folds=2.0)
    with pytest.raises(TypeError, match="integer"):
        deterministic_group_folds(groups, n_folds=np.bool_(True))


def test_nested_haar_controls_are_exact_prefixes_and_overlap_is_absolute():
    ranks = (1, 2, 4)
    bases = deterministic_nested_random_bases(
        8,
        ranks,
        seed=17,
        namespace="nested-control",
        draw=3,
    )
    again = deterministic_nested_random_bases(
        8,
        ranks,
        seed=17,
        namespace="nested-control",
        draw=3,
    )
    assert all(np.array_equal(bases[rank], again[rank]) for rank in ranks)
    assert np.array_equal(bases[1], bases[4][:1])
    assert np.array_equal(bases[2], bases[4][:2])
    independent_namespace = deterministic_nested_random_bases(
        8,
        ranks,
        seed=17,
        namespace="nested-control-test-pool",
        draw=3,
    )
    assert not np.array_equal(bases[4], independent_namespace[4])
    assert mean_squared_principal_cosine(bases[2], bases[2]) == pytest.approx(1.0)
    left = np.eye(8)[:2]
    orthogonal = np.eye(8)[4:6]
    assert mean_squared_principal_cosine(left, orthogonal) == pytest.approx(0.0)


def test_overlap_null_freezes_nuisance_parameters_on_disjoint_calibration_pool():
    ranks = (1, 2)
    observed = {1: 0.40, 2: 0.55}
    calibration = {
        1: np.asarray([0.10, 0.20, 0.30, 0.20]),
        2: np.asarray([0.20, 0.30, 0.40, 0.30]),
    }
    test = {
        1: np.asarray([0.15, 0.25, 0.35]),
        2: np.asarray([0.25, 0.35, 0.45]),
    }
    calibration_ids = ordered_overlap_null_draw_identities(
        namespace="calibration-pool",
        seed=17,
        n_draws=4,
    )
    test_ids = ordered_overlap_null_draw_identities(
        namespace="test-pool",
        seed=17,
        n_draws=3,
    )
    fitted = fixed_calibration_overlap_null(
        observed,
        calibration,
        test,
        rank_grid=ranks,
        calibration_draw_identities=calibration_ids,
        test_draw_identities=test_ids,
    )
    changed_test = {rank: values + 0.01 for rank, values in test.items()}
    refit_test = fixed_calibration_overlap_null(
        observed,
        calibration,
        changed_test,
        rank_grid=ranks,
        calibration_draw_identities=calibration_ids,
        test_draw_identities=test_ids,
    )
    assert fitted.calibration_mean_by_rank == refit_test.calibration_mean_by_rank
    assert fitted.calibration_sd_by_rank == refit_test.calibration_sd_by_rank
    assert fitted.observed_standardized_by_rank == refit_test.observed_standardized_by_rank
    assert not np.array_equal(
        fitted.test_max_standardized,
        refit_test.test_max_standardized,
    )
    record = fitted.record()
    assert record["n_calibration_draws"] == 4
    assert record["n_test_draws"] == 3
    assert record["calibration_pool_sha256"] != record["test_pool_sha256"]
    assert len(record["ordered_calibration_draw_identities_sha256"]) == 64
    assert len(record["ordered_test_draw_identities_sha256"]) == 64
    assert (
        record["calibration_draw_identity_set_sha256"]
        != record["test_draw_identity_set_sha256"]
    )
    assert record["calibration_first_draw_identity"]["draw"] == 0
    assert record["calibration_last_draw_identity"]["draw"] == 3
    assert record["semantic_draw_identity_intersection_count"] == 0
    assert record["draw_identity_sets_disjoint"] is True
    assert record["null_arrays_serialized"] is False

    with pytest.raises(ValueError, match="must be disjoint"):
        fixed_calibration_overlap_null(
            observed,
            calibration,
            calibration,
            rank_grid=ranks,
            calibration_draw_identities=calibration_ids,
            test_draw_identities=calibration_ids,
        )


def test_overlap_null_rejects_permuted_partial_and_duplicate_semantic_draw_reuse():
    ranks = (1, 2)
    observed = {1: 0.4, 2: 0.5}
    calibration = {
        1: np.asarray([0.1, 0.2, 0.3, 0.4]),
        2: np.asarray([0.2, 0.3, 0.4, 0.5]),
    }
    permuted = {rank: values[::-1] for rank, values in calibration.items()}
    calibration_ids = ordered_overlap_null_draw_identities(
        namespace="shared-pool", seed=5, n_draws=4
    )
    with pytest.raises(ValueError, match="draw identities must be disjoint"):
        fixed_calibration_overlap_null(
            observed,
            calibration,
            permuted,
            rank_grid=ranks,
            calibration_draw_identities=calibration_ids,
            test_draw_identities=tuple(reversed(calibration_ids)),
        )

    test = {
        1: np.asarray([0.15, 0.25, 0.35]),
        2: np.asarray([0.25, 0.35, 0.45]),
    }
    novel = ordered_overlap_null_draw_identities(
        namespace="novel-pool", seed=5, n_draws=2
    )
    partially_reused = (calibration_ids[2], *novel)
    with pytest.raises(ValueError, match="draw identities must be disjoint"):
        fixed_calibration_overlap_null(
            observed,
            calibration,
            test,
            rank_grid=ranks,
            calibration_draw_identities=calibration_ids,
            test_draw_identities=partially_reused,
        )

    duplicate_test_ids = (novel[0], novel[0], novel[1])
    with pytest.raises(ValueError, match="duplicate semantic draw identities"):
        fixed_calibration_overlap_null(
            observed,
            calibration,
            test,
            rank_grid=ranks,
            calibration_draw_identities=calibration_ids,
            test_draw_identities=duplicate_test_ids,
        )


@pytest.mark.parametrize("invalid", [True, np.bool_(False), 2.5])
def test_random_basis_integral_arguments_reject_booleans_and_floats(invalid):
    with pytest.raises((TypeError, ValueError), match="integer"):
        deterministic_random_basis(8, invalid, seed=0, namespace="x", draw=0)
    with pytest.raises((TypeError, ValueError), match="integer"):
        deterministic_random_basis(invalid, 2, seed=0, namespace="x", draw=0)
    with pytest.raises((TypeError, ValueError), match="integer"):
        deterministic_random_basis(8, 2, seed=invalid, namespace="x", draw=0)
    with pytest.raises((TypeError, ValueError), match="integer"):
        deterministic_nested_random_bases(8, (1, 2), seed=0, namespace="x", draw=invalid)


def test_random_rank_energy_control_matches_rank_and_per_row_energy():
    rows = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, -1.0, 1.0, -2.0],
        ]
    )
    reference = np.asarray(
        [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.5, 0.0],
        ]
    )
    control = random_rank_energy_control(
        rows,
        reference,
        rank=2,
        seed=7,
        namespace="rank-energy",
        draw=0,
    )
    again = random_rank_energy_control(
        rows,
        reference,
        rank=2,
        seed=7,
        namespace="rank-energy",
        draw=0,
    )
    assert np.array_equal(control["basis"], again["basis"])
    assert projector_diagnostics(control["projector"])["rank"] == 2
    assert np.allclose(
        np.linalg.norm(control["energy_matched_projection"], axis=1),
        np.linalg.norm(reference, axis=1),
    )
    assert np.allclose(
        project_rows(control["energy_matched_projection"], control["projector"]),
        control["energy_matched_projection"],
    )


def test_energy_matching_rejects_an_impossible_zero_candidate():
    with pytest.raises(ValueError, match="zero candidate"):
        energy_match_rows([[1.0, 0.0]], [[0.0, 0.0]])


def test_energy_matching_distinguishes_exact_zero_from_tiny_nonzero_rows():
    tiny = 1e-15  # deliberately below the retained legacy atol keyword
    with pytest.raises(ValueError, match="exact-zero candidate"):
        energy_match_rows([[tiny, 0.0]], [[0.0, 0.0]], atol=1e-12)

    zero_reference = energy_match_rows(
        [[0.0, 0.0]],
        [[tiny, -tiny]],
        atol=1e-12,
    )
    assert np.array_equal(zero_reference, np.zeros((1, 2)))

    tiny_matched = energy_match_rows(
        [[tiny, 0.0]],
        [[tiny, tiny]],
        atol=1e-12,
    )
    assert np.linalg.norm(tiny_matched[0]) == pytest.approx(tiny, rel=1e-15)
    assert not np.array_equal(tiny_matched, np.zeros((1, 2)))

    normal_from_tiny = energy_match_rows(
        [[3.0, 4.0]],
        [[tiny, -tiny]],
        atol=1e-12,
    )
    assert np.linalg.norm(normal_from_tiny[0]) == pytest.approx(5.0)


def test_linear_mlm_inputs_reproduce_selected_full_decoder_logits_and_margin():
    linear_input = np.asarray([[1.0, 2.0], [-1.0, 0.5]])
    decoder_weight = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 2.0],
            [2.0, -1.0],
            [0.5, 0.5],
        ]
    )
    decoder_bias = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, -0.1])
    base_ids = {"A": 4, "C": 1, "G": 5, "T": 2}
    full_logits = linear_input @ decoder_weight.T + decoder_bias
    expected = full_logits[:, [4, 1, 5, 2]]

    selected = select_linear_mlm_margin_inputs(decoder_weight, decoder_bias, base_ids)
    assert selected["base_order"] == ("A", "C", "G", "T")
    observed = linear_mlm_base_logits(
        linear_input,
        decoder_weight,
        decoder_bias,
        base_ids,
    )
    assert np.allclose(observed, expected)
    expected_margin = expected[:, 2] - np.max(expected[:, [0, 1, 3]], axis=1)
    assert np.allclose(hard_g_margin_from_base_logits(observed), expected_margin)
    assert np.allclose(
        linear_mlm_g_margin(linear_input, decoder_weight, decoder_bias, base_ids),
        expected_margin,
    )

    for invalid in (True, np.bool_(False), 4.0):
        bad = dict(base_ids)
        bad["A"] = invalid
        with pytest.raises(TypeError, match="integer decoder rows"):
            select_linear_mlm_margin_inputs(decoder_weight, decoder_bias, bad)


def test_l3_to_u_estimands_use_the_frozen_sufficiency_and_necessity_cells():
    result = l3_to_u_mediation_estimands(
        m_minus=np.asarray([1.0, 2.0]),
        m_l3_plus=np.asarray([2.2, 3.5]),
        m_minus_with_u_from_l3_plus=np.asarray([1.8, 3.0]),
        m_l3_plus_with_u_from_minus=np.asarray([1.4, 2.4]),
        leg_id="partner-donor",
    )
    assert np.allclose(result["estimands"]["direct_l3_rescue"], [1.2, 1.5])
    assert np.allclose(result["estimands"]["S_U"], [0.8, 1.0])
    assert np.allclose(result["estimands"]["N_U"], [0.8, 1.1])
    replayed = replay_l3_to_u_mediation_record(result)
    assert np.array_equal(replayed["estimands"]["S_U"], result["estimands"]["S_U"])


@pytest.mark.parametrize(
    "section,field",
    [
        ("cells", "m_minus"),
        ("cells", "m_l3_plus"),
        ("cells", "m_minus_with_u_from_l3_plus"),
        ("cells", "m_l3_plus_with_u_from_minus"),
        ("estimands", "direct_l3_rescue"),
        ("estimands", "S_U"),
        ("estimands", "N_U"),
    ],
)
def test_l3_to_u_replay_rejects_every_primitive_or_stored_estimand_tamper(
    section,
    field,
):
    record = l3_to_u_mediation_estimands(
        m_minus=np.asarray([1.0, 2.0]),
        m_l3_plus=np.asarray([2.2, 3.5]),
        m_minus_with_u_from_l3_plus=np.asarray([1.8, 3.0]),
        m_l3_plus_with_u_from_minus=np.asarray([1.4, 2.4]),
        leg_id="partner-donor",
    )
    tampered = copy.deepcopy(record)
    tampered[section][field][0] += 0.125
    with pytest.raises(ValueError, match="primitive-cell replay"):
        replay_l3_to_u_mediation_record(tampered)


def test_l3_to_u_replay_requires_the_exact_nested_schema():
    record = l3_to_u_mediation_estimands(
        m_minus=1.0,
        m_l3_plus=2.0,
        m_minus_with_u_from_l3_plus=1.5,
        m_l3_plus_with_u_from_minus=1.25,
        leg_id="partner-donor",
    )
    extra = copy.deepcopy(record)
    extra["untrusted_summary"] = 99.0
    with pytest.raises(ValueError, match="exactly the keys"):
        replay_l3_to_u_mediation_record(extra)

    missing = copy.deepcopy(record)
    del missing["estimands"]["S_U"]
    with pytest.raises(ValueError, match="exactly the keys"):
        replay_l3_to_u_mediation_record(missing)


def test_delivery_sandwich_and_candidate_route_gaps_use_identical_conditional_legs():
    common = {
        "m_minus": np.asarray([1.0, 2.0]),
        "m_c_plus": np.asarray([3.0, 5.0]),
        "leg_id": "partner-donor:L3H1-clean-restore",
    }
    candidate = delivery_sandwich_estimands(
        **common,
        m_c_plus_with_d_minus=np.asarray([2.0, 3.0]),
        m_minus_with_d_from_c_plus=np.asarray([1.8, 3.5]),
        component_id="L4H11",
    )
    route = delivery_sandwich_estimands(
        **common,
        m_c_plus_with_d_minus=np.asarray([2.8, 4.8]),
        m_minus_with_d_from_c_plus=np.asarray([1.1, 2.3]),
        component_id="L4H0",
    )
    assert np.allclose(candidate["estimands"]["N_d"], [1.0, 2.0])
    assert np.allclose(candidate["estimands"]["S_d"], [0.8, 1.5])
    assert np.allclose(candidate["estimands"]["direct_l3_rescue"], [2.0, 3.0])
    gaps = delivery_candidate_route_gaps(candidate, route)
    assert np.allclose(gaps["N_candidate_minus_donor_route"], [0.8, 1.8])
    assert np.allclose(gaps["S_candidate_minus_donor_route"], [0.7, 1.2])
    replayed = replay_delivery_sandwich_record(candidate)
    assert np.array_equal(replayed["estimands"]["N_d"], candidate["estimands"]["N_d"])

    mismatched = delivery_sandwich_estimands(
        m_minus=np.asarray([1.0, 2.1]),
        m_c_plus=common["m_c_plus"],
        m_c_plus_with_d_minus=np.asarray([2.8, 4.8]),
        m_minus_with_d_from_c_plus=np.asarray([1.1, 2.3]),
        leg_id=common["leg_id"],
        component_id="bad-route",
    )
    with pytest.raises(ValueError, match="do not share the m_minus leg"):
        delivery_candidate_route_gaps(candidate, mismatched)


@pytest.mark.parametrize(
    "section,field",
    [
        ("cells", "m_minus"),
        ("cells", "m_c_plus"),
        ("cells", "m_c_plus_with_d_minus"),
        ("cells", "m_minus_with_d_from_c_plus"),
        ("estimands", "direct_l3_rescue"),
        ("estimands", "N_d"),
        ("estimands", "S_d"),
    ],
)
def test_delivery_replay_rejects_every_primitive_or_stored_estimand_tamper(
    section,
    field,
):
    record = delivery_sandwich_estimands(
        m_minus=np.asarray([1.0, 2.0]),
        m_c_plus=np.asarray([3.0, 5.0]),
        m_c_plus_with_d_minus=np.asarray([2.0, 3.0]),
        m_minus_with_d_from_c_plus=np.asarray([1.8, 3.5]),
        leg_id="partner-donor:L3H1-clean-restore",
        component_id="L4H11",
    )
    tampered = copy.deepcopy(record)
    tampered[section][field][0] += 0.125
    with pytest.raises(ValueError, match="primitive-cell replay"):
        replay_delivery_sandwich_record(tampered)


def test_delivery_route_gaps_replays_primitives_instead_of_trusting_stored_estimands():
    common = {
        "m_minus": np.asarray([1.0, 2.0]),
        "m_c_plus": np.asarray([3.0, 5.0]),
        "leg_id": "partner-donor:L3H1-clean-restore",
    }
    candidate = delivery_sandwich_estimands(
        **common,
        m_c_plus_with_d_minus=np.asarray([2.0, 3.0]),
        m_minus_with_d_from_c_plus=np.asarray([1.8, 3.5]),
        component_id="L4H11",
    )
    route = delivery_sandwich_estimands(
        **common,
        m_c_plus_with_d_minus=np.asarray([2.8, 4.8]),
        m_minus_with_d_from_c_plus=np.asarray([1.1, 2.3]),
        component_id="L4H0",
    )
    candidate["estimands"]["N_d"][0] = 99.0
    with pytest.raises(ValueError, match="primitive-cell replay"):
        delivery_candidate_route_gaps(candidate, route)


def test_delivery_replay_requires_the_exact_nested_schema():
    record = delivery_sandwich_estimands(
        m_minus=1.0,
        m_c_plus=3.0,
        m_c_plus_with_d_minus=2.0,
        m_minus_with_d_from_c_plus=1.8,
        leg_id="same-leg",
        component_id="candidate",
    )
    extra = copy.deepcopy(record)
    extra["estimands"]["composite"] = 1.0
    with pytest.raises(ValueError, match="exactly the keys"):
        replay_delivery_sandwich_record(extra)


def test_delivery_sandwich_keeps_necessity_and_sufficiency_separate():
    result = delivery_sandwich_estimands(
        m_minus=0.5,
        m_c_plus=1.5,
        m_c_plus_with_d_minus=1.0,
        m_minus_with_d_from_c_plus=0.9,
        leg_id="same-leg",
        component_id="candidate",
    )
    assert result["estimands"] == {
        "direct_l3_rescue": 1.0,
        "N_d": 0.5,
        "S_d": 0.4,
    }
    assert "sandwich_score" not in result["estimands"]
