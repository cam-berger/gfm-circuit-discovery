import numpy as np
import pytest
import torch

from gfmcd.splicing.context_joint_lowrank import (
    deterministic_signed_permutation,
    deterministic_random_basis,
    fit_orthogonal_alignment,
    fit_lowrank_structure,
    matched_control_sets,
    mean_row_cosine,
    ordered_nonempty_subsets,
    project_rows,
    projection_matrix_error,
    replace_target_vectors,
    transform_row_basis,
)


def test_joint_subsets_and_matched_control_prefixes_are_size_exact():
    candidates = [14, 8, 12]
    assert ordered_nonempty_subsets(candidates) == [
        (14,),
        (8,),
        (12,),
        (14, 8),
        (14, 12),
        (8, 12),
        (14, 8, 12),
    ]
    controls = {
        14: [10, 1, 0],
        8: [9, 11, 5],
        12: [13, 7, 15],
    }
    assert matched_control_sets(candidates, controls, size=2) == [
        (10, 1),
        (9, 11),
        (13, 7),
    ]


def test_lowrank_fit_recovers_task_axis_and_orthonormal_basis():
    generator = np.random.default_rng(4)
    signal = np.asarray([1.0, 2.0, -1.0, 0.5])
    groups = np.asarray([f"g{i // 2}" for i in range(20)], dtype=object)
    behavior = generator.uniform(0.5, 2.0, size=20)
    matrix = behavior[:, None] * signal[None, :]
    matrix += 0.01 * generator.standard_normal(matrix.shape)
    fitted = fit_lowrank_structure(
        matrix,
        behavior,
        groups,
        max_rank=4,
    )
    assert projection_matrix_error(fitted["basis"]) < 1e-10
    assert np.linalg.norm(fitted["task_direction"]) == pytest.approx(1.0)
    assert np.dot(fitted["task_direction"], signal) > 0.99 * np.linalg.norm(
        signal
    )


def test_random_basis_and_projection_are_deterministic_and_dimension_matched():
    first = deterministic_random_basis(8, 3, tag="study", draw=0)
    second = deterministic_random_basis(8, 3, tag="study", draw=0)
    assert np.array_equal(first, second)
    assert projection_matrix_error(first) < 1e-10
    matrix = np.arange(16, dtype=np.float64).reshape(2, 8)
    projected = project_rows(matrix, first)
    assert projected.shape == matrix.shape
    assert np.allclose(project_rows(projected, first), projected)
    assert np.allclose(
        project_rows(matrix, np.zeros((0, 8), dtype=np.float64)),
        0.0,
    )


def test_replace_target_vectors_changes_only_selected_rows_and_positions():
    write = torch.zeros(2, 4, 3)
    positions = torch.tensor([1, 3])
    vectors = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    replaced = replace_target_vectors(write, positions, vectors)
    assert replaced[0, 1].tolist() == [1.0, 2.0, 3.0]
    assert replaced[1, 3].tolist() == [4.0, 5.0, 6.0]
    assert torch.count_nonzero(replaced[0, [0, 2, 3]]) == 0
    assert torch.count_nonzero(replaced[1, [0, 1, 2]]) == 0


def test_orthogonal_alignment_recovers_rotated_code():
    generator = np.random.default_rng(23)
    source = generator.standard_normal((40, 8))
    rotation, _ = np.linalg.qr(generator.standard_normal((8, 8)))
    target = source @ rotation

    fitted = fit_orthogonal_alignment(source, target)

    assert fitted["orthogonality_error"] < 1e-12
    assert fitted["relative_error"] < 1e-12
    assert fitted["mean_row_cosine"] > 1 - 1e-12
    assert np.allclose(source @ fitted["rotation"], target, atol=1e-12)


def test_transform_row_basis_preserves_orthonormality():
    generator = np.random.default_rng(29)
    basis, _ = np.linalg.qr(generator.standard_normal((12, 4)))
    rotation, _ = np.linalg.qr(generator.standard_normal((12, 12)))
    transported = transform_row_basis(basis.T, rotation)
    assert projection_matrix_error(transported) < 1e-12


def test_signed_permutation_is_deterministic_and_norm_preserving():
    matrix = np.arange(24, dtype=np.float64).reshape(4, 6)
    first = deterministic_signed_permutation(matrix, tag="control", draw=2)
    second = deterministic_signed_permutation(matrix, tag="control", draw=2)
    other = deterministic_signed_permutation(matrix, tag="control", draw=3)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.allclose(np.linalg.norm(first, axis=1), np.linalg.norm(matrix, axis=1))


def test_mean_row_cosine_ignores_zero_pairs():
    left = np.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    right = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, -1.0]])
    assert mean_row_cosine(left, right) == pytest.approx(0.0)
