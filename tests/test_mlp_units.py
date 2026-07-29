import numpy as np

from gfmcd.splicing.mlp_units import (
    composed_classifier_margin,
    exact_single_unit_effects,
    output_norm_matched_controls,
    rank_units_bidirectional,
    unit_group_write_delta,
)


def parameters(width):
    return {
        "layer_norm_weight": np.ones(width),
        "layer_norm_bias": np.zeros(width),
        "layer_norm_eps": 1e-5,
        "classifier_direction": np.arange(1, width + 1, dtype=float),
        "classifier_bias": 0.25,
    }


def test_full_unit_group_reconstructs_mlp_write_delta():
    generator = np.random.default_rng(2)
    activations = generator.standard_normal((3, 7))
    weight = generator.standard_normal((5, 7))
    expected = activations @ weight.T
    actual = unit_group_write_delta(activations, weight, range(7))
    assert np.allclose(actual, expected)
    assert np.count_nonzero(
        unit_group_write_delta(activations, weight, [])
    ) == 0


def test_exact_single_unit_effects_match_literal_one_unit_composition():
    generator = np.random.default_rng(3)
    batch, width, units = 4, 5, 6
    clean_residual = generator.standard_normal((batch, width))
    corrupt_residual = generator.standard_normal((batch, width))
    corrupt_activation = generator.standard_normal((batch, units))
    delta = generator.standard_normal((batch, units))
    clean_activation = corrupt_activation + delta
    weight = generator.standard_normal((width, units))
    bias = generator.standard_normal(width)
    corrupt_write = corrupt_activation @ weight.T + bias
    clean_write = clean_activation @ weight.T + bias
    params = parameters(width)
    clean_margin = composed_classifier_margin(
        clean_residual, clean_write, **params
    )
    corrupt_margin = composed_classifier_margin(
        corrupt_residual, corrupt_write, **params
    )
    effects = exact_single_unit_effects(
        delta,
        weight,
        clean_residual=clean_residual,
        clean_write=clean_write,
        corrupt_residual=corrupt_residual,
        corrupt_write=corrupt_write,
        clean_margin=clean_margin,
        corrupt_margin=corrupt_margin,
        parameters=params,
        chunk_size=2,
    )
    for unit in range(units):
        contribution = delta[:, [unit]] @ weight[:, [unit]].T
        expected_damage = clean_margin - composed_classifier_margin(
            clean_residual, clean_write - contribution, **params
        )
        expected_recovery = composed_classifier_margin(
            corrupt_residual, corrupt_write + contribution, **params
        ) - corrupt_margin
        assert np.allclose(effects["damage"][:, unit], expected_damage)
        assert np.allclose(effects["recovery"][:, unit], expected_recovery)


def test_bidirectional_ranking_uses_weaker_gene_mean_direction():
    groups = np.asarray(["a", "a", "b", "b"], dtype=object)
    damage = np.asarray(
        [[3.0, 1.0], [3.0, 1.0], [3.0, 1.0], [3.0, 1.0]]
    )
    recovery = np.asarray(
        [[0.2, 0.9], [0.2, 0.9], [0.2, 0.9], [0.2, 0.9]]
    )
    ranked = rank_units_bidirectional(damage, recovery, groups)
    assert ranked["order"].tolist() == [1, 0]


def test_output_norm_controls_are_exact_unique_and_candidate_free():
    norms = np.linspace(0.1, 2.0, 20)
    candidates = [2, 5, 8, 11]
    first = output_norm_matched_controls(
        candidates, norms, tag="unit-controls", draws=3
    )
    second = output_norm_matched_controls(
        candidates, norms, tag="unit-controls", draws=3
    )
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    for control in first:
        assert control.size == len(candidates)
        assert len(np.unique(control)) == control.size
        assert not set(control) & set(candidates)
