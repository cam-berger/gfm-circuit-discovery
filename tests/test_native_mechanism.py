from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gfmcd.splicing.native_mechanism import (
    FinalLayerRun,
    attention_context_reconstruction_error,
    deterministic_control_heads,
    donor_margin,
    donor_token_positions,
    orthogonal_procrustes_cosine,
    position_mask,
    select_head_candidates,
)


def test_donor_positions_and_mask_select_exact_tokens():
    positions = donor_token_positions([2, 4], prefix_tokens=1)
    assert positions.tolist() == [3, 5]
    mask = position_mask(positions, 8)
    assert mask.nonzero().tolist() == [[0, 3], [1, 5]]

    selected = position_mask(
        positions,
        8,
        selected_examples=torch.tensor([True, False]),
    )
    assert selected.nonzero().tolist() == [[0, 3]]


def test_position_mask_rejects_bad_coordinates():
    with pytest.raises(ValueError):
        position_mask(torch.tensor([3, 8]), 8)
    with pytest.raises(ValueError):
        donor_token_positions([-1])


def test_donor_margin_uses_released_class_difference():
    model = SimpleNamespace(
        classifier=nn.Linear(3, 3, bias=False),
        dropout=nn.Identity(),
    )
    with torch.no_grad():
        model.classifier.weight.copy_(torch.eye(3))
    hidden = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 5.0, 1.0]],
            [[0.0, 0.0, 0.0], [4.0, 1.0, 2.0]],
        ]
    )
    margin = donor_margin(model, hidden, torch.tensor([1, 1]))
    assert margin.tolist() == [3.0, -3.0]


def test_attention_context_reconstruction():
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    attention = torch.tensor(
        [[[[1.0, 0.0], [0.25, 0.75]]]]
    )
    heads = torch.matmul(attention, values[:, None])
    run = FinalLayerRun(
        hidden=torch.empty(0),
        heads=heads,
        mlp_write=torch.empty(0),
        query=torch.empty(0),
        key=torch.empty(0),
        value=values,
        attention=attention,
    )
    assert attention_context_reconstruction_error(run, num_heads=1) == 0.0


def _head_summary(damage, recovery, *, q=0.01, low=0.01):
    return {
        "damage": {
            "estimate": damage,
            "ci_low": low,
            "q_bh_32_tests": q,
        },
        "recovery": {
            "estimate": recovery,
            "ci_low": low,
            "q_bh_32_tests": q,
        },
    }


def test_candidate_rule_requires_both_directions_and_ranks_minimum():
    summaries = {
        0: _head_summary(0.20, 0.06),
        1: _head_summary(0.10, 0.09),
        2: _head_summary(0.40, 0.04),
        3: _head_summary(0.30, 0.20, low=-0.01),
    }
    assert select_head_candidates(summaries) == [1, 0]


def test_control_heads_are_unique_reproducible_and_exclude_candidates():
    first = deterministic_control_heads([1, 4], num_heads=16)
    second = deterministic_control_heads([1, 4], num_heads=16)
    assert first == second
    flat = [head for values in first.values() for head in values]
    assert len(flat) == len(set(flat)) == 6
    assert not ({1, 4} & set(flat))


def test_orthogonal_procrustes_cosine_recovers_rotated_geometry():
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(30, 4, generator=generator).numpy()
    q, _ = torch.linalg.qr(
        torch.randn(4, 4, generator=generator)
    )
    y = x @ q.numpy()
    assert orthogonal_procrustes_cosine(x, y) == pytest.approx(1.0)
