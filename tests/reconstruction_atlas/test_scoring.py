from __future__ import annotations

import math

import torch

from gfmcd.reconstruction_atlas.schema import Interval
from gfmcd.reconstruction_atlas.scoring import score_from_logits
from gfmcd.reconstruction_atlas.tokenization import MaskPlan


def test_true_token_score_is_normalized_by_complete_bpe_cover() -> None:
    plan = MaskPlan(
        site=Interval(1, 4),
        token_indices=(1, 2),
        true_token_ids=(3, 4),
        token_offsets=((0, 2), (2, 5)),
        covered=Interval(0, 5),
        sequence_length=8,
    )
    logits = torch.zeros((4, 7))
    logits[1, 3] = 2.0
    logits[2, 4] = 1.0
    score = score_from_logits(
        "x",
        plan,
        logits,
        alternative_token_ids={"decoy": (2, 2)},
    )
    lp = torch.log_softmax(logits, dim=-1)
    expected = float(lp[1, 3] + lp[2, 4])
    assert math.isclose(score.true_token_log_probability, expected, abs_tol=1e-7)
    assert math.isclose(
        score.true_token_log_probability_per_covered_nt,
        expected / 5,
        abs_tol=1e-7,
    )
    expected_margin = float((lp[1, 3] + lp[2, 4] - lp[1, 2] - lp[2, 2]) / 5)
    assert math.isclose(
        score.alternative_margins_per_covered_nt["decoy"],
        expected_margin,
        abs_tol=1e-7,
    )


def test_score_rejects_wrong_alternative_token_count() -> None:
    plan = MaskPlan(
        site=Interval(0, 2),
        token_indices=(1,),
        true_token_ids=(3,),
        token_offsets=((0, 2),),
        covered=Interval(0, 2),
        sequence_length=4,
    )
    try:
        score_from_logits(
            "x",
            plan,
            torch.zeros((3, 5)),
            alternative_token_ids={"bad": (1, 2)},
        )
    except ValueError as exc:
        assert "different token count" in str(exc)
    else:
        raise AssertionError("wrong-size alternative was accepted")
