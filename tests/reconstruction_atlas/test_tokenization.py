from __future__ import annotations

from gfmcd.reconstruction_atlas.schema import Interval
from gfmcd.reconstruction_atlas.tokenization import (
    build_mask_plan,
    fair_alternative_token_ids,
)


def fixed_encoder(sequence: str):
    offsets = [(0, 0)]
    ids = [101]
    vocab = {}
    for start in range(0, len(sequence), 2):
        end = min(len(sequence), start + 2)
        token = sequence[start:end]
        vocab.setdefault(token, 1000 + sum(ord(x) for x in token))
        offsets.append((start, end))
        ids.append(vocab[token])
    offsets.append((0, 0))
    ids.append(102)
    return ids, offsets


def test_minimal_bpe_cover_and_masking_include_overhang() -> None:
    sequence = "AACCGGTT"
    ids, offsets = fixed_encoder(sequence)
    plan = build_mask_plan(
        sequence,
        Interval(1, 5),
        input_ids=ids,
        offsets=offsets,
    )
    assert plan.token_indices == (1, 2, 3)
    assert plan.covered == Interval(0, 6)
    assert plan.covered_nucleotides == 6
    assert plan.overhang_nucleotides == 2
    masked = plan.masked_input_ids(ids, 99)
    assert masked[1:4] == [99, 99, 99]
    assert masked[0] == 101 and masked[-1] == 102


def test_alternative_is_eligible_only_with_identical_unmasked_geometry() -> None:
    sequence = "AACCGGTT"
    ids, offsets = fixed_encoder(sequence)
    plan = build_mask_plan(
        sequence,
        Interval(2, 4),
        input_ids=ids,
        offsets=offsets,
    )
    alternative = fair_alternative_token_ids(sequence, plan, "TT", fixed_encoder)
    assert alternative is not None
    assert len(alternative) == 1

    def content_dependent_encoder(text: str):
        if text[2:4] == "TT":
            return [101, 1, 2, 3, 102], [(0, 0), (0, 2), (2, 3), (3, 8), (0, 0)]
        return fixed_encoder(text)

    assert (
        fair_alternative_token_ids(
            sequence,
            plan,
            "TT",
            content_dependent_encoder,
        )
        is None
    )
