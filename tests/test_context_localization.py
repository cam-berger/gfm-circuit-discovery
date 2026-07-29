from collections import Counter

import pytest
import torch

from gfmcd.splicing.context_localization import (
    ACCEPTOR_SCAN_WINDOWS,
    CENTRAL_SCAN_WINDOW,
    DONOR_SCAN_WINDOWS,
    RelativeWindow,
    directed_dinucleotide_counts,
    nucleotide_token_mask,
    shuffle_relative_window,
    target_token_mask,
)


def test_scan_windows_are_equal_width_nonoverlapping_and_include_central():
    for windows in (DONOR_SCAN_WINDOWS, ACCEPTOR_SCAN_WINDOWS):
        assert all(window.width == 32 for window in windows)
        assert CENTRAL_SCAN_WINDOW in windows
        assert all(
            left.end == right.start
            for left, right in zip(windows, windows[1:])
        )
    assert len(DONOR_SCAN_WINDOWS) == 9
    assert len(ACCEPTOR_SCAN_WINDOWS) == 7


def test_relative_window_halves_and_bounds():
    assert RelativeWindow(-16, 16).halves() == (
        RelativeWindow(-16, 0),
        RelativeWindow(0, 16),
    )
    assert RelativeWindow(-4, 4).absolute(10, 20) == (6, 14)
    with pytest.raises(ValueError, match="exceeds"):
        RelativeWindow(-11, -1).absolute(10, 20)
    with pytest.raises(ValueError, match="positive"):
        RelativeWindow(2, 2)


def test_shuffle_is_deterministic_and_preserves_exact_local_invariants():
    sequence = "ACGTGCAATGCTAGCTAGGCTAACCGTTACGT" * 4
    window = RelativeWindow(-16, 16)
    kwargs = {
        "sequence": sequence,
        "anchor": 64,
        "window": window,
        "key": ("example-1", "donor", window.name),
        "draw": 0,
        "candidates": 64,
    }
    first, audit = shuffle_relative_window(**kwargs)
    second, second_audit = shuffle_relative_window(**kwargs)
    start, end = window.absolute(64, len(sequence))
    before = sequence[start:end]
    after = first[start:end]

    assert first == second
    assert audit == second_audit
    assert audit.invariants_pass
    assert audit.hamming_fraction >= 0.25
    assert first[:start] == sequence[:start]
    assert first[end:] == sequence[end:]
    assert Counter(before) == Counter(after)
    assert directed_dinucleotide_counts(before) == directed_dinucleotide_counts(
        after
    )


def test_independent_draws_remain_valid_even_if_they_collide():
    sequence = "AACCGGTTAACCGGTTAACCGGTTAACCGGTT" * 4
    window = RelativeWindow(8, 40)
    outputs = []
    for draw in (0, 1):
        shuffled, audit = shuffle_relative_window(
            sequence,
            anchor=48,
            window=window,
            key=("example-2", "acceptor", window.name),
            draw=draw,
        )
        outputs.append(shuffled)
        assert audit.invariants_pass
        assert audit.draw == draw
    assert len(outputs) == 2


def test_token_masks_translate_nucleotide_coordinates_with_special_tokens():
    region = nucleotide_token_mask(
        anchors=[10, 12],
        window=RelativeWindow(-2, 2),
        sequence_length=32,
        prefix_tokens=1,
        device="cpu",
    )
    target = target_token_mask(
        nucleotide_positions=[9, 11],
        sequence_length=32,
        prefix_tokens=1,
        device="cpu",
    )
    assert region.shape == (2, 34)
    assert torch.equal(region[0].nonzero().flatten(), torch.tensor([9, 10, 11, 12]))
    assert torch.equal(region[1].nonzero().flatten(), torch.tensor([11, 12, 13, 14]))
    assert torch.equal(target[0].nonzero().flatten(), torch.tensor([10]))
    assert torch.equal(target[1].nonzero().flatten(), torch.tensor([12]))
