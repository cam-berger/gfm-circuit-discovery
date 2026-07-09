"""Cell B of the necessity 2x2: no shortcut, detector difficulty matched to real hg38.

If the shortcut leaks back in, cell B stops being a no-shortcut cell. If the detector is easier or
harder than real data's, a difference between B and D is uninterpretable. Both are asserted.
"""

import numpy as np
import pytest

from gfmcd.data.genome import upstream_atg_count
from gfmcd.data.uniform_cds import (
    ATG,
    KOZAK_P_MATCHED,
    bayes_detector_frame_accuracy,
    make_batch_hard_detector,
    nucleotide_phase_information,
)

REAL_DECOY_FRACTION = 0.429
REAL_DECOY_MEAN = 0.60
REAL_FRAME_ACC = 0.9215


@pytest.fixture(scope="module")
def batch():
    return make_batch_hard_detector(2500, window=256, seed=3)


def test_decoy_rate_matches_real_hg38(batch):
    tokens, _, phase = batch
    start = np.argmax(phase == 0, axis=1)
    n_up = upstream_atg_count(tokens, start)
    assert abs((n_up > 0).mean() - REAL_DECOY_FRACTION) < 0.05
    assert abs(n_up.mean() - REAL_DECOY_MEAN) < 0.10


def test_detector_difficulty_matches_real_hg38(batch):
    """The whole point of cell B: same anchor difficulty, no shortcut."""
    tokens, _, phase = batch
    _, frame = bayes_detector_frame_accuracy(tokens, phase)
    assert abs(frame - REAL_FRAME_ACC) < 0.02, f"frame accuracy {frame:.4f} vs real {REAL_FRAME_ACC}"


def test_the_shortcut_is_still_absent(batch):
    """Kozak edits and the ATG add a trace of phase information; it must stay negligible."""
    tokens, _, phase = batch
    info = nucleotide_phase_information(tokens, phase)
    assert info < 0.001, f"I(nt;phase) = {info:.6f}; real hg38 CDS is 0.0241"


def test_a_harder_detector_lowers_frame_accuracy(batch):
    """Mutation test: the calibration knob does what it claims."""
    _, _, ph_weak = make_batch_hard_detector(1200, seed=4, kozak_p=0.0)
    tok_weak, _, _ = make_batch_hard_detector(1200, seed=4, kozak_p=0.0)
    _, weak = bayes_detector_frame_accuracy(tok_weak, ph_weak)

    tok_str, _, ph_str = make_batch_hard_detector(1200, seed=4, kozak_p=1.0)
    _, strong = bayes_detector_frame_accuracy(tok_str, ph_str)

    assert weak < 0.60 < strong, f"weak {weak:.3f} strong {strong:.3f}"


def test_start_codon_is_present_and_labelled(batch):
    tokens, aa, phase = batch
    for r in range(200):
        s = int(np.argmax(phase[r] == 0))
        assert tokens[r, s : s + 3].tolist() == ATG.tolist()
        assert aa[r, s] == aa[r, s + 1] == aa[r, s + 2] >= 0
