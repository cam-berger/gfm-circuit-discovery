"""Nucleotide-dependency instrument on synthetic sequences (TO_FIX.md §26 step 7).

Marked `network` (loads SpliceBERT). Checks the method runs and behaves sanely: non-negative scores,
and — on a sequence with a planted local motif — a nearby position generally shows higher dependency
than a distant one. Not a biological claim; a plumbing + sanity check for the instrument.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.network
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def adapter():
    from gfmcd.models.adapters import SpliceBERTAdapter

    try:
        return SpliceBERTAdapter(device="cpu")
    except Exception as e:
        pytest.skip(f"SpliceBERT weights unavailable: {e}")


def test_dependency_is_nonnegative_and_finite(adapter):
    from gfmcd.studies.nucleotide_dependency import substitution_dependency

    seq = "ACGT" * 20  # 80 nt
    d = substitution_dependency(adapter, seq, query_nt=10, target_nt=12)
    assert d >= 0.0 and d == d  # non-negative, not NaN


def test_local_dependency_exceeds_distal_on_average(adapter):
    from gfmcd.studies.nucleotide_dependency import substitution_dependency

    seq = "GT" + "ACGTACGTAC" * 8  # a splice-donor-like GT then filler, 82 nt
    target = 40
    near = sum(substitution_dependency(adapter, seq, q, target) for q in (38, 39, 41, 42)) / 4
    far = sum(substitution_dependency(adapter, seq, q, target) for q in (2, 3, 78, 79)) / 4
    assert near >= far  # local context matters at least as much as distant, on average


def test_partner_vs_decoy_returns_advantage(adapter):
    from gfmcd.studies.nucleotide_dependency import partner_vs_decoy

    seq = "ACGTACGT" * 10
    out = partner_vs_decoy(adapter, seq, anchor_nt=20, true_partner_nt=22, decoy_nt=70)
    assert set(out) == {"dependency_true", "dependency_decoy", "advantage"}
    assert out["advantage"] == out["dependency_true"] - out["dependency_decoy"]
