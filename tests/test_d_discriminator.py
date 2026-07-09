"""Test D — calibrate H3's discriminator against ground truth.

H3 claims Jacobian sparsity separates genuine composition from a bag-of-motifs shortcut. The
proposal asserts the signature and applies it. Here we measure it on models whose mechanism we
wrote ourselves, before it is ever pointed at a real GFM.

Run at neuron level (identity dictionary). The twins' neurons are monosemantic by construction,
and their residual code is not identifiable anyway (D14), so training a dictionary first would
measure the dictionary rather than the discriminator.

Three planted mechanisms, deliberately:

    compositional   attends to the start marker, computes (i-start-1)%3   in-degree 2
    bag             five local motif detectors, independent weak votes    in-degree ~15
    shortcut        one unigram lookup, no context                        in-degree 1

The bag is the alternative H3 names. The shortcut is the one it forgets, and it is the reason
this test exists: sparsity measures IN-DEGREE, not composition.
"""

import pytest
import torch

from gfmcd.data.synthetic import make_batch
from gfmcd.models.planted import build_compositional, build_heuristic, build_shortcut
from gfmcd.tracer.jacobian import active_edge_count, auc, hoyer_sparsity, neuron_jacobian

SEP_THRESHOLD = 0.03


@pytest.fixture(scope="module")
def twins():
    tokens, labels = make_batch(n_seq=32, body_len=60, seed=9)
    n = tokens.shape[-1]
    models = {
        "compositional": build_compositional(match_accuracy=False),
        "bag": build_heuristic(seq_len=n),
        "shortcut": build_shortcut(seq_len=n),
    }
    mask = labels >= 0
    scores, degree = {}, {}
    for name, m in models.items():
        jac = neuron_jacobian(m, tokens)
        scores[name] = hoyer_sparsity(jac)[mask]
        degree[name] = active_edge_count(jac)[mask].float()
    return scores, degree


def test_in_degree_matches_the_planted_mechanism(twins):
    _, degree = twins
    assert degree["compositional"].mean() == pytest.approx(2.0, abs=0.01)
    assert degree["shortcut"].mean() == pytest.approx(3.0, abs=0.01)
    assert degree["bag"].mean() > 12.0


def test_discriminator_separates_composition_from_a_bag_of_motifs(twins):
    """H3's stated claim. Holds, and by a margin that is not on the third decimal place."""
    scores, _ = twins
    separation = float(scores["compositional"].mean() - scores["bag"].mean())
    assert auc(scores["compositional"], scores["bag"]) > 0.8
    assert separation > SEP_THRESHOLD, f"separation only {separation:.4f}"


def test_discriminator_cannot_separate_composition_from_a_lone_shortcut(twins):
    """H3's blind spot, measured.

    A model that cheats with ONE trick has in-degree 1 and is therefore near-maximally sparse.
    The margin against composition collapses by an order of magnitude relative to the bag, so
    the sparsity signature cannot license the claim "this model composes." It can only license
    "this model does not use many weak features."
    """
    scores, _ = twins
    vs_bag = float(scores["compositional"].mean() - scores["bag"].mean())
    vs_shortcut = float(scores["compositional"].mean() - scores["shortcut"].mean())

    assert vs_shortcut < SEP_THRESHOLD, f"shortcut margin {vs_shortcut:.4f} unexpectedly large"
    assert vs_bag > 5 * abs(vs_shortcut), f"bag={vs_bag:.4f} shortcut={vs_shortcut:.4f}"


def test_sparsity_is_monotone_in_in_degree(twins):
    """The mechanism of the metric, stated as a testable fact rather than an interpretation."""
    scores, degree = twins
    order = sorted(scores, key=lambda k: float(degree[k].mean()))
    sparsities = [float(scores[k].mean()) for k in order]
    assert sparsities == sorted(sparsities, reverse=True), dict(zip(order, sparsities))
