"""Rung 0, before any tracer exists: do the twins have the properties we claim?

If these fail, nothing downstream means anything. The twins must be indistinguishable
by behaviour and separable by intervention.
"""

import pytest
import torch

from gfmcd.data.synthetic import IGNORE, ablate_marker, bayes_unigram_accuracy, make_batch
from gfmcd.models.planted import build_compositional, build_heuristic, phase_accuracy

CHANCE = 1.0 / 3.0


@pytest.fixture(scope="module")
def batch():
    return make_batch(n_seq=64, body_len=90, seed=1)


@pytest.fixture(scope="module")
def target(batch):
    """Gate calibration target, measured on a separate calibration split."""
    cal_tokens, cal_labels = make_batch(n_seq=64, body_len=90, seed=1234)
    return phase_accuracy(build_heuristic(seq_len=cal_tokens.shape[-1]), cal_tokens, cal_labels)


def test_compositional_twin_is_exact(batch):
    tokens, labels = batch
    model = build_compositional(match_accuracy=False)
    assert phase_accuracy(model, tokens, labels) > 0.99


def test_heuristic_twin_beats_the_unigram_ceiling(batch):
    """The bag pools five offsets, so it must beat any single-nucleotide predictor."""
    tokens, labels = batch
    acc = phase_accuracy(build_heuristic(seq_len=tokens.shape[-1]), tokens, labels)
    assert acc > bayes_unigram_accuracy() + 0.1


def test_twins_are_behaviourally_matched(batch, target):
    tokens, labels = batch
    comp = phase_accuracy(build_compositional(match_accuracy=True, target_accuracy=target), tokens, labels)
    heur = phase_accuracy(build_heuristic(seq_len=tokens.shape[-1]), tokens, labels)
    assert abs(comp - heur) < 0.03, f"comp={comp:.4f} heur={heur:.4f}"


def test_marker_ablation_separates_the_mechanisms(batch):
    """The compositional twin's frame anchor is causal; the heuristic has no anchor."""
    tokens, labels = batch
    ablated = ablate_marker(tokens, seed=2)

    comp = build_compositional(match_accuracy=False)
    heur = build_heuristic(seq_len=tokens.shape[-1])

    comp_intact = phase_accuracy(comp, tokens, labels)
    comp_ablated = phase_accuracy(comp, ablated, labels)
    heur_intact = phase_accuracy(heur, tokens, labels)
    heur_ablated = phase_accuracy(heur, ablated, labels)

    assert comp_ablated < 0.45, f"marker ablation left frame recoverable: {comp_ablated:.4f}"
    assert comp_intact - comp_ablated > 0.5
    assert abs(heur_intact - heur_ablated) < 0.02, "heuristic must not depend on the marker"


def test_compositional_twin_routes_through_the_marker_head(batch):
    """The layer-0 head should put essentially all its mass on the start marker."""
    tokens, _ = batch
    comp = build_compositional(match_accuracy=False)
    _, cache = comp.run_with_cache(tokens[:8])
    pattern = cache["L0.attn.pattern"]

    starts = (tokens[:8] == 4).float().argmax(dim=-1)
    mass = torch.stack([pattern[s, 0, :, starts[s]].mean() for s in range(8)])
    assert mass.min() > 0.99, f"attention leaked off the marker: {mass.min():.4f}"


def test_phase_feature_is_a_clean_one_hot(batch):
    tokens, labels = batch
    comp = build_compositional(match_accuracy=False)
    _, cache = comp.run_with_cache(tokens[:8])
    feats = cache["L1.resid_post"][..., 6:9]

    mask = labels[:8] != IGNORE
    active = feats[mask]
    assert active.sum(dim=-1).min() > 0.99
    assert active.max(dim=-1).values.min() > 0.99
    assert active.argmax(dim=-1).eq(labels[:8][mask]).all()
