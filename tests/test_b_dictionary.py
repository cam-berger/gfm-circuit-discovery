"""Test B — dictionary recovery under planted superposition.

Test A held the dictionary at identity so attribution error stood alone. Test B holds
attribution out entirely so dictionary error stands alone. Only Test C combines them, and
only because A and B can then say which half moved.

Ground truth is `n_atoms` non-orthogonal directions in a lower-dimensional space, with
independently sampled k-sparse supports. Match learned decoder rows to planted directions
by |cosine| under Hungarian assignment, then ask whether the matched features fire on the
same inputs (MCC).

`test_planted_code_is_not_identifiable` records why the activations cannot simply be
lifted from the planted model.
"""

import pytest
import torch

from gfmcd.data.synthetic import make_batch
from gfmcd.dicts.matching import hungarian_match, mcc
from gfmcd.dicts.toy import random_directions, sample_sparse_code
from gfmcd.dicts.train import explained_variance, train_sae
from gfmcd.models.planted import build_compositional

N_ATOMS = 64
D_PROJ = 32
K = 3
N_FEATURES = 128
N_SAMPLES = 4096


@pytest.fixture(scope="module")
def superposed():
    code = sample_sparse_code(N_SAMPLES, N_ATOMS, K, seed=11)
    directions = random_directions(N_ATOMS, D_PROJ, seed=12)
    return code, directions, code @ directions


@pytest.fixture(scope="module")
def fitted(superposed):
    _, _, acts = superposed
    return train_sae(acts, n_features=N_FEATURES, k=K, steps=4000, seed=1)


def test_superposition_is_real(superposed):
    _, directions, _ = superposed
    gram = directions @ directions.T
    off_diag = gram - torch.diag(torch.diag(gram))
    assert N_ATOMS > D_PROJ
    assert off_diag.abs().max() > 0.3, "directions are near-orthogonal; not superposed"


def test_code_is_exactly_k_sparse(superposed):
    code, _, _ = superposed
    assert code.gt(0).sum(dim=-1).unique().tolist() == [K]


def test_sae_reconstructs_the_activations(fitted, superposed):
    _, _, acts = superposed
    assert explained_variance(fitted, acts) > 0.95


def test_sae_recovers_the_planted_directions(fitted, superposed):
    _, directions, _ = superposed
    _, cosines = hungarian_match(fitted.decoder_weight, directions)
    assert float(cosines.min()) > 0.9, f"worst matched direction: {float(cosines.min()):.4f}"


def test_matched_features_fire_on_the_same_inputs(fitted, superposed):
    """Test B's headline number."""
    code, directions, acts = superposed
    idx, _ = hungarian_match(fitted.decoder_weight, directions)

    with torch.no_grad():
        feats = fitted.encode(acts)

    score = mcc(code.gt(0), feats[:, idx].gt(1e-4))
    assert score > 0.9, f"MCC = {score:.4f}"


def test_planted_code_is_not_identifiable():
    """Why Test B cannot use the planted model's own residual as its activations.

    `phase_feature` is a deterministic function of `pos_phase` and `start_phase`, so the
    residual code takes only 3*3*5 = 45 distinct values. A dictionary learner reconstructs
    such data perfectly while recovering combinations rather than atoms. Any transcoder
    trained on a circuit whose features are deterministically coupled inherits this, and
    the resulting features must not be read as independent causes.
    """
    tokens, _ = make_batch(n_seq=16, body_len=60, seed=5)
    model = build_compositional(match_accuracy=False)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    code = cache["L1.resid_post"][..., :14].reshape(-1, 14).round()
    n_distinct = len(torch.unique(code, dim=0))
    assert n_distinct <= 45, f"expected <= 45 distinct codes, got {n_distinct}"

    pos, start, phase = code[:, 0:3], code[:, 3:6], code[:, 6:9]
    implied = (pos.argmax(-1) - start.argmax(-1) - 1) % 3
    assert torch.equal(phase.argmax(-1), implied), "phase must be determined by pos and start"
