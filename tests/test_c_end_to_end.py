"""Test C — dictionary and attribution together, against the planted circuit.

This is the number the paper reports. It is only interpretable because A and B ran first:
attribution is exact (A), and dictionary recovery works when features are identifiable (B).
So whatever C loses is attributable to this circuit's features not being identifiable.

The compositional twin's MLP emits exactly three distinct outputs. A transcoder reconstructs
them perfectly and still fails to recover them as features:

  - the decoder BIAS absorbs one concept outright, so phase 0 has no feature at all;
  - one learned feature is a *difference* vector meaning "switch phase 0 -> phase 2";
  - three concepts are split across six live features;
  - phase is recoverable from the features at only 0.66, against 1.00 from the neurons.

Perfect reconstruction, meaningless features. In a real paper you would read that feature's
top-activating examples, see phase-2 sites, and name it "the phase-2 feature."

The graph's coarse shape survives: edge recall is 1.0. Its precision does not (0.6): the
learned encoder reads the token dimensions and the constant, which the planted circuit never
touches. Structure is easier to recover than nodes.
"""

import pytest
import torch

from gfmcd.data.synthetic import make_batch
from gfmcd.dicts.train import explained_variance, train_transcoder
from gfmcd.models.planted import build_compositional
from gfmcd.tracer.graph import (
    PLANTED_EDGES_TRANSCODER,
    recovered_edges_relative,
    transcoder_edges,
)

D_MODEL = 36
N_FEATURES = 16
ACTIVE = 0.5


@pytest.fixture(scope="module")
def traced():
    tokens, labels = make_batch(n_seq=32, body_len=60, seed=13)
    model = build_compositional(match_accuracy=False)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    mask = (labels >= 0).reshape(-1)
    x = cache["L1.resid_mid"].reshape(-1, D_MODEL)[mask]
    y = cache["L1.mlp.out"].reshape(-1, D_MODEL)[mask]
    phase = labels.reshape(-1)[mask]

    tc = train_transcoder(x, y, n_features=N_FEATURES, k=1, steps=4000, seed=2)
    with torch.no_grad():
        feats = tc.encode(x)
    return model, tokens, mask, tc, x, y, phase, feats


def test_reconstruction_is_perfect(traced):
    _, _, _, tc, x, y, _, _ = traced
    assert explained_variance(tc, x, y) > 0.999


def test_the_circuit_has_only_three_output_concepts(traced):
    *_, y, _, _ = traced
    assert len(torch.unique(y.round(decimals=3), dim=0)) == 3


def test_features_split_across_more_features_than_concepts(traced):
    *_, feats = traced
    live = int((feats > ACTIVE).any(dim=0).sum())
    assert live > 3, f"expected splitting, got {live} live features"


def test_one_concept_is_absorbed_into_the_decoder_bias(traced):
    """Phase 0 has no feature. It is what the transcoder emits when nothing fires."""
    _, _, _, tc, _, _, _, _ = traced
    bias_readout = tc.b_dec[6:9].detach()
    assert bias_readout.argmax().item() == 0
    assert float(bias_readout[0]) > 0.9
    assert float(bias_readout[1:].abs().max()) < 0.1


def test_a_learned_feature_is_a_difference_vector_not_a_concept(traced):
    """Some feature reads 'switch phase 0 -> phase 2': one positive lobe, one negative."""
    _, _, _, tc, _, _, _, feats = traced
    live = (feats > ACTIVE).any(dim=0)
    readouts = tc.W_dec[:, 6:9].detach()[live]

    signed = [(float(r.max()) > 0.5 and float(r.min()) < -0.5) for r in readouts]
    assert any(signed), "expected at least one signed difference feature"


def test_phase_is_not_cleanly_recoverable_from_features(traced):
    """The end-to-end shortfall. Attribution was exact, so this loss is dictionary-side."""
    *_, phase, feats = traced

    live = torch.where((feats > ACTIVE).any(dim=0))[0]
    assign = torch.full((N_FEATURES,), -1, dtype=torch.long)
    for i in live:
        sel = feats[:, i] > ACTIVE
        assign[i] = torch.bincount(phase[sel], minlength=3).argmax()

    pred = assign[feats.argmax(dim=-1)]
    accuracy = float((pred == phase).float().mean())
    assert accuracy < 0.99, "if this passes cleanly, the identifiability caveat is wrong"
    assert accuracy > 0.5, f"recovery collapsed entirely: {accuracy:.4f}"


def test_edge_recall_survives_even_though_features_do_not(traced):
    """Coarse structure is recovered: the graph's shape is right, its nodes are not."""
    model, tokens, mask, tc, *_ = traced
    edges = transcoder_edges(model, tc, tokens, mask)
    found = recovered_edges_relative(edges, frac=0.05)

    recall = len(found & PLANTED_EDGES_TRANSCODER) / len(PLANTED_EDGES_TRANSCODER)
    assert recall == 1.0, f"missed {PLANTED_EDGES_TRANSCODER - found}"


def test_edge_precision_degrades_relative_to_the_identity_dictionary(traced):
    """Test A had precision 1.0 with neurons as nodes. A learned dictionary invents edges."""
    model, tokens, mask, tc, *_ = traced
    found = recovered_edges_relative(transcoder_edges(model, tc, tokens, mask), frac=0.05)

    spurious = found - PLANTED_EDGES_TRANSCODER
    precision = len(found & PLANTED_EDGES_TRANSCODER) / len(found)
    assert precision < 1.0, "expected the learned encoder to read dims the circuit does not"
    assert {s[0] for s in spurious} <= {"embed.tok", "embed.const"}, spurious
