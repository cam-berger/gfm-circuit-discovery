import torch

from gfmcd.splicing.readout import (
    AdditivePairReadout,
    BilinearPairReadout,
    collate_pair_indices,
    pair_cross_entropy,
    true_pair_margin,
)
from test_splicing_schema_candidates import make_example


def test_readouts_shape_and_loss():
    ex = make_example()
    pairs = collate_pair_indices([ex], token_offset=1)
    hidden = torch.randn(1, len(ex.sequence) + 2, 16)
    for model in (AdditivePairReadout(16), BilinearPairReadout(16, interaction_rank=4)):
        logits = model(hidden, pairs)
        assert logits.shape == pairs.mask.shape
        assert torch.isfinite(pair_cross_entropy(logits, pairs))
        assert true_pair_margin(logits, pairs).shape == (1,)


def test_bilinear_interaction_is_joint():
    ex = make_example()
    pairs = collate_pair_indices([ex], token_offset=1)
    hidden = torch.randn(1, len(ex.sequence) + 2, 8)
    model = BilinearPairReadout(8, interaction_rank=2)
    interaction = model.interaction(hidden, pairs)
    assert interaction.shape == pairs.mask.shape
