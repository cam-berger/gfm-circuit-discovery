"""GFM-nano: the two objective arms must differ only in their loss, never in their wiring.

The causal-masking test is the one that matters. An AR model that can see the future has a trivial
next-nucleotide task, would report a beautiful loss, and would learn nothing about reading frame.
"""

import pytest
import torch

from gfmcd.models.gfm_nano import MASK, GFMNano, NanoConfig, ar_batch, mlm_batch

IGNORE = -100


@pytest.fixture(scope="module")
def tokens():
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 4, (4, 64), generator=g)


def _model(causal: bool) -> GFMNano:
    torch.manual_seed(0)
    return GFMNano(NanoConfig(n_layers=2, d_model=64, n_heads=4, d_mlp=256, max_len=64, causal=causal)).eval()


def test_forward_shape_and_cache_keys(tokens):
    model = _model(causal=False)
    logits, cache = model.run_with_cache(tokens)
    assert logits.shape == (4, 64, 6)

    for key in ("resid.embed", "resid.final", "L0.attn.scores", "L0.attn.pattern", "L0.attn.z",
                "L0.mlp.acts", "L0.resid_mid", "L1.resid_post"):
        assert key in cache, f"missing cache key {key}"


def test_the_two_arms_have_identical_parameter_counts():
    """H2 contrasts objectives, so architecture and scale must be held exactly fixed."""
    assert _model(causal=True).n_params == _model(causal=False).n_params


@torch.no_grad()
def test_causal_model_cannot_see_the_future(tokens):
    model = _model(causal=True)
    base = model(tokens)

    perturbed = tokens.clone()
    perturbed[:, 40:] = (perturbed[:, 40:] + 1) % 4
    after = model(perturbed)

    assert torch.allclose(base[:, :40], after[:, :40], atol=1e-5), "future leaked into the past"
    assert not torch.allclose(base[:, 40:], after[:, 40:]), "perturbation had no effect at all"


@torch.no_grad()
def test_bidirectional_model_does_see_the_future(tokens):
    """The control: without the mask, later tokens must influence earlier positions."""
    model = _model(causal=False)
    base = model(tokens)

    perturbed = tokens.clone()
    perturbed[:, 40:] = (perturbed[:, 40:] + 1) % 4

    assert not torch.allclose(base[:, :40], model(perturbed)[:, :40], atol=1e-5)


def test_ar_targets_are_the_sequence_shifted(tokens):
    inputs, targets = ar_batch(tokens)
    assert inputs.shape == targets.shape == (4, 63)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_mlm_supervises_only_corrupted_positions(tokens):
    g = torch.Generator().manual_seed(1)
    inputs, targets = mlm_batch(tokens, rate=0.15, generator=g)

    supervised = targets != IGNORE
    assert torch.equal(targets[supervised], tokens[supervised])
    assert (targets[~supervised] == IGNORE).all()

    rate = supervised.float().mean()
    assert 0.10 < rate < 0.20, f"masking rate {rate:.3f}"


def test_mlm_leaves_a_tenth_of_targets_uncorrupted(tokens):
    """Without the untouched branch the model learns '[MASK] means predict', not the language."""
    g = torch.Generator().manual_seed(2)
    big = tokens.repeat(64, 1)
    inputs, targets = mlm_batch(big, rate=0.15, generator=g)

    supervised = targets != IGNORE
    is_mask = inputs[supervised] == MASK
    unchanged = inputs[supervised] == big[supervised]

    assert 0.75 < is_mask.float().mean() < 0.85
    assert unchanged.float().mean() > 0.05, "no uncorrupted targets: the model can shortcut on [MASK]"


def test_mask_token_never_appears_in_targets(tokens):
    g = torch.Generator().manual_seed(3)
    _, targets = mlm_batch(tokens, generator=g)
    assert (targets[targets != IGNORE] < 4).all(), "targets must be real nucleotides"
