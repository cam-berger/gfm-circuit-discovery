"""Pre-softmax score capture -- the path M4d's decomposition reads from.

The read-only pre-hook must leave the eager model unchanged: logits must be identical to the
unmodified model, and the softmax of captured scores must reproduce the model's own reported
attention probabilities.  Padded batches are included because the abandoned attention-registry
implementation silently lost its mask on unequal sequence lengths.  ``require()`` turns a
capture that never fired into an explicit error.

Model-free wherever possible; cases needing the checkpoint skip cleanly without it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gfmcd.circuits.attention_scores import CapturedScores, capture_attention_scores
from gfmcd.circuits.hf_bert import edit_heads

MODEL = ("/Users/cberg/Research/projects/gfm-circuit-discovery/data/models/"
         "splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
HAVE_MODEL = Path(MODEL).is_dir()
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="SpliceBERT checkpoint not present")

LAYER, HEAD = 3, 1


@pytest.fixture(scope="module")
def adapter():
    from gfmcd.splicing.circuit import load_adapter, pick_device
    # eager is REQUIRED: sdpa never materialises the probabilities this compares against.
    return load_adapter(MODEL, pick_device(None), eager=True)


@pytest.fixture(scope="module")
def batch(adapter):
    seqs = [" ".join("ACGT" * 24), " ".join("TTGCAA" * 16)]
    return adapter.tokenizer(seqs, return_tensors="pt", padding=True).to(adapter.device)


def _run(adapter, batch, **kw):
    with torch.no_grad():
        return adapter.model(**batch, **kw)


class _ToySelfAttention(torch.nn.Module):
    """Small eager-attention seam for exercising the capture hook without a checkpoint."""

    attention_head_size = 2
    scaling = attention_head_size**-0.5

    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(4, 4, bias=False)
        self.key = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.query.weight.copy_(torch.eye(4))
            self.key.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.5],
                    ]
                )
            )
        self.last_scores = None

    def forward(self, hidden_states, attention_mask=None):
        shape = (*hidden_states.shape[:-1], 2, self.attention_head_size)
        query = self.query(hidden_states).view(*shape).transpose(1, 2)
        key = self.key(hidden_states).view(*shape).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        self.last_scores = scores
        return torch.softmax(scores, dim=-1)


# ------------------------------------------------------------------- no model needed


def test_require_raises_when_nothing_was_captured():
    box = CapturedScores()
    with pytest.raises(RuntimeError, match="no attention scores"):
        box.require()


def test_require_returns_what_was_stored():
    box = CapturedScores()
    box.scores = torch.zeros(2, 3)
    assert box.require().shape == (2, 3)


def test_fake_attention_capture_selects_head_applies_mask_and_cleans_up_without_mutation():
    """The critical hook plumbing must run in CI even when the SpliceBERT checkpoint is absent."""
    module = _ToySelfAttention()
    adapter = SimpleNamespace(
        num_layers=1,
        num_heads=2,
        head_dim=module.attention_head_size,
        attention_self_modules=[module],
    )
    hidden = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 1.0, 3.0], [0.5, 1.0, 2.0, 0.0]],
            [[0.0, 1.0, 2.0, 1.0], [1.5, 0.5, 0.0, 2.0], [3.0, 1.0, 1.0, 0.5]],
        ]
    )
    mask = torch.zeros(2, 1, 1, 3)
    mask[1, :, :, -1] = -10_000.0

    baseline = module(hidden, attention_mask=mask).clone()
    hooks_before = len(module._forward_pre_hooks)
    with capture_attention_scores(adapter, 0, head=1) as box:
        observed = module(hidden, attention_mask=mask)
        expected_scores = module.last_scores[:, 1].clone()

    captured = box.require()
    assert torch.equal(observed, baseline), "the read-only capture changed the eager forward"
    assert captured.shape == (2, 3, 3)
    assert torch.equal(captured, expected_scores), "the capture selected the wrong head"
    assert captured[1, :, -1].max().item() < -1_000, "the padding mask was not applied"
    assert len(module._forward_pre_hooks) == hooks_before, "the capture hook leaked after exit"

    stale = captured
    module(hidden, attention_mask=mask)
    assert box.require() is stale, "a post-context forward still wrote to the capture box"


# ----------------------------------------------------------------------- validation


@needs_model
def test_rejects_an_out_of_range_layer(adapter):
    with pytest.raises(ValueError):
        with capture_attention_scores(adapter, adapter.num_layers):
            pass


@needs_model
def test_rejects_an_out_of_range_head(adapter):
    with pytest.raises(ValueError):
        with capture_attention_scores(adapter, 0, head=adapter.num_heads):
            pass


@needs_model
def test_nested_captures_on_different_layers_both_fire(adapter, batch):
    with capture_attention_scores(adapter, 0, head=HEAD) as first:
        with capture_attention_scores(adapter, LAYER, head=HEAD) as second:
            _run(adapter, batch)
    assert not torch.allclose(first.require(), second.require(), atol=1e-3)


@needs_model
def test_the_hook_is_removed_on_exit(adapter, batch):
    with capture_attention_scores(adapter, LAYER, head=HEAD) as box:
        _run(adapter, batch)
    stale = box.require()
    _run(adapter, batch)
    assert box.require() is stale, "a forward pass after the context still wrote to the box"


@needs_model
def test_the_hook_is_removed_when_the_body_raises(adapter, batch):
    with pytest.raises(RuntimeError):
        with capture_attention_scores(adapter, LAYER, head=HEAD) as box:
            raise RuntimeError("boom")
    _run(adapter, batch)
    assert box.scores is None, "the hook outlived the context and captured a later pass"


@needs_model
def test_the_model_stays_in_eager_mode_throughout(adapter, batch):
    """The registry route silently disabled masking; this capture must not touch the config."""
    before = adapter.model.config._attn_implementation
    with capture_attention_scores(adapter, LAYER, head=HEAD):
        assert adapter.model.config._attn_implementation == before
        _run(adapter, batch)
    assert adapter.model.config._attn_implementation == before


# ------------------------------------------------------------------- the two memorials


@needs_model
def test_softmax_of_the_captured_scores_reproduces_the_models_own_attention(adapter, batch):
    """THE memorial. Two independent paths to the same probabilities must agree."""
    reference = _run(adapter, batch, output_attentions=True).attentions[LAYER]

    with capture_attention_scores(adapter, LAYER) as box:
        _run(adapter, batch)
    got = torch.softmax(box.require().float(), dim=-1)

    assert got.shape == reference.shape
    assert (got - reference.float()).abs().max().item() < 1e-5


@needs_model
def test_capturing_does_not_change_the_models_output(adapter, batch):
    """The custom implementation is eager verbatim; anything else invalidates the run."""
    base = _run(adapter, batch).logits.float().clone()
    with capture_attention_scores(adapter, LAYER):
        got = _run(adapter, batch).logits.float()
    assert (base - got).abs().max().item() < 1e-5


@needs_model
def test_head_slicing_matches_the_all_head_capture(adapter, batch):
    with capture_attention_scores(adapter, LAYER) as every:
        _run(adapter, batch)
    full = every.require().clone()

    with capture_attention_scores(adapter, LAYER, head=HEAD) as one:
        _run(adapter, batch)
    sliced = one.require()

    assert sliced.shape == (full.shape[0], full.shape[2], full.shape[3])
    assert torch.equal(sliced, full[:, HEAD])


@needs_model
def test_captured_scores_are_not_already_probabilities(adapter, batch):
    """Guards against capturing on the wrong side of the softmax, which would look plausible."""
    with capture_attention_scores(adapter, LAYER, head=HEAD) as box:
        _run(adapter, batch)
    scores = box.require().float()
    rows = scores.sum(-1)
    assert not torch.allclose(rows, torch.ones_like(rows), atol=1e-2)
    assert (scores < 0).any(), "raw scores should take negative values; probabilities cannot"


@needs_model
def test_the_capture_targets_the_requested_layer(adapter, batch):
    """Different layers must yield different scores, or the module match is not discriminating."""
    with capture_attention_scores(adapter, 0, head=HEAD) as a:
        _run(adapter, batch)
    first = a.require().clone()
    with capture_attention_scores(adapter, LAYER, head=HEAD) as b:
        _run(adapter, batch)
    assert not torch.allclose(first, b.require(), atol=1e-3)


# ------------------------------------------------------ composition with interventions


@needs_model
def test_capture_sees_the_effect_of_an_upstream_ablation(adapter, batch):
    """M4d's whole design: ablate early, re-read layer 3's scores. Must not be a cached pass."""
    with capture_attention_scores(adapter, LAYER, head=HEAD) as clean:
        _run(adapter, batch)
    before = clean.require().clone()

    with capture_attention_scores(adapter, LAYER, head=HEAD) as ablated:
        with edit_heads(adapter, [(0, 4)], mode="mean"):
            _run(adapter, batch)
    after = ablated.require()

    assert not torch.allclose(before, after, atol=1e-4)


@needs_model
def test_an_ablation_downstream_of_the_captured_layer_changes_nothing(adapter, batch):
    """Layer 5 cannot affect layer 3's scores; if it appears to, the plumbing is wrong."""
    with capture_attention_scores(adapter, LAYER, head=HEAD) as clean:
        _run(adapter, batch)
    before = clean.require().clone()

    with capture_attention_scores(adapter, LAYER, head=HEAD) as ablated:
        with edit_heads(adapter, [(5, 7)], mode="mean"):
            _run(adapter, batch)
    assert torch.equal(before, ablated.require())


@needs_model
def test_padded_columns_are_masked_out_in_the_captured_scores(adapter):
    """The shorter sequence's pad columns must be unreachable, or the margin denominator lies."""
    seqs = [" ".join("ACGT" * 24), " ".join("ACGT" * 8)]
    enc = adapter.tokenizer(seqs, return_tensors="pt", padding=True).to(adapter.device)
    with capture_attention_scores(adapter, LAYER, head=HEAD) as box:
        with torch.no_grad():
            adapter.model(**enc)
    scores = box.require().float()
    valid = int(enc["attention_mask"][1].sum())
    pad = scores[1, :valid, valid:]
    assert pad.numel() > 0, "the test needs the second sequence to actually be padded"
    assert pad.max().item() < -1e3, f"pad columns are reachable: max score {pad.max().item()}"
