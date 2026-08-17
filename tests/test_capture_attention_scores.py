"""Pre-softmax score capture -- the path M4d's decomposition reads from.

The capture replaces the model's attention implementation, so the first thing to establish is
that it changed nothing: the logits must be identical to the unmodified eager model, and the
softmax of what it captured must reproduce that model's own reported attention probabilities.
Both are cheap and both can fail.

The registry makes silence the default failure: ``ALL_ATTENTION_FUNCTIONS.get_interface``
falls back to ``eager_attention_forward`` for a name it does not know, so a broken
registration runs a perfectly normal model and returns no scores at all.  ``require()`` is
what converts that into an error, and it is tested here directly.

Model-free wherever possible; cases needing the checkpoint skip cleanly without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gfmcd.circuits.hf_bert import (
    CapturedScores, capture_attention_scores, edit_heads,
)

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


# ------------------------------------------------------------------- no model needed


def test_require_raises_when_nothing_was_captured():
    box = CapturedScores()
    with pytest.raises(RuntimeError, match="no attention scores"):
        box.require()


def test_require_returns_what_was_stored():
    box = CapturedScores()
    box.scores = torch.zeros(2, 3)
    assert box.require().shape == (2, 3)


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
