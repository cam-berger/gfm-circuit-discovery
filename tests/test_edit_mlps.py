"""MLP-write ablation, the component class this repo deliberately deferred.

``hf_bert``'s module docstring records that "MLP features and edge-level mediation are added
only after a reproducible" base, and ``edit_heads`` has no MLP path -- which is why AC4's scope
column says MLP writes are "out of scope ... no mean-ablation primitive exists for them", while
HCA18 reports MLP writes dominating its acceptor-cluster signals. This adds the primitive.

The write is hooked at ``layer.output.dense``, whose result is what BertOutput adds to the
residual stream before LayerNorm. Mean is taken over batch and sequence, matching
``edit_heads(mode="mean")`` exactly, so head and MLP ablations are on one scale.

Model-free wherever possible; cases needing the checkpoint skip cleanly without it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from gfmcd.circuits.hf_bert import edit_mlp_writes

MODEL = ("/Users/cberg/Research/projects/gfm-circuit-discovery/data/models/"
         "splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
HAVE_MODEL = Path(MODEL).is_dir()
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="SpliceBERT checkpoint not present")


@pytest.fixture(scope="module")
def adapter():
    from gfmcd.splicing.circuit import load_adapter, pick_device
    return load_adapter(MODEL, pick_device(None), eager=False)


@pytest.fixture(scope="module")
def batch(adapter):
    ids = adapter.tokenizer([" ".join("ACGT" * 32)], return_tensors="pt").to(adapter.device)
    return ids


def _logits(adapter, batch):
    with torch.no_grad():
        return adapter.model(**batch).logits.float().clone()


def test_rejects_an_out_of_range_layer():
    class Fake:
        num_layers = 6
        encoder_layers = tuple(range(6))
    with pytest.raises(ValueError):
        with edit_mlp_writes(Fake(), [6]):
            pass


def test_rejects_a_negative_layer():
    class Fake:
        num_layers = 6
        encoder_layers = tuple(range(6))
    with pytest.raises(ValueError):
        with edit_mlp_writes(Fake(), [-1]):
            pass


def test_rejects_an_unknown_mode():
    class Handle:
        def remove(self):
            pass

    class Dense:
        def register_forward_hook(self, hook):
            self.hook = hook
            return Handle()

    class Layer:
        class Output:
            dense = Dense()

        output = Output()

    class Fake:
        num_layers = 6
        encoder_layers = tuple(Layer() for _ in range(6))

    with pytest.raises(ValueError):
        with edit_mlp_writes(Fake(), [0], mode="nonsense"):
            Fake.encoder_layers[0].output.dense.hook(
                None, None, torch.zeros(1, 1, 1)
            )


@needs_model
def test_empty_layer_list_is_a_noop(adapter, batch):
    base = _logits(adapter, batch)
    with edit_mlp_writes(adapter, []):
        got = _logits(adapter, batch)
    assert torch.equal(base, got)


@needs_model
def test_zero_ablation_changes_the_output(adapter, batch):
    """The primitive must actually do something, or every null it produces is vacuous."""
    base = _logits(adapter, batch)
    with edit_mlp_writes(adapter, [0], mode="zero"):
        got = _logits(adapter, batch)
    assert not torch.allclose(base, got)


@needs_model
def test_mean_ablation_changes_the_output(adapter, batch):
    base = _logits(adapter, batch)
    with edit_mlp_writes(adapter, [0], mode="mean"):
        got = _logits(adapter, batch)
    assert not torch.allclose(base, got)


@needs_model
def test_hooks_are_removed_on_exit(adapter, batch):
    base = _logits(adapter, batch)
    with edit_mlp_writes(adapter, [0, 1, 2], mode="zero"):
        pass
    assert torch.equal(base, _logits(adapter, batch))


@needs_model
def test_hooks_are_removed_even_when_the_body_raises(adapter, batch):
    base = _logits(adapter, batch)
    with pytest.raises(RuntimeError):
        with edit_mlp_writes(adapter, [0], mode="zero"):
            raise RuntimeError("boom")
    assert torch.equal(base, _logits(adapter, batch))


@needs_model
def test_mean_mode_makes_the_write_constant_across_positions(adapter, batch):
    """Mean is over batch AND sequence, matching edit_heads; verify at the hook point."""
    seen = {}

    def probe(_m, _i, out):
        seen["w"] = out.detach().clone()

    # Order matters: forward hooks fire in registration order, so the probe must be
    # registered INSIDE the context to observe the ablated write rather than the raw one.
    with edit_mlp_writes(adapter, [0], mode="mean"):
        h = adapter.encoder_layers[0].output.dense.register_forward_hook(probe)
        try:
            _logits(adapter, batch)
        finally:
            h.remove()
    w = seen["w"]
    spread = (w - w.mean(dim=(0, 1), keepdim=True)).abs().max().item()
    assert spread < 1e-5, f"mean-ablated write still varies by {spread}"


@needs_model
def test_ablating_more_layers_moves_the_output_further(adapter, batch):
    base = _logits(adapter, batch)
    with edit_mlp_writes(adapter, [0], mode="zero"):
        one = (_logits(adapter, batch) - base).abs().max().item()
    with edit_mlp_writes(adapter, [0, 1, 2, 3, 4, 5], mode="zero"):
        allsix = (_logits(adapter, batch) - base).abs().max().item()
    assert allsix > one
