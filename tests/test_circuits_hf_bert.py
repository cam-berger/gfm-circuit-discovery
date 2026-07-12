"""G1 instrument tests: verify the head-intervention primitives on the REAL
SpliceBERT module structure before any circuit is traced.

Design:
- Synthetic DNA + a hidden-state score_fn (no trained readout / checkpoint), so
  these test the *mechanics* of ablation/patching, not the science.
- CPU + short sequences + tiny batch for determinism and speed.
- The load-bearing correctness invariant is the capture->patch round-trip
  identity: patching a clean cache back into a clean run reproduces the score
  exactly. (Patching only per-head attention context does NOT recover clean from
  corrupt, because the residual/MLP path still carries corrupt input -- so we do
  not assert that.)
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.circuits.hf_bert import (
    capture_head_outputs,
    edit_heads,
    head_ablation_scan,
    matched_random_head_sets,
    patch_heads,
)
from gfmcd.splicing.config import load_config

REPO = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO / load_config(REPO / "configs/splicebert_partner_selection/base.json")["model"]["path"]

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="SpliceBERT weights not present"
)


def _seqs(n: int, length: int, seed: int) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    bases = "ACGT"
    idx = torch.randint(0, 4, (n, length), generator=g)
    return ["".join(bases[i] for i in row.tolist()) for row in idx]


@pytest.fixture(scope="module")
def adapter():
    return SpliceBertAdapter.from_pretrained(
        MODEL_PATH, tokenizer_mode="auto", device="cpu",
        trust_remote_code=False, local_files_only=True, model_class="masked_lm",
    )


@pytest.fixture(scope="module")
def clean_tok(adapter):
    return adapter.tokenize(_seqs(4, 120, seed=1))


@pytest.fixture(scope="module")
def corrupt_tok(adapter):
    return adapter.tokenize(_seqs(4, 120, seed=2))  # same shape, different content


_PROJ: torch.Tensor | None = None


def _score(adapter, tok):
    """Deterministic, head-sensitive scalar: fixed random projection of an interior
    token's final hidden vector. (A plain hidden.mean() is ~invariant because the
    final states are LayerNorm'd, which masks head interventions.)"""
    global _PROJ
    out = adapter.forward(tok, output_hidden_states=True, output_attentions=False, grad=False)
    h = out.hidden_states[-1]  # [B, S, D]
    if _PROJ is None:
        _PROJ = torch.randn(h.shape[-1], generator=torch.Generator().manual_seed(0))
    return float((h[:, 5, :] @ _PROJ.to(h.dtype)).mean())


def _all_heads(adapter):
    return [(l, h) for l in range(adapter.num_layers) for h in range(adapter.num_heads)]


def test_empty_edit_is_noop(adapter, clean_tok):
    base = _score(adapter, clean_tok)
    with edit_heads(adapter, [], mode="zero"):
        assert _score(adapter, clean_tok) == base


def test_hooks_removed_after_context(adapter, clean_tok):
    base = _score(adapter, clean_tok)
    with edit_heads(adapter, [(0, 0)], mode="zero"):
        pass
    assert _score(adapter, clean_tok) == base  # no lingering hooks


def test_zero_ablate_all_heads_changes_score(adapter, clean_tok):
    base = _score(adapter, clean_tok)
    with edit_heads(adapter, _all_heads(adapter), mode="zero"):
        intervened = _score(adapter, clean_tok)
    assert abs(intervened - base) > 1e-2


def test_capture_patch_roundtrip_identity(adapter, clean_tok):
    """Patching a clean cache back into the same clean run is the identity."""
    base = _score(adapter, clean_tok)
    cache = capture_head_outputs(adapter, lambda: _score(adapter, clean_tok))
    with patch_heads(adapter, cache, _all_heads(adapter), token_mask=None):
        patched = _score(adapter, clean_tok)
    assert abs(patched - base) < 1e-3  # identity up to fp; real effects are O(0.1-1)


def test_patch_false_mask_is_noop_true_mask_is_full(adapter, clean_tok, corrupt_tok):
    cache = capture_head_outputs(adapter, lambda: _score(adapter, clean_tok))
    corrupt = _score(adapter, corrupt_tok)
    S = corrupt_tok.input_ids.shape[1]
    heads = _all_heads(adapter)
    false_mask = torch.zeros((4, S), dtype=torch.bool)
    true_mask = torch.ones((4, S), dtype=torch.bool)
    with patch_heads(adapter, cache, heads, token_mask=false_mask):
        assert abs(_score(adapter, corrupt_tok) - corrupt) < 1e-6  # no-op
    with patch_heads(adapter, cache, heads, token_mask=true_mask):
        masked_full = _score(adapter, corrupt_tok)
    with patch_heads(adapter, cache, heads, token_mask=None):
        unmasked_full = _score(adapter, corrupt_tok)
    assert abs(masked_full - unmasked_full) < 1e-6  # all-True == no mask


def test_patch_moves_corrupt_toward_clean(adapter, clean_tok, corrupt_tok):
    cache = capture_head_outputs(adapter, lambda: _score(adapter, clean_tok))
    corrupt = _score(adapter, corrupt_tok)
    with patch_heads(adapter, cache, _all_heads(adapter), token_mask=None):
        patched = _score(adapter, corrupt_tok)
    assert abs(patched - corrupt) > 1e-2  # patching all heads changes the corrupt score


def test_ablation_modes_produce_effects(adapter, clean_tok):
    zero = head_ablation_scan(adapter, lambda: _score(adapter, clean_tok), mode="zero")
    mean = head_ablation_scan(adapter, lambda: _score(adapter, clean_tok), mode="mean")
    assert len(zero) == adapter.num_layers * adapter.num_heads
    # sorted by effect descending
    assert all(zero[i].effect >= zero[i + 1].effect for i in range(len(zero) - 1))
    # at least one head has a non-trivial effect, and zero vs mean differ somewhere
    assert max(abs(e.effect) for e in zero) > 1e-3
    zmap = {(e.layer, e.head): e.effect for e in zero}
    mmap = {(e.layer, e.head): e.effect for e in mean}
    assert any(abs(zmap[k] - mmap[k]) > 1e-6 for k in zmap)


def test_resample_requires_batch_two(adapter):
    tok1 = adapter.tokenize(_seqs(1, 120, seed=3))
    with pytest.raises(ValueError):
        with edit_heads(adapter, [(0, 0)], mode="resample"):
            _score(adapter, tok1)


def test_matched_random_sets(adapter):
    selected = [(0, 0), (1, 1), (2, 2)]
    sets = matched_random_head_sets(adapter, selected, n=50, seed=0, exclude_selected=True)
    assert len(sets) == 50
    assert all(len(s) == len(selected) for s in sets)
    flat = {h for s in sets for h in s}
    assert not (set(selected) & flat)  # excluded
    assert all(0 <= l < adapter.num_layers and 0 <= h < adapter.num_heads for (l, h) in flat)
    # reproducible with the same seed
    again = matched_random_head_sets(adapter, selected, n=50, seed=0, exclude_selected=True)
    assert sets == again
