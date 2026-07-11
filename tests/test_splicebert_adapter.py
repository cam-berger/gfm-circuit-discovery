"""SpliceBERT adapter smoke tests (TO_FIX.md §19).

Marked `network` (downloads weights) so PR CI skips them. No circuit claim is licensed until these
pass: token alignment, shapes, the context-length guard, self-restoring localized ablation, and
determinism. Run: uv run --extra models pytest -m network tests/test_splicebert_adapter.py
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.network

transformers = pytest.importorskip("transformers")

from gfmcd.config import ContaminationError  # noqa: E402


@pytest.fixture(scope="module")
def adapter():
    from gfmcd.models.adapters import SpliceBERTAdapter

    try:
        return SpliceBERTAdapter(device="cpu")
    except Exception as e:  # network / weights unavailable
        pytest.skip(f"SpliceBERT weights unavailable: {e}")


def test_architecture_facts(adapter):
    assert (adapter.n_layers, adapter.n_heads, adapter.d_model) == (6, 16, 512)
    assert adapter.d_head == 32


def test_token_alignment(adapter):
    enc = adapter.encode("ACGTN")
    ids = enc["input_ids"][0].tolist()
    assert len(ids) == 7  # [CLS] A C G T N [SEP]
    assert enc["nt_to_token"] == [1, 2, 3, 4, 5]
    # nucleotides map to the SpliceBERT vocab ids
    assert ids[1:6] == [6, 7, 8, 9, 5]  # A C G T N


def test_context_length_guard(adapter):
    with pytest.raises(ValueError):
        adapter.encode("A" * (adapter.max_nt + 1))


def test_cache_shapes(adapter):
    enc = adapter.encode("ACGT" * 10)
    logits, cache = adapter.run_with_cache(enc["input_ids"])
    seq = enc["input_ids"].shape[1]
    assert logits.shape[0] == 1 and logits.shape[1] == seq
    assert f"resid.{adapter.n_layers}" in cache  # embeddings + n_layers hidden states
    assert cache["L0.attn.probs"].shape[1] == adapter.n_heads
    assert adapter.head_context(cache, 0).shape == (1, seq, adapter.n_heads, adapter.d_head)


def test_determinism_in_eval(adapter):
    enc = adapter.encode("ACGTACGTAC")
    a = adapter.mlm_logits(enc["input_ids"])
    b = adapter.mlm_logits(enc["input_ids"])
    assert torch.equal(a, b)


def test_head_ablation_is_localized_and_self_restoring(adapter):
    enc = adapter.encode("ACGTACGTACGT")
    base = adapter.mlm_logits(enc["input_ids"])
    with adapter.ablate_head(layer=3, head=5):
        ablated = adapter.mlm_logits(enc["input_ids"])
    restored = adapter.mlm_logits(enc["input_ids"])
    assert not torch.equal(ablated, base), "ablating a head should change the logits"
    assert torch.equal(restored, base), "the intervention must self-restore on exit"


def test_firewall_blocks_generalisation_claims(adapter):
    with pytest.raises(ContaminationError):
        adapter.assert_mechanistic_only()
