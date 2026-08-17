from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.circuits.hf_bert import (
    add_attention_output_delta,
    capture_head_outputs,
    edit_heads,
    patch_heads,
    project_head_context_to_residual,
)


class FakeTokenizer:
    pad_token_id = 0
    mask_token_id = 3
    unk_token_id = 99
    vocab = {"A": 4, "C": 5, "G": 6, "T": 7, "N": 8}

    def _ids(self, text):
        chars = text.split() if " " in text else list(text)
        return [1] + [self.vocab.get(x, self.unk_token_id) for x in chars] + [2]

    def __call__(self, texts, add_special_tokens=True, padding=False, truncation=False, return_tensors=None):
        one = isinstance(texts, str)
        items = [texts] if one else list(texts)
        ids = [self._ids(x) for x in items]
        max_len = max(map(len, ids))
        masks = [[1] * len(x) + [0] * (max_len - len(x)) for x in ids]
        ids = [x + [0] * (max_len - len(x)) for x in ids]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}
        result = {"input_ids": ids[0] if one else ids, "attention_mask": masks[0] if one else masks}
        return result

    def get_special_tokens_mask(self, ids, already_has_special_tokens=True):
        return [1 if x in {0, 1, 2, 3} else 0 for x in ids]

    def num_special_tokens_to_add(self, pair=False):
        return 2


class FakeSelfAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, hidden, attention_mask=None):
        b, s, _ = hidden.shape
        probs = torch.eye(s, device=hidden.device)[None, None].expand(b, 2, s, s)
        return (self.proj(hidden), probs)


class FakeAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self = FakeSelfAttention(width)


class FakeLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = FakeAttention(width)
        self.ff = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.ff.weight)

    def forward(self, hidden, attention_mask=None):
        context, probs = self.attention.self(hidden, attention_mask)
        return self.ff(context), probs


class FakeEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layer = nn.ModuleList([FakeLayer(width), FakeLayer(width)])


class FakeBert(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = FakeEncoder(width)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        width = 8
        self.config = SimpleNamespace(
            hidden_size=width,
            num_attention_heads=2,
            max_position_embeddings=64,
            to_dict=lambda: {
                "hidden_size": width,
                "num_attention_heads": 2,
                "max_position_embeddings": 64,
            },
        )
        self.embedding = nn.Embedding(100, width)
        self.bert = FakeBert(width)
        self.lm_head = nn.Linear(width, 100)

    def forward(self, input_ids, attention_mask, output_hidden_states=True, output_attentions=True, return_dict=True):
        hidden = self.embedding(input_ids)
        hidden_states = [hidden]
        attentions = []
        for layer in self.bert.encoder.layer:
            hidden, probs = layer(hidden, attention_mask)
            hidden_states.append(hidden)
            attentions.append(probs)
        return SimpleNamespace(
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            attentions=tuple(attentions) if output_attentions else None,
            logits=self.lm_head(hidden),
            last_hidden_state=None,
        )


def test_adapter_and_head_edit_patch_roundtrip():
    torch.manual_seed(0)
    adapter = SpliceBertAdapter(FakeModel(), FakeTokenizer(), tokenizer_mode="spaced", device="cpu")
    batch = adapter.tokenize(["ACGT", "TGCA"])
    clean = adapter.forward(batch, output_attentions=False).last_hidden_state

    with edit_heads(adapter, [(0, 0)], mode="zero"):
        edited = adapter.forward(batch, output_attentions=False).last_hidden_state
    assert not torch.allclose(clean, edited)
    restored = adapter.forward(batch, output_attentions=False).last_hidden_state
    assert torch.allclose(clean, restored)

    cache = capture_head_outputs(
        adapter, lambda: adapter.forward(batch, output_attentions=False)
    )
    assert cache[0].shape[:3] == (2, 2, 6)

    corrupt_ids = batch.input_ids.clone()
    corrupt_ids[:, 2] = FakeTokenizer.vocab["N"]
    corrupt_batch = type(batch)(
        input_ids=corrupt_ids,
        attention_mask=batch.attention_mask,
        prefix_tokens=batch.prefix_tokens,
        nucleotide_lengths=batch.nucleotide_lengths,
        texts=batch.texts,
    )
    corrupt = adapter.forward(corrupt_batch, output_attentions=False).last_hidden_state
    with patch_heads(adapter, cache, [(0, 0)]):
        patched = adapter.forward(corrupt_batch, output_attentions=False).last_hidden_state
    assert not torch.allclose(corrupt, patched)


def test_residual_layer_capture_and_patch():
    from gfmcd.circuits.hf_bert_residual import capture_layer_outputs, patch_layer_output

    torch.manual_seed(1)
    adapter = SpliceBertAdapter(FakeModel(), FakeTokenizer(), tokenizer_mode="spaced", device="cpu")
    clean_batch = adapter.tokenize(["ACGT", "TGCA"])
    cache = capture_layer_outputs(
        adapter, lambda: adapter.forward(clean_batch, output_attentions=False)
    )
    assert set(cache) == {0, 1}

    corrupt_ids = clean_batch.input_ids.clone()
    corrupt_ids[:, 2] = FakeTokenizer.vocab["N"]
    corrupt_batch = type(clean_batch)(
        input_ids=corrupt_ids,
        attention_mask=clean_batch.attention_mask,
        prefix_tokens=clean_batch.prefix_tokens,
        nucleotide_lengths=clean_batch.nucleotide_lengths,
        texts=clean_batch.texts,
    )
    corrupt = adapter.forward(corrupt_batch, output_attentions=False).last_hidden_state
    with patch_layer_output(adapter, cache, 0):
        patched = adapter.forward(corrupt_batch, output_attentions=False).last_hidden_state
    assert not torch.allclose(corrupt, patched)


def test_output_projected_head_bridge_and_query_local_delta_hook():
    class OutputAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.output = SimpleNamespace(dense=nn.Linear(4, 4, bias=True))

    projection = OutputAttention().output.dense
    with torch.no_grad():
        projection.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10)
        projection.bias.copy_(torch.tensor([7.0, 8.0, 9.0, 10.0]))
    adapter = SimpleNamespace(
        num_layers=1,
        num_heads=2,
        encoder_layers=(SimpleNamespace(attention=SimpleNamespace(output=SimpleNamespace(dense=projection))),),
    )
    context = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    contribution = project_head_context_to_residual(adapter, (0, 1), context)
    expected = torch.nn.functional.linear(context, projection.weight[:, 2:4], bias=None)
    assert torch.equal(contribution, expected)
    assert not torch.equal(contribution, expected + projection.bias)

    merged = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    baseline = projection(merged)
    mask = torch.tensor([[False, True, False], [False, False, True]])
    delta = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-4.0, -3.0, -2.0, -1.0]])
    with add_attention_output_delta(adapter, 0, delta, token_mask=mask):
        patched = projection(merged)
    expected_patched = baseline.clone()
    expected_patched[0, 1] += delta[0]
    expected_patched[1, 2] += delta[1]
    assert torch.equal(patched, expected_patched)
    assert torch.equal(projection(merged), baseline)
    with pytest.raises(ValueError, match="exactly one token"):
        with add_attention_output_delta(
            adapter, 0, delta, token_mask=torch.ones((2, 3), dtype=torch.bool)
        ):
            pass
