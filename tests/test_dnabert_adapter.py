from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gfmcd.adapters.dnabert import (
    DnaBertAdapter,
    DnaTokenizedSequences,
    normalize_sequence,
)


class FakeFastTokenizer:
    is_fast = True
    model_max_length = 512

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_tensors=None,
        return_offsets_mapping=False,
        return_special_tokens_mask=False,
    ):
        one = isinstance(texts, str)
        items = [texts] if one else list(texts)

        rows = []
        masks = []
        offsets = []
        specials = []

        for text in items:
            # Variable-length fake tokenization:
            # first nucleotide, then remaining sequence.
            token_ids = [1, 10, 11, 2]
            token_offsets = [(0, 0), (0, 1), (1, len(text)), (0, 0)]
            special_mask = [1, 0, 0, 1]

            rows.append(token_ids)
            masks.append([1] * len(token_ids))
            offsets.append(token_offsets)
            specials.append(special_mask)

        result = {
            "input_ids": rows[0] if one else rows,
            "attention_mask": masks[0] if one else masks,
        }

        if return_offsets_mapping:
            result["offset_mapping"] = offsets[0] if one else offsets

        if return_special_tokens_mask:
            result["special_tokens_mask"] = specials[0] if one else specials

        if return_tensors == "pt":
            result = {
                key: torch.tensor(value)
                for key, value in result.items()
            }

        return result


class FakeSelfAttention(nn.Module):
    def forward(self, hidden, *args, **kwargs):
        return hidden


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = FakeSelfAttention()


class FakeMlp(nn.Module):
    def forward(self, hidden):
        return hidden


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FakeAttention()
        self.mlp = FakeMlp()


class FakeEncoder(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layer = nn.ModuleList(
            [FakeLayer() for _ in range(num_layers)]
        )


class FakeDnaBert(nn.Module):
    def __init__(self, hidden_size: int = 8, num_layers: int = 3):
        super().__init__()

        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=2,
            max_position_embeddings=512,
            to_dict=lambda: {
                "hidden_size": hidden_size,
                "num_attention_heads": 2,
                "max_position_embeddings": 512,
            },
        )

        self.encoder = FakeEncoder(num_layers)
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(
        self,
        input_ids,
        attention_mask,
        output_all_encoded_layers=False,
    ):
        hidden = self.embedding(input_ids)
        pooled = None

        if not output_all_encoded_layers:
            return hidden, pooled

        unpadded = hidden[attention_mask.bool()]
        layers = [
            unpadded + float(index)
            for index in range(len(self.encoder.layer))
        ]

        return layers, pooled


@pytest.fixture
def adapter() -> DnaBertAdapter:
    return DnaBertAdapter(
        FakeDnaBert(),
        FakeFastTokenizer(),
        device="cpu",
    )


def test_normalize_sequence():
    assert normalize_sequence("acgu n") == "ACGTN"

    with pytest.raises(ValueError):
        normalize_sequence("")

    with pytest.raises(ValueError):
        normalize_sequence("ACGTX")


def test_tokenization_offsets_and_spans(adapter):
    batch = adapter.tokenize(["ACGT", "TGCA"])

    assert isinstance(batch, DnaTokenizedSequences)
    assert tuple(batch.input_ids.shape) == (2, 4)
    assert batch.prefix_tokens == 1

    assert batch.token_to_nucleotide_spans[0] == [
        None,
        (0, 1),
        (1, 4),
        None,
    ]

    assert adapter.tokens_covering_nucleotide(batch, 0, 0) == [1]
    assert adapter.tokens_covering_nucleotide(batch, 0, 2) == [2]
    assert adapter.token_indices_for_span(batch, 0, 1, 4) == [2]


def test_biological_token_mask(adapter):
    batch = adapter.tokenize(["ACGT"])

    mask = adapter.biological_token_mask(batch)

    assert mask.dtype == torch.bool
    assert mask.tolist() == [[False, True, True, False]]


def test_forward_repads_all_layers(adapter):
    batch = adapter.tokenize(["ACGT", "TGCA"])

    output = adapter.forward(
        batch,
        output_hidden_states=True,
        output_attentions=False,
    )

    assert output.hidden_states is not None
    assert len(output.hidden_states) == 3

    for hidden in output.hidden_states:
        assert tuple(hidden.shape) == (2, 4, 8)

    assert torch.equal(
        output.last_hidden_state,
        output.hidden_states[-1],
    )


def test_forward_final_layer_only(adapter):
    batch = adapter.tokenize(["ACGT"])

    output = adapter.forward(
        batch,
        output_hidden_states=False,
        output_attentions=False,
    )

    assert output.hidden_states is None
    assert tuple(output.last_hidden_state.shape) == (1, 4, 8)


def test_masked_mean_excludes_special_tokens(adapter):
    batch = adapter.tokenize(["ACGT"])

    output = adapter.forward(
        batch,
        output_hidden_states=True,
        output_attentions=False,
    )

    pooled = adapter.masked_mean(output, batch)

    expected = output.last_hidden_state[:, 1:3].mean(dim=1)

    assert tuple(pooled.shape) == (1, 8)
    assert torch.allclose(pooled, expected)


def test_pooler_disabled(adapter):
    assert adapter.model.pooler is None
    assert adapter.pooler_disabled is True


def test_architecture_properties(adapter):
    assert adapter.hidden_size == 8
    assert adapter.num_heads == 2
    assert adapter.head_dim == 4
    assert adapter.num_layers == 3
    assert len(adapter.encoder_layers) == 3
    assert len(adapter.attention_self_modules) == 3
    assert len(adapter.mlp_modules) == 3


def test_attention_outputs_not_supported(adapter):
    batch = adapter.tokenize(["ACGT"])

    with pytest.raises(NotImplementedError):
        adapter.forward(
            batch,
            output_hidden_states=True,
            output_attentions=True,
        )
