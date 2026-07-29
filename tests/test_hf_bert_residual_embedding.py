from types import SimpleNamespace

import torch
from torch import nn

from gfmcd.circuits.hf_bert_residual import (
    capture_embedding_output,
    patch_embedding_output,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Module()
        self.bert.embeddings = nn.Embedding(8, 3)
        with torch.no_grad():
            self.bert.embeddings.weight.copy_(
                torch.arange(24, dtype=torch.float32).reshape(8, 3)
            )

    def forward(self, input_ids):
        return self.bert.embeddings(input_ids)


def test_capture_and_masked_embedding_patch():
    model = TinyModel()
    adapter = SimpleNamespace(model=model)
    clean_ids = torch.tensor([[1, 2, 3, 4]])
    corrupt_ids = torch.tensor([[1, 6, 7, 4]])
    clean = capture_embedding_output(adapter, lambda: model(clean_ids))
    corrupt = model(corrupt_ids)
    mask = torch.tensor([[False, True, True, False]])

    with patch_embedding_output(adapter, clean, token_mask=mask):
        patched = model(corrupt_ids)

    assert torch.equal(patched, clean)
    assert not torch.equal(corrupt, clean)


def test_embedding_patch_false_mask_is_noop_and_handle_is_removed():
    model = TinyModel()
    adapter = SimpleNamespace(model=model)
    ids = torch.tensor([[1, 2, 3]])
    source = model(torch.tensor([[4, 5, 6]])).detach()
    false_mask = torch.zeros((1, 3), dtype=torch.bool)

    with patch_embedding_output(adapter, source, token_mask=false_mask):
        assert torch.equal(model(ids), model.bert.embeddings(ids))

    assert torch.equal(model(ids), model.bert.embeddings(ids))
