import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert_gtex_finetune_shift/run_gtex_native_representation_bridge.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_gtex_native_representation_bridge",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTokenizerOutput(dict):
    def to(self, _device):
        return self


class FakeTask:
    batch_size = 2
    prefix_tokens = 1

    def encode(self, sequences):
        return FakeTokenizerOutput(
            input_ids=torch.zeros((len(sequences), 6), dtype=torch.long)
        )


class FakeBert(nn.Module):
    def forward(self, input_ids, output_hidden_states, return_dict):
        batch, length = input_ids.shape
        first = torch.arange(
            batch * length * 3,
            dtype=torch.float32,
        ).reshape(batch, length, 3)
        second = first + 2
        return SimpleNamespace(hidden_states=(first, second))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = FakeBert()
        self.dropout = nn.Identity()
        self.classifier = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(3))


def test_project_all_depths_returns_matched_margins_and_states():
    margins, states = MODULE.project_all_depths(
        FakeModel(),
        FakeTask(),
        ["AAAA", "CCCC", "GGGG"],
        [1, 2, 1],
        retain_states=True,
    )
    assert margins.shape == (3, 2)
    assert states is not None
    assert len(states) == 2
    assert states[0].shape == (3, 3)
    # Adding the same scalar to all three hidden coordinates does not change
    # class-1 minus class-0 under the identity classifier.
    assert np.allclose(margins[:, 0], margins[:, 1])
