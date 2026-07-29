from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gfmcd.splicing.gtex_finetune import (
    GtexCheckpointError,
    checkpoint_seed,
    state_dict_sha256,
    validate_gtex_encoder,
    validate_gtex_token_classifier_state,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(10, 4)
        self.proj = nn.Linear(4, 4)


class TinyMlm(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = TinyEncoder()
        self.config = SimpleNamespace(hidden_size=4)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = TinyEncoder()
        self.classifier = nn.Linear(4, 3)


def published_state(model: TinyMlm) -> OrderedDict[str, torch.Tensor]:
    state = OrderedDict(
        (f"bert.{name}", value.detach().clone() + 0.25)
        for name, value in model.bert.state_dict().items()
    )
    state["classifier.weight"] = torch.zeros(3, 4)
    state["classifier.bias"] = torch.zeros(3)
    return state


def test_checkpoint_seed_is_explicit_and_bounded():
    assert checkpoint_seed("SpliceBERT.SpliceAI-GTEx.2020.pt") == 2020
    assert checkpoint_seed(Path("x/seed-2024.pt")) == 2024
    assert checkpoint_seed("checkpoint.17.pt") is None


def test_validate_requires_exact_encoder_and_classifier_contract(tmp_path: Path):
    model = TinyMlm()
    state = published_state(model)
    checkpoint = tmp_path / "SpliceBERT.SpliceAI-GTEx.2020.pt"
    torch.save(state, checkpoint)

    report = validate_gtex_encoder(model, state, checkpoint_path=checkpoint)
    assert report.seed == 2020
    assert report.encoder_tensor_count == len(model.bert.state_dict())
    assert report.encoder_parameter_count == sum(
        tensor.numel() for tensor in model.bert.state_dict().values()
    )
    assert all(shift.delta_l2 > 0 for shift in report.tensor_shifts)
    assert not report.missing_encoder_keys
    assert not report.extra_encoder_keys
    assert not report.extra_nonencoder_keys


@pytest.mark.parametrize("failure", ["missing", "extra", "shape", "nonfinite", "classifier"])
def test_validate_rejects_partial_or_ambiguous_transplants(tmp_path: Path, failure: str):
    model = TinyMlm()
    state = published_state(model)
    if failure == "missing":
        state.pop("bert.proj.bias")
    elif failure == "extra":
        state["bert.surprise"] = torch.zeros(1)
    elif failure == "shape":
        state["bert.proj.bias"] = torch.zeros(5)
    elif failure == "nonfinite":
        state["bert.proj.bias"][0] = float("nan")
    elif failure == "classifier":
        state["pooler.weight"] = torch.zeros(4, 4)
    checkpoint = tmp_path / "SpliceBERT.SpliceAI-GTEx.2021.pt"
    torch.save(state, checkpoint)

    with pytest.raises(GtexCheckpointError):
        validate_gtex_encoder(model, state, checkpoint_path=checkpoint)


def test_state_dict_digest_is_order_independent_and_content_sensitive():
    a = OrderedDict([("x", torch.tensor([1.0])), ("y", torch.tensor([2.0]))])
    b = OrderedDict(reversed(list(a.items())))
    c = OrderedDict([("x", torch.tensor([1.0])), ("y", torch.tensor([3.0]))])
    assert state_dict_sha256(a) == state_dict_sha256(b)
    assert state_dict_sha256(a) != state_dict_sha256(c)


def test_native_classifier_requires_exact_finite_state(tmp_path: Path):
    model = TinyClassifier()
    checkpoint = tmp_path / "SpliceBERT.SpliceAI-GTEx.2022.pt"
    state = model.state_dict()
    torch.save(state, checkpoint)
    report = validate_gtex_token_classifier_state(
        model,
        state,
        checkpoint_path=checkpoint,
    )
    assert report.seed == 2022
    assert report.class_labels == ("no_splice_site", "donor", "acceptor")
    assert report.tensor_count == len(state)

    missing = OrderedDict(state)
    missing.pop("classifier.bias")
    with pytest.raises(GtexCheckpointError):
        validate_gtex_token_classifier_state(
            model,
            missing,
            checkpoint_path=checkpoint,
        )

    nonfinite = OrderedDict(
        (name, tensor.detach().clone()) for name, tensor in state.items()
    )
    nonfinite["classifier.bias"][0] = float("nan")
    with pytest.raises(GtexCheckpointError):
        validate_gtex_token_classifier_state(
            model,
            nonfinite,
            checkpoint_path=checkpoint,
        )
