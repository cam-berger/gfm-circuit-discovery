from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from gfmcd.adapters.dnabert2.loader import PinnedEnvironmentError
from gfmcd.adapters.dnabert2.promoter_classifier import (
    Dnabert2PromoterBundle,
    Dnabert2PromoterClassifier,
    PromoterCheckpoint,
    verify_tokenizer_equivalence,
)


def _tokenizer_document(*, list_merges: bool = False) -> dict:
    merges = [["A", "A"], ["AA", "C"]] if list_merges else ["A A", "AA C"]
    return {
        "model": {"type": "BPE", "vocab": {"A": 0, "AA": 1}, "merges": merges},
        "added_tokens": [{"id": 0, "content": "[UNK]"}],
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
    }


def test_tokenizer_equivalence_accepts_new_merge_serialization(tmp_path) -> None:
    base = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    base.write_text(json.dumps(_tokenizer_document()))
    candidate.write_text(json.dumps(_tokenizer_document(list_merges=True)))
    audit = verify_tokenizer_equivalence(base, candidate)
    assert audit["equivalent"]
    assert audit["merge_count"] == 2


def test_tokenizer_equivalence_rejects_semantic_drift(tmp_path) -> None:
    base = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    document = _tokenizer_document(list_merges=True)
    document["model"]["vocab"]["C"] = 2
    base.write_text(json.dumps(_tokenizer_document()))
    candidate.write_text(json.dumps(document))
    with pytest.raises(PinnedEnvironmentError, match="vocabulary"):
        verify_tokenizer_equivalence(base, candidate)


class _FakeTokenizer:
    def __call__(self, sequences, **_kwargs):
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.zeros((len(sequences), width), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            values = [1 if base in "GC" else 0 for base in sequence]
            input_ids[row, : len(values)] = torch.tensor(values)
            attention[row, : len(values)] = 1
        return {"input_ids": input_ids, "attention_mask": attention}


class _FakeModel:
    def __call__(self, input_ids, attention_mask):
        promoter = (input_ids * attention_mask).sum(dim=1).float()
        logits = torch.stack((-promoter, promoter), dim=1)
        return SimpleNamespace(logits=logits)


def _fake_bundle() -> Dnabert2PromoterBundle:
    checkpoint = PromoterCheckpoint(
        name="fake",
        hf_id="owner/fake",
        revision="a" * 40,
        input_length_nt=4,
        tss_upstream_nt=2,
        tss_downstream_nt=1,
        upstream_sha256={},
    )
    return Dnabert2PromoterBundle(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        config=SimpleNamespace(),
        checkpoint=checkpoint,
        snapshot=None,  # type: ignore[arg-type]
        device="cpu",
        provenance={"classification_head_complete": True},
    )


def test_promoter_scorer_returns_label_logit_margin() -> None:
    scorer = Dnabert2PromoterClassifier(_fake_bundle(), maximum_tokens=4)
    scores = scorer.score([("a", "AAAA"), ("b", "GGAA")], batch_size=2)
    assert scores[0].margin == 0.0
    assert scores[0].predicted_label == 0
    assert scores[1].margin == 4.0
    assert scores[1].predicted_label == 1


def test_promoter_scorer_rejects_wrong_length_non_dna_and_token_overflow() -> None:
    scorer = Dnabert2PromoterClassifier(_fake_bundle(), maximum_tokens=4)
    with pytest.raises(ValueError, match="wrong nucleotide length"):
        scorer.score([("short", "AAA")])
    with pytest.raises(ValueError, match="non-ACGT"):
        scorer.score([("ambiguous", "AAAN")])
    with pytest.raises(RuntimeError, match="token"):
        Dnabert2PromoterClassifier(_fake_bundle(), maximum_tokens=3).score(
            [("long", "AAAA")]
        )
