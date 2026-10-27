"""gfmcd.models: verification before loading, eager attention, identity, and registry provenance.

A tiny BERT masked-LM and character tokenizer are saved to a temp directory, so the real
``from_pretrained`` path runs in CI without SpliceBERT weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gfmcd import models, record
from gfmcd.models import REGISTRY, ModelSpec, locate, open_model

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def tiny(tmp_path: Path, monkeypatch) -> ModelSpec:
    from transformers import BertConfig, BertForMaskedLM, BertTokenizer

    directory = tmp_path / "roots/models/Tiny.64nt"
    directory.mkdir(parents=True)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "A", "C", "G", "T", "N"]) + "\n")
    BertTokenizer(str(vocab), do_lower_case=False).save_pretrained(directory)
    config = BertConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=16, max_position_embeddings=66)
    BertForMaskedLM(config).save_pretrained(directory)
    monkeypatch.setenv("GFMCD_DATA_ROOTS", str(tmp_path / "roots"))
    files = {p.name: models._sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
    return ModelSpec(name="Tiny.64nt", family="splicebert", source="data/models/Tiny.64nt", files=files)


def test_open_model_verifies_loads_and_reports_identity(tiny: ModelSpec) -> None:
    handle = open_model(tiny, device="cpu")
    assert type(handle.adapter).__name__ == "SpliceBertAdapter"
    assert handle.adapter.prefix_tokens == 1
    assert handle.identity == {"name": "Tiny.64nt", "revision": None, "files": dict(tiny.files)}
    assert handle.path.name == "Tiny.64nt" and not handle.attention


def test_a_mismatched_file_refuses_before_anything_loads(tiny: ModelSpec, monkeypatch) -> None:
    from gfmcd.adapters import splicebert

    def must_not_load(*args, **kwargs):
        raise AssertionError("loaded a checkpoint that failed verification")

    monkeypatch.setattr(splicebert.SpliceBertAdapter, "from_pretrained", must_not_load)
    tampered = ModelSpec(**{**tiny.__dict__, "files": {**tiny.files, "config.json": "0" * 64}})
    with pytest.raises(FileNotFoundError, match="config.json"):
        open_model(tampered, device="cpu")


def test_attention_flag_loads_eager_attention(tiny: ModelSpec) -> None:
    handle = open_model(tiny, device="cpu", attention=True)
    assert handle.adapter.model.config._attn_implementation == "eager" and handle.attention


def test_unlocatable_checkpoint_names_the_remedy(monkeypatch) -> None:
    monkeypatch.setenv("GFMCD_DATA_ROOTS", "")
    spec = ModelSpec(name="absent", family="splicebert", source="data/models/absent", files={"x": "0" * 64})
    with pytest.raises(FileNotFoundError, match="GFMCD_DATA_ROOTS"):
        locate(spec)


def test_identity_is_what_record_write_accepts(tiny: ModelSpec, tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    handle = open_model(tiny, device="cpu")
    written = record.write(repo / "r.json", {"x": 1}, experiment="e", repo=repo, model=handle.identity)
    assert json.loads(written.manifest.read_text())["model"] == handle.identity


def test_specs_refuse_unknown_conventions_and_unpinned_hub_models() -> None:
    with pytest.raises(ValueError):
        ModelSpec(name="x", family="splicebert", source="s", files={"a": "0" * 64}, layer_convention="hf")
    with pytest.raises(ValueError, match="pinned revision"):
        ModelSpec(name="x", family="dnabert2", source="org/model")


def _pinned_values(document) -> set[tuple[str, str]]:
    """(filename, sha256) pairs anywhere in a committed artifact: file maps and path records."""
    found: set[tuple[str, str]] = set()

    def walk(value) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                found.add((value["path"].rsplit("/", 1)[-1], value["sha256"]))
            for key, item in value.items():
                if isinstance(item, str) and len(item) == 64:
                    found.add((key.rsplit("/", 1)[-1], item))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(document)
    return found


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_registry_values_are_the_ones_committed_artifacts_record(name: str) -> None:
    spec = REGISTRY[name]
    assert spec.cites, f"{name} cites no artifact"
    for cited in spec.cites:
        document = json.loads((REPO / cited).read_text())
        pinned = _pinned_values(document)
        for filename, sha in spec.files.items():
            assert (filename, sha) in pinned, f"{name}: {filename} {sha[:12]} not in {cited}"
        if spec.revision:
            assert spec.revision in json.dumps(document), f"{name}: revision not in {cited}"


@pytest.mark.skipif(not (REPO / models.SPLICEBERT_1024.source / "pytorch_model.bin").exists(),
                    reason="SpliceBERT weights not present")
def test_the_real_checkpoint_verifies() -> None:
    assert locate(models.SPLICEBERT_1024) == REPO / models.SPLICEBERT_1024.source
