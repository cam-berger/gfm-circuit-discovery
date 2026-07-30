"""Pinned DNABERT-2 promoter classifiers and their model-native margin.

The public MagicsLab checkpoints were serialized with a newer ``tokenizers``
version than the atlas runtime.  Their BPE vocabulary and merge table are
semantically identical to the already-pinned DNABERT-2 tokenizer, so this
loader verifies that equivalence and deliberately uses the original tokenizer.
It never executes Hub-hosted Python code.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loader import (
    DNABERT2_REVISION,
    PinnedEnvironmentError,
    _verify_vendor,
    require_pinned_env,
    resolve_snapshot,
    sha256_file,
)

PROMOTER_GUE_DATASET_ID = "leannmlindsey/GUE"
PROMOTER_GUE_DATASET_REVISION = "240df51a2cd2c0fb1d88efffaf86e8d1e477f932"


@dataclass(frozen=True)
class PromoterCheckpoint:
    """Immutable identity and task geometry for one classifier candidate."""

    name: str
    hf_id: str
    revision: str
    input_length_nt: int
    tss_upstream_nt: int
    tss_downstream_nt: int
    upstream_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.input_length_nt != self.tss_upstream_nt + self.tss_downstream_nt + 1:
            raise ValueError(
                "promoter checkpoint TSS geometry does not match its input length"
            )
        object.__setattr__(self, "upstream_sha256", dict(self.upstream_sha256))


PROMOTER_CHECKPOINTS: Mapping[str, PromoterCheckpoint] = {
    "prom_core_all": PromoterCheckpoint(
        name="prom_core_all",
        hf_id="magicslabnu/DNABERT-2-finetuned-prom_core_all",
        revision="39327fe9118ec506899dabd41523bac65012afec",
        input_length_nt=70,
        tss_upstream_nt=34,
        tss_downstream_nt=35,
        upstream_sha256={
            "config.json": "1cfbcdd87f34b4162e10827b89ede88b64272f3a37de6a51998855441987ba27",
            "model.safetensors": "ae0736749cd7efbc5f2df76337ecd6377f8d5bea0584eca660c2f7ae3f116760",
            "special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
            "tokenizer.json": "1b805880bea234d909b35050a412f5f2db91e700f5cfcd16a4b573bfcbc22466",
            "tokenizer_config.json": "ce3e974680079d483cb4dcd66a2000dc331414c21b7b24b0b6b4a690e192a3db",
            "trainer_state.json": "a4619a5a079ea3ff93f18f65307ae28e52cb1afb6fb22e70cb02d2c921e04e68",
            "training_args.bin": "afdfa081135fe03544ccc9cf9f3b5df9c1baea98bd4f13d0dcd766016ae77618",
        },
    ),
    "prom_300_all": PromoterCheckpoint(
        name="prom_300_all",
        hf_id="magicslabnu/DNABERT-2-finetuned-prom_300_all",
        revision="7f2a5f5b8e7e6fffb2dcb396a77d1bc78b7f5f12",
        input_length_nt=300,
        tss_upstream_nt=249,
        tss_downstream_nt=50,
        upstream_sha256={
            "config.json": "9f5c0949193cf3ff81691369610c8fcca8c082fce66aa1ff13dd4b06fb955637",
            "model.safetensors": "ce09ef08160640ac62b68b2995b8ea86c0118d920ee74ba449a56df37a3b4e82",
            "special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
            "tokenizer.json": "1b805880bea234d909b35050a412f5f2db91e700f5cfcd16a4b573bfcbc22466",
            "tokenizer_config.json": "ce3e974680079d483cb4dcd66a2000dc331414c21b7b24b0b6b4a690e192a3db",
            "trainer_state.json": "6ec01510af14b779950af3be4f9582652a18b9c3002d328a22f1c1753bac3e86",
            "training_args.bin": "f77ae2e3c1ef76aa2812d2e52a52ee2a88cc8e89d81280216628021c62b28960",
        },
    ),
}


def resolve_promoter_snapshot(checkpoint: PromoterCheckpoint) -> Path:
    """Locate an exact classifier revision in the local Hub cache; never download."""

    owner, name = checkpoint.hf_id.split("/", 1)
    pattern = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{owner}--{name}/snapshots/"
        f"{checkpoint.revision}"
    )
    hits = glob.glob(pattern)
    if not hits:
        raise FileNotFoundError(
            f"promoter checkpoint {checkpoint.hf_id}@{checkpoint.revision} is absent "
            "from the local Hugging Face cache"
        )
    return Path(hits[0])


def _normalized_merges(document: Mapping[str, Any]) -> list[str]:
    merges = document["model"]["merges"]
    return [
        " ".join(str(part) for part in value) if isinstance(value, list) else str(value)
        for value in merges
    ]


def verify_tokenizer_equivalence(
    base_path: Path, classifier_path: Path
) -> dict[str, Any]:
    """Prove the re-serialized classifier tokenizer has the original BPE semantics."""

    base = json.loads(base_path.read_text())
    classifier = json.loads(classifier_path.read_text())
    comparisons = {
        "model_type": base["model"]["type"] == classifier["model"]["type"] == "BPE",
        "vocabulary": base["model"]["vocab"] == classifier["model"]["vocab"],
        "merge_order": _normalized_merges(base) == _normalized_merges(classifier),
        "added_tokens": base["added_tokens"] == classifier["added_tokens"],
        "pre_tokenizer": base["pre_tokenizer"] == classifier["pre_tokenizer"],
        "post_processor": base["post_processor"] == classifier["post_processor"],
    }
    failed = sorted(name for name, same in comparisons.items() if not same)
    if failed:
        raise PinnedEnvironmentError(
            f"classifier tokenizer is not equivalent to the pinned DNABERT-2 BPE: {failed}"
        )
    return {
        "equivalent": True,
        "comparisons": comparisons,
        "vocab_size": len(base["model"]["vocab"]),
        "merge_count": len(base["model"]["merges"]),
        "runtime_tokenizer_revision": DNABERT2_REVISION,
    }


@dataclass
class Dnabert2PromoterBundle:
    model: Any
    tokenizer: Any
    config: Any
    checkpoint: PromoterCheckpoint
    snapshot: Path
    device: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        return dict(self.provenance)


def load_dnabert2_promoter_classifier(
    candidate: str = "prom_300_all",
    *,
    device: str = "cpu",
    verify_upstream_hashes: bool = True,
) -> Dnabert2PromoterBundle:
    """Load a two-class promoter head with strict code, weights, and tokenizer pins."""

    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    try:
        checkpoint = PROMOTER_CHECKPOINTS[candidate]
    except KeyError as exc:
        raise KeyError(f"unknown promoter checkpoint candidate: {candidate}") from exc
    env = require_pinned_env()
    vendor_hashes = _verify_vendor()
    snapshot = resolve_promoter_snapshot(checkpoint)
    observed_hashes: dict[str, str] = {}
    for name, expected in checkpoint.upstream_sha256.items():
        path = snapshot / name
        if not path.exists():
            raise FileNotFoundError(f"missing pinned promoter artifact: {path}")
        observed = sha256_file(path)
        if verify_upstream_hashes and observed != expected:
            raise PinnedEnvironmentError(
                f"{checkpoint.name}/{name}: sha256 {observed} != pinned {expected}"
            )
        observed_hashes[name] = observed if verify_upstream_hashes else "<unverified>"

    base_snapshot = resolve_snapshot()
    tokenizer_equivalence = verify_tokenizer_equivalence(
        base_snapshot / "tokenizer.json", snapshot / "tokenizer.json"
    )
    from .vendor.bert_layers import BertForSequenceClassification
    from .vendor.configuration_bert import BertConfig

    config = BertConfig.from_pretrained(str(snapshot))
    if config.num_labels != 2:
        raise PinnedEnvironmentError(
            f"promoter classifier must expose two labels, observed {config.num_labels}"
        )
    model = BertForSequenceClassification(config)
    state = load_file(str(snapshot / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise PinnedEnvironmentError(
            f"classifier checkpoint/architecture mismatch: "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )
    classifier_weight = state.get("classifier.weight")
    if classifier_weight is None or tuple(classifier_weight.shape) != (
        2,
        config.hidden_size,
    ):
        raise PinnedEnvironmentError(
            "checkpoint has no complete two-class promoter head"
        )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_snapshot), clean_up_tokenization_spaces=False
    )
    provenance = {
        "hf_id": checkpoint.hf_id,
        "revision": checkpoint.revision,
        "base_hf_id": "zhihan1996/DNABERT-2-117M",
        "base_tokenizer_revision": DNABERT2_REVISION,
        "candidate": checkpoint.name,
        "task_geometry": {
            "input_length_nt": checkpoint.input_length_nt,
            "tss_upstream_nt": checkpoint.tss_upstream_nt,
            "tss_downstream_nt": checkpoint.tss_downstream_nt,
        },
        "device": device,
        "dtype": str(next(model.parameters()).dtype),
        "env": env,
        "vendor_sha256": vendor_hashes,
        "upstream_sha256": observed_hashes,
        "tokenizer_equivalence": tokenizer_equivalence,
        "n_layers": config.num_hidden_layers,
        "n_heads": config.num_attention_heads,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "num_labels": config.num_labels,
        "label_semantics": {"0": "non_promoter", "1": "promoter"},
        "state_dict_keys": len(state),
        "classification_head_complete": True,
    }
    return Dnabert2PromoterBundle(
        model=model,
        tokenizer=tokenizer,
        config=config,
        checkpoint=checkpoint,
        snapshot=snapshot,
        device=device,
        provenance=provenance,
    )


@dataclass(frozen=True)
class PromoterMarginScore:
    request_id: str
    promoter_logit: float
    non_promoter_logit: float
    margin: float
    predicted_label: int
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "promoter_logit": self.promoter_logit,
            "non_promoter_logit": self.non_promoter_logit,
            "margin": self.margin,
            "predicted_label": self.predicted_label,
            "token_count": self.token_count,
        }


class Dnabert2PromoterClassifier:
    """Stable scoring surface for the supervised promoter logit margin."""

    def __init__(self, bundle: Dnabert2PromoterBundle, *, maximum_tokens: int = 256):
        if maximum_tokens < 1:
            raise ValueError("maximum_tokens must be positive")
        self.bundle = bundle
        self.maximum_tokens = maximum_tokens

    def identity(self) -> dict[str, Any]:
        return self.bundle.identity()

    def score(
        self,
        requests: Iterable[tuple[str, str]],
        *,
        batch_size: int = 32,
    ) -> list[PromoterMarginScore]:
        import torch

        rows = [
            (str(request_id), str(sequence).upper())
            for request_id, sequence in requests
        ]
        if not rows:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        wrong_length = [
            request_id
            for request_id, sequence in rows
            if len(sequence) != self.bundle.checkpoint.input_length_nt
        ]
        invalid = [
            request_id for request_id, sequence in rows if set(sequence) - set("ACGT")
        ]
        if wrong_length:
            raise ValueError(
                f"promoter inputs carry the wrong nucleotide length: {wrong_length[:5]}"
            )
        if invalid:
            raise ValueError(f"promoter inputs contain non-ACGT bases: {invalid[:5]}")
        output: list[PromoterMarginScore] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = self.bundle.tokenizer(
                [sequence for _, sequence in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            token_counts = encoded["attention_mask"].sum(dim=1).tolist()
            if max(token_counts) > self.maximum_tokens:
                raise RuntimeError(
                    f"promoter input requires {max(token_counts)} tokens, above the "
                    f"frozen {self.maximum_tokens}-token ceiling"
                )
            encoded = {
                name: tensor.to(self.bundle.device) for name, tensor in encoded.items()
            }
            with torch.inference_mode():
                logits = self.bundle.model(**encoded).logits.detach().cpu()
            if (
                tuple(logits.shape) != (len(batch), 2)
                or not torch.isfinite(logits).all()
            ):
                raise RuntimeError(
                    "promoter classifier returned malformed or non-finite logits"
                )
            for (request_id, _), values, token_count in zip(
                batch, logits.tolist(), token_counts
            ):
                non_promoter, promoter = (float(values[0]), float(values[1]))
                margin = promoter - non_promoter
                if not math.isfinite(margin):
                    raise RuntimeError(f"non-finite promoter margin: {request_id}")
                output.append(
                    PromoterMarginScore(
                        request_id=request_id,
                        promoter_logit=promoter,
                        non_promoter_logit=non_promoter,
                        margin=margin,
                        predicted_label=int(promoter > non_promoter),
                        token_count=int(token_count),
                    )
                )
        return output
