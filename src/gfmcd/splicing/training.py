"""Small, explicit training loop for pair-selection readouts.

This is intentionally not a Trainer abstraction.  Circuit work needs exact
control over which backbone parameters receive gradients, which hidden layer is
read, and which scalar loss is used in later interventions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.splicing.metrics import PairMetrics, evaluate_pair_logits
from gfmcd.splicing.readout import (
    PairIndexBatch,
    collate_pair_indices,
    pair_accuracy,
    pair_cross_entropy,
)
from gfmcd.splicing.schema import SplicePairExample


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    representation_layer: int = -1
    train_backbone: bool = False
    seed: int = 0
    early_stopping_patience: int = 3

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.backbone_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")


@dataclass
class TrainEpoch:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_mrr: float
    val_margin: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches(
    examples: Sequence[SplicePairExample],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterable[list[SplicePairExample]]:
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for i in range(0, len(order), batch_size):
        yield [examples[j] for j in order[i : i + batch_size]]


def forward_examples(
    adapter: SpliceBertAdapter,
    readout: nn.Module,
    examples: Sequence[SplicePairExample],
    *,
    representation_layer: int = -1,
    grad_backbone: bool = False,
) -> tuple[torch.Tensor, PairIndexBatch]:
    tokenized = adapter.tokenize([x.sequence for x in examples])
    outputs = adapter.forward(
        tokenized,
        output_hidden_states=True,
        output_attentions=False,
        grad=grad_backbone,
    )
    if outputs.hidden_states is None:
        raise RuntimeError("SpliceBERT did not return hidden states")
    hidden = outputs.hidden_states[representation_layer]
    pair_batch = collate_pair_indices(
        examples,
        token_offset=tokenized.prefix_tokens,
        device=hidden.device,
    )
    logits = readout(hidden, pair_batch)
    return logits, pair_batch


@torch.no_grad()
def evaluate_readout(
    adapter: SpliceBertAdapter,
    readout: nn.Module,
    examples: Sequence[SplicePairExample],
    *,
    batch_size: int = 8,
    representation_layer: int = -1,
) -> tuple[float, PairMetrics, list[float]]:
    adapter.model.eval()
    readout.eval()
    losses: list[float] = []
    all_logits: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    margins: list[float] = []
    # Pair counts differ, so calculate aggregate metrics per batch and retain
    # per-example margins for bootstrap tests.
    n_total = 0
    correct = 0.0
    rr_sum = 0.0
    nll_sum = 0.0
    margin_sum = 0.0
    pos_margin = 0.0
    from gfmcd.splicing.readout import true_pair_margin

    for batch in batches(examples, batch_size, shuffle=False, seed=0):
        logits, pair_batch = forward_examples(
            adapter,
            readout,
            batch,
            representation_layer=representation_layer,
            grad_backbone=False,
        )
        loss_each = pair_cross_entropy(logits, pair_batch, reduction="none")
        metric = evaluate_pair_logits(logits, pair_batch)
        margin = true_pair_margin(logits, pair_batch)
        losses.extend(float(x) for x in loss_each.cpu())
        margins.extend(float(x) for x in margin.cpu())
        n = len(batch)
        n_total += n
        correct += metric.top1_accuracy * n
        rr_sum += metric.mean_reciprocal_rank * n
        nll_sum += metric.mean_nll * n
        margin_sum += metric.mean_margin * n
        pos_margin += metric.positive_margin_rate * n
    metrics = PairMetrics(
        n=n_total,
        top1_accuracy=correct / n_total,
        mean_reciprocal_rank=rr_sum / n_total,
        mean_nll=nll_sum / n_total,
        mean_margin=margin_sum / n_total,
        positive_margin_rate=pos_margin / n_total,
    )
    return float(np.mean(losses)), metrics, margins


def train_readout(
    adapter: SpliceBertAdapter,
    readout: nn.Module,
    train_examples: Sequence[SplicePairExample],
    val_examples: Sequence[SplicePairExample],
    cfg: TrainConfig | None = None,
) -> list[TrainEpoch]:
    cfg = cfg or TrainConfig()
    cfg.validate()
    set_seed(cfg.seed)
    readout.to(adapter.device)

    for parameter in adapter.model.parameters():
        parameter.requires_grad_(cfg.train_backbone)
    param_groups = [
        {
            "params": [p for p in readout.parameters() if p.requires_grad],
            "lr": cfg.learning_rate,
        }
    ]
    if cfg.train_backbone:
        param_groups.append(
            {
                "params": [p for p in adapter.model.parameters() if p.requires_grad],
                "lr": cfg.backbone_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    history: list[TrainEpoch] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_backbone: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        adapter.model.train(cfg.train_backbone)
        readout.train()
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_n = 0
        for batch in batches(
            train_examples,
            cfg.batch_size,
            shuffle=True,
            seed=cfg.seed + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits, pair_batch = forward_examples(
                adapter,
                readout,
                batch,
                representation_layer=cfg.representation_layer,
                grad_backbone=cfg.train_backbone,
            )
            loss = pair_cross_entropy(logits, pair_batch)
            loss.backward()
            parameters = list(readout.parameters())
            if cfg.train_backbone:
                parameters += list(adapter.model.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
            epoch_correct += float(pair_accuracy(logits.detach(), pair_batch)) * len(batch)
            epoch_n += len(batch)

        val_loss, val_metrics, _margins = evaluate_readout(
            adapter,
            readout,
            val_examples,
            batch_size=cfg.batch_size,
            representation_layer=cfg.representation_layer,
        )
        row = TrainEpoch(
            epoch=epoch,
            train_loss=epoch_loss / epoch_n,
            train_accuracy=epoch_correct / epoch_n,
            val_loss=val_loss,
            val_accuracy=val_metrics.top1_accuracy,
            val_mrr=val_metrics.mean_reciprocal_rank,
            val_margin=val_metrics.mean_margin,
        )
        history.append(row)
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in readout.state_dict().items()}
            if cfg.train_backbone:
                best_backbone = {
                    k: v.detach().cpu().clone() for k, v in adapter.model.state_dict().items()
                }
        else:
            stale += 1
            if stale > cfg.early_stopping_patience:
                break

    if best_state is not None:
        readout.load_state_dict(best_state)
    if best_backbone is not None:
        adapter.model.load_state_dict(best_backbone)
    adapter.model.eval()
    readout.eval()
    return history


def _readout_spec(readout: nn.Module) -> dict[str, int | str]:
    """Return enough architecture metadata to reconstruct a saved readout."""
    from gfmcd.splicing.readout import AdditivePairReadout, BilinearPairReadout

    if isinstance(readout, BilinearPairReadout):
        return {
            "type": "bilinear",
            "hidden_size": int(readout.left.in_features),
            "interaction_rank": int(readout.left.out_features),
            "site_hidden": int(readout.additive.acceptor[1].out_features),
            "length_hidden": int(readout.additive.length[0].out_features),
        }
    if isinstance(readout, AdditivePairReadout):
        return {
            "type": "additive",
            "hidden_size": int(readout.acceptor[1].in_features),
            "site_hidden": int(readout.acceptor[1].out_features),
            "length_hidden": int(readout.length[0].out_features),
        }
    raise TypeError(f"unsupported readout type: {type(readout)!r}")


def build_readout_from_spec(spec: dict) -> nn.Module:
    """Reconstruct a pair readout without consulting the current config file."""
    from gfmcd.splicing.readout import AdditivePairReadout, BilinearPairReadout

    kind = spec.get("type")
    if kind == "additive":
        return AdditivePairReadout(
            int(spec["hidden_size"]),
            site_hidden=int(spec["site_hidden"]),
            length_hidden=int(spec["length_hidden"]),
        )
    if kind == "bilinear":
        return BilinearPairReadout(
            int(spec["hidden_size"]),
            interaction_rank=int(spec["interaction_rank"]),
            site_hidden=int(spec["site_hidden"]),
            length_hidden=int(spec["length_hidden"]),
        )
    raise ValueError(f"unknown readout spec type: {kind!r}")


def save_pair_checkpoint(
    path: str | Path,
    *,
    adapter: SpliceBertAdapter,
    readout: nn.Module,
    readout_type: str,
    train_config: TrainConfig,
    history: Sequence[TrainEpoch],
    save_backbone: bool = False,
    extra_metadata: dict | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "readout_type": readout_type,
        "readout_spec": _readout_spec(readout),
        "readout_state": readout.state_dict(),
        "train_config": asdict(train_config),
        "history": [asdict(x) for x in history],
        "adapter_metadata": asdict(adapter.metadata()),
        "backbone_state": adapter.model.state_dict() if save_backbone else None,
        "extra_metadata": dict(extra_metadata or {}),
    }
    torch.save(blob, p)
    p.with_suffix(".history.json").write_text(
        json.dumps(blob["history"], indent=2) + "\n"
    )
    return p


def load_pair_checkpoint(
    path: str | Path,
    *,
    adapter: SpliceBertAdapter,
    map_location: str | torch.device | None = None,
    load_backbone: bool = True,
) -> tuple[nn.Module, dict]:
    """Load a readout and optional fine-tuned backbone with consistency checks."""
    checkpoint = torch.load(
        Path(path),
        map_location=map_location or adapter.device,
        weights_only=False,
    )
    spec = checkpoint.get("readout_spec")
    if spec is None:
        raise ValueError(
            "checkpoint predates readout_spec; retrain it or migrate it explicitly "
            "rather than guessing architecture parameters"
        )
    if int(spec["hidden_size"]) != adapter.hidden_size:
        raise ValueError(
            f"checkpoint hidden size {spec['hidden_size']} != adapter {adapter.hidden_size}"
        )
    readout = build_readout_from_spec(spec).to(adapter.device)
    readout.load_state_dict(checkpoint["readout_state"])
    backbone_state = checkpoint.get("backbone_state")
    if load_backbone and backbone_state is not None:
        adapter.model.load_state_dict(backbone_state)
    adapter.model.eval()
    readout.eval()
    return readout, checkpoint
