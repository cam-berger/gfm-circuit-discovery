#!/usr/bin/env python
"""Train registered pair-selection readout/backbone regimes for SpliceBERT.

The primary pretrained-mechanism condition is ``additive-frozen``.  Its readout
cannot directly multiply acceptor and donor representations; any partner
information it uses must already be present in contextualized endpoint states.
Bilinear regimes are diagnostic upper bounds, not primary evidence of a
pretrained partner circuit, because the readout itself can create an
interaction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.provenance import ExperimentManifest
from gfmcd.splicing.config import load_config, resolve_repo_path
from gfmcd.splicing.io import load_jsonl
from gfmcd.splicing.readout import AdditivePairReadout, BilinearPairReadout, parameter_count
from gfmcd.splicing.training import (
    TrainConfig,
    evaluate_readout,
    save_pair_checkpoint,
    set_seed,
    train_readout,
)

REGIMES = (
    "additive-frozen",
    "additive-random-frozen",
    "additive-finetuned",
    "interaction-frozen",
    "interaction-finetuned",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/splicebert_partner_selection/base.json")
    p.add_argument("--regime", choices=REGIMES, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default="artifacts/splicebert_partner_selection/readouts")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    args = p.parse_args()

    # Seed before model/readout construction, not only before minibatch shuffling.
    set_seed(args.seed)
    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(resolve_repo_path(repo, args.config))
    data_dir = resolve_repo_path(repo, args.data_dir or cfg["data"]["output_dir"])
    train_examples = load_jsonl(data_dir / "train.jsonl")
    val_examples = load_jsonl(data_dir / "val.jsonl")
    if args.max_train is not None:
        train_examples = train_examples[: args.max_train]
    if args.max_val is not None:
        val_examples = val_examples[: args.max_val]
    if not train_examples or not val_examples:
        raise RuntimeError("train and validation splits must be non-empty")

    model_cfg = cfg["model"]
    model_path = resolve_repo_path(repo, model_cfg["path"])
    adapter = SpliceBertAdapter.from_pretrained(
        model_path,
        tokenizer_mode=model_cfg.get("tokenizer_mode", "auto"),
        device=args.device,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        model_class=model_cfg.get("model_class", "masked_lm"),
    )

    randomized_backbone = args.regime == "additive-random-frozen"
    if randomized_backbone:
        # NOTE: model.init_weights()/apply(_init_weights) are NO-OPS after
        # from_pretrained on transformers >=5 (params are flagged
        # _is_hf_initialized), which silently left the "random" backbone equal to
        # the pretrained one. Build a genuinely fresh model from the config instead.
        from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

        set_seed(args.seed)
        rand_cfg = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            local_files_only=bool(model_cfg.get("local_files_only", True)),
        )
        rand_loader = (
            AutoModelForMaskedLM
            if model_cfg.get("model_class", "masked_lm") == "masked_lm"
            else AutoModel
        )
        rand_model = rand_loader.from_config(
            rand_cfg,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        # Verify the control actually differs from the pretrained weights.
        import torch as _torch

        _ref = next(
            p for n, p in adapter.model.named_parameters() if "encoder" in n and p.dim() == 2
        )
        _new = dict(rand_model.named_parameters())[
            next(n for n, p in adapter.model.named_parameters() if "encoder" in n and p.dim() == 2)
        ]
        if _torch.equal(_ref.detach().cpu(), _new.detach().cpu()):
            raise RuntimeError("random-backbone control failed to randomize the weights")
        adapter.model = rand_model.to(adapter.device)
        adapter.model.eval()
        # The model object changed; re-resolve cached encoder/attention handles.
        adapter._layers = adapter._resolve_encoder_layers()
        adapter._attention_modules = adapter._resolve_attention_modules()

    r = cfg["readout"]
    is_additive = args.regime.startswith("additive")
    if is_additive:
        readout = AdditivePairReadout(
            adapter.hidden_size,
            site_hidden=int(r["site_hidden"]),
            length_hidden=int(r["length_hidden"]),
        )
        readout_type = "additive"
    else:
        readout = BilinearPairReadout(
            adapter.hidden_size,
            interaction_rank=int(r["interaction_rank"]),
            site_hidden=int(r["site_hidden"]),
            length_hidden=int(r["length_hidden"]),
        )
        readout_type = "bilinear"
    train_backbone = args.regime.endswith("finetuned")

    t = cfg["training"]
    train_cfg = TrainConfig(
        epochs=int(t["epochs"]),
        batch_size=int(t["batch_size"]),
        learning_rate=float(t["learning_rate"]),
        backbone_learning_rate=float(t["backbone_learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        grad_clip=float(t["grad_clip"]),
        representation_layer=int(r["representation_layer"]),
        train_backbone=train_backbone,
        seed=args.seed,
        early_stopping_patience=int(t["early_stopping_patience"]),
    )
    history = train_readout(adapter, readout, train_examples, val_examples, train_cfg)
    val_loss, val_metrics, _ = evaluate_readout(
        adapter,
        readout,
        val_examples,
        batch_size=train_cfg.batch_size,
        representation_layer=train_cfg.representation_layer,
    )

    run_dir = resolve_repo_path(repo, args.out_dir) / args.regime / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = save_pair_checkpoint(
        run_dir / "checkpoint.pt",
        adapter=adapter,
        readout=readout,
        readout_type=readout_type,
        train_config=train_cfg,
        history=history,
        save_backbone=train_backbone or randomized_backbone,
        extra_metadata={
            "regime": args.regime,
            "randomized_backbone": randomized_backbone,
            "primary_trace_target": args.regime == "additive-frozen",
        },
    )
    report = {
        "regime": args.regime,
        "seed": args.seed,
        "readout_parameters": parameter_count(readout),
        "train_backbone": train_backbone,
        "randomized_backbone": randomized_backbone,
        "val_loss": val_loss,
        "val_metrics": val_metrics.to_dict(),
        "adapter": asdict(adapter.metadata()),
        "train_config": asdict(train_cfg),
        "interpretation": (
            "PRIMARY: tests information already exposed by the pretrained contextual states"
            if args.regime == "additive-frozen"
            else "CONTROL/SECONDARY: do not use alone to claim a pretrained partner circuit"
        ),
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = ExperimentManifest.create(
        f"{cfg['study_id']}:train:{args.regime}:seed{args.seed}",
        sys.argv,
        repo=repo,
        config=cfg,
        seeds={"training": args.seed},
        notes=[
            "The test split was not loaded.",
            "Additive-frozen is the registered primary pretrained-mechanism condition.",
            "Bilinear readouts can synthesize interactions and are diagnostic upper bounds.",
        ],
    )
    manifest.add_input(data_dir / "train.jsonl")
    manifest.add_input(data_dir / "val.jsonl")
    manifest.add_output(checkpoint)
    manifest.add_output(run_dir / "metrics.json")
    manifest.write(run_dir / "run_manifest.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
