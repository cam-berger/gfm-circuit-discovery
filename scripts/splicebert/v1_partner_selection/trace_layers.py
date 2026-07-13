#!/usr/bin/env python
"""Locate the residual-stream layer at which clean splice information mediates behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.circuits.hf_bert_residual import capture_layer_outputs, layer_patch_scan
from gfmcd.provenance import ExperimentManifest
from gfmcd.splicing.config import load_config, resolve_repo_path
from gfmcd.splicing.context_controls import mask_true_partner_neighborhood, retain_site_windows
from gfmcd.splicing.io import load_jsonl
from gfmcd.splicing.perturb import shuffle_exon_core
from gfmcd.splicing.readout import collate_pair_indices, true_pair_margin
from gfmcd.splicing.training import load_pair_checkpoint


def corrupt(examples, kind, flank, seed):
    out = []
    for i, x in enumerate(examples):
        if kind == "acceptor-mask":
            out.append(mask_true_partner_neighborhood(x, kind="acceptor", flank=flank))
        elif kind == "donor-mask":
            out.append(mask_true_partner_neighborhood(x, kind="donor", flank=flank))
        elif kind == "site-windows-only":
            out.append(retain_site_windows(x, flank=flank, include_all_candidates=True))
        else:
            out.append(shuffle_exon_core(x, protect=flank, seed=seed + i)[0])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/splicebert_partner_selection/base.json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--examples", default="results/splicebert_partner_selection/baselines/val_local_conflicts.jsonl")
    p.add_argument("--corruption", choices=("acceptor-mask", "donor-mask", "site-windows-only", "exon-core-shuffle"), required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--max-examples", type=int, default=16)
    p.add_argument("--flank", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-gap", type=float, default=0.05)
    p.add_argument("--token-patch", choices=("all", "endpoints"), default="all")
    p.add_argument("--out-dir", default="results/splicebert_partner_selection/circuits/layer_patching")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(resolve_repo_path(repo, args.config))
    examples_path = resolve_repo_path(repo, args.examples)
    examples = load_jsonl(examples_path)[: args.max_examples]
    if len(examples) < 2 or any(x.split != "val" for x in examples):
        raise ValueError("layer discovery uses at least two validation examples")

    mcfg = cfg["model"]
    adapter = SpliceBertAdapter.from_pretrained(
        resolve_repo_path(repo, mcfg["path"]),
        tokenizer_mode=mcfg.get("tokenizer_mode", "auto"),
        device=args.device,
        trust_remote_code=bool(mcfg.get("trust_remote_code", False)),
        local_files_only=bool(mcfg.get("local_files_only", True)),
        model_class=mcfg.get("model_class", "masked_lm"),
    )
    checkpoint_path = resolve_repo_path(repo, args.checkpoint)
    readout, checkpoint = load_pair_checkpoint(checkpoint_path, adapter=adapter)
    if checkpoint.get("extra_metadata", {}).get("regime") != "additive-frozen":
        raise SystemExit("primary layer trace is restricted to additive-frozen")

    clean_tokens = adapter.tokenize([x.sequence for x in examples])
    corrupt_sequences = corrupt(examples, args.corruption, args.flank, args.seed)
    corrupt_tokens = adapter.tokenize(corrupt_sequences)
    pair_batch = collate_pair_indices(examples, token_offset=clean_tokens.prefix_tokens, device=adapter.device)
    rep_layer = int(cfg["readout"]["representation_layer"])

    @torch.no_grad()
    def score(tokens):
        output = adapter.forward(tokens, output_hidden_states=True, output_attentions=False)
        logits = readout(output.hidden_states[rep_layer], pair_batch)
        return float(true_pair_margin(logits, pair_batch).mean())

    clean_score, corrupt_score = score(clean_tokens), score(corrupt_tokens)
    if clean_score - corrupt_score < args.min_gap:
        raise RuntimeError("corruption does not create the registered behavior gap")
    cache = capture_layer_outputs(adapter, lambda: score(clean_tokens))

    token_mask = None
    if args.token_patch == "endpoints":
        token_mask = torch.zeros_like(corrupt_tokens.attention_mask, dtype=torch.bool)
        for b, x in enumerate(examples):
            for pos in (x.true_pair.acceptor_pos, x.true_pair.donor_pos):
                lo = max(0, pos - args.flank) + corrupt_tokens.prefix_tokens
                hi = min(len(x.sequence), pos + args.flank) + corrupt_tokens.prefix_tokens
                token_mask[b, lo:hi] = True

    results = layer_patch_scan(
        adapter,
        source_cache=cache,
        clean_score=clean_score,
        corrupt_score=corrupt_score,
        corrupt_score_fn=lambda: score(corrupt_tokens),
        token_mask=token_mask,
    )
    run_name = checkpoint_path.parent.parent.name + "_" + checkpoint_path.parent.name
    out_dir = resolve_repo_path(repo, args.out_dir) / run_name / args.corruption
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "corruption": args.corruption,
        "token_patch": args.token_patch,
        "clean_score": clean_score,
        "corrupt_score": corrupt_score,
        "layers": [x.to_dict() for x in results],
        "claim_scope": "RESIDUAL_MEDIATION_LOCALIZATION",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = ExperimentManifest.create(
        f"{cfg['study_id']}:layer_patch:{run_name}:{args.corruption}",
        sys.argv,
        repo=repo,
        config=cfg,
    )
    manifest.add_input(examples_path)
    manifest.add_input(checkpoint_path)
    manifest.add_output(out_dir / "report.json")
    manifest.write(out_dir / "run_manifest.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
