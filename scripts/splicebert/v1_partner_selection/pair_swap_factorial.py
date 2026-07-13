#!/usr/bin/env python
"""Measure acceptor×donor non-additivity with aligned AA/AB/BA/BB swaps.

A non-zero interaction contrast is evidence that the scored function is not an
independent sum of endpoint-local terms.  It is not yet proof that the effect is
biologically correct or that a compact causal circuit has been identified.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.provenance import ExperimentManifest
from gfmcd.splicing.config import load_config, resolve_repo_path
from gfmcd.splicing.io import load_jsonl
from gfmcd.splicing.metrics import grouped_bootstrap
from gfmcd.splicing.perturb import boundary_pair_factorial
from gfmcd.splicing.readout import collate_pair_indices
from gfmcd.splicing.training import load_pair_checkpoint


def aligned_pairs(examples, max_pairs: int):
    groups = {}
    for x in examples:
        key = (len(x.sequence), x.true_pair.acceptor_pos, x.true_pair.donor_pos)
        groups.setdefault(key, []).append(x)
    out = []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda x: (x.gene_id, x.example_id))
        used = set()
        for i, a in enumerate(items):
            if a.example_id in used:
                continue
            b = next(
                (
                    x
                    for x in items[i + 1 :]
                    if x.example_id not in used and x.gene_id != a.gene_id
                ),
                None,
            )
            if b is None:
                continue
            out.append((a, b))
            used.update((a.example_id, b.example_id))
            if len(out) >= max_pairs:
                return out
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/splicebert_partner_selection/base.json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--examples", default=None, help="validation JSONL; defaults to processed val split")
    p.add_argument("--local-baseline", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-factorials", type=int, default=128)
    p.add_argument("--flank", type=int, default=12)
    p.add_argument("--allow-secondary-regime", action="store_true")
    p.add_argument("--out-dir", default="results/splicebert_partner_selection/behavior/pair_factorial")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(resolve_repo_path(repo, args.config))
    data_dir = resolve_repo_path(repo, cfg["data"]["output_dir"])
    examples_path = resolve_repo_path(repo, args.examples) if args.examples else data_dir / "val.jsonl"
    examples = load_jsonl(examples_path)
    if any(x.split != "val" for x in examples):
        raise ValueError("factorial development assay must use validation examples")
    pairs = aligned_pairs(examples, args.max_factorials)
    if not pairs:
        raise RuntimeError("no equal-length, cross-gene aligned example pairs were found")

    model_cfg = cfg["model"]
    adapter = SpliceBertAdapter.from_pretrained(
        resolve_repo_path(repo, model_cfg["path"]),
        tokenizer_mode=model_cfg.get("tokenizer_mode", "auto"),
        device=args.device,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        model_class=model_cfg.get("model_class", "masked_lm"),
    )
    checkpoint_path = resolve_repo_path(repo, args.checkpoint)
    readout, checkpoint = load_pair_checkpoint(checkpoint_path, adapter=adapter)
    regime = checkpoint.get("extra_metadata", {}).get("regime", "unknown")
    if regime != "additive-frozen" and not args.allow_secondary_regime:
        raise SystemExit(f"refusing secondary regime {regime!r} without explicit override")
    representation_layer = int(cfg["readout"]["representation_layer"])

    baseline = None
    baseline_path = None
    if args.local_baseline:
        baseline_path = resolve_repo_path(repo, args.local_baseline)
        with baseline_path.open("rb") as f:
            baseline = pickle.load(f)

    records = []
    for a, b in pairs:
        sequences = boundary_pair_factorial(a, a, b, flank=args.flank)
        labels = ("AA", "AB", "BA", "BB")
        seq_list = [sequences[x] for x in labels]
        tokenized = adapter.tokenize(seq_list)
        outputs = adapter.forward(
            tokenized,
            output_hidden_states=True,
            output_attentions=False,
            grad=False,
        )
        pair_batch = collate_pair_indices(
            [a, a, a, a],
            token_offset=tokenized.prefix_tokens,
            device=adapter.device,
        )
        with torch.no_grad():
            logits = readout(outputs.hidden_states[representation_layer], pair_batch)
        true_logits = [
            float(logits[i, a.true_pair_index]) for i in range(4)
        ]
        model_contrast = true_logits[0] - true_logits[1] - true_logits[2] + true_logits[3]

        baseline_logits = None
        baseline_contrast = None
        if baseline is not None:
            baseline_logits = []
            for sequence in seq_list:
                proxy = replace(a, sequence=sequence)
                baseline_logits.append(float(baseline.score_pair(proxy, proxy.true_pair)))
            baseline_contrast = (
                baseline_logits[0] - baseline_logits[1] - baseline_logits[2] + baseline_logits[3]
            )

        records.append(
            {
                "background_id": a.example_id,
                "source_b_id": b.example_id,
                "background_gene": a.gene_id,
                "source_b_gene": b.gene_id,
                "exon_length": a.true_pair.exon_length,
                "model_true_pair_logits": dict(zip(labels, true_logits)),
                "model_interaction_contrast": model_contrast,
                "local_true_pair_scores": (
                    dict(zip(labels, baseline_logits)) if baseline_logits is not None else None
                ),
                "local_interaction_contrast": baseline_contrast,
            }
        )

    contrasts = [x["model_interaction_contrast"] for x in records]
    abs_contrasts = [abs(x) for x in contrasts]
    report = {
        "regime": regime,
        "n_factorials": len(records),
        "contrast_definition": "f(AA)-f(AB)-f(BA)+f(BB) on true-pair logit",
        "signed_contrast_by_gene": grouped_bootstrap(
            contrasts, [x[0].gene_id for x in pairs], seed=0
        ),
        "absolute_contrast_by_gene": grouped_bootstrap(
            abs_contrasts, [x[0].gene_id for x in pairs], seed=1
        ),
        "local_baseline_mean_abs_contrast": (
            float(np.mean([abs(x["local_interaction_contrast"]) for x in records]))
            if baseline is not None
            else None
        ),
        "claim_scope": [
            "A non-zero contrast rejects strict endpoint additivity for the scored function.",
            "The assay is chimeric and remains secondary to natural cue-conflict behavior.",
            "Circuit claims require causal localization and patching of the interaction contrast.",
        ],
    }

    run_name = checkpoint_path.parent.parent.name + "_" + checkpoint_path.parent.name
    out_dir = resolve_repo_path(repo, args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "per_factorial.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in records)
    )

    manifest = ExperimentManifest.create(
        f"{cfg['study_id']}:pair_factorial:{run_name}",
        sys.argv,
        repo=repo,
        config=cfg,
        notes=[
            "AA/AB/BA/BB sequences share one background and aligned coordinates.",
            "This is a chimeric non-additivity assay, not the natural primary endpoint.",
        ],
    )
    manifest.add_input(examples_path)
    manifest.add_input(checkpoint_path)
    if baseline_path:
        manifest.add_input(baseline_path)
    manifest.add_output(out_dir / "report.json")
    manifest.add_output(out_dir / "per_factorial.jsonl")
    manifest.write(out_dir / "run_manifest.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
