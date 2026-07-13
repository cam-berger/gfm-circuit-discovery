#!/usr/bin/env python
"""Evaluate a trained SpliceBERT pair-selection regime and context controls.

The script defaults to validation data.  Opening the chromosome-held-out test
split requires both ``--split test`` and ``--unblind-test`` so a terminal agent
cannot accidentally turn the test set into an iterative development set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.provenance import ExperimentManifest
from gfmcd.splicing.config import load_config, resolve_repo_path
from gfmcd.splicing.context_controls import (
    mask_true_partner_neighborhood,
    retain_pair_endpoints_only,
    retain_site_windows,
)
from gfmcd.splicing.io import load_jsonl
from gfmcd.splicing.metrics import bootstrap_difference, grouped_bootstrap
from gfmcd.splicing.scoring import evaluate_sequence_overrides
from gfmcd.splicing.training import load_pair_checkpoint


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def score_condition(adapter, readout, examples, sequences, cfg):
    loss, metrics, records = evaluate_sequence_overrides(
        adapter,
        readout,
        examples,
        sequences,
        batch_size=int(cfg["training"]["batch_size"]),
        representation_layer=int(cfg["readout"]["representation_layer"]),
    )
    return {
        "loss": loss,
        "metrics": metrics.to_dict(),
        "margins": [x.true_margin for x in records],
        "records": records,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/splicebert_partner_selection/base.json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--unblind-test", action="store_true")
    p.add_argument("--subset-file", default=None, help="optional JSONL of frozen conflict examples")
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default="results/splicebert_partner_selection/behavior")
    p.add_argument("--site-flank", type=int, default=12)
    p.add_argument("--max-examples", type=int, default=None)
    args = p.parse_args()

    if args.split == "test" and not args.unblind_test:
        raise SystemExit("refusing to open test split without --unblind-test")

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(resolve_repo_path(repo, args.config))
    data_dir = resolve_repo_path(repo, args.data_dir or cfg["data"]["output_dir"])
    source_path = (
        resolve_repo_path(repo, args.subset_file)
        if args.subset_file
        else data_dir / f"{args.split}.jsonl"
    )
    examples = load_jsonl(source_path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if not examples:
        raise RuntimeError("evaluation set is empty")
    if any(x.split != args.split for x in examples):
        raise ValueError("subset file contains examples from the wrong split")

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

    conditions = {
        "clean": [x.sequence for x in examples],
        "all_candidate_site_windows_only": [
            retain_site_windows(x, flank=args.site_flank, include_all_candidates=True)
            for x in examples
        ],
        "true_pair_endpoints_only": [
            retain_pair_endpoints_only(x, flank=args.site_flank) for x in examples
        ],
        "true_acceptor_masked": [
            mask_true_partner_neighborhood(x, kind="acceptor", flank=args.site_flank)
            for x in examples
        ],
        "true_donor_masked": [
            mask_true_partner_neighborhood(x, kind="donor", flank=args.site_flank)
            for x in examples
        ],
    }
    scored = {
        name: score_condition(adapter, readout, examples, sequences, cfg)
        for name, sequences in conditions.items()
    }
    clean_margins = scored["clean"]["margins"]
    comparisons = {}
    for name, result in scored.items():
        if name == "clean":
            continue
        comparisons[name] = {
            "clean_minus_control_margin": bootstrap_difference(
                clean_margins,
                result["margins"],
                paired=True,
                seed=0,
            ),
            "control_margin_by_gene": grouped_bootstrap(
                result["margins"],
                [x.gene_id for x in examples],
                seed=0,
            ),
        }

    run_name = checkpoint_path.parent.name + "_" + checkpoint_path.parent.parent.name
    out_dir = resolve_repo_path(repo, args.out_dir) / run_name / args.split
    if args.subset_file:
        out_dir = out_dir / Path(args.subset_file).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "split": args.split,
        "subset_source": str(source_path),
        "n": len(examples),
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": checkpoint.get("extra_metadata", {}),
        "conditions": {
            name: {"loss": value["loss"], "metrics": value["metrics"]}
            for name, value in scored.items()
        },
        "comparisons": comparisons,
        "interpretation_guards": [
            "Clean performance above local baselines is behavioral evidence, not a circuit claim.",
            "Loss under context removal supports context dependence but does not localize a mechanism.",
            "A bilinear readout can install an interaction and is not by itself evidence of a pretrained circuit.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, value in scored.items():
        dump_jsonl(out_dir / f"per_example_{name}.jsonl", [x.to_dict() for x in value["records"]])

    manifest = ExperimentManifest.create(
        f"{cfg['study_id']}:behavior:{run_name}:{args.split}",
        sys.argv,
        repo=repo,
        config=cfg,
        notes=[
            "Candidate coordinates are inherited from clean examples under counterfactual edits.",
            "Test unblinding was explicit." if args.split == "test" else "Test split remained unopened.",
        ],
    )
    manifest.add_input(source_path)
    manifest.add_input(checkpoint_path)
    manifest.add_output(out_dir / "metrics.json")
    for path in out_dir.glob("per_example_*.jsonl"):
        manifest.add_output(path)
    manifest.write(out_dir / "run_manifest.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
